# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by globbing for `*.nwb` under `/app/data` using `pathlib.Path.rglob`. Each file is opened with `h5py` (not `pynwb`) and tables are read manually from the HDF5 group structure. Trial tables come from `intervals/trials`, unit tables from `units`, and behavioral events from `acquisition/BehavioralEvents`.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
with h5py.File(path, 'r') as f:
    trial_table = load_trial_table(f)
    units = load_units_table(f)
```

```python
def load_trial_table(f):
    g = f['intervals/trials']
    table = {k: decode_arr(g[k][()]) for k in g.keys() if isinstance(g[k], h5py.Dataset)}
    return table
```

iii. The AI chose h5py over pynwb for speed. The CONVERSION_NOTES state "Used h5py direct reads instead of full pynwb object loading for conversion." This is functionally equivalent since NWB files are HDF5, though it requires manual navigation of the HDF5 group hierarchy.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-440956`). This is assigned per session and accumulated into a list during the main loop.

ii.
```python
subj = path.parent.name
...
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. The directory structure stores one subject per folder, so `path.parent.name` gives the subject identifier. This gives 28 subjects. The subject IDs include the "sub-" prefix (e.g., "sub-440956") rather than just the numeric ID. The CONVERSION_NOTES confirm 28 subjects were detected.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The sorted glob list defines session order. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
for f in files:
    sn, si, so, subj, regions = process_session(f, ...)
    if len(sn) < 2:
        continue
```

iii. The CONVERSION_NOTES document 174 NWB files across 28 subjects. The AI retains all 174 sessions (the verification output confirms 174 sessions). The reference drops 1 session (which has no QC'd units), keeping 173.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). Go cue times are loaded from `acquisition/BehavioralEvents/go_start_times/timestamps`. Each trial corresponds to one go cue event.

ii.
```python
def infer_go_cue_times(trial_table, f=None):
    if f is not None and 'acquisition/BehavioralEvents/go_start_times/timestamps' in f:
        return np.asarray(f['acquisition/BehavioralEvents/go_start_times/timestamps'][:], dtype=float)
```

iii. The AI uses the same go cue event stream as the reference. However, it also includes extensive fallback logic for inferring go cue times from other fields, which is unnecessary for this dataset.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by dropping any trial whose aligned neural matrix is entirely zero (`np.allclose(trial_mats, 0)`). It also drops trials where the aligned window falls outside the pre-binned session array. No explicit `obs_intervals` or `free_water` filtering is performed.

ii.
```python
if start_idx < 0 or end_idx > session_fr.shape[1]:
    continue
trial_mats = session_fr[:, start_idx:end_idx].copy()
...
if np.allclose(trial_mats, 0):
    continue
if np.allclose(trial_mats, 0):
    continue
```

iii. The CONVERSION_NOTES mention: "All-zero neural trials in many sessions: resolved by skipping trials whose aligned neural matrix was entirely zero." The AI uses a heuristic zero-check rather than the principled `obs_intervals` filter. The duplicate `np.allclose` check suggests debugging artifacts left in. The result is 90,999 trials vs the reference's 90,860.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (via `units/spike_times_index` for ragged indexing). Only units passing the quality filter contribute.

ii.
```python
def get_spike_times_for_unit(units_group, idx):
    st = units_group['spike_times']
    if 'spike_times_index' in units_group:
        ind = units_group['spike_times_index'][:]
        end = ind[idx]
        start = 0 if idx == 0 else ind[idx - 1]
        return st[start:end]
```

iii. This correctly reads spike times from the ragged HDF5 storage. The indexing via `spike_times_index` is the standard NWB approach for ragged arrays.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into a session-wide grid of 50ms bins, converted to firing rates (Hz) by dividing by bin size. For each trial, the appropriate slice of this pre-binned array is extracted based on the go cue time.

ii.
```python
session_edges = np.arange(session_start, session_stop + BIN_SIZE, BIN_SIZE)
session_fr = np.zeros((len(good_inds), len(session_edges) - 1), dtype=np.float32)
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
    m = (st >= session_start) & (st < session_stop)
    counts, _ = np.histogram(st[m], bins=session_edges)
    session_fr[j] = counts.astype(np.float32) / BIN_SIZE
for i, go in enumerate(go_times):
    start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
    end_idx = start_idx + N_BINS
    trial_mats = session_fr[:, start_idx:end_idx].copy()
```

iii. The session-wide pre-binning approach is an optimization to avoid per-trial histogramming. However, the session grid is not aligned to each trial's go cue — the grid starts at `min(start_times) + T_START - 0.5` and advances in fixed BIN_SIZE steps. The trial's slice is found by rounding the go-cue-relative offset to the nearest bin index, which can introduce up to ~25ms temporal misalignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses a cascading quality filter in `choose_good_units`: first checks `classification` for 'good'/'single'/'single_unit' values, then boolean flags, then metric-based thresholds (presence_ratio >= 0.9, amplitude_cutoff <= 0.1, isi_violation <= 0.5, nn_hit_rate >= 0.9), and if all metrics pass 0 units, keeps all units.

ii.
```python
def choose_good_units(units):
    for key in ['classification', 'unit_quality', 'quality', 'label', 'cluster_quality']:
        if key in units:
            sval = np.array([str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() for v in vals], dtype=object)
            good_words = {'good', 'single', 'single_unit', 'single unit'}
            if np.isin(sval, list(good_words)).any():
                return np.isin(sval, list(good_words))
    # Metric-based fallback
    mask = np.ones(n, dtype=bool)
    if 'presence_ratio' in units:
        mask &= np.asarray(units['presence_ratio'], dtype=float) >= 0.9
    ...
    if mask.sum() == 0:
        mask = np.ones(n, dtype=bool)
    return mask
```

iii. The primary filter (`classification == 'good'`) matches the reference. However, for the session `sub-440958_ses-20190216T162508` where `classification` values are NaN (never quality-controlled), the AI's `str(NaN).strip().lower()` produces `'nan'`, which doesn't match 'good'. The function then falls through to metric-based fallbacks, which for that session result in 1201 units being kept. The reference correctly drops this session entirely. The AI's total of 70,654 good units exceeds the reference's 69,453 by approximately the 1201 units from this session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The go cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` define alignment. The trial window is extracted from the pre-binned session array by computing a bin index offset from the session start.

ii.
```python
go_times = infer_go_cue_times(trial_table, f)
...
start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
end_idx = start_idx + N_BINS
trial_mats = session_fr[:, start_idx:end_idx].copy()
```

iii. The alignment event (go cue) is correct. However, the pre-binned grid approach means the actual bin boundaries for each trial don't exactly match `go_cue + [-2.5, 1.5]` — there's rounding to the nearest pre-existing bin edge.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins spanning -2.5 to +1.5 s relative to the go cue. No rebinning is applied beyond the initial histogramming.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
```

iii. Matches the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (tone onset events) and the go cue time for each trial. When the number of sample events equals the number of go cue events, they are used one-to-one. Otherwise, for each trial, the first sample event within the trial's `[start_time, stop_time]` window is used.

ii.
```python
sample_starts = np.asarray(trial_table['start_time'], dtype=float)
if 'acquisition/BehavioralEvents/sample_start_times/timestamps' in f:
    sample_event_times = np.asarray(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:], dtype=float)
    if len(sample_event_times) == len(go_times):
        sample_starts = sample_event_times.copy()
    else:
        for i in range(len(go_times)):
            cand = sample_event_times[(sample_event_times >= start_times[i]) & (sample_event_times <= stop_times[i])]
            if len(cand):
                sample_starts[i] = cand[0]
```

iii. The AI uses the first sample event in the trial window, while the reference uses the last sample event before the go cue. For early-lick trials that replay the sample epoch, multiple sample_start_times events exist per trial. The first is the original tone; the last is the final replay (which is what the animal was actually responding to). The verification output shows a minimum time_from_tone of -1.5 for many sessions (vs -0.6 in the reference), suggesting incorrect tone assignment for some trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as the bin center time relative to the go cue, minus the tone-to-go offset.

ii.
```python
def build_time_from_tone_vector(sample_start, go_time):
    return (BIN_CENTERS - (sample_start - go_time)).astype(np.float32)[None, :]
```

iii. The formula is equivalent to `(go_time + bin_center) - sample_start`, giving the absolute bin time minus the tone onset. This produces a continuously increasing ramp across bins. The shape `[None, :]` adds a leading dimension.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same `BIN_CENTERS` array (defined relative to the go cue), so they share the same temporal grid by construction.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. Alignment is guaranteed because both neural and input data reference the same bin center grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Primarily from `photostim_onset` and `photostim_duration` in the trial table. As a fallback, uses `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`.

ii.
```python
if 'photostim_onset' in trial_table and 'photostim_duration' in trial_table:
    onset = safe_float(trial_table['photostim_onset'][i])
    dur = safe_float(trial_table['photostim_duration'][i])
    if np.isfinite(onset) and np.isfinite(dur) and dur > 0:
        abs_on = start_times[i] + onset
        pst = np.array([abs_on], dtype=float)
        pen = np.array([abs_on + dur], dtype=float)
```

iii. The AI correctly identifies `photostim_onset` as trial-relative and converts to absolute time by adding `start_times[i]`. The `safe_float` function handles 'N/A' strings by returning NaN.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary vector is constructed: bins where the center falls between photostim start and stop are set to 1, all others 0.

ii.
```python
def build_photostim_vector(starts, stops, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(starts, stops):
        rs = s - go_time
        re = e - go_time
        on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
        x[on] = 1.0
    return x
```

iii. This matches the reference approach: a bin is 1 if its center falls within the stimulation interval, 0 otherwise. The reference uses the same bin-center comparison.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim onset/offset are converted to go-cue-relative times and compared against `BIN_CENTERS`, ensuring alignment with the neural data grid.

ii.
```python
rs = s - go_time
re = e - go_time
on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
```

iii. Both photostim and neural data reference the same go-cue-aligned bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from actual lick timestamps: `acquisition/BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps`, combined with go cue times and trial stop times to determine which side the animal licked during the response window.

ii.
```python
left_licks = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_licks = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
choice, outcome, early = infer_choice_outcome_early(trial_table, left_licks, right_licks, go_times)
```

iii. Unlike the reference (which derives choice from `trial_instruction` x `outcome`), the AI directly examines lick events in the response window. Both approaches should produce the same result for normal trials, but may diverge on edge cases (e.g., bilateral licks).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the code checks whether left or right lick events occur in the window `[go_time, min(stop_time, go_time + 1.5)]`. If only left licks are found, choice = 0 (left); if only right, choice = 1 (right); otherwise, choice = 2 (no lick / ambiguous).

ii.
```python
for i in range(n):
    l_post = np.any((left_licks >= go_times[i]) & (left_licks < min(stop[i], go_times[i] + 1.5)))
    r_post = np.any((right_licks >= go_times[i]) & (right_licks < min(stop[i], go_times[i] + 1.5)))
    if l_post and not r_post:
        choice[i] = 0
    elif r_post and not l_post:
        choice[i] = 1
    else:
        choice[i] = 2
```

iii. This is a more direct approach than the reference. However, when both left and right licks occur (bilateral), the AI assigns "no lick" (2), which may not match the reference's `trial_instruction` x `outcome` derivation. The output values list matches: `['left', 'right', 'no lick']`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trial table. The AI also has fallback logic using the choice value.

ii.
```python
if 'outcome' in table:
    ov = np.array([str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() for v in table['outcome']], dtype=object)
    for i, s in enumerate(ov):
        if s == 'hit' or 'correct' in s:
            outcome[i] = 2
        elif s == 'miss' or 'error' in s or 'incorrect' in s:
            outcome[i] = 1
        elif s == 'ignore' or 'no' in s:
            outcome[i] = 0
```

iii. The mapping is: ignore=0, miss=1, hit=2, matching the reference. The heuristic matching with 'correct'/'error'/'no' is unnecessary for this dataset but doesn't cause harm.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String outcomes are mapped to integer codes (ignore=0, miss=1, hit=2) and broadcast across all time bins.

ii.
```python
out = np.vstack([
    ...
    np.full(N_BINS, outcome[i], dtype=np.int64),
    ...
])
```

iii. Same approach as the reference: per-trial value repeated across all bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trial table.

ii.
```python
if 'early_lick' in table:
    ev = np.asarray(table['early_lick'])
    early = np.array([
        1 if str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() == 'early' else 0
        for v in ev
    ], dtype=np.int64)
```

iii. The AI reads the same field as the reference: `early_lick` from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string 'early' maps to 1, everything else to 0. The value is broadcast across all time bins.

ii.
```python
np.full(N_BINS, early[i], dtype=np.int64)
```

iii. Same approach as the reference (no=0, yes=1, repeated across bins).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (columns: x, y, likelihood) and the corresponding `timestamps`.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_xy = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
```

iii. Same source variable as the reference. However, the AI loads the data as `tongue_xy` with 3 columns but only uses columns 0 and 1 (x, y), ignoring the likelihood column (column 2) for visibility filtering.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes 40th/60th percentiles directly over all raw frames where y is finite (not NaN), without filtering by tracking likelihood. It then discretizes each frame into categories 0/1/2/3. Per-trial binning uses mode (most frequent label) of frames within each time bin.

ii.
```python
def discretize_tongue_y(data_xy):
    y = np.asarray(data_xy[:, 1], dtype=float)
    visible = np.isfinite(y)
    if visible.sum() == 0:
        return np.full(len(y), 3, dtype=np.int64), (np.nan, np.nan)
    q40, q60 = np.nanpercentile(y[visible], [40, 60])
    out = np.full(len(y), 3, dtype=np.int64)
    out[visible & (y < q40)] = 0
    out[visible & (y >= q40) & (y <= q60)] = 1
    out[visible & (y > q60)] = 2
    return out, (q40, q60)
```

iii. Two major differences from the reference: (1) No likelihood filtering — since raw y values are finite even when the tongue is retracted, nearly all frames are treated as "visible," which corrupts the percentile computation and class assignments. The verification output confirms only 2% "not_visible" vs ~75% in the reference. (2) Per-trial binning uses mode of frame labels rather than mean of y values — the reference averages y within each 50ms bin then digitizes the mean, while the AI digitizes per-frame then takes the mode.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. 40th and 60th percentiles of finite y values across the session define category boundaries: 0 (< 40th), 1 (40th to 60th inclusive), 2 (> 60th), 3 (not visible / not finite).

ii.
```python
q40, q60 = np.nanpercentile(y[visible], [40, 60])
out[visible & (y < q40)] = 0
out[visible & (y >= q40) & (y <= q60)] = 1
out[visible & (y > q60)] = 2
```

iii. The threshold logic itself is correct (same percentile boundaries as the reference), but because the AI doesn't filter by likelihood, the percentiles are computed over all frames (including retracted tongue), yielding very different thresholds than the reference which only uses frames with likelihood >= 0.5.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, frames within `[go + T_START, go + T_END)` are selected, and their pre-discretized labels are binned into 50ms bins using the neural bin edges. The most frequent non-"not visible" label within each bin is taken.

ii.
```python
def bin_tongue_for_trial(timestamps, labels, go_time):
    out = np.full(N_BINS, 3, dtype=np.int64)
    rel = timestamps - go_time
    for b in range(N_BINS):
        m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
        if np.any(m):
            vals = labels[m]
            vals = vals[vals != 3]
            out[b] = 3 if len(vals) == 0 else np.bincount(vals, minlength=3).argmax()
    return out
```

iii. The alignment to go cue uses the same BIN_EDGES as the neural data, so temporal alignment is correct. However, taking the mode of pre-discretized labels differs from the reference's approach of averaging raw y values then discretizing the mean.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three approaches: (1) Sessions with no good units (from the `choose_good_units` cascade) would produce empty neural arrays. (2) Trials with all-zero neural data are dropped. (3) Missing tongue tracking data defaults to "not visible" (class 3).

ii.
```python
if np.allclose(trial_mats, 0):
    continue
...
tongue_trial = np.full(N_BINS, 3, dtype=np.int64)  # fallback
```

iii. The all-zero heuristic catches some problematic trials but is not principled — a trial could legitimately have very low firing rates that round to zero. The AI does not handle the NaN-classification session correctly (it keeps it with metric-based fallback units), unlike the reference which drops it.

## 10-a. What are the most time-consuming steps of the code?

i. Per-unit spike time loading and histogramming over the full session timeline dominates. Each session requires reading spike times for each good unit individually from HDF5.

ii.
```python
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
    m = (st >= session_start) & (st < session_stop)
    counts, _ = np.histogram(st[m], bins=session_edges)
```

iii. The CONVERSION_NOTES report ~1.37s per session after optimization (pre-binning approach). The `get_spike_times_for_unit` function reads `spike_times_index` on every call, which is redundant.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial tongue binning loop iterates over 80 bins per trial, checking frame timestamps against each bin edge. This could be vectorized with searchsorted or digitize. The per-trial main loop in `process_session` also iterates over all trials individually.

ii.
```python
def bin_tongue_for_trial(timestamps, labels, go_time):
    for b in range(N_BINS):
        m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
```

iii. The per-bin tongue loop is O(n_bins * n_frames_per_trial) when it could be O(n_frames_per_trial) with digitize.

## 10-c. What processing does the code repeat multiple times?

i. `get_spike_times_for_unit` re-reads `spike_times_index[:]` from HDF5 on every call (once per good unit per session). The full index array could be read once per session.

ii.
```python
def get_spike_times_for_unit(units_group, idx):
    st = units_group['spike_times']
    if 'spike_times_index' in units_group:
        ind = units_group['spike_times_index'][:]  # re-read every call
```

iii. For a session with 500 good units, this reads the same index array 500 times. The reference reads the offset array once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads the full units table (`load_units_table`) including all columns (quality metrics, waveform stats, etc.) when only `classification`, `spike_times`, `spike_times_index`, `anno_name`/electrode info, and `obs_intervals` are needed. Similarly, `tongue_xy` column 0 (x) is loaded but never used in the discretization.

ii.
```python
def load_units_table(f):
    g = f['units']
    table = {k: decode_arr(g[k][()]) for k in g.keys() if isinstance(g[k], h5py.Dataset)}
```

iii. Loading all unit columns wastes memory and I/O time, especially for large arrays like spike_times that are duplicated in the table dict.
