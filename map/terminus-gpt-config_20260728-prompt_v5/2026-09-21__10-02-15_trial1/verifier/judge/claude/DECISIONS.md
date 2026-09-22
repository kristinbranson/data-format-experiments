# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by recursively globbing for `*.nwb` under `/app/data`. Each file is opened with `h5py` (not `pynwb`) and processed one at a time in `session_from_file()`. Trials, units, and behavioral events are accessed via HDF5 group paths.

ii.
```python
DATA_DIR = Path('/app/data')
files = sorted(DATA_DIR.rglob('*.nwb'))
```

```python
with h5py.File(path, 'r') as h:
    units = h['units']
    trials = h['intervals']['trials']
    acq = h['acquisition']
```

iii. The AI chose `h5py` for direct HDF5 access rather than `pynwb`. CONVERSION_NOTES documents that NWB files are organized by subject folders, with 174 files total covering 28 subjects.

## 1-b. How are the data split into subjects?

i. Subjects are identified by stripping the `sub-` prefix from the parent folder name, e.g. `sub-440956` becomes `'440956'`. This is a numeric ID, not the mouse name used in papers.

ii.
```python
subject = path.parent.name.replace('sub-', '')
```

```python
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. The folder name encodes the subject ID. The AI uses it directly rather than accessing the NWB subject metadata.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are sorted by file path. The session identifier is derived from the file path, not from `nwb.identifier`.

ii.
```python
files = sorted(DATA_DIR.rglob('*.nwb'))
```

```python
session_info = {
    'file': str(path),
    'subject': subject,
    ...
}
```

iii. The AI treats each NWB file as a separate session, which is correct since the dataset stores one session per file.

## 1-d. How are the data split into trials?

i. Trials are defined by the go cue event times from `BehavioralEvents/go_start_times`. The number of trials equals the number of go cue events. Trial table columns (outcome, instruction, etc.) are indexed by trial number, truncated to `n_trials` (the go cue count).

ii.
```python
go_times = np.asarray(beh_events['go_start_times']['timestamps'][()]).astype(float)
n_trials = len(go_times)
trial_instruction = _decode_arr(trials['trial_instruction'][()])[:n_trials]
outcome = _decode_arr(trials['outcome'][()])[:n_trials]
```

iii. The AI uses go cue count as the trial count and truncates trial table columns to match.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses `is_good_trials` from the units table (a units x trials boolean matrix) to determine valid trials. A trial is kept if the mean across good units is > 0 (i.e., at least one good unit considers it valid). Additionally, trials with all-zero neural data are dropped post-hoc. The AI does NOT filter on `obs_intervals` or `free_water`.

ii.
```python
if 'is_good_trials' in units:
    igt = np.asarray(units['is_good_trials'][()])
    n_valid_cols = min(igt.shape[1], n_trials)
    trial_good_frac = igt[good_inds, :n_valid_cols].mean(axis=0) if len(good_inds) else np.zeros(n_valid_cols)
    valid_trial_mask[:] = False
    valid_trial_mask[:n_valid_cols] = trial_good_frac > 0
```

```python
if np.all(neural == 0):
    continue
```

iii. The CONVERSION_NOTES documents that `is_good_trials` had fewer columns than go cue events in some sessions, causing many late trials to have all-zero neural data. The AI resolved this by filtering to valid columns and dropping residual all-zero trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` for units with `classification == 'good'`. Spike times are read per unit via a helper function that handles ragged indexing.

ii.
```python
good_spike_times = [get_unit_spike_times(units, int(ui)) for ui in good_inds]
```

```python
def get_unit_spike_times(units_group, unit_index):
    st_ds = units_group['spike_times']
    ...
    idx = np.asarray(units_group['spike_times_index'][()])
    start = 0 if unit_index == 0 else idx[unit_index - 1]
    end = idx[unit_index]
    return np.asarray(st_all[start:end], dtype=float).ravel()
```

iii. The AI identifies `spike_times` as the source of neural data, consistent with the reference.

## 2-b. How is the `neural` data processed?

i. For each trial and each good unit, spike times are histogrammed into 50 ms bins spanning [-2.5, +1.5] s relative to the go cue using `np.histogram`. Counts are divided by bin width (0.05 s) to get firing rates in Hz.

ii.
```python
for tr in range(n_trials):
    ...
    for j, st in enumerate(good_spike_times):
        rel = st - gt
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
        neural[j] = counts.astype(np.float32) / BIN_SIZE
```

iii. This implements the standard spike-to-rate conversion. The double loop (trials x units) is much less efficient than the reference's vectorized approach but produces the same result.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with no good units would be skipped (0 good units leads to empty neural data, caught by the `len(neural) < 2 or neural[0].shape[0] == 0` check).

ii.
```python
unit_class = _decode_arr(units['classification'][()]) if 'classification' in units else None
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
good_inds = np.flatnonzero(good_mask)
```

iii. This matches the reference's use of the QC classifier verdict from `ChenLiuEtAl2023_SpikeSortingQC.pdf`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to the go cue by subtracting the go cue time from each spike time, then histogramming the relative times into the bin edges defined around t=0 (the go cue).

ii.
```python
gt = go_times[tr]
abs_edges = gt + BIN_EDGES
...
rel = st - gt
counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The alignment approach is correct - all trials share the same relative time grid centered on the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins from -2.5 s to +1.5 s relative to go cue, yielding 80 time bins per trial. No rebinning is applied; spikes are binned directly at this resolution.

ii.
```python
BIN_SIZE = 0.05
WIN_START = -2.5
WIN_END = 1.5
N_BINS = int(round((WIN_END - WIN_START) / BIN_SIZE))
BIN_EDGES = np.linspace(WIN_START, WIN_END, N_BINS + 1)
```

iii. Matches the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times` (tone onset event series) and go cue times. The AI uses an `infer_trial_event_times` function that finds the last sample onset before each go cue (with constraints: at least 0.05 s and at most 5.0 s before).

ii.
```python
sample_times = np.asarray(beh_events['sample_start_times']['timestamps'][()]).astype(float) if 'sample_start_times' in beh_events else None
```

```python
if sample_times is None or len(sample_times) != n_trials:
    sample_times = infer_trial_event_times(sample_times, go_times)
else:
    sample_times = infer_trial_event_times(sample_times[:], go_times)
```

iii. The AI correctly identifies that `sample_start_times` can have more events than trials (due to early lick replays) and uses a matching function to find the correct tone per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center (absolute time), subtract the tone onset time to get time-from-tone.

ii.
```python
stime = sample_times[tr] if np.isfinite(sample_times[tr]) else (gt - 0.6)
time_from_tone = abs_centers - stime
```

iii. The fallback of `gt - 0.6` is used when no valid sample onset is found, approximating a typical sample-to-go delay.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both the neural bins and the tone input use the same bin centers (`gt + BIN_CENTERS`), so they are inherently aligned.

ii.
```python
abs_centers = gt + BIN_CENTERS
...
time_from_tone = abs_centers - stime
```

iii. Using the same time grid for both streams ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses two sources: (1) `photostim_onset` and `photostim_duration` from the trials table, and (2) `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents` as a fallback.

ii.
```python
photostim_start = np.asarray(beh_events['photostim_start_times']['timestamps'][()]).astype(float) if 'photostim_start_times' in beh_events else np.array([])
photostim_stop = np.asarray(beh_events['photostim_stop_times']['timestamps'][()]).astype(float) if 'photostim_stop_times' in beh_events else np.array([])
```

```python
if 'photostim_duration' in trials:
    pdur = _decode_arr(trials['photostim_duration'][()])[:n_trials]
    pon = _decode_arr(trials['photostim_onset'][()])[:n_trials] if 'photostim_onset' in trials else np.array(['N/A'] * n_trials, dtype=object)
```

iii. The AI planned for both trial-table and event-series sources.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. When the trial table has `photostim_onset != 'N/A'`, the AI uses it directly as an absolute time and compares against absolute bin centers. When `'N/A'`, it falls back to the event series. The result is a binary time series (1 when photostim is on, 0 otherwise).

ii.
```python
if pon[tr] != 'N/A' and pdur[tr] != 'N/A':
    try:
        pstart = float(pon[tr])
        pd = float(pdur[tr])
        photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
    except Exception:
        pass
else:
    for ps, pe in zip(photostim_start, photostim_stop):
        if pe >= abs_edges[0] and ps <= abs_edges[-1]:
            photostim_on |= ((abs_centers >= ps) & (abs_centers < pe))
```

iii. **Critical bug**: `photostim_onset` in the trial table is stored as a time offset relative to `start_time` (trial start), not as an absolute time. The AI compares this small relative value (e.g., ~0.5 s) against absolute bin center times (e.g., ~1000+ s), so the comparison never matches. This means photostim is effectively always 0 for stimulated trials. The fallback event-series path only runs for non-stimulated trials (where `pon[tr] == 'N/A'`), so it also produces all zeros. Photostimulation input is therefore broken.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The AI intends to use absolute bin centers for both neural and photostim, which would be correct if the photostim onset were properly converted to absolute time.

ii.
```python
abs_centers = gt + BIN_CENTERS
...
photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
```

iii. The alignment approach (shared time grid) is correct in principle, but the photostim onset value is wrong (see 4-b).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the first lick after the go cue by comparing `left_lick_times` and `right_lick_times` event series. The first lick within the response window `[go, go + WIN_END]` determines the choice.

ii.
```python
left_lick_times = np.asarray(beh_events['left_lick_times']['timestamps'][()]).astype(float) if 'left_lick_times' in beh_events else np.array([])
right_lick_times = np.asarray(beh_events['right_lick_times']['timestamps'][()]).astype(float) if 'right_lick_times' in beh_events else np.array([])
```

```python
lmask = (left_lick_times >= gt) & (left_lick_times < gt + WIN_END)
rmask = (right_lick_times >= gt) & (right_lick_times < gt + WIN_END)
lfirst = left_lick_times[lmask][0] if np.any(lmask) else np.inf
rfirst = right_lick_times[rmask][0] if np.any(rmask) else np.inf
if np.isfinite(lfirst) and (lfirst < rfirst):
    choice = 0  # left
elif np.isfinite(rfirst) and (rfirst < lfirst):
    choice = 1  # right
else:
    choice = 2  # no lick
```

iii. The AI chose to infer choice from raw lick event times rather than deriving it from `trial_instruction` + `outcome` (as the reference does).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left), 1 (right), 2 (no lick) based on the first lick after the go cue. The per-trial value is broadcast across all 80 time bins.

ii.
```python
out = np.vstack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
])
```

iii. The lick-event-based approach is an alternative to the instruction+outcome derivation. Both approaches should yield the same result in most cases, though edge cases (e.g., bilateral licks at the same time, or licks the NWB file may not record) could differ.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table.

ii.
```python
outcome = _decode_arr(trials['outcome'][()])[:n_trials]
...
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = outcome_map.get(outcome[tr], 0)
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values are mapped to integers: ignore=0, miss=1, hit=2. The per-trial value is broadcast across all 80 bins. Unknown values default to 0 (ignore) via `dict.get`.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = outcome_map.get(outcome[tr], 0)
...
np.full(N_BINS, out_val, dtype=np.int64),
```

iii. The mapping matches the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table.

ii.
```python
early_lick = _decode_arr(trials['early_lick'][()])[:n_trials]
```

iii. The trials table stores this directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String values mapped to integers: 'no early'=0, 'early'=1. Per-trial value broadcast across all bins. Unknown values default to 0 via `dict.get`.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early_val = early_map.get(early_lick[tr], 0)
...
np.full(N_BINS, early_val, dtype=np.int64),
```

iii. Matches the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking` data and timestamps. Column 1 is tongue y-position, column 2 is tracking likelihood.

ii.
```python
tongue = np.asarray(beh_ts['Camera0_side_TongueTracking']['data'][()]).astype(float)
tongue_t = np.asarray(beh_ts['Camera0_side_TongueTracking']['timestamps'][()]).astype(float)
y_col, vis_col = choose_tongue_columns(tongue)  # returns 1, 2
tongue_y = tongue[:, y_col]
tongue_vis = tongue[:, vis_col]
```

iii. The AI correctly identifies the tongue tracking time series and the column layout (x, y, likelihood).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.5 are marked not visible. Session-wide 40th and 60th percentiles are computed over raw visible frames. Per trial, the nearest frame to each bin center is selected (nearest-neighbor lookup via `searchsorted`), and its y-value is discretized against the percentiles.

ii.
```python
visible = np.isfinite(tongue_y) & np.isfinite(tongue_vis) & (tongue_vis > 0.5)
if np.any(visible):
    q40, q60 = np.quantile(tongue_y[visible], [0.4, 0.6])
```

```python
idx = np.searchsorted(tongue_t, abs_centers, side='left')
idx = np.clip(idx, 0, len(tongue_t) - 1)
ty = tongue_y[idx]
tv = tongue_vis[idx]
tongue_disc = np.full(N_BINS, 3, dtype=np.int64)
vis_now = np.isfinite(ty) & np.isfinite(tv) & (tv > 0.5)
tongue_disc[vis_now & (ty < q40)] = 0
tongue_disc[vis_now & (ty >= q40) & (ty <= q60)] = 1
tongue_disc[vis_now & (ty > q60)] = 2
```

iii. Two key differences from the reference: (1) percentiles are computed over raw visible frames rather than 50ms bin means, and (2) per-trial values use nearest-frame lookup rather than binned averaging.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Discretized into 4 classes: 0 (< 40th pct), 1 (40th-60th pct inclusive), 2 (> 60th pct), 3 (not visible). The boundary at q60 is inclusive in class 1 (`ty <= q60`), whereas the reference uses `np.digitize` which assigns the q60 boundary to class 2.

ii.
```python
tongue_disc[vis_now & (ty < q40)] = 0
tongue_disc[vis_now & (ty >= q40) & (ty <= q60)] = 1
tongue_disc[vis_now & (ty > q60)] = 2
```

iii. The categorization follows the instructions' description. The inclusive boundary at q60 is a minor difference from the reference.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps are searched to find the nearest frame to each bin center (`abs_centers = gt + BIN_CENTERS`). This uses the same bin grid as the neural data.

ii.
```python
idx = np.searchsorted(tongue_t, abs_centers, side='left')
idx = np.clip(idx, 0, len(tongue_t) - 1)
```

iii. The nearest-neighbor approach differs from the reference's bin-averaging approach but uses the same time grid, so alignment with neural data is maintained.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases handled:
- **Missing classification**: If `classification` is absent, all units are treated as good (fallback to all-True mask).
- **Missing tone onset**: Falls back to `go_time - 0.6` when no suitable sample event is found.
- **Missing tongue visibility**: Bins with low-likelihood frames get class 3 ("not visible").
- **All-zero neural trials**: Dropped post-hoc.
- **Unknown outcome/early_lick values**: Default to 0 via `dict.get`.

ii.
```python
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
```

```python
stime = sample_times[tr] if np.isfinite(sample_times[tr]) else (gt - 0.6)
```

```python
if np.all(neural == 0):
    continue
```

iii. The AI handles edge cases with fallbacks. The fallback of treating all units as good when `classification` is missing is concerning, as the session that lacks classification (sub-440958) has 1,852 units that were never quality-controlled.

## 10-a. What are the most time-consuming steps of the code?

i. The nested trial-by-unit loop for spike histogramming dominates computation time. The AI's CONVERSION_NOTES estimates ~8 s/session and ~23 min for full conversion (before optimization). Spike time extraction per unit is also expensive.

ii.
```python
for tr in range(n_trials):
    ...
    for j, st in enumerate(good_spike_times):
        rel = st - gt
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The nested loop computes one histogram call per (trial, unit) pair, which is O(n_trials * n_units) calls.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop over spike histograms could be vectorized by flattening bin edges across all trials and using `searchsorted` (as the reference does). The per-trial tongue discretization loop could also be vectorized.

ii.
```python
for tr in range(n_trials):
    ...
    for j, st in enumerate(good_spike_times):
        rel = st - gt
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. The reference vectorizes the trial dimension by creating a flat edge array for all trials at once, reducing the computation to one `searchsorted` per unit instead of one `histogram` per unit per trial.

## 10-c. What processing does the code repeat multiple times?

i. Within the trial loop, `st - gt` is computed for every trial for every unit, recomputing the subtraction each time rather than using a vectorized approach. Spike times are cached per unit but the per-trial subtraction is repeated.

ii.
```python
for tr in range(n_trials):
    gt = go_times[tr]
    ...
    for j, st in enumerate(good_spike_times):
        rel = st - gt
```

iii. The full spike time array is traversed once per trial, leading to repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `left_lick_times` and `right_lick_times` event series to derive choice, when choice could be derived more efficiently from `trial_instruction` + `outcome`. Additionally, `photostim_start_times`/`photostim_stop_times` event series are loaded as a fallback but only run for non-stimulated trials (producing zeros), making them functionally unused.

ii.
```python
left_lick_times = np.asarray(beh_events['left_lick_times']['timestamps'][()]).astype(float) if 'left_lick_times' in beh_events else np.array([])
right_lick_times = np.asarray(beh_events['right_lick_times']['timestamps'][()]).astype(float) if 'right_lick_times' in beh_events else np.array([])
```

iii. The lick event loading adds I/O overhead for an alternative derivation path.
