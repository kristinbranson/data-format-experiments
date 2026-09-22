# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by recursively globbing `*.nwb` under `/app/data`. Each file is opened with `pynwb.NWBHDF5IO` and processed in `process_session()`. Trials are read from `nwb.trials`, units from `nwb.units`, and behavioral events from `nwb.acquisition['BehavioralEvents']`.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
if args.sample:
    files = files[:2]
```

```python
io = NWBHDF5IO(str(path), 'r', load_namespaces=True)
nwb = io.read()
```

```python
trial_starts, trial_stops, td = infer_trial_intervals(nwb)
go_times = assign_events_to_trials(get_event_times(nwb, 'go_start_times'), trial_starts, trial_stops)
```

iii. The AI recognized that the dataset is organized as one NWB file per session in subject-specific directories. The CONVERSION_NOTES.md documents 174 NWB files across 28 subjects. Using `pynwb` is the standard approach for reading NWB files.

## 1-b. How are the data split into subjects?

i. The subject is extracted from `nwb.subject.subject_id` via `get_subject_name()`. Subjects are accumulated into a list as sessions are processed, with a mapping dictionary for index lookup.

ii.
```python
def get_subject_name(nwb, path):
    sid = getattr(getattr(nwb, 'subject', None), 'subject_id', None)
    return sid if sid is not None else path.parent.name
```

```python
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_to_idx[subj])
```

iii. The AI uses `subject_id` from the NWB file, falling back to the parent directory name. Subjects are accumulated in encounter order rather than sorted.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed in sorted file order and each session's data is appended to the output lists.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
```

```python
for path in files:
    sess = process_session(path, show_processing=args.show_processing)
```

iii. The one-file-per-session structure is inherent to the dataset. Session info is recorded in `metadata['session_info']`.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. Go cue times are assigned to trials by finding the first `go_start_times` event within each trial's `[start_time, stop_time]` interval. Trials without a valid (finite) go time are skipped.

ii.
```python
def assign_events_to_trials(event_times, trial_starts, trial_stops):
    out = np.full(len(trial_starts), np.nan, dtype=float)
    for i, (a, b) in enumerate(zip(trial_starts, trial_stops)):
        m = (event_times >= a) & (event_times <= b)
        if np.any(m):
            out[i] = event_times[m][0]
    return out
```

```python
for i, go in enumerate(go_times):
    if not np.isfinite(go):
        continue
```

iii. The AI assigns events to trials by searching within trial boundaries. The reference uses a direct assertion that `len(go) == len(trials)` since there is exactly one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. The only trial filter is skipping trials where the go cue time is NaN (not finite). There is no filtering based on `obs_intervals`, `free_water`, or any other quality criterion. Sessions with fewer than 2 valid trials or 0 kept units are skipped entirely.

ii.
```python
for i, go in enumerate(go_times):
    if not np.isfinite(go):
        continue
```

```python
if len(neural_trials) < 2 or len(reg_kept) == 0:
    print('SKIP insufficient trials or units', path)
    continue
```

iii. The AI does not document specific trial quality filtering decisions. The CONVERSION_NOTES.md mentions that "early-lick and no-response trials were excluded for analysis" in the papers but notes these should be retained for the decoder. However, the AI misses the critical `obs_intervals` and `free_water` filters that exclude trials without spike data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']`, read per unit via indexed access. The go cue times from `BehavioralEvents/go_start_times` define the alignment.

ii.
```python
def spike_times_list(nwb, unit_mask):
    idx = np.where(unit_mask)[0]
    spikes = []
    for i in idx:
        spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))
    return idx, spikes
```

iii. Spike times are the only neural representation in the NWB file, consistent with the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins from -2.5 s to +1.5 s relative to the go cue using `np.histogram`, then divided by the bin width to get firing rates in Hz.

ii.
```python
def bin_spikes_for_trial(spike_times, align_time):
    arr = np.zeros((len(spike_times), len(BIN_CENTERS)), dtype=np.float32)
    rel_edges = align_time + BIN_EDGES
    for i, st in enumerate(spike_times):
        arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
    return arr
```

iii. The approach is functionally correct: bin spikes and convert to Hz. However, it processes one trial at a time (called in a per-trial loop), unlike the reference which vectorizes across all trials per unit using `searchsorted`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses a generic `find_first` function to search for a quality column, with candidates `['good', 'quality', 'label', 'unit_quality']`. Due to substring matching, `find_first` actually matches `is_good_trials` (since 'good' is a substring of 'is_good_trials'), which is a per-trial boolean indicator, not a per-unit quality label. This means the unit mask is derived from a completely wrong column.

ii.
```python
def get_unit_mask_and_regions(nwb):
    cols = list(nwb.units.colnames)
    good_col = find_first(cols, ['good', 'quality', 'label', 'unit_quality'])
    ...
    if good_col is not None:
        vals = np.asarray(nwb.units[good_col][:])
        if vals.dtype.kind in 'OUS':
            sval = np.array([str(v).lower() for v in vals])
            mask = np.array([('good' in v) or (v == '1') or (v == 'true') for v in sval], dtype=bool)
        else:
            mask = vals.astype(bool)
```

iii. The AI's `find_first` function searches column names by substring match. With the actual NWB column list, the first candidate 'good' substring-matches `is_good_trials` before reaching `unit_quality`. `is_good_trials` is a ragged per-unit array of per-trial booleans, not a unit quality label. The reference explicitly uses `classification == 'good'`, the verdict of the spike-sorting QC classifier described in the white paper. The AI's approach uses the wrong column entirely, and `region_col` also returns `None` (no column name contains 'location', 'brain_region', 'structure', 'ccf_acronym', or 'acronym'), so all brain regions are set to 'unknown'. The reference uses `anno_name`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue by adding the bin edges to the go cue time for each trial. The go cue time is found by searching `go_start_times` events within each trial's time interval.

ii.
```python
rel_edges = align_time + BIN_EDGES
for i, st in enumerate(spike_times):
    arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
```

iii. The alignment approach is correct in principle: bins are placed relative to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50 ms, spanning -2.5 s to +1.5 s relative to the go cue, giving 80 time bins. The bin edges are computed as `np.arange(T_START, T_END + BIN + 1e-9, BIN)`.

ii.
```python
BIN = 0.05
T_START = -2.5
T_END = 1.5
BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)
BIN_CENTERS = BIN_EDGES[:-1] + BIN / 2
```

iii. The bin parameters match the instructions. However, the use of `T_END + BIN + 1e-9` in `np.arange` produces 82 edges (81 bins) instead of the correct 81 edges (80 bins), because the final edge is at 1.55 s instead of 1.5 s. This is a subtle but impactful bug.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents`, assigned to trials by finding the first event within each trial's `[start_time, stop_time]` interval. The go cue time is also used.

ii.
```python
sample_times = assign_events_to_trials(get_event_times(nwb, 'sample_start_times'), trial_starts, trial_stops)
```

iii. The AI uses the first sample_start_times event within each trial interval, while the reference uses the last one before each go cue. Since early licks replay the sample epoch, there can be multiple tone onsets per trial; the reference takes the last one (the one the animal actually responds to).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as the absolute time of each bin center minus the tone onset time. If the tone time is not finite, a fallback of `go - 1.5` is used.

ii.
```python
def build_inputs(go_time, sample_start, photostim_intervals):
    tone_time = sample_start
    time_from_tone = (go_time + BIN_CENTERS) - tone_time
    ...
```

```python
input_trials.append(build_inputs(go, sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5, phot_int))
```

iii. The fallback `go - 1.5` for missing tone times is undocumented and arbitrary. The reference has no such fallback because tone times are always available.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers defined relative to the go cue, so the time from tone onset array has the same number of time points as the neural data.

ii.
```python
time_from_tone = (go_time + BIN_CENTERS) - tone_time
```

iii. Alignment is correct in principle — both share the same go-cue-relative bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents`, paired into intervals.

ii.
```python
def get_photostim_intervals(nwb):
    keys = list(nwb.acquisition['BehavioralEvents'].time_series.keys())
    if 'photostim_start_times' not in keys or 'photostim_stop_times' not in keys:
        return np.empty((0, 2), dtype=float)
    s = get_event_times(nwb, 'photostim_start_times')
    e = get_event_times(nwb, 'photostim_stop_times')
    n = min(len(s), len(e))
    return np.c_[s[:n], e[:n]] if n else np.empty((0, 2), dtype=float)
```

iii. The AI uses the global event streams for photostimulation, while the reference uses per-trial columns `photostim_onset` and `photostim_duration` from the trials table. The global approach is viable but less direct.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is created where bin centers falling within any photostim interval are set to 1.0, otherwise 0.0.

ii.
```python
def build_inputs(go_time, sample_start, photostim_intervals):
    ...
    phot = np.zeros(len(BIN_CENTERS), dtype=np.float32)
    abs_centers = go_time + BIN_CENTERS
    for a, b in photostim_intervals:
        phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0
    return np.vstack([time_from_tone.astype(np.float32), phot])
```

iii. The binary time series approach matches the reference. However, the AI checks ALL photostim intervals for every trial (not just the relevant one), and uses `<=` for the stop boundary while the reference uses `<`. Also, since photostim intervals are global, they may not be correctly assigned to specific trials.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The same bin centers relative to the go cue are used, ensuring alignment.

ii. Same `BIN_CENTERS` array used for neural and photostim.

iii. Alignment is correct — both use the same go-cue-relative grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI uses a generic `find_first` to search for a choice column with candidates `['choice', 'lick_direction', 'response_side']`. None of these exist in the NWB trials table. If not found, a fallback of `2` (no_lick) is used for all trials.

ii.
```python
def infer_trial_labels(td):
    cols = list(td.keys())
    choice_col = find_first(cols, ['choice', 'lick_direction', 'response_side'])
    ...
```

```python
choice = map_choice(td[choice_col][i]) if choice_col is not None else 2
```

iii. The NWB trials table has no explicit choice column. The reference derives choice from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore). The AI's `find_first` will not find a match, so all trials get choice=2 (no_lick). The trajectory confirms this: verification showed "Choice is constant no_lick for all data." The AI attempted to fix this but ran out of time.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The `map_choice` function maps string values to integers: 'left'->0, 'right'->1, and various no-lick indicators->2. However, since no choice column is found, this function is never called, and all trials get choice=2.

ii.
```python
def map_choice(v):
    s = str(v).lower()
    if 'left' in s: return 0
    if 'right' in s: return 1
    if 'no' in s or 'ignore' in s or 'miss' in s or s in ('nan', ''): return 2
    ...
```

iii. Even if a choice column existed, the `map_choice` function maps 'miss' to no_lick (2), which is incorrect. A miss means the animal licked the wrong side, not that it didn't lick. The reference correctly derives the licked side from instruction x outcome.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, found via `find_first(cols, ['outcome', 'trial_outcome', 'result', 'correctness'])`.

ii.
```python
outcome_col = find_first(cols, ['outcome', 'trial_outcome', 'result', 'correctness'])
```

```python
outcome = map_outcome(td[outcome_col][i]) if outcome_col is not None else 0
```

iii. The `outcome` column exists and would be found by `find_first`. This matches the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The `map_outcome` function maps 'ignore'->0, 'miss'->1, 'hit'->2.

ii.
```python
def map_outcome(v):
    s = str(v).lower()
    if 'ignore' in s: return 0
    if 'miss' in s: return 1
    if 'hit' in s or 'correct' in s or s == '1': return 2
    ...
```

iii. The mapping matches the reference for the three expected values.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, found via `find_first(cols, ['early', 'early_lick'])`.

ii.
```python
early_col = find_first(cols, ['early', 'early_lick'])
```

iii. The `early_lick` column exists and would be found. This matches the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The `map_early` function maps values to 0 (no) or 1 (yes). It checks for 'early' in the string, which would match both 'early' and 'no early'.

ii.
```python
def map_early(v):
    s = str(v).lower()
    if 'true' in s or 'yes' in s or s == '1' or 'early' in s:
        return 1
    try:
        return int(bool(int(v)))
    except Exception:
        return 0
```

iii. The condition `'early' in s` matches both `'early'` and `'no early'`, so ALL trials would be mapped to 1 (yes). The trajectory confirms: "early_lick is constant yes." The reference uses a direct dictionary mapping: `{'no early': 0, 'early': 1}`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 as y and column 2 as likelihood/visibility.

ii.
```python
def tongue_series(nwb):
    ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
    data = np.asarray(ts.data[:])
    t = np.asarray(ts.timestamps[:], dtype=float)
    return t, data
```

iii. The source variable matches the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The tongue y values are interpolated to the bin centers using `np.interp`. Frames with likelihood <= 0.5 are marked not visible. The interpolated values are used for session-wide percentile computation and discretization.

ii.
```python
def interpolate_tongue_y(nwb, go_time):
    t, data = tongue_series(nwb)
    y = data[:, 1] if data.ndim > 1 and data.shape[1] > 1 else data[:, 0]
    vis = np.ones_like(y, dtype=bool)
    if data.ndim > 1 and data.shape[1] > 2:
        vis = np.asarray(data[:, 2] > 0.5)
    sample_t = go_time + BIN_CENTERS
    y_interp = np.interp(sample_t, t, y, left=np.nan, right=np.nan)
    vis_interp = np.interp(sample_t, t, vis.astype(float), left=0, right=0) > 0.5
    y_interp[~vis_interp] = np.nan
    return y_interp
```

iii. The AI uses interpolation to resample the tongue tracking to bin centers, while the reference averages frames within each bin. Interpolation does not account for the visibility threshold correctly: `np.interp` linearly interpolates between visible and non-visible frames, potentially producing spurious intermediate values. The reference sets non-visible frames to NaN before binning and computes bin means of only valid frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide percentiles (40th, 60th) are computed from all valid (finite) interpolated y values across all trials. Values below 40th percentile -> 0, between 40th-60th -> 1, above 60th -> 2, NaN -> 3 (not visible).

ii.
```python
def discretize_tongue_session(trial_y_list):
    all_y = np.concatenate([y[np.isfinite(y)] for y in trial_y_list ...])
    q40, q60 = np.percentile(all_y, [40, 60])
    ...
    d[m & (y < q40)] = 0
    d[m & (y >= q40) & (y <= q60)] = 1
    d[m & (y > q60)] = 2
```

iii. The discretization logic matches the reference in structure, but the percentiles are computed over interpolated per-bin values rather than bin-means of raw frames. This may result in different class boundaries. Also, the boundary conditions differ: the AI uses `<=` for the middle class upper bound (40th-60th inclusive), while the reference uses `np.digitize` which uses `<` for boundaries.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue data is interpolated directly to the neural bin centers (`go_time + BIN_CENTERS`), ensuring the same number of time points.

ii.
```python
sample_t = go_time + BIN_CENTERS
y_interp = np.interp(sample_t, t, y, left=np.nan, right=np.nan)
```

iii. Using the same bin centers ensures temporal alignment, though interpolation vs binning produces different values.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing go cue times result in the trial being skipped. Missing tone times use a fallback of `go - 1.5`. Sessions with no units or fewer than 2 trials are skipped. Tongue tracking NaN values are assigned category 3 (not visible).

ii.
```python
if not np.isfinite(go): continue
```
```python
sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5
```
```python
if len(neural_trials) < 2 or len(reg_kept) == 0:
    print('SKIP insufficient trials or units', path)
    continue
```

iii. The AI does not handle the case where `classification` is NaN for all units (the session that was never quality-controlled), because it uses `unit_quality` instead. It does not filter `obs_intervals` or `free_water`. The fallback `go - 1.5` for missing tones is arbitrary.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files and per-trial spike binning. The trajectory shows sample conversion took ~14,000 seconds for 2 sessions (due to per-trial per-unit histogram calls) and produced a 120 GB output, indicating severe inefficiency.

ii.
```python
def bin_spikes_for_trial(spike_times, align_time):
    arr = np.zeros((len(spike_times), len(BIN_CENTERS)), dtype=np.float32)
    rel_edges = align_time + BIN_EDGES
    for i, st in enumerate(spike_times):
        arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN
    return arr
```

iii. The per-trial, per-unit `np.histogram` call is extremely slow compared to the reference's vectorized `searchsorted` approach. The trajectory shows the AI spent most of its budget waiting for conversion to complete.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has two nested loops: one over trials and one over units. The reference vectorizes the trial dimension by computing all trial edges at once and using `searchsorted`. The `assign_events_to_trials` function also loops over all trials.

ii.
```python
for i, go in enumerate(go_times):
    ...
    neural_trials.append(bin_spikes_for_trial(spikes, go))
```

```python
def bin_spikes_for_trial(spike_times, align_time):
    ...
    for i, st in enumerate(spike_times):
        arr[i] = np.histogram(st, bins=rel_edges)[0]...
```

iii. The doubly-nested loop (trials x units) using `np.histogram` is the primary bottleneck. The reference avoids this by vectorizing across trials.

## 10-c. What processing does the code repeat multiple times?

i. The tongue tracking data is loaded from the NWB file once per trial via `interpolate_tongue_y(nwb, go)`, which calls `tongue_series(nwb)` each time, re-reading the entire tracking array for every trial.

ii.
```python
for i, go in enumerate(go_times):
    ...
    tongue_y_trials.append(interpolate_tongue_y(nwb, go))
```

```python
def interpolate_tongue_y(nwb, go_time):
    t, data = tongue_series(nwb)  # re-reads all data each call
```

iii. The reference reads the tongue tracking data once per session and processes all trials from the cached array. The AI's approach re-reads the entire dataset for each trial, which is extremely wasteful.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reads all unit columns upfront in `get_trial_table_dict`, loading every column of the trials table even though only a few are needed. Also, the spike times are read per-unit in a loop, which is less efficient than reading the ragged array buffer once.

ii.
```python
def get_trial_table_dict(nwb):
    cols = list(nwb.trials.colnames)
    out = {}
    for c in cols:
        out[c] = np.asarray(nwb.trials[c][:])
    return out
```

iii. Loading all trial columns is wasteful but not a major bottleneck. The per-unit spike time reading is more consequential.
