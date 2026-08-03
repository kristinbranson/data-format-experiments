# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using `h5py` to directly read NWB files (HDF5 format). All NWB files are found by recursively globbing `data/**/*.nwb`, sorted, and processed sequentially. Each file is opened with `h5py.File()` and the relevant groups (`intervals/trials`, `units`, `acquisition`) are read directly from the HDF5 hierarchy.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
...
for fp in files:
    sess = load_session(fp, brain_regions)
```

```python
def load_session(nwb_path, brain_regions):
    with h5py.File(nwb_path, 'r') as f:
        trials = f['intervals/trials']
        n_trials = trials['id'].shape[0]
        ...
```

iii. The AI chose `h5py` over `pynwb` for reading NWB files. The CONVERSION_NOTES.md does not explicitly justify this choice, but it is a valid approach since NWB files are HDF5 files. The AI documented the data structure in Step 2 of CONVERSION_NOTES.md.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-440956`). The full directory name including the `sub-` prefix is used as the subject identifier.

ii.
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name
```

```python
subj = sess['subject']
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
```

iii. The AI uses the directory name as subject ID. The CONVERSION_NOTES.md notes 28 subjects were found in the data.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The session ID is taken from the filename stem (without extension). Sessions are processed in sorted file order.

ii.
```python
def get_session_id(nwb_path):
    return nwb_path.stem
```

iii. The AI documented 174 NWB files across 28 subjects in Step 2.

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in each NWB file. The number of trials is determined by `trials['id'].shape[0]`. Go cue times from `BehavioralEvents/go_start_times` are matched 1:1 with trials using `n_match = min(n_trials, len(go_times))`.

ii.
```python
trials = f['intervals/trials']
n_trials = trials['id'].shape[0]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
n_match = min(n_trials, len(go_times))
```

iii. The AI uses `min(n_trials, len(go_times))` rather than asserting equality, which silently handles any mismatch.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on three conditions: (1) the go time must be finite, (2) the trial window `[go + T_START, go + T_END]` must fall within `[trial_start, trial_stop]`, and (3) the trial must have valid values for `choice`, `outcome`, and `early_lick` (i.e., they must be in the expected maps). A session is dropped if fewer than 2 trials survive.

ii.
```python
for i in range(n_match):
    go = go_times[i]
    if not np.isfinite(go):
        continue
    if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
        continue
    if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
        continue
```

iii. The AI does not use `obs_intervals` or `free_water` filtering. CONVERSION_NOTES.md Step 10 mentions that discrepancies with the paper's 173-session / 69,943-unit counts are "likely because the NWB release includes extra non-ogen/extra-region sessions."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (flattened spike time array) and `units/spike_times_index` (per-unit index into the flattened array). Only units passing quality filtering contribute.

ii.
```python
spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
starts = np.concatenate([[0], spikes_index[:-1]])
kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]
```

iii. The AI reads the raw HDF5 datasets directly with h5py rather than using the pynwb abstraction layer.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins spanning [-2.5 s, +1.5 s] relative to the go cue (80 bins). For each unit and trial, spikes are found in the window using `searchsorted`, assigned to bins via `np.floor`, counted with `np.bincount`, and divided by the bin size to get firing rates in Hz.

ii.
```python
def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    a = np.searchsorted(spike_times, lo, side='left')
    b = np.searchsorted(spike_times, hi, side='left')
    if b <= a:
        return np.zeros(N_BINS, dtype=np.float32)
    rel = spike_times[a:b] - lo
    bins = np.floor(rel / BIN_SIZE).astype(np.int64)
    bins = bins[(bins >= 0) & (bins < N_BINS)]
    counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return counts / BIN_SIZE
```

iii. The binning approach produces firing rates in Hz. However, the function is called per-unit per-trial in a nested loop (units loop inside trials loop), which is less efficient than the reference's approach of vectorizing across all trials for each unit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters units using two criteria: (1) `unit_quality == 'good'` and (2) the unit's brain region (from electrode location JSON) must be in a hardcoded set of `MAJOR_REGIONS` (left/right ALM, Striatum, Thalamus, Midbrain, Medulla).

ii.
```python
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
unit_regions = elec_regions[unit_electrodes]
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
```

iii. The AI's CONVERSION_NOTES.md Step 4 notes that `unit_quality == 'good'` yields 154,948 units, and restricting to major regions "gives a total close to the reported 69,943." However, the converted data has 142,233 units, which is still far from 69,943. The reference uses `classification == 'good'` instead, which is a different field based on the QC classifier described in the spike sorting QC paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are binned relative to the go cue. The window `[go + T_START, go + T_END]` is computed for each trial's go cue time, and spikes within that window are binned.

ii.
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```
where `bin_unit_spikes_fast` uses `go_time + T_START` and `go_time + T_END`.

iii. The alignment is correct relative to the go cue as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50 ms, producing 80 bins over the 4-second window. This is defined at the module level.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. Matches the instructions' specification of 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` event timestamps and the go cue time for each trial. The AI finds the last sample event within the trial boundaries that occurs before the go cue.

ii.
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
```

iii. CONVERSION_NOTES.md Step 10 notes this was initially incorrect and was fixed by selecting the last sample event within each trial before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `BIN_CENTERS - tone_relative_to_go`. Negative values (before tone onset) are clamped to 0.

ii.
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The AI clamps negative values to 0, meaning bins before the tone onset are assigned a value of 0 rather than a negative time value.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time from tone onset uses the same `BIN_CENTERS` array as the neural data, which are defined relative to the go cue. This ensures alignment.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
rel = BIN_CENTERS - tone_rel
```

iii. Uses the same bin centers as neural data, ensuring temporal alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in the BehavioralEvents acquisition group, filtered to events within each trial's time boundaries.

ii.
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

iii. The AI uses the event-stream approach rather than the trial-table fields (`photostim_onset`, `photostim_duration`).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is constructed where bins overlapping with any photostim interval are set to 1. The overlap check uses bin edges rather than bin centers.

ii.
```python
def build_photostim_series(start_times, stop_times, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(start_times, stop_times):
        rs = s - go_time
        re = e - go_time
        overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
        x[overlap] = 1.0
    return x
```

iii. The AI checks for any overlap between the bin and the stim interval using bin edges, while the reference uses bin centers (`CENTERS >= stim_on` and `CENTERS < stim_off`).

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim event times are converted to go-cue-relative times before comparing with bin edges, which are also go-cue-relative.

ii.
```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. Alignment is achieved through the shared go-cue-relative coordinate system.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI uses `trial_instruction` directly from the trials table, mapped via `CHOICE_MAP = {'left': 0, 'right': 1}`. This is the *instructed* side, not the actual lick direction choice.

ii.
```python
choice = decode_arr(trials['trial_instruction'][:])
...
CHOICE_MAP = {'left': 0, 'right': 1}
...
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
```

iii. The AI maps `trial_instruction` directly to choice. This means the output represents the instructed direction, not the actual lick direction. For hit trials these are the same, but for miss trials the actual lick was to the opposite side. For ignore trials, the trial is silently skipped (since `choice[i]` will be `'left'` or `'right'`, always in `CHOICE_MAP`, but the skip happens via `outcome[i] not in OUTCOME_MAP` -- actually 'ignore' IS in OUTCOME_MAP, so ignore trials are kept but assigned the instructed direction rather than recognizing no lick occurred).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instruction string is mapped to 0 (left) or 1 (right) and repeated across all 80 time bins. There are only 2 output values, with no "no lick" class for ignore trials.

ii.
```python
'output_values': [
    ['left', 'right'],
    ...
],
```

```python
out = np.vstack([
    np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
    ...
])
```

iii. The AI defines only 2 choice classes (`left`, `right`), not 3 as in the reference which adds a `no lick` class for ignore trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome = decode_arr(trials['outcome'][:])
...
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
```

iii. Direct mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers (ignore=0, miss=1, hit=2) and repeated across all 80 bins.

ii.
```python
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
```

iii. Matches the instruction's encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing `'no early'` and `'early'`.

ii.
```python
early = decode_arr(trials['early_lick'][:])
...
EARLY_MAP = {'no early': 0, 'early': 1}
```

iii. Direct mapping from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are mapped to 0 (no early) or 1 (early), repeated across all 80 bins.

ii.
```python
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
```

iii. Matches the instruction's encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` data (column 1 = tongue y) and timestamps from `acquisition/BehavioralTimeSeries`.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. Same raw data source as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI interpolates the raw tongue y values to bin centers using `np.interp`, without applying any likelihood/confidence thresholding. Then percentiles (40th, 60th) are computed over all finite interpolated values across the session's trials, and bins are discretized into 3 classes (0: below 40th, 1: 40th-60th, 2: above 60th).

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```

```python
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()])
q40, q60 = np.percentile(all_tongue, [40, 60])
for ty, i in zip(sess_tongue_cont, valid_trial_inds):
    disc = np.full(N_BINS, 1, dtype=np.int64)
    disc[ty < q40] = 0
    disc[ty > q60] = 2
```

iii. The AI does NOT filter by tongue tracking likelihood/confidence. All tongue y values with finite positions are used, including those where the tongue is retracted and the tracker reports noise. The reference filters frames with likelihood < 0.5 before processing. Also, the AI uses linear interpolation to resample to bin centers, while the reference averages frames within each bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. 3 categories based on session-level 40th/60th percentiles. No "not visible" category is defined -- NaN values default to class 1 (middle).

ii.
```python
disc = np.full(N_BINS, 1, dtype=np.int64)
disc[ty < q40] = 0
disc[ty > q60] = 2
```

```python
'output_values': [
    ...
    ['low', 'mid', 'high'],
],
```

iii. The AI defines 3 tongue classes. The reference defines 4 classes including a "not visible" class (code 3) for bins with no confident tongue tracking. The AI's approach assigns NaN bins to class 1 (mid) by default rather than a separate class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are converted to go-cue-relative times and interpolated to the same `BIN_CENTERS` used for neural data.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    ...
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], ...)
```

iii. Uses `np.interp` to resample to bin centers, ensuring alignment with neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data in several ways: (1) trials with non-finite go times are skipped, (2) trials where the window extends beyond trial boundaries are skipped, (3) trials with unexpected string values for choice/outcome/early_lick are skipped, (4) trials without a valid sample event are skipped, (5) sessions with fewer than 2 valid trials are dropped. For tongue tracking, `np.interp` returns NaN for extrapolation outside the tracking range, and NaN values default to the middle discretization class.

ii.
```python
if not np.isfinite(go):
    continue
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
    continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
...
if not np.isfinite(sample_time):
    continue
...
if len(sess_neural) < 2:
    return None
```

iii. The AI's approach skips problematic trials rather than explicitly using `obs_intervals` or `free_water` flags as the reference does.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's code has a doubly-nested loop structure: for each trial, it loops over all kept units to bin spikes. This is substantially less efficient than the reference's approach of vectorizing across all trials per unit using a flattened edge array. Loading NWB files with h5py is also a significant cost.

ii.
```python
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. CONVERSION_NOTES.md mentions optimizing spike binning with `searchsorted` + `bincount`, but the per-trial loop remains.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning is done per-unit per-trial. The reference vectorizes across all trials for each unit by flattening all trial edges into one array and using a single `searchsorted` call per unit. The AI's code calls `bin_unit_spikes_fast` once per unit per trial.

ii.
```python
# Per-trial loop (outer), per-unit loop (inner):
for i in range(n_match):
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. This nested loop structure is the main efficiency bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The trial-level processing including spike binning, input construction, and tongue interpolation are all done inside a single trial loop, so there's no redundant recomputation of the same values. However, the per-trial spike binning recomputes `searchsorted` boundaries for each trial separately rather than doing it once for all trials.

ii.
```python
for i in range(n_match):
    go = go_times[i]
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. Each trial independently searches spike times, whereas the reference does this once per unit across all trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `left_lick_times` and `right_lick_times` event arrays but never uses them for any computation.

ii.
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
```

iii. These arrays are loaded but unused, wasting I/O and memory.
