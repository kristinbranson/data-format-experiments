# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files in the `data/` directory using `pynwb`. It globs for all `sub-*_ses-*.nwb` files, sorts them, and processes each file via `load_nwb_session()` which opens the file with `pynwb.NWBHDF5IO`, extracts units, trials, behavioral events, and tongue tracking data. The NWB file handle is closed after loading all needed arrays into memory.

ii.
```python
def get_nwb_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))
```
```python
def load_nwb_session(nwb_path):
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    # ... extracts subject info, units, trials, behavioral events, tongue tracking
    io.close()
    return { ... }
```

iii. The AI noted that NWB is the published format for this dataset on DANDI. It identified 174 NWB files across 28 subjects.

## 1-b. How are the data split into subjects?

i. The AI reads `nwb.subject.subject_id` (a numeric string like `'440956'`) for each session and groups sessions by this ID. It also captures `nwb.subject.description` (e.g., `"SC015"`). Subjects are stored in an `OrderedDict` in the order they are first encountered (which is sorted file order).

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description
```
```python
subjects_set = OrderedDict()
# ...
sid = result['subject_id']
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
```

iii. The AI noted 28 subjects matching the paper's count.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are identified by the filename. The AI processes sessions in sorted file order.

ii.
```python
nwb_files = get_nwb_files()  # sorted glob
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
```

iii. The AI noted 174 NWB files total. After applying session selection criteria, 144 sessions are retained.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The AI reads all trial columns into a dictionary. Go cue times come from `BehavioralEvents/go_start_times`.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trials_data = {
    'start_time': trials['start_time'][:],
    'outcome': trials['outcome'][:],
    # ... etc
}
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. No explicit assertion that go_times count matches trials count (unlike the reference).

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three layers of filtering:
1. **Session-level**: Sessions must have >65% performance, >=50 correct left trials, >=50 correct right trials. Performance is computed on control trials (no photostim, no auto_water, no free_water, no early lick, no ignore).
2. **Trial-level**: Trials with `auto_water == 1` or `free_water == 1` are excluded.
3. **Recording range**: Trials whose go-cue-aligned window falls outside the min/max spike time range are excluded.

The AI does NOT use `obs_intervals` from the NWB file.

ii.
```python
# Session selection
if performance < MIN_PERFORMANCE:  # 0.65
    return None
if correct_left < MIN_CORRECT_LEFT:  # 50
    return None
if correct_right < MIN_CORRECT_RIGHT:  # 50
    return None

# Trial filtering
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

# Recording range filtering
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
```

iii. The AI cited the paper's session selection criteria (>65% performance, >=50 correct each direction). For trial filtering, it excluded auto_water and free_water as "artificial conditions that change task structure." The recording range filter was added to handle sessions where behavioral trials extend beyond the neural recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (via `VectorIndex` and `VectorData`). Only units with `classification == 'good'` are used, and additionally only those with a valid brain region mapping from `anno_name`.

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
# ...
spike_times_vi = units['spike_times']
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])
```

iii. Spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `np.histogram` with bin edges from -2.5s to +1.5s relative to the go cue. Counts are divided by bin_width to get firing rates in Hz. Processing is done per-neuron, per-trial in a nested loop.

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
for n in range(n_neurons):
    st = spike_times_list[n]
    for t in range(n_trials):
        go_t = go_times[t]
        abs_start = go_t + align_start
        abs_end = go_t + align_end
        idx_lo = np.searchsorted(st, abs_start, side='left')
        idx_hi = np.searchsorted(st, abs_end, side='left')
        if idx_hi > idx_lo:
            rel_spikes = st[idx_lo:idx_hi] - go_t
            counts, _ = np.histogram(rel_spikes, bins=bin_edges)
            all_matrices[t, n, :] = counts / bin_width
```

iii. The AI noted using `np.linspace` for bin edges (vs `np.arange` in the reference). No smoothing or normalization is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two filters: (1) `classification == 'good'` and (2) the unit must have an `anno_name` that maps to one of the 14 predefined major brain regions. Units with unmapped annotations are excluded.

ii.
```python
good_mask = classifications == 'good'
# ...
good_anno = data['good_anno_names']
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)
```

iii. The AI documented finding unmapped annotations and iteratively fixing them. The dual filter (classification + region mapping) means some `'good'` units may be dropped if their annotation doesn't match any keyword in `REGION_MAPPING`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue. For each trial, the absolute go-cue time is used to define the window, spikes within that window are extracted, converted to relative times by subtracting the go-cue time, and then histogrammed.

ii.
```python
go_t = go_times[t]
abs_start = go_t + align_start   # go_t - 2.5
abs_end = go_t + align_end       # go_t + 1.5
rel_spikes = st[idx_lo:idx_hi] - go_t
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```

iii. All timestamps in NWB share a common session-absolute clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, producing 80 timepoints per trial (-2.5s to +1.5s). The bin edges are computed with `np.linspace`.

ii.
```python
BIN_WIDTH = 0.050
ALIGN_START = -2.5
ALIGN_END = 1.5
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
```

iii. The AI noted that the 50ms bin width is specified by the decoder task, overriding the reference code's 40ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onset timestamps) in `BehavioralEvents`, together with go-cue times. For each trial, the last `sample_start_times` entry within the trial's time range and before the go cue is taken as the tone onset.

ii.
```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
# ...
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    for i in range(n_trials):
        mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
        matching = sample_start_ts[mask]
        if len(matching) > 0:
            tone_onsets[i] = matching[-1]
```

iii. The AI correctly identified that early lick replays the sample epoch, so the last sample start before the go cue is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, time from tone onset = (absolute time of bin center) - tone_onset = bin_center - (tone_t - go_t). This is computed as `bin_centers - tone_rel` where `tone_rel = tone_t - go_t`.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
# ...
tone_rel = tone_t - go_t  # tone onset relative to go cue (negative)
tone_input = (bin_centers - tone_rel).astype(np.float32)
```

iii. The AI handles the case where no tone onset is found by setting the input to zeros (should not happen for valid trials).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for neural data are used for computing time from tone onset, ensuring alignment. Both use `np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)`.

ii.
```python
bin_centers = np.linspace(ALIGN_START + BIN_WIDTH/2, ALIGN_END - BIN_WIDTH/2, N_BINS)
```

iii. Using the same bin grid guarantees alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents` (absolute session timestamps), NOT from the per-trial `photostim_onset`/`photostim_duration` fields in the trials table.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The AI chose to use the event-level timestamps rather than the trial-level fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, ALL photostim events are checked against the trial window. For each bin, if any photostim event's [start, stop) interval contains the bin center, the bin is set to 1. This is done with a triple-nested loop (trials x photostim events x bins).

ii.
```python
def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts, ...):
    for t in range(n_trials):
        go_t = go_times[t]
        ps = np.zeros(n_bins, dtype=np.float32)
        for si in range(len(photostim_start_ts)):
            ps_start = photostim_start_ts[si] - go_t
            ps_stop = photostim_stop_ts[si] - go_t
            if ps_stop < align_start or ps_start > align_end:
                continue
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0
        photostim_trials.append(ps)
```

iii. The AI iterates over ALL session photostim events for each trial, which is inefficient but functionally produces the same result since non-overlapping events will be skipped.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim event times are converted to go-cue-relative by subtracting `go_t`, then compared against the same bin centers used for neural data.

ii.
```python
ps_start = photostim_start_ts[si] - go_t
ps_stop = photostim_stop_ts[si] - go_t
```

iii. Same bin grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` only (`'left'` or `'right'`). It is NOT derived from the combination of `trial_instruction` and `outcome`.

ii.
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. The AI maps the instructed direction directly to choice, treating it as the animal's lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instructed side is mapped to `left=0, right=1` with only 2 classes. There is no "no lick" class for ignore trials. The per-trial value is broadcast across all 80 time bins.

ii.
```python
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
# ...
out = np.array([
    np.full(N_BINS, choices[t], dtype=np.int64),  # choice
    ...
])
```
```python
'output_values': [
    ['left', 'right'],  # choice: only 2 values
    ...
]
```

iii. The AI treats choice as the instructed side, which is incorrect for miss trials (where the animal licked the opposite side) and for ignore trials (where the animal did not lick).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcomes_raw = td['outcome'][trial_indices]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

iii. Straightforward mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Maps strings to integers: ignore=0, miss=1, hit=2. Uses `.get(o, 0)` as a default (defaults to ignore for unknown values). The per-trial value is broadcast across all time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
out = np.array([
    ...
    np.full(N_BINS, outcomes[t], dtype=np.int64),
    ...
])
```

iii. Matches the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing `'no early'` and `'early'`.

ii.
```python
early_lick_raw = td['early_lick'][trial_indices]
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```

iii. Straightforward mapping from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Maps to no=0, yes=1. Broadcast across all time bins.

ii.
```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
out = np.array([
    ...
    np.full(N_BINS, early_licks[t], dtype=np.int64),
    ...
])
```

iii. Matches the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is `(n_frames, 3)` = x, y, confidence, with matching timestamps.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]  # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies a confidence threshold of 0.9 (vs 0.5 in reference). When tongue is not visible (confidence < 0.9), the y-position is **imputed with the session mean** of visible frames, rather than set to NaN. Percentiles (40th, 60th) are computed over ALL time bin values including imputed values. Discretization produces only 3 classes (0, 1, 2) with no "not visible" class.

ii.
```python
TONGUE_CONFIDENCE_THRESHOLD = 0.9
# ...
visible_mask = tongue_conf >= confidence_threshold
session_mean_y = np.mean(tongue_y[visible_mask])
tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y
```
```python
def discretize_tongue_y(tongue_y_trials, session_mean_y):
    all_values = np.concatenate([t for t in tongue_y_trials])
    p40 = np.percentile(all_values, 40)
    p60 = np.percentile(all_values, 60)
    # ...
    d = np.zeros(len(trial_y), dtype=np.int64)
    d[trial_y >= p40] = 1
    d[trial_y >= p60] = 2
```

iii. The AI cited the reference paper: "when the tongue was occluded... we set the tongue position to its mean value." This approach imputes rather than excludes, and percentiles are computed including imputed values. The AI also handles the edge case where p40==p60 by adding a small epsilon.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes using 40th and 60th percentiles of all tongue y values (including imputed session-mean values): 0 = below 40th percentile, 1 = 40th to 60th percentile, 2 = above 60th percentile. No "not visible" class (class 3).

ii.
```python
p40 = np.percentile(all_values, 40)
p60 = np.percentile(all_values, 60)
d = np.zeros(len(trial_y), dtype=np.int64)
d[trial_y >= p40] = 1
d[trial_y >= p60] = 2
```
```python
'output_values': [
    ...
    ['low', 'mid', 'high'],  # tongue_y: only 3 values
]
```

iii. Since non-visible frames are imputed with the session mean, there is no need for a "not visible" class. However, this means the percentile distribution is heavily influenced by imputed values (the tongue is only visible ~10% of frames).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock. For each trial, tongue data within [go_t + ALIGN_START, go_t + ALIGN_END) is selected, binned using `np.digitize` with the same bin edges, and averaged per bin.

ii.
```python
window_start = go_t + align_start
window_end = go_t + align_end
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
trial_y = tongue_y_imputed[mask]
bin_indices = np.digitize(trial_ts, bin_edges) - 1
```

iii. Same bin grid ensures alignment with neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session with no good units**: Skipped (returns None).
- **Session failing performance criteria**: Skipped.
- **Neurons with unmapped brain region**: Excluded from the session.
- **Trials with auto_water or free_water**: Excluded.
- **Trials beyond recording range**: Excluded by checking min/max spike times.
- **Tongue not visible**: Imputed with session mean y-value.
- **Tongue percentile collapse** (p40==p60): Small epsilon added.

ii.
```python
if n_neurons < 1:
    return None
# ...
if performance < MIN_PERFORMANCE:
    return None
# ...
tongue_y_imputed[~visible_mask] = session_mean_y
# ...
if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
    p40 = p40 - eps
    p60 = p60 + eps
```

iii. The AI documented handling these edge cases in CONVERSION_NOTES.md. Notably, it does NOT handle the case where `classification` is NaN (the un-QC'd session) via a text-conversion helper like the reference does; instead it relies on the comparison `classifications == 'good'` returning False for NaN values, which works but is implicit.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified NWB file loading (~1.5s/session), spike binning (~3.3s/session), and tongue processing (~1.5s/session). Total estimated time was ~22 minutes for the full dataset. The AI optimized tongue processing from 142s to 1.5s and spike binning from 14s to 3s.

ii. N/A (timing info from CONVERSION_NOTES.md)

iii. The AI documented timing per step and optimized bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has a nested loop: per-neuron and per-trial, using `np.histogram` per trial. The reference vectorizes the trial dimension by flattening all bin edges and using a single `np.searchsorted` per neuron. The tongue processing also loops per-trial with a per-bin inner loop.

ii.
```python
for n in range(n_neurons):
    for t in range(n_trials):
        # ... np.histogram per trial
```
```python
for b in range(n_bins):
    in_bin = trial_y[bin_indices == b]
    if len(in_bin) > 0:
        trial_tongue_y[b] = np.mean(in_bin)
```

iii. The AI noted optimizing these from initial versions but the nested neuron x trial loop remains.

## 10-c. What processing does the code repeat multiple times?

i. The photostim computation iterates over ALL session photostim events for every trial, which means each event is checked against every trial window even though most don't overlap. The tone onset computation also loops over all sample_start_ts events per trial.

ii.
```python
for t in range(n_trials):
    for si in range(len(photostim_start_ts)):
        # check every photostim event for every trial
```
```python
for i in range(n_trials):
    mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
```

iii. These are functionally correct but inefficient compared to vectorized approaches.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads several data fields that are not used in the final output: `left_lick_times`, `right_lick_times`, `photostim_power`, `stop_time`, `subject_desc`, `nwb_path`. The session performance computation (involving control trial filtering, correct L/R counting) is done for session selection but the performance value is only stored in metadata, not used downstream.

ii.
```python
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
# ...
'photostim_power': trials['photostim_power'][:],
```

iii. These extra fields add I/O overhead but do not contribute to the final converted data.
