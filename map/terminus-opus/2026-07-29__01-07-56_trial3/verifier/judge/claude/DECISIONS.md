# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files found by globbing `data/sub-*/sub-*.nwb`. Each NWB file is opened with `pynwb.NWBHDF5IO` and processed in `process_session()`. Trials, units, behavioral events, and tongue tracking are extracted from each file.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
# ...
beh_events = nwb.acquisition['BehavioralEvents']
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

iii. The AI notes in CONVERSION_NOTES.md that the dataset has 28 subject directories with 174 NWB files, and uses pynwb as the standard reader for NWB format.

## 1-b. How are the data split into subjects?

i. Each NWB file provides `nwb.subject.subject_id`, which is used to identify subjects. Unique subjects are collected and sorted to create the `subjects` list, and `subject_idx` maps each session to its subject.

ii.
```python
subject_id = nwb.subject.subject_id
# ...
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
```

iii. The AI identifies 28 unique subjects from the data, consistent with the dandiset metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file represents one session. The sorted glob provides all sessions, and each is processed independently in `process_session()`. Sessions may be skipped based on quality filters.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
    if session_data is not None:
        all_sessions.append(session_data)
```

iii. The AI notes that 174 NWB files were found, with 151 sessions surviving after behavioral filtering.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials`, which provides trial-level columns (instruction, outcome, early_lick, etc.). Go cue times from `BehavioralEvents/go_start_times` are used for temporal alignment. Trials are indexed and filtered before processing.

ii.
```python
n_trials_total = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
# ...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

iii. The AI uses the NWB trials table directly to define trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple filters:
1. **Neural coverage**: Trials are filtered by whether the go cue window falls within the recording range (determined from actual spike times of good units, not obs_intervals).
2. **Session-level behavioral filters**: Sessions are excluded if overall correct rate < 65%, or if there are fewer than 50 correct left or 50 correct right trials (computed on control, no-early-lick trials). These filters come from the reference paper's session selection criteria.
3. **Minimum unit count**: Sessions with fewer than 2 good units are skipped.

The AI does NOT filter `free_water` trials and does NOT use `obs_intervals`.

ii.
```python
# Recording range from spike times
rec_start, rec_end = get_recording_range(nwb, good_indices)
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)

# Session-level behavioral filters
correct_rate = hits / denom
if correct_rate < 0.65:
    return None
if correct_left < 50 or correct_right < 50:
    return None
```

iii. The AI's CONVERSION_NOTES.md states: "Session selection: >65% correct rate, >=50 correct L/R trials, >=2 good units" and "Filter trials by actual spike time range." The AI found these criteria from the reference papers' methods section.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']`, loaded per good unit. Go cue times from `go_start_times` are used to define time windows.

ii.
```python
spike_times_all = nwb.units['spike_times'][:]
spike_times_good = []
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

iii. The AI notes that spike_times is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins using `np.histogram` to compute firing rates in Hz. The function `compute_firing_rates_session` iterates over trials and neurons, computing histogram counts divided by bin width.

ii.
```python
def compute_firing_rates_session(spike_times_list, go_times, ...):
    for trial_idx in range(n_trials):
        go_time = go_times[trial_idx]
        bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
        for i, spikes in enumerate(spike_times_list):
            counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
            fr[i, :] = counts / bin_width
```

iii. The AI uses standard histogram binning to convert spike times to firing rates, consistent with the general approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with fewer than 2 good units are skipped.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
n_good = np.sum(good_mask)
if n_good < 2:
    return None
good_indices = np.where(good_mask)[0]
```

iii. The AI uses the classifier-based QC from `ChenLiuEtAl2023_SpikeSortingQC.pdf`, matching the reference code's 'classifier' mode.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. Bin edges are computed as `go_time + window_start + np.arange(n_bins + 1) * bin_width`, centering the window on each trial's go cue time.

ii.
```python
go_time = go_times[trial_idx]
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. All times in NWB are on the same session-absolute clock, so alignment is straightforward.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins spanning -2.5s to +1.5s relative to go cue, giving 80 time bins. No rebinning is applied - spikes are directly histogrammed into the final bins.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```

iii. These match the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI uses a **fixed constant** `TONE_ONSET_REL_GO = -1.85` rather than reading per-trial tone onset times from the data. This constant represents the assumed time of tone onset relative to the go cue.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES.md states: "Trial timing: sample -1.85 to -1.20, delay -1.20 to 0.00, response 0.00 to 1.50." The AI computed a fixed offset from the task structure (0.65s sample + 1.2s delay = 1.85s before go cue).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Since the AI uses a fixed offset, the time-from-tone-onset is simply `BIN_CENTERS - (-1.85) = BIN_CENTERS + 1.85`, a constant vector identical for every trial. This means it does not vary per trial and ignores the effect of early licks replaying the sample epoch.

ii.
```python
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
# This is the same for all trials - computed once outside the loop
```

iii. The AI treats tone onset as a fixed offset from the go cue based on the standard task timing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Since it uses the same `BIN_CENTERS` array as the neural data, alignment is automatic. However, the values themselves are incorrect for trials with early licks.

ii.
```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents` (the event time series), not from the trials table columns `photostim_onset` and `photostim_duration`.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
# ...
stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
```

iii. The AI reads photostim event times from the BehavioralEvents time series and matches them to trials by temporal overlap.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, photostim events overlapping the trial window are found. A binary time series is created where bin centers falling between start and stop times are set to 1. The comparison uses `>=` for start and `<=` for stop (inclusive on both ends).

ii.
```python
def get_photostim_timeseries(photostim_start_times, photostim_stop_times, go_time, bin_centers):
    abs_bin_centers = go_time + bin_centers
    for start, stop in zip(photostim_start_times, photostim_stop_times):
        mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
        photostim[mask] = 1.0
    return photostim
```

iii. The AI creates a binary time series per trial, marking bins where photostimulation is active.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Absolute bin centers are computed as `go_time + bin_centers`, using the same go-cue-aligned grid as the neural data.

ii.
```python
abs_bin_centers = go_time + bin_centers
```

iii. Alignment is through the shared bin center grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from `trial_instruction` (the instructed lick side), NOT from the animal's actual lick direction. It maps left=0, right=1.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
# ...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. The AI's CONVERSION_NOTES.md Step 5 states: "trial_instruction -> output[0]: left=0, right=1."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instructed side string is mapped to 0 (left) or 1 (right) using a dictionary. There are only 2 classes, with no "no lick" class for ignore trials. For ignore trials, the `.get()` default maps to 0 (left), which is incorrect - these trials had no lick.

ii.
```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
# ...
output_values = [['left', 'right'], ...]
```

iii. The AI treats choice as the instruction rather than the animal's actual behavior, and has only 2 output classes instead of 3.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column in the trials table, which contains 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = nwb.trials['outcome'][:]
# ...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. The outcome is read directly from the NWB trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2, matching the instructions.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
# ...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. Straightforward mapping matching the instruction specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column in the trials table, containing 'no early' and 'early'.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
# ...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. Read directly from the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1 using a dictionary.

ii.
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
```

iii. Straightforward mapping matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`, which has columns for x, y, and likelihood. The y-position (column 1) is used.

ii.
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
```

iii. Same source variable as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI extracts tongue y per trial by binning into 50ms bins and averaging. It does NOT filter by tracking likelihood - all tongue y values are used regardless of whether the tongue is visible. Percentiles are computed over all valid (non-NaN) raw values concatenated across trials, not over bin means across the whole session.

ii.
```python
def get_tongue_y_for_trial(tongue_ts, tongue_data, go_time, ...):
    local_y = tongue_data[idx_start:idx_end, 1]  # No likelihood filtering
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            tongue_y[b] = np.mean(local_y[mask])
    return tongue_y

def discretize_tongue_y(tongue_y_session, ...):
    all_values = np.concatenate(all_values)  # Concatenate bin means from trials
    p_low = np.percentile(all_values, percentile_low)
    p_high = np.percentile(all_values, percentile_high)
```

iii. The AI's CONVERSION_NOTES.md Step 5 states: "Use all tongue y values regardless of likelihood for percentile computation."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI discretizes into 3 classes: <40th percentile=0, 40th-60th percentile=1, >60th percentile=2. There is no "not visible" class - bins without tongue data default to class 1 (middle).

ii.
```python
def discretize_tongue_y(tongue_y_session, percentile_low=40, percentile_high=60):
    discrete = np.ones(len(y), dtype=np.int64)  # default middle
    discrete[valid & (y < p_low)] = 0
    discrete[valid & (y >= p_low) & (y <= p_high)] = 1
    discrete[valid & (y > p_high)] = 2
    # NaN bins stay as 1 (middle class)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no_early', 'early'],
    ['low', 'mid', 'high'],  # Only 3 classes, no 'not visible'
],
```

iii. The AI uses 3 classes without a "not visible" category. Bins with no tongue data default to the middle class rather than getting their own category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps (on the same session clock as spikes) are binned using `np.digitize` against the same bin edges as the neural data, aligned to the go cue.

ii.
```python
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
```

iii. Uses the same go-cue-aligned time grid as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three approaches:
- Sessions with classification not equal to 'good' strings: The AI reads `classification[:]` which may return NaN for unlabeled sessions. The comparison `== 'good'` fails for NaN, so those units are excluded. If no good units remain (or < 2), the session is skipped.
- Trials outside recording range: Filtered by comparing go cue windows against the actual spike time range of good units.
- Missing tongue tracking data: Bins with no data remain NaN, which maps to the default middle class (1).

ii.
```python
good_mask = classification == 'good'
n_good = np.sum(good_mask)
if n_good < 2:
    return None

# Recording range check
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
```

iii. The AI handles missing data by skipping sessions without sufficient good units and filtering trials by recording coverage.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports total conversion time of 1143.9s (19.1 minutes). Firing rate computation is the dominant cost, taking 1.6-7.1s per session (compared to 1.0-1.5s for data loading). The nested trial-by-neuron loop in `compute_firing_rates_session` is the main bottleneck.

ii.
```python
# Nested loops: trial x neuron
for trial_idx in range(n_trials):
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
```

iii. The AI's CONVERSION_NOTES.md estimates and timing show firing rates dominating at ~850s total.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main firing rate computation has a nested loop over trials and neurons. The trial loop could be vectorized by computing all bin edges at once (as the reference does with `(go[:, None] + REL_EDGES[None, :]).ravel()` and a single `searchsorted` per neuron). The tongue y binning loop over trials could also be vectorized.

ii.
```python
# AI's nested loop (trial x neuron)
for trial_idx in range(n_trials):
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)

# Reference's vectorized approach (only loops over neurons)
edges = (go[:, None] + REL_EDGES[None, :]).ravel()
for r, u in enumerate(good):
    pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1)
```

iii. The AI's approach is significantly slower due to the doubly-nested loop.

## 10-c. What processing does the code repeat multiple times?

i. The AI reads `spike_times_all = nwb.units['spike_times'][:]` once per session but also calls `get_recording_range()` which reads spike times separately via `nwb.units['spike_times'][:]`. This means the full spike times array is read twice per session.

ii.
```python
def get_recording_range(nwb, good_indices):
    spike_times_all = nwb.units['spike_times'][:]  # First read
    # ...

def process_session(nwb_file, ...):
    # ... later:
    spike_times_all = nwb.units['spike_times'][:]  # Second read
```

iii. Reading spike times is expensive I/O, and doing it twice wastes time.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session-level behavioral statistics (correct_rate, correct_left, correct_right) for session selection that the reference solution does not perform. These computations themselves are cheap, but the session-level filtering they drive excludes 22 sessions of valid data that the downstream decoder could use.

ii.
```python
denom = np.sum(control_no_early & is_not_ignore)
hits = np.sum(control_no_early & (outcome == 'hit'))
correct_rate = hits / denom
correct_left = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'left'))
correct_right = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'right'))
```

iii. The session selection criteria come from the reference paper's analysis pipeline but are not required by the decoder task instructions.
