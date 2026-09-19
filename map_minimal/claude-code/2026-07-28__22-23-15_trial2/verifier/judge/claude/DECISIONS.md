# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found with a glob over `sub-*/sub-*_ses-*.nwb`, and each file is opened with `pynwb` (`NWBHDF5IO`). Subjects, trials, and units are read from each file (`nwb.subject`, `nwb.trials`, `nwb.units`, `nwb.acquisition`).

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, verbose=verbose)
```

Loading one session:
```python
io = NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
trials = nwb.trials.to_dataframe()
be = nwb.acquisition['BehavioralEvents']
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The agent recognized that NWB is the published format and pynwb is the standard reader. It explored the data directory structure and found all NWB files using a glob pattern.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `nwb.subject.description` (e.g., `'SC015'`), which is used as the subject identifier. This is a descriptive name rather than the numeric `subject_id` field. Subjects are collected via an `OrderedDict` keyed by `subject_desc`.

ii.
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description  # e.g. 'SC015'
...
sub_desc = result['subject_desc']
if sub_desc not in all_subjects:
    all_subjects[sub_desc] = result['subject_id']
...
subjects = list(all_subjects.keys())
```

iii. The agent chose to use the descriptive subject name (e.g., `SC015`) rather than the numeric subject_id (e.g., `440956`). Both are valid identifiers present in the NWB file.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. Each session is identified by the filename. Session order follows the sorted file list.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
```

iii. The agent recognized that one NWB file corresponds to one session and no additional splitting is needed.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioural trial. The agent verifies that the number of go cue events matches the number of trials.

ii.
```python
trials = nwb.trials.to_dataframe()
n_trials_total = len(trials)
go_start_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_start_times) == n_trials_total
```

iii. The agent used the trials table directly as the source of trial boundaries, verifying consistency with the go cue event count.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple levels of filtering:
1. **Session-level filtering**: Sessions must have >65% correct performance on control trials, and at least 50 correct lick-left and 50 correct lick-right trials. Control trials are defined as no photostim, no auto_water, no free_water, no early lick, and responded (not ignore).
2. **Trial-level filtering by obs_intervals**: Only trials covered by neural recordings are kept. The agent matches obs_intervals to trials by start time.
3. **Trial-level filtering by auto_water and free_water**: Both auto_water and free_water trials are excluded.
4. **Minimum trial count**: Sessions with fewer than 2 valid trials are dropped.

ii.
```python
# Session filtering
all_valid = (auto_water == 0) & (free_water == 0)
control_mask_all = (all_valid &
                    (early_lick == 'no early') &
                    (outcome != 'ignore') &
                    no_photostim)
n_correct = ((outcome == 'hit') & control_mask_all).sum()
performance = n_correct / n_control
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None

# Trial filtering
for ti in recorded_trial_indices:
    if auto_water[ti] == 0 and free_water[ti] == 0:
        valid_trial_mask[ti] = True
```

iii. The agent adopted the session-level performance criteria from the reference paper (methodpaper.pdf), which states sessions were selected with ">65% performance" and "at least 50 correct lick left and lick right trials each." The agent also chose to exclude auto_water trials in addition to free_water trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units.get_unit_spike_times(ui)` for each good unit. The go cue times (`go_start_times`) are used to define the time window for each trial.

ii.
```python
all_spike_times = []
for ui in good_indices:
    st = units.get_unit_spike_times(ui)
    all_spike_times.append(st)
```

iii. The agent used the spike_times from the units table, which is the only neural representation in the NWB file.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `np.histogram` with bin edges spanning -2.5s to +1.5s relative to go cue. Spike counts are converted to firing rates (Hz) by dividing by bin width. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
for i, st in enumerate(all_spike_times):
    lo = np.searchsorted(st, abs_start)
    hi = np.searchsorted(st, abs_end)
    if hi > lo:
        rel_spikes = st[lo:hi] - go_time
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
        trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The agent used standard histogram-based spike counting with conversion to firing rates, which is the standard approach described in the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `classification == 'good'` AND the unit has a non-empty `anno_name` that can be mapped to one of the 14 broad brain regions. Units with unmappable or empty annotations are excluded.

ii.
```python
classification = units['classification'].data[:]
anno_names = units['anno_name'].data[:]

good_indices = []
unit_regions = []
for ui in range(n_units_total):
    if classification[ui] != 'good':
        continue
    region = map_anno_to_region(anno_names[ui])
    if region is not None:
        good_indices.append(ui)
        unit_regions.append(region)
```

iii. The agent filtered on both `classification` and `anno_name`. The `classification == 'good'` criterion comes from the QC classifier described in the spike sorting paper. The additional `anno_name` filter removes units without valid brain region annotations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go cue times are on the same session-absolute clock. For each trial, the go cue time is used to define the absolute time window, spikes within that window are extracted, and their times are converted to relative times before histogramming.

ii.
```python
go_time = go_start_times[trial_idx]
abs_start = go_time + BEGIN_TIME
abs_end = go_time + END_TIME
for i, st in enumerate(all_spike_times):
    lo = np.searchsorted(st, abs_start)
    hi = np.searchsorted(st, abs_end)
    if hi > lo:
        rel_spikes = st[lo:hi] - go_time
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The agent correctly used the go cue onset as the alignment event, as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50ms, spanning -2.5s to +1.5s relative to the go cue, giving 80 non-overlapping bins per trial. The bin edges are computed using `np.linspace`. No rebinning is applied.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The 50ms bin width and -2.5s to +1.5s window follow the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI uses a FIXED constant `TONE_ONSET_REL = -1.85` (1.85s before the go cue) for all trials and sessions, derived from the known sample period (0.65s) and delay period (1.2s). It does NOT use any per-trial tone onset times from the raw data.

ii.
```python
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. From the trajectory (step 44): "The sample period is 0.65s and delay period is 1.2s, so tone onset is always 1.85s before go cue. This is very consistent." The agent initially investigated per-trial variability but concluded that the timing is fixed. However, this ignores early lick replay trials where the actual sample_start preceding the go cue may differ.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `BIN_CENTERS - TONE_ONSET_REL`, giving a fixed time-varying signal that is identical for all trials. It is precomputed once as a constant array.

ii.
```python
TONE_ONSET_REL = -1.85
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)
```

iii. The agent treated tone onset as a fixed offset from the go cue rather than computing it per trial from the actual event timestamps.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same `BIN_CENTERS` array defined relative to the go cue, so alignment is guaranteed by construction. Since the tone onset offset is constant, the same `TIME_FROM_TONE` array is used for every trial.

ii.
```python
input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
```

iii. Alignment is trivially achieved because the same bin centers are used for both neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents`, which are absolute timestamps of photostimulation onset and offset events across the entire session.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The agent chose to use the event-based photostimulation timestamps rather than the per-trial `photostim_onset` and `photostim_duration` columns in the trials table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code loops over all photostimulation events in the session and checks if any overlap with the trial window. If a photostim event overlaps, bins whose centers fall within the photostim interval are set to 1.0; otherwise they remain 0.0.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float32)
trial_start_abs = go_time + BEGIN_TIME
trial_end_abs = go_time + END_TIME

for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        rel_start = ps_start - go_time
        rel_stop = ps_stop - go_time
        mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
        photostim_binary[mask] = 1.0
```

iii. The agent creates a binary time series for photostimulation, marking bins where the stimulation is active. This approach correctly represents photostimulation as time-varying.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostimulation onset/offset times are converted to relative times (relative to go cue) and compared against the same `BIN_CENTERS` used for the neural data.

ii.
```python
rel_start = ps_start - go_time
rel_stop = ps_stop - go_time
mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
```

iii. Alignment is achieved by expressing both neural and photostim data relative to the same go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore'). For 'hit', the choice matches the instruction. For 'miss', the choice is the opposite of the instruction. For 'ignore', the AI assigns the choice as the INSTRUCTION direction (same as hit), rather than creating a "no lick" category.

ii.
```python
instr = trial_instruction[trial_idx]
outc = outcome[trial_idx]
if outc == 'hit':
    choice = 0 if instr == 'left' else 1
elif outc == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The agent derived choice from instruction and outcome since choice is not stored directly. However, for 'ignore' trials (no response), the AI assigns the instructed direction as the choice rather than defining a separate "no lick" category.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left) or 1 (right) with only 2 categories. The output_values list is `['left', 'right']` with no "no lick" option. The per-trial value is broadcast across all 80 time bins.

ii.
```python
output_values = [
    ['left', 'right'],
    ...
]
output_trial = np.stack([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], axis=0)
```

iii. The agent only uses two categories for choice. The instructions specify three categories: "left, right, no lick", but the AI omits the "no lick" category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, containing 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = trials['outcome'].values
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers: 0=ignore, 1=miss, 2=hit. The per-trial value is broadcast across all 80 time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
output_trial = np.stack([
    ...
    np.full(N_BINS, outcome_val, dtype=np.int64),
    ...
], axis=0)
```

iii. The mapping matches the order specified in the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early_lick = trials['early_lick'].values
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The trials table flags early licking directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) and 1 (yes). The per-trial value is broadcast across all 80 time bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
np.full(N_BINS, early_lick_val, dtype=np.int64),
```

iii. Simple binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 (tongue_y) and the associated timestamps.

ii.
```python
bt = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bt.time_series['Camera0_side_TongueTracking']
tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
tongue_timestamps = tongue_ts.timestamps[:]
```

iii. The agent identified the correct time series and column for tongue y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes the 40th and 60th percentiles of ALL tongue y-position values across the entire session (no likelihood filtering), then discretizes per-trial tongue_y values into 3 categories: 0 (below 40th percentile), 1 (40th-60th), 2 (above 60th). There is NO "not visible" category. The AI uses nearest-neighbor interpolation to find the tongue y value closest in time to each bin center.

ii.
```python
# Percentiles over ALL tongue_y data (no filtering)
tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
tongue_y_p60 = np.percentile(tongue_y_all_data, 60)

# Nearest-neighbor lookup for each bin center
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
use_prev = dist_prev < dist_curr
tongue_indices[use_prev] = prev_indices[use_prev]

tongue_y_values = tongue_y_all_data[tongue_indices]
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

iii. The agent chose not to filter by tongue tracking likelihood and not to include a "not visible" category, despite the instructions specifying "3: not visible" as one of the discretization categories.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories are used: 0 (below 40th percentile), 1 (40th-60th percentile), 2 (above 60th percentile). The "not visible" category (class 3) specified in the instructions is not implemented. Percentiles are computed over raw frame-level tongue_y values without any likelihood-based filtering.

ii.
```python
tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
tongue_y_disc[tongue_y_values > tongue_y_p60] = 2
```

```python
output_values = [
    ...
    ['low', 'mid', 'high'],
]
```

iii. The agent uses only 3 categories instead of the 4 specified in the instructions. The output value names are also different ('low', 'mid', 'high' instead of descriptive percentile-based names).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center, the nearest camera frame (by timestamp) is found using searchsorted and nearest-neighbor comparison. The tongue y value from that frame is used.

ii.
```python
abs_times = go_time + BIN_CENTERS
tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
```

iii. The agent uses nearest-neighbor interpolation rather than binning/averaging within the 50ms window, which gives a single frame's value rather than an average over the bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions with no good units**: Skipped (return None).
- **Sessions failing performance criteria**: Skipped.
- **Trials outside obs_intervals**: Excluded by matching obs_intervals to trial indices.
- **auto_water and free_water trials**: Excluded.
- **Processing errors**: Caught by try/except and session is skipped with error message.
- **Tongue tracking**: No handling of low-confidence frames; all frames are used regardless of likelihood.

ii.
```python
try:
    result = process_session(nwb_file, verbose=verbose)
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    continue
```

iii. The agent handles errors at the session level with a try/except block. Missing or invalid data at the session level causes the session to be skipped.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the per-unit, per-trial spike binning loop. For each trial, the code loops over all good units and calls `np.histogram` for each unit separately. The agent acknowledged this bottleneck in the trajectory (step 50) and attempted some optimization by pre-loading spike times, but the fundamental per-trial, per-unit loop remains.

ii.
```python
for trial_idx in valid_indices:
    ...
    for i, st in enumerate(all_spike_times):
        lo = np.searchsorted(st, abs_start)
        hi = np.searchsorted(st, abs_end)
        if hi > lo:
            rel_spikes = st[lo:hi] - go_time
            counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
            trial_fr[i, :] = counts / BIN_WIDTH
```

iii. The agent identified the spike time extraction as a bottleneck (trajectory step 50) and preloaded spike times, but the nested trial x unit loop using `np.histogram` per call is significantly slower than the reference's approach of flattening all trial edges and using a single `searchsorted` per unit.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop is a nested trial x unit loop for spike binning. The reference solution vectorizes across trials by concatenating all trial bin edges into a single flat array and running one `searchsorted` per unit. The AI's code runs `np.histogram` once per unit per trial, which is much slower. The tongue y-position processing loop (one iteration per trial) could also be vectorized.

ii.
```python
# AI: nested loops (trial x unit)
for trial_idx in valid_indices:
    for i, st in enumerate(all_spike_times):
        counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
```

iii. The agent noted the performance issue but did not fully vectorize the spike binning.

## 10-c. What processing does the code repeat multiple times?

i. The code loops over photostim events for every trial, checking overlap with each trial window. This means all photostim events are iterated for each trial. The tongue nearest-neighbor search is also done per trial rather than vectorized.

ii.
```python
for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
    if ps_stop > trial_start_abs and ps_start < trial_end_abs:
        ...
```

iii. The photostim loop scans all session-level events for each trial, which is redundant since most events won't overlap with a given trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code saves a separate `sample_data.pkl` file with the first 5 sessions, which is not required by the instructions and is not used downstream. The detailed brain region mapping (14 broad categories) involves extensive string matching that could be simpler. The code also computes and stores session performance metrics that are not used downstream.

ii.
```python
# Sample data saving
n_sample = min(5, len(all_sessions))
sample_data = { ... }
with open(sample_output_file, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. The sample data output was likely created for debugging but adds unnecessary I/O.
