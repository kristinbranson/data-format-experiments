# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by globbing all NWB files under `data/sub-*/sub-*.nwb`, sorting them, and processing each file as one session. Within each file it reads trials, units, behavioral event timestamps, and tongue tracking data directly from the NWB object.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
```

```python
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
...
trial_instruction = nwb.trials['trial_instruction'][:]
classification = nwb.units['classification'][:]
beh_events = nwb.acquisition['BehavioralEvents']
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as adapting the paper’s processing to the NWB release: “Reference code loads from .mat files; we use NWB files.” It also noted 28 subject directories and 174 NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are split by `nwb.subject.subject_id`. The final `subjects` list is the sorted unique subject ids across retained sessions, and `subject_idx` maps each retained session to that list.

ii.
```python
subject_id = nwb.subject.subject_id
...
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
...
'subjects': unique_subjects,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The AI did not give a separate detailed justification beyond using the NWB subject field as the canonical subject identifier.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. The output keeps one top-level session entry per successfully processed file.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
session_data = process_session(nwb_file, ...)
if session_data is not None:
    all_sessions.append(session_data)
```

iii. The AI’s notes explicitly describe the NWB layout as one session file per subject/session path, so it used file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are split by indexing all trial-table columns and the go-cue timestamp array with the same trial index. After filtering, the retained trial indices define the session’s trial list.

ii.
```python
n_trials_total = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
...
valid_trial_indices = np.where(valid_trial_mask)[0]
...
for local_idx, trial_idx in enumerate(valid_trial_indices):
    go_time = go_times[trial_idx]
```

iii. The AI’s trajectory shows it understood that NWB trials are row-wise behavioral trials, and it relied on parallel indexing rather than re-deriving trial boundaries from events.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by requiring the full `[-2.5, 1.5] s` neural window around the go cue to lie within a session-wide spike-time range derived from actual spikes. It does not filter `free_water` trials explicitly. It also imposes session-level behavioral inclusion criteria from the paper: control-trial performance `>= 0.65` and at least 50 correct left and 50 correct right trials.

ii.
```python
rec_start, rec_end = get_recording_range(nwb, good_indices)
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
```

```python
is_control = np.array([str(p) == 'N/A' for p in photostim_power])
is_no_early = np.array([str(e) == 'no early' for e in early_lick])
control_no_early = is_control & is_no_early
is_not_ignore = outcome != 'ignore'
...
if correct_rate < 0.65:
    return None
if correct_left < 50 or correct_right < 50:
    return None
```

iii. The AI justified the session filter from `methods.txt`: “overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each.” It justified the trial filter in `CONVERSION_NOTES.md` as using “actual spike time range” to avoid all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for units with `classification == 'good'`, together with `BehavioralEvents/go_start_times` for alignment.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
...
spike_times_all = nwb.units['spike_times'][:]
...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

iii. The AI’s notes say the target neural signal is “spike_times | Bin into 50ms windows, compute firing rate (Hz) | Aligned to go cue.”

## 2-b. How is the `neural` data processed?

i. For each retained trial and each retained neuron, the AI builds 50 ms bin edges around the trial’s go cue, counts spikes with `np.histogram`, and converts counts to firing rates in Hz by dividing by bin width. No smoothing or normalization is applied.

ii.
```python
def compute_firing_rates_session(spike_times_list, go_times, ...):
    ...
    for trial_idx in range(n_trials):
        go_time = go_times[trial_idx]
        bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, spikes in enumerate(spike_times_list):
            ...
            counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
            fr[i, :] = counts / bin_width
```

iii. The AI linked this to the reference preprocessing step `sliding_histogram` and to the task requirement of 50 ms bins around the go cue.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by `classification == 'good'`. Sessions with fewer than 2 such units are dropped.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
n_good = np.sum(good_mask)

if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    io.close()
    return None
```

iii. The AI justified `classification == 'good'` from the paper’s classifier-based QC. In the notes it describes this as “classifier-based QC” matching the reference code’s QC mode.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. The go cue for each trial is read from `go_start_times`, and bin edges are constructed as absolute times relative to that event.

ii.
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
...
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. The AI repeatedly notes in `CONVERSION_NOTES.md` and the trajectory that the task requires “Temporal alignment: Go cue onset (time 0).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins across a 4.0 s window from `-2.5 s` to `+1.5 s`, giving 80 bins per trial. There is no additional rebinning after this.

ii.
```python
BIN_WIDTH = 0.050
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```

iii. This directly follows the decoder task instructions quoted in the prompt and reflected in the AI’s notes.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does not derive this input from any raw per-trial event variable. It assumes a fixed tone onset at `-1.85 s` relative to the go cue and uses only the predefined bin centers.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` the AI justified this by citing the fixed task timing: “Tone at -1.85s rel to go.” There is no code using `sample_start_times`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes one session-invariant 80-element vector by subtracting the assumed constant tone-onset time from each go-cue-centered bin center. The same vector is reused for every trial in the session.

ii.
```python
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
...
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The justification in the notes is again the assumed fixed task geometry, not raw event extraction.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction on the same 80-bin go-cue-centered grid as the neural data. Because the AI uses a fixed offset from the go cue, the time-from-tone vector is identical across trials.

ii.
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The AI’s reasoning was that a fixed task timeline was sufficient for alignment; it did not reference early-lick-driven sample replays.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the behavioral event streams `photostim_start_times` and `photostim_stop_times`, plus each trial’s go-cue time for alignment.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
...
go_time = go_times[trial_idx]
```

iii. The AI’s notes describe this as “photostim_start/stop | Binary: 1 if photostim on, 0 otherwise | Per time bin.”

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI finds all stimulation intervals that overlap the trial window and marks bins whose absolute centers fall between each interval’s start and stop times as `1`; other bins are `0`.

ii.
```python
stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
...
photostim_ts = get_photostim_timeseries(trial_stim_starts, trial_stim_stops, go_time)
```

```python
abs_bin_centers = go_time + bin_centers
for start, stop in zip(photostim_start_times, photostim_stop_times):
    mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
    photostim[mask] = 1.0
```

iii. The AI justified this as producing the requested time-varying binary input rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done by comparing absolute stimulation event times to absolute bin centers computed from the same go-cue-centered grid used for neural binning.

ii.
```python
abs_bin_centers = go_time + bin_centers
...
mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
```

iii. The AI’s trajectory indicates it chose the event stream because it was already on the session clock used for neural timestamps.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice only from `trial_instruction`. It does not combine `trial_instruction` with `outcome`, and it does not create a no-lick category.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. In `CONVERSION_NOTES.md` the mapping table explicitly says `trial_instruction | output[0] | left=0, right=1 | Per trial, broadcast to all time bins`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `left -> 0` and `right -> 1`, then repeats that single value across all 80 time bins for the trial. There is no separate `no lick` value.

ii.
```python
output_choice.append(choice)
...
np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no_early', 'early'],
    ['low', 'mid', 'high'],
],
```

iii. The AI’s notes justify this as a direct mapping from `trial_instruction` to the decoder output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the `outcome` trial-table column.

ii.
```python
outcome = nwb.trials['outcome'][:]
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. The AI treated the NWB outcome field as already matching the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, `hit -> 2`, then broadcasts the per-trial label across all 80 bins.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64)
```

iii. The AI’s notes say this output is directly mapped from the raw trial outcome categories.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the `early_lick` trial-table column.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. The AI treated `early_lick` as an explicit per-trial field in the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats the result across all 80 bins of the trial.

ii.
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
...
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64)
```

iii. The AI’s notes describe this as a direct categorical mapping from the NWB field.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `Camera0_side_TongueTracking.timestamps` and the `y` coordinate in column 1 of `Camera0_side_TongueTracking.data`. The AI does not use the likelihood column when computing visibility.

ii.
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
...
local_y = tongue_data[idx_start:idx_end, 1]
```

iii. In the notes the AI describes “Tongue tracking at ~300Hz (Camera0_side_TongueTracking: x, y, likelihood),” but the implementation uses only the y coordinate.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, the AI averages raw tongue y values within each 50 ms bin, ignoring bins with no frames by leaving them as NaN. It then pools all non-NaN bin values across the retained trials in the session and computes the 40th and 60th percentiles from those values. It does not threshold by tracking likelihood.

ii.
```python
def get_tongue_y_for_trial(...):
    ...
    local_y = tongue_data[idx_start:idx_end, 1]
    ...
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            tongue_y[b] = np.mean(local_y[mask])
```

```python
all_values = np.concatenate(all_values)
p_low = np.percentile(all_values, percentile_low)
p_high = np.percentile(all_values, percentile_high)
```

iii. The AI’s notes explicitly justify a different decision from the reference: “Use all tongue y values regardless of likelihood for percentile computation.”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI thresholds session-level tongue y values into only three classes: `<40th percentile -> 0`, `40th-60th percentile -> 1`, `>60th percentile -> 2`. Missing or invisible bins default to class `1` rather than a separate `not visible` class.

ii.
```python
discrete = np.ones(len(y), dtype=np.int64)  # default middle
valid = ~np.isnan(y)
if np.any(valid):
    discrete[valid & (y < p_low)] = 0
    discrete[valid & (y >= p_low) & (y <= p_high)] = 1
    discrete[valid & (y > p_high)] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no_early', 'early'],
    ['low', 'mid', 'high'],
],
```

iii. The notes say “Tongue y discretization: Use all tongue y values regardless of likelihood for percentile computation.” The code shows no fourth category for hidden tongue.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is aligned per trial by extracting camera frames from the same `[-2.5, 1.5] s` go-cue-centered window and assigning them to the same 50 ms bins used for neural data.

ii.
```python
t_start = go_time + bin_centers[0] - bin_width / 2
t_end = go_time + bin_centers[-1] + bin_width / 2
idx_start = np.searchsorted(tongue_ts, t_start, side='left')
idx_end = np.searchsorted(tongue_ts, t_end, side='right')
...
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
```

iii. The AI’s trajectory shows it intentionally aligned tongue data on the same go-cue-centered timeline as the neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some missing-data cases by dropping sessions with too few good units, dropping trials without full neural window coverage according to the global spike-time range, and leaving tongue bins as NaN until discretization. But the discretization then collapses missing tongue bins into the middle class instead of an explicit `not visible` category.

ii.
```python
if n_good < 2:
    return None
...
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
```

```python
tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
...
discrete = np.ones(len(y), dtype=np.int64)  # default middle
valid = ~np.isnan(y)
```

iii. The AI justified the neural-coverage filter as preventing all-zero neural trials. For tongue data, the notes say to use all values regardless of likelihood, which leads to missing visibility information being folded into the middle class.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step in the AI code is the firing-rate computation, because it loops over every retained trial and every retained neuron and calls `np.histogram` for each pair. NWB loading and tongue/input construction are secondary costs.

ii.
```python
fr_all = []
for trial_idx in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
        fr[i, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` includes timing estimates showing “Firing rates” dominating runtime over “Data loading” and “Inputs/outputs.”

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest vectorization opportunity is the nested trial-by-neuron loop in `compute_firing_rates_session`. The tongue-processing code also loops over bins within each trial and then over trials again during discretization and output assembly.

ii.
```python
for trial_idx in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        ...
```

```python
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
...
for y in tongue_y_session:
    ...
for t_idx in range(n_valid):
    out_arr = np.stack([...], axis=0)
```

iii. The AI’s notes claim `compute_firing_rates_session` is “Vectorized histogram computation with searchsorted optimization,” but the implemented code still uses explicit nested loops and per-trial histograms.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it scans the global photostimulation event arrays separately for each trial, recomputes bin edges separately for each trial, iterates over tongue bins separately for each trial, and scans spike times once to estimate recording range and again to compute firing rates.

ii.
```python
rec_start, rec_end = get_recording_range(nwb, good_indices)
...
fr_all = compute_firing_rates_session(spike_times_good, valid_go_times)
```

```python
for local_idx, trial_idx in enumerate(valid_trial_indices):
    ...
    stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
    ...
    tongue_y = get_tongue_y_for_trial(tongue_ts_all, tongue_data_all, go_time)
```

iii. There is no explicit written justification for these repeated passes beyond the AI’s general focus on getting a working conversion and producing diagnostic plots.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and carries some fields that are not used in the final converted dataset or downstream decoder inputs/outputs: `subject_desc`, `auto_water`, `free_water`, and optional plotting logic. It also returns `subject_desc` from `process_session` but never writes it into the final pickle.

ii.
```python
subject_desc = nwb.subject.description
...
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
```

```python
return {
    ...
    'subject_desc': subject_desc,
    ...
}
```

```python
if show_processing:
    plot_processing(...)
```

iii. The AI did not justify these extra reads beyond general inspection/debugging. They appear to be leftovers from exploratory work and optional diagnostics rather than data needed for the final decoder format.
