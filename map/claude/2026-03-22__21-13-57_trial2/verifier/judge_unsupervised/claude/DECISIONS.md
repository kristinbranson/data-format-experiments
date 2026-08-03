# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored in `/app/data`. It iterates over subject directories (named `sub-XXXXXX`), finds all `.nwb` files within each, and processes each file as one session using `NWBHDF5IO`. Data from each session includes units (spike times, classification, brain region annotations), trials (instruction, outcome, early lick, auto/free water), behavioral events (go cue, sample start, photostim start/stop times), and behavioral time series (tongue tracking).

ii.
```python
def get_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files
```
```python
io = NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The AI documented that the data is in NWB format organized by subject directories, with 174 NWB files from 28 subjects. The reference code originally used `.mat` files from DataJoint export, but the NWB files contain the same information. The AI correctly identified the NWB structure and all necessary variables.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the directory names under `/app/data` (e.g., `sub-440956`). Each subject directory contains one or more NWB session files. A unique subjects list is built as sessions are processed, with each session mapped to its subject via `subject_idx`.

ii.
```python
subjects_seen = {}
# ...
if subject_id not in subjects_seen:
    subjects_seen[subject_id] = len(subjects_seen)
    subjects_list.append(subject_id)
subj_idx = subjects_seen[subject_id]
subject_idx_list.append(subj_idx)
```

iii. The AI noted 28 subjects matching the reference papers. Subject IDs are derived from directory names (e.g., `sub-440956`).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed individually via `process_session()`. The one session with 0 good units (`sub-440958_ses-20190216T162508`) is skipped, yielding 173 sessions matching the papers.

ii.
```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
    if result is None:
        print("SKIPPED (no good units or trials)")
        continue
    all_sessions.append(result)
```

iii. The AI documented that 174 NWB files exist, with 1 having 0 good units, giving 173 usable sessions matching the paper's "173 behavioral sessions."

## 1-d. How are the data split into trials?

i. Trials are defined by the `nwb.trials` table within each NWB session. Each row is one trial. The go cue times for each trial come from `go_start_times` in `BehavioralEvents`, indexed by trial number.

ii.
```python
n_trials_total = len(nwb.trials)
# ...
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The AI uses the trial index directly to index into go_start_times, assuming 1:1 correspondence between trial table rows and go cue events.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `auto_water` and `free_water` trials. Additionally, trials outside valid recording periods (based on `obs_intervals` for good units) are excluded. The AI explicitly chose to KEEP early lick trials (as an output variable), photostimulation trials (as an input variable), and ignore/miss outcomes (as output categories). This departs from the reference code's `get_regular_trial_mask()` which also excludes early lick, no-response, and photostimulation trials.

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]
```

iii. The AI justified keeping early lick, photostim, and ignore trials because they are decoder inputs/outputs per the task specification. The reference code's `get_regular_trial_mask()` excludes these for its analysis, but the decoder task explicitly requires them. The AI also uses `obs_intervals` to exclude trials without valid neural recordings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for units where `nwb.units['classification'] == 'good'`.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
# ...
good_unit_indices = np.where(good_mask)[0]
# ...
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

iii. The AI identified that `classification == 'good'` corresponds to the QC classifier filter used in the reference code (`qc_mode = 'classifier'`).

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins spanning -2.5s to 1.5s relative to go cue onset, producing firing rates in Hz. This differs from the reference code's 40ms Gaussian kernel width with 3.4ms stride, but follows the task specification.

ii.
```python
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80

def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    n_neurons = len(spike_times_list)
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end
    for i, st in enumerate(spike_times_list):
        if len(st) == 0:
            continue
        mask = (st >= abs_start) & (st < abs_end)
        spikes_in_window = st[mask]
        if len(spikes_in_window) == 0:
            continue
        bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        np.add.at(fr[i], bin_indices, 1.0)
    fr /= bin_size
    return fr
```

iii. The AI noted the reference uses 40ms bandwidth/3.4ms stride but the task requires 50ms bins. The non-overlapping binning approach (counting spikes per bin and dividing by bin width) is a standard spike rate computation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are included. Additionally, neurons whose fine CCF annotation (`anno_name`) cannot be mapped to one of the 14 coarse brain regions are excluded. The AI did NOT apply the 2 Hz firing rate threshold mentioned in the method paper.

ii.
```python
good_mask = np.array([c == 'good' for c in classification])
# ...
anno_names = nwb.units['anno_name'][:][good_mask]
region_indices = []
valid_neuron_mask = np.ones(n_good, dtype=bool)
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
```

iii. The AI justified not applying the 2 Hz firing rate threshold by noting it was analysis-specific (used for video prediction in the method paper, not general preprocessing). The reference preprocessing code does not apply a firing rate threshold either - it's applied downstream in specific analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. Go cue times are extracted from `go_start_times` in `BehavioralEvents`. Spike times relative to each trial's go cue are computed by subtracting the go cue time from spike times.

ii.
```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
# ...
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The AI documented that go cue onset is the alignment event, with spike times being absolute in NWB (vs. already go-cue-aligned in the reference .mat files).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms (non-overlapping), producing 80 time bins for the 4-second window (-2.5s to 1.5s). No temporal rebinning is applied - spikes are directly binned at 50ms. The reference code uses 40ms bin width with 3.4ms stride (overlapping), but the task specification requires 50ms bins.

ii.
```python
BIN_SIZE_S = 0.05  # 50 ms bins
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

iii. The AI explicitly documented the difference: reference uses 40ms/3.4ms stride, task requires 50ms. The 50ms value comes directly from the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` in `BehavioralEvents` and `go_start_times`. The tone onset is identified as the last `sample_start_times` before each trial's go cue.

ii.
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
# ...
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
```

iii. The AI noted that `sample_start_times` can have more entries than trials due to early lick replays, so finding the last sample onset before the go cue is necessary.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last `sample_start_time` before the go cue is found. The time from tone onset is computed at each time bin center as: `bin_center_relative_to_go_cue - (tone_onset - go_cue)`.

ii.
```python
def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]

def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time
    time_from_tone = bin_centers - tone_rel
    return time_from_tone.astype(np.float32)
```

iii. If no sample_start_time is found before the go cue, a default of `go_cue - 1.85` is used (based on typical task timing).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same time bin centers relative to the go cue, ensuring alignment. The bin centers are computed as `t_start + bin_index * bin_size + bin_size/2`.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```

iii. Same time axis is used for both neural and input data, ensuring temporal alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents`.

ii.
```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]
```

iii. These are absolute timestamps of photostimulation onset and offset.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is created where 1 indicates photostimulation is active and 0 indicates it is off. For each trial, the photostim events that overlap the trial window are identified, and bin centers within those periods are set to 1.

ii.
```python
def compute_photostim_timeseries(go_cue_time, photostim_starts, photostim_stops,
                                  t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    abs_centers = bin_centers + go_cue_time
    stim = np.zeros(n_bins, dtype=np.float32)
    for start, stop in zip(photostim_starts, photostim_stops):
        mask = (abs_centers >= start) & (abs_centers <= stop)
        stim[mask] = 1.0
    return stim
```

iii. Photostim events are filtered to those overlapping the trial window before computing the binary series.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Uses the same bin centers as neural data (relative to go cue, converted to absolute time for comparison with photostim timestamps).

ii.
```python
abs_centers = bin_centers + go_cue_time
```

iii. Alignment is ensured by using the same time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` in the NWB trials table.

ii.
```python
instructions = trials['trial_instruction'][:]
# ...
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The AI uses `trial_instruction` (left/right) as choice, mapping left=0, right=1.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Direct mapping: 'left' -> 0, 'right' -> 1. This is a per-trial scalar that is replicated across all time bins.

ii.
```python
choice = 0 if instructions[ti] == 'left' else 1
# ...
full_output[0, :] = out_dict['choice']
```

iii. The AI maps `trial_instruction` to binary, consistent with the reference code's `trial_type` variable where `1*(trial_type=='l')` maps left=1, right=0. NOTE: The AI's mapping (left=0, right=1) follows the task specification but is inverted relative to the reference code's convention (left=1, right=0). However, the reference code uses `trial_type` (instructed direction) not actual lick direction. The task instructions specify left=0, right=1 for "Lick direction choice."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `outcome` in the NWB trials table.

ii.
```python
outcomes = trials['outcome'][:]
# ...
outcome_str = outcomes[ti]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

iii. Direct mapping from the NWB outcome strings to integers.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping: 'ignore' -> 0, 'miss' -> 1, 'hit' -> 2. Per-trial scalar replicated across time bins. Unknown outcomes default to 0 (ignore).

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
full_output[1, :] = out_dict['outcome']
```

iii. Mapping matches the task specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `early_lick` in the NWB trials table.

ii.
```python
early_lick = trials['early_lick'][:]
# ...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. The NWB stores early_lick as strings ('no early' or 'early').

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String-to-integer mapping: 'no early' -> 0, anything else -> 1. Per-trial scalar replicated across time bins.

ii.
```python
early_val = 0 if early_lick[ti] == 'no early' else 1
full_output[2, :] = out_dict['early_lick']
```

iii. Binary classification consistent with the task specification.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `Camera0_side_TongueTracking` in `BehavioralTimeSeries`. Specifically, column 1 (y-position) and column 2 (likelihood) of the 3-column tracking data.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. Data from DeepLabCut tracking at ~300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Low-likelihood frames (< 0.9) are set to NaN. Tongue y-values are averaged within each 50ms time bin using digitize and bincount. Per-session percentiles (40th and 60th) are computed over all valid tongue y-values across all trials in the session.

ii.
```python
y_in_window = y_in_window.copy()
y_in_window[like_in_window < 0.9] = np.nan

tongue_y_binned = np.full(n_bins, np.nan, dtype=np.float32)
bin_indices = np.digitize(t_in_window, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
valid_mask = ~np.isnan(y_in_window)
if valid_mask.any():
    valid_bins = bin_indices[valid_mask]
    valid_vals = y_in_window[valid_mask]
    sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
    counts = np.bincount(valid_bins, minlength=n_bins)
    has_data = counts > 0
    tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
```

iii. The AI applies a likelihood threshold of 0.9 to filter unreliable tracking. The reference paper mentions 5-sigma velocity threshold outlier correction but the AI uses likelihood filtering instead.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session discretization using 40th and 60th percentiles of all valid tongue y-values across all trials in the session. Values below 40th percentile -> 0, between 40th and 60th -> 1, above 60th -> 2. NaN values (no valid tracking data) are set to 0.

ii.
```python
if len(tongue_y_session) > 0:
    tongue_y_arr = np.array(tongue_y_session)
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)

tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

iii. Matches the task specification: 0 = below 40th percentile, 1 = 40th-60th, 2 = above 60th. NaN bins default to 0.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Uses the same time bin edges derived from go cue timing, binning tongue tracking timestamps into the same 50ms bins as neural data.

ii.
```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
```

iii. Same temporal framework ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Sessions with 0 good units are skipped.
- Sessions with fewer than 2 valid trials are skipped.
- Neurons with unmappable brain region annotations are excluded.
- Trials outside recording periods (per obs_intervals) are excluded.
- Missing tone onset defaults to `go_cue - 1.85`.
- Low-likelihood tongue tracking frames are set to NaN; NaN tongue positions default to category 0.
- Empty spike time arrays produce zero firing rates.

ii.
```python
if n_good == 0:
    io.close()
    return None
# ...
if len(trial_indices) < 2:
    io.close()
    return None
# ...
if tone_onset is None:
    tone_onset = go_cue - 1.85
```

iii. The AI documented handling of the session with 0 good units and the obs_intervals filtering for partial recordings. Missing tone onset is handled with a sensible default based on typical task timing.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is computing firing rates for each trial, which involves iterating over all neurons and binning spikes. The full conversion took ~38 minutes for 173 sessions (~13 seconds/session average). Loading NWB files and extracting tongue tracking data also contribute significantly.

ii.
```python
for i, st in enumerate(spike_times_list):
    # ... per-neuron spike binning loop
    mask = (st >= abs_start) & (st < abs_end)
    spikes_in_window = st[mask]
    bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
    np.add.at(fr[i], bin_indices, 1.0)
```

iii. The AI estimated ~19 minutes for full conversion (Step 7) but actual time was ~38 minutes, indicating the estimate was optimistic.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop over neurons in `compute_firing_rates` iterates per-neuron to bin spikes. This could potentially be vectorized using numpy histogram operations on all neurons simultaneously, though the variable-length spike time arrays make this non-trivial. The trial-level loop in `process_session` could also be parallelized.

ii.
```python
for i, st in enumerate(spike_times_list):
    if len(st) == 0:
        continue
    mask = (st >= abs_start) & (st < abs_end)
    # ...
```

iii. The AI mentioned vectorization in the code comments and CONVERSION_NOTES but the per-neuron loop remains. The tongue y-position binning was vectorized using numpy's digitize/bincount.

## 10-c. What processing does the code repeat multiple times?

i. For each trial, the code recomputes the same bin edges and bin centers for neural, input, and tongue data. These could be precomputed once. The obs_intervals validation recomputes trial-to-observation matching for different observation interval groups.

ii.
```python
# In compute_firing_rates:
n_bins = int((t_end - t_start) / bin_size)
# In compute_time_from_tone:
n_bins = int((t_end - t_start) / bin_size)
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
# In compute_photostim_timeseries:
n_bins = int((t_end - t_start) / bin_size)
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
# In get_tongue_y_for_trial:
n_bins = int((t_end - t_start) / bin_size)
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
```

iii. Bin calculation is repeated in every function for every trial, though the overhead is minimal compared to I/O and spike binning.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes brain region mapping for all 14 coarse regions, even though the downstream decoder may not use region information for its main analysis. The tongue y-position raw values are collected for percentile computation and then discarded. The full tongue tracking data (x, y, likelihood) is loaded but only y is used.

ii.
```python
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)  # only y used
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. Loading all 3 columns of tongue tracking when only y and likelihood are needed wastes some memory but is relatively minor. The brain region mapping is part of the output format specification so it's necessary.
