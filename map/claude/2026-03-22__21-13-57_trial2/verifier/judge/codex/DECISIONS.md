# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code enumerates subject folders under `/app/data`, then enumerates `.nwb` files inside each subject folder. Each file is opened with `NWBHDF5IO`, and the script reads trials, units, behavioral events, and tongue tracking directly from the NWB object.

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
trials = nwb.trials
events = nwb.acquisition['BehavioralEvents']
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
```

iii. The notes justify this as the NWB-distributed dataset layout: one session per NWB file under `data/sub-*`, with `units`, `trials`, `BehavioralEvents`, and `BehavioralTimeSeries` holding the needed streams.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the directory name, e.g. `sub-440956`, not from `nwb.subject.subject_id`. The script appends a subject the first time it encounters that folder and uses the encounter order to build `subjects` and `subject_idx`.

ii.
```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
    ...
    if subject_id not in subjects_seen:
        subjects_seen[subject_id] = len(subjects_seen)
        subjects_list.append(subject_id)
    subj_idx = subjects_seen[subject_id]
```

```python
'subjects': subjects_list,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The notes treat the `sub-*` directories as the subject organization of the dataset and report 28 such subjects. There is no separate justification in the code comments for preferring folder names over NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The processing loop handles one file at a time and appends one session record to the output lists.

ii.
```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
    ...
    all_sessions.append(result)
```

```python
return {
    ...
    'session_name': os.path.basename(nwb_path),
}
```

iii. The notes explicitly state "Each NWB file = one session" and use the 174 NWB files / 173 kept sessions counts as support.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The code builds a boolean `trial_mask` over the rows of `nwb.trials`, then iterates over the surviving row indices. Trial-aligned event streams are indexed with the same trial index.

ii.
```python
n_trials_total = len(nwb.trials)
trials = nwb.trials
...
trial_indices = np.where(trial_mask)[0]
...
for ti in trial_indices:
    go_cue = go_start_times[ti]
```

iii. The trajectory and notes describe the NWB file as having one row per behavioral trial, with `go_start_times`, `trial_instruction`, `outcome`, and other per-trial fields all indexed by that same trial number.

## 1-e. How are trials filtered based on quality controls?

i. The final code keeps only trials that satisfy three conditions: `auto_water == 0`, `free_water == 0`, and the trial is considered recorded according to `obs_intervals`. The `obs_intervals` handling is stricter than the human reference: it collects one interval set per distinct interval-count among good units and keeps only trials whose start time is within 1 second of an observed interval start for every such set.

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]
```

```python
for n_obs, obs in obs_sets.items():
    obs_starts = obs[:, 0]
    diffs = np.abs(trial_starts[:, None] - obs_starts[None, :])
    min_diffs = diffs.min(axis=1)
    covered = min_diffs < 1.0
    valid_trials &= covered
```

iii. The notes initially planned to exclude `auto_water` and `free_water`, then later note that `obs_intervals` filtering was added after finding sessions where recording covered only part of behavior. The trajectory explicitly describes using `obs_intervals` to avoid trials with no spikes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for units that pass the AI's unit filter, with `go_start_times` providing the alignment event for extracting each trial window.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
...
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
...
fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The notes state that spike times are absolute in NWB and that `classification == 'good'` corresponds to the QC classifier pass used in the reference work.

## 2-b. How is the `neural` data processed?

i. For each surviving trial, the code extracts spikes in `[go + T_START, go + T_END)`, bins them into non-overlapping 50 ms bins, and divides counts by bin width to produce firing rates in Hz. This is done by looping over neurons within each trial.

ii.
```python
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
...
mask = (st >= abs_start) & (st < abs_end)
spikes_in_window = st[mask]
...
bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
np.add.at(fr[i], bin_indices, 1.0)
...
fr /= bin_size
```

iii. The notes justify 50 ms bins and the `-2.5 s` to `+1.5 s` window from the task instructions, while treating firing-rate conversion from spike counts as the direct analogue of the reference preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are first filtered by `classification == 'good'`. They are then filtered again by whether `anno_name` can be mapped into one of 14 coarse brain regions; unmapped units are discarded. A session is dropped if no units remain after this filtering.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
...
anno_names = nwb.units['anno_name'][:][good_mask]
...
region = map_anno_to_region(str(anno))
if region is None:
    valid_neuron_mask[i] = False
...
good_unit_indices = np.where(good_mask)[0]
good_unit_indices = good_unit_indices[valid_neuron_mask]
```

iii. The notes justify `classification == 'good'` from the QC white paper and also say neurons should have histology / `anno_name`. The final code operationalizes the latter as requiring a successful coarse-region mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial index, the code reads `go_start_times[ti]` and defines the neural extraction window relative to that absolute timestamp.

ii.
```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
...
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The notes explicitly describe spike times as already being on the same absolute clock as go-cue events, so only a relative window around the go cue is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code uses 50 ms bins and 80 time bins total from `-2.5 s` to `+1.5 s` around the go cue. No additional rebinning or smoothing is applied beyond this direct binning from raw timestamps.

ii.
```python
BIN_SIZE_S = 0.05
T_START = -2.5
T_END = 1.5
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)
```

iii. The notes explicitly say the reference paper used 40 ms / 3.4 ms stride, but the conversion intentionally switched to 50 ms non-overlapping bins because the decoder task required it.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `sample_start_times` and `go_start_times`. For each trial, the code takes the last sample start before that trial's go cue.

ii.
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
...
def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]
```

iii. The notes and trajectory justify this by observing that `sample_start_times` has more entries than trials because early licks replay the sample epoch, so the relevant tone is the last sample onset before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code computes bin centers relative to go cue, converts the selected tone onset into a go-cue-relative offset, and subtracts that offset from each bin center. If no preceding sample onset exists, it falls back to `go_cue - 1.85`.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
tone_rel = tone_onset_time - go_cue_time
time_from_tone = bin_centers - tone_rel
```

```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)
```

iii. The notes justify the main rule as "last sample_start before go cue." There is no explicit note-based justification for the hard-coded `1.85 s` fallback; that appears only in the final code.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the same 80 bin centers used for the neural data, with those bin centers defined relative to the same go cue used for neural alignment.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
...
fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)
```

iii. The notes frame this input as a time-varying contextual regressor defined on the same go-cue-aligned grid as the firing rates.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The final code derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, both absolute timestamps. It does not use the trial-table `photostim_onset` or `photostim_duration` fields in the final implementation.

ii.
```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]
```

```python
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
ps_starts = photostim_start_abs[stim_mask]
ps_stops = photostim_stop_abs[stim_mask]
```

iii. The trajectory notes that the trial table stores `photostim_onset` relative to trial start, while `BehavioralEvents` stores absolute photostim timestamps. The final code chose the absolute event stream because it already lives on the same time axis as the neural data.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code finds all photostim events overlapping the trial-aligned neural window and marks bins whose absolute bin centers fall between each event's start and stop time. The result is a binary time series per trial.

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

iii. The notes say photostimulation should be represented as a time-varying binary regressor rather than a per-trial flag. The trajectory also mentions aligning absolute photostim start/stop times to the go-cue-based window.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The code converts neural bin centers to absolute time by adding the trial's `go_cue_time`, then compares those absolute centers to absolute photostim start/stop times. This aligns photostim to the same bins used for the neural data.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
abs_centers = bin_centers + go_cue_time
...
photostim_ts = compute_photostim_timeseries(go_cue, ps_starts, ps_stops,
                                              T_START, T_END, BIN_SIZE_S)
```

iii. The trajectory explicitly notes that `photostim_start_times` are already absolute timestamps, so alignment reduces to comparing them to absolute bin centers from the go-cue-aligned trial window.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the final code, choice is derived only from `trial_instruction`. It does not use `outcome` to recover the animal's actual lick direction, and it does not create a separate no-lick class for ignored trials.

ii.
```python
instructions = trials['trial_instruction'][:]
...
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The notes planned to map `trial_instruction` directly to choice and describe it as "left=0, right=1." There is no later note correcting this to an actual lick-direction derivation using `outcome`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instructed side is encoded as `0` for left and `1` for right, then repeated across all 80 bins in the output tensor. The stored `output_values` also contain only two choice labels.

ii.
```python
full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
full_output[0, :] = out_dict['choice']
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['below_40th', '40th_to_60th', 'above_60th'],
],
```

iii. The notes justify the `0/1` coding from the task statement. They do not discuss how to handle `ignore` trials when there is no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `trials['outcome']` field.

ii.
```python
outcomes = trials['outcome'][:]
...
outcome_str = outcomes[ti]
```

iii. The notes identify NWB `outcome` as already containing the required `hit`, `miss`, and `ignore` categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string outcome is mapped with `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeated across all 80 bins in the output array.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
...
full_output[1, :] = out_dict['outcome']
```

iii. The notes explicitly planned the same category mapping because it matches the decoder specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from `trials['early_lick']`.

ii.
```python
early_lick = trials['early_lick'][:]
...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. The notes identify `early_lick` as an explicit NWB trial field and say these trials were intentionally retained because early lick is itself a decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `'no early'` to `0` and anything else to `1`, then repeats that code across all 80 bins.

ii.
```python
early_val = 0 if early_lick[ti] == 'no early' else 1
...
full_output[2, :] = out_dict['early_lick']
```

iii. The notes explicitly planned this binary mapping from the trial field into the decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `Camera0_side_TongueTracking`: column 1 of the `data` matrix for y-position, column 2 for tracking likelihood, and the matching `timestamps`.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. The notes and trajectory both describe the tongue tracking stream as `(x, y, likelihood)` at roughly video frame rate and explicitly identify column 1 as y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, the code selects tongue frames in the go-cue-aligned window, sets y-values with likelihood `< 0.9` to NaN, bins remaining y-values by averaging within each 50 ms bin, and later computes session percentiles from all non-NaN trial-bin values concatenated across trials. Unlike the reference, missing bins are not given a separate class; they stay at 0 because the discretized array is initialized to zeros.

ii.
```python
y_in_window = y_in_window.copy()
y_in_window[like_in_window < 0.9] = np.nan
...
sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
counts = np.bincount(valid_bins, minlength=n_bins)
tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
```

```python
valid_y = tongue_y_trial[~np.isnan(tongue_y_trial)]
if len(valid_y) > 0:
    tongue_y_session.extend(valid_y.tolist())
...
tongue_y_arr = np.array(tongue_y_session)
p40 = np.percentile(tongue_y_arr, 40)
p60 = np.percentile(tongue_y_arr, 60)
```

iii. The notes justify per-session percentile discretization and using the y channel of tongue tracking. The trajectory also notes that likelihood is the DLC confidence. The final code's `0.9` threshold and handling of missing bins are not justified in the notes; the notes had instead planned percentiles over valid values and did not mention collapsing missing bins into class 0.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The final code computes the 40th and 60th percentiles from concatenated valid trial-bin y-values within a session, then applies three visible classes: `< p40 -> 0`, `p40..p60 -> 1`, `> p60 -> 2`. Bins with no valid tongue data remain 0 because the discretized output vector starts at zeros and only valid positions are overwritten.

ii.
```python
p40 = np.percentile(tongue_y_arr, 40)
p60 = np.percentile(tongue_y_arr, 60)
```

```python
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

iii. The notes justify the `40/60` per-session cut points from the task spec, but the code does not preserve a separate missing / not-visible category despite many bins lacking valid tongue data.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking is aligned by selecting frames between `go + T_START` and `go + T_END` and binning them with the same 50 ms edges used for the go-cue-centered trial window.

ii.
```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
bin_indices = np.digitize(t_in_window, bin_edges) - 1
```

iii. The notes describe camera timestamps and neural timestamps as living on the same absolute clock, so the code aligns tongue frames by applying the same go-cue-relative window in absolute time.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several data-quality edge cases by dropping sessions with no remaining good units, dropping trials outside `obs_intervals`, dropping `auto_water` and `free_water` trials, fabricating a tone onset at `go - 1.85` when no sample onset is found, dropping units whose `anno_name` cannot be mapped to a coarse region, and converting low-likelihood tongue samples to NaN before binning. Missing tongue bins are ultimately collapsed into class 0 rather than represented explicitly.

ii.
```python
if n_good == 0:
    io.close()
    return None
...
if n_neurons == 0:
    io.close()
    return None
```

```python
if tone_onset is None:
    tone_onset = go_cue - 1.85
```

```python
y_in_window[like_in_window < 0.9] = np.nan
...
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
```

iii. The notes justify excluding trials without neural recording and excluding water-confounded trials. The trajectory justifies `obs_intervals` filtering after seeing partial-session recordings. There is no clear note-based justification for the hard-coded tone fallback or for converting missing tongue bins into the lowest visible class.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant work in the AI code is the nested per-trial neural binning loop: for every retained trial, `compute_firing_rates` loops over all surviving units, filters spikes to the trial window, and bins them. Secondary costs are per-trial tongue binning and loading full spike-time and tongue-tracking arrays for each session. The conversion log shows the full run took about 2300.8 s.

ii.
```python
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
    neural_trials.append(fr)
```

```python
for i, st in enumerate(spike_times_list):
    mask = (st >= abs_start) & (st < abs_end)
    spikes_in_window = st[mask]
    ...
    np.add.at(fr[i], bin_indices, 1.0)
```

iii. The notes themselves report a much longer runtime than the human reference and estimate about 6.5 s per session during sampling. They do not explicitly analyze the algorithmic cost, but the final code structure makes the repeated trial-by-trial firing-rate computation the clear bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: the per-trial loop over all retained trials, the inner per-neuron loop inside `compute_firing_rates`, the `find_last_sample_before_go` scan repeated per trial, the `obs_intervals` loop over interval sets, and the annotation-to-region mapping loop over units. The tongue binning within `get_tongue_y_for_trial` is partly vectorized once trial frames are selected.

ii.
```python
for ti in trial_indices:
    ...
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

```python
for i, st in enumerate(spike_times_list):
    ...
```

```python
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

iii. The notes do not justify keeping these loops in this form. The trajectory focuses on correctness and matching dataset counts rather than computational efficiency.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations per trial that could have been hoisted or shared: bin-center construction in `compute_time_from_tone` and `compute_photostim_timeseries`, scanning all sample onset times to find the last pre-go tone, and re-running the spike-window filtering/bucketization for every neuron separately on every trial. It also computes final summary counters by re-iterating over the full output after saving.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```

```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
...
photostim_ts = compute_photostim_timeseries(go_cue, ps_starts, ps_stops,
                                              T_START, T_END, BIN_SIZE_S)
```

```python
for session_outputs in data['output']:
    for trial_out in session_outputs:
        all_choices.append(int(trial_out[0, 0]))
```

iii. The notes do not describe these repetitions as deliberate. They appear to be consequences of a straightforward per-trial implementation rather than an optimized design.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains optional plotting and end-of-run reporting work that is not part of the saved decoder dataset. It also builds per-run summary counters by iterating over all outputs after the dataset has already been assembled. In addition, the `session_name` field is carried inside intermediate session dicts only for logging / plotting and is not saved into the final pickle.

ii.
```python
if args.show_processing and i < 2:
    plot_processing(result, i, nwb_path)
```

```python
all_choices = []
all_outcomes = []
all_early = []
for session_outputs in data['output']:
    for trial_out in session_outputs:
        all_choices.append(int(trial_out[0, 0]))
```

```python
return {
    ...
    'session_name': os.path.basename(nwb_path),
}
```

iii. The notes describe the plots as "processing visualization" and the counters as validation / consistency checks. They are justified for inspection, but they are not used by downstream decoder training on the saved pickle itself.
