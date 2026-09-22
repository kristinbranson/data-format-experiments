# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by scanning `/app/data` for `sub-*` directories and `.nwb` files, sorting both directory and file names, then opening each file with `pynwb.NWBHDF5IO`. Within each file it reads the NWB units table, trials table, behavioral events, and behavioral time series.

ii. 
```python
def get_nwb_files(data_dir='/app/data'):
    """Get sorted list of all NWB file paths."""
    nwb_files = []
    for subdir in sorted(os.listdir(data_dir)):
        sub_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(sub_path) or not subdir.startswith('sub-'):
            continue
        for f in sorted(os.listdir(sub_path)):
            if f.endswith('.nwb'):
                nwb_files.append(os.path.join(sub_path, f))
    return nwb_files
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    units = nwb.units
    trials = nwb.intervals['trials']
    be = nwb.acquisition['BehavioralEvents']
    bts = nwb.acquisition['BehavioralTimeSeries']
```

```python
nwb_files = get_nwb_files()
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
```

iii. In `CONVERSION_NOTES.md` Step 5 and the trajectory, the AI explicitly chose direct NWB loading because the dataset is distributed as NWB and the reference code only showed equivalent processing on earlier `.mat` exports. It treated the sorted on-disk NWB layout as the authoritative list of sessions.

## 1-b. How are the data split into subjects?

i. The AI splits subjects by parent directory name, e.g. `sub-440956`, and carries that folder name into `subjects` and `subject_idx`.

ii. 
```python
subject_id = nwb_path.split('/')[-2]
```

```python
all_subjects = sorted(set(s['subject_id'] for s in all_sessions_data))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx.append(subject_to_idx[sess['subject_id']])
```

iii. The notes treat the `/app/data/sub-*` directory structure as the mouse split. The AI did not document using `nwb.subject.subject_id`; instead it assumed the folder name was a sufficient subject identifier.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session and assigns the session id from the NWB filename without the `.nwb` suffix.

ii. 
```python
session_id = os.path.basename(nwb_path).replace('.nwb', '')
```

```python
for i, nwb_path in enumerate(nwb_files):
    print(f"[{i+1}/{len(nwb_files)}] Processing {os.path.basename(nwb_path)}")
    result = process_session(nwb_path, ...)
```

iii. In `CONVERSION_NOTES.md` Step 2 and the trajectory, the AI recorded that there are 174 NWB files and effectively equated NWB file boundaries with sessions.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table and behavioral go-cue timestamps, keeps trial-level arrays aligned by index, and then subsets all of them with a boolean trial mask.

ii. 
```python
trials = nwb.intervals['trials']
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
outcomes = trials['outcome'][:]
...
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
valid_indices = np.where(valid_trials)[0]
...
go_times_valid = go_times[valid_indices]
trial_starts_valid = trial_starts[valid_indices]
trial_stops_valid = trial_stops[valid_indices]
outcomes_valid = outcomes[valid_indices]
```

iii. The trajectory shows the AI inspected NWB trial fields and go-cue arrays, then used indexed trial rows as the trial definition rather than re-deriving trials from event streams.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out `auto_water` and `free_water` trials, then additionally filters out any trial whose full `[-2.5, 1.5]` s go-cue-aligned window does not fit inside the intersection of all good units’ recording intervals. Sessions with fewer than 2 remaining trials are dropped.

ii. 
```python
valid_trials = (auto_water == 0) & (free_water == 0)
```

```python
rec_start, rec_end = get_recording_time_range(nwb, good_indices)
recording_covered = (go_times + t_start >= rec_start) & (go_times + t_end <= rec_end)
valid_trials = valid_trials & recording_covered
```

```python
if n_valid < 2:
    print(f"  Skipping {session_id}: only {n_valid} valid trials")
    return None
```

iii. `CONVERSION_NOTES.md` Step 5 says the AI wanted to keep early-lick, ignore, and photostim trials because they are decoder targets/inputs, but to remove auto/free-water trials. Later, trajectory step 58 and Step 10 in the notes show it added the recording-coverage filter after seeing many zero-neural-data trials and justified it as a fix for trials outside the recording window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units['spike_times']` for units whose `classification` is `'good'`, using go-cue timestamps to place trial-aligned bins.

ii. 
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
good_indices = np.where(good_mask)[0]
```

```python
spike_times_list = []
for idx in good_indices:
    spike_times_list.append(units['spike_times'][idx])
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. In `CONVERSION_NOTES.md` Step 3 and Step 5, the AI states that `classification == "good"` is the NWB representation of the QC classifier output described in the papers, and that spike times are the source for firing rates.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into 50 ms non-overlapping go-cue-aligned bins over `[-2.5, 1.5]` s and divides counts by bin width to obtain firing rates in Hz. It does no smoothing, normalization, or baseline subtraction.

ii. 
```python
def compute_firing_rates_vectorized(spike_times_list, go_times, t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    bin_edges = t_start + np.arange(n_bins + 1) * bin_width
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ...
    for j, spk_times in enumerate(spike_times_list):
        ...
        for i in range(n_trials):
            go = go_times[i]
            lo = np.searchsorted(spk_times, go + t_start)
            hi = np.searchsorted(spk_times, go + t_end)
            if hi > lo:
                relative_spikes = spk_times[lo:hi] - go
                counts, _ = np.histogram(relative_spikes, bins=bin_edges)
                fr_all[i, j, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says to use 50 ms width because the task asked for 50 ms bins, even though the reference code used 100 ms width with 50 ms stride. The notes also describe this as firing-rate computation from spike times aligned to go cue.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'` and drops sessions with zero such units.

ii. 
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    print(f"  Skipping {session_id}: no good units")
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 5 state that this column is the QC classifier verdict from the white paper and should be used as the neuron curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go cue onset by subtracting each trial’s `go_time` from spike times before histogramming within the `[-2.5, 1.5]` s window.

ii. 
```python
go = go_times[i]
lo = np.searchsorted(spk_times, go + t_start)
hi = np.searchsorted(spk_times, go + t_end)
if hi > lo:
    relative_spikes = spk_times[lo:hi] - go
    counts, _ = np.histogram(relative_spikes, bins=bin_edges)
```

iii. `CONVERSION_NOTES.md` Step 3 says all temporal alignment should be relative to go cue onset, matching the task instructions and the reference materials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins from `-2.5` to `1.5` s, producing 80 bins per trial. No additional temporal rebinning is applied beyond this initial binning.

ii. 
```python
t_start = -2.5
t_end = 1.5
bin_width = 0.05
```

```python
n_bins = int(round((t_end - t_start) / bin_width))
bin_edges = t_start + np.arange(n_bins + 1) * bin_width
```

iii. `CONVERSION_NOTES.md` Step 5 says the AI deliberately changed the paper’s 100 ms sliding window to 50 ms non-overlapping bins because the instructions explicitly said “50-ms-width bins.”

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` plus each trial’s `go_time` and `start_time`. For each trial, the AI selects the last sample-start event between trial start and go cue as the tone onset.

ii. 
```python
def get_tone_onset_per_trial(go_times, trial_starts, sample_start_times):
    ...
    samp_in_trial = sample_start_times[(sample_start_times >= ts) & (sample_start_times < go)]
    if len(samp_in_trial) > 0:
        tone_onsets[i] = samp_in_trial[-1]
```

iii. The trajectory shows the AI inspected repeated sample events and concluded the relevant tone is the last one before the go cue because early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a continuous value at each bin center as “time since tone onset.” It subtracts the trial’s tone-onset offset from the go-cue-centered bin centers. If it cannot find a sample event for a trial, it falls back to `go_time - 1.85`.

ii. 
```python
if len(samp_in_trial) > 0:
    tone_onsets[i] = samp_in_trial[-1]
else:
    tone_onsets[i] = go - 1.85
```

```python
tone_onset_rel = tone_onsets[i] - go_times_valid[i]
inputs[0, :] = bin_centers - tone_onset_rel
```

iii. The trajectory and `CONVERSION_NOTES.md` Step 5 say the tone is typically 1.85 s before go cue, and the AI used that as a fallback when no event was found. The documented intent was to represent time from tone onset continuously across the neural bins.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 50 ms bin centers used for neural firing rates, which are themselves defined relative to go cue.

ii. 
```python
fr_list, bin_centers = compute_firing_rates_vectorized(
    spike_times_list, go_times_valid, t_start, t_end, bin_width
)
```

```python
inputs[0, :] = bin_centers - tone_onset_rel
```

iii. The AI’s notes repeatedly state that all streams should share the go-cue-centered time base, so the tone input is expressed on the same bin grid as the neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trials-table strings `photostim_onset` and `photostim_duration`, together with `trial_starts` and `go_times`.

ii. 
```python
photostim_onset_str = trials['photostim_onset'][:]
photostim_duration_str = trials['photostim_duration'][:]
```

```python
stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
stim_dur = float(photostim_duration_valid[i])
stim_end_go_rel = stim_onset_go_rel + stim_dur
```

iii. The trajectory shows the AI explicitly examined these trial fields and concluded that photostimulation onset is stored relative to trial start and must be converted to the go-cue frame.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts each trial into a binary time series over bins: 1 when a bin center falls between stimulation onset and offset, 0 otherwise. Trials with `'N/A'` onset stay all zero.

ii. 
```python
inputs = np.zeros((2, n_bins), dtype=np.float32)
...
if photostim_onset_valid[i] != 'N/A':
    ...
    inputs[1, :] = ((bin_centers >= stim_onset_go_rel) & (bin_centers < stim_end_go_rel)).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 describes photostimulation as a binary time-varying decoder input and says it should be 1 during the stimulation period.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI expresses photostimulation onset and offset relative to go cue and then compares those times against the same `bin_centers` used for the neural data.

ii. 
```python
stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
...
inputs[1, :] = ((bin_centers >= stim_onset_go_rel) & (bin_centers < stim_end_go_rel)).astype(np.float32)
```

iii. The AI justified this in the trajectory by noting that all modalities must be put on the go-cue-centered axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `BehavioralEvents/left_lick_times` and `BehavioralEvents/right_lick_times`, taking the first lick after the go cue and before trial end. If neither occurs, it labels the trial `no_lick`.

ii. 
```python
def determine_lick_choice(go_time, trial_stop, left_lick_times, right_lick_times):
    left_after = left_lick_times[(left_lick_times > go_time) & (left_lick_times < trial_stop)]
    right_after = right_lick_times[(right_lick_times > go_time) & (right_lick_times < trial_stop)]
    first_left = left_after[0] if len(left_after) > 0 else np.inf
    first_right = right_after[0] if len(right_after) > 0 else np.inf
    if first_left < first_right:
        return 0
    elif first_right < first_left:
        return 1
    else:
        return 2
```

iii. The trajectory shows the AI explicitly verified on a session that hits matched instructed-side licks, misses matched opposite-side licks, and ignores matched no lick, then decided to use lick-event timing directly as the choice readout.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0=left`, `1=right`, `2=no_lick`, then repeats that per-trial value across all 80 bins in the output tensor.

ii. 
```python
choice = determine_lick_choice(
    go_times_valid[i], trial_stops_valid[i],
    left_lick_times, right_lick_times
)
...
outputs = np.zeros((4, n_bins), dtype=np.int64)
outputs[0, :] = choice
```

```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
]
```

iii. `CONVERSION_NOTES.md` Step 5 states the planned choice representation was “first lick direction after go cue” with a third `no_lick` class.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI takes outcome directly from the trials-table `outcome` column.

ii. 
```python
outcomes = trials['outcome'][:]
...
outcomes_valid = outcomes[valid_indices]
```

iii. The notes list the NWB trial outcomes as `hit`, `miss`, and `ignore` and treat that field as the source variable for this decoder output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings to integers with `hit=0`, `miss=1`, `ignore=2`, then repeats that value across all bins.

ii. 
```python
outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}
outcome = outcome_map.get(outcomes_valid[i], 2)
...
outputs[1, :] = outcome
```

```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ['hit', 'miss', 'ignore'],
    ...
]
```

iii. `CONVERSION_NOTES.md` Step 5 documents the intended categorical mapping for outcome and treats it as a per-trial output repeated over time.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI derives early lick from the trials-table `early_lick` field.

ii. 
```python
early_lick = trials['early_lick'][:]
...
early_lick_valid = early_lick[valid_indices]
```

iii. The notes identify `early_lick` as an explicit trial flag in the NWB file and therefore a direct source for the decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'early'` to `1` and everything else to `0`, then repeats that per-trial label across all bins.

ii. 
```python
early = 1 if early_lick_valid[i] == 'early' else 0
...
outputs[2, :] = early
```

```python
'output_values': [
    ...,
    ['no_early', 'early'],
    ...
]
```

iii. `CONVERSION_NOTES.md` Step 5 states the desired output coding was a binary early-lick label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 as `y` and column 2 as tracking likelihood, together with frame timestamps.

ii. 
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

```python
tongue_y = tongue_data[:, 1]
tongue_lk = tongue_data[:, 2]
```

iii. The trajectory shows the AI explored the tongue tracking stream, recognized the columns as x/y/likelihood, and decided to use y-position with a visibility threshold.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first computes session-wide 40th and 60th percentiles from all raw visible tongue frames with likelihood `>= 0.9`. For each trial, it bins frames into the go-cue-centered 50 ms bins, averages visible `y` values within each bin, discretizes the mean with those percentile thresholds, and assigns `3` if no visible frame is present in a bin.

ii. 
```python
visible_mask = tongue_data[:, 2] >= 0.9
if np.any(visible_mask):
    y_visible = tongue_data[visible_mask, 1]
    p40 = np.percentile(y_visible, 40)
    p60 = np.percentile(y_visible, 60)
```

```python
for b in range(n_bins):
    mask = bin_indices_v == b
    if not np.any(mask):
        continue
    visible = trial_lk_v[mask] >= likelihood_threshold
    if np.any(visible):
        mean_y = np.mean(trial_y_v[mask][visible])
        if mean_y < p40:
            result[i, b] = 0
        elif mean_y <= p60:
            result[i, b] = 1
        else:
            result[i, b] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 says the AI chose a likelihood threshold of `0.9`, session-wide percentile thresholds, and 50 ms go-cue-aligned binning, based on its exploratory checks that only about 10% of frames had high tongue visibility.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses `p40` and `p60` from raw visible frames to assign `0` for below the 40th percentile, `1` for 40th-60th percentile, `2` for above the 60th percentile, and `3` for bins with no visible tongue.

ii. 
```python
if mean_y < p40:
    result[i, b] = 0
elif mean_y <= p60:
    result[i, b] = 1
else:
    result[i, b] = 2
```

```python
result = np.full((n_trials, n_bins), 3, dtype=np.int64)
```

iii. The notes explicitly describe the target tongue classes as `<p40`, `p40-p60`, `>p60`, and `not_visible`, and the AI implemented that using percentiles from visible tongue frames.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data to go cue by converting each trial’s window into absolute frame edges `go + bin_edges`, selecting the corresponding camera frames, and assigning them into the same 50 ms bins as the neural data.

ii. 
```python
abs_edges = go + bin_edges
idx_start = np.searchsorted(tongue_timestamps, abs_edges[0])
idx_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
...
bin_indices = np.searchsorted(abs_edges, trial_ts, side='right') - 1
```

iii. The trajectory shows the AI wanted the tongue output to share the exact go-cue-centered time base as the neural firing rates.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data in several ad hoc ways: sessions with no good units are skipped; trials outside the estimated recording window are removed; trials with no sample event get a synthetic tone onset at `go-1.85`; sessions without tongue data get all tongue bins set to class `3`; and tongue bins with no visible frames remain `3`.

ii. 
```python
if n_good == 0:
    print(f"  Skipping {session_id}: no good units")
    return None
```

```python
if len(samp_in_trial) > 0:
    tone_onsets[i] = samp_in_trial[-1]
else:
    tone_onsets[i] = go - 1.85
```

```python
if has_tongue and p40 is not None:
    tongue_y_list = compute_tongue_y_vectorized(...)
else:
    tongue_y_list = [np.full(n_bins, 3, dtype=np.int64) for _ in range(n_valid)]
```

iii. The rationale is split between the notes and the trajectory: Step 10 says the recording-coverage filter was added to remove zero-neural-data edge cases; Step 5/trajectory step 36 says the `-1.85` s tone fallback was based on the usual task timing; and the tongue handling comes from the assumption that low-likelihood or absent tongue observations should map to `not_visible`.

## 10-a. What are the most time-consuming steps of the code?

i. The AI treated loading each session, computing firing rates, and to a lesser extent tongue processing as the dominant costs, with firing rates clearly the main compute bottleneck.

ii. 
```python
t_load_spikes = time.time()
print(f"  Loaded {n_good} good units in {t_load_spikes - t0:.1f}s ...")
...
t_fr = time.time()
print(f"  Computed firing rates in {t_fr - t_load:.1f}s")
...
t_tongue = time.time()
print(f"  Computed tongue y in {t_tongue - t_fr:.1f}s")
```

iii. `CONVERSION_NOTES.md` Step 7 reports per-session timing estimates and identifies firing-rate computation as the dominant step; Step 6 separately discusses tongue-processing speedups.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still contains a nested neuron-by-trial loop for firing-rate computation, a trial-by-bin loop for tongue discretization, and per-trial loops for building input and output arrays.

ii. 
```python
for j, spk_times in enumerate(spike_times_list):
    ...
    for i in range(n_trials):
        ...
        counts, _ = np.histogram(relative_spikes, bins=bin_edges)
```

```python
for i in range(n_trials):
    ...
    for b in range(n_bins):
        mask = bin_indices_v == b
```

```python
for i in range(n_valid):
    inputs = np.zeros((2, n_bins), dtype=np.float32)
    ...
for i in range(n_valid):
    outputs = np.zeros((4, n_bins), dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 6 says the AI already optimized some slower versions, but the final code still leaves these loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The AI code repeats several computations trial-by-trial: histogramming spikes separately for every neuron/trial pair, rebuilding input arrays one trial at a time, rebuilding output arrays one trial at a time, and scanning each tongue bin separately within each trial.

ii. 
```python
for j, spk_times in enumerate(spike_times_list):
    ...
    for i in range(n_trials):
        ...
```

```python
for i in range(n_valid):
    inputs = np.zeros((2, n_bins), dtype=np.float32)
    ...
    input_list.append(inputs)
```

```python
for i in range(n_valid):
    ...
    outputs = np.zeros((4, n_bins), dtype=np.int64)
    ...
    output_list.append(outputs)
```

iii. The notes emphasize speedups relative to even slower drafts, but the final code still recomputes many per-trial structures that the reference solution builds more globally.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some work that is not needed in the final saved dataset: it always loads `instructions_valid` even though it is only used for optional plotting, returns `bin_centers` inside each per-session result even though `assemble_dataset` drops it, and contains optional plotting machinery that is not part of the converted pickle.

ii. 
```python
instructions = trials['trial_instruction'][:]
...
instructions_valid = instructions[valid_indices]
```

```python
return {
    'neural': fr_list,
    'input': input_list,
    'output': output_list,
    ...
    'bin_centers': bin_centers,
}
```

```python
if show_processing:
    plot_processing(session_id, bin_centers, fr_list, input_list, output_list,
                   go_times_valid, outcomes_valid, instructions_valid,
                   early_lick_valid, tongue_y_list, session_idx)
```

iii. This is mostly visible from the code itself. The notes also describe the optional processing plots and profiling utilities, which are useful for debugging but not part of the downstream saved dataset.
