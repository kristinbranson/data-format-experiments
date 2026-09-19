# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data by listing every `sub-*` directory under `/app/data`, then listing every `.nwb` file inside each subject directory. Each file is treated as one session and is opened with `NWBHDF5IO`. Within `process_session`, the code reads `nwb.units`, `nwb.trials`, `nwb.acquisition['BehavioralEvents']`, and `nwb.acquisition['BehavioralTimeSeries']`.

ii. 
```python
def get_nwb_files(data_dir):
    """Get list of all NWB files organized by subject."""
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
...
trials = nwb.trials
events = nwb.acquisition['BehavioralEvents']
...
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
```

iii. In `CONVERSION_NOTES.md`, the agent says the NWB files are "organized by subject (sub-XXXXXX directories)" and that "Each NWB file = one session." Trajectory step 58 records the same understanding of the file layout and NWB contents.

## 1-b. How are the data split into subjects?

i. Subjects are split by directory name, not by reading `nwb.subject.subject_id`. The subject label passed through the pipeline is the folder name such as `sub-440956`. In the assembled output, `subjects` is the list of subject folders in first-seen order and `subject_idx` maps sessions to those entries.

ii. 
```python
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for fname in files:
        nwb_files.append((subj, os.path.join(subj_dir, fname)))
```

```python
if subject_id not in subjects_seen:
    subjects_seen[subject_id] = len(subjects_seen)
    subjects_list.append(subject_id)
subj_idx = subjects_seen[subject_id]
...
'subjects': subjects_list,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The notes justify the dataset as already "organized by subject (sub-XXXXXX directories)" and report 28 subjects from that layout. The notes also planned to use subject metadata, but the code itself relies on the directory names instead.

## 1-c. How are the data split into sessions?

i. The code treats each NWB file as one session. Session order is the sorted order produced by `get_nwb_files`, and each processed session stores the NWB basename as `session_name`.

ii. 
```python
files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
for fname in files:
    nwb_files.append((subj, os.path.join(subj_dir, fname)))
```

```python
return {
    ...
    'session_name': os.path.basename(nwb_path),
}
```

iii. `CONVERSION_NOTES.md` explicitly states "Each NWB file = one session" and reports 174 NWB files total, with 173 retained after filtering.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials`, and the code indexes all other per-trial arrays by the same trial index. After filtering, the surviving `trial_indices` are iterated one by one, and the code uses `go_start_times[ti]`, `outcomes[ti]`, `instructions[ti]`, and other trial-aligned arrays for that trial.

ii. 
```python
trials = nwb.trials
outcomes = trials['outcome'][:]
instructions = trials['trial_instruction'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```

```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
...
trial_indices = np.where(trial_mask)[0]
...
for ti in trial_indices:
    go_cue = go_start_times[ti]
```

iii. In the trajectory, the agent noted that `go_start_times` matches trial count whereas `sample_start_times` can have extra entries because of replays after early licks. That led it to use the trials table plus per-trial indexing, rather than trying to reconstruct trials from the repeated event streams.

## 1-e. How are trials filtered based on quality controls?

i. The implemented trial filter keeps only trials that satisfy three conditions: `auto_water == 0`, `free_water == 0`, and membership in a recording mask derived from `obs_intervals`. The recording mask is computed by grouping good units by `len(obs_intervals)`, finding trial starts within 1 second of any `obs_intervals` start for each group, and taking the intersection across groups. Sessions with fewer than two surviving trials are dropped.

ii. 
```python
def get_valid_trial_indices(nwb, good_unit_indices, n_trials):
    ...
    for idx in good_unit_indices:
        obs = nwb.units['obs_intervals'][idx]
        n_obs = len(obs)
        if n_obs not in obs_sets:
            obs_sets[n_obs] = obs
    ...
    for n_obs, obs in obs_sets.items():
        obs_starts = obs[:, 0]
        diffs = np.abs(trial_starts[:, None] - obs_starts[None, :])
        min_diffs = diffs.min(axis=1)
        covered = min_diffs < 1.0
        valid_trials &= covered
```

```python
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]

if len(trial_indices) < 2:
    io.close()
    return None
```

iii. In `CONVERSION_NOTES.md`, the agent says it would exclude auto-water and free-water trials while keeping early-lick, ignore, and photostim trials because those are decoder variables. After debugging, trajectory step 126 adds the justification that `obs_intervals` must also be used because some units were recorded for only part of a session and later trials had no valid spikes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for units passing the code's neuron filters, together with `BehavioralEvents/go_start_times` to place the trial window. The code first identifies `classification == 'good'` units and then further filters them by whether their `anno_name` maps to one of the predefined coarse regions.

ii. 
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
...
anno_names = nwb.units['anno_name'][:][good_mask]
```

```python
spike_times_per_unit = []
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
...
go_start_times = events.time_series['go_start_times'].timestamps[:]
```

iii. The notes say the relevant raw sources are absolute `spike_times`, `classification == 'good'`, and go-cue timestamps. The notes also justify using `anno_name` because they wanted to map units into the 14 coarse regions used by the reference code.

## 2-b. How is the `neural` data processed?

i. Neural activity is processed into 50 ms non-overlapping firing-rate bins in Hz over `[-2.5, 1.5)` seconds relative to each trial's go cue. For each trial and neuron, spikes are filtered into the window, assigned to bins by integer division, counted with `np.add.at`, and divided by the bin size. No smoothing, baseline subtraction, or normalization is applied.

ii. 
```python
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

iii. The notes explicitly say the task specification overrides the paper's 40 ms / 3.4 ms preprocessing, so the agent switched to 50 ms non-overlapping bins for the decoder output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code filters neurons in two stages. First it keeps only `classification == 'good'` units. Then it drops any of those units whose `anno_name` does not map to one of the 14 `COARSE_REGIONS`. If no good units or no mapped units remain, the session is skipped.

ii. 
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
n_good = good_mask.sum()

if n_good == 0:
    io.close()
    return None
```

```python
region_indices = []
valid_neuron_mask = np.ones(n_good, dtype=bool)
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
        region_indices.append(-1)
    else:
        region_indices.append(COARSE_REGIONS.index(region))
...
good_unit_indices = np.where(good_mask)[0]
good_unit_indices = good_unit_indices[valid_neuron_mask]
```

iii. The notes justify `classification == 'good'` as the QC classifier output from the spike-sorting white paper. They also justify the extra `anno_name` mapping by saying they wanted the 14 coarse regions from the reference pipeline, though this second filter is not described there as a QC step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done by converting the relative trial window to absolute times using the go cue for each trial. For each trial, `abs_start = go_cue + T_START` and `abs_end = go_cue + T_END`, and spikes are binned within that absolute interval.

ii. 
```python
T_START = -2.5
T_END = 1.5
...
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

```python
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
mask = (st >= abs_start) & (st < abs_end)
```

iii. The notes repeatedly state that NWB spike times are absolute rather than already go-cue aligned, so the agent's justification was to align everything by the go cue on the shared absolute session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins, with 80 bins per trial spanning `-2.5` to `1.5` s relative to go cue. No additional temporal rebinning is applied beyond this direct binning from raw spikes and video frames.

ii. 
```python
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

iii. `CONVERSION_NOTES.md` explicitly says the agent chose 50 ms non-overlapping bins because the decoder task required that, even though the reference paper used a different spike-binning setup.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `BehavioralEvents/sample_start_times` and `BehavioralEvents/go_start_times`. For each trial, the code finds the last sample onset before that trial's go cue.

ii. 
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
...
go_cue = go_start_times[ti]
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
```

```python
def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]
```

iii. In trajectory steps 54 and 76, the agent observed that `sample_start_times` has more entries than trials because sample epochs can replay after early licks, and concluded that it should use the last sample onset before each go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code first finds the last `sample_start_times` entry before the trial's go cue. If none is found, it fabricates a fallback tone onset at `go_cue - 1.85`. It then computes 80 bin centers relative to the go cue and converts them into "time since tone onset" by subtracting the tone time relative to the go cue.

ii. 
```python
def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time
    time_from_tone = bin_centers - tone_rel
    return time_from_tone.astype(np.float32)
```

```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)
```

iii. The recorded justification is that the last sample onset is the relevant tone because early licks can replay the sample epoch. The fallback `1.85 s` rule does not appear in the notes as a planned decision; it is only visible in the final code.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The tone input uses the same 80 bin centers and the same go-cue-centered trial window as the neural data. It is stacked as the first row of the per-trial input array, with one value per neural time bin.

ii. 
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
...
time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The notes say the decoder should align everything to the go cue and use a single 50 ms time grid, so the tone input follows the same grid as the firing rates.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The code derives photostimulation from the absolute event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from the trial-table columns `photostim_onset` and `photostim_duration`.

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

iii. `CONVERSION_NOTES.md` explicitly states "Use absolute photostim_start/stop_times aligned to go cue." In trajectory step 54, the agent also contrasts the trial-table relative onset values with the absolute event timestamps and chooses the latter.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code finds all photostim intervals that overlap the `[-2.5, 1.5]` s window around that trial's go cue, computes absolute bin centers for the trial, and sets bins to `1.0` when the bin center falls between any interval's start and stop time. Non-overlapping bins remain `0.0`.

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

```python
photostim_ts = compute_photostim_timeseries(go_cue, ps_starts, ps_stops,
                                              T_START, T_END, BIN_SIZE_S)
```

iii. The notes justify photostim as a binary time-varying decoder input rather than a per-trial flag. The trajectory shows the agent wanted to keep photostim trials because photostim is an explicit decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done on the absolute session clock. The neural bins are centered at absolute times `go_cue + bin_center`, and the photostim intervals are already stored as absolute timestamps, so the code compares them directly.

ii. 
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
abs_centers = bin_centers + go_cue_time
```

```python
trial_abs_start = go_cue + T_START
trial_abs_end = go_cue + T_END
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
```

iii. The notes justify this by saying the photostim event times are absolute and should be aligned to the same go-cue-based frame used for neural binning.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the implemented code, choice is derived only from the `trial_instruction` field. A left instruction becomes choice `0`, and anything else becomes choice `1`. The code does not use `outcome` to derive actual executed lick direction or a "no lick" class.

ii. 
```python
instructions = trials['trial_instruction'][:]
...
choice = 0 if instructions[ti] == 'left' else 1
```

iii. In `CONVERSION_NOTES.md`, the mapping table explicitly says `trial_instruction -> output[0]: choice | left=0, right=1`, so the agent appears to have equated instructed side with lick choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code emits a two-class per-trial choice variable (`0` left, `1` right) and repeats that value across all 80 bins. The metadata likewise lists only `['left', 'right']` as legal choice values.

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

iii. The notes justify this only indirectly by defining choice from `trial_instruction` and later reporting "Output ranges: choice [0,1]".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the `outcome` column of the trials table.

ii. 
```python
outcomes = trials['outcome'][:]
...
outcome_str = outcomes[ti]
```

iii. The notes' variable-mapping table explicitly maps `outcome -> output[1]: outcome`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings `'ignore'`, `'miss'`, and `'hit'` are mapped to `0`, `1`, and `2`, then repeated across all 80 bins.

ii. 
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
...
full_output[1, :] = out_dict['outcome']
```

iii. The notes justify this as a direct categorical remapping matching the decoder specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the `early_lick` column of the trials table.

ii. 
```python
early_lick = trials['early_lick'][:]
...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. The notes' variable-mapping table explicitly maps `early_lick -> output[2]: early_lick`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are binarized as `0` for `'no early'` and `1` otherwise, then repeated across all 80 bins.

ii. 
```python
early_val = 0 if early_lick[ti] == 'no early' else 1
...
full_output[2, :] = out_dict['early_lick']
```

iii. The notes justify keeping early-lick trials because early lick is itself a decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The code uses column 1 as `tongue_y` and column 2 as tracking likelihood, with `timestamps` from the same time series.

ii. 
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. The notes say tongue tracking has 3 columns `(tongue_x, tongue_y, tongue_likelihood)`, and trajectory step 54 records that the agent identified column 2 as the likelihood value from DeepLabCut.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code extracts tongue frames in each trial's window, sets `likelihood < 0.9` to `NaN`, averages visible tongue y-values within each 50 ms bin, pools all non-NaN per-trial binned values across the session, computes the 40th and 60th percentiles of that pooled set, and then discretizes each trial's binned values against those two thresholds.

ii. 
```python
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
...
y_in_window = y_in_window.copy()
y_in_window[like_in_window < 0.9] = np.nan
...
sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
counts = np.bincount(valid_bins, minlength=n_bins)
has_data = counts > 0
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

iii. The notes justify the general plan as "Use column 1 (tongue_y) from TongueTracking" and "Discretize per-session using 40th/60th percentiles over ALL valid time points." The trajectory records that the agent identified the likelihood channel, but there is no explicit note justifying the specific `0.9` threshold.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code thresholds visible binned tongue y-values into three categories: `< p40 -> 0`, `p40..p60 -> 1`, and `> p60 -> 2`. It does not create an explicit class `3` for "not visible"; instead, bins with no visible tongue remain at the default value `0` because `tongue_y_disc` is initialized to zeros and only valid bins are reassigned. The metadata also advertises only three tongue categories.

ii. 
```python
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['below_40th', '40th_to_60th', 'above_60th'],
],
```

iii. The notes' mapping table and validation summary both reflect only the three visible classes; step 10 of the notes says "Output ranges ... tongue [0,2]".

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, the code takes camera frames between `go_cue + T_START` and `go_cue + T_END` and bins them on the same 50 ms go-cue-centered grid used for neural activity.

ii. 
```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end

mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
...
bin_indices = np.digitize(t_in_window, bin_edges) - 1
```

iii. The notes say all streams should be aligned to the go cue in the same `[-2.5, 1.5]` s window, so the tongue series follows the same time axis as the neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several cases explicitly: sessions with no `classification == 'good'` units are skipped; sessions where all good units fail coarse region mapping are also skipped; trials outside derived recording coverage are removed via `obs_intervals`; sessions with fewer than two valid trials are skipped; a missing sample onset falls back to `go_cue - 1.85`; low-likelihood tongue frames become `NaN`; and trial bins with no visible tongue stay `NaN` until discretization, where they effectively collapse to class `0`.

ii. 
```python
if n_good == 0:
    io.close()
    return None
...
if n_neurons == 0:
    io.close()
    return None
...
if len(trial_indices) < 2:
    io.close()
    return None
```

```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
```

```python
y_in_window[like_in_window < 0.9] = np.nan
...
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
```

iii. The notes justify session skipping for the one session with no good units and later justify `obs_intervals` filtering after the agent discovered that some probes covered only a subset of trials. The notes also justify discarding low-confidence tongue frames, but they do not give an explicit justification for fabricating a tone onset or for mapping non-visible tongue bins to `0`.

## 10-a. What are the most time-consuming steps of the code?

i. The code structure suggests the dominant work is repeated per-trial neural binning and repeated per-trial tongue extraction. For every kept trial, the code loops through every neuron in `compute_firing_rates`, and it separately scans tongue frames in the same window. The notes report about `~6.5 s/session` in sample mode and `2300.8 s` for full conversion.

ii. 
```python
for ti in trial_indices:
    go_cue = go_start_times[ti]

    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
    neural_trials.append(fr)
    ...
    tongue_y_trial = get_tongue_y_for_trial(tongue_y_all, tongue_timestamps,
                                              tongue_likelihood, go_cue,
                                              T_START, T_END, BIN_SIZE_S)
```

```python
for i, st in enumerate(spike_times_list):
    ...
    np.add.at(fr[i], bin_indices, 1.0)
```

iii. The notes do not include a detailed profiler breakdown, but they do estimate runtime and single out "Load + process" as the dominant per-session cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized further: the outer per-trial loop in `process_session`; the inner per-neuron loop in `compute_firing_rates`; the repeated scan over all `sample_start_times` for each trial; the repeated overlap test against all photostim intervals for each trial; the per-unit `obs_intervals` grouping; and the second pass that builds `final_outputs` trial by trial.

ii. 
```python
for ti in trial_indices:
    ...
    tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
    ...
    stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
```

```python
for i, st in enumerate(spike_times_list):
    ...
```

```python
for idx in good_unit_indices:
    obs = nwb.units['obs_intervals'][idx]
    ...
```

iii. There is no explicit justification in the notes for leaving these loops unvectorized. The recorded trajectory focused on getting the semantics right rather than on optimization.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations across trials: it rebuilds bin centers inside helper functions; it rescans `sample_start_times` for every trial; it rescans the global photostim events for every trial; it loops over all neurons again for every trial when computing firing rates; it bins tongue data once per trial and then makes a second pass across stored trial outputs to discretize tongue values.

ii. 
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```

```python
for ti in trial_indices:
    ...
    tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
    ...
    stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
```

```python
for out_dict in output_trials_raw:
    tongue_y_raw = out_dict['tongue_y_raw']
    ...
    final_outputs.append(full_output)
```

iii. The notes do not defend these repeated computations directly. They instead emphasize correctness and sanity checks, and only later report runtime after the implementation was already in place.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes some work that is only for diagnostics or temporary staging rather than the final dataset: optional `plot_processing` figures, post-save summary statistics with `Counter`, temporary `output_trials_raw` dictionaries and `tongue_y_session` accumulation before building the final arrays, and session bookkeeping fields such as `session_name`, `n_trials`, and `n_neurons` that are used for logging but not written into the saved dataset.

ii. 
```python
output_trials_raw = []
tongue_y_session = []  # for percentile computation
...
output_trials_raw.append({
    'choice': choice,
    'outcome': outcome_val,
    'early_lick': early_val,
    'tongue_y_raw': tongue_y_trial
})
```

```python
if args.show_processing and i < 2:
    plot_processing(result, i, nwb_path)
    print(f"  Saved processing plot")
...
all_choices = []
all_outcomes = []
all_early = []
...
print(f"  Choice: {Counter(all_choices)}")
```

iii. `CONVERSION_NOTES.md` justifies the plotting and summary work as sanity checks and validation aids. The notes explicitly mention processing visualizations and output-distribution checks as part of its review workflow.
