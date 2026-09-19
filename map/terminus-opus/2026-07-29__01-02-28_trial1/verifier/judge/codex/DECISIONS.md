# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the `data/` directory with `os.listdir`, treats each `sub-*` directory as a subject, and each `.nwb` file inside it as one session. Each session file is opened directly with `h5py`, and the converter reads HDF5 groups for trials, units, behavioral events, and behavioral time series.

ii.
```python
def get_nwb_files():
    """Get all NWB file paths organized by subject."""
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj,
                'path': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file
            })
    return all_files
```

```python
f = h5py.File(nwb_path, 'r')
trials = f['intervals']['trials']
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
```

iii. The justification in `CONVERSION_NOTES.md` is that the data are distributed as 28 `sub-*` directories with 174 NWB files total, and that the AI mapped NWB fields directly rather than using the `.mat`-based reference code. The trajectory also shows the AI explicitly chose direct HDF5 reads after exploring the NWB layout.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the parent directory name, e.g. `sub-440956`. The code passes that folder name through per session, accumulates first-seen unique subject ids in `all_subjects`, and stores `subject_idx` as the index of each session's folder-derived subject.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    ...
    all_files.append({
        'subject': subj,
        'path': os.path.join(subj_dir, nwb_file),
        'filename': nwb_file
    })
```

```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

```python
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` says the dataset has 28 subject directories and describes the data as “28 subjects (sub-440956 through sub-484677).” I did not find a separate justification for using the folder name instead of `nwb.subject.subject_id`; the decision appears to follow directly from the directory traversal.

## 1-c. How are the data split into sessions?

i. One `.nwb` file is treated as one session. In full mode the script iterates all sorted session files; in sample mode it subsets that list, but the session boundary is still the individual file.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
for nwb_file in nwb_files:
    all_files.append({
        'subject': subj,
        'path': os.path.join(subj_dir, nwb_file),
        'filename': nwb_file
    })
```

```python
for i, nwb_info in enumerate(nwb_files):
    print(f"\nProcessing session {i+1}/{len(nwb_files)}: {nwb_info['filename']}")
    result = process_session(
        nwb_info['path'],
        nwb_info['subject'],
        show_processing=args.show_processing,
        session_idx=session_count
    )
```

iii. The notes describe “174 NWB files total” and repeatedly equate those with sessions. The trajectory shows the AI adopting the one-file-per-session interpretation after inspecting the dataset structure.

## 1-d. How are the data split into trials?

i. Trials are taken from the HDF5 trial table under `intervals/trials`, with all per-trial variables read as parallel arrays. The code uses integer `trial_indices` into those arrays, and uses the corresponding `go_times[trial_idx]` for temporal alignment.

ii.
```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])

trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
```

```python
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
```

iii. The trajectory shows the AI concluding that `go_start_times` matches trials one-for-one and then using trial-table rows as the trial definition. I did not find an explicit code-level assertion checking trial count versus go-cue count.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in several stages. At the trial level it keeps only trials with `auto_water == 0`, `free_water == 0`, and `neural_valid == True`, where `neural_valid` means the trial index is below `units/is_good_trials.shape[1]` and the full `[-2.5, 1.5] s` window lies within the global spike-time min/max. At the session level it then drops sessions unless they have at least 2 kept trials, at least one valid non-early non-photostim control trial, overall control correct rate at least 65%, and at least 50 correct left and 50 correct right control trials.

ii.
```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
...
neural_valid = np.zeros(n_trials_total, dtype=bool)
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True
```

```python
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
...
if n_selected < 2:
    ...
```

```python
is_control = (photostim_onset_trial == b'N/A') & trial_mask
is_not_early = early_lick == b'no early'
control_non_early = is_control & is_not_early
...
correct_rate = hits / (hits + misses)
correct_left = np.sum((outcome == b'hit') & (trial_instruction == b'left') & control_non_early)
correct_right = np.sum((outcome == b'hit') & (trial_instruction == b'right') & control_non_early)
...
if correct_rate < MIN_CORRECT_RATE:
    return None

if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
    return None
```

iii. `CONVERSION_NOTES.md` says the AI intentionally excluded `auto_water` and `free_water`, kept early-lick / ignore / photostim trials for decoder outputs or inputs, and used `is_good_trials.shape[1]` plus spike-time coverage to handle partial neural recording. The trajectory shows the session performance filter was imported from the methods text after the AI noticed 174 raw sessions versus 173 in the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `units/spike_times_index`, restricted to units where `units/classification == b'good'`, and aligned relative to `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
```

```python
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
```

```python
firing_rates = compute_firing_rates_fast(
    spike_times_flat, spike_times_index, go_times,
    good_indices, trial_indices
)
```

iii. The notes explicitly say “Use classifier-based QC: keep only units with classification='good'” and “Spike times in NWB are absolute timestamps, aligned to go cue during conversion.” The trajectory shows the AI identified `classification` as the QC filter after comparing the NWB structure to the reference pipeline.

## 2-b. How is the `neural` data processed?

i. For each selected trial and each good unit, the AI takes spikes whose absolute timestamps fall within `[go + WINDOW_START, go + WINDOW_END)`, subtracts the trial's go-cue time, histograms them into 50 ms bins, and divides counts by bin width to get firing rates in Hz.

ii.
```python
def compute_firing_rates_fast(spike_times_flat, spike_times_index, go_cue_times,
                               good_indices, trial_indices,
                               window_start=WINDOW_START, window_end=WINDOW_END,
                               bin_width=BIN_WIDTH):
    ...
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
    ...
    for trial_idx in trial_indices:
        go_time = go_cue_times[trial_idx]
        fr_trial = np.zeros((n_good, n_bins), dtype=np.float32)
        
        abs_start = go_time + window_start
        abs_end = go_time + window_end
        
        for i, unit_spikes in enumerate(good_spike_times):
            idx_lo = np.searchsorted(unit_spikes, abs_start)
            idx_hi = np.searchsorted(unit_spikes, abs_end)
            
            if idx_hi > idx_lo:
                aligned = unit_spikes[idx_lo:idx_hi] - go_time
                counts = np.histogram(aligned, bins=bin_edges)[0]
                fr_trial[i, :] = counts / bin_width
```

iii. The notes say the AI used `np.searchsorted` and pre-extracted per-unit spike arrays for speed, and computed firing rate as spike counts divided by bin width. The trajectory shows the AI recognized that spike times were absolute, not pre-aligned, and needed explicit go-cue alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered solely by `units/classification == b'good'`. Sessions with zero such units are dropped. No per-metric thresholds are applied.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
n_good = int(np.sum(good_mask))
n_total = len(classification)
good_indices = np.where(good_mask)[0]
...
if n_good == 0:
    print(f"    Skipping: no good units")
    f.close()
    return None
```

iii. `CONVERSION_NOTES.md` explicitly ties the NWB `classification` field to the classifier-based QC used in the reference pipeline. The trajectory shows the AI decided that `classification == 'good'` in NWB corresponds to the paper's QC-approved units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to its go-cue onset. The converter uses `go_times[trial_idx]` as time zero, takes a window from `-2.5 s` to `+1.5 s` around that go cue, and histograms spikes after subtracting `go_time`.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
```

```python
for trial_idx in trial_indices:
    go_time = go_cue_times[trial_idx]
    ...
    abs_start = go_time + window_start
    abs_end = go_time + window_end
    ...
    aligned = unit_spikes[idx_lo:idx_hi] - go_time
    counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. The notes say “Temporal alignment: go cue = time 0” and the trajectory shows the AI verifying the path to `BehavioralEvents/go_start_times` before implementing the conversion.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms non-overlapping bins over a fixed `[-2.5, 1.5] s` window, yielding 80 time bins per trial. There is no further temporal rebinning after the histogram step.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
WINDOW_START = -2.5  # seconds relative to go cue
WINDOW_END = 1.5  # seconds relative to go cue
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))  # 80 bins
```

```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` says the reference used 40 ms / 3.4 ms sliding windows, but the AI intentionally switched to 50 ms bins over `[-2.5, 1.5] s` because that is what the decoder task specified.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `sample_start_times` and `go_start_times`. For each selected trial, the converter takes the last sample start before the go cue as the tone onset for that trial.

ii.
```python
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
```

```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
```

iii. The notes say the AI manually checked that `time_from_tone_onset` matched `sample_start_times`, and the trajectory shows the AI noticed that `sample_start_times` can have more entries than trials because early licks replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes the center of each 50 ms bin on the go-cue-relative grid, computes the selected trial's tone onset relative to the go cue, and then sets `time_from_tone = bin_centers - tone_onset_rel`. If no earlier sample start exists, it falls back to a hard-coded `tone_onset_rel = -1.85`.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
```

```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. `CONVERSION_NOTES.md` says the AI validated this input against a manual computation from NWB `sample_start_times`. The fallback does not appear to be separately justified beyond defensive handling of a missing preceding sample onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is placed on the exact same per-trial 80-bin grid as the neural data by using the same `bin_centers` relative to the go cue.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
time_from_tone = bin_centers - tone_onset_rel
```

```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
...
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. The justification is implicit in the implementation and in the notes’ statement that the decoder uses a shared 50 ms, go-cue-centered window for all modalities.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from per-trial `photostim_onset`, `photostim_duration`, `start_time`, and the trial's `go_time`. The onset is stored relative to trial start and is converted to a go-cue-relative interval.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
```

```python
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float = float(photostim_duration_trial[trial_idx])
    trial_start = trial_start_times[trial_idx]
    ps_rel_start = (trial_start + ps_onset_float) - go_time
    ps_rel_end = ps_rel_start + ps_dur_float
```

iii. The trajectory shows the AI explicitly checked that `photostim_onset` is stored relative to trial start and that stimulation occurs about 0.5 s before the go cue, matching the methods text.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each selected trial the AI creates a binary time series of length 80, initialized to zeros. If the trial has photostimulation, it marks bins whose centers fall in `[ps_rel_start, ps_rel_end)` as 1.

ii.
```python
photostim_binary = np.zeros(n_bins, dtype=np.float32)
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ...
    photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

```python
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
```

iii. `CONVERSION_NOTES.md` says the AI intentionally kept photostim trials because photostimulation is a required decoder input, even though the reference analysis code excluded stimulated trials for its own analyses.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation interval is converted to time relative to the trial's go cue and then compared to the same go-cue-relative `bin_centers` used for the neural data.

ii.
```python
ps_rel_start = (trial_start + ps_onset_float) - go_time
ps_rel_end = ps_rel_start + ps_dur_float
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The trajectory shows the AI first verified the raw timing relationship between `photostim_onset` and `go_start_times`, then adopted the go-cue-relative alignment used elsewhere in the converter.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives `choice` only from `trial_instruction`. It does not use `outcome`, so misses are not flipped to the opposite lick side and ignores do not get a separate “no lick” class.

ii.
```python
trial_instruction = trials['trial_instruction'][:]
...
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. `CONVERSION_NOTES.md` lists the mapping as `trial_instruction -> output[0] choice`, with left `= 0` and right `= 1`. I did not find any explicit justification in the notes or trajectory for ignoring `outcome` in this derived variable.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps `left` trials to `0` and all other trials to `1`, then repeats that per-trial value across all 80 bins. The output metadata defines only two `choice` labels, `left` and `right`.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
...
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = choice
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['below_p40', 'p40_to_p60', 'above_p60'],
],
```

iii. The only justification I found is the brief mapping table in `CONVERSION_NOTES.md`, which treats `trial_instruction` as if it were equivalent to lick choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` field.

ii.
```python
outcome = trials['outcome'][:]
...
out = outcome[trial_idx]
```

iii. The trajectory shows the AI identified the NWB `outcome` field as already containing `hit`, `miss`, and `ignore`, so no further derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that code across all 80 bins for the trial.

ii.
```python
out = outcome[trial_idx]
if out == b'ignore':
    outcome_val = 0
elif out == b'miss':
    outcome_val = 1
elif out == b'hit':
    outcome_val = 2
else:
    outcome_val = 0
...
output_trial[1, :] = outcome_val
```

iii. `CONVERSION_NOTES.md` states this mapping explicitly in the variable-mapping table and says outcome should be kept because it is a required decoder output.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` field.

ii.
```python
early_lick = trials['early_lick'][:]
...
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. The notes say the AI intentionally retained early-lick trials because `early_lick` is one of the required decoder outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The converter maps `b'early'` to `1` and everything else to `0`, then repeats that per-trial label across the full 80-bin trial.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
...
output_trial[2, :] = early
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['below_p40', 'p40_to_p60', 'above_p60'],
],
```

iii. `CONVERSION_NOTES.md` gives the intended mapping `no=0, yes=1` and explains that early-lick trials were kept because this variable must be decodable.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the `Camera0_side_TongueTracking` time series: column 1 of the `data` array is treated as `y`, and column 2 as the tracking likelihood, with `timestamps` used for alignment.

ii.
```python
has_tongue = 'Camera0_side_TongueTracking' in f['acquisition']['BehavioralTimeSeries']

if has_tongue:
    tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
    tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
```

```python
y_slice = tongue_data[idx_start:idx_end, 1]
lk_slice = tongue_data[idx_start:idx_end, 2]
```

iii. The trajectory shows the AI verified that the tongue-tracking columns are `(x, y, likelihood)` and that the stream is sampled at about 300 Hz before implementing this output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each selected trial, the AI extracts tongue frames from a slightly padded interval around the go-cue window, keeps only frames with `likelihood > 0.1`, and averages the surviving `y` values within each 50 ms bin. It then pools all valid binned tongue values from the selected trials of that session and computes the 40th and 60th percentiles of that pooled set. Those thresholds are later used to discretize each trial.

ii.
```python
def get_tongue_y_for_trials(tongue_data, tongue_timestamps, go_times, trial_indices,
                            window_start=WINDOW_START, window_end=WINDOW_END,
                            bin_width=BIN_WIDTH):
    ...
    for trial_idx in trial_indices:
        go_time = go_times[trial_idx]
        abs_start = go_time + window_start - bin_width
        abs_end = go_time + window_end + bin_width
        ...
        tongue_y_binned = np.full(n_bins, np.nan)
        ...
        if idx_start < idx_end:
            ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
            y_slice = tongue_data[idx_start:idx_end, 1]
            lk_slice = tongue_data[idx_start:idx_end, 2]
            high_conf = lk_slice > 0.1
            
            for b in range(n_bins):
                bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
                n_in_bin = np.sum(bin_mask)
                if n_in_bin > 0:
                    tongue_y_binned[b] = np.mean(y_slice[bin_mask])
        ...
        valid_y = tongue_y_binned[~np.isnan(tongue_y_binned)]
        if len(valid_y) > 0:
            all_tongue_y.extend(valid_y.tolist())
```

```python
if len(all_tongue_y) > 0:
    all_tongue_y_arr = np.array(all_tongue_y)
    p40 = np.percentile(all_tongue_y_arr, 40)
    p60 = np.percentile(all_tongue_y_arr, 60)
else:
    p40, p60 = 0, 0
```

iii. `CONVERSION_NOTES.md` says the AI intended to compute per-session percentiles from “all valid (likelihood>0.1) tracking data.” The trajectory later noticed that the resulting tongue distribution was overly concentrated in the middle category, but the final code was not changed.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The converter creates three categories only: below `p40` is `0`, between `p40` and `p60` is `1`, and above `p60` is `2`. Missing bins are not given a separate class; instead `tongue_y_disc` is initialized to all ones, so bins with `NaN` tongue values remain in the middle category.

ii.
```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['below_p40', 'p40_to_p60', 'above_p60'],
],
```

iii. The trajectory explicitly notes this behavior: “many timepoints have NaN tongue data (tongue not visible), and I'm defaulting to middle.” I did not find a later justification beyond that observation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue frames are aligned to the same go-cue-centered trial window as the neural data. For each selected trial, the code finds the relevant camera-frame slice by searching absolute timestamps around `go_time + window`, then bins frames by their offsets relative to `go_time`.

ii.
```python
go_time = go_times[trial_idx]
abs_start = go_time + window_start - bin_width
abs_end = go_time + window_end + bin_width

idx_start = np.searchsorted(tongue_timestamps, abs_start)
idx_end = np.searchsorted(tongue_timestamps, abs_end)
```

```python
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
...
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. The notes say tongue y is “binned to match neural data,” and the trajectory shows the AI intentionally used the same `[-2.5, 1.5] s`, 50 ms grid for both modalities.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI added several ad hoc handling rules. If a session has no spikes at all, it is dropped. Partial neural coverage is handled by `is_good_trials.shape[1]` plus a spike-time min/max window check. If there is no sample onset before the go cue, the tone is assumed to have started `1.85 s` before the go cue. If tongue tracking is missing entirely, the code fills the session with `NaN` tongue traces, and later those bins become the middle tongue class. Within existing tongue data, low-confidence frames are ignored and bins left `NaN` likewise fall through to the middle class. Electrode-location JSON parse failures become `'unknown'`.

ii.
```python
if len(spike_times_flat) > 0:
    max_spike_time = spike_times_flat.max()
    min_spike_time = spike_times_flat.min()
else:
    f.close()
    return None
```

```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
```

```python
high_conf = lk_slice > 0.1
...
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
valid_mask = ~np.isnan(tongue_y)
```

```python
except (json.JSONDecodeError, AttributeError):
    return 'unknown'
```

iii. The notes justify the neural-coverage rule as a fix for sessions where behavior continued after ephys recording stopped. The trajectory shows the AI discovered the `is_good_trials` size mismatch and then added the min/max spike-time check. The hard-coded tone fallback and the choice to collapse missing tongue bins into the middle class do not receive strong justifications in the notes.

## 10-a. What are the most time-consuming steps of the code?

i. According to the AI's notes, firing-rate computation is the dominant cost, reported as roughly 3 to 14 seconds per session, with data loading around 0.2 to 0.5 seconds and tongue tracking around 0.4 to 0.7 seconds. The full conversion was reported at about 16 minutes for 174 session files.

ii.
```python
t1 = time.time()
print(f"    Data loading: {t1-t0:.1f}s")
...
t2 = time.time()
print(f"    Firing rate computation: {t2-t1:.1f}s")
...
t4 = time.time()
print(f"    Tongue tracking: {t4-t3:.1f}s")
```

```python
for trial_idx in trial_indices:
    ...
    for i, unit_spikes in enumerate(good_spike_times):
        ...
        counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. `CONVERSION_NOTES.md` explicitly reports those timings and attributes most runtime to the firing-rate path. That is consistent with the code's nested trial-by-unit histogram loop.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: the outer trial loop and inner unit loop in `compute_firing_rates_fast`, the per-trial and per-bin loops in `get_tongue_y_for_trials`, the per-trial loops that build inputs and outputs, and the loops over all units/sessions used to compute brain-region lists and region indices.

ii.
```python
for trial_idx in trial_indices:
    ...
    for i, unit_spikes in enumerate(good_spike_times):
        ...
```

```python
for trial_idx in trial_indices:
    ...
    for b in range(n_bins):
        bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

```python
for i in range(n_total):
    start = 0 if i == 0 else int(electrodes_index[i-1])
    elec_idx = electrodes_idx[start]
    region = extract_brain_region(electrode_locations[elec_idx])
    unit_regions.append(region)
```

iii. The notes say the AI already applied some speedups, specifically `searchsorted` and pre-extracting per-unit spike arrays, but they also show that the code still spent most time in the remaining explicit loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats work across passes over the same selected trials: one pass for firing rates, one for inputs, one for tongue extraction, and one for outputs. It also recomputes session-local bin definitions in multiple functions, computes brain regions for every unit before filtering to good units, and later remaps regions again when building `brain_region_idx`.

ii.
```python
firing_rates = compute_firing_rates_fast(
    spike_times_flat, spike_times_index, go_times,
    good_indices, trial_indices
)
...
inputs_list = []
for trial_idx in trial_indices:
    ...
```

```python
if has_tongue:
    tongue_y_trials, all_tongue_y = get_tongue_y_for_trials(
        tongue_data, tongue_timestamps, go_times, trial_indices
    )
...
outputs_list = []
for i, trial_idx in enumerate(trial_indices):
    ...
```

```python
unit_regions = []
for i in range(n_total):
    ...
good_regions = [unit_regions[i] for i in range(n_total) if good_mask[i]]
```

```python
for session_regions in all_brain_region_idx_raw:
    idx = np.array([brain_regions.index(r) for r in session_regions], dtype=np.int64)
    brain_region_idx.append(idx)
```

iii. There is no explicit defense of these repeated passes in the notes. The main justification available is pragmatic: the AI focused on getting a working conversion and only partially optimized the expensive sections.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several quantities that are not part of the saved decoder dataset: session-level control performance statistics (`correct_rate`, `correct_left`, `correct_right`), optional plotting data, and various progress/timing values. It also extracts brain regions for all units before discarding non-good ones, and loads `is_good_trials` only to use its second dimension.

ii.
```python
hits = np.sum(outcome[control_non_early] == b'hit')
misses = np.sum(outcome[control_non_early] == b'miss')
...
correct_rate = hits / (hits + misses)
correct_left = np.sum((outcome == b'hit') & (trial_instruction == b'left') & control_non_early)
correct_right = np.sum((outcome == b'hit') & (trial_instruction == b'right') & control_non_early)
```

```python
if show_processing and session_idx < 2:
    ...
    plt.savefig(f'processing_{sess_name}.png', dpi=100)
```

```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
```

iii. The trajectory shows these extra computations were mainly used for filtering, debugging, plotting, and sanity checks. They are not preserved in the final `neural` / `input` / `output` arrays except indirectly through which sessions were kept.
