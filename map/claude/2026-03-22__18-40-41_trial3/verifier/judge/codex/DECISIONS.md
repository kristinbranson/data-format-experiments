# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data` by subject directory, collects every `.nwb` file, then opens each file with `pynwb.NWBHDF5IO`. Within each file it reads `nwb.trials`, `nwb.units`, and `nwb.acquisition['BehavioralEvents']`.

ii. 
```python
def list_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sub-')])
    all_files = []
    for sub in subjects:
        sub_dir = os.path.join(data_dir, sub)
        nwb_files = sorted([os.path.join(sub_dir, f)
                           for f in os.listdir(sub_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append((sub, nwb_file))
    return all_files
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
trials = nwb.trials
units = nwb.units
be = nwb.acquisition['BehavioralEvents']
```

iii. The notes justify this by saying the dataset is organized as `data/sub-XXXXXX/`, each NWB file is one behavioral session, and NWB is the published storage format. The trajectory also shows the agent exploring the subject-directory layout before writing the loader.

## 1-b. How are the data split into subjects?

i. The AI treats the subject directory and `nwb.subject.subject_id` as the mouse identity. In the final dataset it builds `subjects` in first-seen order and `subject_idx` by indexing into that list.

ii. 
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
subjects = []
subject_idx = []

for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes say the NWB files are already grouped by `sub-XXXXXX` folders and that there are 28 subjects, so no extra inference is needed.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session identity comes from `nwb.identifier`, and the session ordering follows the sorted per-subject file listing.

ii. 
```python
session_id = nwb.identifier
```

```python
for i, (subject, nwb_path) in enumerate(all_files):
    result = process_session(nwb_path,
                            show_processing=args.show_processing,
                            session_idx=i)
```

iii. The notes state that “Each NWB file = one behavioral session,” so the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table directly, with one row per trial, and checks that the number of `go_start_times` matches the number of trial rows.

ii. 
```python
trials = nwb.trials
n_trials = len(trials)
go_times = be.time_series['go_start_times'].timestamps[:]

assert len(go_times) == n_trials, f"Go times ({len(go_times)}) != trials ({n_trials})"
```

iii. The notes describe the `trials` table as the canonical trial structure and describe the behavioral event streams as auxiliary timing information used within those trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI's stated plan in `CONVERSION_NOTES.md` was to keep all trials except `auto_water` and `free_water`, while keeping early-lick, ignore, and stimulation trials because the decoder needs them. The implemented code is stricter: it removes `auto_water`, `free_water`, trials with no detected tone onset, and trials whose `go_time + 1.5 s` exceeds `max_recording_time + 1.0`. It also drops entire sessions if behavioral performance is below 65% or either side has fewer than 50 hit trials.

ii. 
```python
behav_valid = (auto_water == 0) & (free_water == 0)
```

```python
if correct_rate < MIN_CORRECT_RATE:
    return None

if correct_left < MIN_CORRECT_LEFT or correct_right < MIN_CORRECT_RIGHT:
    return None
```

```python
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)

for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

iii. The justification in the notes is that the paper used session-selection criteria of `>65%` correct and at least `50` correct left and right trials, and that early-lick/ignore/stim trials should be retained for the decoder. The agent also says it used a recording-coverage filter to avoid neural windows extending past the recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `units['spike_times']` for quality-filtered units, together with `BehavioralEvents/go_start_times` for alignment.

ii. 
```python
units = nwb.units
classification = units['classification'][:]
spike_times_all = units['spike_times']
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
good_indices = good_indices_units
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

iii. The notes explicitly map `units.spike_times -> neural` and say NWB stores absolute spike times that must be aligned to the go cue.

## 2-b. How is the `neural` data processed?

i. For each valid trial, the AI bins absolute spike times into 80 non-overlapping 50 ms bins from `-2.5 s` to `+1.5 s` around the go cue, counts spikes per bin, and divides by bin width to obtain firing rates in Hz. There is no smoothing or normalization.

ii. 
```python
def compute_firing_rates_vectorized(spike_times_list, go_time, t_start, t_end, bin_width, n_bins):
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    ...
    mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
    spk_window = spk[mask]
    ...
    bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
    np.add.at(fr[i], bin_idx, 1)
    fr /= bin_width
    return fr
```

```python
fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
```

iii. The notes justify 50 ms bins from the decoder specification, and say the original code also computed firing rates from spike times.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'` and also requires `anno_name` to be non-empty. If no such units remain, it drops the session.

ii. 
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

if len(good_indices_units) == 0:
    return None
```

iii. The notes say `classification == 'good'` is the NWB equivalent of the QC classifier and also argue that neurons should have histology/CCF annotation, hence the extra `anno_name` requirement.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to its go cue. The AI uses absolute spike times and absolute go-cue times, then bins spikes in the window `[go_time - 2.5 s, go_time + 1.5 s)`.

ii. 
```python
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
```

iii. The notes say NWB stores absolute spike times, so the conversion must subtract the go-cue timing implicitly through the binning window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins and exactly 80 bins per trial over the 4 s go-cue-centered window. There is no additional temporal rebinning after this first binning step.

ii. 
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. The notes explicitly justify the 50 ms choice as coming from the decoder task, even though the reference analysis code used 40 ms windows.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `trials['start_time']`, and `go_start_times`. For each trial, the AI finds the last sample-start event between trial start and the go cue.

ii. 
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]
tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```

iii. The notes justify using the sample event as tone onset and taking the last occurrence because early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes bin centers relative to the go cue, converts the selected tone onset into go-relative time, and then subtracts that tone time from each bin center to get seconds since tone onset.

ii. 
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time
time_from_tone = bin_centers - tone_relative
```

iii. The notes say this input should be a continuous time-varying signal equal to `current_time - tone_onset_time`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same 80 go-cue-centered bin centers as the neural data, so one value is produced per neural time bin.

ii. 
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
```

```python
fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
time_from_tone = bin_centers - tone_relative
```

iii. The notes explicitly describe `time_from_tone_onset` as a per-bin quantity on the go-cue-relative trial timeline.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `trials['photostim_onset']`, `trials['photostim_duration']`, `trials['start_time']`, and the per-trial go cue.

ii. 
```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
trial_starts = trials['start_time'][:]
```

```python
ps_onset = float(photostim_onset[trial_idx])
ps_duration = float(photostim_duration[trial_idx])
ps_onset_abs = trial_starts[trial_idx] + ps_onset
ps_end_abs = ps_onset_abs + ps_duration
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time
```

iii. The notes justify this by stating that the trial-table onset values are relative to trial start and must be put onto the go-cue-centered axis.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI turns each trial into a binary 80-bin time series. Bins whose centers fall within the stimulation interval are set to `1.0`; all others remain `0.0`.

ii. 
```python
photostim = np.zeros(N_BINS, dtype=np.float32)
if photostim_onset[trial_idx] != 'N/A':
    ...
    for b in range(N_BINS):
        bc = bin_centers[b]
        if ps_onset_rel <= bc < ps_end_rel:
            photostim[b] = 1.0
```

iii. The notes explicitly say the decoder input should be a binary time series indicating whether photostimulation is active at each time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset and offset are converted from trial-start-relative time to go-cue-relative time, then compared against the same bin centers used for neural firing rates.

ii. 
```python
ps_onset_abs = trial_starts[trial_idx] + ps_onset
ps_end_abs = ps_onset_abs + ps_duration
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time
```

iii. The notes say photostimulation should be represented on the same trial-aligned time axis as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `trials['trial_instruction']` and `trials['outcome']`. `hit` means the instructed side, `miss` means the opposite side, and `ignore` is assigned back to the instructed side rather than a separate no-lick class.

ii. 
```python
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1
```

iii. The notes explicitly justify this: “Choice for ignore trials: Set to instruction direction (the ‘correct’ choice), since there’s no actual lick.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI codes choice as binary `0/1` for left/right and repeats that per-trial value across all 80 bins. It does not create a third no-lick class.

ii. 
```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_trial.astype(np.int64),
], dtype=np.int64)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

iii. The notes justify the binary coding by treating ignore trials as instructed-side trials rather than “no lick.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` column.

ii. 
```python
outcomes = trials['outcome'][:]
```

```python
outcome = outcomes[trial_idx]
```

iii. The notes list the NWB `outcome` field as already containing `hit`, `miss`, and `ignore`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the resulting per-trial code across all 80 bins.

ii. 
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

```python
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The notes state that the decoder output specification directly requires these three categorical labels.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` column.

ii. 
```python
early_licks = trials['early_lick'][:]
```

iii. The notes describe `early_lick` as a direct NWB trial variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats the per-trial code across the 80 bins.

ii. 
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

```python
np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The notes justify keeping early-lick trials because early lick is itself a required decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: the `y` coordinate in column 1, with the tracking likelihood in column 2 used only when computing session thresholds.

ii. 
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The notes justify this as the side-camera tongue tracking stream described in the methods and dataset exploration.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI first computes session-level 40th and 60th percentiles from all tongue frames with likelihood `> 0.5` (or from all frames if fewer than 100 visible frames exist). For each trial/bin, it then finds the single nearest camera frame to that bin center and classifies that frame’s `y` value; it does not average all visible frames in the bin.

ii. 
```python
tongue_visible = tongue_likelihood > 0.5
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)
else:
    p40 = np.percentile(tongue_y, 40)
    p60 = np.percentile(tongue_y, 60)
```

```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
```

iii. The notes justify the percentiles as a per-session discretization and say they should be computed over “ALL valid tongue positions in the session.” There is no explicit justification in the notes for using the nearest frame rather than a bin average; that choice is only evident in the code.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI creates only three categories: `< p40 -> 0`, `p40 to p60 -> 1`, and `>= p60 -> 2`. It does not implement the requested fourth “not visible” category.

ii. 
```python
if ty < p40:
    tongue_y_trial[b] = 0
elif ty < p60:
    tongue_y_trial[b] = 1
else:
    tongue_y_trial[b] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

iii. The notes only justify the 40th/60th percentile split. They do not justify omitting the instructed “not visible” class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to the same go-cue-centered 50 ms grid as neural data, but it does so by sampling the nearest video frame to each bin center rather than pooling all frames inside the bin.

ii. 
```python
bc_abs = go_time + bin_centers[b]
t_idx = np.searchsorted(tongue_ts, bc_abs)
t_idx = min(t_idx, len(tongue_ts) - 1)
ty = tongue_y[t_idx]
```

iii. The notes say only that tongue should be a time-varying output on the same per-trial timeline. The specific nearest-frame alignment rule comes from the code rather than an explicit written justification.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases by exclusion or fallback. Missing tone onset causes the trial to be dropped. Sessions with no good, annotated neurons are dropped. Unmapped anatomy labels are forced into `OtherCortex`. If the tongue stream is absent, the code fills all tongue bins with the middle category. It does not create a “not visible” tongue class for low-likelihood frames.

ii. 
```python
valid_mask &= ~np.isnan(tone_onset_per_trial)
```

```python
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
if len(good_indices_units) == 0:
    return None
```

```python
print(f'  WARNING: Unmapped annotation: "{anno_name}"')
return 'OtherCortex'
```

```python
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)  # default to middle
```

iii. The notes justify dropping trials without required timing information and sessions without usable neurons. They also explicitly say unmapped annotations default to `OtherCortex` and describe this as an acceptable fallback.

## 10-a. What are the most time-consuming steps of the code?

i. The AI did not add an explicit profiler, but its notes estimate roughly `5-10 s` per session and about `30 min` for the full run. From the code structure, the main expensive steps are the per-trial firing-rate computation over all neurons and the per-trial/per-bin tongue-frame lookup. The full conversion output shows some large sessions taking tens of seconds each.

ii. 
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
```

iii. The notes’ runtime estimates are the only explicit justification. The trajectory shows no profiling step beyond observing wall-clock runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar that could have been vectorized: per-trial tone matching, per-trial edits to `regular_mask`, per-trial recording-coverage filtering, the per-neuron loop inside `compute_firing_rates_vectorized`, the per-bin photostim loop, and the per-bin tongue lookup loop.

ii. 
```python
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
```

```python
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
    if early_licks[i] == 'early':
        regular_mask[i] = False
```

```python
for i, spk in enumerate(spike_times_list):
    ...
    np.add.at(fr[i], bin_idx, 1)
```

iii. The AI does not explicitly justify leaving these loops unvectorized; the code simply uses straightforward per-trial and per-bin logic.

## 10-c. What processing does the code repeat multiple times?

i. The biggest repeated work is recomputing firing rates from raw spike times independently for every valid trial, even though the same neurons are scanned again each time. The code also repeats `np.searchsorted` once per tongue bin per trial and rebuilds per-session subject indices with repeated list searches.

ii. 
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

```python
for b in range(N_BINS):
    t_idx = np.searchsorted(tongue_ts, bc_abs)
```

```python
if sess['subject_id'] not in subjects:
    subjects.append(sess['subject_id'])
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. There is no explicit written justification for this repeated work in the notes; it is an implementation property of the code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints session-level behavioral performance metrics purely for filtering and reporting, loads `trial_stops` but never uses them, computes `bin_edges_start`/`bin_edges_end` without using them, includes an optional plotting pipeline, and builds some summary statistics that are only printed. These do not become decoder inputs or outputs.

ii. 
```python
trial_stops = trials['stop_time'][:]
```

```python
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
bin_edges_end = bin_edges_start + bin_width
```

```python
if args.show_processing and len(session_results) <= 2:
    make_processing_plots(result, len(session_results) - 1, nwb_path)
```

```python
correct_rate = correct_regular / n_regular
...
print(f'  Output distributions:')
```

iii. The notes justify some of this as sanity checking and documentation, but there is no claim that these quantities are needed by downstream analyses.
