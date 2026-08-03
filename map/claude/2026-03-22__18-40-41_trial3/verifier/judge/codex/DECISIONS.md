# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the `data/sub-*` directory tree, sorts subject folders and `.nwb` files within each folder, and then opens each NWB file with `pynwb.NWBHDF5IO`. Within each session it reads `nwb.trials`, `nwb.units`, and `nwb.acquisition['BehavioralEvents']` / `nwb.acquisition['BehavioralTimeSeries']`.

ii. <Code snippets>
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

iii. The notes say the dataset is "174 NWB files, 28 subjects" and describe the NWB layout as one session per file. The trajectory also shows the AI concluding that NWB is the native source and that it should read trials, units, and behavioral time series directly.

## 1-b. How are the data split into subjects?

i. Subjects are identified per session from `nwb.subject.subject_id`, with a filename fallback if the NWB subject field is absent. In the assembled output, `subjects` is built in first-seen order, not sorted order, and `subject_idx` is the index of each session into that list.

ii. <Code snippets>
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

iii. The notes treat the NWB subject field as the mouse identifier and describe the folder structure as `data/sub-XXXXXX/`. No further justification is given for the encounter-order subject list.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session identity comes from `nwb.identifier`, and only sessions that pass later filtering remain in the output.

ii. <Code snippets>
```python
def process_session(nwb_path, show_processing=False, session_idx=0):
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    session_id = nwb.identifier
```

```python
for i, (subject, nwb_path) in enumerate(all_files):
    result = process_session(nwb_path,
                            show_processing=args.show_processing,
                            session_idx=i)
    if result is None:
        n_skipped += 1
        continue
    session_results.append(result)
```

iii. The notes explicitly state "Each NWB file = one behavioral session." The trajectory also treats the 174 NWB files as the initial session list and then filters from there.

## 1-d. How are the data split into trials?

i. Trials come directly from `nwb.trials`. The AI checks that the number of `go_start_times` matches the number of trial rows, derives per-trial quantities from those rows, and then iterates over `valid_indices` so each retained trial becomes one output trial.

ii. <Code snippets>
```python
trials = nwb.trials
n_trials = len(trials)
go_times = be.time_series['go_start_times'].timestamps[:]

assert len(go_times) == n_trials, f"Go times ({len(go_times)}) != trials ({n_trials})"
```

```python
valid_indices = np.where(valid_mask)[0]
...
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    ...
    neural_trials.append(fr)
    input_trials.append(input_data)
    output_trials.append(output_data)
```

iii. The notes describe the trials table as the source of all per-trial behavioral variables. The code comment says the per-trial sample event is found within each trial window, which shows the AI relied on the trials table boundary rather than re-deriving trial boundaries from events.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are not `auto_water`, not `free_water`, have a detected tone onset, and fall within an approximate neural-recording window defined by the last `obs_intervals` stop time of one representative good unit plus a 1 s margin. It does not use the exact `obs_intervals` start-time matching used by the human reference. At the session level it also drops entire sessions that fail behavioral-performance thresholds.

ii. <Code snippets>
```python
behav_valid = (auto_water == 0) & (free_water == 0)
...
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)

for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

```python
obs_intervals = units['obs_intervals'][good_indices_units[0]]
max_recording_time = obs_intervals[-1, 1] if len(obs_intervals) > 0 else 0
```

```python
if correct_rate < MIN_CORRECT_RATE:
    return None

if correct_left < MIN_CORRECT_LEFT or correct_right < MIN_CORRECT_RIGHT:
    return None
```

iii. The notes say trial filtering should "keep all trials EXCEPT auto_water and free_water," but the final code also requires a non-NaN tone onset and recording coverage. The trajectory shows the AI discovered sessions with behavior extending beyond neural recording and decided to filter trials by recording period instead of using the reference's exact `obs_intervals` mapping.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units['spike_times']` for units marked `classification == 'good'`, with `go_start_times` supplying the per-trial alignment event. Region labels are taken from `units['anno_name']`, but those labels are metadata rather than the neural signal itself.

ii. <Code snippets>
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
...
spike_times_all = units['spike_times']
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The notes and trajectory both state that NWB spike times are stored in absolute time and that good units are identified by the `classification` field. The AI's plan was to align those spike times to each trial's go cue.

## 2-b. How is the `neural` data processed?

i. For each retained trial, the AI extracts spikes from each good unit in the window `[go_time + T_START, go_time + T_END)`, bins them into 50 ms bins by flooring relative time to bin index, counts with `np.add.at`, and divides by bin width to convert counts to Hz. There is no smoothing, baseline subtraction, or normalization.

ii. <Code snippets>
```python
def compute_firing_rates_vectorized(spike_times_list, go_time, t_start, t_end, bin_width, n_bins):
    n_neurons = len(spike_times_list)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    ...
    for i, spk in enumerate(spike_times_list):
        mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
        spk_window = spk[mask]
        ...
        bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(fr[i], bin_idx, 1)

    fr /= bin_width
    return fr
```

```python
fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
```

iii. The notes say the AI wanted 50 ms firing rates aligned to go cue, matching the decoder specification, and the trajectory says it concluded spike times must be converted from absolute time by subtracting go-cue time per trial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `classification == 'good'` and `anno_name` is non-empty/non-`None`. If a session has no such units, the whole session is skipped.

ii. <Code snippets>
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

if len(good_indices_units) == 0:
    print(f'  SKIP: no good neurons')
    io.close()
    return None
```

iii. The notes explicitly say the NWB `classification` field is the QC-equivalent signal from the white paper, and they also say `anno_name` is needed for region assignment, which explains the extra non-empty-annotation requirement.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural window is aligned to the go cue. For each trial, the AI uses `go_time` and extracts spikes from `go_time - 2.5 s` to `go_time + 1.5 s`; bin indices are defined relative to `go_time + T_START`.

ii. <Code snippets>
```python
T_START = -2.5
T_END = 1.5
...
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
```

```python
go_time = go_times[trial_idx]
fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
```

iii. The notes say spike times are absolute in NWB and "need to subtract go cue time." The final code performs that subtraction implicitly via the window and bin-index calculation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins, with 80 bins spanning `[-2.5, 1.5]` seconds around go cue. There is no further rebinning after this direct spike binning.

ii. <Code snippets>
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)
```

```python
fr /= bin_width
...
'time_bin_size': BIN_WIDTH * 1000,
```

iii. The notes justify this as following the decoder task specification even though the paper code used 40 ms / 3.4 ms stride parameters.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `trials['start_time']`, and `go_start_times`. For each trial, the AI chooses the last sample-start event between trial start and go cue as that trial's tone onset.

ii. <Code snippets>
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]
...
tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```

iii. The code comment says the last sample is used "in case of replays from early licking." The trajectory shows the AI debated first-versus-last sample start and ultimately the final code uses the last one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes go-cue-relative bin centers, computes the tone onset relative to go cue, and then turns each bin center into seconds since tone onset. The resulting input is a continuous time series, one value per neural bin.

ii. <Code snippets>
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
```

```python
tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time
time_from_tone = bin_centers - tone_relative
```

iii. The notes describe this variable as "current_time - tone_onset" in go-cue-relative coordinates. No extra smoothing or thresholding is applied.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-cue-relative bin centers used for the neural firing rates, so each time value corresponds directly to one neural time bin.

ii. <Code snippets>
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
...
fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
time_from_tone = bin_centers - tone_relative
```

iii. The notes say all decoder variables are aligned to go cue in a shared `[-2.5, 1.5]` window, so the tone-timing input is computed directly on that shared grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `trials['photostim_onset']`, `trials['photostim_duration']`, `trials['start_time']`, and per-trial `go_time`. The code does not use the behavioral-event photostim time series in the final implementation.

ii. <Code snippets>
```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
```

```python
ps_onset = float(photostim_onset[trial_idx])
ps_duration = float(photostim_duration[trial_idx])
ps_onset_abs = trial_starts[trial_idx] + ps_onset
ps_end_abs = ps_onset_abs + ps_duration
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time
```

iii. The notes say photostim should be a binary time series from onset and duration, and the code comments record the AI's conclusion that `photostim_onset` is stored relative to trial start because the values are small.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts onset and duration to a go-cue-relative interval and then marks each bin as 1 if its center falls within `[stim_on, stim_off)`, otherwise 0. Non-stim trials remain all zeros.

ii. <Code snippets>
```python
photostim = np.zeros(N_BINS, dtype=np.float32)
if photostim_onset[trial_idx] != 'N/A':
    ...
    for b in range(N_BINS):
        bc = bin_centers[b]
        if ps_onset_rel <= bc < ps_end_rel:
            photostim[b] = 1.0
```

```python
input_data = np.stack([time_from_tone.astype(np.float32), photostim], axis=0)
```

iii. The notes explicitly describe this variable as a binary on/off time series. The trajectory shows the AI reasoned from the trial-table values rather than from event streams.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation interval is expressed relative to the same go cue used for the neural bins, and bin membership is tested against the same `bin_centers`.

ii. <Code snippets>
```python
ps_onset_rel = ps_onset_abs - go_time
ps_end_rel = ps_end_abs - go_time
...
if ps_onset_rel <= bc < ps_end_rel:
    photostim[b] = 1.0
```

iii. The notes say all streams are mapped into a common go-cue-aligned window, which is exactly what this conversion does.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trials['trial_instruction']` and `trials['outcome']`. Hits map to the instructed side, misses map to the opposite side, and ignores are assigned to the instructed side as a surrogate choice.

ii. <Code snippets>
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

iii. The notes make this explicit: "Choice for ignore trials: Set to instruction direction (the 'correct' choice), since there's no actual lick." That rationale also appears in the mapping table in `CONVERSION_NOTES.md`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as a two-class label, `left = 0` and `right = 1`, then repeats that label across all 80 time bins for the trial.

ii. <Code snippets>
```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, outcome_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_y_trial.astype(np.int64),
], dtype=np.int64)
```

iii. The notes say the decoder spec requires left/right choice, and the AI elected not to create a separate no-lick class for ignore trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `trials['outcome']`.

ii. <Code snippets>
```python
outcomes = trials['outcome'][:]
...
outcome = outcomes[trial_idx]
```

iii. The notes describe outcome as a direct NWB field with the needed categories `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that code across all 80 bins for the trial.

ii. <Code snippets>
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

```python
np.full(N_BINS, outcome_val, dtype=np.int64),
```

iii. The mapping is stated both in the instructions and in the AI's notes, so the code simply encodes the categories into integers.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from `trials['early_lick']`.

ii. <Code snippets>
```python
early_licks = trials['early_lick'][:]
```

```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

iii. The notes treat early lick as a direct trial-table variable and explicitly say these trials are kept because early lick itself is a decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats the code across all 80 bins for the trial.

ii. <Code snippets>
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

```python
np.full(N_BINS, early_val, dtype=np.int64),
```

iii. The notes justify keeping early-lick trials because the decoder task explicitly asks for early lick as an output variable.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The code uses the timestamps, the y coordinate in column 1, and the likelihood in column 2 for percentile estimation.

ii. <Code snippets>
```python
bts = nwb.acquisition['BehavioralTimeSeries']
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
if has_tongue:
    tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
    tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
    tongue_y = tongue_data[:, 1]
    tongue_likelihood = tongue_data[:, 2]
```

iii. The notes say tongue output should use the side-camera tongue tracking y-coordinate and be discretized per session.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first computes session-wide 40th and 60th percentile thresholds from raw tongue-y frames whose likelihood exceeds 0.5, falling back to all `tongue_y` values if there are too few visible frames. Then, for each trial and each 50 ms bin, it finds the single closest camera frame to the bin center and classifies that frame's y value with those thresholds. If tongue tracking is absent, it fills every bin with the middle class.

ii. <Code snippets>
```python
if has_tongue:
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
if has_tongue:
    tongue_y_trial = np.zeros(N_BINS, dtype=np.float32)
    for b in range(N_BINS):
        bc_abs = go_time + bin_centers[b]
        t_idx = np.searchsorted(tongue_ts, bc_abs)
        t_idx = min(t_idx, len(tongue_ts) - 1)
        ty = tongue_y[t_idx]
        if ty < p40:
            tongue_y_trial[b] = 0
        elif ty < p60:
            tongue_y_trial[b] = 1
        else:
            tongue_y_trial[b] = 2
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)
```

iii. The notes justify per-session percentile discretization. They do not justify the nearest-frame simplification explicitly, but the trajectory shows the AI planned to use the tongue tracking stream directly and later treated the low-class skew as expected because the tongue is usually retracted.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three categories only: `0` for values below the 40th percentile, `1` for values between the 40th and 60th percentiles, and `2` for values above the 60th percentile. It does not create a separate "not visible" class.

ii. <Code snippets>
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

iii. The notes describe the target as a 0/1/2 discretization and do not mention any hidden/not-visible class. The final code follows that interpretation literally.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to neural data by taking each go-cue-relative neural bin center, converting it to absolute session time, and using the closest tongue-tracking frame to represent that bin.

ii. <Code snippets>
```python
bc_abs = go_time + bin_centers[b]
t_idx = np.searchsorted(tongue_ts, bc_abs)
t_idx = min(t_idx, len(tongue_ts) - 1)
ty = tongue_y[t_idx]
```

iii. There is no explicit note defending this approximation. It appears to be an implementation simplification chosen after the AI recognized that the camera timestamps and neural timestamps shared the same absolute clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or imperfect data are handled heuristically. Sessions with no good annotated neurons are dropped. Trials with no detectable tone onset or with estimated recording coverage failure are dropped. If tongue data exist but there are too few visible frames, percentiles are computed from all `tongue_y` values; if the tongue stream is absent entirely, all tongue bins are filled with the middle class. Unmapped brain annotations default to `OtherCortex`, and missing NWB subject metadata falls back to the filename prefix.

ii. <Code snippets>
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
good_mask_units = classification == 'good'
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
if len(good_indices_units) == 0:
    return None
```

```python
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)
```

```python
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)
else:
    p40 = np.percentile(tongue_y, 40)
    p60 = np.percentile(tongue_y, 60)
...
else:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)
```

iii. The notes mention several of these choices directly: defaulting rare unmapped annotations to `OtherCortex`, treating a tiny number of zero-neural-data trials as negligible, and using tongue percentiles even when the tongue is usually retracted. For some heuristics, especially the no-tongue fallback, there is no explicit justification beyond making the decoder format complete.

## 10-a. What are the most time-consuming steps of the code?

i. The slowest parts of this implementation are session-by-session NWB reads and the per-trial firing-rate computation, which re-loops over every good neuron for every retained trial. The per-bin photostim and tongue loops add additional cost inside each trial. The notes estimate roughly 5 to 10 seconds per session and around 30 minutes for a full run.

ii. <Code snippets>
```python
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

```python
for i, spk in enumerate(spike_times_list):
    ...
    np.add.at(fr[i], bin_idx, 1)
```

```python
for b in range(N_BINS):
    ...
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. `CONVERSION_NOTES.md` says the runtime is "~5-10s per session, ~30 min total for full conversion." The trajectory also comments on per-session runtimes and extrapolates total runtime from them.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain easy vectorization targets: the per-trial scan over `sample_starts`, the per-trial edits to `regular_mask` and `valid_mask`, the neuron loop inside `compute_firing_rates_vectorized`, the trial loop that recomputes firing rates from scratch, and the per-bin loops for photostim and tongue alignment/discretization.

ii. <Code snippets>
```python
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```

```python
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
    if early_licks[i] == 'early':
        regular_mask[i] = False
```

```python
for trial_idx in valid_indices:
    ...
    for b in range(N_BINS):
        ...
```

iii. The code does not discuss vectorization explicitly, but the repeated Python loops are visible in the implementation. The human reference avoids several of these with flatter session-level array operations.

## 10-c. What processing does the code repeat multiple times?

i. The main repeated work is neural binning: the AI recomputes windowing, bin assignment, and counting for every neuron on every trial instead of computing all trials for a neuron in one pass. It also redoes bin-wise photostim and nearest-frame tongue alignment for every trial independently.

ii. <Code snippets>
```python
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

```python
for i, spk in enumerate(spike_times_list):
    mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
    ...
```

```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
```

iii. There is no explicit justification in the notes for this repetition. It is a consequence of the per-trial structure the AI chose for `process_session`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra work that is not used in the final dataset values: it reads `trial_stops` without later using them, computes `bin_edges_start` and `bin_edges_end` but never uses `bin_edges_end`, and includes optional plotting code and runtime reporting that do not affect the converted dataset. It also computes session-level `correct_rate` mainly for filtering and reporting, not as an output variable.

ii. <Code snippets>
```python
trial_stops = trials['stop_time'][:]
```

```python
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
bin_edges_end = bin_edges_start + bin_width
```

```python
def make_processing_plots(session_data, session_idx, nwb_path):
    ...
```

```python
'correct_rate': correct_rate,
```

iii. The notes emphasize diagnostic plots, runtime summaries, and paper-matching checks, so this extra processing reflects the AI's validation workflow rather than the target dataset format itself.
