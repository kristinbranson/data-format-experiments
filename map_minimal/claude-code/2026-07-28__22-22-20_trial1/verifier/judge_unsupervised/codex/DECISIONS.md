# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `/app/data` for directories whose names start with `sub-`, then scans each subject directory for `.nwb` files. Each `.nwb` file is opened with `pynwb.NWBHDF5IO` and processed as one session. Within a session it reads the NWB `trials`, `units`, `BehavioralEvents`, and `BehavioralTimeSeries` tables/streams.

ii.
```python
DATA_DIR = '/app/data'

def convert_data(data_dir, sample_mode=False, max_sessions=None):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    ...
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        ...
        for fname in files:
            fpath = os.path.join(subj_dir, fname)
            result = process_session(fpath, sample_mode=sample_mode)
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()

trials = nwb.trials
units = nwb.units
be = nwb.acquisition['BehavioralEvents']
bts = nwb.acquisition['BehavioralTimeSeries']
```

iii. The trajectory shows the agent first inspected the NWB structure and decided the dataset should be read directly from NWB session files rather than reconstructing from the original MATLAB-export pipeline. The notes summarize this as “Dataset: MAP … Format: NWB … Location: `/app/data/`.”

## 1-b. How are the data split into subjects (mice)?

i. Subjects are discovered from the `sub-*` directory layout, but the subject IDs stored in the final dataset come from `nwb.subject.description` (for example `SC015`) rather than the numeric NWB `subject_id` or the folder name.

ii.
```python
subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description
...
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The trajectory notes that the reference code names sessions by mouse strings like `SC038`, and the live NWB inspection showed `subject.description` contains those `SC...` names while `subject.subject_id` is numeric. The agent therefore chose `description` as the subject label.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Sessions are appended to the output only if `process_session(...)` returns a non-`None` result after session-level filtering.

ii.
```python
files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
...
for fname in files:
    fpath = os.path.join(subj_dir, fname)
    result = process_session(fpath, sample_mode=sample_mode)
    if result is not None:
        all_sessions.append(result)
```

```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
```

iii. The trajectory shows the agent inspected the reference preprocessing, which groups probe files into sessions in the original MATLAB export. In the NWB version there is already one NWB file per session, so the agent used that boundary.

## 1-d. How are the data split into trials?

i. Trials are based on NWB trial-table rows, but only rows that can be matched to neural `obs_intervals` are eligible. When `obs_intervals` and `trials` have equal length, trial `i` is assumed to match observation interval `i`. Otherwise the script matches each observation interval to the nearest trial `start_time` within 0.5 s and keeps only trial rows that appear in that mapping. After that it applies the trial filters.

ii.
```python
def build_obs_to_trial_map(nwb, good_indices):
    trials = nwb.trials
    ...
    obs = units['obs_intervals'][good_indices[0]]
    n_obs = obs.shape[0]

    if n_obs == n_trials:
        return list(range(n_trials))

    trial_starts = np.array([trials['start_time'][i] for i in range(n_trials)])
    obs_to_trial = []
    for i in range(n_obs):
        diffs = np.abs(trial_starts - obs[i, 0])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 0.5:
            obs_to_trial.append(int(min_idx))
        else:
            obs_to_trial.append(-1)
```

```python
obs_to_trial = build_obs_to_trial_map(nwb, good_indices)
trials_with_neural = set(t for t in obs_to_trial if t >= 0)
trial_to_obs = {}
for obs_idx, trial_idx in enumerate(obs_to_trial):
    if trial_idx >= 0:
        trial_to_obs[trial_idx] = obs_idx
```

iii. The agent hit an `IndexError` when it assumed one `obs_interval` per trial, then investigated and concluded that some NWB files contain more trial rows than observation intervals. In the trajectory it justified the heuristic by saying those units were recorded for only a subset of trials and that those intervals map to trial starts.

## 1-e. How are trials filtered based on quality controls?

i. Trial inclusion requires all of the following: the session must pass behavioral selection, the trial must have neural coverage according to the `obs_intervals` mapping, the trial must not be an early-lick trial, and the trial must not be an `ignore`/no-response trial. Sessions with fewer than two remaining trials are skipped.

ii.
```python
for i in range(n_trials):
    is_control = trials['photostim_onset'][i] == 'N/A'
    is_no_early = trials['early_lick'][i] == 'no early'
    is_hit = trials['outcome'][i] == 'hit'
    ...
    if is_control and is_no_early:
        control_no_early += 1
        if is_hit:
            correct_control += 1
            ...

perf = correct_control / control_no_early
if perf <= MIN_PERF or correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
```

```python
valid_trial_indices = []
for i in range(n_trials):
    if i not in trials_with_neural:
        continue
    outcome = trials['outcome'][i]
    early = trials['early_lick'][i]
    if early != 'no early':
        continue
    if outcome == 'ignore':
        continue
    valid_trial_indices.append(i)
```

iii. The justification is explicit in both the notes and the trajectory: the methods excerpt says “Early lick trials and no response trials were excluded for analysis” and session selection used “overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each.” The trajectory also shows the agent noticed this makes `early_lick` trivial and `ignore` absent, but it ultimately reverted to paper-style filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from the NWB `units` table, primarily `spike_times`, `obs_intervals`, `classification`, and `anno_name`. `go_start_times` from `BehavioralEvents` supplies the alignment event.

ii.
```python
units = nwb.units
...
if units['classification'][i] == 'good':
    good_indices.append(i)
...
all_spike_times = preload_spike_times(nwb, good_indices)
obs_intervals = units['obs_intervals'][good_indices[0]]
...
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The agent explicitly justified using `classification == 'good'` from NWB as the classifier-based QC output from the reference pipeline. The reference code excerpt in the trajectory also showed that the original preprocessing works from per-unit spike times and QC-selected units.

## 2-b. How is the `neural` data processed?

i. For each valid trial, the script extracts each good unit’s spikes within that trial’s observation interval, then computes firing rates by histogramming spikes into non-overlapping 50 ms bins over `[-2.5, 1.5]` s around go cue and dividing counts by `0.05` to convert to Hz.

ii.
```python
def compute_firing_rates(spike_times_by_neuron, go_cue_time, begin_time, end_time, bin_width):
    n_bins = int(round((end_time - begin_time) / bin_width))
    bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
    ...
    for i, st in enumerate(spike_times_by_neuron):
        ...
        counts, _ = np.histogram(st_window, bins=bin_edges)
        fr[i, :] = counts.astype(np.float32) / bin_width
```

```python
spike_times_trial = get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx)
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
```

iii. The notes explicitly justify this as “Spike counts are histogrammed into 50ms bins and divided by bin width (0.05s) to get firing rates in Hz,” while acknowledging that this is simpler than the original reference preprocessing, which used `sliding_histogram(...)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is implemented by keeping only units whose NWB `classification` is `'good'`, skipping sessions with zero such units, and ignoring trial rows with no mapped observation interval. The script does not apply any further neuron cleanup such as the reference `check_fr(...)` removal of zero-variance neurons.

ii.
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    print(f"  Skipping {os.path.basename(nwb_path)}: 0 good units")
    return None
```

```python
trials_with_neural = set(t for t in obs_to_trial if t >= 0)
...
if i not in trials_with_neural:
    continue
```

iii. The notes say “QC method: Classifier-based (`'good'` classification in NWB)” and the trajectory ties that to the methods excerpt on classifier-based spike-sorting QC. No separate justification is given for omitting `check_fr(...)`; that omission is implicit.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `go_start_times`. The code uses each trial’s go-cue timestamp as time zero and builds the bin edges/centers relative to that event.

ii.
```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
...
go_cue = go_start_times[trial_idx]
...
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

iii. The task instructions explicitly requested go-cue alignment, and the agent repeated that decision in the script header and notes (“Align to go cue onset”).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins. There is no separate rebinning stage; firing rates are computed directly in those bins.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
...
n_bins = int(round((end_time - begin_time) / bin_width))
...
'time_bin_size': BIN_WIDTH * 1000,
'n_bins': n_bins,
'bin_width_s': BIN_WIDTH,
```

iii. The justification is direct from the task instructions and repeated in the notes table (“Bin width: 50 ms”).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents.sample_start_times`, the NWB `trials['start_time']`, and `BehavioralEvents.go_start_times`.

ii.
```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
t_start = trials['start_time'][trial_idx]
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
```

iii. The trajectory shows the agent inspected `sample_start_times`, noticed there can be multiple sample starts because early licks trigger replay, and decided to identify the actual tone onset from those timestamps.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The script picks the last `sample_start_time` that falls between trial start and go cue, treats that as the true tone onset for the trial, converts it to go-cue-relative time, and subtracts it from the bin centers so each bin stores continuous elapsed time since tone onset. If no such sample start exists, it falls back to `go_cue - 1.85`.

ii.
```python
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]  # last sample start = actual tone onset
else:
    tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration

tone_onset_rel = tone_onset - go_cue
time_from_tone = bin_centers - tone_onset_rel
```

iii. The trajectory gives the justification explicitly: “The last sample start before the go cue is always -1.85s relative to go cue for the main sample epoch. Earlier ones are replays due to early licking.” The hard-coded fallback is not separately justified beyond that typical timing claim.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same per-trial `bin_centers` that were generated for the neural firing-rate bins, so the time-from-tone vector has exactly the same length and alignment as the neural matrix.

ii.
```python
fr, bin_centers = compute_firing_rates(...)
...
time_from_tone = bin_centers - tone_onset_rel
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_on], axis=0)
```

iii. The justification is implicit in the implementation: the same `bin_centers` array is reused for both neural and input streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial table’s `photostim_onset` and `photostim_duration`, plus the trial `start_time` and trial-specific `go_start_times`. The script does not use the separate `BehavioralEvents.photostim_start_times` and `photostim_stop_times` streams in the final implementation.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
```

iii. The trajectory shows the agent inspected both trial-table photostim fields and the event streams. It ultimately used the trial-table onset/duration fields because those already encode the per-trial perturbation timing relative to trial start, analogous to the reference `task_stimulation` array.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It creates an all-zero vector for the 80 bins, converts trial-relative photostim onset/duration to absolute session time, converts those to go-cue-relative start/stop times, and marks bins inside that interval as `1.0`.

ii.
```python
photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
...
photostim_start_abs = t_start + onset_val
photostim_stop_abs = photostim_start_abs + dur_val
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The notes describe this as a binary on/off input and explicitly say the code converts photostim onset times from absolute times into the go-cue-relative window.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by applying the on/off mask to the same `bin_centers` used for the neural firing-rate bins, after converting the stimulation interval into go-cue-relative coordinates.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The justification is implicit in the implementation and is consistent with the trajectory’s observation that the reference code subtracts go-cue time from stimulation times.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The final code derives choice from `trials['trial_instruction']` and `trials['outcome']`. It does not use the raw lick-event streams (`left_lick_times`, `right_lick_times`) or the original reference variable `behavior_lick_directions`.

ii.
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 0
```

iii. The notes justify this explicitly: “Hit trials: choice = instruction side … Miss trials: choice = opposite of instruction side.” The trajectory shows the agent considered ignore-trial ambiguity and then relied on paper-style filtering so only hit/miss trials remain.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as `0` for left and `1` for right. The code infers the chosen side: for `hit` trials the animal is assumed to have chosen the instructed side; for `miss` trials it is assumed to have chosen the opposite side. For `ignore` it would default to `0`, but those trials are filtered out before output creation.

ii.
```python
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0  # miss means licked wrong side
else:
    choice = 0  # shouldn't happen after filtering
```

iii. The trajectory shows the agent explicitly reasoned through the ambiguity of ignore trials and reverted to excluding them, which made this inference rule viable for the remaining trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trial-table string field `trials['outcome']`.

ii.
```python
outcome = trials['outcome'][trial_idx]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. No separate justification was needed beyond matching the decoder output specification and the NWB trial labels the agent inspected.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped to integers `ignore=0`, `miss=1`, `hit=2`. Because of the earlier trial filter, the converted dataset only contains `miss` and `hit` values in practice.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. The trajectory shows the agent explicitly noticed that `ignore` would disappear after filtering, but kept the full label mapping anyway because the paper-style trial exclusion remained in place.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the trial-table field `trials['early_lick']`.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. No separate justification is recorded beyond the NWB trial schema and the decoder spec.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `'no early'` to `0` and any other value to `1`. Because early-lick trials are filtered out beforehand, the saved dataset contains only `0`.

ii.
```python
if early != 'no early':
    continue
...
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. The trajectory shows the agent recognized this would make the variable trivial, but chose to preserve the paper’s trial filtering rather than the decoder task’s desire for early-lick prediction.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries['Camera0_side_TongueTracking']`: specifically the y-position column (`data[:, 1]`), the likelihood column (`data[:, 2]`), and that stream’s timestamps.

ii.
```python
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The trajectory explicitly records the agent inspecting the side-camera tongue stream and concluding that it contains `(tongue_x, tongue_y, tongue_likelihood)`.

## 8-b. How is `output` *Tongue y-position* processed?

i. For each kept trial, the script extracts side-camera tongue samples in a window slightly larger than the neural window, filters those samples to likelihood `> 0.5`, assigns the remaining video timestamps to the neural bins, and averages the y-values within each 50 ms bin. Those continuous binned values are stored temporarily and discretized later.

ii.
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
...
lh_mask = tw_lh > 0.5
tw_times_good = tw_times[lh_mask]
tw_y_good = tw_y[lh_mask]

tongue_y_bins = np.full(len(bin_centers), np.nan, dtype=np.float32)
...
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The notes justify the source stream and say “Only positions with tracking likelihood > 0.5 are used.” No trajectory evidence shows this threshold coming from the reference code; it appears to be the agent’s own processing choice.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After all trials in a session are processed, the code concatenates the binned tongue-y values from that session, drops `NaN`s, computes the 40th and 60th percentiles, and discretizes each bin: `< p40 -> 0`, `p40..p60 -> 1`, `> p60 -> 2`. `NaN` bins are forced to class `0`.

ii.
```python
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]

if len(valid_tongue) > 0:
    p40 = np.percentile(valid_tongue, 40)
    p60 = np.percentile(valid_tongue, 60)
...
def discretize_tongue_y(tongue_y_values, p40, p60):
    result = np.zeros(len(tongue_y_values), dtype=np.int64)
    for i, val in enumerate(tongue_y_values):
        if np.isnan(val):
            result[i] = 0
        elif val < p40:
            result[i] = 0
        elif val <= p60:
            result[i] = 1
        else:
            result[i] = 2
```

iii. The notes justify the percentile thresholds and explicitly state the extra missing-data rule: “Time bins without valid tracking data default to 0 (low).” The trajectory does not show any reference-based justification for computing the percentiles only over retained, binned trial windows rather than the full session stream.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned by assigning video timestamps to the same go-cue-centered 50 ms bins used for the neural data. The final discretized tongue output is time-varying with exactly one category per neural bin.

ii.
```python
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
...
output_trial = np.zeros((4, n_bins), dtype=np.int64)
...
output_trial[3, :] = tongue_y_disc
```

iii. The justification is implicit in the implementation and consistent with the overall go-cue alignment policy.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses a collection of ad hoc fallbacks. Missing or empty spike windows yield zeros. Sessions with zero good units, no eligible control trials, or fewer than two valid trials are skipped. Missing region labels become `'unknown'`. If no sample-start time is found before go cue, tone onset defaults to `go_cue - 1.85`. Missing tongue bins remain `NaN` until discretization, then become class `0`. Observation intervals that cannot be matched within 0.5 s become `-1` and are dropped from trial consideration.

ii.
```python
if len(good_indices) == 0:
    return None
...
if control_no_early == 0:
    return None
...
if len(valid_trial_indices) < 2:
    return None
```

```python
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85
```

```python
if anno is None or anno == '' or anno == 'nan':
    anno = 'unknown'
...
if np.isnan(val):
    result[i] = 0
```

iii. The notes explicitly mention one class of residual problem: many all-zero neural trials remained because the `obs_intervals` overlap little with the firing window, and the agent chose to retain them. The other fallbacks are implicit in the code rather than justified from the reference.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive operations are: loading large NWB files; preloading all good-unit spike times; per-trial/per-neuron histogramming in `compute_firing_rates(...)`; and per-trial tongue-video binning, especially because it loops over every neural bin after filtering high-frame-rate video timestamps.

ii.
```python
all_spike_times = preload_spike_times(nwb, good_indices)
...
for i, st in enumerate(spike_times_by_neuron):
    ...
    counts, _ = np.histogram(st_window, bins=bin_edges)
```

```python
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
...
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The trajectory confirms that the agent added spike-time preloading “for speed,” implying it had already identified spike extraction/histogramming as a bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization targets: the per-neuron loop in `compute_firing_rates(...)`; the per-unit loop in `get_spike_times_for_trial_from_cache(...)`; the per-bin loop used to average tongue-y values; the repeated list-building loops for good-unit indices and brain-region mappings; and the subject/session aggregation loops could at least be streamlined.

ii.
```python
for i, st in enumerate(spike_times_by_neuron):
    ...
```

```python
for st in all_spike_times:
    mask = (st >= t_start) & (st < t_stop)
    spike_times_list.append(st[mask])
```

```python
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. No explicit justification was given for these loops; they are straightforward consequences of the implementation style.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly rebuilds bin edges/centers on every trial; filters spikes twice per trial (first by observation interval, then again by the neural analysis window); remaps detailed region names through `map_region_to_major(...)` more than once; and scans the tongue video stream separately for every retained trial.

ii.
```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
...
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
```

```python
mask = (st >= t_start) & (st < t_stop)
spike_times_list.append(st[mask])
...
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
st_window = st[mask]
```

```python
all_region_labels.add(map_region_to_major(label))
...
idx = np.array([brain_regions.index(map_region_to_major(label))
                for label in s['brain_region_labels']], dtype=np.int64)
```

iii. The duplication is not justified in the notes; it follows from composing separate helper functions without caching intermediate products beyond raw spike times.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and stores several intermediate values that are not preserved in the final dataset (`session_id`, `n_good_units`, `n_valid_trials`, `perf`). It also computes continuous tongue-y traces only to discard them after discretization, and it expands the per-trial outputs `choice`, `outcome`, and `early_lick` into 80-bin constant time series even though those variables are trial-constant.

ii.
```python
return {
    'neural': neural_list,
    'input': input_list,
    'output': output_list,
    'brain_region_labels': brain_region_labels,
    'subject_id': subject_id,
    'session_id': session_id,
    'tongue_y_trial_values': tongue_y_trial_values,
    'n_good_units': len(good_indices),
    'n_valid_trials': len(valid_trial_indices),
    'perf': perf,
}
```

```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = base_output[0]
output_trial[1, :] = base_output[1]
output_trial[2, :] = base_output[2]
output_trial[3, :] = tongue_y_disc
```

iii. The only explicit rationale is a code comment: the agent says it made all outputs time-varying so they could be combined into one array. The other discarded intermediates are not justified beyond debugging/statistics convenience.
