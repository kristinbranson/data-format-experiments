# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by listing every `sub-*` directory under `/app/data`, then listing every `.nwb` file inside each subject directory, and calling `process_session(...)` once per file. Trials and units are then read from the opened NWB file inside `process_session`.

ii. 
```python
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
n_trials = len(trials)
units = nwb.units
n_units = len(units)
```

iii. In the trajectory, the AI explicitly decided to use the NWB files under `/app/data/` as the primary source and summarized that choice as: “Data source: NWB files in `/app/data/`.” It first enumerated 28 subject directories and 174 session files, then built the converter around that layout.

## 1-b. How are the data split into subjects (mice)?

i. The AI splits data into subjects using the top-level `sub-*` directories and stores the subject identity for each kept session as `nwb.subject.description` (mouse names such as `SC015`), not `nwb.subject.subject_id`. In the final output, `subjects` preserves first-seen order via `OrderedDict`, and `subject_idx` is the per-session index into that list.

ii. 
```python
subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description
```

```python
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The trajectory shows the AI intentionally chose the mouse name field: “`subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description`.” Earlier inspection steps noted both `subject_id: 440956` and `description: SC015`; the AI chose the description because it matched the mouse naming used in the papers.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. It identifies a session using `nwb.session_id` if present, otherwise the filename. Only sessions that pass later filters are kept in the assembled output.

ii. 
```python
for fname in files:
    fpath = os.path.join(subj_dir, fname)
    print(f"Processing {subj}/{fname}...")

    result = process_session(fpath, sample_mode=sample_mode)
    if result is not None:
        all_sessions.append(result)
```

```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
```

iii. In the trajectory the AI inspected the directory structure, confirmed one NWB file per session, and then structured the converter around one-file-per-session processing.

## 1-d. How are the data split into trials?

i. The AI starts from the NWB trials table, but it does not use all trial rows directly. It builds an `obs_intervals` to trial-table mapping using the first good unit’s observation intervals. If the number of observed intervals equals the number of trials, it assumes one-to-one correspondence by index; otherwise it matches each observed interval to the nearest trial start time within 0.5 s. Only mapped trial indices are considered to have neural data.

ii. 
```python
def build_obs_to_trial_map(nwb, good_indices):
    trials = nwb.trials
    n_trials = len(trials)
    units = nwb.units

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
```

iii. The trajectory shows the AI discovered that some files had more behavioral trials than observed ephys trials, then changed course: “I need to only use trials that are in the obs_intervals.” It justified the mapping by noting that in mismatched files the observed intervals corresponded to a contiguous prefix of the trial table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters. A trial must map to an `obs_intervals` entry so that it has neural data. It then excludes trials with `early_lick != 'no early'` and trials with `outcome == 'ignore'`. It does not explicitly exclude `free_water` trials. A session is dropped if fewer than 2 valid trials remain. Separately, the AI also filters entire sessions by behavioral performance: control, non-early trials must exceed 65% correct, with at least 50 correct left and 50 correct right trials.

ii. 
```python
for i in range(n_trials):
    is_control = trials['photostim_onset'][i] == 'N/A'
    is_no_early = trials['early_lick'][i] == 'no early'
    is_hit = trials['outcome'][i] == 'hit'
    instruction = trials['trial_instruction'][i]
    ...

perf = correct_control / control_no_early
if perf <= MIN_PERF or correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    print(f"  Skipping {os.path.basename(nwb_path)}: perf={perf:.1%}, L={correct_left}, R={correct_right}")
    io.close()
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

if len(valid_trial_indices) < 2:
    ...
    return None
```

iii. The trajectory shows the AI reading the methods text and adopting those paper-level analysis filters: “Trial filtering: Exclude early lick and no-response (ignore) trials per methods” and “Session selection: >65% performance, ≥50 correct lick left and right trials.” It briefly considered keeping all trials because the decoder outputs included `early_lick` and `ignore`, but then reverted to the paper-style exclusion because it wanted to match the paper more closely.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units['spike_times']`, `units['obs_intervals']`, and the go-cue timestamps in `BehavioralEvents/go_start_times`. It also uses `units['classification']` to decide which units contribute.

ii. 
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)
```

```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
...
all_spike_times = preload_spike_times(nwb, good_indices)
obs_intervals = units['obs_intervals'][good_indices[0]]
```

iii. In the trajectory the AI summarized its neural-source decision as: “Spike times: Absolute times, need to align to go cue” and “QC filtering: use `classification == 'good'` from NWB.”

## 2-b. How is the `neural` data processed?

i. The AI preloads each good unit’s spike times, restricts spikes to the trial’s `obs_interval`, then bins those spike times relative to the trial’s go cue with 50 ms bins from -2.5 s to +1.5 s. The output is firing rate in Hz, computed as bin counts divided by bin width. No smoothing, normalization, or baseline subtraction is applied.

ii. 
```python
def get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx):
    spike_times_list = []
    t_start, t_stop = obs_intervals[obs_idx]
    for st in all_spike_times:
        mask = (st >= t_start) & (st < t_stop)
        spike_times_list.append(st[mask])
    return spike_times_list
```

```python
def compute_firing_rates(spike_times_by_neuron, go_cue_time, begin_time, end_time, bin_width):
    n_bins = int(round((end_time - begin_time) / bin_width))
    bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
    ...
    for i, st in enumerate(spike_times_by_neuron):
        mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
        st_window = st[mask]
        if len(st_window) > 0:
            counts, _ = np.histogram(st_window, bins=bin_edges)
            fr[i, :] = counts.astype(np.float32) / bin_width
```

iii. The trajectory states: “Firing rates: Spike counts in 50ms bins (bin width matches task spec).” It chose simple per-trial histogramming after confirming spike times and events were in absolute session time.

## 2-c. How is the `neural` data filtered based on quality controls?

i. At the unit level, only units with `classification == 'good'` are kept. Sessions with zero such units are dropped. At the trial/session level, only trials that map to `obs_intervals` survive, and sessions are also removed by the behavioral performance criteria described in 1-e.

ii. 
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    print(f"  Skipping {os.path.basename(nwb_path)}: 0 good units")
    io.close()
    return None
```

```python
obs_to_trial = build_obs_to_trial_map(nwb, good_indices)
trials_with_neural = set(t for t in obs_to_trial if t >= 0)
```

iii. The AI justified the unit filter by matching the classifier-based QC described in the papers. In the trajectory it repeatedly referred to “good units” from the NWB `classification` column and explicitly skipped sessions with zero good units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to `go_start_times`. For each trial it builds absolute bin edges by adding the go-cue timestamp to the relative window `[-2.5, 1.5]`, then bins spikes against those edges.

ii. 
```python
BIN_WIDTH = 0.05
BEGIN_TIME = -2.5
END_TIME = 1.5
```

```python
go_cue = go_start_times[trial_idx]
...
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
```

```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

iii. The trajectory explicitly says “Align to go cue onset” and records the AI verifying that `go_start_times` had one timestamp per trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins and 80 bins per trial over the 4 s window. There is no additional temporal rebinning beyond this direct binning from spike times.

ii. 
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
```

```python
n_bins = int(round((end_time - begin_time) / bin_width))
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

iii. The AI adopted the temporal resolution directly from the task instructions: “50ms time bins for firing rates.”

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times`, the trials-table `start_time`, and `BehavioralEvents/go_start_times`. The per-trial tone onset is taken as the last sample start between trial start and go cue.

ii. 
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
t_start = trials['start_time'][trial_idx]
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85
```

iii. In the trajectory the AI inspected repeated `sample_start_times` and concluded that the last one before the go cue is the true tone onset because earlier ones are replays caused by early licks.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding the trial’s tone onset time, the AI converts it to go-cue-relative time (`tone_onset_rel = tone_onset - go_cue`) and computes a time-varying input at each neural bin center as seconds since tone onset. If no valid sample start is found, it falls back to `go_cue - 1.85`.

ii. 
```python
tone_onset_rel = tone_onset - go_cue  # relative to go cue (should be ~-1.85)
time_from_tone = bin_centers - tone_onset_rel  # time since tone onset at each bin
```

iii. The trajectory says the AI verified that the last sample start is usually `-1.85 s` relative to the go cue. It added the fallback because it wanted a usable value even when no sample start was found in the current trial bounds.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by using the same `bin_centers` returned from neural firing-rate binning. The continuous time-from-tone vector is computed exactly on the neural time grid.

ii. 
```python
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
...
time_from_tone = bin_centers - tone_onset_rel
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_on], axis=0)
```

iii. The AI’s justification was implicit in the code structure and explicit in its notes: both inputs were constructed after computing neural bin centers so they would live on the same 80-bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trials-table fields `photostim_onset`, `photostim_duration`, and `start_time`, together with `go_start_times` to convert those trial-relative values to go-cue-relative time.

ii. 
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
    ps_start_rel = photostim_start_abs - go_cue
    ps_stop_rel = photostim_stop_abs - go_cue
```

iii. In the trajectory the AI checked both the trials-table photostim fields and the event series, confirmed photostim timing relative to the go cue, and decided to use the per-trial table fields to build the decoder input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts the onset and duration into a binary time series over the 80 bins. A bin is set to 1 if its center lies within the photostim interval, otherwise 0. Trials with `photostim_onset == 'N/A'` stay all zeros.

ii. 
```python
photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
...
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The trajectory shows the AI reasoning that, regardless of exact protocol interpretation, “for the decoder, I just need to mark which time bins have photostim on.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to neural activity by expressing stimulation onset and offset relative to the trial’s go cue and comparing those relative times to the same neural `bin_centers`.

ii. 
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The AI’s trajectory notes that it computed photostim timing “relative to go cue,” so it could place stimulation directly on the decoder time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `trial_instruction` and `outcome`. Because it filtered out `ignore` trials earlier, it only expects `hit` and `miss` at output time.

ii. 
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 0  # shouldn't happen after filtering
```

iii. The trajectory shows the AI reasoning that hit means lick the instructed side and miss means lick the opposite side. It also explicitly struggled with `ignore` trials and ultimately avoided that ambiguity by excluding them.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0` for left and `1` for right. It does not represent a third “no lick” class. The scalar per-trial choice is later repeated across all 80 bins when outputs are combined into a `(4, n_bins)` array.

ii. 
```python
output_trial = np.array([choice, outcome_val, early_lick_val], dtype=np.int64)
```

```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = base_output[0]  # choice (constant across time)
output_trial[1, :] = base_output[1]  # outcome (constant across time)
output_trial[2, :] = base_output[2]  # early_lick (constant across time)
output_trial[3, :] = tongue_y_disc   # tongue y (time-varying)
```

iii. In the trajectory the AI explicitly considered adding a special value for no-lick trials, but after reverting to paper-style trial filtering it kept only a 2-class left/right choice output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials-table `outcome` field.

ii. 
```python
outcome = trials['outcome'][trial_idx]
```

iii. The AI did not need to derive outcome from other variables; it used the trial annotation already present in the NWB file.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings with `{'ignore': 0, 'miss': 1, 'hit': 2}` and stores the resulting per-trial integer. In the final output array that integer is repeated across all 80 bins. Because `ignore` trials are filtered out earlier, only `1` and `2` normally remain.

ii. 
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

```python
output_trial[1, :] = base_output[1]  # outcome (constant across time)
```

iii. The trajectory shows the AI noticing that filtering made `ignore` disappear from the realized output values, but it kept that behavior because it prioritized the paper’s exclusion rules.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` field.

ii. 
```python
trials['early_lick'][trial_idx]
```

iii. The AI treated the NWB trial annotation as authoritative for whether the trial had an early lick.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'no early'` to `0` and `'early'` to `1`, stores the scalar per trial, then repeats it across all 80 bins. Because early-lick trials are excluded before output construction, the realized output is effectively all zeros.

ii. 
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

```python
output_trial[2, :] = base_output[2]  # early_lick (constant across time)
```

iii. The trajectory shows the AI recognizing that this makes `early_lick` trivial, but it accepted that consequence because it chose to preserve the paper’s filtering rule instead of the decoder-oriented output distribution.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The AI uses column 1 as `tongue_y`, column 2 as `tongue_likelihood`, and the matching timestamps to assign frames to bins.

ii. 
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The trajectory shows the AI inspecting the time-series description and confirming the side-camera format `('tongue_x', 'tongue_y', 'tongue_likelihood')`.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each kept trial, the AI extracts tongue frames in an expanded window from `go + BEGIN_TIME - BIN_WIDTH` to `go + END_TIME + BIN_WIDTH`, discards frames with likelihood `<= 0.5`, assigns the remaining frames to bins using `np.digitize`, and takes the mean `tongue_y` per bin. Later, within each kept session, it concatenates all per-bin means from kept trials, removes NaNs, computes the 40th and 60th percentiles of those values, and discretizes each trial’s binned means using those cutoffs.

ii. 
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
...
lh_mask = tw_lh > 0.5
tw_times_good = tw_times[lh_mask]
tw_y_good = tw_y[lh_mask]
```

```python
tongue_y_bins = np.full(len(bin_centers), np.nan, dtype=np.float32)
if len(tw_times_good) > 0:
    abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
    bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
    for b in range(len(bin_centers)):
        b_mask = bin_assignments == b
        if np.any(b_mask):
            tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

```python
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]

if len(valid_tongue) > 0:
    p40 = np.percentile(valid_tongue, 40)
    p60 = np.percentile(valid_tongue, 60)
```

iii. The trajectory shows the AI focusing on side-camera tongue tracking and later trying to “optimize the tongue tracking extraction” by vectorizing frame-to-bin assignment, but the substantive decision remained: threshold by likelihood, average within bins, then percentile-discretize per session.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three categories, not four. Values below the 40th percentile become `0`, values from the 40th to 60th percentile become `1`, and values above the 60th percentile become `2`. Missing bins (`NaN`) are forced into class `0`, effectively treating “not visible” as “low.”

ii. 
```python
def discretize_tongue_y(tongue_y_values, p40, p60):
    result = np.zeros(len(tongue_y_values), dtype=np.int64)
    for i, val in enumerate(tongue_y_values):
        if np.isnan(val):
            result[i] = 0  # no tongue visible = low position
        elif val < p40:
            result[i] = 0
        elif val <= p60:
            result[i] = 1
        else:
            result[i] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

iii. The trajectory directly states this design: “NaN values get class 0 (below 40th percentile - represents no tongue visible).” The AI did not keep a separate “not visible” class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data to neural data by using the same go-cue-relative window and the same 50 ms bin edges, expressed in absolute time for each trial. The discretized tongue classes are then placed into output row 3 across the same 80 bins used for neural firing rates.

ii. 
```python
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

```python
output_trial[3, :] = tongue_y_disc   # tongue y (time-varying)
```

iii. The AI’s alignment rationale was the same as for the other streams: everything is converted onto the go-cue-centered neural bin grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases heuristically. Sessions with zero good units are skipped. Trials without mapped neural observations are dropped via `obs_intervals`. If no sample start is found before a trial’s go cue, it imputes the tone onset as `go_cue - 1.85`. Missing or empty brain-region annotations are replaced with `'unknown'`. If tongue likelihood is too low the frame is discarded, and if a bin ends up with no valid tongue frames its `NaN` is later converted to class `0`. If a session has no valid tongue values, both tongue percentiles default to `0.0`.

ii. 
```python
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration
```

```python
if anno is None or anno == '' or anno == 'nan':
    anno = 'unknown'
```

```python
if len(valid_tongue) > 0:
    p40 = np.percentile(valid_tongue, 40)
    p60 = np.percentile(valid_tongue, 60)
else:
    p40 = 0.0
    p60 = 0.0
```

iii. The trajectory shows the AI adding these fallbacks pragmatically to keep the conversion running even when the raw data did not match its simplifying assumptions.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps in the AI’s code are session-by-session NWB reads, preloading spike times for all good units, the per-trial firing-rate computation over all good units, and per-trial tongue-frame binning. The code itself points to spike-time loading as a notable cost by printing a status line before that step.

ii. 
```python
print(f"  Loading spike times for {len(good_indices)} good units...")
all_spike_times = preload_spike_times(nwb, good_indices)
```

```python
for trial_idx in valid_trial_indices:
    ...
    spike_times_trial = get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx)
    fr, bin_centers = compute_firing_rates(
        spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
    )
```

```python
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The trajectory records the AI optimizing around these areas, especially obs-interval handling and tongue extraction, which indicates that it experienced them as the main bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: scanning units to collect `good_indices`, matching `obs_intervals` to trials, recomputing spike masks for every unit in every trial, histogramming each neuron separately in every trial, per-bin tongue averaging inside each trial, and repeated `brain_regions.index(...)` lookups when building region indices.

ii. 
```python
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)
```

```python
for st in all_spike_times:
    mask = (st >= t_start) & (st < t_stop)
    spike_times_list.append(st[mask])
```

```python
for i, st in enumerate(spike_times_by_neuron):
    ...
    counts, _ = np.histogram(st_window, bins=bin_edges)
```

```python
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The trajectory explicitly mentions attempts to “optimize the tongue tracking extraction” and “vectorize the tongue tracking extraction,” but similar vectorization was not applied to the neural loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations. It repeatedly masks each unit’s spike train by trial interval, then histograms each unit again for each trial. It recomputes bin edges inside `compute_firing_rates` for every trial. It also repeatedly remaps textual brain-region labels to major-region names, rather than caching those mappings once.

ii. 
```python
for trial_idx in valid_trial_indices:
    ...
    spike_times_trial = get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx)
    fr, bin_centers = compute_firing_rates(
        spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
    )
```

```python
def compute_firing_rates(...):
    bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

```python
for s in all_sessions:
    for label in s['brain_region_labels']:
        all_region_labels.add(map_region_to_major(label))
...
idx = np.array([brain_regions.index(map_region_to_major(label))
                for label in s['brain_region_labels']], dtype=np.int64)
```

iii. The trajectory does not state this as an intentional design choice; it appears to be a consequence of implementing the simplest working loops.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and carries some intermediate values that are discarded from the final dataset: `session_id`, `n_good_units`, `n_valid_trials`, and `perf` are returned from `process_session` but not written into the final data structure. It also stores detailed `brain_region_labels` only to immediately remap them to coarser major-region labels later.

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
for s in all_sessions:
    for label in s['brain_region_labels']:
        all_region_labels.add(map_region_to_major(label))
```

iii. The trajectory uses these values for runtime logging and sanity checks, but the final output dictionary does not retain them.
