# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates subject folders under `/app/data`, then enumerates `.nwb` files within each subject folder. Each file is opened with `pynwb.NWBHDF5IO`, read once, and processed session by session.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. `CONVERSION_NOTES.md` states the source data are NWB files in `/app/data/` with 28 subjects and 174 sessions. The code comments and file traversal show the AI assumed the directory layout itself defines the full dataset.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.description` as the subject identifier for each session, then builds the global `subjects` list in first-seen order from retained sessions only. This means subjects are mouse names like `SC015`, not the numeric NWB `subject_id`.

ii.
```python
subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description
```

```python
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The only explicit justification is the inline code comment identifying `description` as the mouse name. No stronger justification appears in `CONVERSION_NOTES.md`.

## 1-c. How are the data split into sessions?

i. One `.nwb` file is treated as one session. The AI processes every file separately and assigns the session id from `nwb.session_id` when present, otherwise from the filename.

ii.
```python
for fname in files:
    fpath = os.path.join(subj_dir, fname)
    result = process_session(fpath, sample_mode=sample_mode)
```

```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
```

iii. The justification is implicit in the file-by-file traversal and in `CONVERSION_NOTES.md`, which reports counts in subjects and sessions as if each NWB file is one session.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table as the behavioral trial list, then maps neural observation intervals to trial-table indices. If the number of `obs_intervals` matches the number of trials, it assumes a 1:1 mapping; otherwise it matches each observation interval to the nearest `trials['start_time']` within 0.5 s.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
```

```python
obs = units['obs_intervals'][good_indices[0]]
if n_obs == n_trials:
    return list(range(n_trials))
...
trial_starts = np.array([trials['start_time'][i] for i in range(n_trials)])
for i in range(n_obs):
    diffs = np.abs(trial_starts - obs[i, 0])
    min_idx = np.argmin(diffs)
    if diffs[min_idx] < 0.5:
        obs_to_trial.append(int(min_idx))
```

iii. `CONVERSION_NOTES.md` justifies this as handling sessions where recording covered only part of the behavior session: it says the conversion "builds a mapping from obs_intervals indices to trial table indices based on temporal overlap between observation windows and trial events."

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that map to `obs_intervals`, then excludes all early-lick trials and all `ignore` trials. It also drops entire sessions unless they exceed 65% correct on control non-early trials and contain at least 50 correct left and 50 correct right trials.

ii.
```python
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
```

iii. `CONVERSION_NOTES.md` says this followed the paper methodology: exclude early lick and no-response trials, and keep only sessions with `>65%` correct and `>= 50` correct per side.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units['spike_times']` for units whose `classification` is `'good'`. It uses `BehavioralEvents/go_start_times` for the alignment event and `obs_intervals` to decide which trial-specific spikes to consider.

ii.
```python
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)
```

```python
all_spike_times = preload_spike_times(nwb, good_indices)
obs_intervals = units['obs_intervals'][good_indices[0]]
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. `CONVERSION_NOTES.md` explicitly says the QC method is classifier-based `'good'` units and that neural activity is aligned to go cue onset.

## 2-b. How is the `neural` data processed?

i. The AI preloads spike times for all good units, restricts them to the trial's observation interval, and then bins the spikes into 50 ms bins from -2.5 s to +1.5 s around the go cue. Counts are divided by bin width to produce firing rates in Hz. No smoothing or normalization is applied.

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
counts, _ = np.histogram(st_window, bins=bin_edges)
fr[i, :] = counts.astype(np.float32) / bin_width
```

iii. `CONVERSION_NOTES.md` says: "Spike counts are histogrammed into 50ms bins and divided by bin width (0.05s) to get firing rates in Hz."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units['classification'] == 'good'` are kept. Sessions with zero such units are skipped entirely.

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

iii. `CONVERSION_NOTES.md` says this matches the classifier-based QC in the papers.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to that trial's go cue. The AI builds absolute bin edges by adding the go cue time to the fixed window `[-2.5, 1.5]` and histograms spikes against those edges.

ii.
```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

```python
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
```

iii. `CONVERSION_NOTES.md` states "All neural data is aligned to go cue onset (time 0)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 50 ms bins over a 4.0 s window, yielding 80 bins per trial. The AI does not apply any further temporal rebinning.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
```

```python
n_bins = int(round((end_time - begin_time) / bin_width))
```

iii. This is justified directly in both the task instructions and `CONVERSION_NOTES.md`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times`, the per-trial `trials['start_time']`, and the go cue time. It looks for sample-start events occurring between trial start and go cue.

ii.
```python
t_start = trials['start_time'][trial_idx]
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
```

iii. `CONVERSION_NOTES.md` says the tone onset is "the last `sample_start_time` before the go cue." The code adds the extra restriction that the event must also occur after trial start.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI finds the last valid `sample_start_time` before the go cue within the same trial. If none exists, it fabricates a fallback onset at `go_cue - 1.85`. It then converts that onset to go-cue-relative time and subtracts it from the neural bin centers, producing a continuous time-since-tone value for every bin.

ii.
```python
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration
```

```python
tone_onset_rel = tone_onset - go_cue
time_from_tone = bin_centers - tone_onset_rel
```

iii. `CONVERSION_NOTES.md` justifies the main rule as taking the last sample start before go cue. The 1.85 s fallback is only justified in code comments as a "typical sample-delay duration."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI computes `time_from_tone_onset` on the exact same `bin_centers` array returned by the firing-rate computation, so the time input and neural data share identical trial and bin alignment.

ii.
```python
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
...
time_from_tone = bin_centers - tone_onset_rel
```

iii. The justification is implicit in the code structure: the input is derived from the same per-trial go-cue-centered bin grid used for neural activity.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trials-table columns `photostim_onset`, `photostim_duration`, and `start_time`, together with the go cue.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
```

iii. `CONVERSION_NOTES.md` says photostim onset times from NWB are converted to times relative to the go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts the onset and duration to absolute time, shifts them into go-cue-relative time, and marks bins as `1.0` wherever the bin center lies within the stimulation interval. Non-stim trials remain all zeros.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. `CONVERSION_NOTES.md` describes this as a binary time-varying input indicating whether photostimulation is on.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are converted into go-cue-relative time and then compared directly against the same `bin_centers` used for neural activity.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The code makes the alignment explicit by using the neural bin centers themselves for the photostim mask.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not stored directly. The AI derives it from `trials['trial_instruction']` and `trials['outcome']`.

ii.
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
```

```python
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
```

iii. `CONVERSION_NOTES.md` justifies this as: hit means the animal chose the instructed side; miss means it chose the opposite side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI codes hit trials as the instructed side and miss trials as the opposite side, with `left=0` and `right=1`. Because it already removed `ignore` trials, it never creates a no-lick choice category. The per-trial choice is later repeated across all time bins.

ii.
```python
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 0  # shouldn't happen after filtering
```

```python
output_trial[0, :] = base_output[0]
```

iii. `CONVERSION_NOTES.md` gives the hit/miss logic explicitly. The absence of a no-lick category is justified there by the earlier trial filtering that excludes no-response trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `trials['outcome']`.

ii.
```python
outcome = trials['outcome'][trial_idx]
```

iii. The AI treats the trials table as the source of record for outcome categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, stores the per-trial code, and then repeats it across all bins. In practice, because `ignore` trials were filtered out earlier, the retained dataset only contains `miss` and `hit`.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

```python
output_trial[1, :] = base_output[1]
```

iii. `CONVERSION_NOTES.md` says the output values still list all possible outcomes "for completeness" even though no-response trials were excluded.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `trials['early_lick']`.

ii.
```python
trials['early_lick'][trial_idx]
```

iii. The AI uses the trial-table flag as the source of record.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, but because it filters out early-lick trials before conversion, the retained dataset should contain only zeros. The value is later repeated across all bins.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

```python
output_trial[2, :] = base_output[2]
```

iii. `CONVERSION_NOTES.md` explicitly notes that after filtering, the `early_lick` output is always 0.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`. The AI uses column 1 of the tracking data as `y`, column 2 as the tracking likelihood, and the camera timestamps for alignment.

ii.
```python
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. `CONVERSION_NOTES.md` says tongue output comes from DeepLabCut side-camera tracking and uses the tracking likelihood to decide which samples count.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial, the AI extracts tongue frames from a slightly padded window around the neural window, keeps only frames with likelihood `> 0.5`, bins those values into go-cue-relative 50 ms bins, and stores the per-bin mean y value. After all trials in a session are processed, it concatenates those retained-trial bin means, computes session percentiles from the non-NaN values, and discretizes each bin.

ii.
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
...
lh_mask = tw_lh > 0.5
```

```python
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

```python
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]
p40 = np.percentile(valid_tongue, 40)
p60 = np.percentile(valid_tongue, 60)
```

iii. `CONVERSION_NOTES.md` justifies the `> 0.5` likelihood threshold and the 40th/60th percentile discretization, but it does not justify computing percentiles only from retained-trial bin values rather than the full-session trace.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI computes the 40th and 60th percentiles from all non-NaN binned tongue y values collected from retained trials in a session. It then assigns `0` below `p40`, `1` between `p40` and `p60`, and `2` above `p60`. Missing bins (`NaN`) are forced into class `0` rather than a separate missing-data class.

ii.
```python
if len(valid_tongue) > 0:
    p40 = np.percentile(valid_tongue, 40)
    p60 = np.percentile(valid_tongue, 60)
else:
    p40 = 0.0
    p60 = 0.0
```

```python
if np.isnan(val):
    result[i] = 0
elif val < p40:
    result[i] = 0
elif val <= p60:
    result[i] = 1
else:
    result[i] = 2
```

iii. `CONVERSION_NOTES.md` explicitly says time bins without valid tracking data default to 0 ("low").

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to each trial's go cue. It extracts camera frames around `go_cue + [-2.5, 1.5]` (with one-bin padding), builds bin edges on the same go-cue-relative grid as the neural data, and assigns frames to those bins before averaging and discretizing.

ii.
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
```

```python
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

iii. The justification is implicit in the code and reinforced by `CONVERSION_NOTES.md`, which describes tongue y as a time-varying output on the same go-cue-centered trial window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or messy data in several ad hoc ways:
- Sessions with no `'good'` units are skipped.
- Missing or unmatched neural coverage is handled by the nearest-start-time `obs_intervals` mapping and by dropping trials that fail that mapping.
- If no tone onset is found within a trial, the AI fabricates one at `go_cue - 1.85`.
- Missing tongue bins are represented as `NaN` during averaging but are finally recoded as tongue class `0`.
- Missing/empty region annotations become `'unknown'`.

ii.
```python
if len(good_indices) == 0:
    ...
    return None
```

```python
if diffs[min_idx] < 0.5:
    obs_to_trial.append(int(min_idx))
else:
    obs_to_trial.append(-1)
```

```python
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85
```

```python
if np.isnan(val):
    result[i] = 0  # no tongue visible = low position
```

iii. The justifications are scattered between code comments and `CONVERSION_NOTES.md`: the notes justify using only trials with neural coverage and explicitly say missing tongue bins default to 0; the tone fallback is justified only by an inline comment.

## 10-a. What are the most time-consuming steps of the code?

i. The code's main expensive steps are: opening every NWB file, preloading spike times for all good units, repeatedly masking those spike times by `obs_intervals` for every retained trial, histogramming each neuron's spikes trial by trial, and binning tongue frames trial by trial.

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

iii. `CONVERSION_NOTES.md` explicitly highlights spike-time loading as a significant operation, and the nested per-trial/per-neuron loops identify the other hot spots.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized further: the `obs_intervals` to trial matching loop, the per-trial loop over retained trials, the per-neuron loop inside `compute_firing_rates`, and the per-bin tongue averaging loop inside each trial.

ii.
```python
for i in range(n_obs):
    diffs = np.abs(trial_starts - obs[i, 0])
    min_idx = np.argmin(diffs)
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

iii. No explicit efficiency justification is given beyond preloading spike times "for speed." The remaining loops are visible in the code.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several computations unnecessarily: it recomputes the same 80-bin time grid inside `compute_firing_rates` for every trial, rescans every good unit's spike train against the observation interval for every trial, rescans `sample_start_times` per trial, and loops across bins per trial for tongue aggregation.

ii.
```python
def compute_firing_rates(...):
    n_bins = int(round((end_time - begin_time) / bin_width))
    bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

```python
for trial_idx in valid_trial_indices:
    ...
    valid_samples = sample_start_times[
        (sample_start_times >= t_start) & (sample_start_times < go_cue)
    ]
```

iii. There is no explicit justification for these repeated computations beyond clarity and incremental construction of each trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and carries some values that are not used in the final exported dataset: per-session performance (`perf`), `n_good_units`, `n_valid_trials`, and the temporary `all_subject_ids` list. It also spends time computing session-selection statistics whose only purpose is to exclude sessions rather than populate the final data object.

ii.
```python
all_subject_ids = []
```

```python
return {
    ...
    'n_good_units': len(good_indices),
    'n_valid_trials': len(valid_trial_indices),
    'perf': perf,
}
```

```python
control_no_early = 0
correct_control = 0
correct_left = 0
correct_right = 0
...
if perf <= MIN_PERF or correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    ...
    return None
```

iii. The code comments justify the session-selection logic as following the paper. There is no explicit downstream use for the temporary counters or for `all_subject_ids`.
