# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates all subject directories matching `sub-*` under `data`, enumerates every `.nwb` file inside each subject directory as a session, opens each file with `pynwb.NWBHDF5IO`, and converts every discovered session in `main()`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects

def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions

with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()

for subject_idx, subject in enumerate(data['subjects']):
    sessions = sessions_per_subject[subject]
    for session in sessions:
        neural, input, output = convert_session_data(...)
```

iii. `CONVERSION_NOTES.md` does not contain a substantive rationale. The only explicit justification is in code/comments: the file naming convention is assumed, and the script intentionally surveys all sessions before converting them.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level `sub-*` directories, and the stripped suffix becomes the subject id stored in `data['subjects']`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects

data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. No separate justification was documented. The decision is implicit in the directory traversal and NWB filename convention the script is built around.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file within a subject directory is treated as one session, with the session number parsed from the `ses-XX` portion of the filename.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions

sessions_per_subject = {
    subject: get_sessions(subject, datadir=args.datadir)
    for subject in data['subjects']
}
```

iii. No additional rationale was preserved in notes. The code comments only state the expected filename pattern and use that as the session split.

## 1-d. How are the data split into trials?

i. Trials are split by taking every nonzero `trial_start` sample as a trial start and every rising edge of `teleport` as a trial end, then slicing arrays with `np.arange(start, end)`.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx

trial_start_idx, trial_end_idx = get_trial_starts_ends(
    behavior_data['trial_start'].data[:],
    behavior_data['teleport'].data[:],
)
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
```

iii. The inline comment is the main justification: "`trial number` and `trial start` do not agree with each other", so the agent chose `trial_start` plus `teleport` instead.

## 1-e. How are trials filtered based on quality controls?

i. The script attempts to filter short trials with a threshold of `min_ntimepoints=50`, but the actual condition is `idx.sum() < min_ntimepoints` rather than `len(idx) < min_ntimepoints`. In practice this compares the sum of sample indices to 50.

ii.
```python
def convert_session_data(..., min_ntimepoints=50, ...):
    ...
    for trial in range(ntrials):
        idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
        if idx.sum() < min_ntimepoints:
            print(f"Skipping trial {trial} ...")
            istrial[trial] = False
            continue
```

iii. No written justification exists in `CONVERSION_NOTES.md`. The code and warning message show an intended short-trial filter, but the implementation uses `idx.sum()` rather than trial length.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `nwbfile.processing["ophys"]["Deconvolved"].roi_response_series[*].data`.

ii.
```python
deconvolved_data = nwbfile.processing["ophys"]["Deconvolved"]
for plane_data in deconvolved_data.roi_response_series.values():
    spike_data_curr = plane_data.data[:]
```

iii. The code comment says this is the "parse neural activity data" step. No separate external justification was recorded.

## 2-b. How is the `neural` data processed?

i. The agent concatenates all imaging planes after filtering each plane to `iscell == 1`. It does not compute additional transforms beyond this plane merge, and it keeps the native time resolution.

ii.
```python
spike_data = []
nplanes = len(deconvolved_data.roi_response_series)
for plane_data in deconvolved_data.roi_response_series.values():
    roi_data = plane_data.rois[:]
    iscell = np.array(roi_data['iscell'].to_list())[:, 0] == 1
    spike_data_curr = plane_data.data[:]
    spike_data.append(spike_data_curr[:, iscell])
spike_data = np.concatenate(spike_data, axis=1)
neural_time_bin_size = nplanes / plane_data.rate * 1000
```

iii. The only preserved rationale is the structure of the code itself and the comment that neural activity is already in the deconvolved series.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code keeps only ROIs whose `iscell` flag is true and drops all others before concatenation.

ii.
```python
roi_data = plane_data.rois[:]
iscell = np.array(roi_data['iscell'].to_list())[:, 0] == 1
spike_data_curr = plane_data.data[:]
spike_data.append(spike_data_curr[:, iscell])
```

iii. No note-level justification was left; the choice is implied by the explicit `iscell` filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural matrices are aligned to trial start implicitly by using the trial slices defined from `trial_start`/`teleport`; no extra shifting is applied.

ii.
```python
trial_start_idx, trial_end_idx = get_trial_starts_ends(...)
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    spike_data_curr = spike_data[idx]
    neural.append(spike_data_curr.T)
```

iii. The metadata and slicing logic are the only rationale available: the output metadata says `temporal_alignment_event` is `"trial start"`, and the code never applies any other alignment transform.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native session resolution. Neural bin size is computed as `nplanes / plane_data.rate * 1000` ms, behavior bin size is checked from timestamp differences, and no temporal rebinning is applied.

ii.
```python
neural_time_bin_size = nplanes / plane_data.rate * 1000  # ms
behavior_time_bin_size = np.median(np.diff(timestamps)) * 1000  # ms
...
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
assert np.isclose(behavior_time_bin_size, time_bin_size, rtol=1e-2)

values, counts = np.unique(info['neural_time_bin_size'], return_counts=True)
time_bin_size = values[np.argmax(counts)].item()
```

iii. The code comments and assertions show the intent: keep the stored sampling resolution and only verify consistency across streams.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `behavior_data['trial number'].timestamps[:]`, then sliced per trial.

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
...
timestamps_curr = timestamps[idx]
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. No separate documentation exists. The choice is visible in the code and in the repeated timestamp-equality checks against other behavioral streams.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp is subtracted so the series starts at 0 seconds and increases within the trial.

ii.
```python
timestamps_curr = timestamps[idx]
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. The processing is self-documenting in code; no further justification was preserved.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input is aligned by indexing behavioral timestamps and neural samples with the same per-trial `idx` slice. If neural and timestamp lengths disagree at session level, both are cropped to the shared minimum first.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]

idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
timestamps_curr = timestamps[idx]
```

iii. The rationale is only implicit: the code asserts shared sampling and then uses identical indices for both streams.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type comes directly from `behavior_data['environment'].data`.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The only explicit justification is that the code treats this as the VR environment stream and checks its timestamps against the shared behavioral timestamps.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No substantive processing is applied beyond cropping to `T`, trial slicing, and casting to integer.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. No separate rationale was documented.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over trial slices, not from the raw `trial number` values. The raw `trial_start` and `teleport` signals determine the boundaries.

ii.
```python
trial_start_idx, trial_end_idx = get_trial_starts_ends(
    behavior_data['trial_start'].data[:],
    behavior_data['teleport'].data[:],
)
...
input_curr[2] = trial
```

iii. The inline comment gives the rationale: "`trial number` and `trial start` do not agree with each other", so the stored trial-number values were not used for segmentation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code writes the current within-session trial index as a constant value across all timepoints in that trial.

ii.
```python
input_curr[2] = trial
```

iii. No further justification was documented.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the `Reward` time series: reward timestamps are mapped into the session time grid and then queried over the previous trial's indices.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The code comments show the rationale: reward events have their own timestamps, so they are aligned onto the behavioral/neural timeline first.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is assigned 0. For later trials, the code checks whether any reward event occurred in the immediately preceding trial and writes that binary value across the current trial.

ii.
```python
if trial == 0:
    input_curr[3, :] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3, :] = int(isprevreward)
```

iii. No note-level explanation exists beyond the code path itself.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the per-timepoint `position` stream plus a per-trial reward-zone label. That label is inferred from `reward_zone` and `position` during the survey step, then stored in `info['reward_zone_labels']`.

ii.
```python
position = behavior_data['position'].data[:][:T]
reward_zone = info['reward_zone_labels'][subject][trial_offset:trial_offset+ntrials]
...
reward_zone_curr = reward_zone[trial].item()
distance_to_reward_zone = compute_distance_to_reward_zone(position_curr, reward_zone_curr)

reward_zone_trial = reward_zone[idx] > 0
info['reward_zone_positions'][trial] = [
    np.min(pos_trial[reward_zone_trial]),
    np.max(pos_trial[reward_zone_trial]),
]
```

iii. The main explicit rationale is in comments around `segment_reward_positions()`: assign trials to one of the canonical reward zones and encourage stable assignments over time.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The survey step first infers each trial's reward-zone identity with a Viterbi pass over reward-zone positions. During conversion, the code computes signed distance to the nearest reward-zone edge: negative before the zone, zero inside, positive after.

ii.
```python
def compute_distance_to_reward_zone(position_curr, reward_zone_curr):
    reward_zone_range = reward_zone_dict[reward_zone_curr]
    distance_to_reward_zone = np.zeros(ntimepoints)
    idx_before_zone = position_curr < reward_zone_range[0]
    distance_to_reward_zone[idx_before_zone] = position_curr[idx_before_zone] - reward_zone_range[0]
    idx_after_zone = position_curr > reward_zone_range[1]
    distance_to_reward_zone[idx_after_zone] = position_curr[idx_after_zone] - reward_zone_range[1]
    return distance_to_reward_zone
```

iii. The comments in `segment_reward_positions()` provide the only preserved justification: use a Gaussian emission over zone distance and a high stay probability to keep assignments stable.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is digitized into 7 categories using the bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
...
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. No separate written rationale exists, but the bins exactly encode the decoder specification from the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing it from `position_curr`, where `position_curr` uses the same `idx` slice as `spike_data_curr` within each trial.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
distance_to_reward_zone = compute_distance_to_reward_zone(position_curr, reward_zone_curr)
```

iii. The rationale is implicit in the shared indexing and earlier timestamp consistency checks.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position comes directly from `behavior_data['position'].data`.

ii.
```python
position = behavior_data['position'].data[:][:T]
position_curr = position[idx]
```

iii. The code treats position as the native VR corridor position and does not record any further justification.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code slices the raw position per trial and discretizes it; there is no other transform.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. No extra rationale was documented.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is digitized into 5 bins with edges `[-inf, 50, 150, 250, 350, inf]`.

ii.
```python
position_bins = [-np.inf, 50, 150, 250, 350, np.inf]
...
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. No separate rationale was left beyond the explicit bin definition.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by using the same per-trial `idx` slice as the neural array.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. The timestamp assertions and shared indexing are the only available justification.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick comes directly from `behavior_data['lick'].data`.

ii.
```python
licks = behavior_data['lick'].data[:][:T]
licks_curr = licks[idx]
```

iii. No separate note-level rationale was preserved.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are binarized as `1` if the raw value is positive and `0` otherwise.

ii.
```python
output_curr[3] = (licks_curr > 0).astype(int)
```

iii. The code itself is the only explicit rationale.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by using the same per-trial `idx` slice as the neural data after verifying timestamp agreement.

ii.
```python
assert np.allclose(behavior_data['lick'].timestamps[:][:T], timestamps)
...
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
licks_curr = licks[idx]
```

iii. The rationale is implicit in the timestamp assertion and shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the raw `reward_zone` and `position` streams during the survey step, then stored as per-trial labels in `info['reward_zone_labels']`.

ii.
```python
reward_zone = behavior_data['reward_zone'].data[:]
position = behavior_data['position'].data[:]
...
info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]), np.max(pos_trial[reward_zone_trial])]
...
all_info['reward_zone_labels'] = {k: v['zone_labels'] for k, v in segment_results.items()}
...
output_curr[4] = ['A', 'B', 'C'].index(reward_zone_curr)
```

iii. The only explicit rationale is in `segment_reward_positions()`: infer one of the canonical zone labels with a stable-state model.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code estimates reward-zone spans per trial, runs a Viterbi algorithm with Gaussian emissions and a high stay probability to assign each trial to A/B/C, then maps the chosen label to integer classes 0/1/2 during conversion.

ii.
```python
pstay = .9
sigma = 10.0
...
log_emission[valid] = -0.5 * (dist / sigma) ** 2
...
zone_labels = np.array([zone_names[s] for s in states])
...
output_curr[4] = ['A', 'B', 'C'].index(reward_zone_curr)
```

iii. The code comments explicitly justify stable assignments: "use viterbi algorithm with a high cost for switching to encourage stable assignments".

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `behavior_data['Reward'].timestamps[:]`, aligned onto the common time grid as `isreward`.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The only available rationale is the comment that reward events have to be matched to the timeline because they come with their own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to the nearest session time bins, checked for sub-bin alignment error, and each trial is labeled `1` if any reward falls in that trial and `0` otherwise.

ii.
```python
reward_time_indices = np.searchsorted(timestamps, reward_times)
err = timestamps[reward_time_indices] - reward_times
assert np.all(np.abs(err) <= time_bin_size/2)
...
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. No further rationale was documented beyond the alignment assertion and the per-trial `np.any`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several issues defensively: it crops neural and behavioral streams to a shared length when they differ; it has an attempted short-trial filter using `idx.sum() < min_ntimepoints`; it leaves trials with missing reward-zone activation as `NaN` during survey and lets the Viterbi model infer them from neighboring context; and it asserts that reward timestamps align within half a bin.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]

if idx.sum() < min_ntimepoints:
    istrial[trial] = False
    continue

if not np.any(reward_zone_trial):
    continue

log_emission = np.zeros((n_trials, n_states))
...
log_emission[valid] = -0.5 * (dist / sigma) ** 2
...
assert np.all(np.abs(err) <= time_bin_size/2)
```

iii. The script contains no prose justification beyond warnings/comments, but the handling strategy is explicit in these guards and assertions.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is repeated NWB I/O: `survey()` loads every NWB once to collect statistics and reward-zone positions, then conversion loads every NWB again to build the final arrays. Plot saving and final pickle serialization are secondary costs.

ii.
```python
def survey(...):
    for subject in subjects:
        for session in sessions:
            info = survey_session_data(subject, session, datadir=datadir)

def convert_session_data(...):
    io = pynwb.NWBHDF5IO(nwb_file_path, 'r')
    nwbfile = io.read()
    ...

with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. No explicit runtime analysis was documented, so this is inferred from the structure of the code and the duplicated full-dataset reads.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious non-vectorized loops are the per-trial loop in `convert_session_data()`, the per-trial reward-zone-position extraction in `survey_session_data()`, the per-neuron plotting loop under `show_processing`, and the nested Viterbi loops over trials and states in `segment_reward_positions()`.

ii.
```python
for trial in range(ntrials):
    ...

for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    ...

for neuroni in range(nneurons):
    ...

for t in range(1, n_trials):
    for k in range(n_states):
        ...
```

iii. The code does not justify these loops explicitly. They appear to be chosen for simplicity around variable-length trials and the Viterbi recurrence.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats full NWB loading and array extraction between `survey_session_data()` and `convert_session_data()`. Neural, position, lick, speed, and other behavioral streams are read once for survey/diagnostics and then read again for conversion.

ii.
```python
info = survey_session_data(subject, session, datadir=datadir)
...
neural, input, output = convert_session_data(subject, session, info, trial_offset, ...)
```

iii. No explicit rationale was preserved beyond the separation of the survey phase from the final conversion phase.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does extra diagnostic work that is not stored in `converted_data.pkl`: survey-only summary statistics (`unique_*`, `range_*`, `max_spike_rate_range`, `reward_zone_dist`, `reward_zone_states`), reward-zone plots saved in three formats, optional `show_processing` plots, `istrial`, and plot helper calculations such as `min_distance_to_reward_zone`/`max_distance_to_reward_zone`.

ii.
```python
info['max_spike_rate_range'] = (max_spike_rate.min(), max_spike_rate.max())
info['unique_reward_amounts'] = np.unique(reward_amounts)
...
fig.savefig('reward_zone_positions.png')
fig.savefig('reward_zone_positions.pdf')
fig.savefig('reward_zone_positions.svg')
...
all_info['reward_zone_states'] = {k: v['states'] for k, v in segment_results.items()}
all_info['reward_zone_dist'] = {k: v['zone_dist'] for k, v in segment_results.items()}
...
istrial = np.ones(ntrials, dtype=bool)
min_distance_to_reward_zone = np.inf
max_distance_to_reward_zone = -np.inf
```

iii. No explicit justification was documented. These steps appear to exist for inspection, sanity checking, or plotting rather than for the downstream decoder dataset itself.
