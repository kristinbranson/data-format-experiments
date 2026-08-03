# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed every NWB file under `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`, then processed each file as one session with `h5py`. Within each file it loaded the behavior and ophys groups directly from HDF5.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, show_processing=args.show_processing, session_idx=i)
```

```python
f = h5py.File(nwb_path, 'r')
beh = f['processing']['behavior']['BehavioralTimeSeries']
deconv_group = f['processing']['ophys']['Deconvolved']
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI justified this as “Loads NWB files with h5py.” The trajectory shows it explored the NWB layout and chose direct HDF5 access rather than `pynwb`.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB file’s metadata, then uniqued and sorted at the end to form `subjects`. Session-to-subject mapping is stored in `subject_idx`.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
...
all_subj_ids.append(result['subj_id'])
...
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
subject_idx = np.array([unique_subjects.index(s) for s in all_subj_ids])
```

iii. Step 2 of `CONVERSION_NOTES.md` says the NWB files are organized by `sub-{id}` and reports 11 subjects. The trajectory shows the AI verified the directory structure and then trusted NWB metadata for subject IDs.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID is read from the NWB metadata, but the actual splitting is file-based.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
sess_id = f['general']['session_id'][()].decode()
```

iii. The AI’s notes consistently describe one NWB file per session, and the trajectory shows it counted 152 NWB files and equated those with 152 sessions.

## 1-d. How are the data split into trials?

i. Trial starts are all indices where `trial_start > 0`. Trial ends are all indices where `teleport > 0`. Trials are the slices `[trial_start_inds[t]:teleport_inds[t]]`. If the counts do not match, the AI truncates both arrays to the shorter length.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
...
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
...
t_start = trial_start_inds[t]
t_end = teleport_inds[t]
```

iii. `CONVERSION_NOTES.md` Step 1 says “Trial boundaries: trial_start_inds and teleport_inds.” The trajectory also says the AI used `trial_start` and `teleport` after finding that other trial-number signals were less reliable.

## 1-e. How are trials filtered based on quality controls?

i. The AI only filters out trials with fewer than 2 timepoints. It does not implement the reference solution’s `<50` timepoint threshold.

ii.
```python
n_tp = t_end - t_start

if n_tp < 2:
    continue
```

iii. No explicit justification for this threshold appears in the notes. The likely implied rationale is the target-format requirement that sessions retain at least two trials, not a paper-matching trial QC rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from `processing/ophys/Deconvolved/plane*/data`.

ii.
```python
deconv_group = f['processing']['ophys']['Deconvolved']
...
deconv_data = deconv_group[plane_keys[0]]['data'][:]
```

iii. `CONVERSION_NOTES.md` Step 1 states “Neural data: `sess.timeseries["events"]` = deconvolved calcium events” and Step 3 says the target neural signal is deconvolved calcium events, not raw fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. The AI concatenates multiple imaging planes along the ROI axis, filters to `iscell==1`, transposes to `(n_neurons, n_timepoints)`, and casts each trial matrix to `float32`. It does not compute dF/F or perform additional temporal processing.

ii.
```python
plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
if len(plane_keys) == 1:
    deconv_data = deconv_group[plane_keys[0]]['data'][:]
else:
    plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
    deconv_data = np.concatenate(plane_data, axis=1)
...
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
...
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. In Step 5, the AI’s notes say “Filter by iscell, transpose” and “Multi-plane pooling: Concatenate plane0 and plane1 data for m17, m18.” The trajectory shows it concluded dF/F computation was unnecessary because deconvolved events were already stored.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered solely by the `iscell` mask from image segmentation. The AI explicitly chose not to implement the paper’s interneuron exclusion.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
...
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. `CONVERSION_NOTES.md` Step 5 lists “Use iscell as-is” as a key decision and says interneuron filtering was skipped because the effect was only about 0.42%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural data are aligned to trial start implicitly by cutting the neural array with the same `[trial_start:teleport]` boundaries used for the behavior arrays.

ii.
```python
t_start = trial_start_inds[t]
t_end = teleport_inds[t]
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. Step 3 of the notes says “Temporal alignment: per-trial, from trial_start to teleport,” and the metadata later describe the alignment event as `trial_start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native sampling resolution. It estimates the bin width as the median difference in `position` timestamps and applies no temporal rebinning.

ii.
```python
pos_timestamps = beh['position']['timestamps'][:]
...
dt = np.median(np.diff(pos_timestamps))
...
'time_bin_size': median_dt * 1000,
```

iii. `CONVERSION_NOTES.md` Step 5 says “Native frame rate: Use ~64.5 ms time bins (no resampling).”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The AI derives it from the trial length `n_tp` and a session-wide `dt` estimated from `position` timestamps, not from the per-sample behavior timestamps themselves.

ii.
```python
pos_timestamps = beh['position']['timestamps'][:]
dt = np.median(np.diff(pos_timestamps))
...
time_from_start = np.arange(n_tp) * dt
```

iii. Step 5 of the notes explicitly maps this variable as “Frame index × dt,” showing that the AI intentionally reconstructed time from frame count rather than using stored timestamps directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI creates a regularly spaced vector `0, dt, 2*dt, ...` with length equal to the number of samples in that trial.

ii.
```python
time_from_start = np.arange(n_tp) * dt
...
input_data[0, :] = time_from_start
```

iii. The notes justify this by assuming a stable native frame rate and no resampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Alignment is by shared sample indices within each trial. The same trial slice used for neural data determines the length of `time_from_start`.

ii.
```python
t_start = trial_start_inds[t]
t_end = teleport_inds[t]
n_tp = t_end - t_start
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
time_from_start = np.arange(n_tp) * dt
```

iii. The AI’s notes say the data use the native frame rate and are aligned per trial from `trial_start` to `teleport`. It did not document any additional cross-stream timestamp verification.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
env_data = beh['environment']['data'][:]
```

iii. Step 5 of the notes maps “environment data” directly to the environment input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI extracts the trial’s `environment` samples, drops any `-1` values, takes the median, converts it to an integer, and repeats that constant value over the whole trial.

ii.
```python
trial_env = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    ...
    env_vals = env_data[t_start:t_end]
    env_vals = env_vals[env_vals >= 0]
    if len(env_vals) > 0:
        trial_env[t] = int(np.median(env_vals))
...
env_type = float(trial_env[t])
input_data[1, :] = env_type
```

iii. `CONVERSION_NOTES.md` Step 5 describes environment as a per-trial binary input, which explains the median-and-repeat treatment.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `t` after trial boundaries have been identified from `trial_start` and `teleport`.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = float(t)
```

iii. The notes map this variable simply as “Trial index, 0-indexed.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing beyond converting the loop index to float and repeating it across all timepoints in the trial.

ii.
```python
trial_num = float(t)
...
input_data[2, :] = trial_num
```

iii. `CONVERSION_NOTES.md` Step 5 says this is a per-trial variable, so the AI stores it as constant through the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` timestamps, which are converted to frame indices and then summarized as whether each trial received reward.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
...
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
...
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
```

iii. The notes map this variable as “Previous trial reward” and repeatedly describe reward outcome as derived from reward deliveries within each trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a per-trial binary `trial_rewarded`, then sets the current trial’s previous-outcome input to `0` for the first trial and `trial_rewarded[t-1]` otherwise. The value is constant within the trial.

ii.
```python
for t in range(n_trials):
    ...
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
    trial_rewarded[t] = 1 if reward_in_trial else 0
...
if t == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(trial_rewarded[t - 1])
...
input_data[3, :] = prev_outcome
```

iii. Step 5 of the notes says the variable is binary `0/1` and per trial, which matches this implementation.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives it from the `position` time series plus a per-trial reward-zone start inferred from the `reward_zone` time series. It uses the minimum trial position where `reward_zone > 0`, maps that to zone A/B/C with a tolerance window, and for unlabeled trials fills from the nearest neighboring labeled trial.

ii.
```python
def get_reward_zone_start_per_trial(pos, rzone, trial_start_inds, teleport_inds):
    ...
    rz_active = trial_rzone > 0
    if np.any(rz_active):
        rz_start_pos = trial_pos[rz_active].min()
        rz_starts[t] = rz_start_pos
        rz_labels[t] = determine_reward_zone_label(rz_start_pos)

def get_reward_zone_for_trial(rz_labels, trial_idx):
    if rz_labels[trial_idx] is not None:
        return rz_labels[trial_idx]
    for offset in range(1, len(rz_labels)):
        if trial_idx - offset >= 0 and rz_labels[trial_idx - offset] is not None:
            return rz_labels[trial_idx - offset]
        if trial_idx + offset < len(rz_labels) and rz_labels[trial_idx + offset] is not None:
            return rz_labels[trial_idx + offset]
```

iii. `CONVERSION_NOTES.md` Step 10 says “Reward zone: Inferred from position data,” and the trajectory shows the AI believed this was equivalent to the reference after its own spot checks.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a trial’s reward-zone start is known, the AI computes signed distance to the interval `[rz_start, rz_start + 50]`: negative before the zone, zero inside, positive after.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start):
    rz_end = rz_start + 50.0
    distance = np.zeros_like(position)
    before_mask = position < rz_start
    distance[before_mask] = position[before_mask] - rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    distance[in_mask] = 0.0
    after_mask = position > rz_end
    distance[after_mask] = position[after_mask] - rz_end
    return distance
```

iii. The notes call this “Distance to reward zone: Computed from position relative to reward zone boundaries.”

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI manually thresholds the signed distance into the seven requested bins.

ii.
```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. Step 5 of the notes says the variable is a “7-bin discretization,” and the code follows the bins from the task instructions literally.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by taking `position[t_start:t_end]` from the same trial slice used to extract `trial_neural`.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
trial_pos = position[t_start:t_end]
distance = compute_distance_to_reward_zone(trial_pos, rz_start)
```

iii. The AI’s notes describe all streams as aligned per trial from `trial_start` to `teleport`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[t_start:t_end]
```

iii. The notes map `Position` directly to the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI discretizes absolute position into five equal-width bins spanning `0` to `450` cm by using `np.linspace(0, TRACK_LENGTH, 6)` and clipping the digitized result to `0..4`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. `CONVERSION_NOTES.md` Step 5 calls for “5-bin equal discretization,” and the AI interpreted that as equal bins over the nominal 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are the equal-width edges implied by `np.linspace(0, 450, 6)`, i.e. 90 cm bins after clipping.

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. The AI justified this in the notes as “equal discretization” of the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial sample indices as the neural data.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
trial_pos = position[t_start:t_end]
pos_bins = discretize_position(trial_pos, n_bins=5)
```

iii. The notes describe all decoder variables as aligned to the same trial-relative sampling grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii.
```python
lick_data = beh['lick']['data'][:]
...
trial_licks = licks_processed[t_start:t_end]
```

iii. Step 5 of the notes maps `lick` directly to the lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies three steps: per-trial lick-error rejection (`>35%` of samples above `2`), clipping all positive counts above `1` to `1`, Gaussian smoothing with `sigma=2`, and then thresholding the smoothed result at `>0.5` to make a binary lick output.

ii.
```python
def process_licks(lick_data, trial_start_inds, teleport_inds):
    ...
    if len(trial_licks) > 0:
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > LICK_ERROR_THRESHOLD:
            licks[t_start:t_end] = np.nan
    licks[licks > 1] = 1
    return licks
...
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 10 explicitly justify this as following the original paper code’s lick-error correction and smoothing logic.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It uses the same trial slice `[t_start:t_end]` as the neural data and the other outputs.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
trial_licks = licks_processed[t_start:t_end]
```

iii. The notes say the full conversion is aligned per trial from `trial_start` to `teleport`.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position` via the same inferred-label procedure used for distance to reward zone.

ii.
```python
rz_starts, rz_labels = get_reward_zone_start_per_trial(
    position, rzone, trial_start_inds, teleport_inds)
...
label = get_reward_zone_for_trial(rz_labels, t)
trial_rz_label[t] = rz_label_map.get(label, 0)
```

iii. The notes justify this the same way as the distance variable: reward zone is inferred from position and reward-zone activity in the NWB file.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers a label A/B/C per trial, fills missing labels from neighboring trials, and maps `A,B,C` to integer classes `0,1,2`.

ii.
```python
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
for t in range(n_trials):
    label = get_reward_zone_for_trial(rz_labels, t)
    trial_rz_label[t] = rz_label_map.get(label, 0)
```

iii. The AI’s trajectory says this was chosen after manual inspection of reward-zone positions and was treated as equivalent to the reference after spot checks.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` timestamps, converted to frame indices against the position timestamp grid.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
```

iii. Step 5 of the notes maps reward delivery directly to the reward-outcome output.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI marks it rewarded if any reward frame index falls between that trial’s start and end. It then repeats the per-trial result across all timepoints in the output matrix.

ii.
```python
for t in range(n_trials):
    ...
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
    trial_rewarded[t] = 1 if reward_in_trial else 0
...
reward_out = trial_rewarded[t]
output_data[5, :] = reward_out
```

iii. The notes say reward outcome is a binary per-trial output, so the AI implemented it as constant within trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases with permissive fallbacks: mismatched counts of `trial_start` and `teleport` are truncated to the shorter count; missing reward-zone labels are filled from the nearest labeled neighboring trial; lick-error trials are replaced with `NaN` before smoothing; `environment==-1` values are ignored when assigning trial environment.

ii.
```python
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
...
if rz_labels[trial_idx] is not None:
    return rz_labels[trial_idx]
for offset in range(1, len(rz_labels)):
    ...
return 'A'
...
env_vals = env_vals[env_vals >= 0]
```

iii. `CONVERSION_NOTES.md` emphasizes lick-error handling and omission-trial handling. The trajectory shows the AI preferred “real data should be kept” and used local fallbacks rather than hard assertions for several edge cases.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps in the AI code are loading large NWB files from disk, concatenating and slicing the neural arrays session by session, and writing the final multi-gigabyte pickle. Optional plotting can also add cost.

ii.
```python
f = h5py.File(nwb_path, 'r')
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, ...)
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes report about 70 seconds for full conversion and a 9.4 GB output file, which supports the I/O-heavy interpretation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI uses several Python trial loops that could be partially vectorized: lick QC, reward detection per trial, environment assignment per trial, reward-zone label filling, and the main per-trial construction of neural/input/output arrays.

ii.
```python
for t in range(len(trial_start_inds)):
    ...
for t in range(n_trials):
    ...
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    ...
    neural_trials.append(trial_neural)
```

iii. No explicit vectorization discussion appears in the notes. The implementation suggests the AI prioritized straightforward trial-wise logic over bulk array transforms.

## 13-c. What processing does the code repeat multiple times?

i. The AI repeats neighbor-based reward-zone lookup twice per session, once to build `trial_rz_label` and again to build `trial_rz_start`. It also loops over trials multiple times for separate summaries instead of combining those passes.

ii.
```python
for t in range(n_trials):
    label = get_reward_zone_for_trial(rz_labels, t)
    trial_rz_label[t] = rz_label_map.get(label, 0)
...
for t in range(n_trials):
    label = get_reward_zone_for_trial(rz_labels, t)
    trial_rz_start[t] = REWARD_ZONES[label][0]
```

iii. There is no explicit justification in the notes. The code structure suggests simplicity was favored over minimizing repeated passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads some data it never uses in the final dataset, notably `scanning` and `sess_id` bookkeeping. In the plotting path it also concatenates `all_pos`, `all_spd`, and `all_lck` even though only the distance distribution is plotted.

ii.
```python
scanning = beh['scanning']['data'][:]
...
all_sess_ids = []
...
all_sess_ids.append(result['sess_id'])
```

```python
all_dist = np.concatenate([o[0, :] for o in output_trials])
all_pos = np.concatenate([o[1, :] for o in output_trials])
all_spd = np.concatenate([o[2, :] for o in output_trials])
all_lck = np.concatenate([o[3, :] for o in output_trials])
ax.bar([0,1,2,3,4,5,6], [np.mean(all_dist==i) for i in range(7)], ...)
```

iii. No explicit justification was given. These look like incidental byproducts of exploration and plotting rather than downstream requirements.
