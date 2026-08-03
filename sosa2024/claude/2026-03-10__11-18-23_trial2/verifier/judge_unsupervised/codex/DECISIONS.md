# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `data/` for every `sub-*` directory, gathers every `*.nwb` file, and processes them one by one. Inside each NWB file it opens the file with `h5py`, reads neural arrays from `processing/ophys/...`, behavioral arrays from `processing/behavior/BehavioralTimeSeries/...`, and then segments them into trials inside `process_session()`.

ii. ```python
def find_nwb_files(data_dir='data'):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    sessions = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
        for fpath in files:
            sessions.append({
                'subject': subj.replace('sub-', ''),
                'filepath': fpath,
                'filename': os.path.basename(fpath),
            })
    return sessions

with h5py.File(filepath, 'r') as f:
    deconv_group = f['processing/ophys/Deconvolved']
    position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
    lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. `CONVERSION_NOTES.md` says the NWB files are organized as `data/sub-{id}/sub-{id}_ses-{ses}_behavior+ophys.nwb` and that all 152 sessions were processed. The trajectory shows the agent deliberately scanning subject folders and then iterating through all NWB files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the `sub-*` folder names. The script strips the `sub-` prefix to get a subject id such as `m11`, stores the per-session subject id, and later builds `subjects` and `subject_idx` from those stored ids.

ii. ```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
'subject': subj.replace('sub-', ''),
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes explicitly describe the dataset as subject-organized and list 11 mice. No alternate subject-splitting rule appears in the trajectory.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script collects one session record per file and processes those records sequentially.

ii. ```python
files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
for fpath in files:
    sessions.append({
        'subject': subj.replace('sub-', ''),
        'filepath': fpath,
        'filename': os.path.basename(fpath),
    })
...
for i, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['filepath'], ...)
```

iii. `CONVERSION_NOTES.md` states there are 152 sessions total and that the files are one session per NWB file.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start > 0` as the start index and `teleport > 0` as the end index. If the number of teleport indices does not match the number of trial starts, the script pairs each trial start with the first later teleport index.

ii. ```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
...
if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
    teleport_inds = np.array(matched_teleports)
```

iii. The notes say trial boundaries come from `trial_start` and `teleport`. The trajectory does not show a separate conceptual defense beyond following those NWB fields.

## 1-e. How are trials filtered based on quality controls?

i. Trials are barely filtered. Entire sessions are skipped if they have fewer than two detected trials, and individual trials are skipped only if they contain fewer than two timepoints.

ii. ```python
if n_trials < 2:
    print(f"  WARNING: Only {n_trials} trials, skipping session")
    return None
...
for t in range(n_trials):
    ...
    if n_tp < 2:
        continue
```

iii. The notes emphasize session-level validation and decoder success, but they do not document a stricter per-trial minimum-duration rule. The trajectory likewise focuses on getting all sessions through conversion rather than culling short trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` output comes from the NWB `Deconvolved` traces, after selecting ROIs with `iscell` and applying an extra interneuron mask. The script also loads `Fluorescence` and `Neuropil` arrays, but only to compute that interneuron exclusion mask.

ii. ```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
...
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
neural_all = deconv[:, final_cell_mask].T
```

iii. `CONVERSION_NOTES.md` explicitly says the NWB `Deconvolved` data are already the processed deconvolved events and should be used directly, while `Fluorescence` and `Neuropil` are loaded to support the extra cell-filtering step.

## 2-b. How is the `neural` data processed?

i. The script concatenates planes, optionally crops neural and behavioral streams to a shared length, filters cells, transposes the deconvolved matrix into neuron-by-time form, slices it by trial, and converts NaNs to zeros inside each trial.

ii. ```python
deconv = np.concatenate(deconv_list, axis=1)
...
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
...
neural_all = deconv[:, final_cell_mask].T
...
trial_neural = neural_all[:, si:ei].copy()
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The notes say the agent chose to use already-deconvolved NWB data rather than recomputing dF/F and OASIS. The trajectory reflects that same decision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps only `iscell[:, 0] == 1` ROIs and then removes any accepted ROI whose simplified dF/F is correlated with running speed above 0.5. That second filter is computed from `Fluorescence - 0.7 * Neuropil` with a median-based baseline, not with the full trialwise maximin baseline described in the notes.

ii. ```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
...
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
if corrs[i] > 0.5:
    interneuron_mask[col_idx] = True
...
final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. The notes cite the paper’s interneuron exclusion criterion and say the agent implemented it. The trajectory later shows the agent optimizing and fixing this correlation-based exclusion step because it was a runtime bottleneck.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from `trial_start_inds[t]` to `teleport_inds[t]`. There is no additional offset; each trial’s first neural sample is the first frame after the start marker.

ii. ```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_neural = neural_all[:, si:ei].copy()
```

iii. The notes state that the temporal alignment event is start of trial and repeatedly describe the conversion as trial-start aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the imaging-frame resolution, using about 15.5 Hz or 64.48 ms per sample. No temporal rebinning is applied.

ii. ```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
...
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
```

iii. `CONVERSION_NOTES.md` says behavior is already aligned to imaging at about 15.5 Hz and records the final time bin as 64.48 ms.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not taken from a behavioral timestamp variable. The script derives it from the number of imaging frames in the trial and the session’s imaging rate.

ii. ```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate
...
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The notes focus on the imaging rate and state that behavior is frame-aligned to imaging. There is no separate note justifying why actual timestamps were not used.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the script creates `0, 1, 2, ... n_tp-1`, multiplies by seconds per frame, reshapes the result to `(1, n_tp)`, and uses that as the first input row.

ii. ```python
frame_time = 1.0 / imaging_rate
...
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_input_tv = time_from_start.reshape(1, -1)
```

iii. The notes describe the final input as “Frame index * time_bin_size, in seconds.” The trajectory does not contain a deeper defense.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction to the same trial slice used for neural data. Each trial gets exactly `n_tp = ei - si` time values, so the time row and neural matrix have the same number of columns.

ii. ```python
si = trial_start_inds[t]
ei = teleport_inds[t]
n_tp = ei - si
trial_neural = neural_all[:, si:ei].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_input = np.vstack([
    trial_input_tv,
    ...
])
```

iii. The notes say all variables are frame-aligned at the imaging rate. The implementation follows that indexing scheme directly.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The stored environment input is derived from the NWB `identifier` string, not from the behavioral `environment` timeseries. The script parses scene names such as `Env1_LocationB_to_A` and converts them to `0` for ENV1 or `1` for ENV2.

ii. ```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_env = env_per_trial.copy()
```

iii. The notes say the agent switched to scene parsing because it considered that “more reliable for cross-env switches,” and trajectory steps 95, 97, and 101 show the agent explicitly fixing scene parsing and then preferring it over the NWB environment field.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script parses the scene string, splits pre-switch and post-switch trials at trial 30 if the scene contains `_to_`, converts `Env1` to `0` and `Env2` to `1`, and repeats the resulting per-trial scalar across all timepoints in that trial.

ii. ```python
def _parse_zone_and_env(parts):
    for p in parts:
        if p.startswith('Env'):
            env_num = int(p.replace('Env', ''))
            env = env_num - 1
...
trial_input = np.vstack([
    trial_input_tv,
    np.full((1, n_tp), env_type, dtype=np.float32),
    ...
])
```

iii. `CONVERSION_NOTES.md` says environment is taken from scene parsing, especially for cross-environment switch sessions. The trajectory explicitly records that design change.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial-number input is not read from a separate NWB variable. It is derived from the loop index over the trial segments defined by `trial_start` and `teleport`.

ii. ```python
for t in range(n_trials):
    ...
    trial_num = np.float32(t)
```

iii. The notes map “Trial number” directly to an integer per trial. There is no separate justification beyond using the segmented trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The script uses the zero-based trial index `t` and broadcasts it across all timepoints in the trial.

ii. ```python
trial_num = np.float32(t)
...
np.full((1, n_tp), trial_num, dtype=np.float32),
```

iii. The notes describe this variable simply as an integer per trial. The code implements exactly that.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward/timestamps` events after they are matched to frame indices and summarized into a per-trial rewarded/not-rewarded flag.

ii. ```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
if np.any(reward_frames[si:ei] > 0):
    trial_rewarded[t] = 1
```

iii. The notes explicitly say reward is determined by matching sparse reward timestamps to behavior frame times and then turned into per-trial reward outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script first computes `trial_rewarded[t]` for every trial, then creates a lagged array where trial `t` receives `trial_rewarded[t-1]`. Trial 0 is hard-coded to 0 because it has no previous trial.

ii. ```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    ...
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. The notes say this input is binary and lagged from the previous trial. The trajectory contains no competing idea.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial position samples and a reward-zone interval for that trial. The position samples come from `BehavioralTimeSeries/position/data`; the reward-zone interval comes from scene-parsed labels mapped through `REWARD_ZONE_DICT`.

ii. ```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_pos = position[si:ei]
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes describe the reward-zone coordinates as coming from the reference code and the zone label as coming from scene parsing. The trajectory around scene parsing supports that.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The script computes signed distance to the reward zone: positions before the zone are measured relative to the zone start, positions after the zone relative to the zone end, and positions inside the zone are set to zero.

ii. ```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    dist[before] = position[before] - rz_start
    dist[after] = position[after] - rz_end
    dist[inside] = 0.0
    return dist
```

iii. The notes map this output to a signed distance variable and describe the binning scheme as the decoder target.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into seven categories using the requested cutpoints: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii. ```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The notes and README-level outputs present this exact seven-bin decoder target. There is no sign the agent considered an alternate thresholding rule.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing position with the same `[si:ei]` trial indices used for neural data, then computing one distance value per neural frame.

ii. ```python
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_bins = discretize_distance(signed_dist)
```

iii. The notes say all behavioral variables were aligned at the imaging frame rate before trial segmentation.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `BehavioralTimeSeries/position/data` on the same trial slices used elsewhere.

ii. ```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. The notes map absolute position directly from the linear-track position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The script divides position by 90 cm, floors the result, clips it to `[0, 4]`, and uses that as the five-category absolute-position output.

ii. ```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. `CONVERSION_NOTES.md` says the agent planned “5 equal bins (90 cm each),” and the code follows that plan exactly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm, implemented with `floor(position / 90)`.

ii. ```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
...
'output_values': [
    ...,
    ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
    ...
]
```

iii. The notes explicitly document this 90 cm binning choice.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced with the same per-trial indices as the neural matrix, so each neural frame gets one aligned position category.

ii. ```python
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. The notes describe all decoder outputs as frame-aligned to the trial-start segmentation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `BehavioralTimeSeries/lick/data`.

ii. ```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
...
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes identify the NWB `lick` field as the source and describe it as cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script first binarizes any positive lick count to 1. It then performs a per-trial error heuristic: if more than 30% of frames in a trial have lick count greater than 2, that trial’s lick trace is overwritten with zeros. The final output is a binary `0/1` vector.

ii. ```python
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1
for t in range(n_trials):
    ...
    if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
        lick_binary[si:ei] = 0
...
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes cite the paper’s lick-sensor error rule and say the agent applied it, but they also acknowledge that the code uses `0` rather than `NaN` “for cleaner output.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks are aligned by using the same per-trial frame indices used for neural data and other behavioral outputs.

ii. ```python
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes repeatedly state that behavior is sampled at the imaging frame rate, so trial slicing is enough to align licks to neural frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The stored reward-zone location is not derived from the behavioral `reward_zone` signal. It is derived from the NWB `identifier` scene string, parsed into reward-zone labels `A/B/C`.

ii. ```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The notes say scene parsing was extended to handle all observed session-name formats and was used to assign reward zones. The trajectory shows explicit scene-parser fixes for cross-environment switches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into pre-switch and post-switch reward-zone labels, with the switch assumed to happen at trial 30. Those labels are converted to integers `A->0`, `B->1`, `C->2` and repeated across timepoints in each trial.

ii. ```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    ...
    labels[:change_trial] = from_zone
    labels[change_trial:] = to_zone
...
rz_loc = zone_map.get(rz_labels[t], 0)
...
np.full((1, n_tp), rz_loc, dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` says reward-zone location was taken from scene parsing and that switch days changed after trial 30. The trajectory corroborates that choice.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the sparse `Reward/timestamps` events aligned to the frame timestamps, then summarized per trial.

ii. ```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
rew_outcome = int(trial_rewarded[t])
```

iii. The notes explicitly say reward is a sparse event field that was matched to behavior timestamps and converted into frame-aligned and trial-aligned reward indicators.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each reward timestamp is assigned to the nearest frame, a binary framewise reward vector is built, and a trial is marked rewarded if any reward frame falls within that trial slice. The resulting scalar is repeated across the trial.

ii. ```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
if np.any(reward_frames[si:ei] > 0):
    trial_rewarded[t] = 1
...
np.full((1, n_tp), rew_outcome, dtype=np.int64),
```

iii. The notes describe exactly this nearest-frame and then per-trial summarization strategy.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several issues heuristically. If neural and behavioral streams differ in length, it crops both to the shorter length. If the number of teleport markers does not match the number of trial starts, it pairs each trial start with the first later teleport. Neural NaNs are replaced with zeros after trial slicing. Suspected lick-sensor-error trials are zeroed rather than set to NaN. The script does not document a separate missing-data policy beyond these fixes.

ii. ```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
    ...

if len(teleport_inds) != n_trials:
    ...
    teleport_inds = np.array(matched_teleports)

trial_neural = np.nan_to_num(trial_neural, nan=0.0)
...
if ... > 0.30:
    lick_binary[si:ei] = 0
```

iii. The notes mention off-by-one neural/behavior mismatches and call out lick-sensor correction. The trajectory also records alignment fixes and parser fixes as practical cleanup steps.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are session-by-session HDF5 reads of multiple large arrays, concatenation of multi-plane data, the simplified dF/F computation for every accepted ROI, and the speed-correlation interneuron screen. Trial-by-trial slicing for all sessions also contributes.

ii. ```python
for plane in planes:
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
    ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
f_corrected = fluorescence - 0.7 * neuropil_data
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
corrs = (speed_z @ dff_z) / len(speed_z)
```

iii. `CONVERSION_NOTES.md` and the trajectory both single out the interneuron correlation check as the main bottleneck, especially before the agent vectorized it.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the `cell_mask_concat` list comprehension, the per-reward `argmin` search over all timestamps, the per-trial reward summary loop, the per-trial lick-error loop, and repeated per-trial slicing loops that walk the same boundaries multiple times.

ii. ```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
...
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
for t in range(n_trials):
    ...
    if len(trial_lick) > 0 and ...:
        lick_binary[si:ei] = 0
```

iii. The trajectory shows the agent already recognized one bottleneck and vectorized it, which supports the same efficiency diagnosis for the remaining loops.

## 13-c. What processing does the code repeat multiple times?

i. The code reuses the same trial boundaries in multiple passes: once to compute reward outcome, again to correct lick errors, and again to build the final neural/input/output trial objects. It also repeatedly slices position, speed, and lick by the same `[si:ei]` intervals.

ii. ```python
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1

for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_lick = lick[si:ei]
    ...

for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_neural = neural_all[:, si:ei].copy()
    trial_pos = position[si:ei]
```

iii. This repeated-pass structure is visible directly in the implementation. The notes do not frame it as a problem, but the trajectory’s optimization work implies the agent was aware performance mattered.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `Fluorescence` and `Neuropil` even though only deconvolved activity is saved; computes `trial_input_pt` but never uses it; loads the raw `environment` timeseries but then ignores it in favor of scene parsing; and carries plotting/debug machinery that does not affect `converted_data.pkl`.

ii. ```python
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)
...
environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
trial_env = env_per_trial.copy()
```

iii. The notes justify fluorescence and neuropil loading only for the interneuron filter and justify scene parsing as a replacement for the NWB environment field. The unused `trial_input_pt` has no written justification and appears to be leftover code.
