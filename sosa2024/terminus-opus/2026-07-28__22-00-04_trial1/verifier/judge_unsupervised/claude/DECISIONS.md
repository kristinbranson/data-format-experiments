# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files using h5py. It discovers all NWB files via a glob pattern `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`, sorts them, and iterates through each file. Each NWB file corresponds to one session. For each file, the AI opens it with `h5py.File()`, extracts behavioral timeseries from `processing/behavior/BehavioralTimeSeries/` and neural data from `processing/ophys/Deconvolved/`. It processes each session independently in a loop and aggregates results into lists.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
# ...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, ...)
```

```python
f = h5py.File(nwb_path, 'r')
beh = f['processing']['behavior']['BehavioralTimeSeries']
position = beh['position']['data'][:]
speed_data = beh['speed']['data'][:]
lick_data = beh['lick']['data'][:]
# ... etc
deconv_group = f['processing']['ophys']['Deconvolved']
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that data is organized as NWB files per subject/session. The reference code originally loaded pickle-based `sess` objects via `multi_anim_sess`, but since the task requires working from NWB files, the AI adapted the loading to read the equivalent fields from NWB format.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from the NWB file metadata field `general/subject/subject_id`. Unique subjects are collected from all sessions and sorted. A `subject_idx` array maps each session to its subject.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
# ...
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
subject_idx = np.array([unique_subjects.index(s) for s in all_subj_ids])
```

iii. The AI identified 11 subjects (m3, m4, m7, m11-m15, m17-m19) consistent with the paper's report of n=11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file represents one session. The session ID is extracted from `general/session_id`. All 152 NWB files are processed as individual sessions.

ii.
```python
sess_id = f['general']['session_id'][()].decode()
```

iii. The AI documented 152 total sessions across 11 subjects, with 12-14 sessions per subject, consistent with the dataset.

## 1-d. How are the data split into trials?

i. Trials are identified by finding indices where `trial_start` markers are > 0 (trial onset) and `teleport` markers are > 0 (trial end). Each trial spans from `trial_start_inds[t]` to `teleport_inds[t]`.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
n_trials = len(trial_start_inds)
```

iii. This matches the reference code's use of `sess.trial_start_inds` and `sess.teleport_inds`. The AI verified trial counts (12,216 total) are consistent with the data.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: trials with fewer than 2 timepoints are skipped. Lick sensor error detection is applied per-trial (if >35% of samples have cumulative lick count >2, lick data is set to NaN for that trial), but the trial is still included. The AI does NOT apply speed threshold masking (speed < 2 cm/s) which the reference code uses to exclude low-speed timepoints.

ii.
```python
if n_tp < 2:
    continue
```

```python
# Lick sensor error correction per trial
if len(trial_licks) > 0:
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > LICK_ERROR_THRESHOLD:
        licks[t_start:t_end] = np.nan
```

iii. The AI noted in CONVERSION_NOTES Step 4 that speed masking was intentionally NOT applied because "speed is a decoder output" and all timepoints are needed for the decoder. The lick error threshold of 0.35 follows the reference code rather than the paper's stated 0.30.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the deconvolved calcium events stored at `processing/ophys/Deconvolved/plane{N}/data` in the NWB files. This corresponds to the reference code's `sess.timeseries['events']`.

ii.
```python
deconv_group = f['processing']['ophys']['Deconvolved']
plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
if len(plane_keys) == 1:
    deconv_data = deconv_group[plane_keys[0]]['data'][:]
else:
    plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
    deconv_data = np.concatenate(plane_data, axis=1)
```

iii. The AI confirmed that NWB files contain pre-computed deconvolved events (OASIS via Suite2p), so dF/F computation is not needed. Multi-plane animals (m17, m18) have data concatenated across planes.

## 2-b. How is the `neural` data processed?

i. The neural data is filtered by `iscell` (keeping only ROIs classified as cells), transposed from (n_timepoints, n_rois) to (n_neurons, n_timepoints), and sliced per trial. Data is cast to float32 for memory efficiency. No smoothing, normalization, or speed masking is applied.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T  # (n_neurons, n_timepoints)
# ...
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The AI justified using iscell as-is since the NWB files include manual curation results. The reference code also uses deconvolved events directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `iscell` filtering is applied — ROIs with `iscell[:,0] == 1` are kept. Interneuron exclusion (cells with speed correlation r > 0.5, ~0.42% of cells) is NOT applied.

ii.
```python
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. The AI noted in CONVERSION_NOTES Step 3 that interneuron exclusion affects only ~0.42% of cells and decided to skip it, relying on the NWB iscell which includes manual curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data is sliced from `trial_start_inds[t]` to `teleport_inds[t]`, so the first timepoint corresponds to the start of the trial (entry to the linear track).

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The instructions specify "Temporally align based on start of the trial." The AI's approach aligns to trial_start markers, consistent with the reference code's use of `trial_start_inds`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame rate of ~15.5 Hz (~64.5 ms per frame). No temporal rebinning is applied. The time bin size is computed as the median inter-frame interval from the position timestamps.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
# ...
'time_bin_size': median_dt * 1000,  # in ms
```

iii. The AI confirmed this matches the paper's reported frame rate of ~15.5 Hz / ~0.0645 s per frame.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index within a trial and the median inter-frame interval (dt) computed from position timestamps.

ii.
```python
dt = np.median(np.diff(pos_timestamps))
# ...
time_from_start = np.arange(n_tp) * dt
```

iii. This produces a monotonically increasing time vector starting at 0 for each trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The number of timepoints in the trial (n_tp = t_end - t_start) is used to create a range [0, 1, ..., n_tp-1], which is multiplied by dt (the median frame duration in seconds).

ii.
```python
n_tp = t_end - t_start
time_from_start = np.arange(n_tp) * dt
input_data[0, :] = time_from_start
```

iii. No complex processing — just a linear time ramp.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same number of timepoints (n_tp) as the neural data for that trial, since both are derived from the same trial boundaries (t_start to t_end). The first timepoint is at time 0.

ii.
```python
input_data = np.zeros((4, n_tp), dtype=np.float32)
input_data[0, :] = time_from_start
```

iii. Alignment is inherent since both use the same frame indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/environment/data` in the NWB file.

ii.
```python
env_data = beh['environment']['data'][:]
```

iii. The environment field contains values 0 (ENV1) or 1 (ENV2), with -1 for pre-scanning samples. This corresponds to the reference code's `morph` variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the environment values within the trial window are extracted, values of -1 are excluded, and the median of remaining values is taken and cast to int. This single value is broadcast across all timepoints.

ii.
```python
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    env_vals = env_data[t_start:t_end]
    env_vals = env_vals[env_vals >= 0]  # exclude -1
    if len(env_vals) > 0:
        trial_env[t] = int(np.median(env_vals))
```

iii. The AI correctly handles the -1 sentinel values that appear before scanning starts.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop index `t` (0-indexed trial index within the session), NOT from the NWB `trial number` field.

ii.
```python
trial_num = float(t)
```

iii. The NWB file contains a `trial number` field that has values [0, 1, 2, ...] at trial start indices, which would give the same result as the loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Simply uses the loop index as a float. Broadcast across all timepoints in the trial.

ii.
```python
trial_num = float(t)
input_data[2, :] = trial_num
```

iii. Minimal processing — just type conversion.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` field in the NWB file. Reward timestamps are converted to frame indices and compared against trial boundaries to determine per-trial reward.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. The AI uses reward timestamps rather than the binary reward data, converting them to frame indices for trial assignment.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A binary `trial_rewarded` array is built: for each trial, check if any reward frame index falls within [t_start, t_end). Previous trial outcome is then `trial_rewarded[t-1]`, with 0 for the first trial.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    t_start = trial_start_inds[t]
    t_end = teleport_inds[t]
    reward_in_trial = np.any(
        (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
    )
    trial_rewarded[t] = 1 if reward_in_trial else 0
# ...
if t == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(trial_rewarded[t - 1])
```

iii. The reference code's `get_trial_types` additionally checks that `rzone > 0` is also active during the reward — the AI only checks for reward timestamps. This could differ for auto-rewarded trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` data in the NWB file. The reward zone start position is inferred by finding the minimum position where `rzone > 0` within a trial.

ii.
```python
trial_pos = position[t_start:t_end]
trial_rzone = rzone[t_start:t_end]
rz_active = trial_rzone > 0
if np.any(rz_active):
    rz_start_pos = trial_pos[rz_active].min()
```

iii. The AI infers reward zone location from position data rather than using the reference code's scene-based lookup (reward_zone_dict). For omission trials (where rzone is never active), it falls back to neighboring trials' reward zone labels.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from position to the nearest boundary of a 50cm-wide reward zone [rz_start, rz_start+50]. Negative = before zone, 0 = inside zone, positive = after zone.

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

iii. The instructions say "Distance to any location in the reward zone." The AI interprets this as signed distance to the nearest edge of the reward zone, with 0 inside the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. 7 bins as specified in the instructions: <-50, -50 to -10, -10 to <0, 0, >0 to +10, +10 to +50, >+50.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. Directly follows the bin definitions from the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Uses the same trial frame indices (t_start:t_end) as the neural data, so alignment is inherent.

ii.
```python
trial_pos = position[t_start:t_end]
distance = compute_distance_to_reward_zone(trial_pos, rz_start)
dist_bins = discretize_distance(distance)
output_data[0, :] = dist_bins
```

iii. Same frame-level alignment as all other variables.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/position/data` in the NWB file.

ii.
```python
position = beh['position']['data'][:]
# ...
trial_pos = position[t_start:t_end]
```

iii. Position is in cm along the 450 cm virtual linear track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins spanning 0 to 450 cm (90 cm per bin) using `np.digitize`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. The bin edges are [0, 90, 180, 270, 360, 450]. `np.digitize(position, bin_edges[1:])` with clipping produces bins 0-4.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal bins: 0-90, 90-180, 180-270, 270-360, 360-450 cm.

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)  # [0, 90, 180, 270, 360, 450]
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. Follows the instruction to discretize into "5 equal-sized bins."

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Uses the same trial frame indices (t_start:t_end) as neural data.

ii.
```python
trial_pos = position[t_start:t_end]
pos_bins = discretize_position(trial_pos, n_bins=5)
output_data[1, :] = pos_bins
```

iii. Frame-level alignment via shared trial indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/lick/data` in the NWB file. The raw data contains cumulative lick counts per imaging frame (integer values 0-6+).

ii.
```python
lick_data = beh['lick']['data'][:]
```

iii. The reference README notes: "lick is a cumulative lick count in each frame, so in my behavior code anything >1 gets set to 1."

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies: (1) lick sensor error correction (if >35% of trial samples have count >2, set to NaN), (2) clip values >1 to 1 (binary), (3) Gaussian smoothing with sigma=2, (4) threshold at 0.5 for final binary output.

ii.
```python
# Error correction
frac_high = np.sum(trial_licks > 2) / len(trial_licks)
if frac_high > LICK_ERROR_THRESHOLD:
    licks[t_start:t_end] = np.nan

# Clip
licks[licks > 1] = 1

# Per-trial smoothing and thresholding
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
```

iii. The AI applied Gaussian smoothing from the reference `nansmooth` function. However, in the reference code's `get_timeseries_data`, the smoothing line is actually **commented out** (`# licks = ut.nansmooth(licks,2)`). The reference code only clips to [0,1] without smoothing.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Uses the same trial frame indices (t_start:t_end) as neural data.

ii.
```python
trial_licks = licks_processed[t_start:t_end]
# ... smoothing and thresholding ...
output_data[3, :] = lick_binary
```

iii. Frame-level alignment via shared trial indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` data in the NWB file. The reward zone start position is inferred by finding the minimum position where `rzone > 0` during each trial, then mapping to predefined zone labels (A=[80,130], B=[200,250], C=[320,370]).

ii.
```python
rz_active = trial_rzone > 0
if np.any(rz_active):
    rz_start_pos = trial_pos[rz_active].min()
    rz_labels[t] = determine_reward_zone_label(rz_start_pos)
```

```python
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
    return None
```

iii. The reference code uses `sess.scene` name to determine reward zones via a lookup dictionary. The AI instead infers the zone from position data, which is a different approach but produces equivalent results.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per trial: (1) find frames where rzone > 0, (2) get minimum position in those frames, (3) match to predefined zone labels with ±20 cm tolerance, (4) for trials where rzone is never active (omission trials), infer from nearest neighbor trial. Map A=0, B=1, C=2.

ii.
```python
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
for t in range(n_trials):
    label = get_reward_zone_for_trial(rz_labels, t)
    trial_rz_label[t] = rz_label_map.get(label, 0)
```

iii. The AI handles omission trials by searching forward/backward for the nearest trial with a known reward zone label.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` in the NWB file, converted to frame indices via `np.searchsorted` against position timestamps.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. The Reward field contains timestamps of reward delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward frame index falls within [t_start, t_end). Binary: 1 if rewarded, 0 if not.

ii.
```python
reward_in_trial = np.any(
    (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
)
trial_rewarded[t] = 1 if reward_in_trial else 0
```

iii. The reference code's `get_trial_types` additionally checks that `rzone > 0` is also present during the trial for a reward to count. The AI only checks reward timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Mismatched trial/teleport counts**: Truncated to the minimum of the two counts.
- **Trials with < 2 timepoints**: Skipped entirely.
- **Lick sensor errors**: If >35% of trial frames have lick count >2, lick data is set to NaN for that trial; lick output defaults to 0.
- **Pre-scanning samples**: Environment values of -1 are excluded when computing per-trial environment type.
- **Omission trials without rzone activation**: Reward zone label is inferred from neighboring trials.
- **NaN in smoothed licks**: Set to 0 in the binary lick output.

ii.
```python
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))

if n_tp < 2:
    continue

if not np.all(np.isnan(trial_licks)):
    # ... process licks
    lick_binary[np.isnan(smoothed_licks)] = 0
```

iii. The AI documented these edge cases in CONVERSION_NOTES Steps 5 and 10.

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files with h5py and reading the full deconvolved neural data arrays (which can be large, e.g., ~5000 ROIs × ~20000 timepoints for multi-plane animals). The full conversion takes ~70 seconds for 152 sessions.

ii.
```python
f = h5py.File(nwb_path, 'r')
deconv_data = deconv_group[plane_keys[0]]['data'][:]  # reads entire array into memory
```

iii. The AI estimated ~0.46s per session, ~70s total, which was well within the 15-minute target.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several per-trial loops could be vectorized:
- The main trial loop (lines 300-380) iterates over each trial to build neural, input, and output arrays.
- The reward determination loop (lines 262-269) checks each trial individually.
- The environment type loop (lines 273-279).
- The reward zone label loop (lines 282-286).
- The lick error correction loop in `process_licks` (lines 174-183).

ii.
```python
for t in range(n_trials):
    # ... all per-trial processing
```

iii. While these loops iterate over ~80 trials per session, the dominant cost is I/O (reading NWB files), so vectorization would yield modest speedups.

## 13-c. What processing does the code repeat multiple times?

i. Position data is accessed multiple times: once for reward zone computation, once for distance-to-reward-zone, once for absolute position binning. Speed data is also loaded and then re-sliced per trial. The trial boundaries are computed once but used across multiple independent loops for reward, environment, and reward zone labels.

ii.
```python
# Position used in get_reward_zone_start_per_trial, then again in trial loop for distance and position bins
trial_pos = position[t_start:t_end]  # accessed in trial loop
```

iii. The repeated access is not a major performance concern since the data is already in memory.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code applies **Gaussian smoothing to lick data** (`nansmooth` with sigma=2), which the reference code has **commented out**. This is unnecessary processing that changes the lick signal.

The code also computes `n_tp` (timepoints) for the last trial of each session and prints it in the summary, but this value is not particularly useful.

The `plot_processing` function generates extensive visualization plots that are only used when `--show-processing` is enabled, which is appropriate.

ii.
```python
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
```

iii. The reference code at line 118 of glmUtils.py has `# licks = ut.nansmooth(licks,2)` (commented out), indicating smoothing was considered but not used in the final processing.
