# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `glob.glob` to find all NWB files matching the pattern `/app/data/sub-*/sub-*_ses-*_behavior+ophys.nwb`. Each NWB file is opened with `h5py.File` (not `pynwb`). All behavioral and neural data arrays are read directly from the HDF5 groups. All files found are processed (152 sessions across 11 subjects).

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
# ...
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    # ... reads neural and behavioral arrays
```

iii. The AI explored the data directory structure and determined that all NWB files follow a consistent naming pattern. Using glob ensures all files are found. The AI confirmed 152 sessions across 11 subjects, matching the paper's description.

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading the `general/subject/subject_id` field from each NWB file. Unique subject IDs are collected from all sessions and sorted. A subject-to-index mapping is created for the `subject_idx` array.

ii.
```python
subjects = sorted(set(s['subject_id'] for s in all_sessions))
sub2idx = {s: i for i, s in enumerate(subjects)}
# ...
subject_idx.append(sub2idx[s['subject_id']])
```

iii. The subject ID is embedded in the NWB file metadata. The AI verified 11 unique subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All sessions are processed independently. The session ID is read from the NWB metadata.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    session_id = f['general/session_id'][()].decode()
```

iii. The NWB file organization (one file per session) naturally defines session boundaries.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by `trial_start_sig == 1` for trial starts and `teleport_sig == 1` for trial ends. The AI uses `np.where` to find indices where these signals equal 1, then takes `min(len(starts), len(ends))` trials.

ii.
```python
trial_start_inds = np.where(trial_start_sig == 1)[0]
teleport_inds = np.where(teleport_sig == 1)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
```

iii. The AI identified that `trial_start` and `teleport` behavioral signals mark trial boundaries. The `min` operation handles cases where the counts don't match.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints (`nf < 2`) are skipped. Sessions with fewer than 2 valid trials are skipped entirely. No other trial quality filtering is applied.

ii.
```python
if nf < 2: continue
# ...
if len(neural_trials) < 2:
    print(f"    Skipping: {len(neural_trials)} valid trials"); return None
```

iii. The AI's CONVERSION_NOTES mention no additional trial filtering beyond the minimum frame count. The reference solution uses a 50-timepoint minimum threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) arrays stored in the NWB file under `processing/ophys/Fluorescence` and `processing/ophys/Neuropil`. The `Deconvolved` array in the NWB is not used.

ii.
```python
fluor = f['processing/ophys/Fluorescence']
neu = f['processing/ophys/Neuropil']
# ...
F_list.append(fluor[pname]['data'][:, pcell])
Fneu_list.append(neu[pname]['data'][:, pcell])
```

iii. The AI correctly identified that the NWB `Deconvolved` data is suite2p's own deconvolution and not the signal the paper analyzes. The paper computes its own dF/F from F and Fneu.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F using a simplified version of the paper's pipeline: (1) neuropil subtraction: `F - 0.7 * Fneu`, (2) maximin baseline: minimum filter then maximum filter (window size 300 or trial length, whichever is smaller), (3) dF/F: `(F_corrected - baseline) / |baseline|`, (4) Gaussian smoothing with sigma=2. The AI does NOT perform OASIS deconvolution -- it uses the dF/F directly as the neural signal, with NaN values replaced by 0.

ii.
```python
def compute_dff(F, Fneu, trial_start_inds, teleport_inds):
    f_ = (F.T - NEU_COEF * Fneu.T).astype(np.float64)  # (neurons, timepoints)
    # ...
    flow[:, s:e] = scipy.ndimage.minimum_filter1d(f_[:, s:e], w, axis=-1)
    flow[:, s:e] = scipy.ndimage.maximum_filter1d(flow[:, s:e], w, axis=-1)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    # ...smooth
    dff[:, s:e] = nansmooth(dff[:, s:e], SMOOTH_SIGMA, axis=1)
    return dff

# Usage:
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES state: "Neural signal: dF/F (not deconvolved) - preserves more info for decoding." The AI also noted that the NWB Deconvolved is not from the reference pipeline's dF/F->OASIS path. However, the pipeline is simplified relative to the reference: it omits the initial Gaussian smoothing (sigma=15) before the min/max filter, omits adding back per-trial mean neuropil after subtraction, omits deconvolution via OASIS, and does not handle the keep_teleports distinction for baseline windowing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered using the `iscell` flag from suite2p (manual curation). No interneuron filtering is applied.

ii.
```python
iscell = seg['iscell'][:]
plane_idx = seg['planeIdx'][:]
# ...
pcell = iscell[pmask, 0] == 1
F_list.append(fluor[pname]['data'][:, pcell])
```

iii. The AI's CONVERSION_NOTES state: "Cell curation: suite2p iscell flag (manual curation). No additional filtering." The paper describes an additional step of filtering putative interneurons (cells with dF/F-speed correlation > 0.5), which the AI does not implement.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by simply extracting the data between trial_start and teleport indices. Since the instructions say to align to trial start, no additional offset is applied.

ii.
```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The trial boundaries are defined by `trial_start` and `teleport` signals, so the neural data naturally starts at trial onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at the original imaging rate of ~15.5 Hz. The time bin size is hardcoded as `1000.0 / 15.5078125 = 64.48 ms`.

ii.
```python
TARGET_RATE_HZ = 15.5078125
time_bin_ms = 1000.0 / TARGET_RATE_HZ
```

iii. The AI determined that multi-plane sessions already have data at the per-plane rate (~15.5 Hz), so no resampling is needed. The hardcoded rate matches the observed timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavioral time series.

ii.
```python
frame_timestamps = behav['position/timestamps'][:]
# ...
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
```

iii. The AI uses actual NWB timestamps rather than computing from the frame rate, which is a reasonable approach.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from all timestamps within the trial to get time relative to trial start.

ii.
```python
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
```

iii. Standard time-from-event computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indexing (same number of frames from trial_start to teleport), so no additional alignment is needed.

ii. Both use the same `ts:te` indexing from `trial_start_inds[ti]` to `teleport_inds[ti]`.

iii. The AI verified that behavioral and neural data have the same number of timepoints.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment/data` behavioral time series.

ii.
```python
environment = behav['environment/data'][:]
# ...
env_vals = environment[ts:te]
valid = env_vals[env_vals >= 0]
if len(valid) > 0:
    trial_env[ti] = int(np.round(np.median(valid)))
```

iii. The AI noted that environment can be 0 or 1, matching the ENV1/ENV2 description.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the median of valid (>= 0) environment values within the trial is computed and rounded. This is stored as a per-trial constant.

ii.
```python
valid = env_vals[env_vals >= 0]
if len(valid) > 0:
    trial_env[ti] = int(np.round(np.median(valid)))
```

iii. The median approach handles potential edge effects. Since environment is constant within a trial, this effectively picks the mode.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop counter over trials (the trial index within the session).

ii.
```python
inp[2] = ti  # trial number
```

iii. The AI uses a 0-indexed sequential trial number within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing -- the loop index `ti` is assigned directly as a constant value across all timepoints.

ii.
```python
inp[2] = ti  # trial number
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` and `reward_zone/data` behavioral time series. Reward outcome is determined per trial by checking if any reward timestamps fall within the trial time window AND the reward zone was active during that trial.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][:]
# ...
isreward = np.zeros(n_trials, dtype=np.int64)
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    has_rzone = np.any(reward_zone_raw[ts:te] > 0)
    t_start = frame_timestamps[ts]
    t_end = frame_timestamps[min(te, n_timepoints-1)]
    has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
    if has_reward and has_rzone:
        isreward[ti] = 1
```

iii. The AI followed the reference code's pattern of checking both reward timestamps and reward zone activity. The additional `has_rzone` check mirrors the reference code's `get_trial_types` function.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's `isreward` value is used. For the first trial, it defaults to 0.

ii.
```python
prev_outcome = int(isreward[ti-1]) if ti > 0 else 0
inp[3] = prev_outcome
```

iii. Standard previous-trial lookback with a sensible default for trial 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and reward zone coordinates determined from the NWB file's `identifier` field (scene name). The scene name is parsed to determine which reward zone (A, B, or C) is active for each trial, with a switch at trial 30.

ii.
```python
scene = parse_scene_name(identifier)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
# ...
rz_s, rz_e = rz_coords[ti]
dist = np.where(trial_pos < rz_s, trial_pos - rz_s,
               np.where(trial_pos > rz_e, trial_pos - rz_e, 0.0))
```

iii. The AI parsed the scene/identifier string from the NWB metadata to determine reward zone locations, following the reference code's `get_reward_zones` function. The reward zone coordinates match: A=[80,130], B=[200,250], C=[320,370].

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position (clipped to [0, 450]) to the nearest edge of the current reward zone is computed. Distance is 0 when inside the zone, negative when before the zone, positive when past it.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
dist = np.where(trial_pos < rz_s, trial_pos - rz_s,
               np.where(trial_pos > rz_e, trial_pos - rz_e, 0.0))
```

iii. Standard signed-distance computation from a range.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boolean masking:

ii.
```python
def discretize_distance(d):
    r = np.zeros_like(d, dtype=np.int64)
    r[d < -50] = 0
    r[(d >= -50) & (d < -10)] = 1
    r[(d >= -10) & (d < 0)] = 2
    r[d == 0] = 3
    r[(d > 0) & (d <= 10)] = 4
    r[(d > 10) & (d <= 50)] = 5
    r[d > 50] = 6
    return r
```

iii. The bin edges follow the instructions: `< -50, -50 to -10, -10 to < 0, 0, >0 to +10, +10 to +50, > +50`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices (ts:te) are used for both neural and behavioral data, so alignment is inherent.

ii. Both use `trial_start_inds[ti]` to `teleport_inds[ti]` indexing.

iii. No additional alignment needed since behavioral and neural data share the same timebase.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behav['position/data'][:]
# ...
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
```

iii. Direct extraction of position data from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm range before discretization.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
```

iii. The AI clips position to ensure values stay within the track boundaries.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each:

ii.
```python
def discretize_position(p):
    r = np.zeros_like(p, dtype=np.int64)
    r[p < 90] = 0
    r[(p >= 90) & (p < 180)] = 1
    r[(p >= 180) & (p < 270)] = 2
    r[(p >= 270) & (p < 360)] = 3
    r[p >= 360] = 4
    return r
```

iii. Five equal-sized 90 cm bins spanning the 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data -- no additional alignment needed.

ii. Same `ts:te` indexing.

iii. Inherent alignment from shared timebase.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick/data'][:]
# ...
trial_lick = lick[ts:te]
```

iii. Direct extraction from NWB behavioral data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_bin = (trial_lick > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data -- no additional alignment needed.

ii. Same `ts:te` indexing.

iii. Inherent alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file's `identifier` field, which contains the scene name encoding the reward zone configuration.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene_name(identifier)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
# ...
rz_label_int = {'A': 0, 'B': 1, 'C': 2}.get(rz_labels[ti], 0)
```

iii. The scene name encodes which reward zones are active and when they switch. The AI parsed this following the pattern in the reference code's `get_reward_zones` function.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine which zone (A, B, or C) is active for each trial. For switch sessions (`_to_` in scene name), the zone changes at trial 30. The zone label is mapped to an integer: A=0, B=1, C=2.

ii.
```python
def get_reward_zones_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        pre_zone = parts[0][-1]
        post_zone = parts[1][-1]
        ct = min(change_trial, n_trials)
        rz_coords[:ct] = zone_to_coords[pre_zone]; rz_labels[:ct] = pre_zone
        rz_coords[ct:] = zone_to_coords[post_zone]; rz_labels[ct:] = post_zone
    # ... similar patterns for other scene types
```

iii. The parsing logic handles multiple scene name formats (fixed location, zone-to-zone switch, zone-to-environment switch).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` and `reward_zone/data` behavioral time series.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][:]
# ...
has_rzone = np.any(reward_zone_raw[ts:te] > 0)
has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
if has_reward and has_rzone:
    isreward[ti] = 1
```

iii. The AI uses both reward timestamps and reward zone activity to determine reward outcome, matching the reference code's approach.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A binary per-trial variable: 1 if a reward event occurred within the trial time window AND the reward zone was active during the trial, 0 otherwise. This is constant across all timepoints in the trial.

ii.
```python
out[5] = isreward[ti]
```

iii. The dual condition (reward timestamp + reward zone active) follows the reference code's `get_trial_types` logic.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **NaN/Inf in dF/F**: Replaced with 0 using `np.nan_to_num`.
- **Trial count mismatch**: `min(len(starts), len(ends))` handles cases where trial_start and teleport counts differ.
- **Short trials**: Trials with fewer than 2 frames are skipped.
- **Few-trial sessions**: Sessions with fewer than 2 valid trials are skipped.
- **Baseline window**: Clamped to trial length when trial is shorter than 300 frames.
- **Position clipping**: Values clipped to [0, 450].

ii.
```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
n_trials = min(len(trial_start_inds), len(teleport_inds))
w = min(BASELINE_WINDOW, e - s)
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
```

iii. The AI documented these in CONVERSION_NOTES as edge case handling.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files (I/O bound, reading large arrays with h5py)
2. Computing dF/F (minimum/maximum filters over full neural arrays)
3. Total conversion runs in about 5 minutes for all 152 sessions

ii. N/A

iii. The AI reported ~2s/session processing time and ~5 min total.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` (lines 209-256) iterates over trials sequentially. Some operations like discretization and distance computation could be applied session-wide before splitting into trials. The reward detection loop (lines 188-196) iterates over all trials.

ii. N/A

iii. The variable-length trials make full vectorization difficult without padding.

## 13-c. What processing does the code repeat multiple times?

i. The code does not have a separate survey step like the reference -- each NWB file is loaded and processed only once. However, the reward zone determination via scene name parsing is done per-session when it could potentially be precomputed.

ii. N/A

iii. The AI's approach of processing each file once is efficient.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The position values are clipped to [0, 450] before discretization. This clipping modifies values that would otherwise fall into the correct bins anyway (the edge bins are open-ended in the reference). The dF/F computation processes all cells including potential interneurons that the reference pipeline would have excluded.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
```

iii. The clipping is unnecessary since the discretization bins handle out-of-range values.
