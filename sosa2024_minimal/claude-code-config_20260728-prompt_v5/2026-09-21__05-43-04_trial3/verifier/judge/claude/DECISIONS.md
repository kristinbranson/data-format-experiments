# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from subdirectories of the data directory matching `sub-*`. It uses `h5py` directly (not `pynwb`) to read the HDF5 files. Subjects are defined by a hardcoded `SESSIONS_DICT` dictionary mapping mouse IDs to session metadata. For each subject, the AI lists all `.nwb` files in the subject directory, extracts the session day from the filename, and matches it to the corresponding entry in `SESSIONS_DICT` to get scene information. Sessions where no matching scene is found are skipped.

ii.
```python
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))

for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, f'sub-{subj}')
    if not os.path.isdir(subj_dir):
        print(f"Warning: directory {subj_dir} not found, skipping")
        continue
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for nwb_file in nwb_files:
        ses_str = nwb_file.split('_ses-')[1].split('_')[0]
        exp_day = int(ses_str)
        # Find matching scene from sessions dict
        scene = None
        for entry in SESSIONS_DICT[subj]:
            if entry['exp_day'] == exp_day:
                scene = entry['scene']
                break
```

Loading NWB data with h5py:
```python
f = h5py.File(nwb_path, 'r')
dec_group = f['processing/ophys/Deconvolved']
fl_group = f['processing/ophys/Fluorescence']
```

iii. The agent explored the NWB file structure directly with h5py and discovered the data organization. Subjects were identified from the SESSIONS_DICT which was derived from the reference code's sessions_dict.py. The agent checked that all 11 switch-task mice were included.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hardcoded `SESSIONS_DICT` dictionary keys (m3, m4, m7, m11-m15, m17-m19), sorted numerically. This is cross-referenced against actual sub-* directories on disk.

ii.
```python
SESSIONS_DICT = {
    'm3': [...],
    'm4': [...],
    ...
}
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))
```

iii. The agent identified subjects from the reference code's sessions_dict.py and verified they matched the directories in the data folder.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session number (experiment day) is parsed from the filename pattern `sub-<subject>_ses-<day>_behavior+ophys.nwb`. Each session is matched to a scene entry in SESSIONS_DICT by experiment day.

ii.
```python
for nwb_file in nwb_files:
    ses_str = nwb_file.split('_ses-')[1].split('_')[0]
    exp_day = int(ses_str)
    scene = None
    for entry in SESSIONS_DICT[subj]:
        if entry['exp_day'] == exp_day:
            scene = entry['scene']
            break
```

iii. The agent identified the filename convention and that each file represents one session/experiment day.

## 1-d. How are the data split into trials?

i. Trials are identified by finding all frames where `trial_start_signal > 0` (trial starts) and all frames where `teleport_signal > 0` (trial ends). The number of trials is set to the minimum of these two arrays' lengths, and the arrays are truncated to this length.

ii.
```python
trial_start_idx = np.where(trial_start_signal > 0)[0]
teleport_idx = np.where(teleport_signal > 0)[0]

n_trials = min(len(trial_start_idx), len(teleport_idx))
trial_start_idx = trial_start_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
```

iii. The agent identified the trial_start and teleport behavioral signals as markers for trial boundaries. It noted that teleport marks the inter-trial interval.

## 1-e. How are trials filtered based on quality controls?

i. Trials are skipped if `t_end <= t_start` or if the number of timepoints is less than 2. Sessions with fewer than 2 valid trials are also skipped.

ii.
```python
for t in range(n_trials):
    t_start = trial_start_idx[t]
    t_end = teleport_idx[t]
    if t_end <= t_start:
        continue
    neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
    n_tp = neural.shape[1]
    if n_tp < 2:
        continue
```

```python
if neural is None or len(neural) < 2:
    print(f"  Skipping {subj} day {exp_day}: insufficient data")
    continue
```

iii. The agent applied minimal filtering: trials must have at least 2 timepoints for the decoder to work.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the pre-computed `Deconvolved` field in the NWB files, which contains suite2p's OASIS deconvolution output. The AI explicitly chose this over recomputing from raw Fluorescence and Neuropil.

ii.
```python
dec_group = f['processing/ophys/Deconvolved']
planes = sorted([k for k in dec_group.keys() if k.startswith('plane')])
dec_planes = [dec_group[p]['data'][:] for p in planes]
deconvolved = np.concatenate(dec_planes, axis=1)
...
deconv_filtered = deconvolved[:, final_cell_indices]
...
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. From the trajectory, the agent inspected the NWB Deconvolved data and found it had the expected sparsity pattern (72% zeros) consistent with OASIS deconvolution. The agent decided to use this pre-computed data directly rather than recomputing from raw fluorescence. The agent stated: "Use NWB Deconvolved data directly as the neural activity."

## 2-b. How is the `neural` data processed?

i. No processing is applied to the neural data. The Deconvolved field is used as-is after cell filtering (iscell and interneuron exclusion). The data is cast to float32 and transposed to (n_neurons, n_timepoints).

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. The agent believed the Deconvolved data was already the result of the full processing pipeline (neuropil subtraction, dF/F, OASIS deconvolution).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) suite2p manual curation (`iscell` field), and (2) exclusion of putative interneurons identified by Pearson correlation > 0.5 between dF/F and running speed. For the interneuron detection, a simplified dF/F computation is performed from raw Fluorescence and Neuropil data.

ii.
```python
# Step 1: iscell filtering
cell_mask = iscell[:, 0].astype(bool)
cell_indices = np.where(cell_mask)[0]
F_cells = fluorescence[:, cell_indices]
Fneu_cells = neuropil_data[:, cell_indices]

# Compute dF/F for interneuron detection
dff = compute_dff_for_interneuron_detection(
    F_cells, Fneu_cells, trial_start_idx, teleport_idx, frame_rate)

# Step 2: interneuron filtering
is_int = detect_interneurons(dff, speed, threshold=0.5)
final_cell_indices = cell_indices[~is_int]
```

iii. The agent identified both filtering steps from the paper methods. The dF/F for interneuron detection is computed separately since the Deconvolved data is used for the neural activity itself.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data is aligned to trial start. Each trial's neural data starts at the `trial_start` frame index, so no additional alignment processing is needed beyond slicing.

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. The instructions specify alignment to trial start, which is the natural boundary from the trial segmentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate is used (~15.5 Hz, ~64.5 ms per bin). No temporal rebinning is applied. The time bin size is computed as 1000 / median_frame_rate ms.

ii.
```python
frame_rates = [info['frame_rate'] for info in all_session_info]
median_frame_rate = np.median(frame_rates)
time_bin_ms = 1000.0 / median_frame_rate
```

iii. The agent determined that the native frame rate was consistent across recordings and used it directly.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is computed from the frame index and the frame rate, NOT from actual timestamps. It is computed as `np.arange(n_tp) * dt` where `dt = 1.0 / frame_rate`.

ii.
```python
dt = 1.0 / frame_rate  # time per frame in seconds
...
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. The agent used the frame rate to compute uniform time steps rather than extracting actual timestamps from the NWB file.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, a time array is created as frame indices (0, 1, 2, ...) multiplied by the frame duration (1/frame_rate). The first timepoint is always 0.

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. This gives uniformly spaced time values starting from 0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both the time array and neural data share the same frame indices within each trial, so they are inherently aligned (same number of timepoints, same ordering).

ii.
```python
# Both use t_start:t_end for the trial
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
n_tp = neural.shape[1]
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. Since both are derived from the same frame count, alignment is guaranteed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The environment type is derived from the scene name in `SESSIONS_DICT`, NOT from the NWB `environment` behavioral time series. The scene name is parsed to extract the environment number (1 or 2).

ii.
```python
def get_env_for_trial(scene_info, trial_idx):
    env_before, _, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return env_after if env_after is not None else env_before
    return env_before

# In process_session:
env = get_env_for_trial(scene, t)
env_binary = 0 if env == 1 else 1
```

iii. The agent parsed scene names to determine environment identity, reasoning that scene metadata provides reliable ground truth for the experimental design.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed using `parse_scene()` which handles three formats: environment switch scenes (e.g., 'Env1_C_to_Env2_B'), within-environment switches (e.g., 'Env1_LocationA_to_B'), and non-switch scenes (e.g., 'Env1_LocationA'). The environment number is extracted and mapped to binary (ENV1=0, ENV2=1). On switch days, trials before index 30 use the "before" environment and trials 30+ use the "after" environment.

ii.
```python
def parse_scene(scene):
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        before = parts[0]
        after = parts[1]
        env_before = int(before[3])
        zone_before = before.split('_')[1]
        env_after = int(after[3])
        zone_after = after.split('_')[1]
        return env_before, zone_before, env_after, zone_after
    ...
```

iii. The agent reasoned that scene names encode the experimental design including environment switches.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop counter over trials within a session (0, 1, 2, ...).

ii.
```python
for t in range(n_trials):
    ...
    input_data = np.stack([
        ...
        np.full(n_tp, t, dtype=np.float32),
        ...
    ], axis=0)
```

iii. The agent used the sequential trial index within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing. The loop index `t` is broadcast to all timepoints in the trial.

ii.
```python
np.full(n_tp, t, dtype=np.float32)
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB behavioral data. Reward event timestamps are compared against trial time windows defined by the behavior timestamps.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
...
t_start_time = timestamps[t_start]
t_end_time = timestamps[t_end - 1]
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
```

iii. The agent identified that reward events have separate timestamps from the behavioral sampling rate and used temporal matching.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward event timestamp falls within the trial's time window. The `prev_rewarded` variable tracks the previous trial's outcome. For the first trial, it defaults to 0.

ii.
```python
prev_rewarded = 0  # For the first trial
for t in range(n_trials):
    ...
    rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
    ...
    input_data = np.stack([
        ...
        np.full(n_tp, prev_rewarded, dtype=np.float32),
    ], axis=0)
    ...
    prev_rewarded = rewarded
```

iii. The instructions specify binary encoding (omitted=0, rewarded=1).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries. The reward zone identity is determined from the session's scene name in SESSIONS_DICT (not from the NWB reward_zone field). Reward zone boundaries are hardcoded: A=(80,130), B=(200,250), C=(320,370).

ii.
```python
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
...
zone_letter = get_reward_zone_for_trial(scene, t)
zone_start, zone_end = REWARD_ZONES[zone_letter]
dist = compute_distance_to_reward_zone(pos_trial, zone_start, zone_end)
```

iii. The agent determined reward zones from scene names because the NWB reward_zone field was noisy and failed on omission trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance from position to the reward zone is computed: negative if before the zone, 0 if inside, positive if past. Position is first clamped to [0, 450].

ii.
```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    dist = np.zeros_like(position, dtype=float)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    dist[before] = position[before] - zone_start
    dist[inside] = 0.0
    dist[after] = position[after] - zone_end
    return dist
```

iii. Standard signed distance computation consistent with the paper's reward-relative framework.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignments matching the instruction bins: <-50, -50 to -10, -10 to 0, 0, >0 to 10, 10 to 50, >50.

ii.
```python
def discretize_distance(dist):
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. Directly implements the bin specification from the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Both position and neural data are indexed by the same frame range (t_start:t_end) within each trial, so alignment is inherent.

ii.
```python
pos_trial = position[t_start:t_end]
neural = deconv_filtered[t_start:t_end, :].T
```

iii. Same frame indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series in the NWB file.

ii.
```python
position = beh['position/data'][:]
...
pos_trial = position[t_start:t_end]
pos_trial = np.clip(pos_trial, 0, 450)
```

iii. The position variable directly records the animal's location in the virtual corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clamped to [0, 450] using `np.clip`, then discretized into 5 equal bins of 90 cm each.

ii.
```python
pos_trial = np.clip(pos_trial, 0, 450)
...
def discretize_position(position):
    out = np.zeros(len(position), dtype=np.int64)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position < 360)] = 3
    out[position >= 360] = 4
    return out
```

iii. The 450 cm track is divided into 5 bins of 90 cm each.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins: 0-90, 90-180, 180-270, 270-360, 360-450 cm using explicit conditionals.

ii.
```python
def discretize_position(position):
    out = np.zeros(len(position), dtype=np.int64)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position < 360)] = 3
    out[position >= 360] = 4
    return out
```

iii. Matches the 5 equal-sized bins specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indexing as neural data within each trial.

ii.
```python
pos_trial = position[t_start:t_end]
neural = deconv_filtered[t_start:t_end, :].T
```

iii. Inherent alignment through shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series in the NWB file.

ii.
```python
lick = beh['lick/data'][:]
...
lick_trial = lick[t_start:t_end]
```

iii. The lick variable records lick events at each frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_binary = (lick_trial > 0).astype(np.int64)
```

iii. Instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indexing as neural data within each trial.

ii.
```python
lick_trial = lick[t_start:t_end]
```

iii. Inherent alignment through shared frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the scene name in SESSIONS_DICT, not from the NWB data. The scene name encodes the reward zone letter(s) and whether a switch occurs on that day.

ii.
```python
def get_reward_zone_for_trial(scene_info, trial_idx):
    env_before, zone_before, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return zone_after
    return zone_before
...
zone_letter = get_reward_zone_for_trial(scene, t)
zone_idx = {'A': 0, 'B': 1, 'C': 2}[zone_letter]
```

iii. The agent found that the NWB reward_zone field was unreliable (noisy values 0-6 that changed within trials) and chose to use scene metadata as a more reliable source.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to extract zone letter(s). On switch days (scenes with "_to_" in the name), trials before index 30 use the pre-switch zone and trials 30+ use the post-switch zone. The zone letter is then mapped to an integer: A=0, B=1, C=2.

ii.
```python
SWITCH_TRIAL = 30
...
def get_reward_zone_for_trial(scene_info, trial_idx):
    env_before, zone_before, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return zone_after
    return zone_before
```

iii. The switch at trial 30 matches the paper's experimental design where reward zone switches occur at trial 30 on switch days.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB behavioral data.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
...
t_start_time = timestamps[t_start]
t_end_time = timestamps[t_end - 1]
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
```

iii. Reward events are recorded with separate timestamps from the behavioral sampling.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event timestamp falls within the trial's time window (from trial start to trial end timestamps). Output is 1 if rewarded, 0 otherwise. The value is broadcast to all timepoints in the trial.

ii.
```python
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
...
output = np.stack([
    ...
    np.full(n_tp, rewarded, dtype=np.int64),
], axis=0)
```

iii. Binary per-trial classification matching the instruction specification.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive checks:
- **Neural/behavior frame mismatch**: Data is cropped to the minimum of neural and behavioral frame counts.
- **Short trials**: Trials with `t_end <= t_start` or fewer than 2 frames are skipped.
- **Session-level filtering**: Sessions with fewer than 2 valid trials are skipped entirely.
- **Missing scene metadata**: Sessions without matching entries in SESSIONS_DICT are skipped with a warning.
- **Position clamping**: Position values are clamped to [0, 450] to avoid out-of-range values.

ii.
```python
n_frames = min(n_frames_neural, n_frames_behav)
...
if t_end <= t_start:
    continue
if n_tp < 2:
    continue
...
if neural is None or len(neural) < 2:
    print(f"  Skipping {subj} day {exp_day}: insufficient data")
    continue
...
pos_trial = np.clip(pos_trial, 0, 450)
```

iii. Defensive handling discovered through data exploration during development.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files with h5py** and reading large arrays (neural data per plane)
2. **dF/F computation for interneuron detection** (`compute_dff_for_interneuron_detection`), which involves per-cell Gaussian smoothing, minimum/maximum filtering per trial
3. **Interneuron correlation computation** (correlating each cell's dF/F with speed)

ii. N/A

iii. Loading NWB files is I/O bound; dF/F computation and correlation are compute-bound.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- The dF/F smoothing loop iterates per-cell (`for c in range(n_cells)`) for Gaussian filtering, which could be done as a single vectorized call with `axis` parameter
- The interneuron detection loop correlates each cell individually, which could be vectorized
- The trial-level processing loop iterates over trials for discretization, which could be done on full session arrays

ii.
```python
# Per-cell smoothing loop
for c in range(n_cells):
    bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)
```

```python
# Per-cell correlation loop
for c in range(n_cells):
    r = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
```

iii. These are performance bottlenecks but functionally correct.

## 13-c. What processing does the code repeat multiple times?

i. The code loads each NWB file only once per session, so there is no repeated file loading. However, the `parse_scene()` function is called multiple times per trial (once for reward zone and once for environment), parsing the same scene string repeatedly.

ii. N/A

iii. The repeated scene parsing is trivial computationally.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation for interneuron detection is extensive (neuropil subtraction, baseline estimation, smoothing) but is only used to compute correlations with speed -- the dF/F itself is discarded and the Deconvolved data is used for neural activity. Additionally, position clamping to [0, 450] modifies position values that would otherwise fall naturally outside this range, potentially altering distance-to-reward-zone calculations.

ii. N/A

iii. The dF/F computation is necessary for the interneuron detection pipeline but its output is not used for the final neural data.
