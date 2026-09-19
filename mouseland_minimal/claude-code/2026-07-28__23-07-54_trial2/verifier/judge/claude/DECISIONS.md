# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded from three subdirectories: `beh/` (behavior), `spk/` (deconvolved calcium traces), and `retinotopy/` (visual area assignments). The master index `Imaging_Exp_info.npy` is loaded first to enumerate all sessions across experiment types. For each unique `(mname, datexp, blk)` session, the behavior is loaded from `Beh_{exp_type}.npy`, neural data from `{mname}_{datexp}_{blk}_neural_data.npy`, and retinotopy from `{mname}_{datexp}_trans.npz`. Behavior files are cached by experiment type to avoid redundant loading.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
spk = load_spk(mname, datexp, blk, DATA_ROOT)
# load_spk does:
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
iarea = load_retino(mname, datexp, DATA_ROOT)
# load_retino does:
dtrans = np.load(ret_path, allow_pickle=True)
return dtrans['iarea']
```

iii. The agent explored the data directory structure, examined `Imaging_Exp_info.npy`, and determined how sessions map to neural and behavior files. Behavior files are cached per experiment type to avoid repeated loading.

## 1-b. How are the data split into subjects?

i. Subject (mouse) names are extracted from the `mname` field in the session info. All unique mouse names are sorted and indexed. There are 19 mice total.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx_all.append(subject_to_idx[mname])
```

iii. The mouse name is directly available from the session metadata; no derivation is needed.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` combination, which corresponds to one neural recording file. When the same recording appears under multiple experiment types, the agent picks the entry with the most non-NaN `stim_id` values rather than simply keeping the first one encountered.

ii.
```python
for exp_type in exp_info:
    for db in exp_info[exp_type]:
        key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
        # ...
        if key not in sessions or n_stim > sessions[key]['n_stim']:
            sessions[key] = { ... }
```

iii. The agent noted that some sessions appear under multiple experiment types (e.g., naive_test1, naive_test2) and chose to keep the one with the most stim_id entries, reasoning that this provides the best behavioral annotation.

## 1-d. How are the data split into trials?

i. Trials are defined by the `StartFr` array in the behavior data. Each trial is extracted as a **fixed-length window of 32 frames** starting at `int(np.round(StartFr[trial]))`. This differs from the reference, which uses variable-length trials based on `ft_trInd` and `ft_CorrSpc`.

ii.
```python
N_TIMEPOINTS = 32
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The agent reasoned that the decoder needs fixed-size arrays and chose 32 frames based on the estimated corridor traversal time at 60 cm/s VR speed with 3.17 Hz imaging: 6m / 0.6 m/s / 3.17 Hz ≈ 32 frames.

## 1-e. How are trials filtered based on quality controls?

i. A trial is filtered out if the 32-frame window extends beyond the neural recording (`start_fr < 0` or `end_fr > n_frames`). Sessions with fewer than 10 valid trials are skipped entirely. There is no filtering based on trial length outliers (unlike the reference which drops trials beyond the 99th percentile in length).

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    continue
if len(valid_neural) < MIN_TRIALS:
    print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
    continue
```

iii. The agent used out-of-bounds checking as the primary filter. The fixed 32-frame window inherently avoids extremely long trials since all trials are the same length.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`{session_id}_neural_data.npy`), which is a list of arrays per imaging plane, concatenated along axis 0. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
iarea = load_retino(mname, datexp, DATA_ROOT)
```

iii. The agent identified the deconvolved calcium traces as the neural data source, matching the paper's methods.

## 2-b. How is the `neural` data processed?

i. The traces are not further processed beyond filtering to visual cortex neurons and slicing to the 32-frame trial window. Data is stored as float32 (the reference uses float16).

ii.
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The agent noted the data is already deconvolved and no additional processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area: only neurons in V1, mHV, lHV, or aHV are kept (same mapping as the reference). Sessions with fewer than 10 visual cortex neurons are skipped.

ii.
```python
def get_brain_region_idx(iarea):
    region_idx = np.full(len(iarea), -1, dtype=int)
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return region_idx
valid_neurons = region_idx >= 0
spk_filtered = spk[valid_neurons]
```

iii. The mapping was taken from `code/utils.py`'s `neu_area_ID` function. The reference applies the same filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start) by using `StartFr` as the start frame. A fixed window of 32 frames is taken from `StartFr`. This differs from the reference which uses variable-length windows based on the frames labeled as inside the corridor texture (`ft_CorrSpc`).

ii.
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints  # N_TIMEPOINTS = 32
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The agent chose a fixed 32-frame window to standardize trial length for the decoder, estimating that 32 frames covers the full corridor traversal.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 1000/3.17 ≈ 315.5 ms, matching the imaging frame rate. No temporal rebinning is applied; each imaging frame is one time bin.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS
```

iii. The imaging frame rate is the finest temporal resolution available, and all behavioral data is already aligned to this grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the sound cue frame for each trial) and `StartFr` (the trial start frame).

ii.
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The agent computed the sound cue position relative to the trial start in frames, then converted each timepoint's distance to the cue into seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The relative cue frame is computed as `SoundFr[trial] - StartFr[trial]`. For each timepoint in the 32-frame window, the time to the cue is `(frame_index - sound_fr_rel) / FS` in seconds. This gives negative values before the cue and positive after — note this is the **opposite sign convention** from the reference, which computes `cue_time - frame_time` (positive before cue, negative after).

ii.
```python
time_to_cue = np.arange(n_timepoints) - sound_fr_rel  # negative before cue, positive after
time_to_cue_sec = time_to_cue / FS
```

iii. The agent reasoned about the sound cue timing relative to each frame. The sign convention differs from the reference.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The input is computed over the same 32-frame window as the neural data, using frame indices 0 through 31 from the trial start.

ii.
```python
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
```

iii. The alignment is inherent since both neural and input data use the same fixed frame window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` (date string) field of each session, parsed as a calendar date.

ii.
```python
def compute_day_of_training(mname, datexp, all_sessions):
    from datetime import datetime
    mouse_dates = []
    for k, s in all_sessions.items():
        if s['mname'] == mname:
            date = datetime.strptime(s['datexp'], '%Y_%m_%d')
            mouse_dates.append(date)
    mouse_dates.sort()
    current_date = datetime.strptime(datexp, '%Y_%m_%d')
    day_idx = (current_date - mouse_dates[0]).days
    return day_idx
```

iii. The agent computed the calendar day difference from the first recorded session for each mouse, rather than counting the ordinal position of the session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is computed as the **calendar day difference** from the mouse's first session date. This differs from the reference, which counts the ordinal session number (0, 1, 2, ...). For example, if a mouse has sessions on days 1, 3, and 7, the AI gives values 0, 2, 6 while the reference gives 0, 1, 2.

ii.
```python
day_idx = (current_date - mouse_dates[0]).days
input_trials[i][1, :] = float(day)
```

iii. The agent chose calendar days because the `days` field in `exp_info` was only available for some experiment types, while `sess#` was too coarse. The datetime approach provides a continuous measure.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is computed purely from the frame index within the trial and the frame rate, not from any raw data variable.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. Since trials start at frame 0 of the window, the time since trial start is simply the frame index divided by the frame rate.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A linear ramp from 0 to ~10s: `np.arange(32) / 3.17`. This differs from the reference, which computes the time difference between each frame's timestamp (`ft`) and the interpolated corridor entry time (`StartFr`), giving slightly different values due to non-uniform frame timing.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. The agent assumed uniform frame spacing, which is approximately but not exactly true.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed over the same 32-frame window, using indices 0-31 from the trial start frame.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. Alignment is inherent since both use the same frame window.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial was in a rewarded corridor.

ii.
```python
is_rew = beh['isRew']
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. Directly available from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing; the boolean `isRew` value is cast to float (0.0 or 1.0) and broadcast across all timepoints of the trial.

ii.
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. No processing needed; the variable is already per-trial binary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which gives the texture name for each trial, and `UniqWalls`, which lists all unique wall names in the session.

ii.
```python
wall_name = wall_names[trial]
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx
```

iii. The agent used `WallName` directly for each trial's stimulus identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent uses **individual texture names** (e.g., `circle1`, `circle2`, `leaf1`, `leaf2`) as separate categories rather than grouping them into the four base textures (circle, leaf, rock, wood) as the reference does. This results in ~15 stimulus categories instead of 4. The categories are the sorted set of all unique wall names across all sessions.

ii.
```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
stim_to_idx[str(w)] = all_stim_sorted.index(str(w))
output_values = [
    [str(s) for s in all_stim_sorted],  # stimulus category names
    ...
]
```

iii. The agent collected all unique wall names from `UniqWalls` across all sessions and sorted them alphabetically. The task says "Visual stimulus category. e.g. circle, leaf, etc." which suggests grouping into base textures, but the agent kept individual variants.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of each lick) and `LickTrind` (trial index for each lick).

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_mask = lick_trinds == trial
trial_lick_frs = lick_frs[trial_lick_mask]
```

iii. The agent identified `LickFr` and `LickTrind` as the lick data sources.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks are identified by matching `LickTrind == trial`, then each lick's frame position relative to the trial start is computed. If it falls within the 32-frame window, that frame is set to 1. This differs from the reference, which creates a session-wide lick array from `LickFr` and then indexes it by trial frames.

ii.
```python
lick_binary = np.zeros(n_timepoints, dtype=float)
trial_lick_mask = lick_trinds == trial
if trial_lick_mask.any():
    trial_lick_frs = lick_frs[trial_lick_mask]
    for lf in trial_lick_frs:
        fr_idx = int(np.round(lf)) - start_fr
        if 0 <= fr_idx < n_timepoints:
            lick_binary[fr_idx] = 1.0
```

iii. The agent used `LickTrind` to filter licks per trial, while the reference uses `LickFr` directly as indices into a session-wide array.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frame indices are converted relative to the trial start frame and placed within the same 32-frame window used for neural data.

ii.
```python
fr_idx = int(np.round(lf)) - start_fr
if 0 <= fr_idx < n_timepoints:
    lick_binary[fr_idx] = 1.0
```

iii. Alignment is achieved by using the same frame window as the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[start_fr:end_fr]
```

iii. Directly from the frame-level position data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is discretized into 4 bins using boundaries at 10, 20, 30 dm (equivalent to 1m, 2m, 3m). The `np.digitize` function is used, which gives values 0-3.

ii.
```python
def discretize_position(pos_values, n_bins=4):
    boundaries = [10, 20, 30]  # in decimeters
    binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
    return binned
```

iii. The task requires 4 equal-length 1m bins for the 4m texture corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Using `np.digitize` with boundaries [10, 20, 30] in decimeters. Values in [0,10) → bin 0, [10,20) → bin 1, [20,30) → bin 2, [30,∞) → bin 3. This is functionally similar to the reference's `np.clip(ft_Pos // 10, 0, 3)` but handles the grey space differently: positions above 40 dm get bin 3 (same as 30-40 dm), matching the reference behavior due to the clip.

ii.
```python
boundaries = [10, 20, 30]
binned = np.digitize(pos_values, boundaries)
```

iii. The 4-bin discretization matches the task requirements.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced from `ft_Pos` using the same `start_fr:end_fr` range as the neural data.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
```

iii. Same frame window ensures alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_speed[start_fr:end_fr]
```

iii. Directly from the frame-level speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using **global quartile boundaries** computed from `ft_RunSpeed` across all sessions' behavior data. The boundaries are computed using `np.percentile` at [25, 50, 75] and applied using `np.digitize`. This differs from the reference, which uses **per-session rank-based quartiles** (ensuring exactly 25% of frames per bin within each session).

ii.
```python
all_speeds = []
for key, sess_info in sessions.items():
    speeds = beh['ft_RunSpeed']
    valid_speed = speeds[~np.isnan(speeds)]
    all_speeds.append(valid_speed)
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
# Then per trial:
speed_binned = np.digitize(trial_speed, speed_quartiles)
```

iii. The agent computed global quartiles for consistent discretization across sessions, but this means individual sessions may not have exactly 25% of data in each bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with global quartile boundaries. Values below Q25 → bin 0, Q25-Q50 → bin 1, Q50-Q75 → bin 2, above Q75 → bin 3. The global boundaries also include all behavior frames, not just the kept trial frames.

ii.
```python
speed_binned = np.digitize(trial_speed, speed_quartiles)
```

iii. The global boundaries were chosen for consistency across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sliced from `ft_RunSpeed` using the same `start_fr:end_fr` range as the neural data.

ii.
```python
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
```

iii. Same frame window ensures alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: trials where the 32-frame window extends beyond the neural recording are marked invalid; sessions where neural or retinotopy files fail to load are skipped; sessions with fewer than 10 visual cortex neurons or 10 valid trials are skipped. However, unlike the reference, the AI does not explicitly handle behavior data extending past imaging frames (truncating to `n_frames`), because the fixed 32-frame window and bounds check implicitly handles this.

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    continue
if valid_neurons.sum() < 10:
    print(f"  Skipping: only {valid_neurons.sum()} visual cortex neurons")
    continue
```

iii. The agent wrapped file loading in try/except blocks and checked for minimum data requirements before processing.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (`{session_id}_neural_data.npy`), which are large files requiring disk I/O and memory allocation. This is the same bottleneck as in the reference.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The agent encountered significant memory issues during full dataset conversion, reaching 237 GB at session 47/89.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick processing loop iterates over individual lick frames per trial rather than vectorizing:
```python
for lf in trial_lick_frs:
    fr_idx = int(np.round(lf)) - start_fr
    if 0 <= fr_idx < n_timepoints:
        lick_binary[fr_idx] = 1.0
```
This could be vectorized using array indexing as the reference does.

ii. See code snippet above.

iii. The reference vectorizes this by creating a session-wide lick array and indexing it directly.

## 12-c. What processing does the code repeat multiple times?

i. Speed quartile computation: the code first computes per-trial speed quartiles in `extract_trial_data` (via `discretize_speed`), then overwrites them with global quartile values in the main loop. The initial per-trial computation is wasted work.

ii.
```python
# In extract_trial_data:
speed_binned = discretize_speed(trial_speed, n_bins=4)  # per-trial, overwritten later
# In main loop:
speed_binned = np.digitize(trial_speed, speed_quartiles)  # global, overwrites above
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The code also iterates over all sessions to collect stimulus names and speed values before processing, adding extra passes over the behavior data.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `stim_id` counts (`n_stim`) for session deduplication, loads `ft_move` which is never used, and loads `UniqWalls` for building stimulus categories. The initial per-trial speed discretization is also discarded. The `stim_id` field from `exp_info` entries is loaded but only used for counting, not for actual stimulus mapping.

ii.
```python
ft_move = beh['ft_move'][:n_frames]  # loaded but never used
stim_id = beh.get('stim_id', None)   # loaded but not used for mapping
uniq_walls = beh['UniqWalls']         # used only for building global stim list
```

iii. These are remnants of the agent's exploration of the data structure.
