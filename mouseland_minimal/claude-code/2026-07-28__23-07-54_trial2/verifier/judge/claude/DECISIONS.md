# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads everything from three subdirectories under `data/`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. It reads `beh/Imaging_Exp_info.npy` as a master index, then builds a session list. Behavior files (`Beh_<exp_type>.npy`) are cached by experiment type. For each session, spike data and retinotopy data are loaded separately.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

# Behavior cache
beh_cache[exp_type] = np.load(
    os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()

# Spike data
spk = load_spk(mname, datexp, blk, DATA_ROOT)
# -> np.concatenate([s for s in spk_data['spks']], axis=0)

# Retinotopy
iarea = load_retino(mname, datexp, DATA_ROOT)
# -> dtrans['iarea']
```

iii. The AI loads data in the same general structure as the reference code, using the experiment info index to discover sessions and then loading behavior, spike, and retinotopy data for each.

## 1-b. How are the data split into subjects (mice)?

i. The mouse name is extracted from `mname` in the experiment info entries. All unique subjects are collected and sorted, and a `subject_to_idx` mapping is created.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The subjects are directly available from the experiment metadata.

## 1-c. How are the data split into sessions?

i. A session is identified by `mname_datexp_blk`. When the same session appears under multiple experiment types, the AI picks the one with the "most stimuli" (highest count of non-NaN stim_id values), rather than simply keeping the first occurrence.

ii.
```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
stim_id = db.get('stim_id', np.array([]))
n_stim = int(np.sum(~np.isnan(stim_id.astype(float))))

if key not in sessions or n_stim > sessions[key]['n_stim']:
    sessions[key] = { ... }
```

iii. The AI chose to pick the "best" behavior file per session based on stimulus count, rather than keeping the first encountered entry.

## 1-d. How are the data split into trials?

i. The AI defines trials using `StartFr` (corridor entry frame) and takes a fixed window of `N_TIMEPOINTS=32` frames starting from `int(np.round(StartFr))`. It does NOT use `ft_trInd` or `ft_CorrSpc` to identify which frames belong to each trial.

ii.
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints

# Check bounds
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    ...
    continue

trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The AI treats trials as fixed-length windows from the start frame, rather than using the data's own trial/corridor-space labeling.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two filters: (1) sessions with fewer than `MIN_TRIALS=10` trials are skipped entirely, and (2) individual trials where `start_fr + N_TIMEPOINTS > n_frames` are marked invalid.

ii.
```python
MIN_TRIALS = 10

if beh['ntrials'] < MIN_TRIALS:
    print(f"  Skipping: only {beh['ntrials']} trials (< {MIN_TRIALS})")
    continue

# Per-trial check
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
```

iii. The MIN_TRIALS=10 threshold is introduced by the AI beyond what the reference code does.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike data files (`spk/<session_id>_neural_data.npy`), which contains deconvolved calcium traces as a list of arrays per imaging plane. These are concatenated across planes. The visual area assignment comes from `iarea` in the retinotopy files.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)

iarea = load_retino(mname, datexp, DATA_ROOT)
```

iii. Same source as the reference.

## 2-b. How is the `neural` data processed?

i. The traces are not further processed beyond slicing. They are stored as float32 (not float16 as in the reference). Since the AI uses a fixed window from StartFr, there is no padding of short trials - all trials are exactly 32 frames or excluded.

ii.
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)  # (n_neurons, n_timepoints)
```

iii. The AI chose float32 for precision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by brain region: only neurons in V1, mHV, lHV, or aHV (using the same iarea codes as the reference) are kept. Additionally, sessions with fewer than 10 valid visual cortex neurons are skipped.

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
if valid_neurons.sum() < 10:
    continue
spk_filtered = spk[valid_neurons]
```

iii. The brain region mapping matches the reference. The min-10-neuron threshold is an additional filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns to corridor entry (StartFr), taking a fixed window of 32 frames starting at `int(np.round(StartFr))`. Unlike the reference, it does not use `ft_CorrSpc` to restrict to corridor-space frames, and does not use variable-length windows with padding.

ii.
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The AI uses StartFr as the alignment event but treats frames as contiguous from that point, which may include gray-space or inter-trial frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is one imaging frame at 3.17 Hz (~315.5 ms). No rebinning is applied.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms per frame
N_TIMEPOINTS = 32
```

iii. Same as reference - the imaging frame rate is used directly.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (frame number of the sound cue) and `StartFr` (trial start frame). Frame indices are used directly rather than converting to actual timestamps.

ii.
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The AI computes relative frame positions and converts to seconds by dividing by the frame rate.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound cue frame is made relative to the trial start frame, then the time-to-cue for each time bin is `(bin_index - relative_sound_frame) / frame_rate`. This gives negative values before the cue and positive values after.

ii.
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The AI computes time relative to the sound cue using frame indices divided by frame rate, rather than using actual timestamps from `ft`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same frame indices (0 to N_TIMEPOINTS-1 relative to StartFr) as the neural data window.

ii.
```python
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
```

iii. The alignment is inherent since both neural and input data use the same frame indexing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (date string) for each session, parsed into actual dates to compute calendar days since the first session for each mouse.

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

iii. The AI uses actual calendar day differences, which can result in large gaps between sessions (e.g., 0, 7, 9, 20, 21, 22).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI parses dates, sorts them per mouse, and computes the difference in calendar days from the earliest session. This is then broadcast across all time bins of each trial. The reference code instead counts the ordinal session number (0, 1, 2, 3, ...).

ii.
```python
day_idx = (current_date - mouse_dates[0]).days
# ...
input_trials[i][1, :] = float(day)
```

iii. Calendar days vs session count is a meaningful difference in how "day of training" is interpreted.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived purely from frame indices and the frame rate constant, not from actual timestamp data.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. Since the window starts at StartFr, the time since trial start starts at 0 by construction.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each time bin is simply `bin_index / frame_rate` in seconds, giving evenly spaced values from 0 to ~9.8 seconds.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. This assumes perfectly regular frame timing. The reference code uses actual timestamps from `ft`, which can have slight irregularities.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same frame count (0 to 31) as the neural window, so alignment is inherent.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. Aligned by construction since both use the same frame indexing.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial boolean indicating whether the corridor is rewarded.

ii.
```python
is_rew = beh['isRew']
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to float and broadcast across all time bins of the trial.

ii.
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. Same as reference - no additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name on each trial's corridor walls, and `UniqWalls`, the unique wall names across sessions.

ii.
```python
wall_name = wall_names[trial]
stim_cat = wall_name  # e.g., 'circle1', 'leaf2', etc.
# ...
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
```

iii. The AI uses WallName directly as the category.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps all 15 individual wall names (circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5) as separate categories. The reference code maps these to 4 broad categories (circle, leaf, rock, wood).

ii.
```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
stim_to_idx = {}
for exp_type in beh_cache:
    for beh_key_inner in beh_cache[exp_type]:
        for w in beh_cache[exp_type][beh_key_inner]['UniqWalls']:
            stim_to_idx[str(w)] = all_stim_sorted.index(str(w))

output_values = [
    [str(s) for s in all_stim_sorted],  # 15 categories
    ...
]
```

iii. The instruction says "Visual stimulus category. e.g. circle1, leaf2, etc." which the AI interpreted as using individual texture names. However, the reference maps these to 4 broad texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']

trial_lick_mask = lick_trinds == trial
if trial_lick_mask.any():
    trial_lick_frs = lick_frs[trial_lick_mask]
    for lf in trial_lick_frs:
        fr_idx = int(np.round(lf)) - start_fr
        if 0 <= fr_idx < n_timepoints:
            lick_binary[fr_idx] = 1.0
```

iii. The AI uses both LickFr and LickTrind to assign licks to trials, while the reference creates a session-wide lick flag array and then slices per trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks belonging to that trial (identified by LickTrind) are mapped to frame indices relative to StartFr. A frame is 1 if a lick falls in it, 0 otherwise. The output_values list has only 2 categories: ['no_lick', 'lick'], without a padding/none category.

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

iii. The approach is functionally similar but uses per-trial lick filtering via LickTrind rather than a session-wide array.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are converted to indices relative to StartFr, the same reference point as the neural data window.

ii.
```python
fr_idx = int(np.round(lf)) - start_fr
if 0 <= fr_idx < n_timepoints:
    lick_binary[fr_idx] = 1.0
```

iii. Aligned by using the same StartFr offset.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame, in decimeters.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[start_fr:end_fr]
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is discretized into 4 bins using `np.digitize` with boundaries at [10, 20, 30] decimeters.

ii.
```python
def discretize_position(pos_values, n_bins=4):
    boundaries = [10, 20, 30]  # in decimeters
    binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
    return binned
```

iii. This gives the same 4 bins of 1m each as the reference, but uses `np.digitize` rather than `np.clip(ft_Pos // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four 1-meter bins: [0,10)dm -> bin 0, [10,20)dm -> bin 1, [20,30)dm -> bin 2, [30+)dm -> bin 3. No padding/none category is included in output_values.

ii.
```python
boundaries = [10, 20, 30]
binned = np.digitize(pos_values, boundaries)
# output_values: ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. Since the AI uses a fixed window from StartFr that may extend into gray space (>40dm), positions beyond 4m would all map to bin 3.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced using the same `start_fr:end_fr` window as the neural data.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. Aligned by using the same frame window.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_speed[start_fr:end_fr]
```

iii. Same source as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes speed quartile boundaries globally across ALL sessions using `np.percentile` at [25, 50, 75], then discretizes each trial's speed using `np.digitize`. The reference code computes rank-based quartiles per session over only the kept frames.

ii.
```python
# Global quartiles
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])

# Per-trial discretization
speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
```

iii. The AI chose global quartiles for consistency across sessions, while the reference uses per-session rank-based splits.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins using global percentile boundaries. Values below 25th percentile -> Q1, 25-50th -> Q2, 50-75th -> Q3, above 75th -> Q4. No padding/none category.

ii.
```python
output_values = [
    ...
    ['Q1', 'Q2', 'Q3', 'Q4'],  # speed quartiles
]
```

iii. Using global percentiles rather than per-session rank-based quartiles means the bins won't each contain exactly 25% of the data within a session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sliced from the same `start_fr:end_fr` window as neural data.

ii.
```python
trial_speed = ft_speed[start_fr:end_fr]
speed_binned = np.digitize(trial_speed, speed_quartiles)
```

iii. Aligned by using the same frame window.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Trials where `start_fr + N_TIMEPOINTS` exceeds the number of available frames are marked invalid and excluded. Sessions with fewer than 10 trials or fewer than 10 visual cortex neurons are skipped. Behavior data is truncated to the number of neural frames.

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)

if beh['ntrials'] < MIN_TRIALS:
    continue

if valid_neurons.sum() < 10:
    continue
```

iii. The AI handles edge cases conservatively by excluding problematic data.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (`load_spk`), which involves reading large numpy files from disk and concatenating across imaging planes.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. I/O is the dominant cost, same as the reference.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking computation loops over individual lick frames per trial rather than using vectorized array operations. The per-trial extraction loop with frame-by-frame checks could also be more vectorized.

ii.
```python
for lf in trial_lick_frs:
    fr_idx = int(np.round(lf)) - start_fr
    if 0 <= fr_idx < n_timepoints:
        lick_binary[fr_idx] = 1.0
```

iii. The lick loop is O(n_licks) per trial but could be vectorized with array indexing.

## 12-c. What processing does the code repeat multiple times?

i. The speed discretization is done twice: once in `extract_trial_data` using per-trial quartiles, then overwritten in the main loop using global quartiles. The behavior cache is also loaded in a first pass for stimulus discovery and then reused.

ii.
```python
# First discretization in extract_trial_data:
speed_binned = discretize_speed(trial_speed, n_bins=4)
# Then overwritten:
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The double speed computation is wasteful but doesn't affect correctness since the second pass overwrites the first.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The initial per-trial speed discretization in `extract_trial_data` is completely discarded and overwritten with global quartile discretization. The `STIM_NAMES` dictionary is defined but never used. The `ft_move` behavioral variable is loaded but never used.

ii.
```python
ft_move = beh['ft_move'][:n_frames]  # loaded but unused

STIM_NAMES = {  # defined but unused
    0: 'circle1', ...
}
```

iii. Minor code cleanliness issues that don't affect the output.
