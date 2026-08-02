# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment info from `Imaging_Exp_info.npy`, which contains metadata organized by experiment type (e.g., `unsup_train1_before_learning`, `sup_test1`). Behavior data is loaded from `Beh_{exp_type}.npy` files. Neural data (deconvolved spikes) is loaded from per-session `.npy` files in `spk/`, and retinotopy data from `retinotopy/`. The AI builds a session list by iterating over all experiment types, deduplicating sessions by their unique key `{mname}_{datexp}_{blk}`, and choosing the behavior entry with the most stimuli when duplicates exist.

ii.
```python
def get_session_list():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    sessions = {}
    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
            ...
            if key not in sessions or n_stim > sessions[key]['n_stim']:
                sessions[key] = { ... }
    return sessions

def load_spk(mname, datexp, blk, root):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
    return spk
```

iii. The AI's approach of loading data from the shared Figshare data structure and iterating over experiment types matches the reference code's approach in `data_process_script.ipynb`, which also iterates over `exp_info` keys. The deduplication by session key is a reasonable decision to avoid processing the same neural recording multiple times.

## 1-b. How are the data split into subjects (mice)?

i. Subjects (mice) are identified by the `mname` field in the experiment info. A sorted list of all unique mouse names is collected, and each session is mapped to its mouse via `subject_to_idx`.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
# ...
subject_idx_all.append(subject_to_idx[mname])
```

iii. This is straightforward and correct. The `subjects` list and `subject_idx` array follow the required data format.

## 1-c. How are the data split into sessions?

i. Each unique combination of `{mname}_{datexp}_{blk}` defines a session. When the same neural recording appears in multiple experiment types, the AI picks the behavior entry with the most valid stimuli. All 89 sessions from the experiment info are processed.

ii.
```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
if key not in sessions or n_stim > sessions[key]['n_stim']:
    sessions[key] = { ... }
```

iii. The deduplication strategy ensures one session per unique recording, which avoids double-counting. The choice to prefer the behavior entry with more stimuli is reasonable.

## 1-d. How are the data split into trials?

i. Trials are defined by the `ntrials` field in the behavior data. Each trial has a start frame (`StartFr`), and the AI extracts a fixed window of 32 frames from each trial start.

ii.
```python
ntrials = beh['ntrials']
start_frs = beh['StartFr']
for trial in range(ntrials):
    start_fr = int(np.round(start_frs[trial]))
    end_fr = start_fr + n_timepoints  # N_TIMEPOINTS = 32
```

iii. The fixed 32-frame window is chosen based on the corridor traversal time: at 60 cm/s VR speed and 3.17 Hz frame rate, the 6m corridor takes ~10s = ~32 frames. This provides a consistent trial length for the decoder.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two ways: (1) if the trial's 32-frame window exceeds the recording bounds, it is marked invalid; (2) sessions with fewer than 10 valid trials are excluded entirely.

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    ...
    continue

if len(valid_neural) < MIN_TRIALS:
    print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
    continue
```

iii. The AI does NOT filter for running-only frames (`ft_move > 0`), which the reference code and paper explicitly do. The paper states: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." This is a significant omission.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` field in the `{mname}_{datexp}_{blk}_neural_data.npy` files. These are deconvolved calcium fluorescence traces from Suite2p.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. This matches the reference code's `load_spk` function exactly, and the paper confirms "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The neural data is extracted as raw deconvolved traces for each trial window (32 frames from trial start), cast to float32. No normalization, z-scoring, or spatial interpolation is applied.

ii.
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The reference code applies various normalizations (z-scoring, spatial interpolation) for its analyses, but those are specific to the paper's figure-generation analyses. For the decoder task, raw traces are a reasonable choice since the decoder can learn its own normalization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by brain region: only neurons in visual cortex areas (V1, mHV, lHV, aHV) are kept. Neurons with `iarea == -1` or `iarea == 7` are excluded. Sessions with fewer than 10 valid visual cortex neurons are also excluded.

ii.
```python
region_idx = get_brain_region_idx(iarea)
valid_neurons = region_idx >= 0  # Exclude iarea -1 and 7
if valid_neurons.sum() < 10:
    print(f"  Skipping: only {valid_neurons.sum()} visual cortex neurons")
    continue
spk_filtered = spk[valid_neurons]
```

iii. This matches the reference code pattern: `idx_neu = (arid!=-1) & (arid != 7)` in `Get_density_map` and the `neu_area_ID` mapping. The brain region ID assignments (V1=8, mHV=0/1/2/9, lHV=5/6, aHV=3/4) match `utils.py:neu_area_ID`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) using `StartFr` from the behavior data. A fixed window of 32 frames is taken from corridor entry onward.

ii.
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints  # 32
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The use of `StartFr` (the neural frame when the animal enters the corridor) is correct.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate of 3.17 Hz, giving ~315.5 ms per time bin. No rebinning is applied.

ii.
```python
FS = 3.17  # Calcium imaging frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms per frame
```

iii. The native frame rate is used directly, which preserves the original temporal resolution. No rebinning is needed since the data is already at a consistent frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (frame of sound cue delivery per trial) and `StartFr` (trial start frame).

ii.
```python
sound_frs = beh['SoundFr']
start_frs = beh['StartFr']
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. Uses frame-level timing from the behavior data to compute the relative time to the sound cue.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the sound cue frame is expressed relative to the trial start frame. Then for each timepoint in the trial window, the time to the cue is computed as `(frame_index - sound_cue_relative_frame) / frame_rate`, giving negative values before the cue and positive after.

ii.
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. This produces a continuous, time-varying signal in seconds as required. However, when the sound cue occurs far into the corridor (beyond the 32-frame window), `sound_fr_rel` can be very large, producing values like -401 seconds for `time_to_cue_sec`, which are unrealistically large. This is because `SoundFr` can be NaN or far beyond the trial window in some sessions.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both are aligned to the same trial start frame (`StartFr`), so the time-to-cue values directly correspond to neural data timepoints within the 32-frame window.

ii.
```python
time_to_cue = np.arange(n_timepoints) - sound_fr_rel  # same indices as neural data
```

iii. Alignment is inherently correct since both use the same frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field (date string in `YYYY_MM_DD` format) across all sessions for each mouse.

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

iii. The AI computes this as the number of calendar days since the mouse's first session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, all session dates are collected and sorted. The day of training for a given session is the number of days between the current session date and the earliest session date for that mouse. This value is constant across all timepoints in a trial and all trials in a session.

ii.
```python
day_idx = (current_date - mouse_dates[0]).days
# ...
input_trials[i][1, :] = float(day)  # constant across timepoints
```

iii. Using calendar days is a reasonable approximation but may differ from the actual training day count if sessions are not daily. The reference data has `days` and `sess#` fields that could have been used directly.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does NOT include an "Environment type" input. The decoder inputs are: time_to_sound_cue, day_of_training, time_since_trial_start, reward_availability. Environment type is not in the instruction's decoder input list.

ii. N/A - not implemented.

iii. The instructions list exactly 4 decoder inputs, none of which is "Environment type." The AI correctly followed the instruction specification.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not implemented - see 4-a above.

ii. N/A.

iii. N/A.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived purely from the frame index within the trial window and the frame rate.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. No raw behavior variables needed since trial start is the alignment event (time 0).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Simply computes `frame_index / frame_rate` for each of the 32 timepoints, giving values from 0 to ~9.8 seconds.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. Straightforward and correct.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Perfectly aligned by construction since both use the same frame indices from trial start.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. No alignment issues possible.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from the `isRew` field in the behavior data, which is a boolean per-trial indicator of whether the trial is a reward trial.

ii.
```python
is_rew = beh['isRew']
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. `isRew` directly indicates whether the corridor is a rewarded corridor for each trial.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The `isRew` boolean value for each trial is broadcast to all timepoints in the trial as a constant value (0.0 or 1.0).

ii.
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The instructions say "1 if in rewarded corridor, 0 if not, discrete, per-trial" which matches this implementation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `WallName` in the behavior data, which gives the stimulus name for each trial (e.g., 'circle1', 'leaf2').

ii.
```python
wall_names = beh['WallName']
wall_name = wall_names[trial]
stim_cat = wall_name
```

iii. Uses the stimulus name directly rather than the `stim_id` mapping from the paper.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique wall names across all sessions are collected, sorted alphabetically, and mapped to integer indices. The stimulus index is then broadcast to all timepoints in the trial.

ii.
```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
stim_to_idx[str(w)] = all_stim_sorted.index(str(w))
# ...
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx  # constant across time
```

iii. This produces 15 unique stimulus categories across all sessions. The per-trial output is replicated across all timepoints, which is redundant but matches the format requirement.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` (neural frame indices of licks) and `LickTrind` (trial indices of licks).

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
```

iii. These are the standard lick data fields documented in the data process notebook.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks assigned to that trial (`LickTrind == trial`) are mapped to frame indices relative to trial start. A binary vector is created where frames with at least one lick get value 1.

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

iii. This creates a binary time-varying lick signal as required. The approach of using `LickTrind` to filter licks to the correct trial is correct.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are expressed relative to the trial start frame (same reference as neural data), ensuring temporal alignment.

ii.
```python
fr_idx = int(np.round(lf)) - start_fr
if 0 <= fr_idx < n_timepoints:
    lick_binary[fr_idx] = 1.0
```

iii. The subtraction of `start_fr` correctly aligns lick frames to the trial window.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos` in the behavior data, which gives the VR position for each neural frame.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[start_fr:end_fr]
```

iii. `ft_Pos` provides position inside the VR corridor for each neural frame.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position values (in decimeters) for the trial window are extracted and then discretized into 4 bins.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. Straightforward extraction and discretization.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 4 bins of 1 meter (10 decimeters) each: [0-10dm), [10-20dm), [20-30dm), [30dm+). Values in the gray space (>40dm) are assigned to the last bin.

ii.
```python
def discretize_position(pos_values, n_bins=4):
    boundaries = [10, 20, 30]  # in decimeters
    binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
    return binned
```

iii. The instructions say "4 equal-length, 1-m-long spatial bins." The implementation uses [0-1m, 1-2m, 2-3m, 3-4m], covering the 4m texture area. However, values beyond 4m (in the 2m gray space) all get assigned to bin 3, which conflates the last texture bin with the gray space.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position data uses the same frame indices as neural data (`ft_Pos[start_fr:end_fr]`), so alignment is inherent.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
```

iii. Direct frame-level alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed` in the behavior data, which gives the running speed for each neural frame.

ii.
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_speed[start_fr:end_fr]
```

iii. `ft_RunSpeed` provides running speed per neural frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first extracted per trial window, then discretized into 4 bins using quartile boundaries. Initially per-trial quartiles are used, but then global quartiles (computed across all sessions) replace the per-trial discretization.

ii.
```python
# Global quartile computation:
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])

# Per-trial re-discretization with global quartiles:
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. Global quartile boundaries are [0.0, 9.67, 31.46] cm/s. The first boundary at 0.0 means any frame with speed exactly 0 gets bin 0, while all frames with speed > 0 but < 9.67 also get bin 0. This is because many stopped frames (speed=0) are included since running-only filtering is not applied.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Uses `np.digitize` with global quartile boundaries to assign each frame to one of 4 bins (Q1-Q4), where each bin should contain approximately 25% of the data.

ii.
```python
speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
```

iii. The instructions say "4 bins, each corresponding to 25% of the data." The global quartile approach achieves this across the full dataset. However, because stopped frames are included, the distribution is heavily skewed (Q1 boundary at 0.0, Q2 at ~9.67).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed data uses the same frame indices as neural data, so alignment is inherent.

ii.
```python
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
```

iii. Direct frame-level alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases of missing or problematic data are handled:
- Trials where the 32-frame window exceeds recording bounds are marked invalid
- Sessions with < 10 valid trials are skipped
- Sessions with < 10 visual cortex neurons are skipped
- Missing behavior keys cause sessions to be skipped
- NumPy string types are converted to Python strings
- Default stim_idx of 0 is used when stimulus name lookup fails

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    ...
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)  # default to 0 if not found
```

iii. The handling is conservative - problematic trials/sessions are excluded. The default to 0 for missing stimulus names could silently misclassify stimuli but unlikely to be triggered in practice.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading the neural data files (each ~GB), (2) computing global speed quartiles across all sessions (requires loading all behavior data), and (3) the trial-by-trial extraction loop for each session.

ii.
```python
# Loading neural data per session:
spk = load_spk(mname, datexp, blk, DATA_ROOT)

# Global speed computation:
for key, sess_info in sessions.items():
    speeds = beh['ft_RunSpeed']
    all_speeds.append(valid_speed)
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
```

iii. Loading ~89 large neural data files is inherently I/O bound. The global speed computation requires iterating all sessions once before processing.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick detection loop iterates over individual lick frames per trial, which could be vectorized. The trial extraction loop processes one trial at a time.

ii.
```python
# Per-lick iteration (could be vectorized):
for lf in trial_lick_frs:
    fr_idx = int(np.round(lf)) - start_fr
    if 0 <= fr_idx < n_timepoints:
        lick_binary[fr_idx] = 1.0
```

iii. Vectorizing lick detection and trial extraction would improve performance but the per-lick loop is a minor bottleneck compared to I/O.

## 12-c. What processing does the code repeat multiple times?

i. Speed discretization is done twice per trial: first in `extract_trial_data` using per-trial quartiles via `discretize_speed()`, then overwritten with global quartiles in the main loop. The behavior data is also effectively iterated twice (once for global speed computation, once for processing).

ii.
```python
# First discretization (in extract_trial_data):
speed_binned = discretize_speed(trial_speed, n_bins=4)

# Second discretization (in main loop, overwrites):
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The double speed discretization is wasteful. The initial per-trial discretization is completely overwritten by the global one.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `discretize_speed()` function is called for each trial in `extract_trial_data` but the results are immediately overwritten by global quartile discretization. The `stim_categories` list is built but could be integrated directly into the output construction. The `session_speed_all` parameter in `extract_trial_data` is accepted but never used.

ii.
```python
def extract_trial_data(spk, beh, n_timepoints, session_speed_all=None):
    # session_speed_all is never used
    ...
    speed_binned = discretize_speed(trial_speed, n_bins=4)  # immediately overwritten later
```

iii. These are minor inefficiencies that don't affect correctness.
