# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three sources: (1) `Imaging_Exp_info.npy` for session metadata and experiment info, (2) `Beh_<exp_type>.npy` files for behavioral data, and (3) per-session `.npy` files for neural spike data and `.npz` files for retinotopy. It iterates over all experiment types in `Imaging_Exp_info.npy`, deduplicates sessions by neural session key (`mname_datexp_blk`), picking the behavior entry with the most stimuli. Behavior files are cached by experiment type.

ii.
```python
def get_session_list():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    sessions = {}
    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
            # ...pick best behavior file...
            if key not in sessions or n_stim > sessions[key]['n_stim']:
                sessions[key] = { ... }

# Behavior loading:
beh_cache[exp_type] = np.load(
    os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()

# Neural loading:
def load_spk(mname, datexp, blk, root):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
    return spk
```

iii. The AI followed the structure of the reference code's `load_spk()` function for neural data loading. The session list construction iterates over all experiment types in the metadata file and builds a deduplicated session list.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field from `Imaging_Exp_info.npy`. A sorted list of unique mouse names is constructed, and each session is assigned a `subject_idx` mapping into this list.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
# Later per session:
subject_idx_all.append(subject_to_idx[mname])
```

iii. This follows standard practice of extracting subject identity from the session metadata.

## 1-c. How are the data split into sessions?

i. Each unique combination of `mname_datexp_blk` defines a session. When multiple behavior entries exist for the same neural session key, the entry with the most valid stimuli is chosen. Sessions with fewer than 10 trials or fewer than 10 visual cortex neurons are skipped.

ii.
```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
# Deduplication: pick entry with most stimuli
if key not in sessions or n_stim > sessions[key]['n_stim']:
    sessions[key] = { ... }

# Filtering:
if beh['ntrials'] < MIN_TRIALS:  # MIN_TRIALS = 10
    continue
if valid_neurons.sum() < 10:
    continue
```

iii. The AI reasoned that each unique neural recording (identified by mouse, date, block) constitutes one session, and selected the best matching behavior data.

## 1-d. How are the data split into trials?

i. Trials are defined by the `ntrials` field from the behavior data. Each trial starts at the frame given by `StartFr[trial]` and extends for a fixed 32-frame window. A trial is valid if `start_fr >= 0` and `start_fr + 32 <= n_frames`.

ii.
```python
ntrials = beh['ntrials']
start_frs = beh['StartFr']
for trial in range(ntrials):
    start_fr = int(np.round(start_frs[trial]))
    end_fr = start_fr + n_timepoints  # n_timepoints = 32
    if start_fr < 0 or end_fr > n_frames:
        valid_mask.append(False)
        continue
```

iii. The AI chose a fixed 32-frame window based on the calculation that at 60 cm/s and 3.17 Hz, a 6m corridor takes approximately 32 frames (~10s).

## 1-e. How are trials filtered based on quality controls?

i. Trials are only filtered based on whether the 32-frame window fits within the recording bounds. There is no filtering based on running behavior, trial completion, stimulus validity, or other quality metrics. Sessions with fewer than 10 valid trials after this filtering are excluded entirely.

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    continue
# ...
if len(valid_neural) < MIN_TRIALS:
    print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
    continue
```

iii. The AI did not implement the running-frame filtering (`ft_move > 0`) that the reference code uses. The reference code's `get_interpPos_spk()` function only includes frames where `VRmove = ft_move > 0`, but the AI included all frames regardless of running state.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the deconvolved calcium fluorescence traces stored in the `spks` field of the neural data files (`{mname}_{datexp}_{blk}_neural_data.npy`).

ii.
```python
def load_spk(mname, datexp, blk, root):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
    return spk
```

iii. This matches the reference code's `load_spk()` function, which loads the same `spks` field from the same file format. The data represents Suite2p deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The neural data is extracted as raw deconvolved traces over a fixed 32-frame time window starting at corridor entry (`StartFr`). No additional processing (e.g., normalization, z-scoring, smoothing, or position interpolation) is applied. The data is cast to float32.

ii.
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)  # (n_neurons, n_timepoints)
```

iii. The reference code performs position interpolation (60 position bins) using `get_interpPos_spk()`, but the AI chose time-based alignment instead, reasoning that the decoder task requires time-varying outputs. The AI uses raw deconvolved traces without any spatial interpolation or temporal rebinning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by brain region: only visual cortex neurons (V1, mHV, lHV, aHV) are kept; neurons with `iarea == -1` or `iarea == 7` are excluded. No other quality filtering is applied (e.g., no minimum firing rate threshold, no SNR filtering).

ii.
```python
region_idx = get_brain_region_idx(iarea)
valid_neurons = region_idx >= 0  # Exclude iarea -1 and 7
spk_filtered = spk[valid_neurons]
```

iii. The AI's brain region mapping follows the reference code's `neu_area_ID()` function exactly: V1=iarea 8, mHV=0/1/2/9, lHV=5/6, aHV=3/4.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by using `StartFr` as the alignment frame. A fixed window of 32 frames is extracted starting from this frame.

ii.
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints  # N_TIMEPOINTS = 32
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)" and `StartFr` represents the frame when the mouse enters the corridor. The AI's alignment matches this requirement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate of 3.17 Hz, giving a time bin size of ~315.5 ms. No temporal rebinning is applied.

ii.
```python
FS = 3.17  # Calcium imaging frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms per frame
N_TIMEPOINTS = 32
```

iii. The AI preserved the native frame rate without any rebinning, which is appropriate since the reference code also works at the native imaging rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundFr']` (frame of sound cue onset per trial) and `beh['StartFr']` (frame of trial start).

ii.
```python
sound_frs = beh['SoundFr']
start_frs = beh['StartFr']
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The AI computed the relative frame of sound onset, then computed a time-varying signal representing time to sound cue at each frame. Negative values indicate before the cue, positive values after.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound frame is converted to a trial-relative index by subtracting `StartFr`. Then for each time bin, the distance in frames from the sound frame is computed and divided by the frame rate to get seconds.

ii.
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. This creates a continuous, time-varying signal as specified by the instructions. Values can be extremely negative (e.g., -401s) when mice pause for long periods, causing the sound cue frame to be much later than the trial start frame.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is inherently aligned because both use the same frame indices. Frame 0 of the neural data corresponds to `StartFr`, and `time_to_cue` is computed relative to the same frame indices.

ii.
```python
# Both start from start_fr and span n_timepoints frames
trial_neural = spk[:, start_fr:end_fr]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
```

iii. The alignment is correct by construction since both the neural and input data use the same temporal frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field (recording date string in format `YYYY_MM_DD`) for each session, computed relative to the earliest recording date for each mouse.

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

iii. The AI chose to compute calendar days since the first recording, rather than using the `days` or `sess#` fields directly from the metadata. This is a reasonable approach but may differ from the reference code's definition.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar day difference between the current session date and the earliest session date for the same mouse. The value is constant across all time bins within a trial and across all trials in a session.

ii.
```python
day_idx = (current_date - mouse_dates[0]).days
# Later:
input_trials[i][1, :] = float(day)  # constant across time
```

iii. The instruction says "Day of training, continuous, time-varying." The AI makes it constant per session but broadcasts it across all time bins. Since each session is one day, it's effectively time-varying across sessions but constant within a session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Computed purely from frame indices and the frame rate; no raw data variable is directly used.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. Since the neural data is aligned to trial start (frame 0 = trial start), time since trial start is simply the frame index divided by the frame rate.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame index (0 to 31) divided by the frame rate (3.17 Hz) to get seconds (0 to ~9.8s).

ii.
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. This is straightforward and correct. Every trial has the same time-since-start values since all trials have the same number of frames.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Inherently aligned: frame 0 of the neural data corresponds to time 0, frame 1 to 1/3.17 seconds, etc.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. Correct by construction.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a per-trial boolean/binary field indicating whether reward was available on that trial.

ii.
```python
is_rew = beh['isRew']
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The AI correctly identified `isRew` as the field indicating whether the corridor was rewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value is broadcast to all time bins as a constant. It is cast to float (1.0 for rewarded, 0.0 for not rewarded).

ii.
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. This matches the instruction "1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which provides the stimulus name (e.g., 'circle1', 'leaf2') per trial.

ii.
```python
wall_names = beh['WallName']
wall_name = wall_names[trial]
stim_cat = wall_name
```

iii. The AI used `WallName` directly rather than mapping through `stim_id`. The reference code uses a `stim_id` mapping (0=circle1, 1=circle2, etc.), but the AI collects all unique `WallName` values across all sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` values are collected across all sessions, sorted alphabetically, and each trial is assigned a categorical index into this sorted list. The index is constant across all time bins.

ii.
```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
# Per trial:
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx  # constant across time
```

iii. The AI found 15 unique stimulus categories, including stimuli not in the reference code's 7-category mapping (e.g., circle3, rock1, rock2, wood1, wood2, wood5). This is because different experiment types use different stimulus sets, and the AI included all of them.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']` (frame indices of licks) and `beh['LickTrind']` (trial index of each lick).

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
```

iii. These are the standard lick-related fields in the behavior data structure.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frames matching that trial index are selected from `LickFr`. Each lick frame is converted to a trial-relative index by subtracting `StartFr`, and if it falls within the 32-frame window, the corresponding bin is set to 1.

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

iii. This creates a binary time-varying signal as required. Note that lick frames are rounded to integer frame indices, and the loop processes each lick individually.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are converted to the same trial-relative frame indices as the neural data by subtracting `start_fr`. This ensures lick bin `i` corresponds to neural data bin `i`.

ii.
```python
fr_idx = int(np.round(lf)) - start_fr
```

iii. Correct alignment by construction since both use the same frame reference.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, the frame-level position data in decimeters.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[start_fr:end_fr]
```

iii. `ft_Pos` contains the animal's position in the virtual corridor at each imaging frame.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw position values (in decimeters) are extracted for the 32-frame trial window and discretized into 4 bins.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. No additional processing (smoothing, clipping) is applied to the position data before discretization.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Discretized into 4 equal-length 1m bins using boundaries at 10, 20, and 30 decimeters. Bin 0: 0-1m, Bin 1: 1-2m, Bin 2: 2-3m, Bin 3: 3-4m and beyond. Values in the gray space (>4m) are assigned to the last bin.

ii.
```python
def discretize_position(pos_values, n_bins=4):
    boundaries = [10, 20, 30]  # in decimeters
    binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
    return binned
```

iii. This matches the instruction "4 equal-length, 1-m-long spatial bins." The 4m texture corridor is divided into 4 bins of 1m each. Positions beyond 4m (gray space) are collapsed into the last bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted from the same frame indices as the neural data (`start_fr:end_fr`), ensuring frame-level alignment.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
```

iii. Correct alignment by construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, the frame-level running speed data.

ii.
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
```

iii. `ft_RunSpeed` provides the instantaneous running speed at each imaging frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Raw speed values are extracted per trial and discretized into 4 bins using global quartile boundaries computed across all sessions. The boundaries are [0.0, 9.67, 31.46].

ii.
```python
# Global quartile computation:
all_speeds = []
for key, sess_info in sessions.items():
    speeds = beh['ft_RunSpeed']
    valid_speed = speeds[~np.isnan(speeds)]
    all_speeds.append(valid_speed)
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])

# Per trial:
speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
```

iii. The AI computed global quartiles across all frames of all sessions. Because many frames have zero speed (mouse not running), Q1=0.0, leading to heavily imbalanced bins (Q1=7.2%, Q2=58.2%, Q3=30.4%, Q4=4.2%).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with global quartile boundaries [0.0, 9.67, 31.46] to assign each frame to one of 4 bins (Q1, Q2, Q3, Q4).

ii.
```python
speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
```

iii. The instruction says "4 bins, each corresponding to 25% of the data." The AI attempted this via quartiles, but computing across all frames (including stationary frames with speed=0) results in Q1 boundary at 0.0, making the bins highly imbalanced in the actual data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted from the same frame indices as neural data (`start_fr:end_fr`).

ii.
```python
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
```

iii. Correct alignment by construction.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing data in several ways: (1) sessions with missing behavior keys are skipped with a warning; (2) sessions where neural or retinotopy files fail to load are skipped; (3) trials where the 32-frame window extends beyond the recording are marked invalid; (4) NaN speed values are handled during global quartile computation by filtering with `~np.isnan`. However, NaN values in position or speed within individual trials are not explicitly handled — `np.digitize` assigns NaN values to bin 0.

ii.
```python
# Missing behavior:
if beh_key not in beh_cache[exp_type]:
    print(f"  WARNING: beh_key {beh_key} not in {exp_type}, skipping")
    continue

# Failed loads:
except Exception as e:
    print(f"  ERROR loading spk: {e}")
    continue

# Out-of-bounds trials:
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    continue
```

iii. The AI's error handling is functional but minimal. Missing data is handled by skipping entire sessions/trials rather than interpolation or imputation.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading neural data files, which can be several GB each (e.g., 4GB for some sessions); (2) filtering neurons by brain region for each session (slicing large arrays); (3) the inner loop over all trials extracting fixed-length windows from the large neural matrix. The full conversion ran out of memory at session 47/89.

ii.
```python
spk = load_spk(mname, datexp, blk, DATA_ROOT)  # loads entire session
spk_filtered = spk[valid_neurons]  # copies large array
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)  # per-trial extraction
```

iii. The AI's approach stores all session data in memory simultaneously, leading to the memory exhaustion at session 47/89 (~237 GB RAM).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick processing loop iterates over individual lick frames per trial, which could be vectorized using array indexing. The trial extraction loop could also be partially vectorized by pre-computing all trial slices.

ii.
```python
# Inner loop over lick frames:
for lf in trial_lick_frs:
    fr_idx = int(np.round(lf)) - start_fr
    if 0 <= fr_idx < n_timepoints:
        lick_binary[fr_idx] = 1.0
```

iii. While the lick loop is inefficient, the main bottleneck is memory, not computation speed of these inner loops.

## 12-c. What processing does the code repeat multiple times?

i. Speed discretization is done twice: first in `extract_trial_data()` using per-trial quartiles via `discretize_speed()`, and then overridden in the main loop using global quartiles via `np.digitize()`. The first computation is wasted.

ii.
```python
# In extract_trial_data():
speed_binned = discretize_speed(trial_speed, n_bins=4)

# Later in process_all_sessions(), overwriting:
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The `discretize_speed()` function is called inside `extract_trial_data()` but its results are immediately overwritten in the main processing loop with global quartile values.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of unnecessary processing: (1) the per-trial speed discretization in `extract_trial_data()` is overwritten; (2) the `stim_categories` list is built during trial extraction but the same information is re-derived from `stim_cats` in the main loop; (3) the `session_speed_all` parameter in `extract_trial_data()` is accepted but never used; (4) global speed quartile computation collects all speeds from all sessions including sessions that may be skipped later.

ii.
```python
# Unused parameter:
def extract_trial_data(spk, beh, n_timepoints, session_speed_all=None):

# Overwritten speed binning:
speed_binned = discretize_speed(trial_speed, n_bins=4)  # computed but overwritten
```

iii. These inefficiencies suggest iterative development where earlier implementations were partially replaced without cleaning up the residual code.
