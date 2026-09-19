# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `data/beh/Imaging_Exp_info.npy` as a master index, collapses entries by unique `(mname, datexp, blk)` session key, and chooses one behavior record per neural session by keeping the entry with the largest count of non-NaN `stim_id` values. It later caches `Beh_<exp_type>.npy` files by experiment type, scans them to collect stimulus names and speed values, and loads spikes and retinotopy per session.

ii. ```python
def get_session_list():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    sessions = {}
    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
            stimtype = db.get('stimtype', '')
            beh_key = key if not stimtype else f'{key}_{stimtype}'
            stim_id = db.get('stim_id', np.array([]))
            n_stim = int(np.sum(~np.isnan(stim_id.astype(float))))
            if key not in sessions or n_stim > sessions[key]['n_stim']:
                sessions[key] = {
                    'mname': db['mname'],
                    'datexp': db['datexp'],
                    'blk': db['blk'],
                    'exp_type': exp_type,
                    'beh_key': beh_key,
                    'n_stim': n_stim,
                }
```
```python
if exp_type not in beh_cache:
    beh_cache[exp_type] = np.load(
        os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'),
        allow_pickle=True
    ).item()
```
```python
spk = load_spk(mname, datexp, blk, DATA_ROOT)
iarea = load_retino(mname, datexp, DATA_ROOT)
```

iii. In the trajectory, the agent said the same neural recording appears in multiple behavior datasets, so it should map each unique neural recording to one decoder session and "select whichever behavior interpretation has the most complete stimulus coverage for that recording" (steps 37, 79, 81).

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The output subject list is the sorted set of unique `mname` values from the chosen sessions, and each kept session stores the corresponding index in `subject_idx`.

ii. ```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_all.append(subject_to_idx[mname])
```

iii. No separate justification was given beyond treating each unique `mname` as one mouse/session owner.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` combination. If the same neural session appears multiple times in `Imaging_Exp_info.npy`, the code keeps one entry and picks the behavior record with the greatest apparent stimulus coverage (`n_stim`).

ii. ```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
...
if key not in sessions or n_stim > sessions[key]['n_stim']:
    sessions[key] = {
        'mname': db['mname'],
        'datexp': db['datexp'],
        'blk': db['blk'],
        'exp_type': exp_type,
        'beh_key': beh_key,
        'n_stim': n_stim,
    }
```

iii. The trajectory justification was that each unique neural recording should become one decoder session, while duplicate behavior records should be resolved by selecting the version with the most non-NaN `stim_id` entries (steps 37, 79, 81).

## 1-d. How are the data split into trials?

i. Trials are split by iterating over `range(beh['ntrials'])`. For each trial, the code rounds `StartFr[trial]` to an integer frame and extracts a fixed 32-frame window from that start frame, instead of using trial frame labels such as `ft_trInd` and `ft_CorrSpc`.

ii. ```python
for trial in range(ntrials):
    start_fr = int(np.round(start_frs[trial]))
    end_fr = start_fr + n_timepoints
    ...
    trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The trajectory justification was that the decoder needed fixed-size time bins aligned to corridor entry, and that roughly 10 seconds, or about 32 imaging frames, should cover the full traversal of the 6 m corridor (steps 33, 50, README text in step 102).

## 1-e. How are trials filtered based on quality controls?

i. A trial is rejected only if its 32-frame window would go out of bounds (`start_fr < 0` or `end_fr > n_frames`). After per-trial extraction, an entire session is skipped if it has fewer than 10 valid trials. Sessions are also skipped earlier if they have fewer than 10 visual-cortex neurons.

ii. ```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    neural_trials.append(None)
    input_trials.append(None)
    output_trials.append(None)
    stim_categories.append('')
    continue
```
```python
if beh['ntrials'] < MIN_TRIALS:
    print(f"  Skipping: only {beh['ntrials']} trials (< {MIN_TRIALS})")
    continue
...
if len(valid_neural) < MIN_TRIALS:
    print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
    continue
```

iii. The trajectory does not show a paper-based quality-control rationale for this filtering. The only explicit motivation was satisfying decoder practicality and minimum-data requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the `spks` arrays in each session's `*_neural_data.npy` file. Region labels come from `iarea` in the matching retinotopy `.npz` file.

ii. ```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```
```python
dtrans = np.load(ret_path, allow_pickle=True)
return dtrans['iarea']
```

iii. The trajectory consistently described the neural source as deconvolved fluorescence traces plus retinotopy-derived region assignments.

## 2-b. How is the `neural` data processed?

i. The code concatenates all imaging planes, filters neurons to four visual areas, slices a fixed 32-frame segment beginning at rounded `StartFr`, and stores each trial array as `float32`.

ii. ```python
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
spk_filtered = spk[valid_neurons]
...
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The trajectory justification was that the decoder needed fixed-size, trial-start-aligned windows, and that using `float32` was a practical compromise for memory and file size (steps 33, 67, 68, 75, 83, 87).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if `iarea` maps to `V1`, `mHV`, `lHV`, or `aHV`; all others receive region index `-1` and are dropped. Entire sessions are skipped if fewer than 10 valid neurons remain.

ii. ```python
def get_brain_region_idx(iarea):
    region_idx = np.full(len(iarea), -1, dtype=int)
    region_idx[iarea == 8] = 0
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
    region_idx[(iarea == 5) | (iarea == 6)] = 2
    region_idx[(iarea == 3) | (iarea == 4)] = 3
    return region_idx
...
valid_neurons = region_idx >= 0
if valid_neurons.sum() < 10:
    print(f"  Skipping: only {valid_neurons.sum()} visual cortex neurons")
    continue
```

iii. The trajectory justification was that the paper and utilities focus on visual cortex areas, so neurons outside those areas should be excluded (steps 33, 35, README text in step 102). No paper-based justification was given for the extra `< 10 neurons` session filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to corridor entry by taking the rounded `StartFr` as frame 0. The code then keeps exactly 32 consecutive imaging frames after that point.

ii. ```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The trajectory repeatedly justified this as temporal alignment to trial start with a fixed window covering the expected traversal duration of the 6 m corridor (steps 33, 50, 102).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one imaging frame, using `FS = 3.17` Hz and `TIME_BIN_MS = 1000.0 / FS`. No temporal rebinning or resampling is applied; the code directly slices frame-aligned arrays.

ii. ```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS
...
'time_bin_size': TIME_BIN_MS,
```

iii. The trajectory justification was that calcium imaging frames already provide a common time base, about 315 ms per bin (steps 26, 33, README in step 102).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The code derives this variable from `SoundFr` and `StartFr`, together with the assumed constant frame rate `FS`. It does not use the `ft` timestamp array.

ii. ```python
start_frs = beh['StartFr']
sound_frs = beh['SoundFr']
...
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The trajectory justification was that the decoder should represent time relative to cue onset within a fixed trial-start-aligned frame window (steps 33, 50, 64).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code subtracts the cue frame offset from `np.arange(n_timepoints)`, then divides by `FS` to convert frames to seconds. This produces negative values before the cue and positive values after it.

ii. ```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The trajectory explicitly justified the sign convention by saying the large negative values occur when the cue is delayed far into the trial, so early bins are many frames before the cue (steps 50, 64).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same 32-frame trial window used for neural data, with one value per neural frame.

ii. ```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. The trajectory justification was that every decoder input should be frame-by-frame and aligned to the same fixed trial-start window as the neural activity (step 33).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `day_of_training` is derived from the mouse name `mname` and session date string `datexp` across all selected sessions for that mouse.

ii. ```python
def compute_day_of_training(mname, datexp, all_sessions):
    from datetime import datetime
    mouse_dates = []
    for k, s in all_sessions.items():
        if s['mname'] == mname:
            date = datetime.strptime(s['datexp'], '%Y_%m_%d')
            mouse_dates.append(date)
```

iii. The trajectory justification was that training day should be computed relative to the first recording session for each mouse (README text in step 102 and the function name/comment).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code parses all session dates for a mouse as calendar dates, sorts them, and computes `day_idx` as the absolute day difference between the current session date and the mouse's earliest session date. That scalar is then broadcast across all time bins of each valid trial.

ii. ```python
mouse_dates.sort()
current_date = datetime.strptime(datexp, '%Y_%m_%d')
day_idx = (current_date - mouse_dates[0]).days
return day_idx
```
```python
for i in range(len(input_trials)):
    if valid_mask[i]:
        input_trials[i][1, :] = float(day)
```

iii. The trajectory justification was simply "Days since first recording session for this mouse" (README text in step 102). No justification was given for using elapsed calendar days instead of session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The code derives it from `StartFr` and the fixed frame index within the extracted 32-frame window, converted to seconds using `FS`.

ii. ```python
start_frs = beh['StartFr']
...
time_since_start = np.arange(n_timepoints) / FS
```

iii. The trajectory justification was that trial start should be the temporal alignment event and that all decoder variables should share the same fixed frame grid (step 33).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The value is simply `0, 1/FS, 2/FS, ...` for the 32 extracted bins. Because the code starts the window exactly at rounded `StartFr`, no interpolation is applied.

ii. ```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. The trajectory justification was that trial start should define frame 0 for a fixed temporal window (steps 33, 102).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is defined directly on the same 32 bins as the neural trial matrix, so the two are aligned by construction.

ii. ```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. The trajectory justification was the general choice to align all streams to the same fixed trial-start window (step 33).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is taken directly from `isRew` for each trial.

ii. ```python
is_rew = beh['isRew']
...
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. No additional justification was given beyond using the trial's rewarded/not-rewarded flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code performs no transformation beyond broadcasting each trial's `isRew` value across all 32 time bins of that trial.

ii. ```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The trajectory and README both frame this variable as a per-trial indicator: "1 if rewarded corridor, 0 if not" (step 102).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The output category is derived from `WallName` for each trial. The code also scans `UniqWalls` across sessions to build the list of all possible output values.

ii. ```python
wall_names = beh['WallName']
...
wall_name = wall_names[trial]
stim_cat = wall_name
stim_categories.append(stim_cat)
```
```python
for w in beh['UniqWalls']:
    all_stim_names.add(w)
```

iii. The trajectory justification was that duplicate behavior files mostly differ in stimulus annotations, so the code should preserve the most complete stimulus labeling for each neural session (steps 37, 79, 81).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code keeps the raw wall name as the category, builds a global sorted list of all unique wall labels seen in `UniqWalls`, assigns each wall name an integer index, and writes that same integer to every time bin of the trial.

ii. ```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
...
stim_to_idx[str(w)] = all_stim_sorted.index(str(w))
...
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx
```

iii. The trajectory justification was that different behavior records for the same session mainly change stimulus assignments, so the version with the richest stimulus information should define the categories (steps 37, 79, 81). The README also described categories as raw names such as `circle1` and `leaf1` (step 102).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind`. `LickFr` gives lick frames, and `LickTrind` is used to select the licks that belong to a given trial.

ii. ```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
...
trial_lick_mask = lick_trinds == trial
trial_lick_frs = lick_frs[trial_lick_mask]
```

iii. No separate paper-based justification was given; the code simply treats the available lick frame/trial arrays as the source of binary licking output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code initializes a 32-bin zero vector, selects licks whose `LickTrind` matches that trial, rounds each lick frame, subtracts `start_fr` to convert it into a window-relative index, and sets the corresponding bin to 1 if it falls inside the fixed window.

ii. ```python
lick_binary = np.zeros(n_timepoints, dtype=float)
trial_lick_mask = lick_trinds == trial
if trial_lick_mask.any():
    trial_lick_frs = lick_frs[trial_lick_mask]
    for lf in trial_lick_frs:
        fr_idx = int(np.round(lf)) - start_fr
        if 0 <= fr_idx < n_timepoints:
            lick_binary[fr_idx] = 1.0
```

iii. The trajectory noted that some sample sessions had all-zero licking because they were unsupervised, and then changed the sample-session selection to include supervised sessions (step 50 and step 59).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by expressing each lick as a frame offset from the same rounded `StartFr` used to slice neural data, then placing it into the same 32-bin window.

ii. ```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
fr_idx = int(np.round(lf)) - start_fr
if 0 <= fr_idx < n_timepoints:
    lick_binary[fr_idx] = 1.0
```

iii. The trajectory justification was that all time-varying outputs should be frame-by-frame within the same trial-start-aligned window (step 33).

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from the framewise `ft_Pos` array.

ii. ```python
ft_pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_pos[start_fr:end_fr]
```

iii. No additional justification was given beyond using the available framewise position trace.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code takes the 32-frame `ft_Pos` slice for a trial and discretizes each value into one of four bins using 10, 20, and 30 decimeter boundaries. Values beyond the textured 4 m corridor are still kept and land in the last bin.

ii. ```python
def discretize_position(pos_values, n_bins=4):
    boundaries = [10, 20, 30]
    binned = np.digitize(pos_values, boundaries)
    return binned
...
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. The trajectory justification was that the task asked for four equal 1 m bins, while the fixed 32-frame window was intended to span the full corridor traversal (step 33 and README in step 102).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are hard-coded at 10, 20, and 30 decimeters, producing categories `0-1m`, `1-2m`, `2-3m`, and `3-4m`.

ii. ```python
boundaries = [10, 20, 30]  # in decimeters
binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
```
```python
['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The only explicit justification was the task requirement for four equal-length, 1 m spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position uses the same `[start_fr:end_fr]` slice as neural data, so each position bin corresponds to one neural frame in the fixed 32-bin window.

ii. ```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. The trajectory justification was the general fixed-window, trial-start alignment for all time-varying streams (step 33).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the framewise `ft_RunSpeed` array.

ii. ```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
...
trial_speed = ft_speed[start_fr:end_fr]
```

iii. No separate justification was given beyond using the behavior speed trace.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code first discretizes each trial's 32-frame speed vector with a local percentile-based helper, but then overwrites those values with bins computed from global quartile boundaries estimated across all non-NaN speed samples from all selected sessions. The final saved output is the second, global version.

ii. ```python
def discretize_speed(speed_values, n_bins=4):
    valid = speed_values[~np.isnan(speed_values)]
    if len(valid) == 0:
        return np.zeros_like(speed_values, dtype=int)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(valid, percentiles)
    binned = np.digitize(speed_values, boundaries)
    return binned
```
```python
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The trajectory justification was "Compute speed quartile boundaries across ALL sessions for consistent discretization," while also noting that Q1 became 0 because many frames were stationary (code comments and step 64).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The final thresholds are the global 25th, 50th, and 75th percentiles of all non-NaN `ft_RunSpeed` samples gathered across the chosen sessions, and `np.digitize` maps each frame into one of four categories `Q1`-`Q4`.

ii. ```python
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
```
```python
['Q1', 'Q2', 'Q3', 'Q4']
```

iii. The trajectory explicitly justified this as a globally consistent discretization scheme, even though it observed that the lowest quartile boundary was often 0 because of many stationary frames (step 64).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is taken from the same `[start_fr:end_fr]` slice as neural data and then discretized frame by frame, so the saved speed labels align one-to-one with neural columns.

ii. ```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
trial_speed = ft_speed[start_fr:end_fr]
speed_binned = np.digitize(trial_speed, speed_quartiles)
```

iii. The trajectory justification was the general frame-by-frame trial-start alignment used for all decoder outputs (step 33).

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly handles bad or missing cases by skipping them. It rejects out-of-bounds trials, drops sessions with too few trials or neurons, ignores NaNs when estimating speed quartiles, and skips sessions entirely if behavior keys, spike files, or retinotopy files cannot be loaded.

ii. ```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    ...
    continue
```
```python
valid = speed_values[~np.isnan(speed_values)]
if len(valid) == 0:
    return np.zeros_like(speed_values, dtype=int)
```
```python
if beh_key not in beh_cache[exp_type]:
    print(f"  WARNING: beh_key {beh_key} not in {exp_type}, skipping")
    continue
...
except Exception as e:
    print(f"  ERROR loading spk: {e}")
    continue
```

iii. The trajectory does not show a detailed data-cleaning rationale. Most of these branches appear to be defensive handling added during implementation.

## 12-a. What are the most time-consuming steps of the code?

i. The heaviest steps are loading and concatenating large spike files for each session, plus building and holding the full converted dataset in memory. The trajectory shows that this became the dominant runtime and memory bottleneck during the attempted full conversion.

ii. ```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```
```python
neural_all.append(valid_neural)
input_all.append(valid_input)
output_all.append(valid_output)
```

iii. The trajectory explicitly focuses on spike-file load time and the memory blow-up from holding all converted sessions at once, eventually reporting 237 GB RAM in use and killing the full run (steps 83, 85, 87, 113, 114).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest obvious candidates are the per-trial loop in `extract_trial_data`, the inner per-lick loop that marks lick bins one event at a time, and the multiple full-session passes used to collect stimulus names and global speed quartiles.

ii. ```python
for trial in range(ntrials):
    ...
    if trial_lick_mask.any():
        trial_lick_frs = lick_frs[trial_lick_mask]
        for lf in trial_lick_frs:
            fr_idx = int(np.round(lf)) - start_fr
            if 0 <= fr_idx < n_timepoints:
                lick_binary[fr_idx] = 1.0
```
```python
for key, sess_info in sessions.items():
    ...
    for w in beh['UniqWalls']:
        all_stim_names.add(w)
...
for key, sess_info in sessions.items():
    ...
    all_speeds.append(valid_speed)
```

iii. The trajectory never explicitly discusses vectorization, but it does discuss time and memory problems caused by repeated large passes over the dataset.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it scans behavior files once to collect all stimulus names and again to collect all speed values; it discretizes running speed once inside `extract_trial_data` and then overwrites it with a second discretization using global quartiles; and it repeatedly parses session dates when computing day of training.

ii. ```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```
```python
for key, sess_info in sessions.items():
    ...
    for w in beh['UniqWalls']:
        all_stim_names.add(w)
...
for key, sess_info in sessions.items():
    ...
    all_speeds.append(valid_speed)
```

iii. The trajectory justification for the second speed pass was the desire for "global speed quartiles" for consistency; no explicit justification was given for the other repeated work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The most obvious discarded work is the first speed discretization inside `extract_trial_data`, which is later overwritten. The code also reads `ft_move`, `stim_id`, `uniq_walls`, and `exp_info` in places where those variables do not affect the saved dataset, and it gathers some metadata fields that are only used for logging.

ii. ```python
ft_move = beh['ft_move'][:n_frames]
...
stim_id = beh.get('stim_id', None)
uniq_walls = beh['UniqWalls']
```
```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
```

iii. The only explicit trajectory justification here is for the overwritten speed discretization: the agent wanted a globally consistent quartile definition. The rest appears to be leftover exploratory or intermediate logic rather than something needed downstream.
