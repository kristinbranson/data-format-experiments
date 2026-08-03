# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `data/beh/Imaging_Exp_info.npy`, collapses entries to one record per unique `(mname, datexp, blk)` session, then loads the corresponding behavior dictionary from `Beh_<exp_type>.npy`, neural activity from `data/spk/<mouse>_<date>_<blk>_neural_data.npy`, and retinotopy labels from `data/retinotopy/<mouse>_<date>_trans.npz`. Its deduplication heuristic picks the behavior entry with the largest number of non-NaN `stim_id` values.

ii. 
```python
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
                sessions[key] = {...}
```

```python
def load_spk(mname, datexp, blk, root):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(os.path.join(root, 'spk', fn), allow_pickle=True).item()
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
    return spk

def load_retino(mname, datexp, root):
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(root, 'retinotopy', fn), allow_pickle=True)
    return dtrans['iarea']
```

iii. No `CONVERSION_NOTES.md` was present in `/app`, so the justification comes from the trajectory and README. In trajectory step 33 the agent said sessions are “each unique `(mname, datexp, blk)` combination with corresponding behavior data and neural data,” and in the module docstring/README it described loading Suite2p deconvolved traces, behavior, and retinotopy together.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by mouse name `mname`. The script builds a sorted list of unique mouse IDs across the deduplicated sessions and maps each processed session back to that subject list.

ii. 
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_all.append(subject_to_idx[mname])
```

iii. The trajectory inspection of `Imaging_Exp_info.npy` emphasized mouse IDs such as `DR10`, `TX60`, and `VR2`, and the README states the dataset contains recordings from 19 mice. The code follows that organization exactly.

## 1-c. How are the data split into sessions?

i. Sessions are split by the unique key `f"{mname}_{datexp}_{blk}"`. If multiple behavior entries refer to the same neural session, the AI keeps only one, chosen by the “most valid stimuli” heuristic, while preserving a `beh_key` that may include `stimtype`.

ii. 
```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
stimtype = db.get('stimtype', '')
beh_key = key if not stimtype else f'{key}_{stimtype}'
...
sorted_keys = sorted(sessions.keys())
for sess_idx, key in enumerate(sorted_keys):
    sess_info = sessions[key]
```

iii. In trajectory step 27 the agent counted 142 metadata entries but only 89 unique sessions, so it chose to collapse metadata records onto unique neural recordings. That is the explicit rationale for the session split.

## 1-d. How are the data split into trials?

i. Within each session, trials are split by iterating `range(beh['ntrials'])`. For each trial, the script rounds `StartFr[trial]` to an integer frame, then extracts a fixed 32-frame window starting there. It does not use `EndFr`, `GrayFr`, `ft_trInd`, or variable-length trial boundaries.

ii. 
```python
ntrials = beh['ntrials']
start_frs = beh['StartFr']
...
for trial in range(ntrials):
    start_fr = int(np.round(start_frs[trial]))
    end_fr = start_fr + n_timepoints
    ...
    trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The rationale appears in the docstring, README, and trajectory step 33: the agent decided to align to corridor entry and use a fixed-length window because at 60 cm/s a 6 m traversal is about 32 imaging frames.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. A trial is marked invalid only if the 32-frame window would go out of bounds. After that, the script keeps all remaining trials and only drops entire sessions if they end up with fewer than 10 valid trials.

ii. 
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    neural_trials.append(None)
    input_trials.append(None)
    output_trials.append(None)
    stim_categories.append('')
    continue
...
if len(valid_neural) < MIN_TRIALS:
    print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
    continue
```

iii. The trajectory shows the agent knew the paper “only considered timepoints during running,” but it did not convert that into an actual trial-level QC rule. The practical justification in code is just decoder feasibility via `MIN_TRIALS = 10`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `spk_data['spks']` in the per-session neural `.npy` file, with neurons subsequently filtered by retinotopy `iarea` from the matching `_trans.npz` file.

ii. 
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
dtrans = np.load(ret_path, allow_pickle=True)
return dtrans['iarea']
```

iii. The module docstring and README both say the neural source is “Deconvolved calcium fluorescence traces (Suite2p)” plus retinotopy-based area assignments. The trajectory inspection of a spike file also found only a `spks` field.

## 2-b. How is the `neural` data processed?

i. The AI concatenates all imaging planes along the neuron axis, keeps the resulting framewise deconvolved traces, converts each per-trial slice to `float32`, and stores fixed 32-frame trial windows starting at `StartFr`.

ii. 
```python
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. In trajectory step 33 the agent explicitly decided to “extract per-trial segments aligned to corridor entry” from the deconvolved traces. The README repeats that the window is 32 frames from trial start.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filtering is anatomical: neurons with retinotopy labels outside the four selected visual regions are discarded. Sessions are skipped if fewer than 10 such neurons remain. There is no explicit neuron quality metric, no running-frame mask, and no other neural QC.

ii. 
```python
region_idx = get_brain_region_idx(iarea)
valid_neurons = region_idx >= 0  # Exclude iarea -1 and 7
if valid_neurons.sum() < 10:
    print(f"  Skipping: only {valid_neurons.sum()} visual cortex neurons")
    continue
spk_filtered = spk[valid_neurons]
```

iii. The justification comes from the docstring/README wording “All visual cortex neurons (V1, mHV, lHV, aHV) kept; neurons outside visual cortex excluded.” The trajectory also inspected `iarea` counts and adopted those groups.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial is aligned to trial start, operationalized as `StartFr`, which the behavior notebook defines as the neural frame when the animal enters the corridor. The saved window starts exactly at that frame.

ii. 
```python
start_frs = beh['StartFr']
...
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. This was a deliberate choice: the task instructions specified “Temporally aligned based on trial start (corridor entry),” and the trajectory notes repeatedly identify `StartFr` as corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one original imaging frame, defined as `1000.0 / 3.17` ms, about 315.5 ms per bin. No temporal rebinning or smoothing is applied.

ii. 
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS
...
'time_bin_size': TIME_BIN_MS,
```

iii. The behavior reference excerpt read in trajectory step 18 says `fs = 3.17Hz`, and the README repeats “Frame rate: 3.17 Hz (~315 ms per frame).” The script simply preserves that frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `beh['SoundFr']` and `beh['StartFr']`.

ii. 
```python
start_frs = beh['StartFr']
sound_frs = beh['SoundFr']
...
sound_fr_rel = sound_frs[trial] - start_frs[trial]
```

iii. The trajectory’s behavior-variable excerpt explicitly defines `SoundFr` as the neural frame when the sound cue was delivered and `StartFr` as corridor entry, so the agent used those two fields.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the AI computes the cue frame relative to trial start, subtracts that from each frame index in the 32-frame window, and divides by `FS` to convert the signed offset to seconds. Negative values are before the cue and positive values after the cue.

ii. 
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The README describes this variable as “Time relative to sound cue onset (seconds), negative before cue,” which matches the implementation. The trajectory also shows the agent inspected `SoundFr` ranges before writing the code.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on the same 32 trial-start-aligned frames as the neural data, so every neural frame gets a corresponding signed time-to-cue value.

ii. 
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. The AI’s general design in the README and trajectory was to put all inputs and outputs on the same trial-start-aligned frame grid as `neural`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The implemented value is derived from session metadata `mname` and `datexp` across all deduplicated sessions for that mouse. Although `get_session_list()` stores `days` or `sess#`, the later computation ignores them.

ii. 
```python
'days': db.get('days', db.get('sess#', 0)),
...
def compute_day_of_training(mname, datexp, all_sessions):
    for k, s in all_sessions.items():
        if s['mname'] == mname:
            date = datetime.strptime(s['datexp'], '%Y_%m_%d')
```

iii. The README says “Days since first recording session for this mouse.” That matches the function body rather than the `days` metadata field captured earlier.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI sorts all recording dates for a mouse, subtracts the first date from the current session date in calendar days, then fills the entire trial with that constant scalar.

ii. 
```python
mouse_dates.sort()
current_date = datetime.strptime(datexp, '%Y_%m_%d')
day_idx = (current_date - mouse_dates[0]).days
...
input_trials[i][1, :] = float(day)
```

iii. The trajectory does not show a deeper justification beyond the README phrasing “Days since first recording session for this mouse.” There is no evidence the AI used the provided `days`/`sess#` fields for this variable.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the alignment to `StartFr` plus the global imaging frame rate `FS`. No additional raw behavior field is used.

ii. 
```python
FS = 3.17
...
time_since_start = np.arange(n_timepoints) / FS
```

iii. Once the agent chose `StartFr` as the alignment event, elapsed time from trial start follows directly from the frame index. That is the implicit justification in the code and README.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script creates `0, 1, 2, ..., 31`, divides by `FS`, and uses the resulting seconds as a monotonically increasing time series for every trial.

ii. 
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. The README describes the variable simply as “Time from corridor entry (seconds),” which matches this direct frame-to-seconds conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is defined on the exact same 32-frame trial-start-aligned window as the neural data, starting at 0 on the first saved neural frame.

ii. 
```python
start_fr = int(np.round(start_frs[trial]))
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
time_since_start = np.arange(n_timepoints) / FS
```

iii. The common frame grid is part of the overall design stated in the README and trajectory step 33.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `beh['isRew']`.

ii. 
```python
is_rew = beh['isRew']
...
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The behavior-variable excerpt loaded in the trajectory defines `isRew` as the boolean indicating whether the trial is a reward trial, which is the field the agent chose.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the per-trial boolean to `0.0` or `1.0` and broadcasts it across all 32 time bins of that trial.

ii. 
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. The README explicitly justifies this as “1 if rewarded corridor, 0 if not,” and the task instructions also defined reward availability as a per-trial discrete input.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The output category is derived from `beh['WallName']`, with the set of possible categories collected from `beh['UniqWalls']` across sessions.

ii. 
```python
wall_names = beh['WallName']
...
wall_name = wall_names[trial]
stim_cat = wall_name
...
for w in beh['UniqWalls']:
    all_stim_names.add(w)
```

iii. The behavior reference excerpt defines `WallName` as the stimulus name of each corridor and `UniqWalls` as the set of stimulus names in the session. The AI followed those field descriptions directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. For each trial, the script takes the `WallName` string, maps it to a global index using a sorted list of all stimulus names seen in all sessions, and writes that same index into every time bin of the trial. It does not pool the two `leaf1_swap` variants.

ii. 
```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
...
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx
```

iii. The README justifies this as “Stimulus category (e.g., circle1, leaf1, etc.) - per trial.” The trajectory also shows the agent built a global stimulus vocabulary from all sessions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `output` licking is derived from `beh['LickFr']` and `beh['LickTrind']`, together with the trial’s `StartFr`.

ii. 
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
...
trial_lick_mask = lick_trinds == trial
trial_lick_frs = lick_frs[trial_lick_mask]
fr_idx = int(np.round(lf)) - start_fr
```

iii. The trajectory’s behavior-variable excerpt defines `LickFr` as the neural frames when the animal licked and `LickTrind` as the trial stamp of each lick. That is the agent’s stated raw source.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI creates a zero vector of length 32 for each trial, finds licks assigned to that trial, rounds each lick frame to an integer, converts it to an offset from `StartFr`, and sets those bins to 1.

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

iii. The README summarizes the same choice as “Binary (0=no lick, 1=lick) - time varying.” The trajectory shows the agent inspected real `LickFr` and `LickTrind` examples before coding it.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned by converting lick frames into offsets from the trial’s `StartFr`, so the binary lick series lives on the same 32 neural frames saved for that trial.

ii. 
```python
fr_idx = int(np.round(lf)) - start_fr
if 0 <= fr_idx < n_timepoints:
    lick_binary[fr_idx] = 1.0
```

iii. The AI’s general alignment scheme was trial-start alignment for all streams. That is stated in the README and trajectory step 33.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from the per-frame neural-aligned position trace `beh['ft_Pos']`.

ii. 
```python
ft_pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_pos[start_fr:end_fr]
```

iii. The behavior reference excerpt defines `ft_Pos` as position inside VR for each neural frame, which is exactly the variable the AI used.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script slices `ft_Pos` over the same 32 trial-start-aligned frames as the neural data, then discretizes those raw positions directly. It does not interpolate by position, restrict to running frames, or stop at `GrayFr`.

ii. 
```python
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
...
def discretize_position(pos_values, n_bins=4):
    boundaries = [10, 20, 30]
    binned = np.digitize(pos_values, boundaries)
```

iii. The README says position is “discretized into 4 x 1m bins,” and the trajectory shows the agent chose a fixed 32-frame corridor-entry window rather than the reference code’s interpolation helpers such as `spk_pos_interp`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are hard-coded at 10, 20, and 30 decimeters, corresponding to 0-1 m, 1-2 m, 2-3 m, and 3-4 m bins. Any value above 40 dm, including gray-space values, also falls into the last bin.

ii. 
```python
def discretize_position(pos_values, n_bins=4):
    """... Values in gray space (>40dm) get assigned to the last bin."""
    boundaries = [10, 20, 30]
    binned = np.digitize(pos_values, boundaries)
    return binned
```

iii. The justification is explicit in the function docstring and README: the agent wanted four equal 1 m bins across the 4 m textured corridor, and then chose to absorb gray-space positions into the last bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position uses the same frame indices as the neural slice: `ft_Pos[start_fr:end_fr]`, where `start_fr` is the trial’s corridor-entry frame.

ii. 
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
trial_pos = ft_pos[start_fr:end_fr]
```

iii. The AI’s stated overall choice was trial-start alignment for all streams. The position code follows that exactly.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the neural-frame-aligned running-speed trace `beh['ft_RunSpeed']`.

ii. 
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
...
trial_speed = ft_speed[start_fr:end_fr]
```

iii. The behavior reference excerpt defines `ft_RunSpeed` as running speed for each neural frame. The AI used that field directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first discretizes each trial’s speed slice inside `extract_trial_data()`, then later recomputes speed bins using global quartile thresholds estimated from all non-NaN `ft_RunSpeed` values across all sessions and overwrites the per-trial result.

ii. 
```python
trial_speed = ft_speed[start_fr:end_fr]
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
...
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The trajectory and README justify the final version as using “4 quartile bins,” and the trajectory shows the agent explicitly computed “global speed quartiles” for consistency across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Final thresholds are the 25th, 50th, and 75th percentiles of all non-NaN `ft_RunSpeed` values pooled across all sessions. Values are digitized into categories `0, 1, 2, 3`, later labeled `Q1` to `Q4`.

ii. 
```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
boundaries = np.percentile(valid, percentiles)
...
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
...
['Q1', 'Q2', 'Q3', 'Q4']
```

iii. The README states “running_speed: Running speed discretized into 4 quartile bins,” and the conversion log printed the resulting global quartile boundaries.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by slicing the same 32 frame indices used for the neural data, starting at `StartFr`.

ii. 
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_speed = ft_speed[start_fr:end_fr]
```

iii. This is part of the AI’s consistent trial-start frame-grid alignment described in the README and trajectory.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic data are handled only in a limited way. Sessions are skipped if the behavior key is absent, spike loading fails, retinotopy loading fails, there are too few trials, or too few kept neurons remain. Trials are skipped only for out-of-bounds frame windows. NaNs in speed are ignored when computing quartiles, but otherwise missing values are not treated specially.

ii. 
```python
if beh_key not in beh_cache[exp_type]:
    print(f"  WARNING: beh_key {beh_key} not in {exp_type}, skipping")
    continue
...
except Exception as e:
    print(f"  ERROR loading spk: {e}")
    continue
...
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    ...
valid_speed = speeds[~np.isnan(speeds)]
```

iii. There is no separate justification file in `/app`. The practical rationale visible in the code is robustness of execution rather than principled data curation: skip what breaks, otherwise keep the data.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the very large spike files, concatenating all imaging planes, computing global speed quartiles across all sessions, and iterating trial-by-trial to build full per-trial arrays while holding everything in RAM. The trajectory later confirmed the full run became memory-bound.

ii. 
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
all_speeds = np.concatenate(all_speeds)
...
for trial in range(ntrials):
    ...
    neural_trials.append(trial_neural)
```

iii. In trajectory step 113 the agent explicitly noted that the full dataset build was using too much memory because it was “building up in RAM.” That matches the structure of `process_all_sessions()`.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could have been vectorized: the per-trial extraction loop, the inner lick loop over every lick frame, the pass that fills `day_of_training`, the pass that overwrites running-speed bins, and the pass that fills stimulus indices.

ii. 
```python
for trial in range(ntrials):
    ...
    for lf in trial_lick_frs:
        ...
...
for i in range(len(input_trials)):
    if valid_mask[i]:
        input_trials[i][1, :] = float(day)
...
for i in range(len(output_trials)):
    if valid_mask[i]:
        ...
```

iii. The trajectory does not contain a separate efficiency rationale. This conclusion follows directly from the script’s structure.

## 12-c. What processing does the code repeat multiple times?

i. The code discretizes running speed twice, walks through all trials multiple extra times to fill day-of-training and stimulus labels, and repeatedly indexes behavior arrays from `StartFr` inside separate passes instead of computing all per-trial annotations in one pass.

ii. 
```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
for i in range(len(input_trials)):
    ...
for i in range(len(output_trials)):
    ...
for i in range(len(output_trials)):
    ...
```

iii. This repeated processing is visible in the final script and was not otherwise justified in the trajectory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script reads `ft_move` but never uses it, reads `stim_id` and `uniq_walls` in `extract_trial_data()` without using them, loads `exp_info` a second time in `process_all_sessions()` without using it, and computes the first per-trial speed discretization only to overwrite it later with global quartile bins.

ii. 
```python
ft_move = beh['ft_move'][:n_frames]
...
stim_id = beh.get('stim_id', None)
uniq_walls = beh['UniqWalls']
...
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
...
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. There is no explicit justification for these extra steps. They appear to be remnants of exploratory development rather than intentional downstream requirements.
