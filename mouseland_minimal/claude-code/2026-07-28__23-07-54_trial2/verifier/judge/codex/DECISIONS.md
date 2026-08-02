# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy`, collapses it to one record per unique neural recording key `mname_datexp_blk`, then loads one behavior dictionary `Beh_{exp_type}.npy` per experiment type, one spike file per session from `data/spk`, and one retinotopy file per session from `data/retinotopy`. When multiple behavior views exist for the same neural recording, it keeps the one with the largest number of non-NaN `stim_id` entries.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

for exp_type in exp_info:
    for db in exp_info[exp_type]:
        key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
        stimtype = db.get('stimtype', '')
        beh_key = key if not stimtype else f'{key}_{stimtype}'
```

```python
spk_path = os.path.join(root, 'spk', fn)
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The justification appears in trajectory step 79: the agent decided that duplicated `Imaging_Exp_info` rows were “the same recording but with different stim_id assignments” and that it should “select the behavior dataset with the most complete stimulus information for each unique neural recording.”

## 1-b. How are the data split into subjects?

i. Subjects are split by unique mouse name `mname`. The script builds `subjects` as a sorted set of all `mname` values and maps each kept session to `subject_idx`.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_all.append(subject_to_idx[mname])
```

iii. The trajectory does not show a separate long justification for this; it is an implementation choice consistent with the target schema.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `mname_datexp_blk` keys, sorted lexicographically. This collapses the 142 `Imaging_Exp_info` entries down to 89 unique neural recordings.

ii.
```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
...
sorted_keys = sorted(sessions.keys())
for sess_idx, key in enumerate(sorted_keys):
    sess_info = sessions[key]
```

iii. In trajectory step 33 the agent stated that “each unique (mname, datexp, blk) combination” should define a session. In step 79 it explicitly justified collapsing repeated behavior views onto one neural recording.

## 1-d. How are the data split into trials?

i. Trials are split by iterating `range(beh['ntrials'])` and taking a fixed 32-frame window starting at each rounded `StartFr`. The code does not use `GrayFr`, `EndFr`, `ft_trInd`, or the reference position interpolation pipeline to define trial contents.

ii.
```python
for trial in range(ntrials):
    start_fr = int(np.round(start_frs[trial]))
    end_fr = start_fr + n_timepoints
    trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. Trajectory step 33 says the agent chose “fixed-size time bins across all trials and sessions” and planned to “extract a consistent number of time bins per trial regardless of how long the mouse takes to traverse.”

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. A trial is dropped only if the 32-frame window would start before frame 0 or end after the recording. Afterward, the whole session is dropped if it has fewer than 10 valid trials. There is no trial-level filtering for non-running periods, gray-space periods, cue anomalies, or missing behavior variables.

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    ...
    continue
...
if len(valid_neural) < MIN_TRIALS:
    print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
    continue
```

iii. The trajectory shows the agent noticed very long trial durations and late cue times in step 33 and step 50, but it did not add extra trial QC beyond the bounds check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the Suite2p deconvolved fluorescence traces stored as `spk_data['spks']` in each `*_neural_data.npy` file.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The README and trajectory repeatedly describe the neural source as deconvolved calcium/Suite2p output.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, filters neurons to four visual cortex groups using retinotopy, and then slices raw deconvolved traces into 32-frame trial windows aligned to `StartFr`. It does not interpolate neural data to 60 position bins, restrict to running frames, or normalize the activity the way the reference analysis code often does.

ii.
```python
region_idx = get_brain_region_idx(iarea)
valid_neurons = region_idx >= 0
spk_filtered = spk[valid_neurons]
...
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. Trajectory step 33 records the key decision: the agent considered using the reference position-interpolated representation, but rejected it as too expensive and instead chose fixed raw-frame windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopy: keep only neurons mapped to `V1`, `mHV`, `lHV`, or `aHV`; exclude neurons with no mapped region. Entire sessions are skipped if fewer than 10 such neurons remain. No frame-level QC is applied to neural data after loading.

ii.
```python
region_idx = get_brain_region_idx(iarea)
valid_neurons = region_idx >= 0

if valid_neurons.sum() < 10:
    print(f"  Skipping: only {valid_neurons.sum()} visual cortex neurons")
    continue
```

iii. The docstring and trajectory step 35 say the agent matched the retinotopy area mapping and exclusion of neurons outside visual cortex.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start, operationalized as corridor entry at `StartFr`.

ii.
```python
start_frs = beh['StartFr']
...
start_fr = int(np.round(start_frs[trial]))
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. This choice is explicit in the script docstring and in trajectory step 33, where the agent says “Align to trial start (corridor entry).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one calcium imaging frame, `1000 / 3.17 = 315.46 ms`. No temporal rebinning is applied; the code uses raw frames directly and fixes the number of frames per trial at 32.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS
N_TIMEPOINTS = 32
```

iii. Trajectory step 33 states the agent chose to “use the calcium imaging frame rate as my time bins (~315 ms per frame).”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and `StartFr`.

ii.
```python
start_frs = beh['StartFr']
sound_frs = beh['SoundFr']
sound_fr_rel = sound_frs[trial] - start_frs[trial]
```

iii. The trajectory discusses cue timing repeatedly, especially in step 50 when the agent examined extreme cue offsets.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code computes frame offsets relative to the cue, then divides by `FS` to express them in seconds. The resulting vector is negative before the cue and positive after the cue.

ii.
```python
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. Trajectory step 50 explicitly says the large negative values are “mathematically correct” because they represent frames before a late cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned to the same 32-frame trial window used for neural data, with time index 0 corresponding to the first neural frame after rounded `StartFr`.

ii.
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
```

iii. The alignment follows the agent’s broader step-33 decision to align every variable to corridor entry and keep fixed-length time windows.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session dates `datexp` collected from the collapsed `sessions` dictionary, not from a dedicated per-session training-day field in the behavior metadata.

ii.
```python
for k, s in all_sessions.items():
    if s['mname'] == mname:
        date = datetime.strptime(s['datexp'], '%Y_%m_%d')
        mouse_dates.append(date)
```

iii. The trajectory does not show a separate defense of this choice. The code stores `days`/`sess#` in the session dictionary but never uses them.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI sorts recording dates for a mouse, subtracts the first date from the current session date, and uses the resulting number of calendar days. It then repeats that scalar across all time bins in every trial of the session.

ii.
```python
mouse_dates.sort()
current_date = datetime.strptime(datexp, '%Y_%m_%d')
day_idx = (current_date - mouse_dates[0]).days
...
input_trials[i][1, :] = float(day)
```

iii. This is an inferred implementation decision from the code itself; there is no matching explanation in a notes file because `CONVERSION_NOTES.md` is missing.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. The AI did not create an `environment type` input at all, so it does not derive this variable from any raw field.

ii.
```python
input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
```

iii. There is no trajectory evidence that the agent attempted to add this variable. It followed the decoder-input list from the user instructions instead.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. No processing is performed because the variable is omitted.

ii.
```python
trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. The omission is implicit in both the code and the README.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the trial alignment point `StartFr` plus the global frame rate `FS`; it is not read from a raw timestamp vector such as `beh['ft']`.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. The trajectory shows the agent intentionally used imaging frames as the time basis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI creates a simple linear ramp `0, 1/FS, 2/FS, ...` for 32 bins and copies that same ramp into every trial.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. This follows directly from the step-33 decision to use raw frame bins instead of the reference position interpolation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned to the neural window by construction: the first value is the first neural frame after `StartFr`, and the vector has the same 32 time bins as the neural matrix.

ii.
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
time_since_start = np.arange(n_timepoints) / FS
```

iii. The script’s core alignment logic uses the same `start_fr` for all trial-wise variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `beh['isRew']`.

ii.
```python
is_rew = beh['isRew']
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The agent’s README describes this variable as “1 if rewarded corridor, 0 if not.”

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code converts the per-trial Boolean into a constant 32-bin float vector.

ii.
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. No extra justification appears in the trajectory; this is a straightforward reading of the instructions.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` for each trial and from `UniqWalls` when collecting all possible categories across sessions.

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

iii. The trajectory and README both frame this output as the corridor texture identity (for example `circle1`, `leaf2`).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects all wall names observed across kept sessions, sorts them globally, maps each trial’s `WallName` to an integer index, and writes that integer into every time bin of the trial.

ii.
```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
...
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx
```

iii. In the trajectory the agent treats this as a per-trial categorical label but keeps it time-varying in shape to satisfy the decoder format.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, `LickTrind`, and the per-trial alignment frame `StartFr`.

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
...
fr_idx = int(np.round(lf)) - start_fr
```

iii. The trajectory step 32 shows the agent explicitly inspected `LickFr` and `LickTrind` to decide how to build a per-frame binary lick output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI creates a zero vector of length 32 for each trial and sets entries to 1 when a rounded lick frame falls inside that trial window.

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

iii. The justification is implicit in the trajectory: the agent wanted a time-varying binary output aligned to trial start.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned by subtracting the trial’s `start_fr` from each lick frame and placing the result into the same 32-bin trial window used for neural data.

ii.
```python
fr_idx = int(np.round(lf)) - start_fr
if 0 <= fr_idx < n_timepoints:
    lick_binary[fr_idx] = 1.0
```

iii. This follows the global corridor-entry alignment choice from trajectory step 33.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level position `beh['ft_Pos']`.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[start_fr:end_fr]
```

iii. The code ignores the reference interpolation inputs `ft_PosCum`, `ft_trInd`, and `ft_move` even though the trajectory acknowledged them.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI slices raw `ft_Pos` over the 32-frame time window and directly discretizes those decimeter positions. It does not interpolate activity or behavior to uniform position bins per trial.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. Trajectory step 33 records that the agent considered the reference position-interpolation approach but deliberately replaced it with raw time windows for practicality.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is thresholded with fixed boundaries at 10, 20, and 30 decimeters, producing four bins corresponding to `0-1m`, `1-2m`, `2-3m`, and `3-4m`. Any values above 40 dm, including gray-space values, also land in the last bin.

ii.
```python
boundaries = [10, 20, 30]
binned = np.digitize(pos_values, boundaries)
...
['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The docstring states the rationale explicitly and notes that gray-space values are forced into the last bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by taking the same raw frame indices `start_fr:end_fr` used for the neural data. Because the window is fixed at 32 frames rather than bounded by `GrayFr`/`EndFr`, some trials can include gray-space or even next-trial positions.

ii.
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
trial_pos = ft_pos[start_fr:end_fr]
```

iii. The trajectory shows the agent knew trials had widely varying durations but still chose the fixed 32-frame alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['ft_RunSpeed']`.

ii.
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_speed[start_fr:end_fr]
```

iii. The behavior-variable notebook and the trajectory both identify `ft_RunSpeed` as the frame-level running-speed signal.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes per-trial bins locally, then overwrites them using global quartile boundaries computed from all non-NaN `ft_RunSpeed` values across all kept sessions. It does not restrict the quartiles to running-only frames, so the first quartile boundary is 0 because stationary periods are included.

ii.
```python
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
...
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. Trajectory step 50 notes the suspicious quartile result: “Q1 is at 0.0—indicating many frames where the mouse isn't moving at all.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is thresholded into four quantile bins using dataset-wide percentile boundaries at 25%, 50%, and 75%.

ii.
```python
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
speed_binned = np.digitize(trial_speed, speed_quartiles)
...
['Q1', 'Q2', 'Q3', 'Q4']
```

iii. The agent justified this as “global speed quartiles” for consistency across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is aligned by slicing `ft_RunSpeed` over the same fixed raw-frame trial window as the neural data.

ii.
```python
trial_speed = ft_speed[start_fr:end_fr]
trial_output = np.stack([
    ...,
    speed_binned.astype(int),
])
```

iii. This is part of the same fixed-window corridor-entry alignment chosen in trajectory step 33.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is ad hoc. Out-of-bounds trials are set to `None` and later dropped. Missing speed values are ignored when computing quartiles, and if an entire vector were NaN the code would output zeros. Missing stimulus mappings fall back to category index 0. There is no dedicated handling for unusually long cue delays, gray-space contamination, duplicated behavior views beyond the heuristic collapse, or other anomalies.

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    ...
    continue
```

```python
valid = speed_values[~np.isnan(speed_values)]
if len(valid) == 0:
    return np.zeros_like(speed_values, dtype=int)
...
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
```

iii. Trajectory step 50 shows the agent recognized extreme cue timings and zero-heavy speed quartiles, but the final code leaves those issues in place.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading multi-gigabyte spike files, concatenating all planes into one dense neuron-by-frame matrix, looping over every trial in every session to slice neural and behavior arrays, computing global speed quartiles over all sessions, and serializing the enormous pickle output.

ii.
```python
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
for trial in range(ntrials):
    ...
all_speeds = np.concatenate(all_speeds)
...
pickle.dump(data, f)
```

iii. The trajectory repeatedly comments on runtime and memory pressure, especially when loading 4-7 GB spike files and writing multi-GB output.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial extraction loop, the per-lick loop inside each trial, the separate post-processing loops that fill day-of-training, re-discretize speed, and fill stimulus categories, and the first-pass loop over behavior files to gather categories and speeds could all be reduced or vectorized.

ii.
```python
for trial in range(ntrials):
    ...
    for lf in trial_lick_frs:
        ...
```

```python
for i in range(len(input_trials)):
    ...
for i in range(len(output_trials)):
    ...
for i in range(len(output_trials)):
    ...
```

iii. No explicit justification is given; this follows directly from the final implementation structure.

## 12-c. What processing does the code repeat multiple times?

i. The code discretizes running speed twice, scans behavior files once to gather stimulus names and again to gather all speeds, iterates over trials once to build outputs and then over trials again to fill day-of-training, speed bins, and stimulus IDs, and repeatedly rounds/derives frame windows from `StartFr`.

ii.
```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. This repeated work is visible in the code; there is no separate note defending it.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes an initial per-trial speed discretization that is always overwritten later, loads `ft_move` but never uses it, stores `days` in the session metadata without using it for `day_of_training`, defines `STIM_NAMES` but never uses it, and expands per-trial stimulus labels into 32-bin time series even though they are constant within trial.

ii.
```python
ft_move = beh['ft_move'][:n_frames]
...
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
output_trials[i][3, :] = speed_binned.astype(int)
```

```python
STIM_NAMES = {
    0: 'circle1',
    ...
}
```

iii. These are direct observations from the final script. The trajectory does not indicate the agent noticed most of these redundancies.
