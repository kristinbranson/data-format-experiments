# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `beh/Imaging_Exp_info.npy` as a master index, builds one unique session per `(mname, datexp, blk)`, caches behavior files by experiment type, then loads spikes and retinotopy per session during processing. Unlike the reference, it resolves duplicate session listings by choosing the behavior entry with the most non-`NaN` `stim_id` values.

ii. 
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
```
```python
if key not in sessions or n_stim > sessions[key]['n_stim']:
    sessions[key] = {
        'mname': db['mname'],
        'datexp': db['datexp'],
        'blk': db['blk'],
        'exp_type': exp_type,
        'beh_key': beh_key,
        'n_stim': n_stim,
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

iii. There is no `CONVERSION_NOTES.md`; the rationale comes from the trajectory and README. In the trajectory, the AI explicitly says each unique neural recording should map to one decoder session and that it should pick the behavior interpretation with the “most complete stimulus coverage.”

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The output `subjects` list is the sorted set of unique mouse names, and each session’s `subject_idx` is assigned from that list.

ii. 
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```
```python
subject_idx_all.append(subject_to_idx[mname])
```

iii. The AI’s code does not justify this separately; it follows the obvious mouse identifier in the dataset and matches the README’s description of “19 mice.”

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` triple. If the same neural session appears multiple times across experiment types or swap variants, the AI keeps only one session record and chooses the behavior entry with the highest count of non-missing `stim_id` values.

ii. 
```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
stimtype = db.get('stimtype', '')
beh_key = key if not stimtype else f'{key}_{stimtype}'
```
```python
if key not in sessions or n_stim > sessions[key]['n_stim']:
    sessions[key] = {
        ...
        'beh_key': beh_key,
        'n_stim': n_stim,
```

iii. In the trajectory, the AI states that the same neural recording can appear in multiple behavior datasets and that it should select the one with the “most complete stimulus information” instead of taking the first occurrence.

## 1-d. How are the data split into trials?

i. Trials are split by taking a fixed 32-frame window beginning at each trial’s `StartFr` value. The code does not use `ft_trInd` or `ft_CorrSpc` to define within-trial frames; it assumes a trial is the 32 frames immediately after rounded corridor entry.

ii. 
```python
start_frs = beh['StartFr']
...
for trial in range(ntrials):
    start_fr = int(np.round(start_frs[trial]))
    end_fr = start_fr + n_timepoints
```
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The README and trajectory justify this as “32 frames from corridor entry,” with the AI reasoning that about 10 seconds at 3.17 Hz should cover most corridor traversals while giving fixed-length trial tensors.

## 1-e. How are trials filtered based on quality controls?

i. Trials are marked invalid only if the fixed 32-frame window would go out of bounds of the imaged frames. Sessions are also skipped if they have fewer than 10 total trials before extraction or fewer than 10 valid trials after extraction.

ii. 
```python
MIN_TRIALS = 10
```
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    neural_trials.append(None)
    input_trials.append(None)
    output_trials.append(None)
```
```python
if beh['ntrials'] < MIN_TRIALS:
    ...
if len(valid_neural) < MIN_TRIALS:
    ...
```

iii. The AI does not provide a strong paper-based justification. The trajectory frames the filtering as a practical decoder requirement rather than a reference-matching curation rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spk_data['spks']` in each session’s spike file, concatenated across imaging planes. Brain-region assignments are derived from retinotopy `iarea`.

ii. 
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```
```python
dtrans = np.load(ret_path, allow_pickle=True)
return dtrans['iarea']
```

iii. The README describes the neural source as “Deconvolved calcium fluorescence traces from Suite2p” with retinotopy-based region assignment.

## 2-b. How is the `neural` data processed?

i. After concatenating imaging planes and filtering neurons by visual area, the AI slices a 32-frame window per trial starting at `StartFr` and stores it as `float32`. It does not interpolate, pad, or otherwise transform the traces.

ii. 
```python
spk_filtered = spk[valid_neurons]
```
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The README says the processing is “Fixed-length time windows per trial” aligned to corridor entry. The trajectory shows the AI chose this for fixed-size decoder inputs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if retinotopy maps them into one of four visual regions: `V1`, `mHV`, `lHV`, or `aHV`. Sessions with fewer than 10 retained neurons are skipped entirely.

ii. 
```python
region_idx = np.full(len(iarea), -1, dtype=int)
region_idx[iarea == 8] = 0
region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
region_idx[(iarea == 5) | (iarea == 6)] = 2
region_idx[(iarea == 3) | (iarea == 4)] = 3
```
```python
valid_neurons = region_idx >= 0
...
if valid_neurons.sum() < 10:
    print(f"  Skipping: only {valid_neurons.sum()} visual cortex neurons")
    continue
```

iii. The README justifies the region filter as “All visual cortex neurons ... kept; neurons outside visual cortex excluded.” The extra minimum-neuron session filter is not separately justified.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data is aligned to trial start, interpreted as corridor entry, by taking frames from `StartFr` onward. The alignment is purely frame-index based, using the rounded `StartFr` as the first bin.

ii. 
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The README explicitly states “Align to trial start (corridor entry)” and “Window: 32 frames from corridor entry.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame rate, `FS = 3.17`, so each time bin is about 315.5 ms. No temporal rebinning is applied.

ii. 
```python
FS = 3.17  # Calcium imaging frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms per frame
```
```python
'time_bin_size': TIME_BIN_MS,
```

iii. The README says the neural data is sampled at about 3.17 Hz and treats that as the decoder time grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives `time_to_sound_cue` from `SoundFr`, `StartFr`, and the constant frame rate `FS`. It does not use the recorded frame timestamps `ft`.

ii. 
```python
start_frs = beh['StartFr']
sound_frs = beh['SoundFr']
```
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The README describes this as “Time relative to sound cue onset (seconds), negative before cue.” The trajectory shows the AI accepted large cue offsets as a consequence of delayed cues in long trials.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code computes the cue’s frame offset from `StartFr`, subtracts that offset from `0..31`, and divides by `FS` to express it in seconds. The resulting sign convention is negative before the cue and positive after it.

ii. 
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. The trajectory explicitly discusses “negative before cue” and treats extreme negative values as mathematically correct when cues happen long after trial start.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is defined on exactly the same 32 bins as the neural window, using the trial’s `StartFr` as time zero and one value per frame in the extracted neural segment.

ii. 
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. The AI’s general justification is that all decoder variables should share the same fixed frame-aligned trial window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `day_of_training` is derived from mouse name `mname` and session date string `datexp` across all sessions. It is not taken from a raw “day” field.

ii. 
```python
for k, s in all_sessions.items():
    if s['mname'] == mname:
        date = datetime.strptime(s['datexp'], '%Y_%m_%d')
        mouse_dates.append(date)
```

iii. The README describes this as “Days since first recording session for this mouse.”

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI parses each session’s calendar date, finds the first date for that mouse, computes the integer day difference to the current session, and broadcasts that constant value across all 32 bins of every trial in the session.

ii. 
```python
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

iii. The only explicit justification is in the README; the trajectory does not defend this against the reference solution’s session-count definition.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives `time_since_trial_start` from the fixed frame index within the 32-frame trial window and the constant frame rate `FS`. It does not use `ft` timestamps except indirectly through `StartFr` for alignment.

ii. 
```python
time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. The README describes this input as “Time from corridor entry (seconds),” consistent with counting elapsed time from the aligned start frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code sets the first bin to 0 and increments by `1 / FS` for each later bin, producing a simple linear time axis `[0, 1/FS, 2/FS, ...]` for every trial.

ii. 
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. The AI’s rationale is implicit: once trials are represented as fixed 32-frame windows, elapsed time is just frame index divided by frame rate.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by construction to the same 32 neural frames starting at `StartFr`, with time 0 assigned to the first neural frame of each trial.

ii. 
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. The AI consistently treats all signals as sharing the same extracted frame window.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is taken directly from per-trial `isRew`.

ii. 
```python
is_rew = beh['isRew']
```
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The README describes this as “1 if rewarded corridor, 0 if not.”

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation is applied beyond broadcasting the per-trial `isRew` value across all 32 time bins of the trial.

ii. 
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The AI provides no additional justification; this is a direct per-trial flag adapted to a time-varying input matrix.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The stimulus output is derived from `WallName` for each trial.

ii. 
```python
wall_names = beh['WallName']
...
wall_name = wall_names[trial]
stim_cat = wall_name
```

iii. The README says the output is the trial’s stimulus category, for example `circle1` or `leaf1`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps the full wall-name categories rather than collapsing them to four base textures. It first collects all unique wall names across sessions, sorts them, maps each trial’s `WallName` to an integer index in that 15-class list, and broadcasts that index across the full trial.

ii. 
```python
for w in beh['UniqWalls']:
    all_stim_names.add(w)
...
all_stim_sorted = sorted(str(s) for s in all_stim_names)
```
```python
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx
```
```python
output_values = [
    [str(s) for s in all_stim_sorted],
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['Q1', 'Q2', 'Q3', 'Q4'],
]
```

iii. The README explicitly states “15 stimulus categories across all sessions,” so the AI intentionally used the full stimulus identities instead of broad texture groups.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind`: the code uses `LickTrind` to find licks belonging to the current trial and `LickFr` to place them into per-frame bins.

ii. 
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
```
```python
trial_lick_mask = lick_trinds == trial
if trial_lick_mask.any():
    trial_lick_frs = lick_frs[trial_lick_mask]
```

iii. The AI does not explain this separately, but the code reflects a trial-by-trial reconstruction of licking from lick event frame numbers.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, it initializes a 32-bin zero vector, rounds each lick frame to the nearest integer frame, subtracts the rounded trial start frame, and sets bins with licks to 1. Multiple licks in the same bin remain a single 1.

ii. 
```python
lick_binary = np.zeros(n_timepoints, dtype=float)
...
for lf in trial_lick_frs:
    fr_idx = int(np.round(lf)) - start_fr
    if 0 <= fr_idx < n_timepoints:
        lick_binary[fr_idx] = 1.0
```

iii. The AI’s implicit rationale is to make licking a binary time-varying output on the same fixed frame grid as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned to the same 32-frame `StartFr`-anchored window as the neural activity, using frame offsets relative to `start_fr`.

ii. 
```python
fr_idx = int(np.round(lf)) - start_fr
if 0 <= fr_idx < n_timepoints:
    lick_binary[fr_idx] = 1.0
```
```python
trial_output = np.stack([
    np.zeros(n_timepoints, dtype=int),
    lick_binary.astype(int),
    pos_binned.astype(int),
    speed_binned.astype(int),
])
```

iii. This follows the AI’s overall fixed-window alignment strategy.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from frame-level `ft_Pos`.

ii. 
```python
ft_pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_pos[start_fr:end_fr]
```

iii. The README states that position is a time-varying output discretized from the corridor position.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI takes the 32 `ft_Pos` values starting at `StartFr`, bins them by decimeter boundaries into four 1 m bins, and uses those categorical values directly for the output.

ii. 
```python
def discretize_position(pos_values, n_bins=4):
    boundaries = [10, 20, 30]
    binned = np.digitize(pos_values, boundaries)
    return binned
```
```python
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. The README justifies this as “Position in corridor discretized into 4 x 1m bins.”

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The code uses thresholds at 10, 20, and 30 decimeters, which produce four categories corresponding to `0-1m`, `1-2m`, `2-3m`, and `3-4m`. Values beyond 40 dm are still assigned to the last bin.

ii. 
```python
boundaries = [10, 20, 30]  # in decimeters
binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
```
```python
['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The README and code comments explicitly describe the categories as 1 m bins over the textured corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position values are taken from the same 32-frame `StartFr`-aligned trial window used for neural data.

ii. 
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. The AI’s fixed-window trial representation is the only stated alignment principle.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from frame-level `ft_RunSpeed`.

ii. 
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
```
```python
trial_speed = ft_speed[start_fr:end_fr]
```

iii. The README describes running speed as a time-varying output later discretized into quartiles.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first defines a per-array percentile-based discretizer, then separately computes global speed quartile boundaries across all sessions, and finally re-discretizes each trial’s 32-frame speed segment with those global thresholds.

ii. 
```python
def discretize_speed(speed_values, n_bins=4):
    valid = speed_values[~np.isnan(speed_values)]
    ...
    boundaries = np.percentile(valid, percentiles)
    binned = np.digitize(speed_values, boundaries)
    return binned
```
```python
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
speed_binned = np.digitize(trial_speed, speed_quartiles)
output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The conversion log and README say the agent chose “global speed quartiles” for consistency across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The final thresholds are the global 25th, 50th, and 75th percentiles computed across all non-`NaN` framewise speeds from all sessions. The output values are labeled `Q1` to `Q4`.

ii. 
```python
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
print(f"Global speed quartile boundaries: {speed_quartiles}")
```
```python
output_values = [
    ...,
    ['Q1', 'Q2', 'Q3', 'Q4'],
]
```

iii. The AI explicitly reports “Computing global speed quartiles...” in `conversion_full_out.txt`, so this was an intentional dataset-wide thresholding choice.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by slicing the same `start_fr:end_fr` window used for neural data and discretizing those 32 framewise values.

ii. 
```python
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
...
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
```

iii. As with the other time-varying outputs, the alignment follows the AI’s fixed 32-frame trial window.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles issues mostly by skipping invalid items rather than repairing or trimming them. Trials are dropped if their 32-frame window would exceed the imaged frames, sessions are skipped if behavior keys, spike files, or retinotopy files cannot be loaded, and licks outside the current trial window are ignored. The code also truncates `ft_Pos`, `ft_move`, and `ft_RunSpeed` to `n_frames`, but it does not systematically use the raw per-frame timestamp array to reconcile behavior extending beyond imaging.

ii. 
```python
ft_pos = beh['ft_Pos'][:n_frames]
ft_move = beh['ft_move'][:n_frames]
ft_speed = beh['ft_RunSpeed'][:n_frames]
```
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
    ...
    continue
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

iii. There is no explicit justification beyond practical robustness. The agent’s notes file that was supposed to document such handling is missing.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading and concatenating the large spike files for every session, the extra full-dataset pass to gather global speed quartiles, and the per-trial extraction loops that build full `(neurons x 32)` tensors. The trajectory also shows the agent was primarily worried about neural-data I/O and dataset size.

ii. 
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```
```python
for key, sess_info in sessions.items():
    ...
    all_speeds.append(valid_speed)
all_speeds = np.concatenate(all_speeds)
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
for trial in range(ntrials):
    ...
    trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. In the trajectory, the AI repeatedly discusses session loading times, spike-file size, and the impracticality of very large pickles, which points to I/O and tensor construction as the dominant costs.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized: the per-trial extraction loop, the per-lick loop inside each trial, the loop that fills `day_of_training`, the loop that re-discretizes speed per trial, and the loop that fills stimulus indices. These are all explicit Python loops over data that is mostly already array-structured.

ii. 
```python
for trial in range(ntrials):
    ...
    for lf in trial_lick_frs:
        fr_idx = int(np.round(lf)) - start_fr
```
```python
for i in range(len(input_trials)):
    if valid_mask[i]:
        input_trials[i][1, :] = float(day)
```
```python
for i in range(len(output_trials)):
    if valid_mask[i]:
        trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
        speed_binned = np.digitize(trial_speed, speed_quartiles)
        output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The AI does not discuss vectorization, but the loops are visible in the implementation.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it scans behavior once to collect all stimulus names, again to collect all speeds for quartiles, then again to process sessions; it computes a preliminary speed discretization inside `extract_trial_data` and later overwrites it with global discretization; and `compute_day_of_training` re-scans all sessions for each processed session.

ii. 
```python
for key, sess_info in sessions.items():
    ...
    for w in beh['UniqWalls']:
        all_stim_names.add(w)
```
```python
for key, sess_info in sessions.items():
    ...
    all_speeds.append(valid_speed)
```
```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
```
```python
for i in range(len(output_trials)):
    if valid_mask[i]:
        ...
        speed_binned = np.digitize(trial_speed, speed_quartiles)
        output_trials[i][3, :] = speed_binned.astype(int)
```

iii. There is no explicit justification for these repeated passes. They appear to be artifacts of how the script evolved.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several intermediate values that are unused or later discarded: `ft_move`, `stim_id`, `uniq_walls`, and the initial per-trial speed discretization are not needed in the final dataset; output row 0 is first filled with zeros as a placeholder and later overwritten; and the code keeps metadata fields like `stim_id`/`n_stim` mainly to choose session variants rather than to build the final decoder data.

ii. 
```python
ft_move = beh['ft_move'][:n_frames]
...
stim_id = beh.get('stim_id', None)
uniq_walls = beh['UniqWalls']
```
```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
output_trials[i][3, :] = speed_binned.astype(int)
```
```python
trial_output = np.stack([
    np.zeros(n_timepoints, dtype=int),  # placeholder for stim category
    lick_binary.astype(int),
    pos_binned.astype(int),
    speed_binned.astype(int),
])
```

iii. The trajectory does not justify these extra computations. They look like implementation leftovers rather than deliberate downstream requirements.
