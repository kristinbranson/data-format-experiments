# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. It reads `Imaging_Exp_info.npy`, caches each selected `Beh_<exp_type>.npy`, and loads the spike and retinotopy file per session. Duplicate recording IDs are resolved by retaining the index entry with the most non-NaN `stim_id` values.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
...
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
spk = load_spk(mname, datexp, blk, DATA_ROOT)
iarea = load_retino(mname, datexp, DATA_ROOT)
```

iii. The trajectory says duplicate behavior views share the recording and that choosing the view with the most complete stimulus information preserves all trials. It intended all 89 sessions. The final conversion was killed for memory use, however, leaving `converted_data.pkl` truncated and unreadable.

## 1-b. How are the data split into subjects?

i. Subjects are mouse names (`mname`). A sorted unique subject list is built from selected sessions, and each retained session receives the corresponding integer index.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_all.append(subject_to_idx[mname])
```

iii. The trajectory treated `mname` as the explicit mouse identifier; no inferred split was needed.

## 1-c. How are the data split into sessions?

i. A session is uniquely keyed by mouse, date, and block. Repeated index entries are collapsed, choosing the entry with the greatest number of non-NaN stimulus IDs.

ii.
```python
key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
...
if key not in sessions or n_stim > sessions[key]['n_stim']:
    sessions[key] = {...}
```

iii. The agent reasoned that repeated entries are alternate analyses of the same neural recording and selected the most stimulus-complete behavior view.

## 1-d. How are the data split into trials?

i. Each behavior trial is indexed from `0` to `ntrials-1`. It is represented by a fixed 32-frame slice beginning at rounded `StartFr`, rather than by `ft_trInd`/`ft_CorrSpc` through the actual texture traversal.

ii.
```python
for trial in range(ntrials):
    start_fr = int(np.round(start_frs[trial]))
    end_fr = start_fr + n_timepoints
    trial_neural = spk[:, start_fr:end_fr]
```

iii. The agent estimated that a 6 m corridor at 60 cm/s lasts about 10 seconds or 32 imaging frames and chose that common window.

## 1-e. How are trials filtered based on quality controls?

i. A trial is valid only if its fixed window is in imaging bounds. Sessions are prefiltered at 10 raw trials and postfiltered at 10 valid trials; load failures, behavior-key failures, sessions with fewer than 10 visual neurons, and affected sessions are skipped. No stalled-trial/99th-percentile filter is used.

ii.
```python
if start_fr < 0 or end_fr > n_frames:
    valid_mask.append(False)
...
if beh['ntrials'] < MIN_TRIALS: continue
...
if len(valid_neural) < MIN_TRIALS: continue
```

iii. The stated motive for `MIN_TRIALS = 10` was decoder evaluation. The trajectory did not justify the absence of the reference stalled-trial control.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each session's `spks` arrays (concatenated across planes); `iarea` supplies region labels used for filtering.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
return dtrans['iarea']
```

iii. The agent identified `spks` as already-deconvolved Suite2p activity and retinotopy as the region source.

## 2-b. How is the `neural` data processed?

i. The concatenated traces are neuron-filtered, sliced into 32-frame trials, and cast/copied as float32. There is no normalization, deconvolution, padding, or temporal averaging.

ii.
```python
spk_filtered = spk[valid_neurons]
...
trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The trajectory notes the raw neural arrays are already float32 and the paper analyses deconvolved traces; it retained that representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons mapped to V1, mHV, lHV, or aHV are retained. A whole session is additionally skipped if fewer than 10 such neurons remain.

ii.
```python
region_idx = get_brain_region_idx(iarea)
valid_neurons = region_idx >= 0
if valid_neurons.sum() < 10: continue
```

iii. The agent followed the visual-area mapping in the paper code and described `iarea == -1` and `7` as outside the retained visual cortex.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials start at rounded corridor-entry frame `StartFr` and always extend for 32 frames. This can include gray-space or later frames for fast trials and truncate long/paused traversals.

ii.
```python
start_fr = int(np.round(start_frs[trial]))
end_fr = start_fr + n_timepoints
trial_neural = spk[:, start_fr:end_fr]
```

iii. The agent chose a fixed window based on nominal corridor length and VR speed so every trial would have equal length.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin at 3.17 Hz, reported as about 315.46 ms. No rebinning or resampling is applied.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS
...
'time_bin_size': TIME_BIN_MS
```

iii. The agent used the acquisition rate from the methods and regarded the frame grid as the native common grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It derives the quantity from per-trial `SoundFr` and `StartFr`, plus the constant frame rate; it does not use measured frame timestamps `ft`.

ii.
```python
sound_fr_rel = sound_frs[trial] - start_frs[trial]
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
```

iii. The trajectory interpreted the event-frame difference as sufficient at 3.17 Hz.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It subtracts the cue's relative frame from indices 0..31 and divides by 3.17. Thus it stores time relative to cue (negative before, positive after), the opposite sign from literal/reference time-to-cue.

ii.
```python
time_to_cue = np.arange(n_timepoints) - sound_fr_rel
time_to_cue_sec = time_to_cue / FS
```

iii. Comments and README explicitly justify negative values before cue, showing this was deliberate rather than accidental.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The resulting 32 values correspond index-for-index to the same fixed slice as neural data.

ii.
```python
trial_input = np.stack([time_to_cue_sec, ...])
trial_neural = spk[:, start_fr:end_fr]
```

iii. The agent relied on shared frame indices for alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day is derived from each mouse's `datexp` strings over the deduplicated session table.

ii.
```python
if s['mname'] == mname:
    date = datetime.strptime(s['datexp'], '%Y_%m_%d')
    mouse_dates.append(date)
```

iii. The agent used session dates as the available longitudinal ordering information.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It computes calendar days elapsed since that mouse's earliest recording, then broadcasts the scalar to all 32 bins.

ii.
```python
day_idx = (current_date - mouse_dates[0]).days
...
input_trials[i][1, :] = float(day)
```

iii. The function docstring calls this relative to the first session; no justification was given for calendar-day gaps versus recorded-session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses `StartFr` only indirectly: after slicing at rounded `StartFr`, it constructs elapsed time from local sample indices and the constant frame rate, not from raw `ft` timestamps.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. The agent assumed uniform imaging intervals and that the first extracted frame is time zero.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It generates 0..31 frames divided by 3.17, so all valid trials receive the identical time vector.

ii.
```python
time_since_start = np.arange(n_timepoints) / FS
```

iii. The chosen fixed-window design made a synthetic uniform time vector straightforward.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The 32 time values are stacked with the neural slice and therefore align by column.

ii.
```python
trial_input = np.stack([... time_since_start, ...])
```

iii. Alignment was justified through common frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability comes directly from per-trial `isRew`.

ii.
```python
is_rew = beh['isRew']
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The variable directly marks whether the corridor is rewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial flag is cast to float and repeated across all 32 time bins; no other transformation is applied.

ii.
```python
reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. The agent treated reward availability as a per-trial contextual flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Stimulus is taken from per-trial `WallName`; `UniqWalls` is used to collect the global vocabulary.

ii.
```python
wall_name = wall_names[trial]
stim_cat = wall_name
...
for w in beh['UniqWalls']: all_stim_names.add(w)
```

iii. The trajectory found `WallName` complete even when `stim_id` is masked in swap views.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct wall-name string is sorted into a global 15-label vocabulary and its index is broadcast over the trial. It does not collapse variants into four base textures.

ii.
```python
all_stim_sorted = sorted(str(s) for s in all_stim_names)
stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
output_trials[i][0, :] = stim_idx
```

iii. The agent intentionally preserved named variants, describing them as the stimulus categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking uses `LickFr` and `LickTrind`.

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
```

iii. The agent identified these as lick frame and trial assignment fields.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial it selects that trial's licks, rounds each lick frame, subtracts the rounded trial start, and marks an in-window bin 1; other bins remain 0.

ii.
```python
trial_lick_frs = lick_frs[lick_trinds == trial]
for lf in trial_lick_frs:
    fr_idx = int(np.round(lf)) - start_fr
    if 0 <= fr_idx < n_timepoints: lick_binary[fr_idx] = 1.0
```

iii. The intended representation was a binary event series on imaging frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick indices are converted to offsets from the same `start_fr` and clipped to the same 32-bin window as neural activity.

ii.
```python
fr_idx = int(np.round(lf)) - start_fr
...
trial_neural = spk[:, start_fr:end_fr]
```

iii. The agent used the shared neural-frame numbering for alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position comes from frame-level `ft_Pos` sliced to imaging length and then to each trial window.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[start_fr:end_fr]
```

iii. The agent recognized the values as decimeters on the imaging-frame grid.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. It passes each 32-frame position slice through `np.digitize`; frames beyond the 4 m texture, which fixed windows can include, are folded into the last category.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
pos_binned = discretize_position(trial_pos)
```

iii. The agent aimed to fulfill the requested four 1 m categories and documented gray-space values as assigned to the last bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 decimeters, yielding labels 0 through 3: 0–1, 1–2, 2–3, and 3 m or farther.

ii.
```python
boundaries = [10, 20, 30]
binned = np.digitize(pos_values, boundaries)
```

iii. Ten decimeters equals one meter, matching the requested spatial bin width.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position slice uses the exact `start_fr:end_fr` indices used for neural columns.

ii.
```python
trial_pos = ft_pos[start_fr:end_fr]
trial_neural = spk[:, start_fr:end_fr]
```

iii. The agent relied on behavior already being sampled on the neural-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Speed comes from frame-level `ft_RunSpeed`.

ii.
```python
ft_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_speed[start_fr:end_fr]
```

iii. The field directly provides running speed per imaging frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is first discretized within each trial inside `extract_trial_data`, but that row is later overwritten using percentile thresholds computed from all non-NaN speeds across all selected behavior sessions.

ii.
```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
...
output_trials[i][3, :] = np.digitize(trial_speed, speed_quartiles)
```

iii. The trajectory says global thresholds were chosen for consistent discretization across sessions. The initial per-trial result has no downstream effect.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global 25th/50th/75th percentile value thresholds are used with `np.digitize`. Ties need not produce equally populated bins.

ii.
```python
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
speed_binned = np.digitize(trial_speed, speed_quartiles)
```

iii. The agent sought a common definition of Q1–Q4 across the dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed values are sliced at the same `start_fr:end_fr` as neural data and the resulting categories occupy corresponding columns.

ii.
```python
trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
output_trials[i][3, :] = speed_binned
```

iii. Shared frame indices provide alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Out-of-range fixed windows are dropped; sessions with missing behavior keys, load errors, too few trials, too few neurons, or too few valid trials are skipped with warnings. Frame streams are sliced to neural length in some places, NaNs are removed for quartile calculation, and absent speed values return zeros. The code does not repair missing data. The final pickle itself is truncated because conversion was killed.

ii.
```python
if start_fr < 0 or end_fr > n_frames: ... continue
except Exception as e:
    print(f'  ERROR loading spk: {e}')
    continue
valid = speed_values[~np.isnan(speed_values)]
```

iii. The approach favors continuing past bad sessions. The trajectory's terminal action killed the memory-heavy conversion, so the promised full artifact was not successfully produced.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant work is loading and concatenating very large spike files, copying many neuron-by-32 arrays for every trial, retaining them all in memory, and pickling the result.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([...], axis=0)
...
neural_trials.append(trial_neural)
```

iii. The trajectory timed large file loads and observed 237 GB RAM at session 47, then killed conversion; it explicitly identified loading and accumulated output size as bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loop over every lick can be vectorized; trial filtering/list construction and the separate loops that fill day, overwrite speed, map stimulus, and filter valid trials could be consolidated or partly vectorized.

ii.
```python
for lf in trial_lick_frs: ...
for i in range(len(input_trials)): ...
for i in range(len(output_trials)): ...
```

iii. The trajectory did not discuss these loop optimizations; this follows directly from the code structure.

## 12-c. What processing does the code repeat multiple times?

i. Speed is discretized twice. Trials are traversed repeatedly to fill day, redo speed, map stimulus, and filter outputs. `compute_day_of_training` rescans all sessions for every session, and stimulus vocabularies are traversed in multiple passes.

ii.
```python
speed_binned = discretize_speed(trial_speed, n_bins=4)
...
output_trials[i][3, :] = np.digitize(trial_speed, speed_quartiles)
```

iii. No explicit trajectory justification was provided; the repeated work is visible in the final script.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The first per-trial speed categorization is discarded when global categorization overwrites it. `ft_move`, `stim_id`, `uniq_walls`, `session_speed_all`, and several corridor constants are read/declared but unused in trial outputs.

ii.
```python
ft_move = beh['ft_move'][:n_frames]
stim_id = beh.get('stim_id', None)
uniq_walls = beh['UniqWalls']
...
speed_binned = discretize_speed(trial_speed, n_bins=4)
```

iii. The trajectory originally considered running-frame logic and several dataset fields, but the implemented fixed-window/global-speed path leaves these computations unused.
