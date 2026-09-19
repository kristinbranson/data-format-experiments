# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every behavior `.npy` file under `data/beh`, keeps dictionary entries whose keys look like session IDs, resolves duplicate session keys by keeping the "richer" entry, recursively finds every `*_neural_data.npy`, and processes only the intersection of behavior and neural session names. It does not use `beh/Imaging_Exp_info.npy` or retinotopy files.

ii.
```python
def load_behavior_sessions(data_dir: Path):
    sessions = {}
    session_source = {}
    for p in sorted((data_dir / 'beh').glob('*.npy')):
        obj = np.load(p, allow_pickle=True).item()
        ...
        for sess, dat in obj.items():
            if isinstance(sess, str) and SESSION_RE.match(sess) and isinstance(dat, dict):
                if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
                    sessions[sess] = dat
                    session_source[sess] = p.name
```
```python
def load_neural_files(data_dir: Path):
    files = {}
    for p in sorted(data_dir.rglob('*_neural_data.npy')):
        files[p.name.replace('_neural_data.npy', '')] = p
    return files
```
```python
beh_sessions, beh_source = load_behavior_sessions(data_dir)
neural_files = load_neural_files(data_dir)
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The explicit justification in the notes is that behavior files contain duplicate session IDs and the loader should prefer the richer session dictionary; Step 9 and Step 10 say this was added after a missing-session bug was traced to a poorer duplicate overwriting a richer one.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session ID prefix before the first underscore, then sorted uniquely. `subject_idx` is built from that derived subject list.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. The notes justify this indirectly by stating that the session naming convention embeds the subject ID.

## 1-c. How are the data split into sessions?

i. Sessions are defined purely by shared string IDs between behavior keys and neural filenames. Duplicate behavior entries are collapsed by richness, and only `common` session names are kept.

ii.
```python
SESSION_RE = re.compile(r'^[A-Za-z0-9]+_\d{4}_\d{2}_\d{2}_\d+(?:_swap[12])?$')
...
common = sorted(set(beh_sessions) & set(neural_files))
```
```python
if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
    sessions[sess] = dat
```

iii. The explicit justification is again the duplicate-session fix: the notes say a poorer duplicate entry had overwritten `TX109_2023_03_27_1`, so the loader was changed to keep the richer entry.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from `ft_trInd`. The code trims all frame-aligned streams to a common `n_frames`, converts `ft_trInd` to integers where finite, finds all valid frame indices, and then groups contiguous stretches of those indices by unique trial number. It does not additionally restrict trials to `ft_CorrSpc`.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
...
valid_frame = np.isfinite(ft_tr)
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)
```
```python
valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {
    int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:]
    for i, tr in enumerate(uniq_tr)
}
```

iii. The notes and trajectory justify this as "frame-aligned conversion using `ft_trInd`" and say the frame-level fields in behavior allow trial segmentation without reconstructing timing from scratch.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by basic viability checks: the trial must have at least 2 frames, and its trial index must be in range for `WallName`, `isRew`, `SoundTime`, and `Trial_start_time`. There is no explicit outlier-length filter.

ii.
```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
```

iii. No explicit quality-control rationale was written beyond making sessions decodable; the only recorded rationale is the Step 6 optimization around precomputing `trial_frame_idx`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from the session's `spks` entry in the neural `.npy` file, concatenated across list elements. The final neural values are not derived from behavior variables.

ii.
```python
def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
...
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. The trajectory explicitly notes that reference code concatenates the `spks` list along axis 0 and interprets the list elements as neuron blocks or planes rather than trials.

## 2-b. How is the `neural` data processed?

i. Neural planes are concatenated, cast to `float32`, trimmed to the minimum length shared with several behavior streams, and then sliced into per-trial matrices using the frame indices from `trial_frame_idx`. No additional neural normalization, deconvolution, or temporal rebinning is applied.

ii.
```python
spk = concat_spks(nobj['spks'])
...
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
```
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
...
neural_trials.append(tr_spk)
```

iii. The notes say the agent implemented "frame-aligned conversion" using the existing deconvolved `spks` stream and the trajectory says the dataset already exposes frame-aligned fields, so the agent reused them directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neural quality-control filtering. All concatenated neurons are kept, every neuron is assigned brain-region index 0, and the only named brain region is `"unknown"`.

ii.
```python
data = {
    ...
    'brain_regions': ['unknown'],
    'brain_region_idx': [],
    ...
}
```
```python
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. No explicit justification was given for ignoring retinotopy; the only related evidence is that the agent focused on behavior-neural frame alignment and did not document using retinotopy in the final script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script declares the alignment event to be "corridor entry / trial start" and aligns neural trials to the first kept frame of each `ft_trInd` segment. Trials then run for the full kept segment length from that trial index.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'corridor entry / trial start',
    'off_start': 0.0,
    'off_end': None,
},
```
```python
for tr, idx in trial_frame_idx.items():
    ...
    tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The trajectory repeatedly describes the implementation as "frame-aligned" and says the continuous VR streams "must be segmented by trial boundaries."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No explicit rebinning is applied. Instead, the script estimates a per-session frame duration `dt_frame` from `Trial_end_time - Trial_start_time` divided by the number of frames assigned to each trial, uses that inferred bin width for timing variables, and leaves `metadata['time_bin_size']` as `None`.

ii.
```python
trial_dur_sec = (tend - tstart) * 86400.0
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
```
```python
'time_bin_size': None,
```

iii. No explicit justification for omitting a fixed `time_bin_size` was written; the code itself shows the decision to infer timing from trial durations instead of using a constant imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime` and `Trial_start_time`, together with the inferred per-trial frame index.

ii.
```python
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
sound = np.asarray(beh['SoundTime'], dtype=float)
...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. The notes do not justify this variable choice explicitly. The surrounding notes indicate the agent preferred trial-level timing variables that were already available in behavior dictionaries.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The script converts `SoundTime - Trial_start_time` from days to seconds to get a per-trial sound-cue time, builds a synthetic within-trial time axis `0, dt_frame, 2*dt_frame, ...`, and subtracts that axis from the cue time. Missing cue times are converted to `NaN` vectors.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. No explicit written justification was found beyond using available trial timestamps.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by constructing the vector with exactly the same number of bins as the neural slice for that trial, using the same `idx.size`.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32) ...
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The trajectory justification is that all streams were being aligned through shared trial/frame indexing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the behavior source filename, not from session chronology. The code parses strings like `before_learning`, `after_learning`, `trainN`, or `testN` from the filename.

ii.
```python
def infer_day(source_name: str):
    name = source_name.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 1.0
    if 'after_learning' in name or 'after_grating' in name:
        return 5.0
    m = re.search(r'test(\\d+)', name)
    ...
    m = re.search(r'train(\\d+)', name)
```

iii. The notes do not justify this directly. The closest justification is in Step 3, where the notes summarize different training stages from the methods text.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The filename-derived stage label is converted to a single float per session and then broadcast across every time bin of every trial in that session.

ii.
```python
day = infer_day(beh_source)
...
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. No explicit justification was found beyond the stage-name parsing in `infer_day`.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time`, `Trial_end_time`, and the number of frames assigned to the trial.

ii.
```python
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
tend = np.asarray(beh['Trial_end_time'], dtype=float)
trial_dur_sec = (tend - tstart) * 86400.0
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. No explicit written justification was found.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script estimates `dt_frame` from per-trial duration divided by the trial's frame count, then builds `time_since_trial_start` as a synthetic regularly spaced vector from 0 upward.

ii.
```python
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. No explicit justification was recorded beyond using trial-level timing fields already present in behavior.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by using the same trial slice length as the neural data; each trial gets one `time_since` vector with `idx.size` bins.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The trajectory justification is the same shared frame-indexing logic used for other trial-wise variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `isRew`.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
...
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. No additional justification was written; the variable is used directly from the behavior dictionary.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script just casts the trial's reward flag to an integer and broadcasts it across that trial's time bins.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. No explicit extra justification was given.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii.
```python
wall = np.asarray(beh['WallName'])
...
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
```

iii. The Step 1 notes say stimulus structure uses fields such as `WallType`, `WallName`, and `UniqWalls`, and that the converter should preserve category mapping.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code collects the sorted unique raw `WallName` strings across all kept sessions, builds a global `wall_to_id` lookup, and then fills the first output channel of each trial with that raw wall-name ID. It does not collapse variants such as `circle1`, `circle2`, or swap conditions into four broad texture categories.

ii.
```python
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
...
for wall_name, out in output_trials:
    out = out.copy()
    out[0, :] = wall_to_id[wall_name]
```
```python
'output_values': [wall_names, ['no_lick', 'lick'], ['bin1', 'bin2', 'bin3', 'bin4'], ['q1', 'q2', 'q3', 'q4']],
```

iii. No explicit justification was given for using raw `WallName` values rather than the four-category mapping mentioned in the notes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTrind`, `LickTime`, and `Trial_start_time`.

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
...
rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
```

iii. The notes say behavioral processing in the reference code uses lick positions and lick trial indices to reconstruct per-trial lick rasters, and Step 6 says the final implementation used `LickTrind` and `LickTime`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial containing licks, lick times are converted to seconds relative to `Trial_start_time`, converted into within-trial bin numbers by dividing by `dt_frame`, clipped into range, deduplicated per bin, and then written into a session-wide binary `lick_series`.

ii.
```python
lick_series = np.zeros(n_frames, dtype=np.int64)
...
for tr in np.unique(lick_tr_i):
    idx = trial_frame_idx.get(int(tr))
    ...
    rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
    rel = rel[np.isfinite(rel)]
    ...
    bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
    lick_series[idx[np.unique(bins)]] = 1
```

iii. The notes justify the choice at a high level by saying the converter should reconstruct trial-organized lick rasters from the lick variables in behavior.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned to the neural data by placing the derived lick bins into the same frame indices `idx` that define each trial's neural slice, and then slicing `lick_series[idx]`.

ii.
```python
idx = trial_frame_idx.get(int(tr))
...
lick_series[idx[np.unique(bins)]] = 1
...
lick = lick_series[idx]
```

iii. The recorded rationale is again that all trial-wise streams are aligned through the same frame-index segmentation.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_PosCum` and `Corridor_Length`.

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
...
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
...
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. Step 6 notes explicitly say the implementation used `ft_PosCum`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The cumulative position is reduced modulo the corridor length to get a within-corridor position, then scaled into four equal fractional bins across the corridor.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. No explicit written justification was found beyond using the cumulative position field.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The within-corridor position is converted to four bins by dividing by the corridor length, multiplying by 4, casting to integer, and clipping to `[0, 3]`.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. No explicit justification beyond "4 bins" was recorded.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by reading one `ft_PosCum` value at every neural frame index in the trial slice and storing the resulting `pos_bin` vector with the same length as `tr_spk`.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
pos_bin = ...
```

iii. The notes and trajectory consistently justify these frame-aligned outputs by the presence of frame-level behavior fields.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-to-frame differences of `ft_PosCum` together with the inferred frame duration `dt_frame`. The loaded `ft_move` array is not used in the final speed output.

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
ft_move = np.asarray(beh['ft_move'], dtype=float)
...
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Step 6 notes explicitly list `ft_move` among the loaded fields, but the final code computes speed from position differences instead.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. For each trial, the within-corridor position trace is differenced to estimate speed. All trial speed vectors for the session are concatenated, and session-specific quartile thresholds are computed from those values.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
...
all_speeds.append(speed)
...
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0], dtype=np.float32)
```

iii. No explicit written justification for this computation was found.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The code thresholds speed with `np.quantile(..., [0.25, 0.5, 0.75])` and then uses `np.digitize` to assign bins 0 through 3.

ii.
```python
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) ...
...
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. No explicit justification beyond "4 bins" was recorded.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed per frame on the same trial slice `idx` used for the neural data, so each speed vector has one value per neural time bin.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
```

iii. The rationale is implicit in the frame-aligned trial slicing used throughout the script.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles mismatches by trimming all frame-wise streams to the minimum shared length, ignoring non-finite `ft_trInd` values, clipping lick bins into valid ranges, dropping trials with too few frames or missing per-trial variables, and resolving duplicate behavior session keys by keeping the richer entry.

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
...
valid_frame = np.isfinite(ft_tr)
...
bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
...
if idx.size < 2:
    continue
if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
    continue
```
```python
if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
    sessions[sess] = dat
```

iii. The only explicit justification in the notes concerns duplicate behavior entries: the agent says the richer session dictionary should win to avoid dropping valid sessions.

## 12-a. What are the most time-consuming steps of the code?

i. The notes imply that building trial frame indices and repeated per-trial frame scans were a major runtime cost; the full conversion runtime is also reported as about 36 minutes for 76 sessions.

ii.
```python
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    ...
```
```python
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = { ... }
```

iii. Step 6 says precomputing `trial_frame_idx` reduced sample runtime from about 160 s to about 83 s, which is the clearest recorded efficiency justification.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code contains several per-trial loops that could be vectorized or consolidated, especially the loop that recomputes frame counts with `np.sum(ft_tr_int == tr)` and the loop over unique lick trials.

ii.
```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
```
```python
for tr in np.unique(lick_tr_i):
    idx = trial_frame_idx.get(int(tr))
    ...
```

iii. The notes explicitly mention that precomputing `trial_frame_idx` removed repeated frame-index scans, which is the agent's own optimization rationale.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats trial-wise processing in multiple passes: once to estimate frame duration, once to build the trial index map, once to rasterize licks by trial, once to collect cached trial outputs, and then a second pass over the cached trials to assign speed bins.

ii.
```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    ...
```
```python
for tr in np.unique(lick_tr_i):
    ...
```
```python
for tr, idx in trial_frame_idx.items():
    ...
    cache.append(...)
    all_speeds.append(speed)
...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. No explicit written justification was found for these repeated passes; the notes only mention one optimization to reduce repeated scans.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads and trims `ft_move` and `ft_WallID` but never uses them downstream, computes `Trial_end_time` only to estimate `dt_frame`, stores `wall_name` strings temporarily only to overwrite the first output row later, and builds session metadata without recording the inferred frame duration in the output metadata.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
...
ft_move = ft_move[:n_frames]
ft_wall = ft_wall[:n_frames]
```
```python
tend = np.asarray(beh['Trial_end_time'], dtype=float)
...
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
...
out[0, :] = wall_to_id[wall_name]
```

iii. No explicit justification was written for these unused or transient computations.
