# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads behavior by iterating over every `data/beh/*.npy` file, keeping only dictionary entries whose keys look like session IDs. It loads neural data by recursively finding every `*_neural_data.npy` file. It then keeps only the intersection of behavior-session IDs and neural-session IDs. Within each common session, it builds trials from frame-level behavior indices.

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

def load_neural_files(data_dir: Path):
    files = {}
    for p in sorted(data_dir.rglob('*_neural_data.npy')):
        files[p.name.replace('_neural_data.npy', '')] = p
    return files

beh_sessions, beh_source = load_behavior_sessions(data_dir)
neural_files = load_neural_files(data_dir)
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The notes say the agent decided to treat `spks` as the deconvolved neural signal, filter behavior keys to session-like IDs, and prefer the “richer” duplicate behavior entry when the same session appeared in multiple files. The trajectory also shows it explicitly fixed a dropped-session bug caused by duplicate behavior entries.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session-name prefix before the first underscore, and unique prefixes become `subjects`. Each kept session gets `subject_idx` from that prefix.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. The notes describe session names as embedding subject ID, date, and session number, so the agent used the prefix as mouse ID.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the matched session IDs shared by a behavior dict entry and a neural filename. The converter creates one output session per matched key in `common`.

ii.
```python
common = sorted(set(beh_sessions) & set(neural_files))
...
for sess in common:
    neural_trials, input_trials, output_trials = build_session(
        beh_sessions[sess], beh_source[sess], neural_files[sess]
    )
    if len(neural_trials) < 2:
        continue
    data['neural'].append(neural_trials)
```

iii. The agent justified this from the reference code structure: per-session dictionaries keyed by session IDs, plus one neural file per session.

## 1-d. How are the data split into trials?

i. Trials are split using the frame-aligned `ft_trInd` vector. The agent trims all frame-level arrays to a shared `n_frames`, converts finite `ft_trInd` values to integers, groups contiguous valid frame indices by trial ID, and uses each group as one trial.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
...
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
...
valid_frame = np.isfinite(ft_tr)
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)

valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {
    int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:]
    for i, tr in enumerate(uniq_tr)
}
```

iii. In the trajectory, the agent called the frame-aligned `ft_*` fields a “major breakthrough” and switched from a placeholder trial reconstruction to `ft_trInd`-based segmentation.

## 1-e. How are trials filtered based on quality controls?

i. The code applies only minimal structural filtering: it skips trials with fewer than 2 frames, skips trials whose trial index would exceed the lengths of `WallName`, `isRew`, `SoundTime`, or `Trial_start_time`, and drops sessions with fewer than 2 remaining trials. It does not implement paper-style movement, corridor-only, odd/even, or validity filtering.

ii.
```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
...
if len(neural_trials) < 2:
    continue
```

iii. The notes say explicit trial-curation rules were still unresolved from the methods/code, so the agent fell back to minimal validity checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken directly from the neural file’s `spks` field, which is assumed to contain deconvolved fluorescence traces. The agent concatenates all entries in the `spks` list.

ii.
```python
def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)

nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. The notes explicitly state: “Treat `spks` as the deconvolved neural activity stream,” based on the methods quote that analyses were based on deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Processing is minimal. The agent concatenates `spks`, trims time to the minimum shared frame count with frame-level behavior arrays, and slices per-trial frame blocks. It does not perform the reference interpolation by corridor position, z-scoring, or neuron selection.

ii.
```python
spk = concat_spks(nobj['spks'])
...
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The trajectory shows the agent knew the reference code used `get_interpPos_spk(...)` and `stats.zscore(...)`, but it still described its own implementation as “still approximate” and “likely not yet faithful enough.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not apply neuron-quality filtering. It keeps every concatenated neuron and sets every neuron’s brain-region index to zero/`unknown`.

ii.
```python
data = {
    ...
    'brain_regions': ['unknown'],
    'brain_region_idx': [],
}
...
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. The notes mention Suite2p cell classification in the paper but say exact downstream neuron-inclusion criteria were not identified, so no such filtering was implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural time series are aligned to corridor entry / trial start only implicitly through trial segmentation. The session is split by `ft_trInd`, and each trial’s first kept frame becomes time zero for that trial.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'corridor entry / trial start',
    'off_start': 0.0,
    'off_end': None,
}
...
for tr, idx in trial_frame_idx.items():
    ...
    tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The instructions required trial-start alignment. The agent’s notes say it used frame-aligned fields and trial indices as the primary time base.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent does not preserve the real imaging frame time. Instead, it estimates one session-level `dt_frame` as the median over `(trial_end - trial_start) / number_of_frames_in_trial`. It does not rebin neural data; it keeps one sample per kept frame. Metadata leaves `time_bin_size` as `None`.

ii.
```python
trial_dur_sec = (tend - tstart) * 86400.0
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
...
'time_bin_size': None,
```

iii. The trajectory says this was a “proxy time axis based on median trial duration because neural frame timestamps are still unresolved.”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime`, `Trial_start_time`, and the inferred per-frame time axis within each trial.

ii.
```python
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
sound = np.asarray(beh['SoundTime'], dtype=float)
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) ...
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. The notes cite the paper’s sound-cue timing description and say the agent chose direct behavior fields for reward/sound structure.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code converts `SoundTime - Trial_start_time` from days to seconds, then subtracts the trial’s framewise `time_since` vector. Missing cue times produce an all-`NaN` vector.

ii.
```python
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. The trajectory shows this was part of the approximate frame-time reconstruction after the agent failed to identify an explicit neural timestamp stream.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is sampled on the same per-trial frame grid as `tr_spk`, because both are built using the same `idx` frame indices and the same inferred `dt_frame`.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
...
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The notes say the converter uses frame-aligned behavior fields and concatenated `spks`, with all time-varying inputs placed on that frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived from an explicit raw day variable. The code infers day only from the behavior source filename, such as `before_learning`, `after_learning`, `testN`, or `trainN`.

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

iii. The notes cite the paper’s 5-day training schedule, and the agent appears to have used filename heuristics as a shortcut rather than reading a session-level day field.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. After inferring one scalar `day`, the code broadcasts it across all time points in the trial as a constant vector.

ii.
```python
day = infer_day(beh_source)
...
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. The trajectory shows this was a heuristic mapping from file naming conventions, not a reconstruction from experimental metadata.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. It is not derived at all. The agent did not include an environment-type input in the converted dataset.

ii.
```python
'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
```

iii. No justification for omitting it appears in the code. The likely reason is that “environment type” was not one of the decoder inputs required by the task instructions.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. No environment-type variable is computed or stored.

ii.
```python
'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
```

iii. The omission is consistent with the fact that the instructions did not ask for this input, even though it appears in the question list.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from trial frame count and the inferred session-level frame period `dt_frame`, not from an explicit per-frame timestamp variable.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The trajectory explicitly calls this a proxy time axis because frame timestamps were unresolved.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Within each trial, the first kept frame is assigned 0 and subsequent frames increment by `dt_frame`.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. This follows directly from the agent’s `ft_trInd`-based trial segmentation plus its estimated frame period.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is exactly frame-aligned to the neural matrix because it has the same number of time points as `tr_spk.shape[1]` for each trial.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The notes say the converter is frame-aligned and trial-based.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial `isRew` field.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
...
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. The notes say reward structure in the behavior files matched the paper/methods, so the agent used `isRew` directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. For each trial, the integer reward flag is copied into a constant vector across all frames of the trial.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. This is a straightforward trial-level broadcast of `isRew`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial `WallName` string. The agent collects all unique wall names across sessions and maps them to integer IDs.

ii.
```python
wall = np.asarray(beh['WallName'])
...
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
...
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
...
out[0, :] = wall_to_id[wall_name]
```

iii. The notes mention `WallType`, `WallName`, `UniqWalls`, and category mapping from the reference code, but the implemented converter simplified this to `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code converts each trial’s `WallName` to one categorical integer and then repeats that label across all time bins in the first output row.

ii.
```python
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
...
out[0, :] = wall_to_id[wall_name]
```

iii. The trajectory suggests the agent knew `stim_id` existed but ultimately used the simpler `WallName` encoding.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTrind`, `LickTime`, and `Trial_start_time`, plus the inferred frame period and trial-frame index mapping.

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
...
rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
lick_series[idx[np.unique(bins)]] = 1
```

iii. The notes say the reference code uses lick positions and trial indices to build lick rasters. The implemented converter instead rasterizes lick timestamps onto inferred frame bins.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code builds a session-wide binary `lick_series`, then for each trial assigns 1 to any frame bin containing one or more licks. Multiple licks in the same frame are collapsed.

ii.
```python
lick_series = np.zeros(n_frames, dtype=np.int64)
...
for tr in np.unique(lick_tr_i):
    ...
    bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
    lick_series[idx[np.unique(bins)]] = 1
...
lick = lick_series[idx]
```

iii. The trajectory and notes frame this as a practical alignment choice, not a direct copy of the reference lick-position logic.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by writing lick events into the same frame indices `idx` used for the neural trial matrix.

ii.
```python
lick_series[idx[np.unique(bins)]] = 1
...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
lick = lick_series[idx]
```

iii. The notes describe the implementation as frame-aligned conversion.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level cumulative position `ft_PosCum` and `Corridor_Length`.

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
...
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
...
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The trajectory shows the agent deliberately switched to `ft_PosCum` after finding the richer frame-aligned behavior fields.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is wrapped with modulo corridor length and kept framewise within each trial.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The notes say the converter uses frame-aligned `ft_PosCum`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The code divides normalized corridor position into 4 equal-length bins by scaling by `4 / corridor_len`, converting to `int`, and clipping to `[0, 3]`.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. This matches the decoder-task requirement for four equal 1 m bins when corridor length is 4 m.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is frame-aligned because it is computed from `ft_PosCum[idx]` using the same frame indices as `tr_spk`.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The notes repeatedly describe the conversion as frame-aligned.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived indirectly from `ft_PosCum` and the inferred `dt_frame`. The code does not use available raw speed variables such as `ft_RunSpeed` or `SubjMove`.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The trajectory shows the agent originally considered running speed from position/time derivatives and kept that approximation.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code takes the first difference of wrapped position within the trial, prepends the first position, and divides by `dt_frame`.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. No extra smoothing or movement masking is applied.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The code concatenates all per-trial speed values within a session, computes the 25th/50th/75th percentiles, and uses `np.digitize` to assign each frame to one of four quantile bins.

ii.
```python
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0], dtype=np.float32)
...
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. This directly follows the decoder-task instruction to make four bins corresponding to 25% of the data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is frame-aligned because speed is computed from the same framewise position samples indexed by `idx`.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The implementation keeps one speed value per neural frame.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code trims streams to the minimum shared frame count, ignores non-dictionary behavior files/entries, drops non-session keys, skips `NaN` trial labels in `ft_trInd`, fills invalid frame labels with `-1`, propagates missing `SoundTime` as `NaN` in `time_to_sound`, uses defaults for missing `Corridor_Length`, and skips trials or sessions that fail basic structural checks. Duplicate session entries are resolved by keeping the richer behavior dict.

ii.
```python
if not isinstance(obj, dict):
    continue
...
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
...
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)
...
time_to_sound = ... if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
...
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
...
if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
    sessions[sess] = dat
```

iii. The notes explicitly document the duplicate-session overwrite bug and its fix. The rest of the handling is ad hoc defensive programming rather than a paper-specified curation rule.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading huge behavior/neural files, concatenating very large `spks` arrays, constructing per-session trial-frame mappings, rasterizing licks trial-by-trial, and iterating through every trial to build per-trial outputs.

ii.
```python
for p in sorted((data_dir / 'beh').glob('*.npy')):
    obj = np.load(p, allow_pickle=True).item()
...
spk = concat_spks(nobj['spks'])
...
trial_frame_idx = { ... for i, tr in enumerate(uniq_tr)}
...
for tr in np.unique(lick_tr_i):
    ...
for tr, idx in trial_frame_idx.items():
    ...
```

iii. The notes say full conversion took about 36 minutes and that precomputing `trial_frame_idx` materially reduced sample runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over trials to estimate `dt_frame`, the lick-raster loop over unique trials, and the main per-trial construction loop are all candidates. Speed-bin assignment is also split into a cache pass and a second loop.

ii.
```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
...
for tr in np.unique(lick_tr_i):
    ...
for tr, idx in trial_frame_idx.items():
    ...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    ...
```

iii. The notes mention only one optimization: precomputing `trial_frame_idx` instead of repeated scans.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly converts arrays with `np.asarray`, repeatedly scans trial IDs while estimating frame periods, and performs a two-pass trial workflow: once to cache `speed` and again to convert it to quantile bins. It also recomputes subject ID from the session string in multiple places.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
ft_move = np.asarray(beh['ft_move'], dtype=float)
...
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
...
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The notes acknowledge some inefficiency and document the one optimization the agent added.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `ft_move` and `ft_WallID` but never uses them downstream; computes `trial_frame_idx` for all valid trials before later dropping some; stores `wall_name` strings in a cache only to replace them with integer IDs later; and computes continuous `speed` values only to discard them after converting to bins.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
...
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
...
out[0, :] = wall_to_id[wall_name]
```

iii. The notes emphasize runtime and trial-index optimization, but they do not claim these extra computations are needed by the final decoder format.
