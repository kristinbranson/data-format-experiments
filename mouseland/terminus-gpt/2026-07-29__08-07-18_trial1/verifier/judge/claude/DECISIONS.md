# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans all `.npy` files in `data/beh/` and matches session keys via a regex pattern (`SESSION_RE`). It does NOT use `Imaging_Exp_info.npy` as the master index. Neural files are discovered by globbing for `*_neural_data.npy`. Sessions are defined as the intersection of behavior session IDs and neural file IDs.

ii. Loading behavior:
```python
def load_behavior_sessions(data_dir: Path):
    sessions = {}
    session_source = {}
    for p in sorted((data_dir / 'beh').glob('*.npy')):
        obj = np.load(p, allow_pickle=True).item()
        if not isinstance(obj, dict):
            continue
        for sess, dat in obj.items():
            if isinstance(sess, str) and SESSION_RE.match(sess) and isinstance(dat, dict):
                if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
                    sessions[sess] = dat
                    session_source[sess] = p.name
    return sessions, session_source
```
Loading neural:
```python
def load_neural_files(data_dir: Path):
    files = {}
    for p in sorted(data_dir.rglob('*_neural_data.npy')):
        files[p.name.replace('_neural_data.npy', '')] = p
    return files
```
Intersection:
```python
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The AI chose to discover sessions by scanning all behavior files and matching keys by regex, rather than using the master experiment index. From CONVERSION_NOTES.md: "Behavior files map session IDs to per-session dictionaries."

## 1-b. How are the data split into subjects?

i. Subjects are extracted from session IDs by splitting on underscore and taking the first element. The sorted unique set of subject names forms the subjects list.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
```

iii. The session ID naming convention embeds the subject name as the first field, so splitting on `_` extracts it.

## 1-c. How are the data split into sessions?

i. A session is identified by the intersection of behavior session IDs (matched by regex) and neural file names. The AI finds 76 common sessions.

ii.
```python
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. From CONVERSION_NOTES.md: "76 common behavior+neural sessions." The AI matches sessions by ID rather than using the experiment info index. This yields fewer sessions than the reference (89 vs 76) because the regex-based approach may miss sessions or use different session key formats (e.g., swap sessions).

## 1-d. How are the data split into trials?

i. Trials are identified by unique values in `ft_trInd` (the frame-level trial index). Frame indices for each trial are precomputed and stored in `trial_frame_idx`. Trials with fewer than 2 frames are dropped.

ii.
```python
valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:] for i, tr in enumerate(uniq_tr)}
```

iii. The AI uses `ft_trInd` to split into trials. Unlike the reference, it does NOT filter to corridor-space frames using `ft_CorrSpc`, and does NOT apply a fixed trial length of 32 frames. Trials have variable length.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 frames are dropped. Sessions with fewer than 2 trials are also dropped.

ii.
```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
```
```python
if len(neural_trials) < 2:
    continue
```

iii. No explicit quality filtering beyond minimum frame count. The AI's CONVERSION_NOTES.md does not discuss trial curation beyond excluding sessions with insufficient trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`*_neural_data.npy`), which is a list of arrays per imaging plane, concatenated into a single neurons-by-frames array.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```
```python
def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
```

iii. The AI correctly identified `spks` as the deconvolved neural activity. From CONVERSION_NOTES.md: "Neural files expose `spks` entries."

## 2-b. How is the `neural` data processed?

i. The traces are concatenated across planes and cast to float32. For each trial, the neural data is sliced by the trial's frame indices. No additional processing (no dF/F, no deconvolution) is applied. Trials are NOT padded to a fixed length.

ii.
```python
spk = concat_spks(nobj['spks'])
spk = spk[:, :n_frames]
...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The AI recognized that the data is already deconvolved and needs no further processing. However, unlike the reference, it does not pad/truncate trials to a fixed length.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons from all imaging planes are kept. Brain regions are set to `['unknown']` for all neurons. The retinotopy data is NOT loaded.

ii.
```python
data['brain_regions'] = ['unknown']
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. From the AI's CONVERSION_NOTES.md: "Neural storage format ... Conversion code must robustly handle both representations." The AI did not filter neurons by visual area or use retinotopy data at all.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to trial start by taking the frames belonging to each trial (as identified by `ft_trInd`). The frames used include ALL frames of the trial, not just the corridor-space portion. Trials are variable length.

ii.
```python
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:] ...}
...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The AI uses `ft_trInd` to identify trial frames but does not restrict to corridor space (`ft_CorrSpc`) and does not truncate/pad to a fixed number of frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate is used. However, the AI sets `time_bin_size` to `None` in the metadata rather than computing it from the frame rate.

ii.
```python
'time_bin_size': None,
```
The frame period is estimated per session:
```python
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
```

iii. The AI computes `dt_frame` locally but does not set the metadata `time_bin_size` to a value. The reference uses 1000/3.17 = ~315.5 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the absolute timestamp of the sound cue) and `Trial_start_time` (the absolute timestamp of trial start).

ii.
```python
sound = np.asarray(beh['SoundTime'], dtype=float)
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0)
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. The AI uses `SoundTime` minus `Trial_start_time` converted to seconds, rather than using `SoundFr` (the frame index of the sound cue) interpolated onto frame timestamps as in the reference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound time is computed as `(SoundTime - Trial_start_time) * 86400` to get seconds. Then `time_to_sound = sound_sec - time_since` where `time_since = np.arange(idx.size) * dt_frame`. This gives positive values before the sound cue and negative after.

ii.
```python
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. The computation is conceptually similar to the reference (time difference between sound cue and each frame), but uses different source variables and a different time axis.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same frame indices used for the neural data of that trial.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. The time axis is constructed from the same frame indices, so alignment is maintained.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the behavior source filename (e.g., `Beh_before_learning.npy`, `Beh_after_learning.npy`).

ii.
```python
def infer_day(source_name: str):
    name = source_name.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 1.0
    if 'after_learning' in name or 'after_grating' in name:
        return 5.0
    m = re.search(r'test(\d+)', name)
    if m:
        return float(m.group(1))
    m = re.search(r'train(\d+)', name)
    if m:
        return float(m.group(1))
    return 0.0
```

iii. The AI infers the day from the filename pattern rather than counting session dates per mouse. The reference counts how many recording days each mouse has had.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day is inferred from the behavior filename and broadcast across all frames of a trial. "before_learning" maps to 1.0, "after_learning" to 5.0, etc.

ii.
```python
day = infer_day(beh_source)
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. This approach gives absolute day values inferred from experiment type names, not per-mouse training day counts. The reference counts 0 to 7 per mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Constructed as `np.arange(idx.size) * dt_frame`, where `dt_frame` is the median estimated frame period for the session.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The AI uses a synthetic time axis based on estimated frame period rather than actual frame timestamps from `ft`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A linearly spaced time axis starting at 0, with spacing `dt_frame` (estimated from trial durations). This is simpler than the reference which uses actual frame timestamps from `ft` and interpolates `StartFr` onto them.

ii.
```python
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The approximation assumes uniform frame spacing, which is generally close but not exact.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same frame indices used for the neural data of that trial, so alignment is maintained.

ii. Same frame index `idx` is used for both.

iii. Aligned by construction.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial flag indicating whether the corridor is rewarded.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
...
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. Same source variable as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The `isRew` value is cast to int and broadcast across all frames of the trial.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. Same approach as the reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
wall = np.asarray(beh['WallName'])
...
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each unique wall name (e.g., circle1, circle2, leaf1, leaf2, rock1, wood1, etc.) is treated as its own category. The AI does NOT group wall names into 4 base texture categories (circle, leaf, rock, wood) as the reference does.

ii.
```python
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
...
out[0, :] = wall_to_id[wall_name]
```
Output values:
```python
'output_values': [wall_names, ['no_lick', 'lick'], ['bin1', 'bin2', 'bin3', 'bin4'], ['q1', 'q2', 'q3', 'q4']],
```

iii. The AI uses individual wall names as categories. The task instructions say "Visual stimulus category. e.g. circle1, leaf2, etc." which could be interpreted either way, but the reference groups them into 4 base textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTrind` (the trial index of each lick) and `LickTime` (the absolute timestamp of each lick).

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
```

iii. The AI uses `LickTrind` and `LickTime` rather than `LickFr` (the frame index of each lick). This requires reconstructing lick timing from absolute timestamps.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick times are converted to relative times since trial start, then binned into frame indices using `dt_frame`. A binary lick series is constructed.

ii.
```python
if lick_tr.size and lick_time.size:
    lick_tr_i = lick_tr.astype(int)
    for tr in np.unique(lick_tr_i):
        idx = trial_frame_idx.get(int(tr))
        if idx is None or idx.size == 0 or tr >= len(tstart):
            continue
        rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
        rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            continue
        bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
        lick_series[idx[np.unique(bins)]] = 1
```

iii. This is a more complex reconstruction than the reference, which simply uses `LickFr` as frame indices directly. The AI's approach involves time-based reconstruction which may introduce timing inaccuracies.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick series is indexed by the same frame indices used for neural data of each trial.

ii.
```python
lick = lick_series[idx]
```

iii. Aligned by frame index.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_PosCum` (the cumulative position), NOT `ft_Pos` (the position within the current corridor).

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
...
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The AI uses cumulative position with modulo to get within-corridor position. The reference uses `ft_Pos` directly.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The cumulative position is taken modulo the corridor length (4.0 m by default), then divided by corridor length and multiplied by 4 to get bin indices (0-3).

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. The reference uses `ft_Pos` directly with `// 10` (dividing decimeters by 10 to get meter bins).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The position is divided by corridor length, multiplied by 4, and truncated to int, then clipped to 0-3. This creates 4 bins.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. Functionally similar to the reference's approach of `// 10`, but using different source data (`ft_PosCum` vs `ft_Pos`).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same frame indices used for neural data.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. Aligned by frame index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is COMPUTED as the derivative of position (`ft_PosCum`), NOT taken from the raw `ft_RunSpeed` variable.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The AI derives speed from position rather than using the provided `ft_RunSpeed`. This means the speed values will differ from the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is computed as `np.diff(pos, prepend=pos[0]) / dt_frame`. All speeds from a session are collected, then quartile thresholds are computed using `np.quantile` at [0.25, 0.5, 0.75], and `np.digitize` is used to bin.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
...
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75])
...
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The reference uses `ft_RunSpeed` directly and rank-based quartiles. The AI's approach is different in both source data and discretization method.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Quartile thresholds are computed from all speeds in the session using `np.quantile`, then `np.digitize` assigns bins 0-3.

ii.
```python
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0])
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The reference uses rank-based quartiles (`argsort` + rank indexing), which guarantees exactly 25% per bin even with tied values. `np.quantile` + `np.digitize` can produce unequal bins when many values are equal (e.g., zero speed).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from the position at the same frame indices used for neural data.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Aligned by frame index.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: truncates all streams to the minimum of neural and behavioral frame counts; skips trials where trial index exceeds available metadata arrays; uses `np.isfinite` checks for sound times; handles missing `LickTrind`/`LickTime` with `.get()` defaults; handles duplicate behavior sessions by preferring the "richer" entry.

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
...
if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
    continue
...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
```

iii. The AI was relatively careful about edge cases, handling NaN values and dimension mismatches. The duplicate-session handling was documented in CONVERSION_NOTES.md as a bug fix.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating neural data files (the `spks` arrays). The AI reports ~41s per session for the full pipeline, with total conversion time ~36 minutes for 76 sessions.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. From CONVERSION_NOTES.md: "Sample conversion ~41 s/session after trial-index optimization... Rough full conversion time ~52 minutes for 76 sessions."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking reconstruction loop iterates over unique lick trials, and within each trial computes relative lick times. This could potentially be vectorized. The main session loop iterates over trials individually.

ii.
```python
for tr in np.unique(lick_tr_i):
    idx = trial_frame_idx.get(int(tr))
    ...
    rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
    ...
```

iii. The AI improved efficiency by precomputing trial frame indices but did not vectorize the licking reconstruction.

## 12-c. What processing does the code repeat multiple times?

i. The behavior file for each session is loaded independently (since `load_behavior_sessions` loads all behavior upfront). No repeated loading. However, the `all_speeds` collection and subsequent `np.quantile` computation requires two passes through trials within a session.

ii.
```python
cache.append(...)
all_speeds.append(speed)
...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False)
```

iii. The two-pass approach for speed quantiles is necessary for the chosen discretization method.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ft_move` and `ft_WallID` arrays but does not use them for any output variable. These are loaded unnecessarily.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
...
ft_move = ft_move[:n_frames]
ft_wall = ft_wall[:n_frames]
```

iii. `ft_move` and `ft_WallID` are loaded and truncated but never used in constructing inputs or outputs.
