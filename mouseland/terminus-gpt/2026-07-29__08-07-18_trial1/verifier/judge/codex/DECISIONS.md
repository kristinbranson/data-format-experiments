# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavior by scanning every `data/beh/*.npy` file, taking dictionary entries whose keys look like session IDs, and keeping the "richer" duplicate when the same session appears in multiple files. Neural data are loaded by recursively finding every `*_neural_data.npy` file. The session list is the intersection of discovered behavior-session keys and neural filenames. It does not use `beh/Imaging_Exp_info.npy` as a master index, and it does not load retinotopy files at all.

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
    return sessions, session_source
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

iii. The justification in the notes is that behavior files are already "substantially preprocessed into numpy dictionaries" and should be reused directly, and that behavior keys should be filtered to "session-like IDs when matching to neural files." The notes also document the duplicate-session fix: the loader was changed to prefer the "richer" duplicate behavior entry after `TX109_2023_03_27_1` was initially lost.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session ID string by taking the substring before the first underscore, then sorting the unique names and mapping each session to that subject index.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. There is no separate explicit justification beyond the notes' statement that session naming embeds the subject ID.

## 1-c. How are the data split into sessions?

i. A session is any behavior-dictionary key matching the session-ID regex and also present as a neural filename stem. Duplicate behavior entries with the same session key are merged by keeping the richer entry. The AI does not use the paper's `(mname, datexp, blk)` master-index definition.

ii.
```python
SESSION_RE = re.compile(r'^[A-Za-z0-9]+_\d{4}_\d{2}_\d{2}_\d+(?:_swap[12])?$')
...
for sess, dat in obj.items():
    if isinstance(sess, str) and SESSION_RE.match(sess) and isinstance(dat, dict):
        if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
            sessions[sess] = dat
```
```python
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The notes justify this as filtering to "session-like IDs when matching to neural files" and later justify the duplicate handling as necessary because some behavior files repeated the same session with poorer metadata.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the frame-level trial index `ft_trInd`. The code keeps all frames with finite trial labels, groups contiguous occurrences of the same trial index, and treats each resulting frame block as one trial. It does not use `ft_CorrSpc` to restrict trials to the textured corridor and does not impose the reference's fixed 32-frame trial length.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
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

iii. The notes describe this generally as a "frame-aligned conversion using `ft_trInd`" and emphasize precomputing `trial_frame_idx` once per session for speed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they contain fewer than 2 frames after the `ft_trInd` grouping, or if the trial index would fall past the lengths of `WallName`, `isRew`, `SoundTime`, or `Trial_start_time`. Sessions are later dropped if fewer than 2 trials survive. There is no explicit trial-quality filter tied to the reference task structure.

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

iii. No explicit scientific justification is given. The notes only mention that precomputing `trial_frame_idx` improved speed, and the code treats the length checks as defensive handling for imperfect data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from the `spks` arrays in each `*_neural_data.npy` file, concatenated across planes. No other raw neural variable is used, and retinotopy is not consulted for the neural signal itself.

ii.
```python
def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
```
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. The notes justify this by saying the methods/code refer to deconvolved traces and that `spks` should be treated as the deconvolved neural activity stream.

## 2-b. How is the `neural` data processed?

i. The code concatenates planes, truncates all frame-level streams to a common `n_frames`, then slices the concatenated spike matrix by each trial's frame indices. The resulting per-trial neural arrays are left at variable length and stored as `float32`. There is no padding, no fixed 32-frame window, and no explicit cut to the textured corridor.

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
...
neural_trials.append(tr_spk)
```

iii. The notes describe the implementation as "frame-aligned conversion using `ft_trInd`, `ft_PosCum`, `ft_move`, `LickTrind`, `LickTime`, and concatenated `spks`." No further justification for variable-length trials is given.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no neuron-quality or brain-region filter to the neural matrix. All concatenated neurons are kept. Separately, it sets every neuron's brain-region index to 0 and records the only brain region as `"unknown"`.

ii.
```python
data = {
    ...
    'brain_regions': ['unknown'],
    'brain_region_idx': [],
    ...
}
...
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. The notes never identify the retinotopy-based curation rule from the reference. Their Step 3 says exact downstream neuron inclusion criteria were "not yet identified," which appears to explain why no neuron filter was implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural data are aligned only to the start of each `ft_trInd` frame block. The code does not explicitly align to corridor entry using `StartFr`, and it does not define a fixed-length post-entry window.

ii.
```python
for tr, idx in trial_frame_idx.items():
    ...
    tr_spk = spk[:, idx].astype(np.float32, copy=False)
```
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The notes call the approach "frame-aligned conversion" but do not give a more precise event-alignment rationale.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code estimates a frame duration `dt_frame` per session from trial duration divided by the number of frames in each trial, using the median across trials. No explicit temporal rebinning is applied. The metadata field `time_bin_size` is left as `None`, so the chosen bin size is implicit in the arrays rather than recorded in metadata.

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
'metadata': {
    ...
    'time_bin_size': None,
    ...
}
```

iii. There is no explicit justification beyond the general use of frame-level variables and trial start/end times. The notes never recover the reference imaging rate and instead leave behavior timing as an inferred quantity.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime` and `Trial_start_time`, together with the inferred frame duration `dt_frame`.

ii.
```python
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
sound = np.asarray(beh['SoundTime'], dtype=float)
...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
```

iii. The notes identify `SoundTime` and trial start/end times as available trial-structured behavior variables and generally justify reusing those behavior fields directly.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the sound time is converted from MATLAB-day units to seconds relative to trial start. The per-bin value is then `sound_sec - time_since`, where `time_since` is a synthetic uniformly spaced time axis created from `np.arange(idx.size) * dt_frame`. If the sound or start time is missing, the whole vector is filled with `NaN`.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
...
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. No explicit justification is given beyond the implicit assumption that trial-level `SoundTime` can be placed onto a uniformly spaced within-trial time axis.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same per-trial frame index array `idx` and the same number of time bins as the neural matrix for that trial. There is no separate interpolation to recorded frame timestamps.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32) ...
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The notes' general justification is that the conversion is frame-aligned and uses precomputed per-trial frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived from the session chronology in the master index. Instead it is derived from the source behavior filename, for example whether the filename contains `before_learning`, `after_learning`, `testN`, or `trainN`.

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

iii. No explicit justification is written for this choice. The best available evidence is the Step 2 note that behavior filenames encode experiment-group structure.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code maps filename patterns to scalar day values: `before_*` to `1.0`, `after_*` to `5.0`, `testN` to `N`, `trainN` to `N`, otherwise `0.0`. That scalar is then broadcast across all time bins of every trial in the session.

ii.
```python
day = infer_day(beh_source)
...
day_arr = np.full(idx.size, day, dtype=np.float32)
...
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The notes do not explicitly justify the mapping. It is only implicit in the code and in the AI's emphasis on experiment-group filenames during dataset exploration.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time` plus the inferred frame duration `dt_frame`. The frame-level trial index `ft_trInd` determines how many bins belong to each trial.

ii.
```python
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. No explicit justification is given beyond using trial-structured behavior fields and frame-grouped trial segments.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code creates a uniformly spaced vector beginning at exactly `0.0` for the first kept bin of each trial and increasing by the inferred `dt_frame`. It does not use recorded frame timestamps and does not allow values before corridor entry.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
...
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. There is no explicit justification beyond the assumption of a regular within-trial frame grid.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by giving each trial exactly the same number of bins as its neural array and using the same `idx`-defined trial segmentation. No separate timestamp interpolation is used.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The notes justify the overall approach as frame-aligned per-trial conversion.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the trial-level `isRew` array.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
...
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. The notes explicitly say reward/sound structure is consistent across code, data, and methods and that `isRew` should be used directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No substantive processing is applied other than casting the per-trial value to an integer and broadcasting it across all time bins in that trial.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. The notes justify direct use of the reward field without further transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trial-level `WallName` array.

ii.
```python
wall = np.asarray(beh['WallName'])
...
cache.append((tr_spk, inp, str(wall[tr]), ...))
```

iii. The notes describe `WallName` as one of the key trial-structured stimulus/category variables available in the behavior files.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code keeps each distinct wall name as its own category rather than collapsing them to four base textures. It first gathers all unique wall names across the kept sessions, builds `wall_to_id`, and then fills the first output row of each trial with that wall-name ID across all time bins.

ii.
```python
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
...
for wall_name, out in output_trials:
    out = out.copy()
    out[0, :] = wall_to_id[wall_name]
    sess_out.append(out)
```

iii. The notes do not justify this choice explicitly. Step 7 sample statistics show the AI was intentionally monitoring the distribution of fine-grained labels like `circle1`, `circle2`, `leaf1`, `leaf2`, `rock1`, `wood1`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickTrind` and `LickTime`, together with `Trial_start_time` and the inferred `dt_frame`.

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
...
rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
lick_series[idx[np.unique(bins)]] = 1
```

iii. The notes justify using `LickTrind` and `LickTime` because those fields were prominent in both the reference utilities the AI inspected and the dataset exploration summary.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code converts lick timestamps to seconds relative to trial start, floors those times into inferred frame bins, marks unique bins as 1, and leaves all other bins 0. The resulting per-frame lick flag is then indexed by the trial's `idx`.

ii.
```python
lick_series = np.zeros(n_frames, dtype=np.int64)
...
rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
rel = rel[np.isfinite(rel)]
...
bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
lick_series[idx[np.unique(bins)]] = 1
```

iii. The notes do not explicitly defend the time-binning step. The closest justification is the Step 1 note that the reference code reconstructs lick rasters from lick times and lick trial indices.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. After building a session-wide `lick_series`, the code aligns licking to neural data by indexing that series with the same per-trial frame indices `idx` used to slice the neural matrix.

ii.
```python
lick = lick_series[idx]
...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
```

iii. The notes' stated rationale is the general frame-aligned design using precomputed trial frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_PosCum`, not from `ft_Pos`.

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
...
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The notes list `VRposCum` as a key available behavior variable and Step 6 explicitly says the implementation used `ft_PosCum`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code wraps cumulative position into the corridor by taking modulo `corridor_len`, then rescales the resulting position to four equal fractions of the corridor and converts those to integer bins 0-3.

ii.
```python
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
...
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. The notes do not explicitly justify the modulo-and-rescale choice. The best implicit justification is the task instruction requiring four equal-length corridor bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded by dividing the corridor into four equal fractional segments using `pos / corridor_len * 4`, then clipping the integer result into the range 0-3.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. No explicit justification is provided beyond the implied desire to create four equal bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by computing `pos` from the same trial frame indices `idx` used to extract the neural data, so the position vector and neural matrix have the same number of bins.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The notes justify this only at the general level of using frame-aligned per-trial indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is not read from a raw speed variable. Instead it is derived from consecutive differences of the wrapped position signal `pos`, divided by the inferred `dt_frame`.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. No explicit justification is given. The notes only say the implementation used `ft_PosCum` and `ft_move`; in the final code `ft_move` is loaded but not actually used.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code computes a speed trace per trial from position differences, concatenates all trial speeds within a session, computes the 25th/50th/75th percentile thresholds from those values, and then bins each trial's speed trace with `np.digitize`.

ii.
```python
all_speeds = []
...
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
...
all_speeds.append(speed)
...
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0], dtype=np.float32)
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The notes do not explicitly justify the percentile-threshold approach; the only related evidence is that Step 7 monitored the resulting output distribution.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It uses three within-session quantile thresholds at 25%, 50%, and 75%, then assigns categories with `np.digitize`.

ii.
```python
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) ...
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. No explicit justification is given. The choice appears to come from the decoder-task instruction that the output should be split into four bins corresponding to 25% of the data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by being computed from the same per-trial frame sequence `idx` used for neural extraction, so each trial's speed vector has the same number of bins as its neural matrix.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The notes again justify this only through the general frame-aligned trial-indexing approach.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles data issues in several ad hoc ways: it trims all streams to the minimum shared frame count; ignores non-finite `ft_trInd` values; skips trials whose supporting trial-level arrays are too short; fills `time_to_sound` with `NaN` if sound timing is missing; keeps the richer duplicate behavior session when behavior files disagree; and drops sessions with fewer than two surviving trials.

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
...
valid_frame = np.isfinite(ft_tr)
...
if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
    continue
...
time_to_sound = ... if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```
```python
if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
    sessions[sess] = dat
```

iii. The notes explicitly justify only the duplicate-session handling, documenting that a poorer duplicate behavior entry had to be replaced by the richer one. The other safeguards are implicit defensive coding choices.

## 12-a. What are the most time-consuming steps of the code?

i. The AI's notes emphasize per-session trial indexing and the overall full-conversion runtime, implying that session construction is the main bottleneck. From the code, the most expensive steps are loading and concatenating large `spks` arrays, then looping over trials to estimate `dt_frame`, bin licks, compute per-trial features, and concatenate all speeds for quantile binning.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```
```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    ...
```
```python
for tr in np.unique(lick_tr_i):
    ...
for tr, idx in trial_frame_idx.items():
    ...
speed_all = np.concatenate(all_speeds)
```

iii. The explicit justification in the notes is performance-oriented rather than scientific: precomputing `trial_frame_idx` once per session cut sample conversion runtime roughly in half, and the full run still took about 36 minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code could further vectorize the `dt_frame` estimation loop over trials, the lick-assignment loop over unique lick trials, and the second per-trial pass that recomputes/broadcasts features after the cache is built. The AI did partially optimize one area by replacing repeated per-trial frame searches with a precomputed `trial_frame_idx`.

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
```python
for tr, idx in trial_frame_idx.items():
    ...
```

iii. The notes explicitly say the AI added a speedup by "precomputing `trial_frame_idx` mapping once per session instead of repeated `np.where` scans."

## 12-c. What processing does the code repeat multiple times?

i. The code repeats trial-level work in multiple passes: one pass counts frames per trial to estimate `dt_frame`, another bins licks by trial, another constructs and caches each trial's neural/input/output intermediates, and a final pass revisits every cached trial to quantize speed and finalize outputs.

ii.
```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
```
```python
for tr in np.unique(lick_tr_i):
    ...
```
```python
for tr, idx in trial_frame_idx.items():
    ...
    cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
    all_speeds.append(speed)
...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. There is no explicit written justification for these repeated passes beyond the general performance note that the AI tried to optimize trial indexing.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `ft_move` and `ft_WallID` but never uses them downstream. It also computes `ntrials` only for the `dt_frame` loop, creates a cached `wall_name` string only to remap it later in `main`, and copies each output trial again in order to overwrite the first row with the wall-name ID.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
...
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
```
```python
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
```
```python
for wall_name, out in output_trials:
    out = out.copy()
    out[0, :] = wall_to_id[wall_name]
    sess_out.append(out)
```

iii. No explicit justification is given for these extra steps. They appear to be leftover implementation choices rather than deliberate processing required by the task.
