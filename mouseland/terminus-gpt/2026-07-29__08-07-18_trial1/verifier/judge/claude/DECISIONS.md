# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans all `.npy` files in `data/beh/` and loads each one as a dictionary. It then iterates over all keys in each dictionary, filtering those matching a session-ID regex pattern (`^[A-Za-z0-9]+_\d{4}_\d{2}_\d{2}_\d+(?:_swap[12])?$`). It separately scans for all `*_neural_data.npy` files and takes the intersection of behavior session keys and neural file names. It does NOT use the `Imaging_Exp_info.npy` master index.

ii.
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

def load_neural_files(data_dir: Path):
    files = {}
    for p in sorted(data_dir.rglob('*_neural_data.npy')):
        files[p.name.replace('_neural_data.npy', '')] = p
    return files

common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The AI chose to discover sessions by scanning all files and matching session IDs via regex, rather than using the provided `Imaging_Exp_info.npy` master index. It resolved duplicate session entries across behavior files by preferring the "richer" dictionary (higher key count and `ft_trInd` size).

## 1-b. How are the data split into subjects?

i. Subjects are extracted by splitting the session ID string on underscores and taking the first element. Unique subjects are sorted alphabetically.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
# ...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. The session naming convention embeds the subject ID as the first component before the date, so the AI extracts it from the session ID string.

## 1-c. How are the data split into sessions?

i. A session is identified by the full session ID string (e.g., `TX109_2023_05_12_1`). Sessions are the intersection of behavior and neural file session IDs, yielding 76 sessions. The AI does not use `Imaging_Exp_info.npy` to enumerate sessions.

ii.
```python
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The AI found 100 behavior sessions and 89 neural sessions, with 76 in common. Some behavior session keys include swap suffixes that don't match neural file names, and some sessions listed in `Imaging_Exp_info.npy` have behavior data keyed differently (with `_stimtype` suffixes).

## 1-d. How are the data split into trials?

i. Trials are identified using `ft_trInd`, the per-frame trial index. The AI finds unique trial indices and groups frame indices by trial. It does NOT use `ft_CorrSpc` to restrict to corridor-space frames only.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts)
                   else valid_idx[starts[i]:]
                   for i, tr in enumerate(uniq_tr)}
```

iii. The AI uses `ft_trInd` to group frames by trial but does not filter by `ft_CorrSpc` (corridor space), so frames outside the texture corridor (e.g., in the grey space) are included.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only filtered by having at least 2 frames and by having valid metadata indices (trial index within range of `wall`, `is_rew`, `sound`, `tstart` arrays). Sessions with fewer than 2 surviving trials are dropped. There is no trial length filtering (no percentile-based outlier removal).

ii.
```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
```

iii. The AI does not mention any trial length outlier filtering in CONVERSION_NOTES.md. It only removes trials with fewer than 2 frames or out-of-bounds metadata.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the `*_neural_data.npy` files. The AI concatenates the list of per-plane spike arrays into a single neurons-by-frames matrix.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])

def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
```

iii. The AI identified `spks` as the deconvolved neural activity and concatenated across imaging planes.

## 2-b. How is the `neural` data processed?

i. The traces are concatenated across planes, cast to float32, and truncated to the minimum of neural frames and behavior array lengths. No further processing (no normalization, no dF/F). Neural data is stored as float32.

ii.
```python
spk = concat_spks(nobj['spks'])
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
# ...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The AI noted in CONVERSION_NOTES.md that analyses are based on deconvolved fluorescence traces, so no further processing was applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons from all imaging planes are kept. The retinotopy data is not loaded, and brain regions are all labeled as `'unknown'`. The AI's output has 4,047,169 total neurons across 76 sessions.

ii.
```python
data['brain_regions'] = ['unknown']
# ...
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. The AI did not implement any visual-area-based neuron filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by selecting frames belonging to each trial based on `ft_trInd`. The first frame of each trial is the alignment point.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The AI aligns using frame indices grouped by `ft_trInd`, which starts at corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The AI sets `time_bin_size` to `None` in the metadata, but estimates `dt_frame` per session from trial start/end times and frame counts.

ii.
```python
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
# ...
'time_bin_size': None,
```

iii. The AI does not record the known imaging rate (3.17 Hz) from the reference paper and leaves `time_bin_size` as None. It estimates frame duration per session for timing calculations.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the absolute time of the sound cue per trial) and `Trial_start_time` (the absolute time of trial start), both as MATLAB datenums.

ii.
```python
sound = np.asarray(beh['SoundTime'], dtype=float)
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0)
```

iii. The AI uses absolute time fields rather than frame-indexed fields (`SoundFr`).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound time relative to trial start is computed in seconds (converting from MATLAB datenums via `* 86400`). The time-to-sound for each frame is `sound_sec - time_since_trial_start`. This gives positive values before the sound and negative after, matching the "time to" semantics.

ii.
```python
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. The AI computes time_to_sound as `sound_sec - time_since_trial_start`, yielding positive values before the cue and negative after.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed for the same frame indices used for the neural data of that trial.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. The input is computed for each frame index in the trial, matching the neural data alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the behavior source file name (e.g., `Beh_before_learning.npy`, `Beh_after_learning.npy`, `Beh_test1.npy`), parsed with regex to infer a day number.

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

day = infer_day(beh_source)
```

iii. The AI inferred training day from file naming conventions rather than computing it from session ordering per mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The file name is parsed: `before_learning` maps to 1.0, `after_learning` to 5.0, `testN` to N, `trainN` to N, and anything else to 0.0. The day value is broadcast across all frames of each trial.

ii.
```python
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. The AI uses heuristic name parsing. This produces only values {0.0, 1.0, 2.0, 5.0} as seen in the verification output, which does not correctly capture training progression for all mice (the reference counts 0-7).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From frame count and estimated frame duration (`dt_frame`), not from actual frame timestamps.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The AI reconstructs time from frame index and an estimated frame period.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed as `frame_index * dt_frame` where `dt_frame` is the median of `(Trial_end_time - Trial_start_time) * 86400 / n_frames_in_trial` across trials in the session.

ii.
```python
trial_dur_sec = (tend - tstart) * 86400.0
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The AI estimates frame duration per session rather than using the known frame rate or actual frame timestamps (`ft`).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed for the same frame indices as the neural data, since `idx` is the set of frames for that trial.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. Aligned by construction since it's computed per-frame for the trial's frames.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial was in the rewarded corridor.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. Direct use of the `isRew` field.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value is broadcast across all frames of the trial. No further processing.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. Straightforward broadcast of per-trial value.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the name of the wall texture for each trial.

ii.
```python
wall = np.asarray(beh['WallName'])
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
# ...
out[0, :] = wall_to_id[wall_name]
```

iii. The AI uses `WallName` directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each individual wall name (e.g., `circle1`, `leaf1_swap1`, `wood5`) is treated as its own category. The AI does NOT group wall names into the four base texture categories (circle, leaf, rock, wood). This produces 13 categories instead of 4. The value is broadcast per-trial across all frames.

ii.
```python
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
# output_values:
[wall_names, ['no_lick', 'lick'], ['bin1', 'bin2', 'bin3', 'bin4'], ['q1', 'q2', 'q3', 'q4']]
```

iii. The AI treats each individual wall texture variant as a separate category rather than grouping into the four base categories specified in the instructions ("e.g. circle, leaf, etc.").

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTrind` (trial index of each lick event) and `LickTime` (absolute timestamp of each lick event).

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
```

iii. The AI uses `LickTrind`/`LickTime` rather than `LickFr` (lick frame indices), which requires complex reconstruction of lick-to-frame mapping.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events are found by matching `LickTrind`. Lick times are converted to relative time since trial start, then divided by `dt_frame` to estimate which frame they fall in. A frame is marked 1 if a lick falls in it.

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

iii. This is a complex reconstruction that involves estimating frame timing, which introduces approximation error compared to using `LickFr` directly.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick series is pre-computed for all frames, then indexed by the same trial frame indices used for neural data.

ii.
```python
lick = lick_series[idx]
```

iii. Aligned by sharing the same frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_PosCum`, the cumulative position at each frame.

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The AI uses cumulative position (`ft_PosCum`) rather than corridor position (`ft_Pos`).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The cumulative position is taken modulo the corridor length (default 4.0 m), converting it to position within the current corridor.

ii.
```python
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The AI wraps cumulative position back into corridor bounds using modulo.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is scaled to the corridor length and multiplied by 4 to produce bin indices 0-3, each representing 1 m.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. Four equal-length bins of 1 m each.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is computed for the same frame indices as neural data.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. Same frame indexing as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Speed is derived from position by computing `np.diff(pos)` divided by `dt_frame`. It does NOT use `ft_RunSpeed`.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The AI derives speed from the numerical derivative of position rather than using the provided running speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is computed as the frame-to-frame difference in position divided by estimated frame duration. Then all speeds across all trials in a session are collected, and quartile boundaries are computed using `np.quantile` at [0.25, 0.5, 0.75]. Speed is then binned into 4 categories using `np.digitize`.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
# ...
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75])
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The AI uses value-based quantiles via `np.quantile` + `np.digitize` rather than rank-based quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.quantile` at [0.25, 0.5, 0.75] to find three threshold values, then `np.digitize` to assign each speed value to one of 4 bins. This is value-based rather than rank-based.

ii.
```python
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0])
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. Value-based quantile binning. When many values are tied (e.g., at zero speed), this can produce very unequal bin sizes. The verification output shows bins of approximately [0.040, 0.295, 0.415, 0.250] - far from equal quarters.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from the same frame-indexed position data used for neural alignment.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Same frame-level alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates all arrays to the minimum of neural frames and behavior array lengths. It checks for NaN/infinite values in timing fields and skips lick events with invalid times. Duplicate behavior sessions across files are resolved by preferring the "richer" entry. If `LickTrind`/`LickTime` are missing, licking defaults to all zeros.

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
# ...
rel = rel[np.isfinite(rel)]
# ...
if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
```

iii. The AI handles length mismatches and NaN values but does not handle missing corridor space information or trial length outliers.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files is the most time-consuming step. The full conversion took 2177 seconds (~36 min) for 76 sessions. The resulting pickle file was extremely large (344 GB) due to keeping all neurons (no retinotopy filtering).

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. The AI noted in CONVERSION_NOTES.md that the full decoder training was impractical due to the 344 GB pickle size.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick assignment loop iterates over unique trial indices and reconstructs lick timing per trial, which could potentially be vectorized. The trial frame index computation uses `np.unique` with return_index, which is reasonably efficient.

ii.
```python
for tr in np.unique(lick_tr_i):
    idx = trial_frame_idx.get(int(tr))
    # ... per-trial lick reconstruction
```

iii. The AI noted optimizing trial index computation in CONVERSION_NOTES.md.

## 12-c. What processing does the code repeat multiple times?

i. The AI does not appear to repeat major processing steps. However, it loads all behavior files upfront into memory, which could be wasteful for the sample case.

ii. N/A

iii. N/A

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ft_move` and `ft_WallID` per-frame arrays but does not use them in the final output. It also computes `corridor_len` from the data but uses a default of 4.0 in most cases.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
```

iii. These arrays are loaded and truncated but never used in the output construction.
