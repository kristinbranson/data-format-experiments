# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every behavior `.npy`, retains dictionary entries whose keys look like sessions, scans recursively for neural files, and processes the intersection. For duplicate behavior keys it keeps the “richer” dictionary. It does not use `Imaging_Exp_info.npy` as the master index or load retinotopy.

ii.
```python
for p in sorted((data_dir / 'beh').glob('*.npy')):
    obj = np.load(p, allow_pickle=True).item()
    ...
for p in sorted(data_dir.rglob('*_neural_data.npy')):
    files[p.name.replace('_neural_data.npy', '')] = p
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The notes say session-like keys were filtered and duplicate overwriting was fixed by preferring the entry with more keys and `ft_trInd` values. This yielded 76 common sessions.

## 1-b. How are the data split into subjects?

i. The subject is the substring before the first underscore in each common session ID; sorted unique values form `subjects`, and each retained session receives the corresponding index.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. The notes identify 19 true animal IDs and state that session naming embeds subject, date, and session number.

## 1-c. How are the data split into sessions?

i. Session dictionaries and neural files are matched by regex-like session ID, including optional swap suffixes. Only matches having both sources are processed; sessions with fewer than two resulting trials are discarded.

ii.
```python
SESSION_RE = re.compile(r'^[A-Za-z0-9]+_\d{4}_\d{2}_\d{2}_\d+(?:_swap[12])?$')
common = sorted(set(beh_sessions) & set(neural_files))
if len(neural_trials) < 2:
    continue
```

iii. The notes justify matching true session-like keys to separate neural files and report 76 matched, retained sessions after resolving a duplicate-key bug.

## 1-d. How are the data split into trials?

i. Finite `ft_trInd` values are converted to integers and frames are grouped by trial ID. Trials with fewer than two frames or missing per-trial metadata are skipped. The agent does not restrict frames to `ft_CorrSpc` (corridor texture).

ii.
```python
valid_idx = np.flatnonzero(ft_tr_int >= 0)
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] ... for i, tr in enumerate(uniq_tr)}
if idx.size < 2:
    continue
```

iii. The notes describe “frame-aligned conversion using `ft_trInd`” and precomputing the trial-index map for speed, but do not justify including frames outside the corridor.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed only if they contain fewer than two indexed frames or lack entries in `WallName`, `isRew`, `SoundTime`, or `Trial_start_time`; sessions with fewer than two surviving trials are removed. There is no long/stopped-trial filter.

ii.
```python
if idx.size < 2:
    continue
if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
    continue
```

iii. The notes acknowledge that explicit trial exclusion criteria were not confirmed. No justification for omitting the reference’s extreme-duration control is recorded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from every plane in the `spks` list of `<session>_neural_data.npy`, concatenated over neurons. Retinotopy `iarea` is not used.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. The notes identify `spks` as the Suite2p-deconvolved activity used by the paper and say its list-backed representation needs concatenation.

## 2-b. How is the `neural` data processed?

i. Plane arrays are cast to float32 and concatenated; columns are truncated to the shortest behavioral/neural stream and then selected by trial frame indices. No fluorescence transformation, padding, or rebinning is applied.

ii.
```python
return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
spk = spk[:, :n_frames]
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The agent states that `spks` already contains deconvolved fluorescence, consistent with the methods, so recomputing dF/F or deconvolution is unnecessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered: all concatenated ROIs are retained and every neuron is labeled as one `unknown` region.

ii.
```python
'brain_regions': ['unknown'],
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. The notes mention Suite2p cell classification but say exact downstream inclusion criteria were not identified. They do not justify omitting the available retinotopic-area selection.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each array begins at the first frame labeled with that trial, which the agent treats as corridor entry/trial start, and continues over every grouped frame. It does not explicitly align with `StartFr` or restrict to corridor frames.

ii.
```python
for tr, idx in trial_frame_idx.items():
    tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. Metadata declares “corridor entry / trial start.” The notes generally call the conversion frame-aligned, without validating that the first `ft_trInd` frame equals corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No neural rebinning is applied, but a session-specific `dt_frame` is estimated as the median trial duration divided by its number of labeled frames, with a 0.1 s fallback. Metadata leaves `time_bin_size` as `None`.

ii.
```python
frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
...
'time_bin_size': None,
```

iii. The notes do not justify this estimate and leave the behavior time bin unresolved, despite the reference’s 3.17 Hz (about 315 ms) imaging grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and `Trial_start_time`, plus the inferred session frame period.

ii.
```python
sound = np.asarray(beh['SoundTime'], dtype=float)
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
```

iii. The notes identify the sound cue as a task variable present across conditions but provide no detailed rationale for choosing timestamps rather than frame-aligned fields.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. MATLAB-day timestamps are differenced and multiplied by 86400. A synthetic elapsed-time axis (`arange * dt_frame`) is subtracted, producing positive values before and negative after the cue; invalid timestamps produce NaNs.

ii.
```python
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. The implicit rationale is to convert MATLAB dates to seconds and express time remaining to the cue. The agent did not document why a constant inferred period accurately represents frame timestamps.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. One synthetic value is created for each neural column, beginning at zero elapsed time for the first frame in the grouped trial.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr])
```

iii. The notes characterize all streams as frame-aligned. Lengths match, but the timing is inferred rather than taken from each frame’s `ft` timestamp.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is inferred from the name of the behavior file that supplied the session, using substrings such as `before_learning`, `after_learning`, `testN`, and `trainN`.

ii.
```python
def infer_day(source_name: str):
    ...
    m = re.search(r'test(\d+)', name)
    ...
    return 0.0
```

iii. No explicit justification is recorded. The notes discuss experiment-group filenames, but not why their labels are a reliable per-mouse training-day index.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Before sessions map to 1, after sessions to 5, numbered test/train filenames to that number, and everything else to 0. This scalar is broadcast across the trial.

ii.
```python
if 'before_learning' in name or 'before_grating' in name: return 1.0
if 'after_learning' in name or 'after_grating' in name: return 5.0
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. The apparent intention was to recover nominal experimental training days from filenames. The notes do not compare it to chronological session counting per mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived only from the number/order of grouped trial frames and the estimated `dt_frame`; `Trial_start_time` participates indirectly in estimating trial durations but not the per-frame axis.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The notes state the conversion is frame aligned; no further justification is provided.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame indices 0 through N−1 are multiplied by the session’s estimated frame period, so every retained trial starts exactly at zero.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. This is a simple synthetic regular time axis. The agent did not justify ignoring measured frame times or the precise fractional start frame.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Its length is exactly the number of selected neural columns, and element j is assigned to neural column j.

ii.
```python
tr_spk = spk[:, idx]
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The notes’ justification is common frame indexing; array alignment is direct even though absolute timing can be wrong.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` array.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
```

iii. The notes say `isRew` agrees with the code and methods’ rewarded/unrewarded corridor structure.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The integer trial flag is converted to float32 and broadcast across all retained time points.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. The notes support using the field directly; no additional transformation is needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from each trial’s `WallName` string. Categories are all distinct wall names found across processed sessions.

ii.
```python
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
```

iii. The notes cite `WallName`, `WallType`, and category utilities, but the implementation’s distinct-wall choice is not justified against the requested broad categories (circle, leaf, rock, wood).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each unique wall-name string receives a sorted integer ID, and that ID is broadcast over the trial. No collapsing of crops or swapped walls to four base textures occurs.

ii.
```python
out[0, :] = wall_to_id[wall_name]
```

iii. The notes report sample distributions for names such as `circle1` and `circle2`, confirming the fine-grained interpretation, but do not justify it.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTrind`, `LickTime`, and `Trial_start_time`.

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
```

iii. The notes emphasize reference utilities that reconstruct trial lick rasters from lick trial indices and positions/times.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick timestamps are made relative to trial start, converted to seconds, divided by inferred `dt_frame`, floored to a bin, clipped to the trial’s frame range, deduplicated, and marked binary.

ii.
```python
rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
lick_series[idx[np.unique(bins)]] = 1
```

iii. The agent intended a binary frame-aligned lick raster. It gives no justification for clipping out-of-range licks onto trial endpoints or for preferring inferred time bins over `LickFr`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Computed lick bins are written into a session-length array at trial frame indices, then sliced by the same `idx` used for neural data.

ii.
```python
lick_series[idx[np.unique(bins)]] = 1
lick = lick_series[idx]
```

iii. The notes call this frame-aligned. Array lengths align, though event-to-frame timing depends on the inaccurate inferred period.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-wise cumulative position `ft_PosCum` and optional `Corridor_Length` (default 4.0).

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
```

iii. The notes state frame-aligned conversion uses `ft_PosCum`; no rationale is given for choosing it over within-corridor `ft_Pos`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Cumulative position is reduced modulo corridor length to obtain a nominal within-corridor coordinate.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The implied intention is to turn cumulative distance into repeated 0–4 m corridor position. The notes do not validate units or behavior in gray space.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is normalized by corridor length, multiplied by four, integer-truncated, and clipped to 0–3.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. This implements four equal fractions and would be four 1 m bins only when length is correctly represented as 4 m.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_PosCum` is truncated to the common stream length and indexed with the identical per-trial frame indices used for neural columns.

ii.
```python
ft_pos = ft_pos[:n_frames]
pos = np.mod(ft_pos[idx], corridor_len)
```

iii. The notes justify alignment through common frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is re-derived from successive modulo-transformed `ft_PosCum` values and inferred `dt_frame`; raw `ft_RunSpeed` is not used.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. No rationale is documented. Although `ft_move` is loaded, it is not used; the notes merely say the conversion used it.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. First differences in within-corridor position are divided by the estimated frame period. The first speed is forced to zero; modulo wraparound can create negative jumps.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The code’s apparent goal is physical speed from displacement/time, but the notes do not discuss wraparound artifacts or why the provided speed stream is ignored.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speeds from all retained trials in a session are pooled; the 25th, 50th, and 75th value quantiles are thresholds for `np.digitize`.

ii.
```python
qs = np.quantile(speed_all, [0.25, 0.5, 0.75])
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. This aims for quartiles, as requested. The sample notes reveal very unequal bins, consistent with tied values and value-threshold quantiles rather than rank quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed directly from the position series already selected by the same trial indices, yielding one speed value per neural frame.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The notes characterize outputs as frame aligned; lengths do match exactly.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Streams are truncated to their common minimum length; nonfinite trial indices are invalidated; malformed objects/keys are skipped; missing optional lick arrays become empty; bad timestamps produce NaNs; missing per-trial metadata and undersized trials are skipped; duplicate behavior sessions select the richer entry.

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
valid_frame = np.isfinite(ft_tr)
lick_tr = np.asarray(beh.get('LickTrind', []))
if tr >= len(wall) or tr >= len(is_rew) ...: continue
```

iii. The notes specifically document diagnosing a duplicate overwrite and preferring the richer dictionary. Other guards are defensive, though NaNs are retained and out-of-range licks are clipped rather than dropped.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating very large neural files and serializing the repeatedly copied per-trial neural matrices dominate. Full conversion took about 36 minutes and produced a 344 GB pickle that could not practically be loaded for full training.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
tr_spk = spk[:, idx].astype(np.float32, copy=False)
pickle.dump(data, f)
```

iii. The notes estimate roughly 41 seconds per sampled session and explicitly identify the giant full pickle as the Step 11 blocker.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The frame-period loop, lick loop over trials, trial construction loop, session loop, and output-remapping loop remain. The agent already replaced repeated per-trial `np.where` scans with a precomputed mapping; lick binning could be grouped/vectorized further.

ii.
```python
for tr in range(min(ntrials, len(trial_dur_sec))): ...
for tr in np.unique(lick_tr_i): ...
for tr, idx in trial_frame_idx.items(): ...
for wall_name, out in output_trials: ...
```

iii. The notes credit precomputing `trial_frame_idx` with reducing the two-session sample runtime from about 160 to 83 seconds.

## 12-c. What processing does the code repeat multiple times?

i. It traverses trials first to estimate frame periods, again to place licks, again to construct/cache trials, and again to finalize speed bins. Neural data is also copied into every trial and later traversed during serialization.

ii.
```python
for tr in range(...): ... frame_periods.append(...)
for tr in np.unique(lick_tr_i): ...
for tr, idx in trial_frame_idx.items(): ... cache.append(...)
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache: ...
```

iii. The notes discuss only the eliminated repeated frame-index scans, not these remaining repeated passes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/truncates `ft_move` and `ft_WallID` but never uses them; stores an initial all-−1 stimulus row only to overwrite it later; and materializes a cache plus `all_speeds` before building final outputs. The huge float32 per-trial neural copies also inflate storage without adding resolution.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
np.full(tr_spk.shape[1], -1, dtype=np.int64)
out = out.copy(); out[0, :] = wall_to_id[wall_name]
```

iii. The notes incorrectly list `ft_move` as part of the implemented frame alignment and do not identify this discarded work; they do acknowledge that the representation needs compression.
