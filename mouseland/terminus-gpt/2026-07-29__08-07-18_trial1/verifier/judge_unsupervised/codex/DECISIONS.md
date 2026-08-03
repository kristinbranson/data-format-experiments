# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads behavior by scanning only top-level `data/beh/*.npy`, filtering keys that look like session IDs, and keeping the “richer” duplicate when the same session appears in multiple files. It loads neural data by recursively finding every `*_neural_data.npy`, then keeps only the intersection of behavior and neural session IDs. Trials are not preloaded globally; they are created later from frame-level trial indices.

ii. ```python
for p in sorted((data_dir / 'beh').glob('*.npy')):
    obj = np.load(p, allow_pickle=True).item()
    ...
    if isinstance(sess, str) and SESSION_RE.match(sess) and isinstance(dat, dict):
        if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
            sessions[sess] = dat
...
for p in sorted(data_dir.rglob('*_neural_data.npy')):
    files[p.name.replace('_neural_data.npy', '')] = p
...
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. In `CONVERSION_NOTES.md`, the agent justified filtering to session-like keys because behavior files also contain metadata/group keys, and later justified the duplicate-session rule after finding that `TX109_2023_03_27_1` was overwritten by a poorer example entry. The trajectory also records that the reference code operated on per-session dictionaries keyed by session ID.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session name prefix before the first underscore, and each kept session gets a `subject_idx` based on that prefix.

ii. ```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. The agent’s notes say the session naming convention embeds the subject ID, date, and session number, so splitting on `'_'` was treated as the natural way to recover mouse IDs.

## 1-c. How are the data split into sessions?

i. Each unique session ID present in both the behavior loader and neural loader becomes one converted session. Sessions with fewer than two usable trials are dropped.

ii. ```python
common = sorted(set(beh_sessions) & set(neural_files))
...
for sess in common:
    neural_trials, input_trials, output_trials = build_session(...)
    if len(neural_trials) < 2:
        continue
    data['neural'].append(neural_trials)
```

iii. The notes repeatedly describe the conversion as operating on “common behavior+neural sessions,” and the full-conversion log confirms the final retained set was 76 such sessions.

## 1-d. How are the data split into trials?

i. Trials are split with the frame-aligned behavior field `ft_trInd`. The code converts valid frame labels to integers, groups contiguous valid frames by trial index, and uses those frame indices to cut both neural and behavioral arrays into per-trial matrices.

ii. ```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
...
valid_frame = np.isfinite(ft_tr)
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)
...
valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] ...}
```

iii. In the trajectory, the agent explicitly switched from timestamp-based reconstruction to the `ft_*` frame-aligned fields after noticing that the reference `utils.py` also used `ft_trInd`, `ft_WallID`, `ft_PosCum`, and related arrays.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. A trial is kept only if it has at least 2 frames and if the per-trial metadata arrays are long enough to index that trial. There is no explicit behavioral-quality, movement-quality, or stimulus-quality curation beyond those checks.

ii. ```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
```

iii. The notes focus on format validity and session recoverability, not on reproducing any stricter trial QC. The trajectory shows the agent mainly used these checks to avoid broken indexing and to satisfy the decoder’s requirement of at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived entirely from the neural file key `spks`, which is a list of arrays that the script concatenates across axis 0.

ii. ```python
def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
...
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. The trajectory records the agent finding the reference-code line `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']],0)` in `code/utils.py:341`, and it cites the methods text stating that analyses used deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The script concatenates the `spks` blocks, casts them to `float32`, truncates them to the common frame count with the frame-aligned behavior arrays, and then slices them into per-trial matrices using `trial_frame_idx`. It does not z-score, smooth, interpolate by position, or select subsets of neurons.

ii. ```python
spk = concat_spks(nobj['spks'])
...
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The agent justified using `spks` as the already processed deconvolved activity stream. In the trajectory it also admits the converter is “still approximate” and does not reproduce all of the reference code’s downstream interpolation/normalization steps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neuron-level QC in the converter. The only filtering is indirect: frames are truncated to shared length, invalid `ft_trInd` frames are ignored, and trials shorter than 2 frames are skipped. All neurons in the concatenated `spks` arrays are retained, and brain regions are set to `unknown`.

ii. ```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
...
if idx.size < 2:
    continue
...
'brain_regions': ['unknown'],
'brain_region_idx': [],
```

iii. The notes mention Suite2p cell classification and deconvolution from the paper, but the converter itself performs no additional cell-quality filtering. The trajectory does not show the agent implementing the corridor-neuron or area-based selections used in the reference figure code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to corridor entry / trial start by taking all frames whose `ft_trInd` equals a given trial index. Within each trial, time zero is implicitly the first retained frame of that trial.

ii. ```python
'temporal_alignment_event': 'corridor entry / trial start',
'off_start': 0.0,
...
for tr, idx in trial_frame_idx.items():
    ...
    tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The instructions asked for temporal alignment based on trial start, and the notes say the agent deliberately pivoted to the frame-aligned `ft_*` fields because they matched the reference code’s trial/frame organization.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converter does not declare a fixed time bin size in metadata. Instead it estimates a per-session frame duration `dt_frame` from median `(trial duration / number of frames)` and then uses the native frames as bins. No explicit temporal rebinning is applied.

ii. ```python
trial_dur_sec = (tend - tstart) * 86400.0
...
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
...
'metadata': {
    'time_bin_size': None,
    ...
}
```

iii. The trajectory explicitly says the agent used a “proxy time axis” and “median trial duration because neural frame timestamps are still unresolved.” That is the main justification for leaving the bin size implicit.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime`, `Trial_start_time`, and the synthetic within-trial time axis built from `dt_frame`.

ii. ```python
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
sound = np.asarray(beh['SoundTime'], dtype=float)
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) ...
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. The agent’s Step 5 notes planned exactly this mapping from `SoundTime` to a continuous trial-aligned cue-timing variable, and the trajectory shows it chose the sign convention `sound_time - t` for “time to sound cue.”

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The code converts MATLAB-style day fractions to seconds by subtracting `Trial_start_time` from `SoundTime` and multiplying by `86400`. It then subtracts the within-trial elapsed time series so the value counts down to the cue. If cue timing is missing, it fills the whole series with `NaN`.

ii. ```python
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. The notes identify this as a continuous, time-varying decoder input. The trajectory shows the agent knew this was approximate because `time_since` itself is only an estimated frame-time axis.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same per-trial frame indices as the neural matrix. Each trial’s `time_to_sound` vector has the same length as `tr_spk.shape[1]`.

ii. ```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
...
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The agent’s justification was structural: by deriving everything from `idx`, all trialwise inputs and outputs stay synchronized with the neural frames even though the true per-frame timestamps were not available.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived from a field inside the session dictionary. It is inferred from the source behavior filename recorded in `beh_source`, for example `before_learning`, `after_learning`, `test1`, or `train2`.

ii. ```python
def infer_day(source_name: str):
    name = source_name.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 1.0
    if 'after_learning' in name or 'after_grating' in name:
        return 5.0
    m = re.search(r'test(\d+)', name)
    ...
```

iii. The Step 5 notes already flagged that day-of-training would likely have to be inferred from file/condition order rather than a clean raw variable, and the final script followed that plan.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script maps filename patterns to scalar day values: `before_* -> 1.0`, `after_* -> 5.0`, `testN -> N`, `trainN -> N`, else `0.0`. That scalar is then repeated across every time bin in the trial.

ii. ```python
day = infer_day(beh_source)
...
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. The justification in the notes is pragmatic rather than evidential: the agent needed a continuous per-trial day variable and could not find a dedicated raw field, so it fell back to filename-based inference.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `ft_trInd`, `Trial_start_time`, and `Trial_end_time` through the estimated frame size `dt_frame`.

ii. ```python
trial_dur_sec = (tend - tstart) * 86400.0
...
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    ...
    frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The trajectory explicitly says the agent used a proxy time axis because it had frame-aligned trial labels but not resolved neural-frame timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code estimates one session-level frame duration from the median trial duration divided by the number of frames in each trial, then generates a uniform elapsed-time vector with `np.arange(idx.size) * dt_frame` for every trial.

ii. ```python
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. In the trajectory the agent characterizes this as approximate and temporary, but it remained in the final script because no cleaner per-frame timebase was implemented.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by construction: the elapsed-time vector is built directly from the same `idx` used to slice the neural trial.

ii. ```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The justification is the same as for the cue-time input: all modalities are forced to share the same per-trial frame count, even if the frame-time spacing is only estimated.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial behavior field `isRew`.

ii. ```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
...
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. The methods excerpt in the trajectory distinguishes rewarded and unrewarded corridors, and the notes say the behavior files contain the same `isRew` flag used by the reference analysis.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial Boolean/integer `isRew` value is cast to an integer and then repeated across every frame of the trial as a constant time series.

ii. ```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The agent’s notes describe reward availability as a discrete per-trial variable, so repeating it across the trial was the chosen way to fit the decoder’s time-varying input shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial field `WallName`. Session-specific wall-name strings are collected globally, sorted, and mapped to integer category IDs.

ii. ```python
wall = np.asarray(beh['WallName'])
...
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
```

iii. The notes and trajectory both point to `WallName`, `UniqWalls`, and related category helpers in the reference code as the natural source of stimulus identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. During trial assembly the code stores the wall name string for each trial, initializes output row 0 as `-1`, and after all trials are built it overwrites that row with the integer ID for that wall category across all time bins of the trial.

ii. ```python
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
...
out[0, :] = wall_to_id[wall_name]
```

iii. The justification is mostly structural: the decoder expects categorical outputs encoded numerically, so the agent translated the native wall-name labels into dataset-wide integer categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTrind`, `LickTime`, and `Trial_start_time`. The script does not use `LickPos` for the final output.

ii. ```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
...
rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
```

iii. The notes say the reference behavioral code often used `LickPos` and `LickTrind`, but for the decoder output the agent chose time-binned licking from lick timestamps so it could align directly to neural frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code builds a full-session binary lick vector, converts each lick time to seconds relative to trial start, bins those relative times with `floor(rel / dt_frame)`, clips them to the trial length, and marks the corresponding frame bins as `1`. If there are no lick events, the series stays all zeros.

ii. ```python
lick_series = np.zeros(n_frames, dtype=np.int64)
...
for tr in np.unique(lick_tr_i):
    idx = trial_frame_idx.get(int(tr))
    ...
    rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
    ...
    bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
    lick_series[idx[np.unique(bins)]] = 1
```

iii. The methods excerpt in the trajectory highlights cue-relative lick analyses, but the decoder task asked for time-varying binary licking, so the agent implemented a full lick raster rather than the paper’s anticipatory-lick summary.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned to neural data by converting each lick to a trial-relative frame bin and then indexing the session-level `lick_series` with the same trial frame indices `idx` used for `tr_spk`.

ii. ```python
lick_series[idx[np.unique(bins)]] = 1
...
lick = lick_series[idx]
...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
```

iii. The agent’s stated reason was that all outputs had to be synchronized to the per-trial neural frames. The trajectory repeatedly notes that this synchronization is only as exact as the estimated `dt_frame`.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-aligned behavior variable `ft_PosCum` and the scalar `Corridor_Length`.

ii. ```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
...
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
```

iii. The trajectory shows the agent specifically moved to the `ft_PosCum` field after discovering that the reference code also used frame-aligned accumulated position (`ft_PosCum` / `ft_AcumPos`) together with `Corridor_Length`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For each trial the script takes `ft_PosCum[idx]`, wraps it into corridor coordinates with `np.mod(..., corridor_len)`, and treats the resulting position trace as the time-varying location variable.

ii. ```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The notes say the task corridor is effectively 4 m and the reference code uses position modulo corridor length for cue/reward-position analyses, so the converter mirrors that convention.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is normalized by corridor length, multiplied by 4, converted to integers, and clipped to `[0, 3]`, yielding four equal-length 1 m bins when corridor length is 4 m.

ii. ```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. The agent’s Step 5 notes explicitly planned “4 equal 1-m bins” from corridor position to satisfy the decoder specification.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position trace is already frame-aligned (`ft_PosCum`), and the converter selects the same per-trial frame indices `idx` used for neural activity.

ii. ```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The trajectory emphasizes that using `ft_*` fields was the key alignment decision, because those arrays are already sampled on the same frame grid as the neural traces.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_PosCum` plus the estimated frame duration `dt_frame`. The code loads `ft_move` but does not actually use it in the final speed computation.

ii. ```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
ft_move = np.asarray(beh['ft_move'], dtype=float)
...
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The Step 5 notes originally described running speed as a derivative of position over time. The trajectory does not show any later justification for ignoring `VRposTime` or the loaded `ft_move` mask.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Within each trial the code differentiates the wrapped corridor position trace with a first-order difference and divides by `dt_frame`, storing the result as a floating-point speed trace before binning.

ii. ```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
...
cache.append((..., speed.astype(np.float32)))
```

iii. The agent treated this as the simplest time-aligned speed estimate available from the frame-aligned position series. There is no evidence in the notes that it matched this against a reference speed computation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed bins are computed per session, not globally. The converter concatenates all trial speeds within one session, takes the 25th/50th/75th percentiles, and uses `np.digitize` to assign bin IDs 0 to 3.

ii. ```python
all_speeds = []
...
all_speeds.append(speed)
...
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0], dtype=np.float32)
...
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The agent’s earlier planning notes said “4 quantile bins,” but the final code implements those quantiles session by session. There is no later justification for choosing session-local rather than dataset-wide quantiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from the same trial-specific frame indices `idx` used for the neural data, so each speed vector has one value per neural time bin.

ii. ```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
...
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
```

iii. The agent’s justification was again structural alignment through shared frame indices, not fidelity to a reference speed-alignment routine.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles minor problems with defensive truncation and skipping. It truncates all frame-level arrays to the shortest shared length, ignores `NaN` trial labels, skips trials whose metadata are out of range or shorter than 2 frames, fills missing cue timing with `NaN`, treats missing lick arrays as empty, and resolves duplicate behavior-session entries by keeping the richer one.

ii. ```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
...
valid_frame = np.isfinite(ft_tr)
...
if idx.size < 2:
    continue
if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
    continue
...
time_to_sound = ... if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
...
if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
    sessions[sess] = dat
```

iii. The notes document the duplicate-session fix explicitly, and the trajectory shows the agent added the other checks mainly to survive inconsistent lengths and missing values without crashing the conversion.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading very large neural files, concatenating the `spks` blocks, scanning all frames to build per-trial frame groupings, looping over trials to build trial objects, and then writing the enormous pickle. The notes report a full conversion time of about 2177 seconds (36.3 minutes).

ii. ```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
...
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
...
for tr, idx in trial_frame_idx.items():
    ...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. `CONVERSION_NOTES.md` says the first sample runs were very slow and that the optimized full conversion still took ~36 minutes, which the agent attributed mainly to session-scale loading and trialization work.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that repeatedly computes `np.sum(ft_tr_int == tr)` to estimate frame periods could be vectorized with grouped counts. The lick loop over unique trials, the main per-trial assembly loop, and the second pass that bins speed from the cached trial data are also obvious vectorization targets.

ii. ```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
...
for tr in np.unique(lick_tr_i):
    ...
for tr, idx in trial_frame_idx.items():
    ...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The notes explicitly mention one optimization: replacing repeated frame-index scans with a precomputed `trial_frame_idx` mapping. That also implies the remaining loops are still largely Python-level and not fully vectorized.

## 12-c. What processing does the code repeat multiple times?

i. It repeats trialwise scans several times: once to estimate `dt_frame`, once to map lick events into trial bins, once to assemble cached trial tensors, and once more to convert cached speeds into quantile bins. It also repeatedly converts arrays with `np.asarray(...)` at load time.

ii. ```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    ...
for tr in np.unique(lick_tr_i):
    ...
for tr, idx in trial_frame_idx.items():
    ...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    ...
```

iii. The trajectory around the optimization step makes this explicit: the agent sped the script up by precomputing `trial_frame_idx` once because earlier versions were re-scanning frames repeatedly.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several loaded or computed values are unused or only intermediate: `ft_move` and `ft_WallID` are loaded but never used downstream; visual category row 0 is first filled with `-1` and later overwritten entirely; floating-point `speed` is cached only to be converted into bins; and the full wall-name string is carried through `cache` only to be remapped to an integer later.

ii. ```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
...
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
...
out[0, :] = wall_to_id[wall_name]
```

iii. The notes do not dwell on these, but they follow directly from the final code structure: some quantities are prepared during development or caching and then discarded once the categorical output arrays are finalized.
