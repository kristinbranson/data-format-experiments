# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master session index from `beh/Imaging_Exp_info.npy`, deduplicates recordings into 89 unique sessions, then does a two-pass conversion. In pass 1 it loads each `Beh_<exp_type>.npy` once and extracts per-trial behavioral data. In pass 2 it loads per-session retinotopy from `retinotopy/<mouse>_<date>_trans.npz` and spike data from `spk/<mouse>_<date>_<blk>_neural_data.npy`.

ii. 
```python
def load_exp_info():
    return np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
```

```python
B = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type),
            allow_pickle=True).item()
```

```python
ret = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
              allow_pickle=True)
spks = np.load(fn, allow_pickle=True).item()['spks']
```

iii. In `CONVERSION_NOTES.md`, Step 5 and Step 6 justify this as an I/O-efficient two-pass design: behavior files are very large and shared across many sessions, while spikes and retinotopy are session-specific.

## 1-b. How are the data split into subjects?

i. Subjects are defined by mouse name (`mname`). The script carries the mouse name through the session list, then builds `subjects` as the sorted unique mouse names and `subject_idx` as the per-session index into that list.

ii. 
```python
def session_key(db):
    return (db['mname'], db['datexp'], str(db['blk']))
```

```python
subjects = sorted({k[0] for k, _, _, _ in sessions})
...
data['subject_idx'].append(subjects.index(key[0]))
```

iii. The justification in the notes is that `mname` is the explicit subject identifier in the raw dataset, so no derived split is needed.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` recording. The agent deduplicates entries because the same recording can appear under multiple experiment types in `Imaging_Exp_info.npy`, and it keeps the first occurrence.

ii. 
```python
def unique_sessions(exp_info):
    reps = {}
    exp_types = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = session_key(db)
            exp_types.setdefault(key, []).append(exp_type)
            if key not in reps:
                reps[key] = (exp_type, db)
    ordered = sorted(reps.keys())
    return [(k, reps[k][0], reps[k][1], exp_types[k]) for k in ordered]
```

iii. `CONVERSION_NOTES.md` Step 4 says the agent verified duplicate entries across experiment types and concluded they were the same session, so each recording should be converted once.

## 1-d. How are the data split into trials?

i. Trials are defined by `beh['ntrials']`, but within each trial the agent keeps only frames that satisfy `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`. It groups those valid frame indices by trial and stores only trials with at least one such frame.

ii. 
```python
valid = move & corr & np.isfinite(ftr)
vidx = np.where(valid)[0]
vtr = ftr[vidx].astype(int)
order = np.argsort(vtr, kind='stable')
vidx, vtr = vidx[order], vtr[order]
lo = np.searchsorted(vtr, np.arange(ntrials), side='left')
hi = np.searchsorted(vtr, np.arange(ntrials), side='right')
```

```python
for t in range(ntrials):
    frames = vidx[lo[t]:hi[t]]
    if len(frames) == 0:
        continue
```

iii. The notes and trajectory justify this by citing the paper’s “running only” analyses and the reference mask `fr_valid = (ft_move>0) & ft_CorrSpc`, arguing that stationary periods are not useful for the decoder.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have no retained valid frames, if their retained frames extend beyond the neural recording, or if fewer than 5 retained frames remain after filtering. The agent does not use the reference solution’s global 99th-percentile long-trial filter.

ii. 
```python
if len(frames) == 0:
    continue
```

```python
frames = tr['frames'][tr['frames'] < nfr_spk]
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
```

```python
MIN_FRAMES_PER_TRIAL = 5
```

iii. In `CONVERSION_NOTES.md` Step 5 the agent says trials with fewer than 5 retained running frames are “too short to be informative / truncated at the end of the recording.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the per-session `spks` arrays in `spk/<session>_neural_data.npy`, and brain-region labels come from `iarea` in the session’s retinotopy file.

ii. 
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
```

```python
ret = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
              allow_pickle=True)
iarea = ret['iarea']
```

iii. The notes say these are the same primary variables used by the reference code: deconvolved traces plus retinotopic area labels.

## 2-b. How is the `neural` data processed?

i. The agent concatenates plane-wise spike arrays conceptually but only materializes selected neuron rows and selected frames. It stores the retained trial matrices as contiguous `float32` arrays. It does not apply dF/F, deconvolution, z-scoring, or temporal rebinning, but it does apply neuron subsampling and running-frame selection before trial assembly.

ii. 
```python
for plane in spks:
    n = plane.shape[0]
    sel = rows[(rows >= offset) & (rows < offset + n)] - offset
    if len(sel):
        out[filled:filled + len(sel)] = plane[sel][:, frames]
```

```python
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. `CONVERSION_NOTES.md` Step 5 says the traces are already suite2p deconvolved and should not be further normalized, while neuron subsampling was added for tractability because the full dataset was considered too large.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered to those whose `iarea` maps to one of four visual areas (`V1`, `mHV`, `lHV`, `aHV`), then further subsampled to at most 2,000 neurons per session with proportional stratification across areas.

ii. 
```python
if total > MAX_NEURONS:
    alloc = np.floor(counts / total * MAX_NEURONS).astype(int)
    ...
for r in range(len(BRAIN_REGIONS)):
    if alloc[r] <= 0:
        continue
    pick = rng.choice(per_region[r], size=alloc[r], replace=False)
```

```python
MAX_NEURONS = 2000
```

iii. The agent’s notes justify dropping unlabeled neurons because the reference area analyses exclude them, and justify the 2,000-neuron cap on memory and decoder-size grounds.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The stated alignment event is trial start / corridor entry (`StartFr`), but the actual retained neural samples are only the running frames inside the corridor, so each trial is aligned to corridor entry while omitting stationary frames between entry and exit.

ii. 
```python
# alignment     : trial start = corridor entry (beh['StartFr'])
```

```python
valid = move & corr & np.isfinite(ftr)
...
frames = vidx[lo[t]:hi[t]]
```

iii. The justification in the notes is that the paper’s analyses use running-only timepoints and that the decoder can still use true elapsed-time inputs even when some frames are dropped.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal bin is the imaging frame interval, about 315 ms. No temporal rebinning or resampling is applied.

ii. 
```python
dt=float(np.median(np.diff(t_frame)))
```

```python
time_bin_size=float(np.median(dts) * 1000.0),
frame_rate_hz=float(1.0 / np.median(dts)),
```

iii. The notes say the behavior is already on the imaging-frame clock, so preserving the native frame grid gives exact alignment.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and the per-frame timestamps `ft`.

ii. 
```python
t_frame = beh['ft'] * 86400.0
t_cue = np.interp(beh['SoundFr'], frame_idx, t_frame)
```

iii. The notes say cue timing should come from the true cue frame rather than from delayed reward variables.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The fractional cue-frame index is interpolated onto the frame-time axis, then each retained frame gets `cue_time - frame_time`, in seconds. Values are positive before the cue and negative after it.

ii. 
```python
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32),
```

iii. The agent explicitly documents this sign convention in `metadata['input_units']` and in the notes.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same retained frame indices as the neural trial matrix.

ii. 
```python
frames = vidx[lo[t]:hi[t]]
...
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32),
```

iii. The justification is that all behavioral streams are already on the neural frame clock, so using the same `frames` array aligns them directly.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date encoded in `datexp`, relative to each mouse’s first imaging-session date.

ii. 
```python
def first_session_dates(all_sessions):
    first_date = {}
    for key, _, _, _ in all_sessions:
        m, date = key[0], key[1]
        d = datetime.date(*map(int, date.split('_')))
        if m not in first_date or d < first_date[m]:
            first_date[m] = d
```

```python
day = (datetime.date(*map(int, key[1].split('_'))) - first_date[key[0]]).days
```

iii. `CONVERSION_NOTES.md` Step 5 says the raw `days` field is incomplete, so the agent chose elapsed calendar days since the mouse’s first imaging session as a continuous proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script finds the first session date for each mouse over the whole dataset, computes the day difference for each session, stores that scalar, and broadcasts it over every retained frame in each trial.

ii. 
```python
results[i] = process_behaviour(beh, day)
results[i]['day_of_training'] = day
```

```python
inp[1] = tr['day_of_training']
```

iii. The trajectory shows the agent deliberately changed this to be computed over all 89 sessions so `--sample` and `--full` would agree.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and the frame timestamps `ft`.

ii. 
```python
t_frame = beh['ft'] * 86400.0
t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
```

iii. The notes identify corridor entry (`StartFr`) as the alignment event required by the task.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `StartFr` is interpolated onto the frame-time axis, then each retained frame gets `frame_time - start_time`, in seconds.

ii. 
```python
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
```

iii. The notes emphasize that these are true elapsed times, even though some stationary frames are not retained.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is indexed on the same retained frames used for the neural trial matrix.

ii. 
```python
frames = vidx[lo[t]:hi[t]]
...
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
```

iii. The justification is again direct alignment on the common frame clock.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`, the per-trial rewarded-corridor flag.

ii. 
```python
is_rew = np.asarray(beh['isRew']).astype(bool)
...
reward_available=np.float32(1.0 if is_rew[t] else 0.0),
```

iii. The notes state that `isRew` directly encodes whether a trial is in the rewarded corridor.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is converted to `0.0` or `1.0` per trial and broadcast over time within the retained frames of that trial.

ii. 
```python
reward_available=np.float32(1.0 if is_rew[t] else 0.0),
```

```python
inp[3] = tr['reward_available']
```

iii. The notes justify keeping unrewarded sessions and simply labeling them with reward availability 0.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. 
```python
wall = np.asarray(beh['WallName'])
stim_cat = np.array([STIM_CATEGORIES.index(texture_family(w)) for w in wall])
```

iii. The notes say `WallName` is the stable per-trial texture identity, while some alternate stimulus fields are not reliable across swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent maps each wall name to a texture family by stripping digits and swap suffixes, then remaps `wood` to `brick`. The resulting four categories are `circle`, `leaf`, `rock`, and `brick`, and the per-trial category is broadcast over the trial’s retained frames.

ii. 
```python
def texture_family(wall_name):
    base = wall_name.split('_')[0]
    base = ''.join(ch for ch in base if not ch.isdigit())
    if base == 'wood':
        base = 'brick'
    return base
```

```python
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'brick']
...
out[0] = tr['stim_cat']
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly justifies using texture family and renaming `wood` to `brick` because the paper describes the fourth texture as brick.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`.

ii. 
```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
lick_bin[lf] = True
```

iii. The notes treat `LickFr` as already expressed on the neural frame grid.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent floors fractional lick-frame indices to integer frames, removes non-finite and out-of-range licks, then marks a frame as 1 if at least one lick fell in that frame.

ii. 
```python
lick_bin = np.zeros(nfr_beh, dtype=bool)
lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
lick_bin[lf] = True
```

```python
lick=lick_bin[frames].astype(np.int64),
```

iii. The notes and comments justify this as matching the reference cue/lick alignment utilities, which bin licks by neural frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is indexed on the same retained frame indices as the neural trial matrix.

ii. 
```python
lick=lick_bin[frames].astype(np.int64),
...
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. The justification is direct frame-index alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. 
```python
pos = beh['ft_Pos'][:nfr_beh].astype(float)
```

iii. The notes identify `ft_Pos` as the within-trial position in decimeters on the neural frame clock.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw position values are carried through behavior extraction, then at assembly the script converts them into 4 one-meter bins across the 4-meter texture corridor.

ii. 
```python
pos=pos[frames],
```

```python
out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                 0, N_POS_BINS - 1).astype(np.int64)
```

iii. The notes justify 4 equal 1 m bins because `Texture_Length` is 40 dm and the decoder spec asks for 4 equal-length bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided by 10 dm (equivalently by `TEXTURE_LENGTH_DM / 4`) and clipped into category indices `0..3`.

ii. 
```python
TEXTURE_LENGTH_DM = 40.0
N_POS_BINS = 4
...
out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                 0, N_POS_BINS - 1).astype(np.int64)
```

iii. This is the categorical discretization the agent documents in Step 5 of the notes.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken on the same retained frames used for the neural trial matrix.

ii. 
```python
pos=pos[frames],
...
frames = tr['frames'][tr['frames'] < nfr_spk]
```

iii. The notes justify this by the common frame clock.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
speed = beh['ft_RunSpeed'][:nfr_beh].astype(float)
```

iii. The notes treat this as the direct per-frame running-speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent carries the continuous per-frame speed values through trial extraction, then computes global speed quartile thresholds across all retained frames in all processed sessions and bins trial frames with those thresholds.

ii. 
```python
speed=speed[frames],
```

```python
speeds = np.concatenate([tr['speed'] for res in beh_results for tr in res['trials']])
speed_edges = np.percentile(speeds, [25, 50, 75])
```

```python
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 says global quartiles were chosen so the four bins each hold 25% of the data across the whole dataset and so `--sample` and `--full` use the same thresholds.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global percentile edges at the 25th, 50th, and 75th percentiles of retained running speeds are computed, and `np.digitize` turns each frame into one of four categories.

ii. 
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
...
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
```

iii. The notes say the target is 25% of data per bin globally, not per session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is read on the same retained frame indices as the neural trial matrix.

ii. 
```python
speed=speed[frames],
...
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. The justification is the shared neural-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script drops non-finite `ft_trInd` values from trial assignment, floors and filters non-finite or out-of-range lick frames, truncates trial frames to the spike recording length, and drops trials left with too few retained frames. It does not appear to impute missing values.

ii. 
```python
valid = move & corr & np.isfinite(ftr)
```

```python
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
```

```python
frames = tr['frames'][tr['frames'] < nfr_spk]
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
```

iii. The notes frame these as edge-case handling for behavior arrays longer than neural recordings, missing labels, and truncated end-of-recording trials.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive part is the neural pass: loading the per-session spike files and slicing out selected neurons and frames. The behavior pass is explicitly designed to be cheaper by loading each behavior file once.

ii. 
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
```

```python
if nproc == 1:
    it = map(convert_session, jobs)
else:
    pool = mp.Pool(nproc, maxtasksperchild=1)
    it = pool.imap_unordered(convert_session, jobs)
```

iii. The notes repeatedly emphasize that spike I/O dominates runtime because the spike directory is hundreds of GB.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized the main trial-grouping step with `argsort` and `searchsorted`, but there are still small Python loops over trials and over frame-index remapping that could be vectorized further, especially the per-trial `frame_pos` lookup and repeated array assembly.

ii. 
```python
for t in range(ntrials):
    frames = vidx[lo[t]:hi[t]]
    ...
```

```python
cols = np.array([frame_pos[f] for f in frames], dtype=np.int64)
```

```python
for tr in trials:
    T = len(tr['frames'])
    inp = np.empty((4, T), dtype=np.float32)
    ...
```

iii. The notes show the agent intentionally vectorized the larger frame-grouping problem, leaving mostly bookkeeping loops that are minor compared with spike I/O.

## 12-c. What processing does the code repeat multiple times?

i. The code carries raw `pos` and `speed` arrays through the behavior pass and only bins them later during assembly, so position and speed discretization are deferred and then repeated across trials. It also iterates over trials once to build behavior records and again to assemble final input/output arrays.

ii. 
```python
pos=pos[frames],
speed=speed[frames],
```

```python
out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                 0, N_POS_BINS - 1).astype(np.int64)
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
```

iii. The notes justify the two-pass structure for memory and I/O reasons, but this organization does defer some work instead of finalizing it once in the first pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes extra metadata (`cohort`, `exp_types`, `reward_mode`, detailed `session_info`) and supports optional plotting. It also stores raw per-frame position and speed in intermediate trial dicts even though the downstream decoder only uses the discretized categories.

ii. 
```python
def cohort_of(db, exp_types):
    ...
```

```python
session_info.append(dict(session_id='_'.join(key), mouse=key[0], date=key[1], blk=key[2],
                         exp_type=exp_type, exp_types=ets, cohort=cohort_of(db, ets),
                         day_of_training=beh_results[isess]['day_of_training'],
                         reward_session=bool(info['n_rew_trials'] > 0),
                         reward_mode=str(db.get('rewType', '')),
                         ...))
```

```python
def plot_processing(session_id, neural, trials, speed_edges, info, outfile):
    ...
```

iii. The agent’s notes frame these additions as diagnostics and documentation rather than decoder necessities.
