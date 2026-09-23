# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `beh/Imaging_Exp_info.npy`, builds unique session records keyed by `mname_datexp_blk`, then makes a single pass over every `Beh_*.npy` file to collect one behavior dict per session and merge stimulus labels across duplicate behavior keys. Neural data and retinotopy are then loaded per session from `spk/<session>_neural_data.npy` and `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
def load_exp_info():
    exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    sessions = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = session_key(db['mname'], db['datexp'], db['blk'])
            rec = sessions.setdefault(key, {'mname': db['mname'], 'datexp': db['datexp'],
                                            'blk': db['blk'], 'exp_types': [],
                                            'exptype': None, 'rewType': db.get('rewType', None)})
            rec['exp_types'].append(exp_type)
```

```python
for f in sorted(glob.glob(os.path.join(BEH_DIR, 'Beh_*.npy'))):
    B = np.load(f, allow_pickle=True).item()
    for k, beh in B.items():
        base = '_'.join(k.split('_')[:5])
        if base not in sessions:
            continue
        if base not in beh_by_session:
            beh_by_session[base] = beh
```

```python
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (info['mname'], info['datexp'])),
              allow_pickle=True)
spk, n_spk_neurons = load_spk_rows(key, rows)
```

iii. In `CONVERSION_NOTES.md`, the AI says it wanted one pass over the 23 behavior files, 89 unique sessions from the experiment index, and per-session spike loading to avoid materializing the full 405 GB spike dataset at once.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`) from the experiment index. The final `subjects` list is the sorted unique set of mouse names across retained sessions, and `subject_idx` maps each session to that list.

ii.
```python
key = session_key(db['mname'], db['datexp'], db['blk'])
rec = sessions.setdefault(key, {'mname': db['mname'], 'datexp': db['datexp'],
                                'blk': db['blk'], 'exp_types': [],
                                'exptype': None, 'rewType': db.get('rewType', None)})
```

```python
subjects = sorted(set(r['mname'] for r in results))
data = {
    'subjects': subjects,
    'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64),
```

iii. The notes explicitly state that the 89 sessions are grouped into 19 mice using the mouse identifier already present in `Imaging_Exp_info.npy`.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` recording. If the same recording appears under multiple experiment types or duplicate `_swap1`/`_swap2` behavior keys, it is merged into one session.

ii.
```python
def session_key(mname, datexp, blk):
    return '%s_%s_%s' % (mname, datexp, blk)
```

```python
for exp_type, dbs in exp_info.items():
    for db in dbs:
        key = session_key(db['mname'], db['datexp'], db['blk'])
        rec = sessions.setdefault(key, {'mname': db['mname'], 'datexp': db['datexp'],
                                        'blk': db['blk'], 'exp_types': [],
                                        'exptype': None, 'rewType': db.get('rewType', None)})
        rec['exp_types'].append(exp_type)
```

```python
base = '_'.join(k.split('_')[:5])
if base not in beh_by_session:
    beh_by_session[base] = beh
```

iii. The AI’s notes repeatedly justify this as necessary because the released behavior files contain 99 keys but only 89 unique recordings, with duplicate keys used to provide complementary stimulus labels for swap sessions.

## 1-d. How are the data split into trials?

i. Trials are not taken as all corridor frames. Instead, the AI defines a valid-frame mask inside each session, keeps only frames that are both in the textured corridor and moving, sorts those valid frames by `ft_trInd`, and groups each run of equal trial index into one trial. Each stored trial therefore contains only included frames from that trial.

ii.
```python
ft_trind = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
ft_corr = np.asarray(beh['ft_CorrSpc'], dtype=bool)[:nfr]
ft_move = np.asarray(beh['ft_move'], dtype=float)[:nfr] > 0

valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))
```

```python
idx_valid = np.where(valid)[0]
tr_valid = ft_trind[idx_valid].astype(int)
order = np.argsort(tr_valid, kind='stable')
tr_sorted = tr_valid[order]
idx_sorted = idx_valid[order]
bounds = np.where(np.diff(tr_sorted) != 0)[0] + 1
for fr, tt in zip(np.split(idx_sorted, bounds), np.split(tr_sorted, bounds)):
    tr = int(tt[0])
```

iii. In the notes and trajectory, the AI explicitly justifies this with the reference paper’s analysis mask `fr_valid = (ft_move>0) & ft_CorrSpc`, arguing that only running timepoints inside the texture should be kept.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with fewer than 5 valid frames, trials whose wall name has no merged stimulus label, and trials with non-finite cue or start times. After session processing, sessions with fewer than 2 usable trials are dropped from the final dataset.

ii.
```python
if len(fr) < MIN_FRAMES_PER_TRIAL:
    n_drop_short += 1
    continue
if wname not in smap:
    n_drop_stim += 1
    continue
if not np.isfinite(t_sound[tr]) or not np.isfinite(t_start[tr]):
    n_drop_cue += 1
    continue
```

```python
results = [r for r in results if len(r['trials']) >= 2]
```

iii. The notes justify these filters as removing unusable trials for decoding: too-short trials, unlabeled stimuli, and missing cue/start timing. The AI also notes that `circle3` trials were dropped because it could not assign them a canonical `stim_id`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the deconvolved spike-like traces in `spks` from each session’s spike file, plus `iarea` from the retinotopy file for anatomical selection and region indexing.

ii.
```python
planes = np.load(path, allow_pickle=True).item()['spks']
```

```python
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (info['mname'], info['datexp'])),
              allow_pickle=True)
iarea = ret['iarea']
```

iii. The notes say the AI followed `utils.load_spk` and `utils.load_retino` from the reference code and treated the provided traces as already deconvolved neural activity.

## 2-b. How is the `neural` data processed?

i. The AI concatenates imaging planes conceptually via row offsets, selects up to 1,000 neurons per session from visual cortex, truncates to the imaged frames, z-scores each neuron over included frames, removes zero-variance neurons, and stores each trial as a float32 neuron-by-time matrix.

ii.
```python
rows, area_idx = select_neurons(iarea, N_NEURONS_PER_SESSION, rng)
spk, n_spk_neurons = load_spk_rows(key, rows)
```

```python
mu = spk[:, valid].mean(axis=1, keepdims=True)
sd = spk[:, valid].std(axis=1, keepdims=True)
keep_neu = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
spk = (spk - mu) / np.where(sd > 0, sd, 1.0)
spk = spk[keep_neu]
```

```python
neural_s.append(t['neural'].astype(np.float32))
```

iii. The notes justify subsampling as a memory/I/O constraint on a 405 GB raw dataset, and justify z-scoring by reference analyses that z-scored traces before population analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only neurons whose `iarea` maps into V1, mHV, lHV, or aHV, subsamples to at most 1,000 neurons per session, and then drops neurons with zero or non-finite standard deviation over the included frames.

ii.
```python
AREA_OF_IAREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}
```

```python
valid = np.where(area_idx >= 0)[0]
if len(valid) <= n_target:
    return valid, area_idx[valid]
```

```python
keep_neu = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
spk = spk[keep_neu]
area_idx = area_idx[keep_neu]
```

iii. In the notes, the AI says it copied the reference anatomical filter, then added proportional subsampling and a zero-variance guard for practicality and numerical stability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to corridor entry, but the stored timepoints are only the valid corridor-and-running frames from each trial. The trial start event is described as `Trial_start_time`/`StartFr`.

ii.
```python
"""Convert Zhong et al. 2025 VR-corridor imaging data to decoder format.
...
  alignment   : corridor entry (beh['Trial_start_time'] / beh['StartFr'])
"""
```

```python
'temporal_alignment_event': 'corridor entry (trial start; beh["Trial_start_time"]/beh["StartFr"])',
```

```python
trials_out.append({
    'frames': fr,
    'neural': np.ascontiguousarray(spk[:, fr]),
```

iii. The notes explicitly say the alignment event is corridor entry and that the valid-frame mask from the reference should define which timepoints remain inside each aligned trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging-frame resolution. It estimates frame interval from the median difference in `ft`, stores the mean session `dt` in metadata, and does not rebin the time series.

ii.
```python
dt': float(np.median(np.diff(ft)) * SEC_PER_DAY),
```

```python
dt_ms = float(np.mean([r['dt'] for r in results]) * 1000.0)
...
'time_bin_size': dt_ms,
```

iii. The notes state that the imaging frame at about 3.17 Hz is the natural time bin and that no extra temporal resampling was needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and per-frame timestamps `ft`.

ii.
```python
ft = np.asarray(beh['ft'], dtype=float)[:nfr]
...
t_sound = np.asarray(beh['SoundTime'], dtype=float)
```

```python
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```

iii. The notes say the AI preferred the explicit cue time variable and treated all behavioral time series as living on the same frame-time clock.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame of a trial, the AI subtracts the frame time from the trial’s cue time and converts the MATLAB-day units to seconds. It also derives an extra binary `sound_cue_onset` input marking the first frame at or after the cue.

ii.
```python
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```

```python
cue_onset = np.zeros(T, dtype=np.float32)
after = np.where(t['time_to_cue'] <= 0)[0]
if after.size:
    cue_onset[after[0]] = 1.0
```

iii. The notes justify the signed time difference as the requested continuous decoder input and say the extra cue-onset flag was added to give the decoder a scale-free cue timing signal.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same retained frame indices as the neural trial matrix.

ii.
```python
trials_out.append({
    'frames': fr,
    'neural': np.ascontiguousarray(spk[:, fr]),
    'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```

iii. The notes repeatedly state that all streams were aligned on the shared frame clock and then cut with the same per-trial frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session’s `datexp` string, parsed as a calendar date, together with the mouse identity `mname`.

ii.
```python
def parse_date(datexp):
    y, m, d = datexp.split('_')
    return date(int(y), int(m), int(d))
```

```python
for r in results:
    d = parse_date(r['datexp'])
    if r['mname'] not in first_day or d < first_day[r['mname']]:
        first_day[r['mname']] = d
```

iii. In the notes, the AI describes this input as “days elapsed since that mouse’s first recording,” not session count.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the earliest recording date for each mouse, takes the calendar-day difference between the current session and that first day, and broadcasts that scalar across all timepoints in every trial of the session.

ii.
```python
day = (parse_date(r['datexp']) - first_day[r['mname']]).days
```

```python
inp = np.stack([
    t['time_to_cue'].astype(np.float32),
    cue_onset,
    np.full(T, float(day), dtype=np.float32),
    t['time_since_start'].astype(np.float32),
    np.full(T, float(t['is_rew']), dtype=np.float32),
])
```

iii. The notes explicitly justify this as elapsed training days rather than recording-session index.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the per-frame timestamps `ft` and the per-trial start timestamps `Trial_start_time`.

ii.
```python
ft = np.asarray(beh['ft'], dtype=float)[:nfr]
...
t_start = np.asarray(beh['Trial_start_time'], dtype=float)
```

```python
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. The notes say trials are aligned to corridor entry and use the behavioral timestamps directly on the frame-time base.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each kept frame, the AI subtracts the trial’s `Trial_start_time` from the frame time and converts from days to seconds.

ii.
```python
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. The notes describe this as the continuous elapsed time since corridor entry for each included frame.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame indices as the neural activity for that trial.

ii.
```python
trials_out.append({
    'frames': fr,
    'neural': np.ascontiguousarray(spk[:, fr]),
    'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. The notes say all trial-wise arrays are sliced with the same per-trial frame list.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`, the per-trial rewarded-corridor flag.

ii.
```python
is_rew = np.asarray(beh['isRew'], dtype=bool)
```

```python
'is_rew': int(is_rew[tr]),
```

iii. The notes identify `isRew` as the direct source variable.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the boolean per-trial flag to `0`/`1` and broadcasts it across all frames of that trial in the decoder input matrix.

ii.
```python
np.full(T, float(t['is_rew']), dtype=np.float32),
```

iii. The notes justify this as the natural per-trial indicator of whether water could be earned in that corridor.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` per trial, but the actual category assignment comes from a session-specific merged mapping built from `UniqWalls` and `stim_id` across all behavior files in which that session appears. The final categories are the seven canonical stimulus IDs in `CANONICAL_STIM`.

ii.
```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
```

```python
sid = np.asarray(beh['stim_id'], dtype=float)
for w, s in zip(list(beh['UniqWalls']), sid):
    if not np.isnan(s):
        stim_map[base][str(w)] = int(s)
```

```python
wname = str(wall[tr])
if wname not in smap:
    n_drop_stim += 1
    continue
...
'stim': smap[wname],
```

iii. The notes justify this as following the reference code’s canonical, role-based `stim_id` labels and merging duplicate `_swap1`/`_swap2` behavior keys so both swapped stimuli can be labeled.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI first merges non-NaN `stim_id` values for each wall name across all behavior entries for the session, then assigns each trial’s `WallName` to a canonical stimulus ID, and finally broadcasts that ID across the frames of the trial. Trials whose wall name never receives a valid label are dropped.

ii.
```python
for w, s in zip(list(beh['UniqWalls']), sid):
    if not np.isnan(s):
        stim_map[base][str(w)] = int(s)
```

```python
if wname not in smap:
    n_drop_stim += 1
    continue
```

```python
outp = np.stack([
    np.full(T, t['stim'], dtype=np.int64),
    t['lick'].astype(np.int64),
    pos_bin.astype(np.int64),
    spd_bin.astype(np.int64),
])
```

iii. The notes explicitly justify the merged `stim_id` map because some stimulus labels are NaN in one behavior file but defined in another, and say the remaining unlabeled `circle3` trials were therefore removed.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`, the frame indices of licks.

ii.
```python
lick_fr = np.atleast_1d(np.asarray(beh['LickFr'], dtype=float))
```

iii. The notes identify `LickFr` as the raw source and emphasize that it already lives on the neural frame clock.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI converts lick times to a framewise binary vector by flooring finite lick frame indices to integers, dropping out-of-range values, and setting those frame bins to 1 if any lick lands there.

ii.
```python
lick_bin = np.zeros(nfr, dtype=np.int64)
if lick_fr.size:
    li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
    li = li[(li >= 0) & (li < nfr)]
    lick_bin[li] = 1
```

iii. The notes justify this as the appropriate categorical framewise representation for decoder output, including sessions with no licks.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is built on the session’s frame grid and then sliced with the same frame indices used for each trial’s neural data.

ii.
```python
'lick': lick_bin[fr],
```

```python
outp = np.stack([
    np.full(T, t['stim'], dtype=np.int64),
    t['lick'].astype(np.int64),
    pos_bin.astype(np.int64),
    spd_bin.astype(np.int64),
])
```

iii. The notes say all output streams are aligned by neural frame index and then cut per trial with the same `fr` array.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, the per-frame corridor position in decimeters.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
...
'pos_dm': ft_pos[fr],
```

iii. The notes explicitly state that `ft_Pos` is in decimeters over a 6 m cycle and that the textured corridor occupies 0-40 dm.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI carries forward each retained frame’s position in decimeters and later discretizes it into 1 m bins during assembly.

ii.
```python
'pos_dm': ft_pos[fr],
```

```python
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
```

iii. The notes justify this as matching the decoder task’s 4 equal-length bins while respecting the reference data’s native decimeter position variable.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It uses fixed decimeter thresholds at 10, 20, and 30 dm, clipped into four categories `0..3`.

ii.
```python
POS_BIN_EDGES_DM = np.array([10.0, 20.0, 30.0])
```

```python
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
```

iii. The notes explicitly justify these as the task-required four 1 m bins across the 4 m textured corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same retained frame indices as the neural data for each trial.

ii.
```python
'pos_dm': ft_pos[fr],
```

```python
outp = np.stack([
    np.full(T, t['stim'], dtype=np.int64),
    t['lick'].astype(np.int64),
    pos_bin.astype(np.int64),
    spd_bin.astype(np.int64),
])
```

iii. The notes state that `ft_Pos` is already frame-aligned and then cut per trial using the same retained-frame array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`, the per-frame running speed.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
...
'speed': ft_speed[fr],
```

iii. The notes identify `ft_RunSpeed` as the direct source variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI collects all retained running-speed samples across all retained sessions and trials, computes global 25th/50th/75th percentile edges, and digitizes each trial’s retained speed values against those global thresholds.

ii.
```python
all_speed = np.concatenate([np.concatenate([t['speed'] for t in r['trials']])
                            for r in results])
speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
```

```python
spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

iii. The notes justify this as the most literal way to satisfy the decoder-task instruction that the four speed bins each correspond to 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded with dataset-wide quartile edges, producing categories `0..3`.

ii.
```python
speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
```

```python
spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

iii. The notes explicitly discuss these as global quartile bins and note that this causes session-to-session bin imbalance by design.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is taken on the same retained frame indices as the neural data for each trial.

ii.
```python
'speed': ft_speed[fr],
```

```python
outp = np.stack([
    np.full(T, t['stim'], dtype=np.int64),
    t['lick'].astype(np.int64),
    pos_bin.astype(np.int64),
    spd_bin.astype(np.int64),
])
```

iii. The notes say speed is already frame-aligned and is sliced with the same per-trial frame list.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavioral streams to the available neural frame count, excludes non-finite trial/frame annotations from the valid-frame mask, drops out-of-range lick frames, merges stimulus labels across duplicate behavior entries to recover missing labels where possible, drops trials with missing cue/start time or missing stimulus label, and removes zero-variance neurons before z-scoring.

ii.
```python
nfr = min(spk.shape[1], len(beh['ft']))
spk = spk[:, :nfr]
```

```python
valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))
```

```python
li = li[(li >= 0) & (li < nfr)]
```

```python
if wname not in smap:
    n_drop_stim += 1
    continue
if not np.isfinite(t_sound[tr]) or not np.isfinite(t_start[tr]):
    n_drop_cue += 1
    continue
```

iii. The notes emphasize that behavior can be one frame longer than imaging, some `stim_id` values are NaN in one behavior file but recoverable from another, and the conversion should fail gracefully on missing or invalid trial data rather than keep malformed examples.

## 12-a. What are the most time-consuming steps of the code?

i. The time-consuming part is per-session spike loading and session processing over the large raw spike files, not behavior loading. The notes report about 65 s of the 76 s full run spent in session processing for all 89 sessions.

ii.
```python
def load_spk_rows(key, rows):
    path = os.path.join(SPK_DIR, '%s_neural_data.npy' % key)
    planes = np.load(path, allow_pickle=True).item()['spks']
```

```python
print('\nProcessing %d sessions with %d workers ...' % (len(keys), args.nworkers), flush=True)
...
results = pool.map(process_session, keys, chunksize=1)
```

iii. In both the notes and trajectory, the AI repeatedly frames the spike files as the dominant cost and designs the code around reducing spike I/O and memory pressure.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized trial segmentation compared with a trial-by-trial scan, but some loops remain: the `iarea`-to-region assignment loop, the per-trial Python loop that assembles input/output arrays, and repeated `subjects.index(...)` lookups when building `subject_idx`.

ii.
```python
for ia, a in AREA_OF_IAREA.items():
    area_idx[iarea == ia] = a
```

```python
for r in results:
    day = (parse_date(r['datexp']) - first_day[r['mname']]).days
    neural_s, input_s, output_s = [], [], []
    for t in r['trials']:
        ...
        inp = np.stack([...])
        outp = np.stack([...])
```

```python
'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64),
```

iii. The AI’s notes say trial segmentation was intentionally vectorized for speed; they do not give further vectorization rationale beyond that, so the remaining vectorization opportunities are inferred directly from the code.

## 12-c. What processing does the code repeat multiple times?

i. The code recomputes discretizations for diagnostics after already computing the core arrays: `show_processing()` recalculates position bins and speed bins from stored raw per-trial values, and cue onset is derived from `time_to_cue` after `time_to_cue` itself has already been computed. It also repeatedly reparses session dates while assembling outputs.

ii.
```python
cue_onset = np.zeros(T, dtype=np.float32)
after = np.where(t['time_to_cue'] <= 0)[0]
if after.size:
    cue_onset[after[0]] = 1.0
```

```python
pb = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
sb = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

```python
day = (parse_date(r['datexp']) - first_day[r['mname']]).days
```

iii. The notes do not call these out as problems; this is mostly repeated work for metadata/diagnostics and minor assembly overhead rather than the main bottleneck.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion includes extra processing and diagnostics beyond the requested core dataset: it adds an extra `sound_cue_onset` decoder input not requested in the task, builds rich intermediate trial dictionaries (`frames`, `wall`, `trial_index`, raw `pos_dm`, raw `speed`) that are then collapsed into final arrays, and optionally recomputes bins again in `show_processing()` purely for plots.

ii.
```python
'input_names': ['time_to_sound_cue', 'sound_cue_onset', 'day_of_training',
                'time_since_trial_start', 'reward_availability'],
```

```python
trials_out.append({
    'frames': fr,
    'neural': np.ascontiguousarray(spk[:, fr]),
    'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
    'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
    'pos_dm': ft_pos[fr],
    'speed': ft_speed[fr],
    'lick': lick_bin[fr],
    'stim': smap[wname],
    'wall': wname,
    'is_rew': int(is_rew[tr]),
    'trial_index': tr,
})
```

```python
def show_processing(result, speed_edges):
    ...
    pb = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
    sb = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

iii. The notes justify some of this as helping decoder performance and auditing the conversion, but these additions are not required to reproduce the human reference output structure.
