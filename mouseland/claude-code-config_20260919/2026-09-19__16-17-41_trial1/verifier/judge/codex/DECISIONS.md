# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds a session index from `beh/Imaging_Exp_info.npy`, deduplicates repeated experiment-type entries down to physical recordings keyed by `mname_datexp_blk`, loads each `Beh_<exp_type>.npy` once while keeping only selected behavior fields, merges `stim_id` information across experiment types, and then loads spikes and retinotopy per session from `spk/` and `retinotopy/`.

ii.
```python
def build_session_index():
    exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=1).item()
    sessions = {}
    for exp_type, db in exp_info.items():
        for ndb in db:
            key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
            beh_key = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
            rec = sessions.setdefault(key, dict(
                key=key, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'],
                exp_types=[], beh_keys=[], rewtype=set(), exptype=set(), stim_id_entries=[]))
```
```python
def load_behaviour(sessions, verbose=True):
    beh = {}
    stim_map = {k: {} for k in sessions}
    for exp_type in sorted(set(sum([r['exp_types'] for r in sessions.values()], []))):
        B = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=1).item()
```
```python
fn = os.path.join(ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
spks = np.load(fn, allow_pickle=True).item()['spks']

fn = os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
with np.load(fn, allow_pickle=True) as d:
    return d['iarea']
```

iii. In `CONVERSION_NOTES.md` Step 1/2/4/5, the AI says the dataset is organized around the three raw folders, that `Imaging_Exp_info.npy` overcounts recordings because the same physical session appears under multiple experiment types, and that behavior should be read once per file while spikes and retinotopy are session-specific.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname`. Each deduplicated session record stores the mouse name, and during assembly the script creates `subjects` from first encounter order and records a `subject_idx` for each session.

ii.
```python
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
rec = sessions.setdefault(key, dict(
    key=key, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'],
```
```python
subjects, subject_idx, session_info = [], [], []
...
if r['mname'] not in subjects:
    subjects.append(r['mname'])
subject_idx.append(subjects.index(r['mname']))
```

iii. The notes repeatedly describe `mname` as the mouse identifier and report 19 unique mice, so the split into subjects is taken directly from the raw index rather than inferred.

## 1-c. How are the data split into sessions?

i. A session is one unique physical recording identified by `mname`, `datexp`, and `blk`, joined into `mname_datexp_blk`. Multiple entries from different experiment types are merged into one session record.

ii.
```python
for exp_type, db in exp_info.items():
    for ndb in db:
        key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        beh_key = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
        rec = sessions.setdefault(key, dict(
            key=key, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'],
            exp_types=[], beh_keys=[], rewtype=set(), exptype=set(), stim_id_entries=[]))
        rec['exp_types'].append(exp_type)
        rec['beh_keys'].append(beh_key)
```

iii. In Step 4 the AI justifies this by noting that `exp_info` contains 141 entries but only 89 unique physical recordings, exactly matching the paper’s “89 recordings in 19 mice”.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from `ft_trInd`, but only for frames that also satisfy `ft_CorrSpc`, `ft_move > 0`, and finite trial index. The code groups valid frames by trial id with a sort/count pass and keeps only those frames per trial. As a result, each stored trial is a variable-length subset of running frames inside the textured corridor, not the full contiguous corridor traversal.

ii.
```python
ft = b['ft'][:nfr]
tr = b['ft_trInd'][:nfr]
corr = b['ft_CorrSpc'][:nfr].astype(bool)
move = b['ft_move'][:nfr] > 0
...
valid = corr & move & np.isfinite(tr)
tr_int = np.where(valid, np.nan_to_num(tr, nan=-1), -1).astype(np.int64)
tr_int[(tr_int < 0) | (tr_int >= ntrials)] = -1
```
```python
order = np.argsort(tr_int, kind='stable')
counts = np.bincount(tr_int[tr_int >= 0], minlength=ntrials)
ordered = order[int(np.sum(tr_int < 0)):]
...
fi = ordered[ptr:ptr + n]
fi = np.sort(fi)
```

iii. The script header and Step 5 of the notes say this is meant to mirror the reference’s `fr_valid = ft_CorrSpc & (ft_move>0)` and to keep only timepoints inside the 4 m textured corridor while the mouse is running.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has zero valid frames after truncation and frame masking, or if its wall name cannot be mapped to a canonical stimulus id (`stim_of_trial[t] < 0`, mainly `circle3`). There is no separate long-trial outlier filter.

ii.
```python
stim_of_trial = np.full(ntrials, -1, dtype=np.int64)
smap = rec['stim_map']
for t in range(ntrials):
    stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)
...
for t in range(ntrials):
    n = counts[t]
    if n == 0:
        continue
    fi = ordered[ptr:ptr + n]
    ptr += n
    if stim_of_trial[t] < 0:
        continue
```

iii. The notes justify the stimulus-based drop by saying the reference never analyzes walls with missing canonical `stim_id`, and they treat the running-only frame mask itself as the main curation against pause-filled trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays come from the per-plane `spks` arrays in each `*_neural_data.npy` spike file, combined with `iarea` from the matching retinotopy file to filter and label neurons by visual region.

ii.
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
...
with np.load(fn, allow_pickle=True) as d:
    return d['iarea']
```

iii. The notes say the released neural data are already Suite2p deconvolved traces, and `iarea` is the only neuron-level curation signal used from raw metadata.

## 2-b. How is the `neural` data processed?

i. The AI keeps only neurons in the selected visual areas, gathers the chosen frame indices for all kept trials, optionally z-scores each neuron over all frames of the full session, and then splits the gathered neural matrix back into per-trial arrays. It stores these as contiguous `float32` matrices rather than raw per-plane arrays.

ii.
```python
def extract_neural(blocks, keep_mask, frame_idx, zscore=True, nthreads=16, chunk=4096):
    out = np.empty((int(keep_mask.sum()), T), dtype=np.float32)
```
```python
g = sub[:, frame_idx][km]
if zscore:
    nf = sub.shape[1]
    s1 = sub.sum(axis=1, dtype=np.float64)
    s2 = np.einsum('ij,ij->i', sub, sub, dtype=np.float64)
    mu = s1 / nf
    sd = np.sqrt(np.maximum(s2 / nf - mu * mu, 0.0))
    sd[sd == 0] = 1.0
    g -= mu[km, None].astype(np.float32)
    g /= sd[km, None].astype(np.float32)
```
```python
trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. The notes defend z-scoring by citing `stats.zscore(spk, axis=1)` in the reference population-analysis code and by arguing that deconvolved trace magnitudes vary strongly across neurons and sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neuron filter keeps neurons whose `iarea` maps to one of four visual cortical region groups: V1, mHV, lHV, or aHV. Anything else is dropped.

ii.
```python
reg = np.full(len(iarea), -1, dtype=np.int64)
reg[iarea == 8] = 0
reg[np.isin(iarea, [0, 1, 2, 9])] = 1
reg[np.isin(iarea, [5, 6])] = 2
reg[np.isin(iarea, [3, 4])] = 3
keep = reg >= 0
return keep, reg[keep]
```

iii. In Step 1/5 the AI says the raw release is already Suite2p-cell-classifier curated and deconvolved, so anatomical inclusion by `iarea` is the only further quality control it applies.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The stored trials are aligned to corridor entry conceptually, but the actual neural samples are the subset of that trial’s frames that meet the running/textured-corridor mask. Alignment is therefore by trial start with variable trial duration and dropped non-running bins.

ii.
```python
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```
```python
ti = build_trials(r, b, nfr_true, day[k])
...
neural = extract_neural(blocks, keep, ti['frame_idx'], zscore=not args.no_zscore)
...
trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. The script header and metadata say the temporal alignment event is `Trial_start_time` / `StartFr` at corridor entry, while `CONVERSION_NOTES.md` says the reference frame mask should still exclude non-running points.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at native imaging-frame resolution. No temporal rebinning is applied; the metadata bin size is derived from the session frame intervals.

ii.
```python
dt = float(np.mean([s['frame_interval_s'] for s in session_info]))
data['metadata'] = {
    'time_bin_size': dt * 1000.0,
    'sampling_rate_hz': 1.0 / dt,
```
```python
'temporal_alignment_event':
    'trial start = corridor entry (beh["Trial_start_time"] / beh["StartFr"]); '
    'timepoints are the native two-photon imaging frames',
```

iii. The notes explicitly say the reference uses raw imaging frames for time-domain analyses and that rebinning would only blur the 3.17 Hz data.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from `SoundTime` and `ft`, not from frame-number interpolation via `SoundFr`.

ii.
```python
ft = b['ft'][:nfr]
...
tcue = np.asarray(b['SoundTime'], dtype=float)
...
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
```

iii. The notes say this variable should be a signed continuous time relative to the cue, and using `SoundTime` is the most direct raw timestamp source.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame of each trial, the code subtracts that frame’s timestamp from the trial’s cue timestamp and converts MATLAB-day units to seconds. The sign convention is positive before the cue and negative after it.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
```

iii. The notes explicitly justify keeping it signed and continuous because the decoder specification asked for “time to sound cue” as a continuous time-varying input.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same frame indices `fi` used to extract neural columns for the trial.

ii.
```python
fi = np.sort(fi)
T = len(fi)
ftt = ft[fi]
...
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
...
g = sub[:, frame_idx][km]
```

iii. The notes repeatedly state that the behavior streams and neural data are aligned on the imaging-frame grid after truncation to the neural recording length.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp` for each session, grouped by mouse name. The code finds the first recording date for each mouse and measures later sessions relative to it.

ii.
```python
def datenum(datexp):
    y, m, d = [int(x) for x in datexp.split('_')]
    import datetime
    return datetime.date(y, m, d).toordinal()
...
for k in keys_all:
    mn = sessions[k]['mname']
    first_day[mn] = min(first_day.get(mn, 1 << 30), datenum(sessions[k]['datexp']))
```

iii. The notes say the release does not contain a trustworthy absolute training-day counter, so `datexp` is the stable raw field available for ordering and spacing sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code converts each session date string to an ordinal day number, subtracts the subject’s first imaging day, and then broadcasts that scalar across every timepoint of every trial in the session.

ii.
```python
day = {k: datenum(sessions[k]['datexp']) - first_day[sessions[k]['mname']] for k in keys_all}
```
```python
inp[1] = day_of_training
```

iii. The notes justify this as “days since this mouse’s first imaging session”, arguing it is more literal than a per-session ordinal when recording days are not consecutive.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time` and `ft`.

ii.
```python
tstart = np.asarray(b['Trial_start_time'], dtype=float)
...
ftt = ft[fi]
...
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```

iii. The notes describe `Trial_start_time` / `StartFr` as corridor entry and treat that event as the temporal anchor for the decoder.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each kept frame, the frame time is subtracted from the trial-start timestamp and converted from days to seconds, yielding a continuous elapsed-time trace.

ii.
```python
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```

iii. The notes justify keeping this as real elapsed time on the native imaging grid, not as a binary event marker or rebinned representation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same per-trial frame subset `fi` that the neural signal uses.

ii.
```python
fi = np.sort(fi)
ftt = ft[fi]
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
...
g = sub[:, frame_idx][km]
```

iii. The notes say all aligned quantities are taken after truncating behavior to the imaged frame count and then indexing by imaging-frame number.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is taken directly from `isRew`.

ii.
```python
isrew = np.asarray(b['isRew']).astype(np.float32)
...
inp[3] = isrew[t]
```

iii. The notes describe `isRew` as the trial-level flag for whether the corridor is the rewarded one.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. There is no transformation beyond casting to float and broadcasting the trial value across all timepoints of that trial.

ii.
```python
isrew = np.asarray(b['isRew']).astype(np.float32)
...
inp[3] = isrew[t]
```

iii. The notes justify this as a direct trial attribute that already exists in the raw behavior without any derived computation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives the stimulus label from `WallName`, but only after building a cross-file `stim_map` from each session’s `UniqWalls` and `stim_id` values merged across all experiment-type entries for that physical recording.

ii.
```python
walls = list(b['UniqWalls'])
sid = np.asarray(b['stim_id'], dtype=float)
for w, s in zip(walls, sid):
    if not np.isnan(s):
        prev = stim_map[key].get(str(w))
        assert prev is None or prev == int(s)
        stim_map[key][str(w)] = int(s)
```
```python
stim_of_trial = np.full(ntrials, -1, dtype=np.int64)
smap = rec['stim_map']
for t in range(ntrials):
    stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)
```

iii. The notes justify this by saying some walls are `NaN` in one experiment type and labeled in another, so merging `stim_id` across duplicate records gives the canonical cross-mouse stimulus identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code assigns each trial one of 7 canonical stimulus ids, drops trials whose wall still lacks a canonical id, and broadcasts the kept trial label across all timepoints of that trial.

ii.
```python
N_STIM = 7
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
...
if stim_of_trial[t] < 0:
    continue
...
out = np.empty((4, T), dtype=np.int64)
out[0] = stim_of_trial[t]
```

iii. In Step 5 and Step 10 the notes say this preserves the reference’s canonical `stim_id` scheme and keeps test-stimulus distinctions that would be lost by collapsing to two or four broad texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`.

ii.
```python
lick_frames = np.floor(b['LickFr']).astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick = np.zeros(nfr, dtype=bool)
lick[lick_frames] = True
```

iii. The notes describe `LickFr` as already indexed to imaging frames, so it is the natural raw variable for a frame-aligned binary lick output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code floors each lick frame to an integer imaging frame, clips away out-of-range licks, and marks each frame as `True` if at least one lick occurred there.

ii.
```python
lick_frames = np.floor(b['LickFr']).astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick = np.zeros(nfr, dtype=bool)
lick[lick_frames] = True
...
out[1] = lick[fi]
```

iii. The notes cite reference code that casts lick frames to integers and justify the binary-per-frame representation as the decoder-friendly form of lick timing.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is indexed by the same trial frame indices `fi` used for neural extraction.

ii.
```python
out[1] = lick[fi]
...
g = sub[:, frame_idx][km]
```

iii. The notes say licking, position, speed, and inputs are all synchronized to the imaging-frame grid before trial splitting.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`.

ii.
```python
pos = b['ft_Pos'][:nfr]
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. The notes describe `ft_Pos` as position in decimeters along the corridor and grey region, making it the direct source for the required 4 spatial bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code divides decimeter position by 10, floors it, and clips it to 4 categories corresponding to the 0-1 m, 1-2 m, 2-3 m, and 3-4 m segments of the textured corridor.

ii.
```python
POS_BIN_DM = 10.0
N_POS_BINS = 4
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. The notes explicitly say the decoder requires four 1 m spatial bins and that the kept frames are already restricted to the 4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded by fixed spatial edges at 0, 10, 20, 30, and 40 decimeters.

ii.
```python
POS_BIN_DM = 10.0
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. The notes justify this as the exact four equal-length 1 m bins requested by the decoder specification.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same kept frame indices `fi` as the neural data.

ii.
```python
ftt = ft[fi]
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. The notes emphasize frame-synchronous alignment across behavior and neural data after truncating to the neural frame count.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`.

ii.
```python
speed = b['ft_RunSpeed'][:nfr]
...
return dict(..., speed=speed[frame_idx].astype(np.float32), ...)
```

iii. The notes say `ft_RunSpeed`, not `RunFr`, is the correct frame-level running-speed trace.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code first pools all kept-frame running speeds across all recordings in a behavior-only pass, computes global 25th/50th/75th percentile thresholds, then replays those thresholds session by session in the main conversion pass.

ii.
```python
speeds, ntr1 = [], 0
for k in keys_all:
    ti = build_trials(sessions[k], beh[k], len(beh[k]['ft']), day[k])
    speeds.append(ti['speed'])
...
speed_all = np.concatenate(speeds)
speed_edges = np.percentile(speed_all, [25, 50, 75])
```
```python
sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
for i in range(len(ti['inputs'])):
    a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
    ti['outputs'][i][3] = sb[a:bnd]
```

iii. In Step 5 and Step 12, the AI justifies this as matching the instruction “4 bins, each corresponding to 25% of the data” at the whole-dataset level and providing consistent semantics across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by global percentile edges and binned with `np.searchsorted(..., side='right')`.

ii.
```python
speed_edges = np.percentile(speed_all, [25, 50, 75])
...
sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
```

iii. The notes say these are intended to be global quartiles of the kept-timepoint speed distribution.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed values are extracted on the same `frame_idx` as the neural data and then split trial by trial with the same bounds.

ii.
```python
return dict(..., speed=speed[frame_idx].astype(np.float32), ...)
```
```python
sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
for i in range(len(ti['inputs'])):
    a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
    ti['outputs'][i][3] = sb[a:bnd]
```

iii. The notes describe all per-frame outputs as built on the frame grid first and then split into trials with shared bounds.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates all behavior frame streams to the true neural frame count, excludes non-finite or out-of-range trial indices, drops out-of-range lick frames, merges incomplete `stim_id` annotations across duplicate experiment-type entries, guards zero-variance neurons during z-scoring, and keeps but documents one session with corrupted `ft_RunSpeed` rather than inventing a correction.

ii.
```python
ft = b['ft'][:nfr]
tr = b['ft_trInd'][:nfr]
...
valid = corr & move & np.isfinite(tr)
tr_int[(tr_int < 0) | (tr_int >= ntrials)] = -1
```
```python
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
```
```python
if not np.isnan(s):
    prev = stim_map[key].get(str(w))
    assert prev is None or prev == int(s)
    stim_map[key][str(w)] = int(s)
```
```python
sd = np.sqrt(np.maximum(s2 / nf - mu * mu, 0.0))
sd[sd == 0] = 1.0
```

iii. The Step 10 and Step 12 notes explicitly enumerate these cases, including `ft_trInd = NaN`, out-of-range indices, missing `stim_id` entries, zero-variance neurons, and the defective running-speed trace in `DR10_2022_07_12`.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading the very large spike files and extracting/z-scoring neural activity across many neurons and frames. The code also treats this as an engineering bottleneck and adds prefetching and multithreaded extraction.

ii.
```python
def load_spk_blocks(mname, datexp, blk):
    fn = os.path.join(ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
    spks = np.load(fn, allow_pickle=True).item()['spks']
```
```python
def extract_neural(blocks, keep_mask, frame_idx, zscore=True, nthreads=16, chunk=4096):
    ...
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(work, tasks))
```

iii. The notes report that the 89 spike files total about 404 GB and show timing tables where spike loading and neural extraction dominate runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes the biggest former bottleneck by grouping frames with `argsort`/`bincount`. The main remaining Python loops are per-trial loops for assigning stimulus ids, building per-trial input/output arrays, and splitting the gathered neural matrix back into trial arrays.

ii.
```python
order = np.argsort(tr_int, kind='stable')
counts = np.bincount(tr_int[tr_int >= 0], minlength=ntrials)
```
```python
for t in range(ntrials):
    stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)
...
for i in range(len(ti['inputs'])):
    a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
    trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. The notes say the agent rewrote extraction for performance, so the remaining loops are smaller secondary costs rather than the core bottleneck.

## 12-c. What processing does the code repeat multiple times?

i. The largest repeated step is `build_trials`: it is run once in pass 1 on behavior-only data to gather running-speed values and again in pass 2 with the exact neural frame count to build the final trials. The code also constructs full per-trial input/output structures in pass 1 even though that pass only needs speeds.

ii.
```python
for k in keys_all:
    ti = build_trials(sessions[k], beh[k], len(beh[k]['ft']), day[k])
    speeds.append(ti['speed'])
```
```python
for si in range(len(keys)):
    ...
    ti = build_trials(r, b, nfr_true, day[k])
```

iii. The notes justify the two-pass structure because the global speed quartiles must be fixed over all recordings before final conversion, but they also acknowledge this as an implementation tradeoff.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several intermediate products are only scaffolding: pass 1 builds full `inputs`/`outputs`/`trial_ids` structures just to collect `speed`, pass 2 attaches `ti['outputs_speedbin_edges']` but the saved dataset does not use that field, the script gathers `pos` and `ft` inside `build_trials` mainly for diagnostics, and the full gathered neural matrix is built before being split and deleted. Optional plotting is also purely diagnostic.

ii.
```python
return dict(frame_idx=frame_idx, bounds=bounds, inputs=inputs, outputs=outputs,
            trial_ids=np.array(trial_ids), speed=speed[frame_idx].astype(np.float32),
            pos=pos[frame_idx].astype(np.float32), ft=ft[frame_idx],
            ntrials_raw=ntrials, nvalid_frames=int(valid.sum()))
```
```python
ti['outputs_speedbin_edges'] = speed_edges
```
```python
neural = extract_neural(blocks, keep, ti['frame_idx'], zscore=not args.no_zscore)
...
trials.append(np.ascontiguousarray(neural[:, a:bnd]))
...
del neural
```

iii. The notes frame these as acceptable costs for validation, plotting, and dataset-level speed binning, not as processing needed by the final decoder pickle itself.
