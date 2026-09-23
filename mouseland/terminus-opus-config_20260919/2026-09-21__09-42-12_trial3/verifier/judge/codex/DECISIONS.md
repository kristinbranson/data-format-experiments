# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a master recording table from `beh/Imaging_Exp_info.npy`, deduplicates repeated `(mouse, date, block)` recordings across experiment types, then loads behavior from each `Beh_<exp_type>.npy` file, spikes from `spk/<session>_neural_data.npy`, and retinotopy labels from `retinotopy/<mouse>_<date>_trans.npz`. Behavior wall-to-stimulus maps are merged across duplicated appearances of the same recording.

ii. 
```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
```
```python
for exp_type, db in exp_info.items():
    for ndb in db:
        key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
```
```python
B = np.load(os.path.join(BEH_DIR, 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
```
```python
d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])),
              allow_pickle=True)
```

iii. `CONVERSION_NOTES.md` says the agent verified that duplicate behavior entries are identical except for `UniqWalls`/`stim_id`, so it kept one behavior record per recording and merged wall-label maps. The trajectory also shows this was chosen to avoid counting duplicated sessions twice.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by `mname`. The AI groups recordings by mouse name, computes per-mouse training-day metadata from recording dates, and later builds `subjects` as sorted unique mouse names with `subject_idx` pointing into that list.

ii. 
```python
by_mouse = collections.defaultdict(list)
for key, rec in recs.items():
    ...
    by_mouse[rec['mname']].append(rec)
```
```python
subjects = sorted({r['mname'] for r in results})
'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64),
```

iii. The notes state there are 89 sessions from 19 mice and that `mname` is the natural subject identifier from the index.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `mname_datexp_blk`. If the same recording appears under multiple experiment types, it is deduplicated into a single session while retaining all associated behavior keys for later label merging.

ii. 
```python
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
rec = recs.setdefault(key, {'key': key, 'mname': ndb['mname'],
                            'datexp': ndb['datexp'], 'blk': ndb['blk'],
                            'exp_types': [], 'beh_keys': [], ...})
```

iii. In `CONVERSION_NOTES.md`, the AI says there are 142 `(exp_type, session)` entries but only 89 unique recordings, so deduplication was necessary to match the paper’s session count.

## 1-d. How are the data split into trials?

i. Trials are built from framewise trial labels `ft_trInd`, but only using frames inside the textured corridor and while the mouse is running. The AI groups valid frames by trial label, so each trial becomes the ordered set of kept frames for that traversal rather than all frames from start to end.

ii. 
```python
tr = beh['ft_trInd'][:nfr]
valid = (beh['ft_CorrSpc'][:nfr].astype(bool) & (beh['ft_move'][:nfr] > 0)
         & ~np.isnan(tr))
frames = np.flatnonzero(valid)
labels = tr[frames].astype(int)
order = np.lexsort((frames, labels))
frames, labels = frames[order], labels[order]
bounds = np.flatnonzero(np.diff(labels)) + 1
groups = np.split(np.arange(len(frames)), bounds)
trial_ids = [int(labels[g[0]]) for g in groups]
```

iii. The notes say this mirrors the reference `fr_valid` mask from `utils.Get_dprime_selective_neuron`, prioritizing the paper’s “inside the 0-4 m corridor” and “running only” rules over retaining every frame between `StartFr` and `GrayFr`.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not explicitly drop long or low-quality trials. Any trial with at least one kept frame after applying the corridor-and-running mask is retained; in the full run it reported that all behavior trials remained and the minimum kept length was 11 bins.

ii. 
```python
for g, tid in zip(groups, trial_ids):
    T = len(g)
    neural_out.append(np.ascontiguousarray(spk[:, g]))
    ...
    kept_trials.append(tid)
```
```python
print('all trials kept? %s (behaviour trials %d, converted trials %d)'
      % (all(len(r['neural']) == r['ntrials_beh'] for r in results),
         sum(r['ntrials_beh'] for r in results), int(np.sum(ntrials))))
```

iii. The justification in the notes is that every trial had enough valid corridor+running frames, so no additional trial rejection was needed once non-running frames were already removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the deconvolved fluorescence traces in `spks` inside each session’s spike file, together with `iarea` from the retinotopy file to assign region labels and choose neurons in named visual areas.

ii. 
```python
d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
planes = d['spks']
```
```python
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])),
              allow_pickle=True)
iarea = ret['iarea']
```

iii. The notes explicitly state the data are already suite2p deconvolved traces and that no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes logically, filters to named visual areas, randomly subsamples to at most 2000 neurons per session, keeps only selected frames, and stores per-trial neuron-by-time matrices in native frame units as `float32`.

ii. 
```python
def select_neurons(iarea, max_neurons, seed=RNG_SEED):
    ...
    if len(in_area) > max_neurons:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(in_area, size=max_neurons, replace=False))
```
```python
spk, _, _ = load_spk_selected(key, sel, frames)
...
neural_out.append(np.ascontiguousarray(spk[:, g]))
```

iii. The notes justify the subsample on storage grounds: keeping all in-area neurons was estimated at about 152 GB, whereas 2000 neurons per session produced a 6.6 GB dataset and the downstream decoder itself projects to 100 PCs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cell-quality metric is applied beyond the released suite2p curation. The AI keeps only neurons whose `iarea` maps to one of `V1`, `mHV`, `lHV`, or `aHV`, then may randomly subsample those retained neurons to `MAX_NEURONS`.

ii. 
```python
region_idx = np.full(len(iarea), -1, dtype=np.int64)
for r, name in enumerate(BRAIN_REGIONS):
    region_idx[np.isin(iarea, AREA_CODES[name])] = r
in_area = np.flatnonzero(region_idx >= 0)
```
```python
if len(in_area) > max_neurons:
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(in_area, size=max_neurons, replace=False))
else:
    sel = in_area
```

iii. The AI’s notes say the reference applies no further neuron-quality filtering, but that storing all in-area neurons was too large, so subsampling was added as a tractability choice rather than as biological QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (`StartFr`), but the actual stored neural sequence contains only valid corridor+running frames from that trial. The alignment event is documented in metadata as trial start / corridor entry.

ii. 
```python
'temporal_alignment_event':
    'trial start = entry into the virtual-reality corridor (beh["StartFr"])',
'off_start': 0.0,
'off_end': None,
```
```python
for g, tid in zip(groups, trial_ids):
    neural_out.append(np.ascontiguousarray(spk[:, g]))
```

iii. The notes say corridor entry is the instruction-specified alignment event, but they also justify dropping non-running frames because the reference analyses use `fr_valid = ft_CorrSpc & ft_move>0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is one imaging frame. The AI estimates the session frame period from `ft` and writes the dataset time bin as the median session `dt` in milliseconds. No temporal rebinning is applied.

ii. 
```python
def frame_times_s(beh, nfr):
    ft = beh['ft'][:nfr]
    return (ft - ft[0]) * 86400.0
```
```python
'time_bin_size': dt * 1000.0,
'frame_rate_hz': 1.0 / dt,
```

iii. The notes cite the reference 3.17 Hz imaging rate and state that the native frame grid is already the common time base for neural and behavioral streams.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and `ft`.

ii. 
```python
t_frames = frame_times_s(beh, nfr)
fr_axis = np.arange(nfr)
t_cue = np.interp(beh['SoundFr'], fr_axis, t_frames)
```

iii. The notes say cue times are recorded as fractional frame indices and are converted to seconds on the imaging-frame time axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI interpolates cue frame indices onto session time in seconds, then computes `t_cue - t_frame` on each kept frame so the value is positive before the cue and negative after it.

ii. 
```python
t_cue = np.interp(beh['SoundFr'], fr_axis, t_frames)
...
t_sel = t_frames[frames]
time_to_cue = t_cue[labels] - t_sel
```

iii. The notes and trajectory both state that `SoundFr` is fractional and must be put on the same seconds scale as the frame timestamps before taking the difference.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `t_sel`, the same selected frame times used to build the neural trials, and then split by the same per-trial groups.

ii. 
```python
t_sel = t_frames[frames]
time_to_cue = t_cue[labels] - t_sel
...
inp[0] = time_to_cue[g]
```

iii. The notes say all streams are aligned in imaging-frame units, and the code uses the identical grouped frame index `g` as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date in `datexp`, grouped by mouse `mname`.

ii. 
```python
y, m, d = [int(v) for v in rec['datexp'].split('_')]
rec['date'] = datetime.date(y, m, d)
...
d0 = min(r['date'] for r in rs)
for r in rs:
    r['day_of_training'] = float((r['date'] - d0).days)
```

iii. The notes say the AI interpreted “day of training” literally as elapsed calendar days since each mouse’s first recording, not merely session count.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the AI finds the earliest recording date and assigns every session the integer number of elapsed days since that first date. That scalar is then broadcast across all bins of each trial.

ii. 
```python
d0 = min(r['date'] for r in rs)
for r in rs:
    r['day_of_training'] = float((r['date'] - d0).days)
```
```python
inp[1] = day
```

iii. `CONVERSION_NOTES.md` explicitly records this as “days since the mouse’s first imaging session.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and `ft`.

ii. 
```python
t_start = np.interp(beh['StartFr'], fr_axis, t_frames)
```

iii. The notes identify `StartFr` as corridor entry and therefore the trial-start event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI interpolates `StartFr` onto the frame-time axis and computes `t_frame - t_start` for each selected frame.

ii. 
```python
t_start = np.interp(beh['StartFr'], fr_axis, t_frames)
...
time_since_start = t_sel - t_start[labels]
```

iii. The notes justify this as the continuous elapsed time from corridor entry, even when non-running frames are omitted from the stored sequence.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned on the same selected frame times `t_sel` and split into trials with the same grouped indices `g` used for neural activity.

ii. 
```python
time_since_start = t_sel - t_start[labels]
...
inp[2] = time_since_start[g]
```

iii. The AI’s notes and sample validations say it re-derived these values directly from raw behavior for spot-checked trials and matched the stored trial inputs.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from canonicalized stimulus identity built from `WallName`, `UniqWalls`, and `stim_id`, plus session reward status inferred from `RewardFr`.

ii. 
```python
for w, s in zip(beh['UniqWalls'], np.atleast_1d(beh['stim_id']).astype(float)):
    if not np.isnan(s):
        w2id[str(w)] = int(s)
```
```python
rec['is_task'] = bool(np.any(~np.isnan(beh['RewardFr'])))
```
```python
stim_ids, wall_names = stim_categories(beh, rec['wall2id'])
reward_available = (stim_ids == 2) & rec['is_task']
```

iii. The trajectory and notes both say the AI investigated `isRew` and concluded it marks delivered reward, not rewarded-corridor availability, so it used canonical stimulus identity instead.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI first merges wall-to-`stim_id` mappings across duplicate behavior files, infers whether a session is a reward-bearing task session from `RewardFr`, then marks a trial as reward-available when its canonical `stim_id` is 2 in a task session. The result is broadcast across time bins in that trial.

ii. 
```python
def stim_categories(beh, wall2id):
    wn = np.array([str(w) for w in beh['WallName']])
    ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
    return ids, wn
```
```python
reward_available = (stim_ids == 2) & rec['is_task']
...
inp[3] = float(reward_available[tid])
```

iii. The AI justified this in the notes as matching the task statement “1 if in rewarded corridor” and the paper code’s rewarded-corridor logic better than `isRew`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, canonicalized through the per-recording wall-to-`stim_id` map assembled from `UniqWalls` and `stim_id`.

ii. 
```python
def stim_categories(beh, wall2id):
    wn = np.array([str(w) for w in beh['WallName']])
    ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
    return ids, wn
```

iii. The notes say raw texture names differ across mice, so the AI preferred canonical task-relative IDs over physical texture names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses canonical `stim_id` codes rather than broad physical categories. It keeps 8 categories (`STIM_VALUES`), including an extra `UNRESOLVED_STIM_ID = 7` for `circle3`, and broadcasts the per-trial category across all bins of that trial.

ii. 
```python
STIM_VALUES = ['nonrew_crop1', 'nonrew_crop2', 'rew_crop1', 'rew_crop2',
               'rew_crop3', 'rew_crop1_swap1', 'rew_crop1_swap2', 'nonrew_crop3']
UNRESOLVED_STIM_ID = 7
```
```python
out = np.empty((4, T), dtype=np.int16)
out[0] = stim_ids[tid]
```

iii. The notes justify this as a mouse-invariant labeling scheme and explicitly document the special handling of `circle3`, which lacked a canonical ID in the source files.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`.

ii. 
```python
lick_any = np.zeros(nfr, dtype=bool)
if len(beh['LickFr']):
    lf = np.round(beh['LickFr']).astype(int)
    lf = lf[(lf >= 0) & (lf < nfr)]
    lick_any[lf] = True
```

iii. The notes state that lick events are already recorded on the imaging-frame axis and that non-task sessions simply have empty lick arrays.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame indices are rounded to the nearest frame, clipped to valid frame bounds, converted to a per-frame boolean flag, and then sampled on the selected kept frames to produce a binary time series.

ii. 
```python
lf = np.round(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
lick_any[lf] = True
licking = lick_any[frames].astype(np.int64)
```

iii. The AI justified the all-zero non-task sessions as behaviorally correct because those mice were never water restricted and their source files contain no licks.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is first placed on the full frame grid, then restricted to the same selected frame indices and trial groups as the neural data.

ii. 
```python
licking = lick_any[frames].astype(np.int64)
...
out[1] = licking[g]
```

iii. The notes describe this as exact alignment on the imaging-frame basis shared by all streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. 
```python
pos = beh['ft_Pos'][:nfr][frames]
```

iii. The notes say positions are in decimetres, with the textured corridor spanning 0-40 dm.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI uses the selected-frame positions, divides by 10 dm per meter, converts to integers, and clips into four bins.

ii. 
```python
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. The notes justify this directly from the task requirement of 4 equal 1 m bins within the 4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` meter-equivalent bins by integer division in decimetres and clipping to `0..3`.

ii. 
```python
POS_BIN_DM = 10.0
N_POS_BINS = 4
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. The notes repeatedly describe this as “4 x 1 m bins.”

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is evaluated only on the selected frame indices and then split into trials using the same grouped frame indices as neural activity.

ii. 
```python
pos = beh['ft_Pos'][:nfr][frames]
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
...
out[2] = pos_bin[g]
```

iii. The notes say this is aligned on the same kept imaging frames as all other streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
speed = beh['ft_RunSpeed'][:nfr][frames]
```

iii. The notes say the quartile thresholds are computed from run-speed values on the kept frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI collects running speeds from all kept frames across all sessions, computes global 25th/50th/75th percentile thresholds, and then bins each selected frame’s speed with `np.digitize`.

ii. 
```python
speeds = []
for key, rec in recs.items():
    beh = beh_by_rec[key]
    speeds.append(session_speed_values(beh, len(beh['ft']) - 2))
speeds = np.concatenate(speeds)
speed_thr = np.percentile(speeds, [25, 50, 75])
```
```python
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
```

iii. The notes justify this as matching the task requirement that the four bins correspond to 25% of the data globally.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global thresholds are taken at the 25th, 50th, and 75th percentiles of all kept-frame speeds, and `np.digitize` maps each frame to categories 0-3.

ii. 
```python
speed_thr = np.percentile(speeds, [25, 50, 75])
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
```

iii. The trajectory shows the agent explicitly computed global run-speed quartiles before writing the converter and carried those thresholds through as metadata.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same selected frame indices as the neural data and then split using the same per-trial groups.

ii. 
```python
speed = beh['ft_RunSpeed'][:nfr][frames]
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
...
out[3] = speed_bin[g]
```

iii. The notes describe this as the common imaging-frame alignment used throughout the conversion.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior streams to the number of imaging frames, removes `NaN` trial labels when defining valid frames, clips lick-frame indices to valid bounds, merges duplicated behavior label maps across experiment types, and assigns unresolved wall names to category 7 rather than dropping them.

ii. 
```python
valid = (beh['ft_CorrSpc'][:nfr].astype(bool) & (beh['ft_move'][:nfr] > 0)
         & ~np.isnan(tr))
```
```python
lf = np.round(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
```
```python
ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
```

iii. The notes justify these choices as practical fixes for known data quirks: behavior arrays being 1-2 frames longer than spikes, duplicate behavior entries with incomplete `stim_id` maps, and `circle3` lacking a canonical ID.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant cost is loading the very large spike files and converting sessions; the notes also mention pickling the final dataset as a nontrivial cost.

ii. 
```python
d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
planes = d['spks']
```
```python
with Pool(min(args.nproc, len(jobs))) as pool:
    for i, res in enumerate(pool.imap_unordered(convert_session, jobs)):
```

iii. `CONVERSION_NOTES.md` explicitly says the 405 GB spike files are the expensive part and that the main optimizations focused on spike I/O and parallel session conversion.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes most framewise work, but it still loops over imaging planes in `load_spk_selected` and over trials when packaging outputs. Those loops could only be reduced further with a more complex in-memory layout.

ii. 
```python
for p in planes:
    n = p.shape[0]
    rows = sel[(sel >= off) & (sel < off + n)] - off
    if len(rows):
        out.append(np.asarray(p[np.ix_(rows, frames)], dtype=np.float32))
    off += n
```
```python
for g, tid in zip(groups, trial_ids):
    T = len(g)
    neural_out.append(np.ascontiguousarray(spk[:, g]))
    ...
    output_out.append(out)
```

iii. The notes say per-session frame computations were intentionally vectorized already and that remaining overhead outside spike I/O was negligible.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats some work: it computes trial-frame masks once for global speed-threshold estimation and again during per-session conversion, and it opens each spike file once to get frame counts and again to fetch selected data.

ii. 
```python
speeds.append(session_speed_values(beh, len(beh['ft']) - 2))
```
```python
_, n_neu_spk, nfr = load_spk_selected(key, None, None)
...
spk, _, _ = load_spk_selected(key, sel, frames)
```

iii. The notes accept these repeats because they simplify the pipeline and still keep the full conversion runtime well under the agent’s budget.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores substantial bookkeeping and visualization-oriented information that the decoder does not use: wall names, per-session timing stats, exptype/reward summaries, optional processing plots, and detailed metadata fields such as `session_info`.

ii. 
```python
stim_ids, wall_names = stim_categories(beh, rec['wall2id'])
```
```python
if show_processing:
    plot_processing(...)
```
```python
'session_info': [
    {'session': i, 'key': r['key'], 'mouse': r['mname'], 'date': r['datexp'],
     'block': r['blk'], 'exptypes': r['exptypes'], 'is_task_session': r['is_task'],
     ...}
    for i, r in enumerate(results)],
```

iii. The notes frame these extras as validation and documentation support rather than downstream decoder inputs, so they were kept despite not affecting training.
