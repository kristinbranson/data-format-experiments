# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master index `Imaging_Exp_info.npy` from `beh/`, deduplicates recordings across experiment types into 89 unique sessions, loads behaviour files (`Beh_<exp_type>.npy`) in parallel using multiprocessing, loads spike data per session from `spk/<session_id>_neural_data.npy`, and loads retinotopy data from `retinotopy/<mouse>_<date>_trans.npz`. Behaviour is loaded with parallel workers, and behaviour across duplicate experiment-type entries is merged to build a wall-to-stim_id mapping.

ii.
```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# ...
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
# ...
B = np.load(os.path.join(BEH_DIR, 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
# ...
d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
# ...
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])), allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md that recordings can appear under multiple experiment types and verified that behaviour arrays are identical across duplicates, so one behaviour record per session suffices. The wall-to-stim_id maps are merged from all appearances.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info. Sessions are grouped by mouse name. 19 unique mice are identified, matching the paper.

ii.
```python
subjects = sorted({r['mname'] for r in results})
data['subject_idx'] = np.array([subjects.index(r['mname']) for r in results], dtype=np.int64)
```

iii. The AI noted that the index already contains the mouse name. No derivation needed.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `mname_datexp_blk`. Recordings appearing under multiple experiment types are deduplicated via an OrderedDict keyed by this ID. 89 unique sessions are produced.

ii.
```python
recs = collections.OrderedDict()
for exp_type, db in exp_info.items():
    for ndb in db:
        key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        rec = recs.setdefault(key, {...})
```

iii. The AI verified that 142 (exp_type, session) entries collapse to 89 unique recordings, matching the paper's count.

## 1-d. How are the data split into trials?

i. Trials are corridor traversals identified by `ft_trInd`. The AI keeps frames that are (a) inside the texture corridor (`ft_CorrSpc`) AND (b) the mouse is running (`ft_move > 0`), truncated to the number of imaging frames. Frames are grouped by trial index.

ii.
```python
def trial_frames(beh, nfr):
    tr = beh['ft_trInd'][:nfr]
    valid = (beh['ft_CorrSpc'][:nfr].astype(bool) & (beh['ft_move'][:nfr] > 0)
             & ~np.isnan(tr))
    frames = np.flatnonzero(valid)
    labels = tr[frames].astype(int)
    # group by trial
    order = np.lexsort((frames, labels))
    frames, labels = frames[order], labels[order]
    bounds = np.flatnonzero(np.diff(labels)) + 1
    groups = np.split(np.arange(len(frames)), bounds)
    trial_ids = [int(labels[g[0]]) for g in groups]
    return frames, labels, groups, trial_ids
```

iii. The AI justified this by citing the reference code `utils.Get_dprime_selective_neuron` which uses `VRmove = beh['ft_move'][:nfr]>0` and `isCorridor = beh['ft_CorrSpc'][:nfr]` to define valid frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter any trials based on quality controls. All trials with at least one valid (corridor + running) frame are kept. No trial length filtering is applied.

ii. All trials from `trial_frames` are kept:
```python
for g, tid in zip(groups, trial_ids):
    T = len(g)
    neural_out.append(np.ascontiguousarray(spk[:, g]))
```

iii. The AI noted that every trial has >= 11 valid frames after the corridor + running filter, so no trial is lost. The AI did not implement any outlier trial length filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` (a list of per-plane arrays concatenated). The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
planes = d['spks']
# ...
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])))
iarea = ret['iarea']
```

iii. These are the suite2p deconvolved traces, as documented in the reference code.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, selects neurons in named visual areas, then subsamples to at most 2000 neurons per session using a fixed random seed. Only the kept (corridor + running) frames of each trial are stored. Data is stored as float32.

ii.
```python
def select_neurons(iarea, max_neurons, seed=RNG_SEED):
    region_idx = np.full(len(iarea), -1, dtype=np.int64)
    for r, name in enumerate(BRAIN_REGIONS):
        region_idx[np.isin(iarea, AREA_CODES[name])] = r
    in_area = np.flatnonzero(region_idx >= 0)
    if len(in_area) > max_neurons:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(in_area, size=max_neurons, replace=False))
    else:
        sel = in_area
    return sel, region_idx[sel], len(iarea), len(in_area)
```

iii. The AI justified the 2000 neuron cap by noting the full dataset would be ~152 GB and the decoder reduces each session to 100 PCs anyway. The area composition is preserved proportionally.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside the four named visual areas (V1, mHV, lHV, aHV) are dropped. Additionally, a random subsample of at most 2000 neurons per session is taken from those in a named area.

ii.
```python
in_area = np.flatnonzero(region_idx >= 0)
if len(in_area) > max_neurons:
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(in_area, size=max_neurons, replace=False))
```

iii. The AI noted that the reference code applies no further neuron quality filtering beyond suite2p cell classification. The subsampling is a practical constraint to keep file size manageable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). Only the kept frames (corridor + running) of each trial are stored. Trials have variable length. The neural data for a trial is the selected neurons at the selected frames of that trial.

ii.
```python
spk, _, _ = load_spk_selected(key, sel, frames)
# ...
neural_out.append(np.ascontiguousarray(spk[:, g]))
```

iii. The AI documented that the alignment event is trial start (corridor entry via `StartFr`), and that trials have variable length because mice run at different speeds.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. The imaging frame rate (~3.17 Hz, ~315 ms per bin) is the native temporal resolution. The bin size is computed from the median frame-to-frame interval.

ii.
```python
dt = float(np.median([r['dt_s'] for r in results]))
# ...
'time_bin_size': dt * 1000.0,
```

iii. The AI noted that the imaging frame is the finest resolution the data has, and all behaviour streams are already on the same grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame index of the sound cue for each trial) and `ft` (timestamps of imaging frames).

ii.
```python
t_cue = np.interp(beh['SoundFr'], fr_axis, t_frames)
```

iii. The cue frame is fractional, so it's interpolated onto the frame time axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue frame is interpolated to seconds via the frame time axis. The input is `t_cue[trial] - t_frame`, which is positive before the cue and negative after.

ii.
```python
time_to_cue = t_cue[labels] - t_sel  # > 0 before the cue, < 0 after
```

iii. The sign convention follows the name "time TO sound cue": positive means the cue is in the future.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same kept frames used for the neural data of each trial.

ii.
```python
t_sel = t_frames[frames]
# ...
time_to_cue = t_cue[labels] - t_sel
# ...
inp[0] = time_to_cue[g]
```

iii. All data streams are aligned via the frame index.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` (date string) of each session, parsed into a `datetime.date`.

ii.
```python
y, m, d = [int(v) for v in rec['datexp'].split('_')]
rec['date'] = datetime.date(y, m, d)
# ...
d0 = min(r['date'] for r in rs)
for r in rs:
    r['day_of_training'] = float((r['date'] - d0).days)
```

iii. The date is parsed from the session ID to compute calendar day differences.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is found. Day of training is the number of calendar days between the session's date and the mouse's earliest recording date. This is a float value broadcast across all bins of every trial.

ii.
```python
d0 = min(r['date'] for r in rs)
for r in rs:
    r['day_of_training'] = float((r['date'] - d0).days)
```

iii. This gives the actual calendar day difference (e.g., 0, 7, 14, ...) rather than a session count (0, 1, 2, ...).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (frame index of corridor entry for each trial) and `ft` (timestamps of imaging frames).

ii.
```python
t_start = np.interp(beh['StartFr'], fr_axis, t_frames)
```

iii. The start frame is fractional and interpolated onto the frame time axis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is `t_frame - t_start[trial]`, in seconds. It is positive after corridor entry and near zero at the start of each trial.

ii.
```python
time_since_start = t_sel - t_start[labels]  # >= 0
```

iii. A simple time difference from the corridor entry timestamp.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same kept frames used for the neural data.

ii.
```python
t_sel = t_frames[frames]
# ...
inp[2] = time_since_start[g]
```

iii. Aligned via frame index.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the canonical `stim_id` (derived from `WallName` and `UniqWalls`/`stim_id` mappings) and whether the session is a "task" session (determined by checking if any `RewardFr` values are non-NaN).

ii.
```python
stim_ids, wall_names = stim_categories(beh, rec['wall2id'])
reward_available = (stim_ids == 2) & rec['is_task']
```

iii. The AI reasoned that `isRew` means "reward was actually delivered" (lick-triggered), not "trial was in the rewarded corridor". The decoder spec asks for "1 if in rewarded corridor, 0 if not", so the AI uses `stim_id == 2` (rewarded corridor) AND task session status.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The canonical stimulus ID is determined by merging the `UniqWalls`-to-`stim_id` mappings across all experiment types for each session. A trial is in the rewarded corridor if its canonical `stim_id` is 2 AND the session is a task session (has water rewards). The value is per-trial, broadcast across time bins.

ii.
```python
reward_available = (stim_ids == 2) & rec['is_task']
# ...
inp[3] = float(reward_available[tid])
```

iii. The AI verified that `isRew == ~isnan(RewardFr)` (reward actually delivered) and deliberately chose the corridor-based definition instead.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (the texture name of each trial's corridor walls) and the merged `UniqWalls`-to-`stim_id` mapping.

ii.
```python
def stim_categories(beh, wall2id):
    wn = np.array([str(w) for w in beh['WallName']])
    ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
    return ids, wn
```

iii. The mapping is built by merging `stim_id` labels from all experiment types in which a recording appears.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the canonical `stim_id` codes (0-7) as stimulus categories. This gives 8 categories: `nonrew_crop1` (0), `nonrew_crop2` (1), `rew_crop1` (2), `rew_crop2` (3), `rew_crop3` (4), `rew_crop1_swap1` (5), `rew_crop1_swap2` (6), `nonrew_crop3` (7). The value is per-trial, broadcast across time bins.

ii.
```python
STIM_VALUES = ['nonrew_crop1', 'nonrew_crop2', 'rew_crop1', 'rew_crop2',
               'rew_crop3', 'rew_crop1_swap1', 'rew_crop1_swap2', 'nonrew_crop3']
# ...
out[0] = stim_ids[tid]
```

iii. The AI argued that canonical `stim_id` encodes the task role consistently across mice (rewarded vs non-rewarded, crop variants, swaps), whereas the physical texture names differ between mice.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame indices of lick events in each session.

ii.
```python
lick_any = np.zeros(nfr, dtype=bool)
if len(beh['LickFr']):
    lf = np.round(beh['LickFr']).astype(int)
    lf = lf[(lf >= 0) & (lf < nfr)]
    lick_any[lf] = True
licking = lick_any[frames].astype(np.int64)
```

iii. Lick frame indices are rounded to the nearest integer frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A frame is 1 if at least one lick falls in it, else 0. The lick frame is fractional, so it is rounded (via `np.round`) to the nearest frame. Lick frames outside the imaging range are excluded.

ii.
```python
lf = np.round(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
lick_any[lf] = True
licking = lick_any[frames].astype(np.int64)
```

iii. The AI noted that non-task sessions have empty lick arrays, so licking = 0 for all 61 non-task sessions.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames directly, so the licking flag is already on the same grid as neural data. It is indexed with the same kept frames.

ii.
```python
licking = lick_any[frames].astype(np.int64)
# ...
out[1] = licking[g]
```

iii. Aligned via frame index.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame, in decimetres (0 to 40 across the texture, up to 60 through grey space).

ii.
```python
pos = beh['ft_Pos'][:nfr][frames]
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. Positions are in decimetres.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimetres is divided by 10 (1 m = 10 dm) and truncated to integer to get 4 bins: [0-1m), [1-2m), [2-3m), [3-4m). Values are clipped to [0, 3].

ii.
```python
POS_BIN_DM = 10.0
N_POS_BINS = 4
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. The 4 equal 1 m bins match the instruction specification.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Divided by 10 dm (1 m), truncated to integer, clipped to [0, 3]. This gives 4 categories for positions 0-1m, 1-2m, 2-3m, 3-4m.

ii. Same as 9-b above.

iii. Since frames are already filtered to `ft_CorrSpc` (0-4m texture region) and `ft_move > 0`, all positions are within the 0-40 dm range.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one position per imaging frame. It is indexed at the same kept frames as the neural data.

ii.
```python
pos = beh['ft_Pos'][:nfr][frames]
# ...
out[2] = pos_bin[g]
```

iii. Aligned via frame index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr][frames]
```

iii. Directly from the behavioural variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using **global** quartile thresholds computed across the kept frames of ALL sessions. The thresholds are computed once as the 25th, 50th, and 75th percentiles of the speed values of all kept frames, then `np.digitize` is used to assign bins.

ii.
```python
# Global thresholds
speeds = []
for key, rec in recs.items():
    beh = beh_by_rec[key]
    speeds.append(session_speed_values(beh, len(beh['ft']) - 2))
speeds = np.concatenate(speeds)
speed_thr = np.percentile(speeds, [25, 50, 75])
# Per trial
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
```

iii. The AI chose global quartile thresholds so speed bins are consistent across sessions. This means each bin contains ~25% of the data globally, but per-session distributions can be uneven.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global thresholds (25th, 50th, 75th percentiles of all kept-frame speeds) split the data into 4 bins using `np.digitize`.

ii.
```python
speed_thr = np.percentile(speeds, [25, 50, 75])
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
```

iii. The thresholds are ~12.4, 25.4, 40.9 cm/s.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one speed per imaging frame. It is indexed at the same kept frames as the neural data.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr][frames]
# ...
out[3] = speed_bin[g]
```

iii. Aligned via frame index.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behaviour arrays are truncated to the number of imaging frames (`nfr`). Lick frames outside the imaging range are dropped. Trials with no valid frames after the corridor+running filter would be dropped (though none exist). The AI handles sessions where `LickFr` is empty (non-task sessions). Wall names not in any stim_id mapping get a default code of 7.

ii.
```python
# nfr from neural file
_, n_neu_spk, nfr = load_spk_selected(key, None, None)
# Lick frames filtered
lf = np.round(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
# Unknown wall names
ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
```

iii. The AI noted this is a clean dataset with few edge cases.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (405 GB total). The AI uses multiprocessing (12 workers) to parallelize session processing, including spike loading.

ii.
```python
with Pool(min(args.nproc, len(jobs))) as pool:
    for i, res in enumerate(pool.imap_unordered(convert_session, jobs)):
        results.append(res)
```

iii. The AI documented timing per session and estimated total conversion time.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial grouping in `trial_frames` uses lexsort, np.diff and np.split, which is already fairly vectorized. The per-trial loop to build neural/input/output arrays is sequential but unavoidable since trials have variable length.

ii.
```python
for g, tid in zip(groups, trial_ids):
    T = len(g)
    neural_out.append(np.ascontiguousarray(spk[:, g]))
```

iii. The AI's approach is reasonably efficient, using numpy operations for frame selection and grouping.

## 12-c. What processing does the code repeat multiple times?

i. The spike file is loaded twice per session: once to get `nfr` (number of frames) and once to load the actual data. The speed computation also recomputes `trial_frames` for the global speed calculation before session conversion.

ii.
```python
# First load: just to get nfr
_, n_neu_spk, nfr = load_spk_selected(key, None, None)
# Second load: actual data
spk, _, _ = load_spk_selected(key, sel, frames)
```

iii. This doubles the I/O time for every session.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `nfr` for speed computation uses `len(beh['ft']) - 2` instead of the actual neural frame count from the spike file, which means the global speed thresholds are computed on slightly different frame sets than the final conversion. The AI also stores extensive metadata per session that may not be used by the decoder.

ii.
```python
speeds.append(session_speed_values(beh, len(beh['ft']) - 2))
```

iii. The discrepancy in frame count for speed threshold computation is minor (1-2 frames per session).
