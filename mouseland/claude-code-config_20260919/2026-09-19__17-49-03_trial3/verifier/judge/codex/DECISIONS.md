# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset from `/app/data/beh`, `/app/data/spk`, and `/app/data/retinotopy`. It first reads `Imaging_Exp_info.npy`, de-duplicates `(mouse, date, block)` recordings into 89 sessions, then reads each `Beh_<exp_type>.npy` once and loads each session's spike planes and retinotopy file separately. It also merges canonical stimulus ids across all experiment-type entries that reference the same recording.

ii. 
```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
```
```python
B = np.load(os.path.join(BEH_DIR, 'Beh_' + exp_type + '.npy'), allow_pickle=True).item()
return np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']
d = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (mname, datexp)), allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as reconciling the 142 experiment-type entries with the paper's 89 unique recordings, and as the only way to recover the merged canonical stimulus-role map for each session.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `mname` from `Imaging_Exp_info.npy`. The final `subjects` list is the sorted set of mouse names, and session-to-subject assignment is stored by indexing each recording's `mname` into that list.

ii. 
```python
recs[key] = {
    'key': key, 'mname': db['mname'], 'datexp': db['datexp'],
    'blk': db['blk'], 'exp_types': [], 'beh_source': (exp_type, behkey),
    'stim_map': {}, 'exptype': db.get('exptype', None),
}
```
```python
subjects = sorted({r['mname'] for r in recs})
'subject_idx': np.array([subjects.index(r['mname']) for r in recs], dtype=np.int64),
```

iii. The AI's notes treat mouse identity as directly given by the reference index, so no extra inference is needed beyond carrying `mname` through the pipeline.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` recording. The AI de-duplicates repeated appearances of the same recording across experiment types, but retains all experiment-type metadata on that session so it can merge stimulus-role information later.

ii. 
```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
if key not in recs:
    recs[key] = {
        'key': key, 'mname': db['mname'], 'datexp': db['datexp'],
        'blk': db['blk'], 'exp_types': [], 'beh_source': (exp_type, behkey),
        'stim_map': {}, 'exptype': db.get('exptype', None),
    }
```
```python
r['_db_list'] = r.get('_db_list', []) + [(exp_type, behkey, db)]
order = sorted(recs.values(), key=lambda r: (r['mname'], r['datexp'], r['blk']))
```

iii. The notes explicitly justify de-duplication because the same recording appears in multiple behavioral groupings, while the merged `_db_list` is needed to recover all canonical stimulus ids for that recording.

## 1-d. How are the data split into trials?

i. Trials are not taken from full corridor traversals. Instead, the AI first filters to retained frame-level timepoints with `frame_mask(...)`, then groups the retained frames by `ft_trInd`; trial boundaries are the change points in the retained `ft_trInd` vector. This means each trial contains only the kept running/corridor frames, not all corridor frames.

ii. 
```python
keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
idx = np.flatnonzero(keep)
tri = b['ft_trInd'][idx].astype(np.int64)
```
```python
bounds = np.flatnonzero(np.diff(tri)) + 1
starts = np.concatenate(([0], bounds))
stops = np.concatenate((bounds, [len(tri)]))
trial_ids = tri[starts]
```

iii. The notes justify this by claiming the retained timepoints should follow the paper's `fr_valid = (ft_move > 0) & ft_CorrSpc` analysis mask, so the trial representation should live on that reduced frame grid rather than on the full traversal.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not drop whole trials for duration or emptiness. It keeps every trial that still has at least one retained frame after frame-level filtering, and only removes 21 ambiguous boundary frames where position wraps to the next corridor before `ft_trInd` changes.

ii. 
```python
keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
wrap = np.flatnonzero((np.diff(pos) < 0) & (np.diff(tri) == 0)) + 1
if len(wrap):
    keep[idx[wrap]] = False
```
```python
trial_ids = tri[starts]
for s, e, t in zip(starts, stops, trial_ids):
    ...
```

iii. `CONVERSION_NOTES.md` explicitly says "No trials are dropped" and justifies that choice by arguing the reference analyses are timepoint-filtered rather than trial-filtered, while the only removed samples are the 21 label-ambiguous boundary frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices come from the `spks` arrays in each session's `*_neural_data.npy`, together with `iarea` from the matching retinotopy file to determine region membership and neuron selection.

ii. 
```python
return np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']
```
```python
d = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (mname, datexp)), allow_pickle=True)
return d['iarea']
```

iii. The notes say this follows the paper's raw deconvolved neural representation directly: no dF/F or deconvolution is recomputed.

## 2-b. How is the `neural` data processed?

i. The AI concatenates selected neurons across imaging planes, restricts columns to retained running/corridor frames, subsamples to at most 2000 visual-cortex neurons per session, z-scores each kept neuron across retained session timepoints, and then slices the session matrix into per-trial matrices. Trials remain variable length.

ii. 
```python
blocks.append(p[sel][:, keep_idx])
X = np.concatenate(blocks, axis=0).astype(np.float32) if blocks else \
    np.zeros((0, len(keep_idx)), np.float32)
```
```python
mu = X.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
sd = X.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
sd[sd == 0] = 1.0
X -= mu
X /= sd
```
```python
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. The notes justify neuron subsampling as a runtime/memory concession tied to the decoder's `svd_max_neurons=2000`, and justify z-scoring as a decoder-conditioning step because raw deconvolved amplitudes have highly heterogeneous scales.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only neurons assigned to one of four visual regions, then performs an additional proportional random subsample over those regions to cap each session at 2000 neurons. It does not drop zero-variance neurons, but it avoids division by zero by setting zero standard deviations to 1 before z-scoring.

ii. 
```python
per_area = [np.flatnonzero(area_idx[ar]) for ar in BRAIN_REGIONS]
if n_vis <= n_keep:
    chosen = [a for a in per_area]
else:
    ...
    chosen = [rng.choice(a, size=k, replace=False) if k < len(a) else a
              for a, k in zip(per_area, take)]
```
```python
n_dead = int((sd[:, 0] == 0).sum())
sd[sd == 0] = 1.0
```

iii. The notes explicitly justify the visual-area restriction from the reference code, and justify the extra 2000-neuron cap as necessary for tractable decoding while preserving region composition.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to corridor entry conceptually, but the stored per-trial neural data starts at the first retained frame after applying the running/corridor mask, not at every raw frame from entry onward. Neural and behavioral arrays for a trial are all sliced from the same retained-frame spans.

ii. 
```python
'temporal_alignment_event':
    'trial start = entry into the virtual-reality corridor (beh["Trial_start_time"], '
    'the frame at which VR position resets to 0)',
```
```python
keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
...
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. The AI's stated rationale is that the reference paper's analyses only keep timepoints inside the 0-4 m corridor while the VR is moving, so the decoder representation should be aligned on that same retained frame grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps native imaging frames as bins. It computes a dataset-level median frame rate from `ft` and stores the corresponding median bin width in milliseconds; no temporal rebinning is applied.

ii. 
```python
fs_all = np.array([s['frame_rate_hz'] for s in session_info])
bin_ms = float(1000.0 / np.median(fs_all))
```
```python
'time_bin_size': bin_ms,
'frame_rate_hz': float(np.median(fs_all)),
```

iii. The notes justify this as preserving the native time grid because the decoder output includes position, so the paper's position-rebinning analyses are not directly reusable.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from the retained frame timestamps `ft` and per-trial `SoundTime`, not from `SoundFr`.

ii. 
```python
ft = b['ft'][idx]
...
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
```

iii. The justification in the notes is that wall-clock times are the relevant continuous time variables for the decoder, and that using `SoundTime` directly gives the cue timing on that same clock.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the AI computes seconds until cue as `SoundTime[trial] - ft[frame]`, then clips the result to `[-30, 30]` seconds.

ii. 
```python
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
t_to_cue = np.clip(t_to_cue, -TIME_CLIP_S, TIME_CLIP_S)
```
```python
inp[0] = t_to_cue[s:e]
```

iii. The notes explicitly justify clipping because some animals pause for tens or hundreds of seconds, and the AI wanted to keep the decoder numerically well conditioned without discarding those trials.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same retained frame indices that define the per-trial neural matrix, so each trial's `time_to_sound_cue` vector has the same length as the neural time axis.

ii. 
```python
idx = np.flatnonzero(keep)
ft = b['ft'][idx]
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
```
```python
inp[0] = t_to_cue[s:e]
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. The AI's notes describe all streams as sharing the same retained imaging-frame grid by construction.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives this from each session's `datexp` field grouped by mouse, not from session count alone. It computes the calendar-day difference from that mouse's first recording date.

ii. 
```python
for mname, rs in by_mouse.items():
    d0 = min(datetime.date(*map(int, r['datexp'].split('_'))) for r in rs)
    for r in rs:
        r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days
```

iii. The only explicit justification is in the metadata/notes wording: the input is described as "days elapsed since this mouse's first recording session."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI converts date strings to `datetime.date`, subtracts the first date seen for that mouse, stores the resulting integer day offset per session, and then broadcasts that scalar across every time bin in each trial.

ii. 
```python
r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days
```
```python
inp[1] = rec['day']
```

iii. The notes frame this as a continuous day variable rather than a session ordinal, with no further processing beyond broadcast across time bins.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives this from per-trial `Trial_start_time` and retained frame timestamps `ft`, not from `StartFr`.

ii. 
```python
ft = b['ft'][idx]
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
```

iii. The notes justify wall-clock time as the intended notion of elapsed time since corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the AI computes elapsed seconds since `Trial_start_time` and clips the result to `[0, 30]` seconds.

ii. 
```python
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
t_since_start = np.clip(t_since_start, 0.0, TIME_CLIP_S)
```
```python
inp[2] = t_since_start[s:e]
```

iii. The notes use the same clipping rationale as for cue time: some retained frames come from very long pauses, and clipping avoids ill-conditioned large time values while preserving samples.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated on the exact same retained frame indices and then sliced with the same trial boundaries as the neural matrices.

ii. 
```python
idx = np.flatnonzero(keep)
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
```
```python
inp[2] = t_since_start[s:e]
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. The notes state that neural, inputs, and outputs all share the retained imaging-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The AI derives reward availability from `WallName` together with `isRew`: it first identifies the rewarded corridor type from a rewarded trial, then labels every trial in that corridor as reward-available. If a session has no rewarded trials, all trials are labeled unavailable.

ii. 
```python
wall = b['WallName']
if b['isRew'].any():
    rew_stim = wall[b['isRew']][0]
    rew_avail = (wall == rew_stim).astype(np.float32)
else:
    rew_avail = np.zeros(ntr, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as avoiding leakage: the AI concluded `isRew` means reward delivered, not merely rewarded corridor, so using `isRew` directly would leak the licking output into the input.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the rewarded corridor identity into a per-trial binary flag and broadcasts it across all time bins of that trial.

ii. 
```python
rew_avail = (wall == rew_stim).astype(np.float32)
...
inp[3] = rew_avail[t]
```

iii. The notes say this is a deliberate deviation from the naive `isRew` interpretation, motivated by semantic correctness and leakage avoidance.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives the stimulus output from `WallName` plus the merged canonical `stim_id` map recovered from `Imaging_Exp_info.npy`. Unmapped `circle3` is given a hard-coded extra id.

ii. 
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2', 'circle3']
STIM_EXTRA = {'circle3': 7}
```
```python
stim_id = np.array([rec['stim_map'].get(w, STIM_EXTRA.get(w, -1)) for w in wall],
                   dtype=np.int64)
```

iii. The notes justify this by saying the reference code uses canonical stimulus roles across mice and experiment types, so rock/brick mice should be pooled onto those roles rather than using coarse raw texture names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each trial's wall name to an 8-class canonical role id, then broadcasts that single class label across all time bins of the trial.

ii. 
```python
out = np.empty((4, T), dtype=np.int64)
out[0] = stim_id[t]
```
```python
'output_values': [
    list(STIM_NAMES),
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4'],
],
```

iii. The notes explicitly say this pools stimuli by canonical role across mice and preserves distinctions like `leaf1` vs `leaf2` instead of collapsing to four texture super-categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`, interpreted as lick event times in imaging-frame units.

ii. 
```python
lick_fr = b['LickFr']
lick_fr = lick_fr[np.isfinite(lick_fr)]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
```

iii. The notes say this follows the reference event-frame convention of flooring `LickFr` to the containing frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI drops non-finite and out-of-range lick frames, floors the remaining frame indices to integers, creates a boolean lick raster over session frames, and converts it to integer labels on the retained-frame grid.

ii. 
```python
lick_frame = np.zeros(nfr, dtype=bool)
lick_frame[lick_fr] = True
licking = lick_frame[idx].astype(np.int64)
```

iii. The justification is partly explicit and partly implicit: the notes reference the paper code's `.astype(int)` convention, while the finite/range filtering is a defensive cleanup for minor data issues.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by indexing the lick raster with the same retained frame indices and trial slices used for neural data.

ii. 
```python
licking = lick_frame[idx].astype(np.int64)
...
out[1] = licking[s:e]
```
```python
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. The notes describe this as all variables living on the same retained imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos` on the retained frame grid.

ii. 
```python
pos = b['ft_Pos'][idx]
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. The AI's notes justify `ft_Pos` as the direct per-frame corridor-position signal already used by the reference analyses.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI converts position from decimeters into four equal 1 m bins over the 0-4 m textured corridor by dividing by `POS_BIN_DM = 10`.

ii. 
```python
CORRIDOR_DM = 40.0
N_POS_BINS = 4
POS_BIN_DM = CORRIDOR_DM / N_POS_BINS
```
```python
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. The notes say this matches the decoder specification of four equal-length 1 m bins inside the 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are fixed at 0, 10, 20, 30, and 40 decimeters, corresponding to 0-1 m, 1-2 m, 2-3 m, and 3-4 m.

ii. 
```python
N_POS_BINS = 4
POS_BIN_DM = CORRIDOR_DM / N_POS_BINS
```
```python
'output_values': [
    list(STIM_NAMES),
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4'],
],
```

iii. The AI's rationale is just the decoder spec: equal-length 1 m spatial bins across the textured 4 m corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is evaluated on the retained frame indices and sliced into the same trial segments as the neural data.

ii. 
```python
pos = b['ft_Pos'][idx]
...
out[2] = pos_bin[s:e]
```
```python
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. The notes describe neural and position outputs as sharing the retained frame grid by construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed` at the retained frame indices.

ii. 
```python
speed = b['ft_RunSpeed'][idx]
speed_bin = np.searchsorted(speed_edges, speed, side='right').astype(np.int64)
```

iii. The notes identify `ft_RunSpeed` as the per-imaging-frame running-speed signal in centimeters per second.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI pools all retained running-speed samples across sessions, computes global 25th/50th/75th percentile edges, and bins each retained frame by thresholding against those edges.

ii. 
```python
all_speed = np.concatenate([r['beh']['ft_RunSpeed'][:r['nfr']][r['keep']] for r in recs])
speed_edges = np.percentile(all_speed, [25, 50, 75])
```
```python
speed_bin = np.searchsorted(speed_edges, speed, side='right').astype(np.int64)
speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this by saying the task specification asks for bins corresponding to 25% of the data, which is exactly satisfied only by pooled global quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The categories are defined by the three global percentile thresholds stored in `speed_edges`, with class ids 0-3 corresponding to the four quartile bins.

ii. 
```python
speed_edges = np.percentile(all_speed, [25, 50, 75])
```
```python
'running_speed_bin_edges_cm_s': [float(x) for x in speed_edges],
'running_speed_bin': 'running speed quartile, bin edges from the pooled '
                     'distribution over every retained timepoint',
```

iii. The justification is the same global-quartile argument from the notes.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is evaluated on the retained frame indices and split with the same trial boundaries as the neural data.

ii. 
```python
speed = b['ft_RunSpeed'][idx]
...
out[3] = speed_bin[s:e]
```
```python
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. The notes treat this as another variable on the shared retained imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI adds several defensive checks: it uses spike-file frame counts rather than assuming `len(ft)-1`; drops frames with `NaN` trial indices; removes 21 ambiguous corridor-boundary frames; filters non-finite or out-of-range lick events; and guards zero-variance neurons during z-scoring.

ii. 
```python
keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
...
if len(wrap):
    keep[idx[wrap]] = False
```
```python
lick_fr = lick_fr[np.isfinite(lick_fr)]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
```
```python
sd = X.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
sd[sd == 0] = 1.0
```

iii. The notes explicitly justify these as bug fixes found during validation: mismatched frame counts, ambiguous wrapped frames, and numerical edge cases that could silently misalign or destabilize the decoder inputs.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant cost is reading and processing the 405 GB spike files. The AI parallelizes this with a `ProcessPoolExecutor`, and the notes say this is the only real bottleneck.

ii. 
```python
with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tasks))) as ex:
    for i, res in enumerate(ex.map(process_neural, tasks)):
        ...
```
```python
planes = load_spk_planes(task['mname'], task['datexp'], task['blk'])
...
blocks.append(p[sel][:, keep_idx])
```

iii. `CONVERSION_NOTES.md` explicitly says reading the spike files is the only real cost and documents the measured speedup from six parallel readers.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Most hot-path numerical work is already vectorized, but the code still uses Python loops over sessions, experiment-type entries, imaging planes, and ragged trials. The most avoidable repeated looping is the second pass over sessions to recompute frame masks and the per-trial Python loop that packages ragged trial arrays.

ii. 
```python
for r in recs:
    r['nfr'] = results[r['key']]['nfr']
    r['keep'] = compute_frame_selection(r, r['nfr'])
```
```python
for s, e, t in zip(starts, stops, trial_ids):
    ...
    inputs.append(inp)
    outputs.append(out)
    slices.append((s, e))
```

iii. The notes emphasize that I/O dominates and that major vectorization opportunities were already addressed earlier, so the remaining loops are mostly around ragged trial packaging and bookkeeping rather than large numeric bottlenecks.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats frame-mask computation: once inside each neural worker to build the reduced session matrix, and again in the main process to validate worker results and compute global speed quartiles. It also effectively reuses the same retained-frame grouping logic both to slice neural data and to build trial inputs/outputs.

ii. 
```python
keep = task['keep_fn'](nfr)
keep_idx = np.flatnonzero(keep)
```
```python
for r in recs:
    r['nfr'] = results[r['key']]['nfr']
    r['keep'] = compute_frame_selection(r, r['nfr'])
    assert r['keep'].sum() == results[r['key']]['X'].shape[1]
```

iii. The notes explicitly mention the cross-process mask assertion as a safeguard after a bug with frame counts, so this duplicated work is deliberate validation overhead rather than an accident.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some diagnostics and bookkeeping that are not needed by the final decoder dataset: `raw_stats` are returned from workers but never used downstream, `exp_info` is loaded and returned to `main()` but then ignored, and full `diag` structures are built and attached to every session even when `--show-processing` is off and only a small part of them is later used.

ii. 
```python
raw_stats = {'mean': float(X.mean()) if X.size else 0.0,
             'max': float(X.max()) if X.size else 0.0}
...
'raw_stats': raw_stats,
```
```python
recs, exp_info = build_recording_table()
```
```python
diag = {'idx': idx, 'tri': tri, 'pos': pos, 'speed': speed, 'licking': licking,
        't_since_start': t_since_start, 't_to_cue': t_to_cue,
        'pos_bin': pos_bin, 'speed_bin': speed_bin, 'trial_ids': trial_ids,
        'starts': starts, 'stops': stops, 'stim_id': stim_id, 'rew_avail': rew_avail}
```

iii. The notes frame most of this as validation/diagnostic support. It was useful during conversion review, but not necessary for downstream decoding once the final pickle is written.
