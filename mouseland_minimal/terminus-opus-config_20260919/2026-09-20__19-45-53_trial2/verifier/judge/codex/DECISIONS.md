# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script uses `beh/Imaging_Exp_info.npy` as the master index, loops over every experiment type, loads the matching `Beh_<exp_type>.npy`, and builds one unique recording per `mname/datexp/blk` triple. It caches per-recording behavior fields in `recs`, merges experiment-type metadata and a wall-name-to-`stim_id` map across duplicate listings, and later loads the spike file and retinotopy file for each session during the main session loop.

ii. ```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
for exp_type, db in exp_info.items():
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=True).item()
    for ndb in db:
        rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        ...
        beh = Beh[key]
...
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                           % (r['mname'], r['datexp'])), allow_pickle=True)
```

iii. In trajectory steps 41 and 59, the agent explicitly said it would use all 89 recordings from `Imaging_Exp_info.npy`, keep duplicate listings as one recording, and then load per-session spike and retinotopy data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique mouse names `mname`, sorted once into `subjects`. Each session gets a `subject_idx` by looking up `r['mname']` in that sorted list.

ii. ```python
subjects = sorted(set(recs[r]['mname'] for r in rids))
...
data['subject_idx'].append(subjects.index(r['mname']))
```

iii. The trajectory summary in step 59 says there are 19 mice and that sessions are grouped by recording while `mname` identifies the mouse.

## 1-c. How are the data split into sessions?

i. A session is one unique recording id `rid = mname_datexp_blk`. Duplicate appearances of the same recording under multiple experiment types are merged into the same session record rather than emitted multiple times.

ii. ```python
rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
...
if rid not in recs:
    recs[rid] = dict(...)
...
rids = sorted(recs.keys(), key=lambda r: (recs[r]['mname'], recs[r]['datexp'], recs[r]['blk']))
```

iii. In steps 41 and 59, the agent described sessions as the 89 unique imaging recordings keyed by mouse, date, and block, with repeated experiment-type listings merged.

## 1-d. How are the data split into trials?

i. Trials are built by grouping imaging frames by `ft_trInd`, but only after first masking to frames that are both inside the textured corridor (`ft_CorrSpc`) and running (`ft_move > 0`). Each trial is therefore represented by a possibly non-contiguous subset of running-only corridor frames; trials with no kept frames are skipped later.

ii. ```python
def trial_frames(rec, nfr_spk):
    nfr = min(len(rec['ft']), nfr_spk)
    keep = rec['ft_CorrSpc'][:nfr] & (rec['ft_move'][:nfr] > 0)
    tr = rec['ft_trInd'][:nfr]
    idx = np.where(keep)[0]
    tr_idx = tr[idx]
    return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr
...
ii = frames[ti]
if len(ii) == 0:
    continue
```

iii. In steps 38, 41, and 59, the agent justified this as matching the paper's running-only analysis, saying it would extract 'running corridor frames per trial' using `ft_move > 0` and `ft_CorrSpc`.

## 1-e. How are trials filtered based on quality controls?

i. The script drops trials that have no retained running-corridor frames and drops trials whose wall name has no mapped stimulus id (`sid is None`, which the agent identified as `circle3` cases). It does not implement the reference solution's long-trial outlier filter.

ii. ```python
ii = frames[ti]
if len(ii) == 0:
    continue
sid = r['stim_map'].get(r['WallName'][ti], None)
if sid is None:                     # undefined stimulus category
    n_dropped_stim += 1
    continue
```

iii. In steps 32, 41, and 59, the agent said 309 `circle3` trials had no stimulus mapping and should be dropped, and it also treated empty running-only trials as unusable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the session spike file's `spks` arrays, concatenated across imaging planes, together with retinotopy `iarea` labels used to decide which neurons to keep and how to assign brain-region indices.

ii. ```python
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
...
iarea = np.asarray(ret['iarea'], dtype=float)
```

iii. In steps 20, 38, 41, and 59, the agent repeatedly summarized the neural source as deconvolved traces from `spk/*.npy` plus retinotopic area assignments from `iarea`.

## 2-b. How is the `neural` data processed?

i. After concatenating planes, the code restricts neurons to valid retinotopic groups, randomly subsamples up to 1000 neurons per session, converts to `float32`, z-scores each kept neuron over the full recording, drops zero-variance neurons, and finally slices each trial's frame indices into `X[:, ii]` arrays.

ii. ```python
valid = np.where(region_of_neuron >= 0)[0]
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))

X = spk[sel].astype(np.float32)
mu = X.mean(axis=1, keepdims=True)
sd = X.std(axis=1, keepdims=True)
good = (sd[:, 0] > 0)
X = (X[good] - mu[good]) / sd[good]
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. In steps 24, 41, and 59, the agent justified subsampling by the dataset's size (405 GB of spikes) and explicitly said it would z-score neurons over the whole recording and save 1000 random neurons per session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if their `iarea` belongs to one of four groups (`V1`, `mHV`, `lHV`, `aHV`). After subsampling, zero-variance neurons are also removed. No other explicit neuron-quality metric is used.

ii. ```python
region_of_neuron = np.full(nneu_all, -1, dtype=int)
for ri, reg in enumerate(BRAIN_REGIONS):
    m = np.isin(iarea, AREA_GROUPS[reg])
    region_of_neuron[m] = ri
valid = np.where(region_of_neuron >= 0)[0]
...
good = (sd[:, 0] > 0)
X = (X[good] - mu[good]) / sd[good]
```

iii. In steps 33, 41, and 59, the agent said it would keep only neurons assigned by retinotopy to the four visual area groups and drop unassigned cells; zero-variance removal follows from the z-scoring step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script states that trial alignment is corridor entry/trial start, but the actual neural arrays contain only the running frames inside the textured corridor for that trial. So alignment is implicit through `Trial_start_time`/corridor-entry semantics while the stored neural samples omit stopped periods.

ii. ```python
* Trial             : one corridor traversal, aligned to corridor entry
                      (beh['StartFr'] / beh['Trial_start_time']), ending when the animal
                      leaves the 4 m texture area (beh['ft_CorrSpc']).
...
keep = rec['ft_CorrSpc'][:nfr] & (rec['ft_move'][:nfr] > 0)
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. In steps 41 and 59, the agent described trials as corridor traversals aligned to entry but also emphasized that only running timepoints (`ft_move > 0`) would be kept because it believed this matched the paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native imaging-frame resolution. No temporal rebinning is applied. The metadata reports the mean session frame interval in milliseconds.

ii. ```python
* Timepoints        : the native imaging frames (no re-binning)
...
dts.append(np.median(np.diff(r['ft'][:nfr])) * 86400.)
...
time_bin_size=float(np.mean(dts) * 1000.)
```

iii. In steps 38, 41, and 59, the agent said the frame rate was about 3.18 Hz (`dt = 0.3147 s`) and that it would keep the native frames with no rebinning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the session frame timestamps `ft` and the per-trial cue timestamp `SoundTime`.

ii. ```python
SoundTime=np.asarray(beh['SoundTime'], dtype=np.float64)
...
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
```

iii. In steps 15, 16, 38, and 59, the agent identified `SoundTime` as an available behavioral variable and chose to compute cue timing directly from timestamps rather than from `SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame in a trial, the code subtracts the frame timestamp from the trial's `SoundTime` and converts days to seconds by multiplying by `86400`. The resulting value is positive before the cue and negative after it.

ii. ```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
inp = np.stack([t_cue,
                np.full(len(ii), r['day']),
                t,
                np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
```

iii. The trajectory summary in step 59 explicitly describes this input as 'signed time to sound cue'.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same frame indices `ii` that are used to slice the neural matrix for that trial.

ii. ```python
ii = frames[ti]
...
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. In step 59, the agent described the inputs and neural data as being defined on the same running corridor frames per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each recording's mouse name `mname` and date string `datexp`, which is parsed into a `datetime.date`.

ii. ```python
rids = sorted(recs.keys(), key=lambda r: (recs[r]['mname'], recs[r]['datexp'], recs[r]['blk']))
...
d = datetime.date(*[int(x) for x in r['datexp'].split('_')])
...
r['day'] = float((r['date'] - first_day[r['mname']]).days)
```

iii. In steps 41 and 59, the agent said it would encode day of training as 'days since that mouse's first recording'.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code finds the earliest recording date for each mouse, computes the elapsed calendar-day difference for each session, stores that as `r['day']`, and then broadcasts it across all kept frames of every trial in the session.

ii. ```python
first_day = {}
for rid in rids:
    ...
    if r['mname'] not in first_day or d < first_day[r['mname']]:
        first_day[r['mname']] = d
for rid in rids:
    r = recs[rid]
    r['day'] = float((r['date'] - first_day[r['mname']]).days)
...
np.full(len(ii), r['day'])
```

iii. The agent justified this in steps 41 and 59 as an interpretable continuous training-progress variable based on elapsed days from the first recording.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the frame timestamps `ft` and the per-trial start timestamp `Trial_start_time`.

ii. ```python
Trial_start_time=np.asarray(beh['Trial_start_time'], dtype=np.float64)
...
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
```

iii. In steps 38, 41, and 59, the agent described trial start as corridor entry and used `Trial_start_time` for that reference point.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each kept frame, the code subtracts `Trial_start_time[ti]` from the frame timestamp and converts the result from days to seconds.

ii. ```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
inp = np.stack([t_cue,
                np.full(len(ii), r['day']),
                t,
                np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
```

iii. The trajectory summaries in steps 41 and 59 call this input 'time since trial start' and tie trial start to corridor entry.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same per-trial frame index array `ii` as the neural data.

ii. ```python
ii = frames[ti]
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The agent's step-59 summary presents all decoder inputs as frame-aligned to the stored neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial boolean `isRew`.

ii. ```python
isRew=np.asarray(beh['isRew']).astype(bool)
...
np.full(len(ii), float(r['isRew'][ti]))
```

iii. In steps 40 and 59, the agent described reward availability as whether the corridor is the rewarded one.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond boolean-to-float casting and broadcasting across the kept frames of a trial.

ii. ```python
inp = np.stack([t_cue,
                np.full(len(ii), r['day']),
                t,
                np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
```

iii. The trajectory summaries treat `isRew` as a direct per-trial input and do not describe any additional processing.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The code derives it from the trial's `WallName`, using a per-recording `stim_map` built from `UniqWalls` and `stim_id` values seen in the behavior files.

ii. ```python
for wall, sid in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], dtype=float)):
    if not np.isnan(sid):
        r['stim_map'][str(wall)] = int(sid)
...
sid = r['stim_map'].get(r['WallName'][ti], None)
```

iii. In steps 28, 32, 41, and 59, the agent reasoned about wall-name-to-stimulus-id mappings and decided to use those ids as the category labels, while dropping unmapped `circle3` trials.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script defines seven output labels in `STIM_NAMES`, looks up the trial's `sid` from `stim_map`, broadcasts that integer across the retained frames of the trial, and drops the trial if no mapping exists. It does not collapse textures into four broad categories.

ii. ```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
...
output_values=[STIM_NAMES, ...]
...
sid = r['stim_map'].get(r['WallName'][ti], None)
if sid is None:
    n_dropped_stim += 1
    continue
...
out = np.stack([np.full(len(ii), sid),
                lick[ii].astype(int),
                pos_bin,
                spd_bin]).astype(np.int64)
```

iii. In step 59, the agent explicitly summarized this output as a 7-category stimulus label and justified dropping `circle3` because it considered that stimulus undefined in its mapping.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`, the list of lick frame indices for the session.

ii. ```python
LickFr=np.asarray(beh['LickFr'], dtype=np.float64)
...
lf = np.round(r['LickFr']).astype(int)
```

iii. In steps 15, 16, 40, and 59, the agent identified `LickFr` as the lick source and described output licking as binary per frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code rounds lick frame indices to the nearest integer, keeps only indices in `[0, nfr)`, writes them into a boolean per-frame raster, and then converts the selected trial frames to integers for the output array.

ii. ```python
lick = np.zeros(nfr, dtype=bool)
lf = np.round(r['LickFr']).astype(int)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
...
lick[ii].astype(int)
```

iii. In step 59, the agent summarized this as 'licking (binary per frame from `LickFr`)', and in step 40 it checked how many licks survived its running-only frame filter.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The per-frame lick raster is indexed by the same trial-frame indices `ii` used for the neural data.

ii. ```python
out = np.stack([np.full(len(ii), sid),
                lick[ii].astype(int),
                pos_bin,
                spd_bin]).astype(np.int64)
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The agent's trajectory consistently described licking as a framewise output defined on the same stored trial frames as the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position comes from the per-frame corridor-position variable `ft_Pos`.

ii. ```python
ft_Pos=np.asarray(beh['ft_Pos'], dtype=np.float64)
...
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
```

iii. In steps 20, 38, and 59, the agent described `ft_Pos` as the VR corridor position used to form the position output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code bins `ft_Pos` with edges at 10, 20, and 30 decimeters and clips the result into four categories.

ii. ```python
POS_BIN_EDGES = [10., 20., 30.]
...
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. In step 59, the agent summarized this as 'position in 4 x 1 m bins'.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are 10, 20, and 30 dm, producing bins labeled `0-1m`, `1-2m`, `2-3m`, and `3-4m`.

ii. ```python
POS_BIN_EDGES = [10., 20., 30.]
...
output_values=[STIM_NAMES,
               ['no_lick', 'lick'],
               ['0-1m', '1-2m', '2-3m', '3-4m'],
               ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']]
```

iii. The agent states this discretization directly in the code comments and in the step-59 summary.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is evaluated on the same per-trial frame indices `ii` used to slice the neural data.

ii. ```python
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
...
out = np.stack([np.full(len(ii), sid),
                lick[ii].astype(int),
                pos_bin,
                spd_bin]).astype(np.int64)
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The trajectory presents position as one of the time-varying outputs on the stored trial frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the per-frame variable `ft_RunSpeed`.

ii. ```python
ft_RunSpeed=np.asarray(beh['ft_RunSpeed'], dtype=np.float64)
...
speeds.append(r['ft_RunSpeed'][ii])
...
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. In steps 39, 41, and 59, the agent discussed the running-speed distribution and chose to discretize `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before processing sessions, the script pools all retained running-corridor speeds across all sessions, computes global 25th/50th/75th percentile thresholds, and then uses those global thresholds to bin each frame's running speed.

ii. ```python
speeds = []
for rid in rids:
    r = recs[rid]
    frames, _ = trial_frames(r, len(r['ft']))
    ii = np.concatenate([f for f in frames if len(f)])
    speeds.append(r['ft_RunSpeed'][ii])
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
```

iii. In steps 39, 41, 44, and 59, the agent explicitly referred to 'global running speed quartiles' and 'global quartile bins'.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The global thresholds in `speed_edges` are passed to `np.digitize`, yielding four categories labeled `speed_q1` through `speed_q4`.

ii. ```python
speed_edges = np.percentile(speeds, [25, 50, 75])
...
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
...
['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']
```

iii. The step-59 summary states that running speed is binned into global quartile bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is computed on the same retained frame indices `ii` used for neural slicing.

ii. ```python
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
...
out = np.stack([np.full(len(ii), sid),
                lick[ii].astype(int),
                pos_bin,
                spd_bin]).astype(np.int64)
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The agent treated running speed as a framewise output aligned to the stored neural frames in steps 41 and 59.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates behavior-derived framewise streams to the imaged frame count via `nfr = min(len(ft), nfr_spk)`, drops lick indices outside the imaging range, skips trials that end up with no retained frames, and drops trials whose stimulus category cannot be mapped. There is no broader missing-data imputation or exception handling inside the converter.

ii. ```python
nfr = min(len(rec['ft']), nfr_spk)
keep = rec['ft_CorrSpc'][:nfr] & (rec['ft_move'][:nfr] > 0)
...
lf = np.round(r['LickFr']).astype(int)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
...
if len(ii) == 0:
    continue
...
if sid is None:
    n_dropped_stim += 1
    continue
```

iii. In step 59, the agent highlighted dropping the unmapped `circle3` trials; the frame-count truncation and valid-index checks are the code's main safeguards against minor data mismatches.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading every large spike file, concatenating planes, computing full-recording z-scores for the sampled neurons, and making the global all-session pass to collect running speeds. Behavior loading is smaller but still touches every behavior file.

ii. ```python
Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
              allow_pickle=True).item()
...
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
...
mu = X.mean(axis=1, keepdims=True)
sd = X.std(axis=1, keepdims=True)
...
for rid in rids:
    ...
    speeds.append(r['ft_RunSpeed'][ii])
```

iii. In step 24, the agent explicitly noted that the spike files total about 405 GB and based several later decisions on that cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the Python-level grouping of frame indices into one list per trial, the all-session loop used only to collect global speed thresholds, the loop assigning region ids by area group, and the per-trial assembly loop that repeatedly stacks arrays and appends them to lists.

ii. ```python
return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr
...
for rid in rids:
    ...
    speeds.append(r['ft_RunSpeed'][ii])
...
for ri, reg in enumerate(BRAIN_REGIONS):
    m = np.isin(iarea, AREA_GROUPS[reg])
    region_of_neuron[m] = ri
...
for ti in range(r['ntrials']):
    ii = frames[ti]
    ...
    input_s.append(inp)
    output_s.append(out)
```

iii. The trajectory does not call these out explicitly, but the code structure makes those vectorization opportunities clear.

## 12-c. What processing does the code repeat multiple times?

i. It computes `trial_frames(...)` twice conceptually: once in the global pre-pass that builds `speed_edges`, and again when each session is processed for output generation. It also performs repeated per-session linear `subjects.index(...)` lookups and carries duplicate experiment-type bookkeeping even though the exported arrays are session-based.

ii. ```python
for rid in rids:
    r = recs[rid]
    frames, _ = trial_frames(r, len(r['ft']))
    ...
...
for si, rid in enumerate(rids):
    ...
    frames, nfr = trial_frames(r, nfr_spk)
...
data['subject_idx'].append(subjects.index(r['mname']))
```

iii. This repeated work follows from the agent's global-speed-binning design described in steps 41, 44, and 59.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The decoder only needs the converted arrays plus minimal metadata, but the script also accumulates `exp_types`, `cohorts`, detailed `session_info`, mean-frame-interval bookkeeping, and several descriptive metadata fields that are not used by downstream decoding. `CORRIDOR_TEXTURE_LEN` is defined but never used. The global speed-threshold pass is also extra work relative to the reference solution because it exists only to support the AI's custom speed discretization choice.

ii. ```python
recs[rid] = dict(
    rid=rid, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'],
    exp_types=[], stim_map={}, cohorts=set(),
    ...)
...
session_info = []
dts = []
...
CORRIDOR_TEXTURE_LEN = 40.
...
data['metadata'] = dict(
    ...
    speed_bin_edges_cm_s=[float(x) for x in speed_edges],
    position_bin_edges_m=[1.0, 2.0, 3.0],
    licking_note=(...),
    n_trials_dropped_unknown_stimulus=int(n_dropped_stim),
    session_info=session_info,
    source='Zhong et al. 2025, Unsupervised pretraining in biological neural networks',
)
```

iii. The agent's trajectory focused on making the output decodable and interpretable, but several of these bookkeeping fields and passes are not consumed by `train_decoder.py`.
