# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` for behavior, `spk/` for deconvolved neural traces, and `retinotopy/` for visual area assignments. It first reads `beh/Imaging_Exp_info.npy` as a master index of all recordings grouped by experiment type. For each experiment type, it loads the corresponding `Beh_<exp_type>.npy` file and iterates over entries. A recording key is `mname_datexp_blk`. The AI merges behavior data across experiment types, reading behavior arrays once per recording and merging `stim_id` across all experiment types to fully label every trial. Neural data is loaded per-session from `spk/<key>_neural_data.npy`, and retinotopy from `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=1).item()
for exp_type, db in exp_info.items():
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=1).item()
    for d in db:
        key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
        behkey = key + ('_%s' % d['stimtype'] if 'stimtype' in d else '')
        beh = Beh[behkey]
        rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})
        ...
```

iii. The AI reasoned that a recording can appear in several experiment types and that behavior arrays are identical across copies, differing only in `stim_id`. It therefore reads behavior once per recording and merges `stim_id` over all experiment types so every stimulus the paper labels gets its canonical category.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info entries. The AI builds a list of unique subjects in the order they are encountered, and assigns each session a `subject_idx` by looking up the mouse name in that list. Result: 19 mice.

ii.
```python
if rec['mname'] not in subjects:
    subjects.append(rec['mname'])
subject_idx.append(subjects.index(rec['mname']))
```

iii. Mouse names are directly available from the data entries; no additional derivation needed.

## 1-c. How are the data split into sessions?

i. A session is one recording, identified by the triple `mname_datexp_blk`. When the same recording appears under multiple experiment types, behavior data is merged (keeping one copy of the arrays, merging `stim_id`). Sessions are sorted by (mname, datexp, blk). Result: 89 sessions.

ii.
```python
key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})
```

iii. The AI noted recordings can appear under multiple experiment types and merged them into a single entry per recording key.

## 1-d. How are the data split into trials?

i. Trials are individual corridor traversals identified by `ft_trInd`. The AI selects frames that satisfy three conditions: (a) assigned to a trial (`~np.isnan(trind)`), (b) inside the textured corridor (`ft_CorrSpc`), and (c) the mouse was running (`ft_move > 0`). Frames meeting all criteria are grouped by trial index.

ii.
```python
def add_frame_selection(rec, nframes):
    n = min(nframes, len(rec['ft']))
    trind = rec['ft_trInd'][:n]
    keep = (rec['ft_CorrSpc'][:n] & (rec['ft_move'][:n] > 0) & ~np.isnan(trind))
    frames = np.nonzero(keep)[0]
    return frames, trind[frames].astype(int)
```

iii. The AI justified the running filter with a quote from the paper: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The AI noted this reduced trial lengths from 11-600+ frames to 11-30 frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two trial-level filters: (1) trials with an unlabelled stimulus (where `stim_id` maps to -1, e.g. "circle3" that the paper never assigns a category to) are dropped -- 309 trials total. (2) Trials with zero valid frames after the frame-level filtering (CorrSpc + running) are skipped. The AI does NOT apply any trial length outlier filtering (no 99th percentile cutoff).

ii.
```python
for t in range(rec['ntrials']):
    if stim_of_trial[t] < 0:
        n_dropped_stim += 1
        continue
    f = frames[trials == t]
    if len(f) == 0:
        continue
```

iii. The AI reasoned that circle3 trials "cannot be labelled on the common scale and are dropped." The running filter already eliminates the long-idle trials, so the AI did not see a need for additional length-based filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains per-plane arrays of deconvolved calcium traces. The visual area comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
...
iarea = np.load(fn, allow_pickle=True)['iarea']
```

iii. The AI identified these as the deconvolved two-photon calcium fluorescence signals from Suite2p.

## 2-b. How is the `neural` data processed?

i. The AI selects neurons belonging to V1/mHV/lHV/aHV, then randomly subsamples to at most 2000 neurons per session using a fixed seed. Selected neurons are loaded plane-by-plane to avoid materializing the full matrix. Neural values are stored as float32 (not float16). No normalization is applied.

ii.
```python
NEURONS_PER_SESSION = 2000
...
valid = np.nonzero(area_idx >= 0)[0]
if len(valid) > NEURONS_PER_SESSION:
    valid = np.sort(rng.choice(valid, NEURONS_PER_SESSION, replace=False))
...
out = np.empty((len(sel), nframes), dtype=np.float32)
```

iii. The AI reasoned: "The recordings contain 20,547-89,577 neurons each; keeping all of them would make the dataset several hundred GB. 2000 is also the largest number of neurons for which the reference decoder computes an exact SVD initialisation of the per-session projection (decoder.svd_max_neurons), so nothing is gained by keeping more."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two ways: (1) only neurons assigned to V1, mHV, lHV, or aHV by the retinotopic map are kept (neurons with iarea -1 or 7 are dropped); (2) the remaining neurons are randomly subsampled to at most 2000 per session.

ii.
```python
AREA_CODES = {'V1': (8,), 'mHV': (0, 1, 2, 9), 'lHV': (5, 6), 'aHV': (3, 4)}
...
area_idx = np.full(len(iarea), -1, dtype=np.int64)
for a, name in enumerate(AREA_NAMES):
    area_idx[np.isin(iarea, AREA_CODES[name])] = a
valid = np.nonzero(area_idx >= 0)[0]
if len(valid) > NEURONS_PER_SESSION:
    valid = np.sort(rng.choice(valid, NEURONS_PER_SESSION, replace=False))
```

iii. The area filter matches the paper's analysis. The subsampling is justified by the dataset size and decoder architecture constraints.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Trials start at corridor entry (`StartFr`) and include frames that are inside the textured corridor and during running. Trials have variable length. `off_start` is 0.0, `off_end` is None.

ii.
```python
neural_s.append(spk[:, f])
```
where `f` is the array of frame indices for that trial from `add_frame_selection()`.

iii. The AI stated: "Timepoints are the native two-photon frames from corridor entry to the end of the textured corridor (4 m) that were acquired while the mouse was running."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate (~3.18 Hz, ~314.6 ms per frame) is used with no rebinning. The `time_bin_size` is computed as the mean of per-session median inter-frame intervals, converted to milliseconds.

ii.
```python
dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)
...
'time_bin_size': float(np.mean(dts)) * 1000.0,
```

iii. The AI uses the native frame rate as the temporal resolution since no rebinning is needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr`, the frame index at which the sound cue was played in each trial, and the frame indices `f` of the trial's timepoints, scaled by `dt` (seconds per frame).

ii.
```python
time_to_cue = (rec['SoundFr'][t] - f) * dt
```

iii. The AI computes the time difference between the sound cue frame and each frame, converting to seconds by multiplying by the inter-frame interval.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(SoundFr[t] - frame_index) * dt` for each frame in the trial. This gives positive values before the cue and negative values after. The conversion uses a constant `dt` (median inter-frame interval converted to seconds) rather than interpolating onto a frame time axis.

ii.
```python
dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)
time_to_cue = (rec['SoundFr'][t] - f) * dt
```

iii. The AI described this as "seconds until the sound cue (positive before the cue, negative after it)."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices `f` used for the neural data of that trial.

ii.
```python
f = frames[trials == t]
time_to_cue = (rec['SoundFr'][t] - f) * dt
neural_s.append(spk[:, f])
```

iii. All data streams share the same frame indices per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp`, the date string in each recording's metadata, parsed as a calendar date.

ii.
```python
date = {k: datetime.date(*map(int, recordings[k]['datexp'].split('_'))) for k in keys}
first = {}
for k in keys:
    m = recordings[k]['mname']
    first[m] = min(first.get(m, date[k]), date[k])
for k in keys:
    recordings[k]['day'] = float((date[k] - first[recordings[k]['mname']]).days)
```

iii. The AI computed calendar days since each mouse's first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is found. Day of training is computed as calendar days elapsed since that mouse's first recording date. This gives 0 for the first session and grows through training. The value is broadcast across all frames of a trial.

ii.
```python
recordings[k]['day'] = float((date[k] - first[recordings[k]['mname']]).days)
...
day = np.full(len(f), rec['day'])
```

iii. The AI reasoned: "the date of each recording is the only training-time information in the metadata... 0 for every 'naive'/'before learning' session and grows through training." Values range from 0 to 92.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr`, the corridor entry frame for each trial, and the frame indices `f` of the trial's timepoints, scaled by `dt`.

ii.
```python
time_since_start = (f - rec['StartFr'][t]) * dt
```

iii. Same frame-based time computation approach as for time_to_sound_cue.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes `(frame_index - StartFr[t]) * dt` for each frame in the trial. This gives the time in seconds since corridor entry, starting near zero.

ii.
```python
time_since_start = (f - rec['StartFr'][t]) * dt
```

iii. Simple time difference computation using constant dt.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame indices `f` used for the neural data of that trial.

ii.
```python
f = frames[trials == t]
time_since_start = (f - rec['StartFr'][t]) * dt
neural_s.append(spk[:, f])
```

iii. All data streams share the same frame indices per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` (which trials had reward delivered) and `WallName` (the wall texture of each trial). The AI identifies which corridor texture was rewarded and marks all trials of that corridor type.

ii.
```python
rewarded_walls = set(rec['WallName'][rec['isRew']])
rew_of_trial = np.isin(rec['WallName'], list(rewarded_walls))
```

iii. The AI reasoned that using `isRew` directly would "leak the licking output into the decoder's input" because in some sessions reward delivery requires the mouse to have licked. Instead, it identifies the rewarded corridor and marks all trials of that corridor.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI finds which wall textures had `isRew=True`, then marks every trial with that wall texture as reward-available (1.0), regardless of whether the mouse actually licked/received reward on that specific trial. Unsupervised and naive mice get 0 for all trials. The value is broadcast across all frames.

ii.
```python
rewarded_walls = set(rec['WallName'][rec['isRew']])
assert len(rewarded_walls) <= 1, (k, rewarded_walls)
rew_of_trial = np.isin(rec['WallName'], list(rewarded_walls))
...
rew = np.full(len(f), 1.0 if rew_of_trial[t] else 0.0)
```

iii. The AI explicitly noted the information leakage concern: "beh['isRew'] marks the trials on which water was actually delivered, which in the 'active after cue' sessions requires the mouse to have licked -- using it directly would leak the licking output into the decoder's input."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `stim_id` (merged across experiment types via the `wallmap` dictionary) and `UniqWalls`. The AI maps each wall texture to the paper's canonical stimulus category numbering.

ii.
```python
for wall, cat in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], float)):
    if not np.isnan(cat):
        rec['wallmap'][str(wall)] = int(cat)
...
stim_of_trial = np.array([wallmap.get(w, -1) for w in rec['WallName']])
```

iii. The AI reasoned that different mice were trained with different texture pairs but `stim_id` maps them onto a common set of roles so categories are comparable across mice.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the paper's canonical 7-category labelling from `stim_id`: `['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']`. Trials where no category is assigned (e.g. circle3) are dropped. The value is broadcast per trial.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
...
out[0] = stim_of_trial[t]
```

iii. The AI chose the paper's canonical labelling to ensure cross-mouse comparability.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of every lick in the session.

ii.
```python
lick_fr = np.round(rec['LickFr']).astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
licks[lick_fr] = True
```

iii. Lick times are directly available from `LickFr`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds `LickFr` to integer frame indices, filters to valid frame range, then sets a boolean array to True at those frames. A frame is 1 if at least one lick was detected in it, 0 otherwise.

ii.
```python
licks = np.zeros(nframes, dtype=bool)
lick_fr = np.round(rec['LickFr']).astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
licks[lick_fr] = True
```

iii. Simple conversion from lick frame indices to a binary time series.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the lick flag is on the same grid as neural data. It is taken at the same frame indices `f` used for neural data.

ii.
```python
out[1] = licks[f]
```

iii. All data streams share the same frame indices per trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame.

ii.
```python
pos = rec['ft_Pos']
...
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. Position is directly available from the behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI uses `np.digitize` with edges `[10.0, 20.0, 30.0]` to bin position into 4 bins. Since the corridor is 40 units (4m) long, this creates four 1m bins.

ii.
```python
POSITION_EDGES = [10.0, 20.0, 30.0]
...
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. The corridor is 40 units (=4m), so edges at 10, 20, 30 create four 1m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `np.digitize(pos[f], [10.0, 20.0, 30.0])` produces values 0, 1, 2, 3 for position ranges [0-10), [10-20), [20-30), [30+) in decimeters, corresponding to [0-1m), [1-2m), [2-3m), [3-4m).

ii.
```python
POSITION_EDGES = [10.0, 20.0, 30.0]
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. Four equal-length 1m bins as specified in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one position per imaging frame, so it is already on the same grid as neural data. Taken at the same frame indices `f`.

ii.
```python
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. All data streams share the same frame indices per trial.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = rec['ft_RunSpeed']
...
out[3] = np.digitize(speed[f], speed_edges)
```

iii. Running speed is directly available from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes speed quartile edges globally across all sessions by collecting all valid timepoints' speeds and computing the 25th, 50th, and 75th percentiles. Then `np.digitize` is used to bin each frame's speed into one of 4 quartile bins.

ii.
```python
speeds = []
for k in keys:
    rec = recordings[k]
    frames, _ = add_frame_selection(rec, len(rec['ft']))
    speeds.append(rec['ft_RunSpeed'][frames])
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
...
out[3] = np.digitize(speed[f], speed_edges)
```

iii. The AI reasoned that computing edges globally ensures "a speed bin means the same thing in every session."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed[f], speed_edges)` with `speed_edges` being the 25th/50th/75th percentiles of all running speeds. This produces values 0-3 representing speed quartiles.

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.digitize(speed[f], speed_edges)
```

iii. Four quartile bins, each holding approximately 25% of all data points.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one speed per imaging frame, so it is already on the same grid as neural data. Taken at the same frame indices `f`.

ii.
```python
out[3] = np.digitize(speed[f], speed_edges)
```

iii. All data streams share the same frame indices per trial.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several handling mechanisms: (1) Behavior arrays are truncated to match the number of imaged frames: `n = min(nframes, len(rec['ft']))`. (2) Lick frames outside valid range are filtered: `lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]`. (3) Trials with unmapped stimuli are dropped (309 trials). (4) Trials with zero valid frames are skipped.

ii.
```python
n = min(nframes, len(rec['ft']))
...
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
...
if stim_of_trial[t] < 0:
    n_dropped_stim += 1
    continue
f = frames[trials == t]
if len(f) == 0:
    continue
```

iii. The AI noted "the behaviour arrays are sometimes one frame longer (the paper's code truncates the same way)."

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files from disk, which are very large (tens of thousands of neurons per session). The AI optimized this by selecting neurons plane-by-plane to avoid materializing the full matrix.

ii.
```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
...
for p, plane in enumerate(planes):
    m = (sel >= offsets[p]) & (sel < offsets[p + 1])
    if m.any():
        out[m] = plane[sel[m] - offsets[p]]
```

iii. The AI noted: "Selecting inside the loop avoids ever materialising the full (nneurons x nframes) matrix, which can be 10 GB."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop iterates over each trial individually to extract frame indices and build per-trial arrays. This could potentially be vectorized using groupby-style operations on `ft_trInd`.

ii.
```python
for t in range(rec['ntrials']):
    ...
    f = frames[trials == t]
    ...
```

iii. No explicit justification provided.

## 12-c. What processing does the code repeat multiple times?

i. The `add_frame_selection` is called twice for each recording: once during the global speed quartile computation pass, and once during the main conversion pass. The first pass computes frame selections using `len(rec['ft'])` as nframes, while the second uses the actual spike file nframes.

ii.
```python
# First pass (speed quartiles):
frames, _ = add_frame_selection(rec, len(rec['ft']))

# Second pass (main conversion):
frames, trials = add_frame_selection(rec, nframes)
```

iii. The first pass uses behavior file length while the second uses spike file length, so the frame selections may differ slightly.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores detailed metadata and session info that is not used by the decoder, including cohort, experiment types, reward mode, frame rate per session, etc. Additionally, the neuron subsampling means most of the neural data loaded from each spike file is discarded.

ii.
```python
session_info.append(dict(
    session=k, mouse=rec['mname'], date=rec['datexp'], block=rec['blk'],
    cohort=rec['cohort'], experiment_types=sorted(set(rec['exp_types'])),
    day_of_training=rec['day'], reward_mode=rec['reward_mode'],
    ntrials=len(neural_s), nneurons=len(area),
    ntimepoints=int(sum(x.shape[1] for x in neural_s)),
    frame_rate_hz=1.0 / dt,
))
```

iii. The extra metadata provides context but is not consumed by the decoder.
