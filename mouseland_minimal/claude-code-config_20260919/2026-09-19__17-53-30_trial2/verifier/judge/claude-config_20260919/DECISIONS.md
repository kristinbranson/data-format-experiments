# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats `beh/Imaging_Exp_info.npy` as the master index. It iterates over its 23 experiment-type keys and, for each entry, builds the recording id `mname_datexp_blk` and the behavior key (`+ '_' + stimtype` for the swap entries). All 23 `Beh_<exp_type>.npy` files are loaded up front into one dict (`beh_files`) and every entry of a recording is merged into a single per-recording record holding the behavior dict, the experiment types, the cohort, and a merged `UniqWalls -> stim_id` map. Deconvolved activity is read per session from `spk/<rec>_neural_data.npy` (planes concatenated), and the per-neuron visual area from `retinotopy/<mname>_<datexp>_trans.npz`. The conversion of the 89 recordings is then farmed out to a `multiprocessing` fork pool (default 8 workers), each worker loading its own spike file.

ii.
```python
def load_exp_info():
    return np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()

beh_files = {et: np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % et),
                         allow_pickle=True).item() for et in exp_info}
sessions = collections.OrderedDict()
for exp_type, db in exp_info.items():
    for ndb in db:
        rec = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        key = rec + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
        entry = sessions.setdefault(rec, {...})
        ...
        beh = beh_files[exp_type][key]
        entry['beh'] = beh
        for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
            if not np.isnan(float(sid)):
                entry['wall2stim'][str(wall)] = int(sid)
```
```python
def load_spikes(rec):
    """Deconvolved activity, neurons x frames, planes concatenated (utils.load_spk)."""
    path = os.path.join(DATA_ROOT, 'spk', '%s_neural_data.npy' % rec)
    return np.concatenate(np.load(path, allow_pickle=True).item()['spks'], 0)

def load_areas(mname, datexp):
    path = os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
    return np.load(path, allow_pickle=True)['iarea']
```

iii. The loaders are written to "mirror `code/utils.py`" (the AI's own section header); `load_spikes` is annotated as reproducing `utils.load_spk` and `load_areas` reproduces `utils.load_retino`. The AI verified up front (trajectory steps 19, 28) that the 89 index recordings have exactly 89 matching spike files and that no retinotopy file is missing. Merging every index entry of a recording is justified in the `load_all_behavior` docstring: a single entry "only names the stimuli that are relevant for that particular comparison in the paper -- the others are left as the placeholder string `'stimulus_of_trial'`", so the maps have to be merged to recover the stimulus of every trial.

## 1-b. How are the data split into subjects?

i. The mouse is read straight from `mname` in the index entry and carried on the per-recording record as `entry['mname']` / `info['mouse']`. `subjects` is the sorted set of `mname` over all converted recordings (19 mice), and `subject_idx` is each session's index into that list. No subject-level curation is done — all 19 mice, including the naive, unsupervised, task and grating cohorts, are kept.

ii.
```python
subjects = sorted({s['mname'] for s in sessions.values()})
session_info = [r[4] for r in results]
...
'subject_idx': np.array([subjects.index(i['mouse']) for i in session_info], dtype=np.int64),
```

iii. No justification needed beyond the data: the index names the mouse of every recording. The AI confirmed the count in the trajectory (step 50: `19 ['DR10', 'DR15', ... 'VR2']`) and the cohort of each session is additionally recorded in `session_info['cohort']` so the cohorts stay recoverable downstream.

## 1-c. How are the data split into sessions?

i. A session is one recording = one mouse × one date × one block, keyed `mname_datexp_blk`. Because `Imaging_Exp_info.npy` lists the same recording under several experiment types (e.g. `unsup_test1` and `unsup_train2_before_learning`, or the `swap1`/`swap2` stimtype pairs), the AI de-duplicates with `sessions.setdefault(rec, ...)`, so each recording is converted exactly once: 89 sessions. The list of experiment types that referenced the recording is kept in `session_info['experiment_types']`.

ii.
```python
rec = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
key = rec + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
entry = sessions.setdefault(rec, {
    'mname': ndb['mname'], 'datexp': ndb['datexp'], 'blk': ndb['blk'],
    'exp_types': [], 'cohorts': set(), 'wall2stim': {}, 'beh': None})
entry['exp_types'].append(exp_type)
```

iii. From the `load_all_behavior` docstring: entries of one recording "share the same behaviour", so the AI merges their stimulus maps and converts "each recording exactly once". It had checked in the trajectory (step 32) that duplicate entries of a recording agree on `ntrials`, and that the number of unique recordings equals the number of spike files (89).

## 1-d. How are the data split into trials?

i. Trials are the ones the data declares (`beh['ntrials']`), and each imaging frame is assigned to a trial through `beh['ft_trInd']`. A frame belongs to a trial's window only if three conditions hold: it is inside the textured corridor (`ft_CorrSpc`), the virtual reality was advancing, i.e. the mouse was running faster than the 6 cm/s threshold (`ft_move > 0`), and it carries a non-NaN trial index. Frames are grouped per trial with one stable argsort + `searchsorted`, keeping them in time order. Trials therefore have variable length (median 21 frames, min 11, max 178) and, because stationary frames are removed from the *middle* of a traversal, the frames of a trial are not necessarily contiguous in real time.

ii.
```python
def frame_mask(beh, nframes):
    corridor = beh['ft_CorrSpc'][:nframes]
    running = beh['ft_move'][:nframes] > 0
    in_trial = ~np.isnan(beh['ft_trInd'][:nframes])
    return corridor & running & in_trial

def trial_frames(beh, nframes):
    """dict trial index -> array of imaging-frame indices, ordered in time."""
    mask = frame_mask(beh, nframes)
    frames = np.where(mask)[0]
    trials = beh['ft_trInd'][:nframes][frames].astype(int)
    order = np.argsort(trials, kind='stable')
    frames, trials = frames[order], trials[order]
    splits = np.searchsorted(trials, np.arange(int(beh['ntrials']) + 1))
    return {t: frames[splits[t]:splits[t + 1]] for t in range(int(beh['ntrials']))}
```

iii. The `frame_mask` docstring gives all three conditions with their source: `ft_CorrSpc` restricts to the 4 m over which the stimulus is shown, "which also makes the trial window match the 4 x 1 m position bins requested by the decoder task"; `ft_move > 0` implements the paper's rule quoted from methods.txt — "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards" — and it points at the released code that builds exactly this mask (`VRmove = beh['ft_move']>0`, `corr_fr = beh['ft_CorrSpc'] & VRmove`, e.g. `utils.py:536-537`). The final summary repeats this: trial window is "corridor entry → corridor exit (the 4 m textured section only ...), keeping frames with `ft_CorrSpc & (ft_move > 0)` — the paper's 'only timepoints during running' rule, exactly as `utils.Get_coding_direction` builds `corr_fr`".

## 1-e. How are trials filtered based on quality controls?

i. Two filters, applied identically in the behavior pre-pass and in the conversion: a trial needs at least `MIN_FRAMES_PER_TRIAL = 5` running-corridor frames, and its wall texture must map to one of the seven `stim_id` names. 309 of 38,110 trials are dropped, all of them by the second rule (`circle3` trials in four naive/grating sessions whose `UniqWalls → stim_id` map never names `circle3`); the ≥5-frame rule removes nothing at all. 37,801 trials survive, the smallest session keeping 84. No upper bound on trial length is imposed — the running mask already strips the stationary periods that make a traversal long, so the longest kept trial is 178 frames.

ii.
```python
MIN_FRAMES_PER_TRIAL = 5
...
keep = [t for t in range(int(beh['ntrials']))
        if len(frames[t]) >= MIN_FRAMES_PER_TRIAL and names[t] is not None]
```
```python
for t in range(int(beh['ntrials'])):
    fr = frames[t]
    if len(fr) < MIN_FRAMES_PER_TRIAL or names[t] is None:
        continue
```

iii. The constant is documented in the source: "A trial is kept only if the animal actually traversed the corridor while the imaging was running; 4 m of virtual corridor at the fixed VR speed takes ~21 imaging frames, so this only removes truncated (start/end of recording) trials." The stimulus-based drop is reported in the final summary: "Trials dropped: 309 of 38,110 — those whose wall texture is never mapped to a paper stimulus id (a stray `circle3` in 4 naive sessions) or with <5 running frames." The AI explicitly examined long traversals first (trajectory step 59: durations, 3.2% of trials > 30 s, 0.13% > 300 s) and chose not to add a duration cut.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<rec>_neural_data.npy`, a list of one (neurons × frames) array per imaging plane, concatenated along the neuron axis into one matrix. The per-neuron area label comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`, and the code asserts that its length equals the number of concatenated neurons, so the two orders correspond.

ii.
```python
spk = load_spikes(rec)
nneurons_total, nframes = spk.shape
iarea = load_areas(entry['mname'], entry['datexp'])
assert len(iarea) == nneurons_total, rec
```

iii. `load_spikes` is labelled as `utils.load_spk`; the AI verified in the trajectory (step 38) that the concatenated neuron count equals `iarea.shape` for several sessions and that the traces are non-negative deconvolved values. The metadata records the provenance: "two-photon mesoscope calcium imaging, Suite2p deconvolved traces (0.75 s decay timescale), ~3.17 Hz".

## 2-b. How is the `neural` data processed?

i. Three steps. (1) Neurons outside the four visual-area groups, and neurons with zero standard deviation, are removed. (2) Of the survivors, at most `nsub = 2000` per session are kept, chosen by a seeded uniform random draw and re-sorted into their original order. (3) Each kept neuron is z-scored over the whole recording (mean and SD over all imaging frames, not just the corridor frames) and stored as `float32`. Per trial the kept columns are copied out with `np.ascontiguousarray`; trials keep their own length, nothing is padded or rebinned.

ii.
```python
sd = spk.std(axis=1)
usable = np.where((region >= 0) & (sd > 0))[0]   # silent neurons cannot be z-scored

rng = np.random.default_rng(SEED + session_index)
if nsub is not None and len(usable) > nsub:
    usable = np.sort(rng.choice(usable, size=nsub, replace=False))

# z-score every neuron over the whole recording, as the paper's code does
# (utils.get_kfold_reward_response: `spk = stats.zscore(spk, axis=1)`).
activity = spk[usable]
activity = (activity - activity.mean(axis=1, keepdims=True)) / sd[usable][:, None]
activity = activity.astype(np.float32)
...
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. The z-score is justified in the comment by the released analysis code (`utils.py:820`, `stats.zscore(spk, axis=1)`) and by putting "sessions with very different deconvolved amplitudes on a common scale". The subsample is called out as "the one deliberate deviation" in the final summary, with three reasons: keeping all 20k–90k neurons gives a ~150 GB dataset, the decoder compresses every session to 100 components and already random-projects to 2000 neurons for its SVD initialisation, and a 4-session pilot A/B test showed the 2000-neuron version scored *higher* than the all-neuron version (stimulus 0.977 vs 0.924, position 0.897 vs 0.841, speed 0.590 vs 0.582).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only two neuron-level filters, plus the subsample. A neuron is kept if its retinotopy `iarea` falls in V1 (8), mHV (0,1,2,9), lHV (5,6) or aHV (3,4) — `iarea` of −1 or 7 is dropped — and if its SD over the recording is non-zero. Then at most 2000 of the survivors per session are randomly retained, so every session in the delivered pickle has exactly 2000 neurons (out of 20,547–89,577 recorded). `brain_region_idx` is the region code of exactly those kept neurons, in the same row order as `neural`.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
AREA_CODE_TO_REGION = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}
...
region = np.full(nneurons_total, -1, dtype=np.int64)
for code, ridx in AREA_CODE_TO_REGION.items():
    region[iarea == code] = ridx
sd = spk.std(axis=1)
usable = np.where((region >= 0) & (sd > 0))[0]
```

iii. The area grouping is annotated as "exactly as `utils.neu_area_ID` groups the retinotopy `iarea` codes", and the exclusion of −1/7 is justified because "the paper's code (`utils.Get_density_map`) explicitly calls these 'neurons from outside of visual cortex' and excludes them". No further quality filter is applied because Suite2p's cell classifier already curated the ROIs. The 2000-neuron cap is justified by size and by the pilot decoding comparison described in 2-b.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is on trial start = corridor entry. Each trial's neural matrix is the columns of the z-scored activity for that trial's running-corridor frames, in time order, starting at the first running frame inside the corridor and ending at the last one before the grey space. Trials are variable length, nothing is padded or truncated; `off_start = 0.0` and `off_end = None` are written to the metadata.

ii.
```python
for t in range(int(beh['ntrials'])):
    fr = frames[t]
    ...
    neural.append(np.ascontiguousarray(activity[:, fr]))
```
```python
'temporal_alignment_event':
    'trial start = corridor entry (the frame at which the animal enters the '
    'textured corridor, beh["Trial_start_time"]/beh["StartFr"])',
'off_start': 0.0,
'off_end': None,
```

iii. From the metadata `trial_window` field: "From corridor entry to corridor exit (4 m later). Only frames during which the virtual reality advanced ... are kept, following the paper ... so trials have a variable number of frames (median 21, i.e. ~6.6 s of running) and `off_end` is not a fixed time." The format spec permits `off_end = None`, and the decoder treats every time bin as an independent sample, so a common window is not required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, no resampling, no smoothing: one column of `spks` per imaging frame is one time bin. The bin size written to the metadata is the median inter-frame interval over the sessions, 314.69 ms (~3.177 Hz), and it is computed from the timestamps rather than hard-coded. Because non-running frames are dropped, consecutive stored bins within a trial can be separated by more than one frame period.

ii.
```python
'frame_period_s': float(np.median(np.diff(ft)) * 86400.0),
...
'time_bin_size': float(np.median([i['frame_period_s'] for i in session_info]) * 1000.0),
```

iii. The imaging frame is the finest resolution available, and the behavior streams (`ft_Pos`, `ft_RunSpeed`, `ft_trInd`, `ft_move`) are already sampled on the same grid, so no resampling is needed. The AI measured the frame period in the trajectory (step 30, `dt median ≈ 0.3146 s`) rather than taking the notebook's nominal 3.17 Hz.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundTime']`, the MATLAB datenum timestamp of the sound cue in each trial, and `beh['ft']`, the datenum timestamp of each imaging frame (truncated to the imaged frames).

ii.
```python
ft = beh['ft'][:nframes]
...
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
```

iii. Using the timestamps rather than the frame numbers (`SoundFr`) keeps the value in physical seconds and, as the input-units metadata says, is "seconds until the sound cue (negative after the cue)". The AI checked beforehand (trajectory step 57) that `SoundTime` has no NaNs over the retained frames of any session.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. A single subtraction per bin, converted from days to seconds (`× 86400`), stored as `float32` in row 0 of the input array. It is positive before the cue and negative after, and it is time-varying within the trial. No clipping or normalisation is applied, so the range over the dataset is [−1763.3, 723.5] s — the tail comes from the ~1% of trials in which the mouse stood still for minutes inside a corridor.

ii.
```python
inp = np.empty((4, len(fr)), dtype=np.float32)
inp[0] = t_to_cue
```
```python
'input_names': ['time_to_sound_cue', 'day_of_training',
                'time_since_trial_start', 'reward_available'],
```

iii. The decoder-task spec asks for "Time to sound cue, continuous, time-varying", so a signed continuous seconds-to-cue is the literal implementation. The AI notes that seconds "are derived from the frame timestamps (MATLAB datenum -> seconds), so gaps created by removing non-running frames are kept", i.e. it deliberately reports true elapsed time rather than a running-frame count.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `ft[fr]` — exactly the same frame indices `fr` that index the neural columns of that trial — so it is sample-for-sample aligned and has the same length as the trial's neural matrix.

ii.
```python
fr = frames[t]
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
...
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. Every stream in this dataset (`ft`, `ft_Pos`, `ft_RunSpeed`, `spks`) is on the imaging-frame grid, so indexing all of them with the same `fr` is the alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp`, the calendar date of the recording in the index entry, relative to the earliest `datexp` of the same mouse.

ii.
```python
first_day = {}
for entry in sessions.values():
    date = datetime.date(*map(int, entry['datexp'].split('_')))
    m = entry['mname']
    first_day[m] = min(first_day.get(m, date), date)

def day_of_training(entry, first_day):
    date = datetime.date(*map(int, entry['datexp'].split('_')))
    return float((date - first_day).days)
```

iii. From the function docstring: "The dataset does not record the calendar date on which each animal started its virtual-reality training, so the first imaging session of a mouse is used as its reference day. For every mouse that has a 'before learning' session this is literally day 0 of training / of unsupervised exposure; for the naive cohort it is the day of the naive recording." The AI printed the per-mouse day offsets in the trajectory (step 53) before choosing this, and looked for but found no explicit training-start field (step 55 checks `days` / `Note`).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. One scalar per session, `(date − first date of that mouse).days`, as a float, broadcast across every bin of every trial of the session (row 1 of the input array). Values run from 0 to 92; gaps between recordings are preserved (e.g. TX108 goes 0, 67, 76, 79, 86, 89, 92). It is also stored per session in `session_info['day_of_training']`.

ii.
```python
inp[1] = day
```
```python
jobs = [(rec, sessions[rec], speed_edges,
         day_of_training(sessions[rec], first_day[sessions[rec]['mname']]),
         nsub, i) for i, rec in enumerate(recs)]
```

iii. Calendar days are the literal reading of "day of training" and preserve the real spacing of the recordings; the AI flags in the docstring and in its final summary that the reference day is a "documented proxy" because the dataset has no VR-training start date.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `beh['Trial_start_time']`, the datenum "time when animal enters each corridor" (the notebook's own definition), and `beh['ft']`, the frame timestamps.

ii.
```python
t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
```

iii. Same rationale as 3-a: timestamps give physical seconds; `Trial_start_time` is the documented corridor-entry time and is the event the whole dataset is aligned to. The AI verified it has no NaNs over the retained frames (trajectory step 57).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. One subtraction per bin, days → seconds, stored as `float32` in row 2. It starts at ~0 at the first retained frame of the trial and increases; because stationary frames are dropped but real timestamps are kept, the value can jump. The distribution is median 3.8 s, 99th percentile 61 s, maximum 1765 s. No clipping or normalisation.

ii.
```python
inp[2] = t_since_start
```
```python
# inputs: seconds are derived from the frame timestamps (MATLAB datenum
# -> seconds), so gaps created by removing non-running frames are kept.
```

iii. Explicit design choice recorded in the code comment above: the gaps created by the running filter are *kept* rather than being collapsed into a running-frame count, so the input always reports true elapsed time since corridor entry. The spec asks for "Time since trial start, continuous, time varying".

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on the same `fr` frame indices as the neural columns, so it is bin-for-bin aligned and the same length. `off_start = 0.0` in the metadata states that the trial window begins at the alignment event.

ii.
```python
fr = frames[t]
t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. As in 3-c: all streams share the imaging-frame grid, so one index array aligns them.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. `isRew` is the dataset's own flag for the rewarded corridor, which is exactly the spec's "1 if in rewarded corridor, 0 if not". The AI additionally stores `session_info['rewarded_session'] = bool(np.any(beh['isRew']))` and the cohort label, so that the unsupervised/naive sessions (which are all `isRew = False`) remain identifiable.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool→float cast and broadcasting the per-trial scalar across all bins of the trial (row 3). The value is 0 for every trial of the unsupervised, naive and grating cohorts; within task sessions the fraction of rewarded trials is up to 0.54.

ii.
```python
inp = np.empty((4, len(fr)), dtype=np.float32)
...
inp[3] = float(beh['isRew'][t])
```
```python
'reward_available': '1 if this corridor is the rewarded one (task cohort only), else 0'
```

iii. Nothing to process — the raw flag already encodes the requested variable. The AI checked the per-session rewarded fraction in the trajectory (step 50) before accepting it.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']` (the texture on the walls of each trial) mapped through the recording's merged `UniqWalls → stim_id` table, and then through the notebook's canonical id list `0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3, 5:leaf1_swap1, 6:leaf1_swap2`. `beh['TrialStim']` is deliberately not used.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']

def session_stimulus_names(entry):
    """Canonical stimulus name of each trial, or None where the paper gives none."""
    wall2stim = entry['wall2stim']
    return [STIM_NAMES[wall2stim[str(w)]] if str(w) in wall2stim else None
            for w in entry['beh']['WallName']]
```

iii. The AI found (trajectory step 42) that `TrialStim` carries the placeholder `'stimulus_of_trial'` for stimuli irrelevant to a given experiment type, so it reconstructs the identity from `UniqWalls`/`stim_id` merged over all the entries of a recording — explained at length in the `load_all_behavior` docstring and repeated in the final summary. The constant is annotated with the exact source: "indexed by `beh['stim_id']` (see `data_process_script.ipynb`: '0:circle1, 1:circle2, 2:leaf1, ...')".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The per-trial wall name becomes the integer `stim_id` (0–6) and is broadcast across all the trial's bins as output row 0, with `output_values[0] = STIM_NAMES`, i.e. a 7-way categorical. Because `stim_id` is the dataset's *role* index rather than the physical texture, the same code means different physical textures in different mice: id 0 is `circle1` in 62 recordings, `leaf1` in 8 and `rock1` in 19; id 2 is `leaf1` in 62, `circle1` in 6 and `wood1` in 19. The `leaf1_swap1`/`leaf1_swap2` spatial shuffles are kept as their own two categories rather than being pooled with `leaf1`. Trials whose wall has no id (309 `circle3` trials) are dropped rather than labelled. Resulting class fractions: circle1 0.319, leaf1 0.335, leaf2 0.170, circle2 0.060, leaf3 0.059, leaf1_swap2 0.029, leaf1_swap1 0.027.

ii.
```python
out[0] = STIM_NAMES.index(names[t])
```
```python
'output_values': [
    list(STIM_NAMES),
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['speed_Q1', 'speed_Q2', 'speed_Q3', 'speed_Q4'],
],
```

iii. From the final summary: "Stimulus output uses the paper's 7 canonical identities (`circle1/2`, `leaf1/2/3`, `leaf1_swap1/2`) rather than a coarse leaf-vs-circle split, since the test stimuli are central to the paper and the 2-way grouping is recoverable from it." The module docstring acknowledges the naming convention it inherits: these are the "canonical names used by the paper even when the physical textures were rock/brick/wood", which is the convention methods.txt states ("we denote the stimuli as 'leaf' and 'circle', even though other visual stimuli were also used in some mice ('rock' and 'bricks')").

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
def lick_frames(beh, nframes):
    """Boolean per imaging frame: did the animal lick during this frame?"""
    licks = np.zeros(nframes, dtype=bool)
    lick_fr = np.asarray(beh['LickFr'], dtype=float)
    lick_fr = lick_fr[np.isfinite(lick_fr)]
    if lick_fr.size:
        idx = np.round(lick_fr).astype(int)
        idx = idx[(idx >= 0) & (idx < nframes)]
        licks[idx] = True
    return licks
```

iii. `LickFr` is already expressed in imaging frames, which is the grid the converted data uses, so it is the direct source. The spec asks for licking as a binary time series, which is what a per-frame flag gives.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length boolean vector is built once: a frame is 1 if at least one lick rounds to it, else 0. Non-finite lick frames and licks outside the imaged range are discarded. Sessions with no licks at all (`LickFr` empty) give an all-zero vector. Per trial the flag is sliced and cast to `int64` as output row 1. Over the delivered dataset 3.5% of bins are licks (the running filter removes much of the stopped-at-the-spout licking).

ii.
```python
licks = lick_frames(beh, nframes)
...
out[1] = licks[fr]
```
```python
['no_lick', 'lick'],
```

iii. Rounding rather than truncating puts a lick in the frame it is closest to; the finiteness and range guards are the AI's handling of behavior that runs past the imaging (see 11). The AI checked the resulting per-session lick fractions before committing (trajectory step 50: median 0, 75th pct 0.066, max 0.379 — zero for the unsupervised/naive sessions, as expected).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The flag vector is indexed with the same `fr` frames as the neural columns, giving one label per neural time bin.

ii.
```python
out[1] = licks[fr]
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. `LickFr` indexes imaging frames directly, so no interpolation or resampling is needed to put licking on the neural grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the position inside the corridor at each imaging frame, in decimetres (0–40 across the texture, continuing to 60 through the grey space), truncated to the imaged frames.

ii.
```python
pos = beh['ft_Pos'][:nframes]
...
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
```

iii. `ft_Pos` is the only per-frame position stream and is already on the imaging grid. The AI verified the units and range (trajectory steps 46 and 57: within corridor frames `ft_Pos` spans 0 to 40, no NaNs) and confirmed `Corridor_Length 60 / Texture_Length 40 / Gray_Space_length 20` are constant across all 89 recordings (step 44).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Divide by 10 dm (= 1 m), truncate to an integer, clip into [0, 3], store as `int64` in output row 2. Because the trial window is restricted to `ft_CorrSpc`, only the 0–4 m textured section contributes, so the clip only catches the single frame that sits exactly at 40.0. The resulting distribution is essentially uniform: 0.250 / 0.249 / 0.250 / 0.252.

ii.
```python
TEXTURE_LENGTH_DM = 40.0
POSITION_BIN_DM = 10.0          # 1 m spatial bins -> 4 bins over the 4 m corridor
N_POSITION_BINS = 4
...
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
```

iii. Directly implements the decoder-task requirement: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins". The `frame_mask` docstring explains that restricting the window to `ft_CorrSpc` "also makes the trial window match the 4 x 1 m position bins requested by the decoder task".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-length spatial thresholds at 1, 2 and 3 m (10, 20, 30 dm), labelled `'0-1m'`, `'1-2m'`, `'2-3m'`, `'3-4m'` — not data-driven quantiles.

ii.
```python
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
...
['0-1m', '1-2m', '2-3m', '3-4m'],
```

iii. The task specifies equal-length 1 m bins, and since the virtual reality advances at a constant speed whenever the mouse runs, equal-length bins are also nearly equally occupied, which the observed 25/25/25/25 split confirms.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, and the same `fr` index array used for the neural columns selects the position labels, so the two are bin-for-bin aligned.

ii.
```python
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. Common imaging-frame grid, as for every other stream.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame (cm/s), truncated to the imaged frames.

ii.
```python
speed = beh['ft_RunSpeed'][:nframes]
...
speeds.append(beh['ft_RunSpeed'][:nframes][frames[t]])
```

iii. `ft_RunSpeed` is the per-frame speed stream, already on the neural grid. The AI checked it against `RunFr` and `ft_move` (trajectory step 48) and confirmed there are no NaNs on the retained frames (step 57).

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first pass over behaviour only (`collect_behavior`) collects the speed of every timepoint that will be kept — all retained trials of all 89 sessions pooled — and takes the 25th/50th/75th percentiles as three global bin edges (12.40, 25.28, 40.75 cm/s). In the conversion pass each trial's speeds are turned into bin indices with `np.searchsorted(..., side='right')`, giving `int64` output row 3. Edges are computed once, globally, not per session, and are recorded in the metadata.

ii.
```python
def collect_behavior(sessions):
    """First pass (behaviour only): trial structure, and the global speed quartiles."""
    speeds = []
    ...
    speeds = np.concatenate(speeds)
    edges = np.percentile(speeds, [25, 50, 75])
    return per_session, edges
```
```python
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
...
'speed_bin_edges_cm_per_s': [float(e) for e in speed_edges],
```

iii. From the `collect_behavior` docstring: the speed output "has to be binned into four bins 'each corresponding to 25% of the data', so the bin edges are the quartiles of the running speed over every timepoint that ends up in the dataset (pooled over all sessions, which is what makes each bin hold 25% of the data)." The AI pre-computed the same quartiles in exploration (step 57) and reports them in the final summary.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three data-driven thresholds (the global quartiles) split speed into `speed_Q1`–`speed_Q4`; values equal to an edge fall in the upper bin (`side='right'`). Negative speeds (the sensor reports down to about −19 cm/s) simply land in Q1. Because the non-running frames were already removed, almost no timepoint sits at exactly zero (0.01% within the retained frames), so there is no tie that would unbalance the bins.

ii.
```python
N_SPEED_BINS = 4
edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
...
['speed_Q1', 'speed_Q2', 'speed_Q3', 'speed_Q4'],
```

iii. Quartiles are the literal implementation of "4 bins, each corresponding to 25% of the data"; pooling over the whole dataset makes the statement true of the dataset as a whole, and keeps the labels comparable across sessions (a "Q4" bin means the same speed in every mouse).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. One speed per imaging frame; the trial's labels are `speed[fr]` with the same `fr` used for the neural columns, so they are bin-for-bin aligned.

ii.
```python
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. Common imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards, all inline. Behaviour can run past the imaging, so every behaviour stream is truncated to `nframes = spk.shape[1]` in the conversion pass. Frames with `NaN` trial index are excluded from every trial window. `LickFr` entries that are non-finite, negative, or beyond the last imaged frame are dropped. Neurons whose `iarea` is missing (−1) or outside visual cortex (7) are removed, as are neurons with zero SD (which cannot be z-scored). The neuron/area correspondence is asserted rather than assumed. Trials whose stimulus cannot be identified, or which have fewer than 5 retained frames, are skipped. There is no `try/except` around a session: a failure would abort the whole run.

ii.
```python
nneurons_total, nframes = spk.shape
iarea = load_areas(entry['mname'], entry['datexp'])
assert len(iarea) == nneurons_total, rec
```
```python
in_trial = ~np.isnan(beh['ft_trInd'][:nframes])
```
```python
lick_fr = lick_fr[np.isfinite(lick_fr)]
idx = np.round(lick_fr).astype(int)
idx = idx[(idx >= 0) & (idx < nframes)]
```
```python
usable = np.where((region >= 0) & (sd > 0))[0]   # silent neurons cannot be z-scored
```

iii. The truncation is justified by reference to the released code: "All behaviour arrays are truncated to the number of imaging frames, as in `utils.Get_dprime_selective_neuron` (`nfr = spk.shape[1]`; the behaviour files can contain a few frames more than the imaging files)." The AI had confirmed beforehand (trajectory steps 46, 57) that the only real gaps are the handful of NaN `ft_trInd` frames and the frame-count mismatch, and that position, speed, cue time and start time contain no NaNs on the retained frames.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 405 GB of `spk/*_neural_data.npy` (up to 7.4 GB per session) and concatenating the planes, followed by the full-matrix `spk.std(axis=1)` and mean over every one of the 20k–90k recorded neurons. Both are paid before the 2000-neuron subsample shrinks the data. The AI mitigates the cost by running the per-session conversion in a `multiprocessing` fork pool (8 workers for the final run), which brought the whole 89-session conversion to roughly 4 minutes of wall clock. The behaviour-only pre-pass (`collect_behavior`) touches no spike file and is cheap.

ii.
```python
spk = load_spikes(rec)
nneurons_total, nframes = spk.shape
...
sd = spk.std(axis=1)
```
```python
if args.workers > 1:
    import multiprocessing as mp
    with mp.get_context('fork').Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap(_worker, jobs)):
```

iii. The AI measured a single spike-file load early on (trajectory step 36, `time python -c ...`) and checked node memory and core count (step 16) before choosing the worker count; it monitored `free -g` during the run (step 82) to keep 8 concurrent multi-GB loads inside RAM.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Very little remains. The per-trial frame grouping is already vectorized — one mask, one stable `argsort`, one `searchsorted` for all trials at once — rather than scanning the frame index once per trial. What is left is the per-trial Python loop in `convert_session`, which does ~400 small slices and `np.stack`s per session; the per-trial subtraction of `Trial_start_time` and `SoundTime` could have been computed once as whole-session vectors (`ft − Trial_start_time[ft_trInd]`) and then sliced, and `collect_behavior`'s per-trial append into a list could have been a single masked gather. These are small next to the I/O.

ii.
```python
order = np.argsort(trials, kind='stable')
frames, trials = frames[order], trials[order]
splits = np.searchsorted(trials, np.arange(int(beh['ntrials']) + 1))
return {t: frames[splits[t]:splits[t + 1]] for t in range(int(beh['ntrials']))}
```
```python
for t in range(int(beh['ntrials'])):
    fr = frames[t]
    ...
    t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
    t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
```

iii. Not commented on by the AI; the vectorized `trial_frames` is evidently a deliberate implementation choice, and the remaining per-trial loop is where the arrays have to be split into the per-trial lists the output format requires.

## 12-c. What processing does the code repeat multiple times?

i. The behaviour pre-pass and the conversion pass both call `trial_frames` and `session_stimulus_names` for every session, and both apply the same `keep` predicate — the trial structure of all 89 sessions is therefore built twice (the two passes differ only in whether `nframes` comes from `beh['ft']` or from `spk.shape[1]`). `load_all_behavior` also loads all 23 behaviour files at once and holds them for the whole run, and with the `fork` pool each of the 8 workers inherits that whole structure. The lick vector, position vector and speed vector are each built once per session, which is correct.

ii.
```python
kept_trials, speed_edges = collect_behavior(sessions)   # calls trial_frames for all sessions
...
frames = trial_frames(beh, nframes)                     # and again inside convert_session
names = session_stimulus_names(entry)
```

iii. Not justified explicitly, but the second pass is structurally necessary: the speed quartiles have to be known before any trial can be written, and only the conversion pass knows the true imaging frame count. The duplicated work is behaviour-only and negligible against the spike I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three things. `collect_behavior` computes and returns `per_session`, the list of kept trial indices for every session, and it is used only to print a trial count — the conversion recomputes the same list. `sd` and the mean are computed over *all* recorded neurons, including the ~12% outside visual cortex and the ~95% that the subsample will throw away, when they are only needed for the kept rows. `session_info` stores a full `trial_indices` list (up to 722 integers) per session and `frame_period_s`, none of which the decoder reads. Minor extras: `np.ascontiguousarray` copies each trial slice, `experiment_types`/`cohort`/`rewarded_session` are bookkeeping only, and the module-level `TEXTURE_LENGTH_DM` and `SEED`-derived RNG per session are carried but barely used.

ii.
```python
kept_trials, speed_edges = collect_behavior(sessions)
print('speed quartile edges (cm/s):', speed_edges)
print('%d recordings, %d trials' % (len(recs), sum(len(v) for v in kept_trials.values())))
```
```python
sd = spk.std(axis=1)           # over all 20k-90k neurons, 2000 of which survive
```
```python
'trial_indices': trial_ids,
```

iii. Not addressed by the AI. The metadata extras are defensible as provenance — they make it possible to trace any converted trial back to its raw index, cohort and experiment type — and the cost of the rest is trivial next to reading 405 GB of traces.
