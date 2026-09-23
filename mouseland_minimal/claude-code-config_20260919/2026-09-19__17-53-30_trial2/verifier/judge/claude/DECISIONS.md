# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `Imaging_Exp_info.npy` as the master index, then loads all behavior files (`Beh_<exp_type>.npy`) upfront into a dict keyed by experiment type. It iterates over all entries in the index, building a per-recording dict (`sessions`) keyed by `mname_datexp_blk`. When the same recording appears under multiple experiment types, it merges the `UniqWalls -> stim_id` maps so every trial gets its canonical stimulus name. Neural spike data (`spk/<rec>_neural_data.npy`) and retinotopy (`retinotopy/<mname>_<datexp>_trans.npz`) are loaded per-session during conversion.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
beh_files = {et: np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % et),
                         allow_pickle=True).item() for et in exp_info}
# ...
rec = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
key = rec + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
```

```python
spk = load_spikes(rec)  # np.concatenate(...['spks'], 0)
iarea = load_areas(entry['mname'], entry['datexp'])
```

iii. The AI explored the data structure extensively (steps 17-59 of the trajectory), discovering how recordings are indexed and how behavior keys work. It merged stimulus maps across experiment types to recover stimulus identity for all trials.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the index entries. The sorted unique set of mouse names forms the subjects list; each session's subject is mapped to its index via `subjects.index(...)`.

ii.
```python
subjects = sorted({s['mname'] for s in sessions.values()})
'subject_idx': np.array([subjects.index(i['mouse']) for i in session_info], dtype=np.int64),
```

iii. Mouse names are taken directly from the index entries, which is the standard approach.

## 1-c. How are the data split into sessions?

i. A session is one unique recording, identified by `mname_datexp_blk`. When the same recording appears under multiple experiment types, it is converted only once (using `sessions.setdefault(rec, ...)`). This yields 89 sessions.

ii.
```python
rec = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
entry = sessions.setdefault(rec, {
    'mname': ndb['mname'], 'datexp': ndb['datexp'], 'blk': ndb['blk'],
    'exp_types': [], 'cohorts': set(), 'wall2stim': {}, 'beh': None})
```

iii. The AI recognized that the same recording can be listed under several experiment types and ensured each is converted exactly once.

## 1-d. How are the data split into trials?

i. Trials are identified by `ft_trInd` (frame-to-trial index). The AI groups frames by trial index, keeping only frames where `ft_CorrSpc` is True (inside the textured corridor) AND `ft_move > 0` (the animal was running). This follows the paper's "only timepoints during running" rule.

ii.
```python
def frame_mask(beh, nframes):
    corridor = beh['ft_CorrSpc'][:nframes]
    running = beh['ft_move'][:nframes] > 0
    in_trial = ~np.isnan(beh['ft_trInd'][:nframes])
    return corridor & running & in_trial
```

iii. The AI cites the paper's methods: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The released code uses `VRmove = beh['ft_move']>0` combined with `ft_CorrSpc`.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has fewer than `MIN_FRAMES_PER_TRIAL = 5` running frames in the corridor, or if its wall texture cannot be mapped to a canonical stimulus name (i.e., `names[t] is None`). This removes 309 of 38,110 trials.

ii.
```python
MIN_FRAMES_PER_TRIAL = 5
# ...
keep = [t for t in range(int(beh['ntrials']))
        if len(frames[t]) >= MIN_FRAMES_PER_TRIAL and names[t] is not None]
```

iii. The AI justified the minimum frame threshold as removing truncated trials at start/end of recording. The stimulus-name filter ensures every trial has a decodable stimulus label.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` (deconvolved calcium traces, one array per imaging plane, concatenated). Visual area from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
def load_spikes(rec):
    path = os.path.join(DATA_ROOT, 'spk', '%s_neural_data.npy' % rec)
    return np.concatenate(np.load(path, allow_pickle=True).item()['spks'], 0)

def load_areas(mname, datexp):
    path = os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
    return np.load(path, allow_pickle=True)['iarea']
```

iii. The AI reads the same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps beyond simply extracting trial frames: (1) filters to neurons in the four visual areas (V1/mHV/lHV/aHV), (2) removes silent neurons (std == 0), (3) z-scores each neuron over the whole recording, (4) subsamples to 2000 neurons per session, and stores as float32.

ii.
```python
region = np.full(nneurons_total, -1, dtype=np.int64)
for code, ridx in AREA_CODE_TO_REGION.items():
    region[iarea == code] = ridx
sd = spk.std(axis=1)
usable = np.where((region >= 0) & (sd > 0))[0]

rng = np.random.default_rng(SEED + session_index)
if nsub is not None and len(usable) > nsub:
    usable = np.sort(rng.choice(usable, size=nsub, replace=False))

activity = spk[usable]
activity = (activity - activity.mean(axis=1, keepdims=True)) / sd[usable][:, None]
activity = activity.astype(np.float32)
```

iii. The AI justified z-scoring by citing the paper's code (`utils.get_kfold_reward_response: spk = stats.zscore(spk, axis=1)`), arguing it puts sessions with different deconvolved amplitudes on a common scale. Subsampling to 2000 was justified by a pilot experiment showing comparable decoder accuracy at 1/23 the dataset size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if they belong to one of the four visual-area groups (V1, mHV, lHV, aHV) AND have a non-zero standard deviation (not silent). Additionally, a random subsample of 2000 neurons per session is taken.

ii.
```python
usable = np.where((region >= 0) & (sd > 0))[0]
if nsub is not None and len(usable) > nsub:
    usable = np.sort(rng.choice(usable, size=nsub, replace=False))
```

iii. The AI argues that the paper's code excludes neurons outside visual cortex (`iarea == -1` and `7`), and that silent neurons cannot be z-scored. Subsampling is a practical concession for dataset size.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). Only frames within the textured corridor where the animal was running are kept, so the trial starts at corridor entry and ends at corridor exit, with non-running frames removed. Variable-length trials with `off_start = 0.0` and `off_end = None`.

ii.
```python
neural.append(np.ascontiguousarray(activity[:, fr]))
```

Where `fr` are the frame indices from `trial_frames()` which filters by `ft_CorrSpc & (ft_move > 0)`.

iii. The AI notes that trials have variable numbers of frames (median 21, ~6.6 s of running) and that `off_end` is not a fixed time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is the imaging frame period (~315 ms at 3.17 Hz), computed as the median of actual frame periods across sessions.

ii.
```python
'time_bin_size': float(np.median([i['frame_period_s'] for i in session_info]) * 1000.0),
```

iii. The imaging frame is the finest resolution available, and all behavioral streams are already on the same grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the MATLAB datenum timestamp of the sound cue for each trial) and `ft` (the timestamp of every imaging frame).

ii.
```python
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
```

iii. The AI uses `SoundTime` directly rather than `SoundFr`, computing time differences in seconds via MATLAB datenum conversion (multiply by 86400).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the time to the sound cue is computed as `SoundTime[trial] - ft[frame_indices]`, converted from days to seconds by multiplying by 86400. This gives positive values before the cue and negative after.

ii.
```python
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
inp[0] = t_to_cue
```

iii. The AI computes the time difference directly from timestamps.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`fr`) used for the neural data of that trial, so alignment is inherent.

ii.
```python
fr = frames[t]
# ...
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. All data streams use the same frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the date string in the session entry), which is parsed into a `datetime.date` object. The first recording date of each mouse is used as the reference.

ii.
```python
def day_of_training(entry, first_day):
    date = datetime.date(*map(int, entry['datexp'].split('_')))
    return float((date - first_day).days)
```

```python
first_day = {}
for entry in sessions.values():
    date = datetime.date(*map(int, entry['datexp'].split('_')))
    m = entry['mname']
    first_day[m] = min(first_day.get(m, date), date)
```

iii. The AI notes that the dataset does not record the calendar date on which VR training started, so the first imaging session is used as the reference.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is computed as the number of calendar days elapsed since the first recording of that mouse. This is a float value broadcast across all frames of each trial.

ii.
```python
def day_of_training(entry, first_day):
    date = datetime.date(*map(int, entry['datexp'].split('_')))
    return float((date - first_day).days)
```

iii. This counts actual calendar days rather than recorded session count, which differs from the reference approach.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the MATLAB datenum of each trial's start) and `ft` (the timestamp of every imaging frame).

ii.
```python
t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
```

iii. The AI uses the trial's start timestamp directly rather than `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the time since start is computed as `ft[frame_indices] - Trial_start_time[trial]`, converted from days to seconds by multiplying by 86400. This gives positive values after the start.

ii.
```python
t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
inp[2] = t_since_start
```

iii. Simple timestamp difference computation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame indices (`fr`) as the neural data, so alignment is inherent.

ii.
```python
fr = frames[t]
t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. Same frame indexing as all other data streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks the trials in the rewarded corridor.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. Directly taken from the data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing beyond casting to float. The value is broadcast across all frames of the trial.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. No processing needed; `isRew` is a binary per-trial flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (the wall texture name per trial) and `UniqWalls`/`stim_id` (mapping unique walls to canonical stimulus identifiers). The AI merges stim_id maps across experiment types for each recording.

ii.
```python
def session_stimulus_names(entry):
    wall2stim = entry['wall2stim']
    return [STIM_NAMES[wall2stim[str(w)]] if str(w) in wall2stim else None
            for w in entry['beh']['WallName']]
```

```python
for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
    if not np.isnan(float(sid)):
        entry['wall2stim'][str(wall)] = int(sid)
```

iii. The AI uses the paper's stim_id mapping rather than directly parsing texture names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses 7 individual stimulus names (`circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`) rather than grouping them into 4 broad texture categories (circle, leaf, rock, wood). Each trial's stimulus is the index into `STIM_NAMES`.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
# ...
out[0] = STIM_NAMES.index(names[t])
```

```python
'output_values': [list(STIM_NAMES), ...]
```

iii. The AI argues that the 7 canonical identities are "central to the paper" and "the 2-way grouping is recoverable from it." However, the instructions say "Visual stimulus category. e.g. circle, leaf, etc." suggesting broader categories, and the data actually has textures like rock and wood that are absent from STIM_NAMES.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of every lick in the session.

ii.
```python
def lick_frames(beh, nframes):
    licks = np.zeros(nframes, dtype=bool)
    lick_fr = np.asarray(beh['LickFr'], dtype=float)
    lick_fr = lick_fr[np.isfinite(lick_fr)]
    if lick_fr.size:
        idx = np.round(lick_fr).astype(int)
        idx = idx[(idx >= 0) & (idx < nframes)]
        licks[idx] = True
    return licks
```

iii. Directly from `LickFr`, the same source as the reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A frame is 1 (lick) if at least one lick's frame number rounds to it, 0 (no lick) otherwise. The AI uses `np.round` to convert fractional frame numbers, filters out NaN/Inf values, and clips to valid frame range.

ii.
```python
lick_fr = np.asarray(beh['LickFr'], dtype=float)
lick_fr = lick_fr[np.isfinite(lick_fr)]
if lick_fr.size:
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < nframes)]
    licks[idx] = True
```

iii. The AI handles edge cases (NaN, out-of-range) more carefully than the reference.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick flag array is indexed by the same frame indices (`fr`) as the neural data.

ii.
```python
out[1] = licks[fr]
```

iii. Same frame-based alignment as all other streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame, in decimeters.

ii.
```python
pos = beh['ft_Pos'][:nframes]
# ...
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by `POSITION_BIN_DM = 10.0` (i.e., 1 m bins), truncated to int, and clipped to [0, 3] to produce 4 bins.

ii.
```python
POSITION_BIN_DM = 10.0
N_POSITION_BINS = 4
# ...
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
```

iii. Matches the instruction's "4 equal-length, 1-m-long spatial bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Division by 10 dm and truncation gives bins [0, 1, 2, 3] for positions [0-10), [10-20), [20-30), [30-40) dm, i.e., 0-1m, 1-2m, 2-3m, 3-4m. Values outside [0,3] are clipped.

ii.
```python
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
```

iii. Same approach as the reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Indexed by the same frame indices (`fr`) as neural data.

ii.
```python
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
```

iii. Same frame-based alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][:nframes]
```

iii. Same source as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartile edges (25th, 50th, 75th percentiles) pooled across ALL retained timepoints of ALL sessions. Then for each frame, the speed bin is assigned via `np.searchsorted(speed_edges, speed[fr])`, giving 4 bins.

ii.
```python
def collect_behavior(sessions):
    speeds = []
    # ... collects all speeds from kept frames ...
    edges = np.percentile(speeds, [25, 50, 75])
    return per_session, edges
```

```python
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
```

iii. The AI computes quartiles globally to ensure each bin holds 25% of all data points across the entire dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global percentile edges (25th, 50th, 75th) are computed over all retained frames. Each frame's speed is assigned to a bin via `searchsorted`, giving bins 0-3.

ii.
```python
edges = np.percentile(speeds, [25, 50, 75])
# ...
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
```

iii. The AI notes the edges are about 12.4 / 25.3 / 40.8 cm/s.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Indexed by the same frame indices (`fr`) as the neural data.

ii.
```python
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
```

iii. Same frame-based alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behavior arrays are truncated to `nframes` from the spike data, (2) NaN and Inf values in `LickFr` are filtered out, (3) lick frame indices outside [0, nframes) are dropped, (4) trials where the wall texture has no known stim_id mapping are dropped (`names[t] is None`), (5) trials with fewer than 5 running frames are dropped.

ii.
```python
lick_fr = lick_fr[np.isfinite(lick_fr)]
idx = idx[(idx >= 0) & (idx < nframes)]
# ...
if len(fr) < MIN_FRAMES_PER_TRIAL or names[t] is None:
    continue
```

iii. The AI's handling of edge cases is more explicit than the reference, particularly around NaN lick frames and unmapped stimuli.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (`spk/<session_id>_neural_data.npy`), which are large numpy files (total ~405 GB across all sessions). The AI also uses multiprocessing (default 6 workers) to parallelize session processing.

ii.
```python
def load_spikes(rec):
    path = os.path.join(DATA_ROOT, 'spk', '%s_neural_data.npy' % rec)
    return np.concatenate(np.load(path, allow_pickle=True).item()['spks'], 0)
```

```python
with mp.get_context('fork').Pool(args.workers) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs)):
```

iii. The I/O cost dominates; parallelization helps overlap loading across sessions.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_frames` function uses `np.where`, `np.argsort`, and `np.searchsorted` to group all frames by trial in one pass, which is already fairly vectorized. The `session_stimulus_names` function iterates per-trial with a list comprehension, but this is negligible.

ii.
```python
def trial_frames(beh, nframes):
    mask = frame_mask(beh, nframes)
    frames = np.where(mask)[0]
    trials = beh['ft_trInd'][:nframes][frames].astype(int)
    order = np.argsort(trials, kind='stable')
    frames, trials = frames[order], trials[order]
    splits = np.searchsorted(trials, np.arange(int(beh['ntrials']) + 1))
    return {t: frames[splits[t]:splits[t + 1]] for t in range(int(beh['ntrials']))}
```

iii. The frame grouping is done in a single vectorized pass, which is efficient.

## 12-c. What processing does the code repeat multiple times?

i. The `collect_behavior` function (first pass) computes trial frames and stimulus names for all sessions to determine speed quartile edges. Then `convert_session` recomputes `trial_frames` and `session_stimulus_names` for each session during the actual conversion. This duplication is deliberate: the first pass is behavior-only (no spike loading), while the second pass processes each session fully.

ii.
```python
# First pass (collect_behavior):
frames = trial_frames(beh, nframes)
names = session_stimulus_names(entry)

# Second pass (convert_session):
frames = trial_frames(beh, nframes)
names = session_stimulus_names(entry)
```

iii. The two-pass design is intentional to avoid loading all spike data at once.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI subsamples to 2000 neurons per session, which discards 90-98% of recorded neurons. Additionally, the z-scoring and silent-neuron removal add processing overhead. The two-pass behavior collection also repeats work. The AI computes and stores extensive metadata per session (`session_info`) including experiment types, cohort labels, trial indices, and frame periods.

ii.
```python
if nsub is not None and len(usable) > nsub:
    usable = np.sort(rng.choice(usable, size=nsub, replace=False))
```

iii. The subsampling is the most significant — it loads all neurons from disk only to discard most of them.
