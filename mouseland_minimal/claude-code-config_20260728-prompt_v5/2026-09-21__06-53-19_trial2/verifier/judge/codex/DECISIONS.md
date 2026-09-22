# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master session index from `beh/Imaging_Exp_info.npy`, then iterates over every experiment type and loads each matching behavior file (`Beh_<exp_type>.npy`) to build a `sessions` dictionary keyed by `mname_datexp_blk`. Each unique session stores its behavior struct in memory immediately. Later, each session loads its spike file from `spk/` and retinotopy file from `retinotopy/` during `process_session`.

ii. ```python
def collect_sessions():
    exp_info = np.load(os.path.join(BEH_ROOT, 'Imaging_Exp_info.npy'), allow_pickle=True).item()

    sessions = {}
    for exp_type in exp_info:
        beh_file = os.path.join(BEH_ROOT, f'Beh_{exp_type}.npy')
        Beh = np.load(beh_file, allow_pickle=True).item()

        for ndb in exp_info[exp_type]:
            session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if session_key in sessions:
                continue
            ...
            if beh_key in Beh:
                sessions[session_key] = {
                    'ndb': ndb,
                    'beh': Beh[beh_key],
                    'exp_type': exp_type,
                }
```
```python
def load_spk(ndb):
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    return spk
```

iii. In the trajectory the agent explicitly decided to include all 89 unique sessions across experiment types to maximize data. It reasoned that `reward_availability` and `day_of_training` would let the decoder absorb differences across supervised, unsupervised, naive, and grating sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name. For each session the agent reads `ndb['mname']`, appends new mice to `subjects_list` in first-seen order over sorted session keys, and stores `subject_idx` per session.

ii. ```python
for i, session_key in enumerate(sorted_keys):
    info = sessions[session_key]
    mname = info['ndb']['mname']
    ...
    if mname not in subject_to_idx:
        subject_to_idx[mname] = len(subjects_list)
        subjects_list.append(mname)
    session_subject_idx.append(subject_to_idx[mname])
```

iii. The trajectory treated `mname` as the subject identifier from the outset and did not describe any alternative split.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` triple. The agent forms `session_key = "<mouse>_<date>_<block>"` and keeps only the first occurrence when the same recording appears under multiple experiment types.

ii. ```python
for ndb in exp_info[exp_type]:
    session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
    if session_key in sessions:
        continue
```

iii. The trajectory states that some recordings reappear under different experiment types but the neural data are the same, so each recording should be used only once.

## 1-d. How are the data split into trials?

i. Trials are split using the per-frame trial index `ft_trInd` intersected with the corridor mask `ft_CorrSpc`. This makes each trial equal to the texture-corridor traversal only; gray-space frames are excluded. The code keeps the original variable trial lengths.

ii. ```python
for trial_idx in range(ntrials):
    mask = (ft_trInd == trial_idx) & ft_CorrSpc
    frame_indices = np.where(mask)[0]
```

iii. The trajectory repeatedly justifies this as matching the 4 m texture corridor and the requested 4 one-meter position bins, while preserving contiguous time series for the decoder.

## 1-e. How are trials filtered based on quality controls?

i. The only explicit trial-quality filter is `len(frame_indices) < 2`, which drops trials with fewer than two corridor frames. Entire sessions are skipped if fewer than two valid trials remain. The agent does not implement the reference long-trial outlier filter.

ii. ```python
for trial_idx in range(ntrials):
    mask = (ft_trInd == trial_idx) & ft_CorrSpc
    frame_indices = np.where(mask)[0]

    if len(frame_indices) < 2:
        continue
...
if len(neural_trials) < 2:
    return None
```

iii. The trajectory discussed keeping all corridor frames, including non-running frames, to preserve contiguous samples. It did not provide a separate justification for omitting the long-trial percentile filter used by the reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from `spks` in each session's `*_neural_data.npy` file, concatenated across imaging planes. Brain-region labels come from `iarea` in the matching retinotopy `.npz` file.

ii. ```python
spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
...
ret = np.load(os.path.join(RET_ROOT, fn))
brain_reg_idx = map_brain_regions(ret['iarea'])
```

iii. The trajectory explicitly chose deconvolved fluorescence traces from `spks` because the methods say all analyses were based on deconvolved traces.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: the agent slices each trial's columns out of the concatenated `spk` array and casts each trial to `float32`. It does not pad, smooth, normalize, or rebin the traces.

ii. ```python
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The trajectory says to use the native deconvolved traces directly and preserve contiguous native-frame bins for the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural quality control is limited to assigning each neuron a grouped visual-area label. Unlike the reference solution, the agent does not drop neurons outside `V1`, `mHV`, `lHV`, and `aHV`; instead it keeps them in an extra `unassigned` category.

ii. ```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
BRAIN_REGION_MAP = {
    8: 0,
    0: 1, 1: 1, 2: 1, 9: 1,
    5: 2, 6: 2,
    3: 3, 4: 3,
    -1: 4, 7: 4,
}

def map_brain_regions(iarea):
    return np.array([BRAIN_REGION_MAP.get(int(ia), 4) for ia in iarea], dtype=np.int64)
```

iii. The trajectory says this was done to preserve more neurons, using an `unassigned` fallback rather than discarding them.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to corridor entry in the sense that each trial begins at the first kept corridor frame for that trial and runs through the texture portion only. The arrays remain variable length.

ii. ```python
mask = (ft_trInd == trial_idx) & ft_CorrSpc
frame_indices = np.where(mask)[0]
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The trajectory explicitly chose corridor entry / trial start as the alignment event and rejected running-only subsampling because it would introduce temporal gaps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use one imaging frame per time bin. The bin size is estimated as the median `np.diff(ft)` from the first session and stored in milliseconds. No temporal rebinning or resampling is applied.

ii. ```python
first_key = sorted(sessions.keys())[0]
ft = sessions[first_key]['beh']['ft']
dt_seconds = float(np.median(np.diff(ft)) * 86400)
dt_ms = dt_seconds * 1000
...
'time_bin_size': dt_ms,
```

iii. The trajectory checked the frame rate empirically and concluded that imaging frames are already evenly spaced at about 315 ms, so native frames should be the decoder bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `time_to_sound_cue` is derived from `SoundFr` and the kept neural frame indices. The code does not use `ft` timestamps for the actual per-trial calculation.

ii. ```python
SoundFr = beh['SoundFr']
...
time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. The trajectory says the agent intended to rely on floating-point `SoundFr` values directly rather than rounding them.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the agent subtracts the integer frame index from the floating-point cue frame `SoundFr[trial_idx]`, multiplies by a constant `dt_seconds`, and stores the result as a continuous vector. No interpolation onto `ft` is done.

ii. ```python
time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. The trajectory justifies native-frame timing and constant frame spacing, but it does not explain the departure from the reference interpolation-based computation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The cue-timing vector is aligned by using exactly the same `frame_indices` array used to slice the neural data for that trial.

ii. ```python
neural_trial = spk[:, frame_indices].astype(np.float32)
...
time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. The trajectory consistently treats all per-frame outputs and inputs as living on the neural frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `day_of_training` is derived from the per-session metadata in `ndb`, specifically the mouse name `mname` and the date string `datexp`.

ii. ```python
def parse_date(datexp):
    return datetime.strptime(datexp, '%Y_%m_%d')

def compute_days_of_training(sessions):
    mouse_dates = {}
    for session_key, info in sessions.items():
        mname = info['ndb']['mname']
        dt = parse_date(info['ndb']['datexp'])
```

iii. The trajectory explicitly says date parsing looked more reliable than using fields like `sess#` or other inconsistent session counters.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The agent computes day of training as calendar days since each mouse's first recording date. It finds the earliest `datexp` for each mouse, subtracts that date from each session date, and broadcasts the resulting integer day count across each trial.

ii. ```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}

return {
    sk: (parse_date(info['ndb']['datexp']) - mouse_first_date[info['ndb']['mname']]).days
    for sk, info in sessions.items()
}
```
```python
day_train = np.full(T, day_of_training, dtype=np.float32)
```

iii. The trajectory explicitly justifies this as “calendar days from each mouse's first recording date.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The implemented `time_since_trial_start` is derived from the kept frame indices for that trial, not from `StartFr` or `ft`. Although `StartFr` is loaded, it is not used in the computation.

ii. ```python
StartFr = beh['StartFr']
...
time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. The trajectory says the intended alignment event was corridor entry (`StartFr`), but the code actually anchors to the first kept corridor frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Processing is a simple zeroing relative to the first kept frame of the trial: subtract `frame_indices[0]` from every kept frame index and multiply by the constant frame duration. This ignores fractional `StartFr` values.

ii. ```python
time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. The trajectory discusses alignment to trial start, but the implementation is a coarser approximation using the first selected corridor frame.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The time-since-start vector is aligned by being computed on the same `frame_indices` used for the neural trial slice, so it has the same length and bin order as the neural data.

ii. ```python
neural_trial = spk[:, frame_indices].astype(np.float32)
...
time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. The trajectory consistently frames alignment in terms of using one common per-frame index for neural and behavioral streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `reward_availability` is derived directly from the per-trial boolean `isRew` array in the behavior struct.

ii. ```python
isRew = beh['isRew']
...
reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)
```

iii. The trajectory treated reward availability as the session-provided rewarded-corridor indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No additional computation is done beyond converting the trial's `isRew` value to float and broadcasting it across all time bins of that trial.

ii. ```python
reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)
```

iii. The trajectory used this variable mainly to justify why all experiment types could be pooled in one dataset.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The visual-stimulus output is derived from `WallName`, the per-trial corridor texture name.

ii. ```python
WallName = beh['WallName']
...
stim_idx = STIM_TO_IDX[str(WallName[trial_idx])]
```

iii. The trajectory notes that `WallName` reflects the actual trial texture and was safer than relying on experiment-type-specific stimulus metadata.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent encodes the raw wall names as 15 separate categories using `ALL_STIMULI` and `STIM_TO_IDX`. It does not collapse variants like `circle1`, `circle2`, and `circle3` into the four broad categories used by the reference solution.

ii. ```python
ALL_STIMULI = sorted([
    'circle1', 'circle2', 'circle3',
    'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3',
    'rock1', 'rock2',
    'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5',
])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
...
stim_idx = STIM_TO_IDX[str(WallName[trial_idx])]
stim_out = np.full(T, stim_idx, dtype=np.int64)
```

iii. The trajectory explicitly reasoned that because the dataset contains many named wall textures, it should “use all these as possible values.”

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind`. The code first selects licks belonging to the current trial and then maps their frame locations onto the kept trial frames.

ii. ```python
def compute_lick_per_frame(beh, trial_idx, frame_indices):
    lick_mask = beh['LickTrind'] == trial_idx
    ...
    lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
```

iii. The trajectory explicitly chose to use both lick frame locations and trial indices when deriving the binary licking output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code rounds each lick's floating-point `LickFr` to the nearest integer frame, puts those frames in a set, and then emits a binary vector indicating whether each kept frame index appears in that set.

ii. ```python
def compute_lick_per_frame(beh, trial_idx, frame_indices):
    lick_mask = beh['LickTrind'] == trial_idx
    if not np.any(lick_mask):
        return np.zeros(len(frame_indices), dtype=np.float32)

    lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
    return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

iii. The trajectory says the agent wanted a binary “lick happened in this imaging frame” signal and chose per-frame assignment rather than keeping counts.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking vector is aligned by evaluating lick occurrence only on the same `frame_indices` used for the neural slice of that trial.

ii. ```python
lick_out = compute_lick_per_frame(beh, trial_idx, frame_indices).astype(np.int64)
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The trajectory repeatedly states that all behavioral streams should be projected onto the neural imaging frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position in corridor is derived from `ft_Pos`, the per-frame position signal.

ii. ```python
ft_Pos = beh['ft_Pos'][:nfr]
...
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. The trajectory explicitly tied this to the 4 m corridor and the requested 1 m position bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Processing consists of taking the kept per-frame positions for each trial, integer-dividing by 10 decimeters, clipping to `[0, 3]`, and storing the result as categorical integers.

ii. ```python
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. The trajectory justifies restricting trials to corridor frames so these bins naturally cover only the 4 m texture portion.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholding into categories is done with fixed spatial boundaries at 0, 10, 20, 30, and 40 decimeters, yielding four equal-length 1 m bins encoded as 0 to 3.

ii. ```python
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. The trajectory consistently argues for four equal spatial bins covering the texture corridor only.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by indexing `ft_Pos` with the same `frame_indices` used for the neural trial matrix.

ii. ```python
neural_trial = spk[:, frame_indices].astype(np.float32)
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. The trajectory treats `ft_Pos` as already sampled on the neural frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`, the per-frame running-speed trace.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
```

iii. The trajectory always treats running speed as a framewise behavioral output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent computes global running-speed quartile edges across all corridor frames from all sessions, then uses those edges to bin each frame's speed. It uses numeric percentiles, not rank-based equal-count bins within session.

ii. ```python
def compute_speed_quartiles(sessions):
    all_speeds = []
    for session_key in sorted(sessions.keys()):
        ...
        corridor_mask = ft_CorrSpc & ~np.isnan(ft_trInd)
        all_speeds.append(ft_RunSpeed[corridor_mask])

    all_speeds = np.concatenate(all_speeds)
    q25, q50, q75 = np.percentile(all_speeds, [25, 50, 75])
    return np.array([q25, q50, q75])
```

iii. The trajectory explicitly says it wanted quartiles computed across all corridor frames in all sessions, including zero-speed frames.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The actual thresholding uses `np.digitize` with the three global percentile cut points returned by `compute_speed_quartiles`, producing categories 0 to 3.

ii. ```python
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. The trajectory frames this as a straightforward quartile-thresholding step after the global percentile pass.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by reading `ft_RunSpeed` at the same `frame_indices` used for the neural trial.

ii. ```python
neural_trial = spk[:, frame_indices].astype(np.float32)
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. The trajectory consistently keeps running speed on the same frame grid as neural activity.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles minor inconsistencies mainly by trimming framewise behavior arrays to the neural frame count `nfr`, skipping trials with fewer than two kept frames, and only adding sessions that actually contain the expected behavior key. It does not add a broader missing-data repair layer.

ii. ```python
ft_trInd = beh['ft_trInd'][:nfr]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
ft_Pos = beh['ft_Pos'][:nfr]
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
...
if len(frame_indices) < 2:
    continue
...
if beh_key in Beh:
    sessions[session_key] = { ... }
```

iii. The trajectory mostly treats the dataset as clean. Its main explicit robustness discussion was about preserving fixed time bins and not relying on noisier or inconsistent metadata fields when dates could be parsed directly.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading large spike files and doing the extra pass used for global speed quartiles. The trajectory explicitly notes that loading neural data dominated runtime and memory use.

ii. ```python
def get_nfr(ndb):
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    nfr = spk_data['spks'][0].shape[1]
    return nfr
```
```python
def load_spk(ndb):
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    return spk
```

iii. In the trajectory the agent explicitly says the script is slow because it is “loading all the neural data files,” and later notes that the speed-quartile pass was wasting time and memory by loading spike files only to read frame counts.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loops are the per-frame list comprehension in `compute_lick_per_frame` and the Python-level list comprehension in `map_brain_regions`. Both could be replaced by array operations.

ii. ```python
def map_brain_regions(iarea):
    return np.array([BRAIN_REGION_MAP.get(int(ia), 4) for ia in iarea], dtype=np.int64)
```
```python
return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

iii. The trajectory does not dwell on vectorization beyond general runtime concerns, but these are the obvious remaining Python loops in the agent's own implementation.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats spike-file loading: once in `get_nfr` during the global speed-quartile pass and again in `load_spk` during actual session conversion. It also scans all sessions once to collect global speed thresholds and then scans them again to build the final dataset.

ii. ```python
def compute_speed_quartiles(sessions):
    for session_key in sorted(sessions.keys()):
        ...
        nfr = get_nfr(ndb)
```
```python
def process_session(session_key, info, day_of_training, speed_quartiles, dt_seconds):
    spk = load_spk(ndb)
```

iii. The trajectory explicitly recognizes this repetition and calls out that speed-quartile computation was loading full neural files before the real conversion pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is some unnecessary processing that does not affect downstream decoding: `StartFr` is loaded in `process_session` but never used, `n_neurons` is computed but unused, and the code constructs extra metadata such as `session_keys` and `speed_quartile_edges` that the decoder does not consume. The extra `get_nfr` pass also exists only to support the chosen global speed-thresholding strategy.

ii. ```python
n_neurons, nfr = spk.shape
...
StartFr = beh['StartFr']
```
```python
'metadata': {
    ...
    'session_keys': session_keys_ordered,
    'speed_quartile_edges': speed_quartiles.tolist(),
    'stimulus_names': ALL_STIMULI,
}
```

iii. The trajectory never presents these as important downstream requirements; they are side effects of the implementation choices rather than necessary decoder inputs.
