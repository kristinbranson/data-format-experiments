# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as the master index, builds a `session_map` keyed by `mname_datexp_blk`, caches each `Beh_<exp_type>.npy` behavior dictionary, then loads spikes per session from `spk/<session>_neural_data.npy` and retinotopy per session from `retinotopy/<mouse>_<date>_trans.npz`.

ii. 
```python
exp_info = np.load(
    os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'),
    allow_pickle=True
).item()
session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))
```
```python
if exp_type not in beh_cache:
    beh_path = os.path.join(data_root, 'beh', f'Beh_{exp_type}.npy')
    beh_cache[exp_type] = np.load(beh_path, allow_pickle=True).item()
```
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
return dtrans['iarea']
```

iii. `CONVERSION_NOTES.md` says the source data are `data/beh`, `data/spk`, and `data/retinotopy`, with `Imaging_Exp_info.npy` as the experiment index.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the sorted unique mouse IDs extracted from the session keys; `subject_idx` uses the first underscore-delimited token of each session key.

ii.
```python
all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
```
```python
subject_idx_list.append(mouse_to_idx[mname])
```

iii. The notes describe each recording as belonging to one mouse, and the code uses mouse name as the subject identifier.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `mname_datexp_blk` recording. If the same recording appears in multiple experiment types, the agent does not keep the first occurrence; it prefers the entry with the largest non-NaN `stim_id` count.

ii.
```python
key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
```
```python
if key not in session_map:
    session_map[key] = (exp_type, beh_key, ndb)
else:
    old_ndb = session_map[key][2]
    old_nstim = np.sum(~np.isnan(old_ndb['stim_id'].astype(float)))
    new_nstim = np.sum(~np.isnan(ndb['stim_id'].astype(float)))
    if new_nstim > old_nstim:
        session_map[key] = (exp_type, beh_key, ndb)
```

iii. `CONVERSION_NOTES.md` explicitly says duplicate recordings are resolved by selecting the experiment-type entry with “the most stimuli”.

## 1-d. How are the data split into trials?

i. The agent uses `beh['ntrials']` as the trial count and creates one 60-bin position-normalized trial for every trial index. It does not derive trial frame windows from `ft_trInd` and `ft_CorrSpc`.

ii.
```python
ntrials = beh['ntrials']
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove],
    ft_AcumPos[VRmove],
    ntrials,
    n_bins=int(CL),
    lengths=CL
)
```
```python
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The notes justify this as “spatial interpolation into 60 bins per trial (40 corridor + 20 gray), each 1 dm”.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality filter beyond skipping entire sessions with fewer than 2 trials. All remaining trial indices are kept.

ii.
```python
if ntrials < 2:
    print(f"    Skipping: only {ntrials} trials")
    skipped_sessions.append(sess_key)
    continue
```
```python
for trial in range(ntrials):
    ...
```

iii. No separate trial-quality justification was found beyond the notes’ claim that all sessions should have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from `spks` in the per-session spike file, concatenated across imaging planes, and `iarea` in the retinotopy file for neuron region labels/filtering.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
return dtrans['iarea']
```

iii. `CONVERSION_NOTES.md` says the script follows the reference in using deconvolved fluorescence traces and retinotopy-based area assignments.

## 2-b. How is the `neural` data processed?

i. The agent filters to running frames (`ft_move > 0`), uses cumulative position (`ft_PosCum`) to interpolate each neuron onto 60 spatial bins per trial, and stores each trial as `float32`.

ii.
```python
VRmove = beh['ft_move'][:nfr] > 0
ft_AcumPos = beh['ft_PosCum'][:nfr]
```
```python
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove],
    ft_AcumPos[VRmove],
    ntrials,
    n_bins=int(CL),
    lengths=CL
)
```
```python
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The notes justify this by citing the paper statement that “We only considered timepoints during running for analysis” and by treating 1-dm spatial bins as the analysis grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept if `iarea` is not `-1` and not `7`, then sessions are skipped if fewer than 10 such neurons remain.

ii.
```python
valid_neuron_mask = (iarea != -1) & (iarea != 7)
n_valid = valid_neuron_mask.sum()
```
```python
if n_valid < 10:
    print(f"    Skipping: only {n_valid} valid neurons")
    skipped_sessions.append(sess_key)
    continue
```

iii. The notes say this is intended to exclude non-visual-cortex or unassigned neurons; no justification is given for the extra `n_valid < 10` session filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent says alignment is at trial start/corridor entry, but operationally the neural data are indexed by 60 position bins spanning the full corridor plus gray space after interpolation from running frames.

ii.
```python
time_bin_sec = 1.0 / 6.0
n_bins = 60
```
```python
interp_spk = get_interpPos_spk(..., ntrials, n_bins=int(CL), lengths=CL)
...
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
```

iii. The notes justify this by treating corridor entry as bin 0 of a 60-bin position axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent treats each 1-dm spatial bin as `1/6` s, i.e. `166.67 ms`, and rebins neural data by spatial interpolation into 60 bins per trial.

ii.
```python
n_bins = 60
time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms
```
```python
'time_bin_size': time_bin_sec * 1000,
```

iii. The notes explicitly argue that constant VR speed makes 1-dm spatial bins “equivalent to temporal binning when the VR is moving”.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The code derives this variable from `SoundPos` and a synthetic position axis `positions = np.arange(n_bins)`, not from frame times.

ii.
```python
SoundPos = beh['SoundPos']  # sound cue position in dm
positions = np.arange(n_bins)
```

iii. The notes say time to cue is computed as `(SoundPos - position) * (1/6)`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the agent subtracts the spatial-bin index from the cue position and multiplies by `1/6` s per bin, making the value positive before the cue and negative after.

ii.
```python
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec
```

iii. `CONVERSION_NOTES.md` gives this exact formula and explains it via constant VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned to the same 60 spatial bins as the interpolated neural trial.

ii.
```python
input_trial = np.stack([
    time_to_cue.astype(np.float32),
    day_array,
    time_since_start.astype(np.float32),
    reward_avail
], axis=0)
```
```python
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The notes justify this by representing all trial-varying signals on the same spatial grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and `datexp` in `Imaging_Exp_info.npy`.

ii.
```python
key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')
mouse_dates[ndb['mname']].append(date)
```

iii. The notes say day of training is computed from dates in the experiment metadata.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The agent computes calendar-day offset from each mouse’s first recorded session, then broadcasts that scalar across every time bin of each trial.

ii.
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
...
session_days[key] = (date - mouse_first_date[mname]).days
```
```python
day_array = np.full(n_bins, day, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says this is “Days since the mouse's first recording session”.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The code derives it from the synthetic position axis and the assumed constant VR speed, not from `StartFr` and frame timestamps.

ii.
```python
positions = np.arange(n_bins)
time_since_start = positions * time_bin_sec
```

iii. The notes describe this as `position * (1/6)`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent multiplies each spatial-bin index by `1/6` s, producing a deterministic `0` to `9.83` s ramp for every trial.

ii.
```python
time_since_start = positions * time_bin_sec
```

iii. `CONVERSION_NOTES.md` states this exact formula and range.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned to the same 60 position bins used for the neural interpolation.

ii.
```python
input_trial = np.stack([
    time_to_cue.astype(np.float32),
    day_array,
    time_since_start.astype(np.float32),
    reward_avail
], axis=0)
```

iii. The notes treat trial start as position 0 on the shared spatial grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `beh['isRew']`.

ii.
```python
isRew = beh['isRew']
```

iii. The notes explicitly identify `beh['isRew']` as the source.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is cast to float and broadcast across all 60 bins of the trial.

ii.
```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. The notes say this is a binary per-trial variable from `isRew`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh['WallName']`.

ii.
```python
WallName = beh['WallName']
stim_idx = stim_to_idx[str(WallName[trial])]
```

iii. The notes say the visual stimulus category comes from `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent gathers all unique `WallName` strings across the dataset, maps each trial’s exact `WallName` to an index in that 15-item list, and broadcasts the result across all bins of the trial.

ii.
```python
all_stim_names_set = set()
...
for name in np.unique(beh_data[beh_key_inner]['WallName']):
    all_stim_names_set.add(str(name))
all_stim_names = sorted(all_stim_names_set)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```
```python
stim_idx = stim_to_idx[str(WallName[trial])]
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` justifies this as preserving the full stimulus set rather than collapsing to broader texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickPos` and `LickTrind`, not from frame-based lick times.

ii.
```python
lick_spatial = make_lick_spatial_bins(
    beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL)
)
```

iii. The notes explicitly say licking is “mapped from `LickPos` and `LickTrind` to spatial bins”.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick is assigned to a spatial bin by `floor(position)`, clipped into `[0, 59]`, and the corresponding trial/bin entry is set to `1`.

ii.
```python
lick_arr = np.zeros((ntrials, n_bins), dtype=np.int64)
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
    if 0 <= tr < ntrials:
        lick_arr[tr, bin_idx] = 1
```

iii. The notes justify this by representing licking on the same 60-bin spatial grid as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by trial and spatial bin, not by frame number.

ii.
```python
lick_trial = lick_spatial[trial, :].astype(np.int64)
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The notes say all decoder variables are expressed in the shared position-binned trial representation.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The actual code does not read `ft_Pos` for this output. It derives position categories from a fixed bin index `positions = np.arange(n_bins)` and the assumed 60-bin corridor length.

ii.
```python
n_bins = 60
positions = np.arange(n_bins)
CL = beh['Corridor_Length']
```

iii. The notes describe the output as deterministic from the chosen spatial binning rather than from per-frame position samples.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 60 spatial bins are digitized into 5 categories: four 1-m corridor bins plus a fifth gray-space category.

ii.
```python
position_bins_edges = [10, 20, 30, 40]
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
```
```python
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` justifies this by retaining the full 4 m corridor plus 2 m gray space spatial structure.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are `[10, 20, 30, 40]` decimeter bins, producing categories `[0-10), [10-20), [20-30), [30-40), [40-60)`.

ii.
```python
position_bins_edges = [10, 20, 30, 40]
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. The notes describe the fifth category as gray space rather than padding or “none”.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned to the same 60 spatial bins as the interpolated neural data and is identical across trials.

ii.
```python
pos_trial = pos_category.copy()
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The notes treat spatial bin index as the common alignment axis for the whole dataset.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['run_pos']`, the per-trial, per-position running-speed array.

ii.
```python
run_pos = beh['run_pos']    # (ntrials, 60) running speed at each position
```

iii. `CONVERSION_NOTES.md` says the quartiles are computed from `beh['run_pos']`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent concatenates `run_pos` from all sessions, computes global percentile edges at 25/50/75%, and digitizes each trial’s 60-bin speed profile using those edges.

ii.
```python
all_speeds = np.concatenate([rp.ravel() for rp in run_pos_all])
all_speeds = all_speeds[~np.isnan(all_speeds)]
edges = np.percentile(all_speeds, [25, 50, 75])
```
```python
speed_bins = speed_to_bins(run_pos, speed_edges)
```

iii. The notes justify this as making each bin contain about 25% of the global dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global percentile edges returned by `np.percentile`, and `np.digitize` assigns categories `0` to `3`.

ii.
```python
edges = np.percentile(all_speeds, [25, 50, 75])
result = np.digitize(run_pos, edges)
```

iii. The notes say the quartile edges are computed globally across all 89 sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned per trial on the same 60 spatial bins used for the interpolated neural data.

ii.
```python
speed_trial = speed_bins[trial, :].astype(np.int64)
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The notes justify this by using the same position-binned representation for all time-varying variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code skips sessions if neural or retinotopy files fail to load, skips sessions with fewer than 2 trials or fewer than 10 “valid” neurons, removes NaN/Inf from interpolated neural trials with `np.nan_to_num`, and otherwise does not explicitly reconcile behavior that extends past imaging.

ii.
```python
try:
    spk = load_spk(...)
except Exception as e:
    ...
    continue
...
try:
    iarea = load_retino(...)
except Exception as e:
    ...
    continue
```
```python
if ntrials < 2:
    ...
if n_valid < 10:
    ...
```
```python
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes mention NaN/Inf cleanup and file-integrity checks, but they do not document the reference-style truncation of behavior to imaged frames.

## 12-a. What are the most time-consuming steps of the code?

i. The heavy steps are loading the large spike files, interpolating neural activity neuron-by-neuron into 60 spatial bins, and the first pass over all sessions to collect running speeds for global quartiles.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
for s in range(raw_spk.shape[0]):
    spk_resh.append(np.reshape(
        interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos),
        (int(new_shape[0]), int(new_shape[1]))
    ))
```
```python
for sess_key in all_sessions_full:
    ...
    run_pos_all.append(beh['run_pos'])
```

iii. No explicit timing justification was documented, but these are the obvious dominant costs from the code structure.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron loop in `spk_pos_interp`, the lick loop in `make_lick_spatial_bins`, and the per-trial assembly loop are all scalar or Python-loop heavy and could be vectorized or batched further.

ii.
```python
for s in range(raw_spk.shape[0]):
    spk_resh.append(...)
```
```python
for i in range(len(lick_pos)):
    ...
```
```python
for trial in range(ntrials):
    ...
```

iii. No explicit vectorization discussion was documented.

## 12-c. What processing does the code repeat multiple times?

i. The code makes a first full pass over all sessions to gather `run_pos` for speed quartiles, a separate scan over all behavior files to collect unique stimulus names, and then a second pass to build the actual dataset.

ii.
```python
for sess_key in all_sessions_full:
    ...
    run_pos_all.append(beh['run_pos'])
```
```python
for exp_type_key in exp_info.keys():
    ...
    for beh_key_inner in beh_data:
        for name in np.unique(beh_data[beh_key_inner]['WallName']):
            all_stim_names_set.add(str(name))
```
```python
for sess_idx, sess_key in enumerate(all_sessions):
    ...
```

iii. No explicit justification was documented beyond the need to compute global quartiles and global stimulus labels.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints summary-only quantities (`total_trials`, `total_neurons`, `neurons_per_session`), defines an unused `n_position_categories`, and performs dataset-wide stimulus-name collection and label formatting that are only used as metadata labels rather than analysis content.

ii.
```python
n_position_categories = 5  # 4 corridor bins + gray space
```
```python
all_stim_names_set = set()
...
all_stim_names = sorted(all_stim_names_set)
```
```python
total_trials = sum(len(s) for s in neural_all)
total_neurons = sum(brain_region_idx_all[i].shape[0] for i in range(len(neural_all)))
neurons_per_session = [brain_region_idx_all[i].shape[0] for i in range(len(neural_all))]
```

iii. No explicit justification was documented for these extra computations.
