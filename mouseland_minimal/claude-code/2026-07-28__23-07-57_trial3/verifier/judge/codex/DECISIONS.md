# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads the master index from `beh/Imaging_Exp_info.npy`, builds a `session_map` keyed by `mname_datexp_blk`, then uses that map to pick one behavior entry per recording, load spikes from `spk/<session>_neural_data.npy`, and load retinotopy from `retinotopy/<mouse>_<date>_trans.npz`. Unlike the reference, duplicate metadata entries are resolved by preferring the entry with the largest non-NaN `stim_id` count.

ii. ```python
exp_info = np.load(
    os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'),
    allow_pickle=True
).item()
session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))
...
spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
```
```python
if key not in session_map:
    session_map[key] = (exp_type, beh_key, ndb)
else:
    old_nstim = np.sum(~np.isnan(old_ndb['stim_id'].astype(float)))
    new_nstim = np.sum(~np.isnan(ndb['stim_id'].astype(float)))
    if new_nstim > old_nstim:
        session_map[key] = (exp_type, beh_key, ndb)
```

iii. In trajectory steps 28 and 40, the agent said behavior was duplicated across experiment types and concluded it could use any one copy, then added a heuristic to keep the version with the "most stimuli" as the "more complete" record.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique mouse names parsed from the session keys, sorted, and then each session gets a `subject_idx` from that sorted list.

ii. ```python
all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
...
subject_idx_list.append(mouse_to_idx[mname])
```

iii. In steps 25 and 40, the agent treated each `mname_datexp_blk` recording as belonging to one mouse, so the subject split followed `mname`.

## 1-c. How are the data split into sessions?

i. A session is one unique `mname_datexp_blk` recording. The code deduplicates repeated metadata entries for the same key and processes each surviving key once.

ii. ```python
key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
...
all_sessions = sorted(session_map.keys())
for sess_idx, sess_key in enumerate(all_sessions):
    exp_type, beh_key, ndb = session_map[sess_key]
```

iii. In steps 25, 28, and 40, the agent explicitly defined each unique recording session as one `mname_datexp_blk` combination and noted that duplicate experiment-type listings should collapse to one session.

## 1-d. How are the data split into trials?

i. Trials are taken from `beh['ntrials']`, and the code assumes each trial can be represented as 60 fixed spatial bins after interpolation. It does not use `ft_trInd` or frame windows to recover per-frame trial boundaries; instead, it reshapes interpolated running-frame data into `(n_trials, 60)`.

ii. ```python
ntrials = beh['ntrials']
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove],
    ft_AcumPos[VRmove],
    ntrials,
    n_bins=int(CL),
    lengths=CL
)
...
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. In steps 25, 28, and 40, the agent decided trials should be corridor traversals represented by 60 spatial bins because it believed constant VR speed made spatial binning the right way to handle variable trial lengths.

## 1-e. How are trials filtered based on quality controls?

i. The script does not apply per-trial quality control analogous to the reference. It keeps all `ntrials` once a session is accepted, with session-level skips only for sessions with fewer than 2 trials, missing files, or fewer than 10 valid neurons.

ii. ```python
if ntrials < 2:
    ...
if n_valid < 10:
    ...
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The trajectory focuses on session-level validity and storage cost, not trial QC. In steps 40, 51, and 80, the agent emphasized fixed 60-bin trials and dataset size rather than filtering anomalous trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the concatenated `spks` arrays in each session's spike file, with `iarea` from retinotopy used to decide which neurons to keep and how to label their brain region.

ii. ```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
...
return dtrans['iarea']
```

iii. In steps 25, 28, and 40, the agent described the neural data as deconvolved Suite2p traces concatenated across planes, with retinotopy used for area assignment.

## 2-b. How is the `neural` data processed?

i. The script concatenates planes, filters neurons, keeps only running frames (`ft_move > 0`), interpolates the traces into 60 spatial bins per trial using cumulative position, casts each trial to `float32`, and replaces any NaN/Inf values with zero.

ii. ```python
VRmove = beh['ft_move'][:nfr] > 0
ft_AcumPos = beh['ft_PosCum'][:nfr]
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
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. In steps 25, 28, and 40, the agent justified this by citing the paper's statement that analysis used running periods only and by assuming constant VR speed made spatial interpolation equivalent to time alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopy: the code excludes `iarea == -1` and `iarea == 7`, then asserts that all remaining neurons map into V1, mHV, lHV, or aHV.

ii. ```python
valid_neuron_mask = (iarea != -1) & (iarea != 7)
...
area_idx = neu_area_ID(iarea)
neuron_region_idx = np.full(nneu, -1, dtype=np.int64)
for region_name, region_i in brain_region_map.items():
    neuron_region_idx[area_idx[region_name]] = region_i
...
assert np.all(neuron_region_idx_filtered >= 0)
```

iii. In steps 28 and 40, the agent said it would keep all neurons with valid visual-cortex assignments and exclude unassigned or outside-visual-cortex cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent claims alignment to trial start, but in code the per-trial neural data are aligned to fixed spatial bins from corridor position 0 through gray space, not to per-frame time from corridor entry. Every trial is forced to the same 60-bin position axis.

ii. ```python
n_bins = 60
time_bin_sec = 1.0 / 6.0
...
interp_spk = get_interpPos_spk(..., ntrials, n_bins=int(CL), lengths=CL)
...
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. In steps 25 and 40, the agent argued that corridor entry should be the alignment event but that variable frame counts should be normalized by spatial binning at constant VR speed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script sets the effective bin size to `1/6` s, or 166.67 ms, based on 1 decimeter bins at 60 cm/s. It does not keep the original 3.17 Hz imaging frames; instead it performs position-based interpolation/rebinning into 60 bins per trial.

ii. ```python
n_bins = 60
time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms
...
'metadata': {
    'time_bin_size': time_bin_sec * 1000,
}
```

iii. In steps 25 and 40, the agent first noticed the imaging frame rate, then replaced it with a spatial-bin-derived bin size because it believed constant VR speed justified that conversion.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundPos` and a synthetic per-trial position axis `positions = np.arange(n_bins)`, not from frame timestamps or `SoundFr`.

ii. ```python
SoundPos = beh['SoundPos']
...
positions = np.arange(n_bins)
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec
```

iii. In step 25, the agent reasoned that because the cue occurs at a random corridor position and the VR moves at fixed speed, cue timing should be represented from position rather than imaging-frame timestamps.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code subtracts the 0..59 spatial-bin index from the cue position and multiplies by `1/6` s per bin, producing a continuous 60-sample trace that is positive before the cue and negative after.

ii. ```python
time_to_cue = (sound_pos - positions) * time_bin_sec
input_trial = np.stack([
    time_to_cue.astype(np.float32),
    day_array,
    time_since_start.astype(np.float32),
    reward_avail
], axis=0)
```

iii. In steps 25 and 40, the agent justified this as the most natural way to express time-to-cue once trials had been converted to fixed spatial bins.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same 60 spatial bins as the interpolated neural data; each trial's cue trace has exactly one value per spatial bin.

ii. ```python
neural_trial = interp_spk[:, trial, :].astype(np.float32)
...
time_to_cue = (sound_pos - positions) * time_bin_sec
...
input_trial = np.stack([...], axis=0)
```

iii. In the trajectory, the agent repeatedly described all modalities as sharing one spatially normalized 60-bin trial representation.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp` in the experiment metadata, grouped by mouse name from `Imaging_Exp_info.npy`.

ii. ```python
key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')
mouse_dates[ndb['mname']].append(date)
session_dates[key] = (ndb['mname'], date)
```

iii. In steps 25 and 40, the agent treated training day as something recoverable from the session date metadata rather than from session order among converted recordings.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code computes each mouse's earliest recording date, subtracts it from each session date in calendar days, and broadcasts that scalar across all 60 bins of every trial in the session.

ii. ```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
...
session_days[key] = (date - mouse_first_date[mname]).days
...
day_array = np.full(n_bins, day, dtype=np.float32)
```

iii. The agent's justification in steps 25 and 40 was that elapsed days since first recording were a reasonable proxy for training progress.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the synthetic 0..59 position-bin axis rather than from `StartFr` and frame timestamps.

ii. ```python
positions = np.arange(n_bins)
time_since_start = positions * time_bin_sec
```

iii. In steps 25 and 40, the agent decided elapsed time should follow the fixed spatial representation of each trial.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script multiplies each spatial bin index by `1/6` s to obtain a 60-sample monotonic trace from 0 to about 9.83 s for every trial.

ii. ```python
time_since_start = positions * time_bin_sec
...
input_trial = np.stack([
    time_to_cue.astype(np.float32),
    day_array,
    time_since_start.astype(np.float32),
    reward_avail
], axis=0)
```

iii. The trajectory frames this as a direct consequence of spatial interpolation at fixed VR speed.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It shares the same fixed 60 spatial bins as the interpolated neural trials.

ii. ```python
neural_trial = interp_spk[:, trial, :].astype(np.float32)
time_since_start = positions * time_bin_sec
```

iii. In steps 25 and 40, the agent treated all trial-wise arrays as different channels on the same 60-bin axis.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `beh['isRew']`.

ii. ```python
isRew = beh['isRew']
...
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. The trajectory treated reward availability as a direct trial label that should be carried into the decoder input.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond broadcasting the per-trial reward flag across all 60 bins of the trial.

ii. ```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. In step 25, the agent grouped reward availability with other decoder inputs and treated it as a per-trial contextual signal.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh['WallName']`.

ii. ```python
WallName = beh['WallName']
...
stim_idx = stim_to_idx[str(WallName[trial])]
```

iii. The trajectory treated corridor wall identity as the source of the visual stimulus label.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code collects every unique `WallName` string across the dataset, builds a 15-class lookup table, converts each trial's `WallName` to that class index, and broadcasts it across the 60 bins of the trial. It does not collapse variants into four base texture families.

ii. ```python
all_stim_names_set = set()
...
for name in np.unique(beh_data[beh_key_inner]['WallName']):
    all_stim_names_set.add(str(name))
all_stim_names = sorted(all_stim_names_set)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
...
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```

iii. In steps 83 and 148, the agent explicitly validated the dataset by noting there were 15 stimulus categories and treated that as evidence the conversion was correct.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickPos` and `LickTrind`, not from lick frame indices.

ii. ```python
lick_spatial = make_lick_spatial_bins(
    beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL)
)
```

iii. In steps 83 and 84, the agent focused on whether unsupervised versus task mice had licks and reasoned about licking in spatial-bin terms.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code initializes a `(n_trials, 60)` zero array, then for each lick uses `floor(LickPos)` clipped into `[0, 59]` to mark that spatial bin as 1 in the corresponding `LickTrind` trial.

ii. ```python
lick_arr = np.zeros((ntrials, n_bins), dtype=np.int64)
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
    if 0 <= tr < ntrials:
        lick_arr[tr, bin_idx] = 1
```

iii. The agent justified this in the trajectory by treating lick location as the natural representation once trials were converted to position bins.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by sharing the same 60 spatial bins as the interpolated neural data.

ii. ```python
lick_trial = lick_spatial[trial, :].astype(np.int64)
output_trial = np.stack([
    stim_category,
    lick_trial,
    pos_trial,
    speed_trial
], axis=0)
```

iii. In steps 25 and 40, the agent's main alignment principle was a common spatially normalized trial axis.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The output position is derived from a synthetic fixed bin index `0..59` and corridor-length metadata, not from per-frame `ft_Pos` values. The neural interpolation itself uses `ft_PosCum`, but the output label is just a deterministic function of the bin number.

ii. ```python
positions = np.arange(n_bins)
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. The trajectory assumed every trial should live on a fixed 60-bin corridor-plus-gray axis, so position labels became intrinsic to the binning scheme rather than measured per frame.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code uses `np.digitize` on the fixed bin indices with edges `[10, 20, 30, 40]`, yielding deterministic position categories for all trials before any trial loop runs.

ii. ```python
position_bins_edges = [10, 20, 30, 40]
...
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. In steps 25 and 40, the agent reasoned from the 4 m corridor plus 2 m gray-space geometry rather than from per-frame behavioral position measurements.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into five categories: 0-9 dm, 10-19 dm, 20-29 dm, 30-39 dm, and 40-59 dm gray space. This adds a fifth `gray_space` class beyond the four corridor bins requested.

ii. ```python
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
...
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. In steps 25 and 40, the agent explicitly chose 60 bins spanning both corridor and gray space, which is why it created a fifth position category.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by assigning one position category to each of the same 60 spatial bins used for neural interpolation.

ii. ```python
pos_trial = pos_category.copy()
...
neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The trajectory repeatedly described a single shared spatial-bin axis for neural and behavioral outputs.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['run_pos']`, the running-speed values already arranged over position bins, not from framewise `ft_RunSpeed`.

ii. ```python
run_pos = beh['run_pos']    # (ntrials, 60) running speed at each position
...
speed_bins = speed_to_bins(run_pos, speed_edges)
```

iii. In steps 64 and 83, the agent explicitly reasoned about `run_pos` as the natural source for speed once everything was represented in spatial bins.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script flattens `run_pos` from all sessions, removes NaNs, computes global 25th/50th/75th percentile edges, and then discretizes each trial's 60-bin speed trace with `np.digitize`.

ii. ```python
all_speeds = np.concatenate([rp.ravel() for rp in run_pos_all])
all_speeds = all_speeds[~np.isnan(all_speeds)]
edges = np.percentile(all_speeds, [25, 50, 75])
...
result = np.digitize(run_pos, edges)
```

iii. In steps 64, 76, and 148, the agent justified this as producing globally balanced quartile bins and treated balanced quartiles as a validation target.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global percentile edges stored in `speed_edges`, and the categories are the `np.digitize` bins induced by those three thresholds.

ii. ```python
speed_edges = discretize_speed_quartiles(run_pos_all)
...
speed_trial = speed_bins[trial, :].astype(np.int64)
...
speed_labels = [
    f'Q1(<{speed_edges[0]:.1f})',
    f'Q2({speed_edges[0]:.1f}-{speed_edges[1]:.1f})',
    f'Q3({speed_edges[1]:.1f}-{speed_edges[2]:.1f})',
    f'Q4(>{speed_edges[2]:.1f})'
]
```

iii. The trajectory shows the agent intentionally switched to dataset-wide quartile thresholds and even corrected the code so the sample run would still use full-dataset edges.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by giving each of the 60 spatial bins a speed category on the same axis used by the neural data.

ii. ```python
speed_trial = speed_bins[trial, :].astype(np.int64)
output_trial = np.stack([
    stim_category,
    lick_trial,
    pos_trial,
    speed_trial
], axis=0)
```

iii. In the trajectory, the agent's alignment story was consistently that neural, inputs, and outputs all share the same position-normalized trial grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script skips sessions if neural or retinotopy files fail to load, skips sessions with fewer than 2 trials or fewer than 10 valid neurons, removes NaNs before computing global speed quartiles, and replaces NaN/Inf values in interpolated neural trials with zero. It does not implement the reference's framewise truncation of behavior to imaged frames.

ii. ```python
try:
    spk = load_spk(...)
except Exception as e:
    ...
try:
    iarea = load_retino(...)
except Exception as e:
    ...
if ntrials < 2:
    ...
if n_valid < 10:
    ...
```
```python
all_speeds = all_speeds[~np.isnan(all_speeds)]
...
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The trajectory mostly frames these as pragmatic safeguards for a huge conversion job; it does not mention the reference behavior/imaging-length mismatch handling, and instead emphasizes keeping the run from failing.

## 12-a. What are the most time-consuming steps of the code?

i. The heaviest work is loading very large spike files and then interpolating every neuron's running-frame trace into 60 spatial bins per trial. The first-pass global speed scan and the final write of the massive pickle also add cost, but the per-neuron interpolation dominates the extra compute the agent introduced.

ii. ```python
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

iii. In steps 64, 80, and 102, the agent repeatedly described the full conversion as expensive because of spatial interpolation over 89 sessions with 50K+ neurons per session and noted the job's very high memory use.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: the per-neuron interpolation loop in `spk_pos_interp`, the chunked `while` loop in `get_interpPos_spk`, the per-lick loop in `make_lick_spatial_bins`, and the per-trial loop that stacks inputs/outputs one trial at a time.

ii. ```python
for s in range(raw_spk.shape[0]):
    spk_resh.append(...)
```
```python
while i <= spk.shape[0]:
    interp_spk[i:i+step_size, :] = spk_pos_interp(...)
    i += step_size
```
```python
for i in range(len(lick_pos)):
    ...
for trial in range(ntrials):
    ...
```

iii. The trajectory does not propose these optimizations, but its repeated complaints about runtime and memory make clear these loops are the bottlenecks created by the chosen processing path.

## 12-c. What processing does the code repeat multiple times?

i. The script reads and iterates through behavior data multiple times: once to build `session_map`, once in the first pass to collect `run_pos` for global quartiles, once again to collect all unique stimulus names, and again in the main conversion loop via the cached behavior dicts. It also repeatedly constructs identical position/time templates for every trial.

ii. ```python
session_map = get_session_beh_mapping(exp_info, ...)
...
for sess_key in all_sessions_full:
    ...
    run_pos_all.append(beh['run_pos'])
...
for exp_type_key in exp_info.keys():
    beh_data = np.load(beh_path, allow_pickle=True).item()
    ...
```
```python
positions = np.arange(n_bins)
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
for trial in range(ntrials):
    time_to_cue = (sound_pos - positions) * time_bin_sec
    time_since_start = positions * time_bin_sec
```

iii. The trajectory explicitly added extra whole-dataset passes for speed quartiles and stimulus names in steps 64, 69, and 76, prioritizing validation over efficiency.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Relative to the reference decoder target, the code does substantial extra work that is not needed downstream: it interpolates gray-space bins and creates a fifth `gray_space` position class, computes and stores a 15-way wall-name label instead of the four requested base texture categories, sanitizes NaN/Inf neural values after interpolation, and records large metadata fields such as skipped sessions and textual mappings that the decoder never uses.

ii. ```python
n_bins = 60  # spatial bins per trial (40 corridor + 20 gray)
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
```
```python
all_stim_names_set = set()
...
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```
```python
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The trajectory shows the agent deliberately broadened the representation to corridor-plus-gray, 15 stimulus categories, and large diagnostic metadata because it believed more complete structure would help validation, even though those additions are outside the reference processing.
