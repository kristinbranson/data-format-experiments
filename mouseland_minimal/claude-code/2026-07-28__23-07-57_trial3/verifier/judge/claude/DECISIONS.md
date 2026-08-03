# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` for behavior, `spk/` for deconvolved traces, and `retinotopy/` for brain region assignment. `Imaging_Exp_info.npy` is loaded first as a master index. Behavior files are loaded per experiment type (`Beh_<exp_type>.npy`), neural data per session (`<mname>_<datexp>_<blk>_neural_data.npy`), and retinotopy per mouse-date (`<mname>_<datexp>_trans.npz`). A two-pass approach is used: first pass collects running speed data for global quartiles, second pass builds the dataset.

ii.
```python
exp_info = np.load(os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))
spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
```

iii. The AI follows the reference code structure. It uses `Imaging_Exp_info.npy` as the master index and reads behavior, neural, and retinotopy data as described by the reference code.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from the session key (first field of `mname_datexp_blk`). All unique mouse names are sorted and assigned indices.

ii.
```python
all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subject_idx_list.append(mouse_to_idx[mname])
```

iii. The mouse name is directly available in the session key and experiment metadata.

## 1-c. How are the data split into sessions?

i. A session is a unique recording identified by `mname_datexp_blk`. When a recording appears in multiple experiment types, the AI selects the one with the most stimuli (most complete `stim_id` array). This yields 89 unique sessions.

ii.
```python
def get_session_beh_mapping(exp_info, data_root='data/beh'):
    session_map = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
            if key not in session_map:
                session_map[key] = (exp_type, beh_key, ndb)
            else:
                old_nstim = np.sum(~np.isnan(old_ndb['stim_id'].astype(float)))
                new_nstim = np.sum(~np.isnan(ndb['stim_id'].astype(float)))
                if new_nstim > old_nstim:
                    session_map[key] = (exp_type, beh_key, ndb)
```

iii. The AI prefers the experiment type with the most stimuli when a recording appears multiple times. The underlying behavioral data is identical across experiment types for the same recording.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the behavior data. The number of trials is given by `beh['ntrials']`. Each trial is processed as a whole, with neural data spatially interpolated into 60 position bins (40 corridor + 20 gray space). No per-trial frame selection is done; instead, the spatial interpolation handles the trial structure via cumulative position.

ii.
```python
ntrials = beh['ntrials']
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :]
```

iii. The AI uses the spatial interpolation approach from the reference code, where trial boundaries are defined by cumulative position resets. Each trial gets exactly 60 spatial bins.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped. Sessions with fewer than 10 valid neurons are skipped. No trial-level quality filtering is applied. NaN/Inf values in neural data are replaced with zeros.

ii.
```python
if ntrials < 2:
    skipped_sessions.append(sess_key)
    continue
if n_valid < 10:
    skipped_sessions.append(sess_key)
    continue
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The AI applies minimal filtering, keeping all trials from valid sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`spk/<session_id>_neural_data.npy`), which contains deconvolved fluorescence traces as a list of arrays per imaging plane, and `iarea` from retinotopy files for brain region assignment.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
iarea = np.load(os.path.join(root, fn), allow_pickle=True)['iarea']
```

iii. The neural data is read directly from the Suite2p deconvolved traces, consistent with the paper.

## 2-b. How is the `neural` data processed?

i. The neural data is spatially interpolated into 60 position bins per trial using only running frames (`ft_move > 0`). The interpolation maps neural activity from frame-space into position-space using cumulative position. The reference code's `get_interpPos_spk()` function is adapted. Data is stored as float32.

ii.
```python
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

iii. The AI follows the reference code's spatial interpolation approach, which maps neural activity into position bins. The paper states analysis was done on position-interpolated activity during running.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` (unassigned) or `iarea == 7` (outside visual cortex) are excluded. The remaining neurons are assigned to V1, mHV, lHV, or aHV. No further quality filtering is applied.

ii.
```python
valid_neuron_mask = (iarea != -1) & (iarea != 7)
spk_filtered = spk[valid_neuron_mask]
```

iii. The AI follows the reference code's brain region assignment. The paper used Suite2p's cell classifier for initial curation, and the AI does not apply additional filters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to corridor entry (trial start) via spatial interpolation. Position 0 corresponds to corridor entry, and each of the 60 bins represents 1 decimeter. The alignment is inherent in the position-based binning.

ii.
```python
linPos = np.arange(0, new_shape[0], 1 / new_shape[1])
# Interpolation maps frame-level activity to position bins starting at corridor entry
```

iii. The spatial interpolation automatically aligns to corridor entry since position 0 is where the corridor begins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses spatial bins rather than temporal bins. Each bin represents 1 decimeter of corridor. At the constant VR speed of 60 cm/s (6 dm/s), each bin corresponds to approximately 166.67 ms. There are 60 bins per trial. No temporal rebinning is applied; instead, the data is spatially interpolated from frame-level to position-level.

ii.
```python
n_bins = 60
time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms
'time_bin_size': time_bin_sec * 1000,  # 166.67 ms
```

iii. The AI treats each spatial bin as equivalent to a temporal bin at constant VR speed. The metadata reports 166.67 ms as the time bin size.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos`, the position (in dm) where the sound cue occurs in each trial, and the spatial bin positions (0 to 59).

ii.
```python
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec
```

iii. `SoundPos` provides the cue position directly. The time is derived from position differences at constant VR speed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundPos - position) * time_bin_sec` where `time_bin_sec = 1/6` seconds. This gives the time in seconds from the current position to the sound cue position, positive before the cue and negative after.

ii.
```python
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. The computation converts spatial distance to time using the constant VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is computed at the same 60 spatial bin positions as the neural data, so it is inherently aligned.

ii.
```python
positions = np.arange(n_bins)  # 0 to 59
time_to_cue = (sound_pos - positions) * time_bin_sec
```

iii. Both neural and input data use the same 60-bin spatial grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field in the experiment metadata, which contains the recording date as a string in `YYYY_MM_DD` format.

ii.
```python
def compute_day_of_training(exp_info):
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')
            mouse_dates[ndb['mname']].append(date)
            session_dates[key] = (ndb['mname'], date)
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
    for key, (mname, date) in session_dates.items():
        session_days[key] = (date - mouse_first_date[mname]).days
```

iii. The dates from the experiment metadata are parsed and used to compute calendar day offsets.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is found. The day of training for each session is the number of calendar days since that first date. This means non-consecutive recording days have gaps (e.g., day 0, 2, 5, ...). The value is constant within a session, broadcast across all 60 bins.

ii.
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
session_days[key] = (date - mouse_first_date[mname]).days
day_array = np.full(n_bins, day, dtype=np.float32)
```

iii. The AI interprets "day of training" as calendar days since the first session, which captures the actual passage of time.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the spatial bin positions (0 to 59) and the constant VR speed (6 dm/s).

ii.
```python
positions = np.arange(n_bins)
time_since_start = positions * time_bin_sec
```

iii. Since the data is spatially interpolated, the time since trial start is derived from position divided by VR speed.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `position * time_bin_sec` where `time_bin_sec = 1/6` seconds. This gives the time in seconds from corridor entry, linearly increasing from 0 to 9.83 s across 60 bins.

ii.
```python
time_since_start = positions * time_bin_sec  # shape (60,)
```

iii. The time is deterministically derived from position at constant VR speed.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The time since trial start is computed at the same 60 spatial bin positions as the neural data, so it is inherently aligned.

ii.
```python
positions = np.arange(n_bins)
time_since_start = positions * time_bin_sec
```

iii. Both neural and input data use the same 60-bin spatial grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavior data, which is a binary flag per trial indicating whether the corridor is rewarded.

ii.
```python
isRew = beh['isRew']
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. `isRew` is directly available in the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The binary `isRew` value is broadcast across all 60 bins of each trial. No additional processing is applied.

ii.
```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. No processing needed; the value is directly used from the behavior data.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` in the behavior data, which names the texture on the corridor walls for each trial.

ii.
```python
WallName = beh['WallName']
stim_idx = stim_to_idx[str(WallName[trial])]
```

iii. `WallName` directly provides the stimulus identity for each trial.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each unique `WallName` across all sessions becomes its own category (15 categories total: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5). The stimulus index is constant per trial, broadcast across all 60 bins.

ii.
```python
all_stim_names = sorted(all_stim_names_set)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
stim_idx = stim_to_idx[str(WallName[trial])]
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```

iii. The AI uses all 15 individual texture names as separate categories, rather than grouping them into 4 broad texture types.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickPos` (position of each lick in the corridor) and `LickTrind` (trial index of each lick).

ii.
```python
lick_spatial = make_lick_spatial_bins(
    beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL)
)
```

iii. The AI uses the spatial position of licks rather than the frame-based `LickFr`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array of shape (ntrials, 60) is created. For each lick event, the `LickPos` is mapped to a spatial bin (0-59), and that bin is set to 1 for the corresponding trial.

ii.
```python
def make_lick_spatial_bins(lick_pos, lick_trind, ntrials, n_bins=60):
    lick_arr = np.zeros((ntrials, n_bins), dtype=np.int64)
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
        if 0 <= tr < ntrials:
            lick_arr[tr, bin_idx] = 1
    return lick_arr
```

iii. The spatial approach is consistent with the position-based binning of the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is mapped to the same 60 spatial bins as the neural data using `LickPos`, so it is inherently aligned.

ii.
```python
lick_spatial = make_lick_spatial_bins(beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL))
lick_trial = lick_spatial[trial, :]
```

iii. Both licking and neural data are in position-space bins.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived deterministically from the spatial bin index (0 to 59), where each bin represents 1 dm.

ii.
```python
positions = np.arange(n_bins)
pos_category = np.digitize(positions, position_bins_edges)
```

iii. Since the data is spatially binned, position is known from the bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 60 spatial bins are grouped into 5 categories using `np.digitize` with edges at [10, 20, 30, 40]: bins 0-9 -> category 0 (0-1m), 10-19 -> 1 (1-2m), 20-29 -> 2 (2-3m), 30-39 -> 3 (3-4m), 40-59 -> 4 (gray space).

ii.
```python
position_bins_edges = [10, 20, 30, 40]
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
```

iii. The AI includes a 5th category for gray space (2m beyond the textured corridor).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 5 categories: 4 corridor bins of 1m (10 dm) each plus a gray space bin. Thresholds are at 10, 20, 30, and 40 dm.

ii.
```python
position_bins_edges = [10, 20, 30, 40]
n_position_categories = 5  # 4 corridor bins + gray space
pos_category = np.digitize(positions, position_bins_edges)
```

iii. The 5th bin captures gray space activity, which the reference solution excludes by only taking corridor-space frames.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is deterministic from the spatial bin index, so it is inherently aligned with the neural data.

ii.
```python
pos_trial = pos_category.copy()  # same 60 bins as neural
```

iii. Alignment is automatic since both use the same spatial grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `run_pos` in the behavior data, which contains running speed at each spatial position for each trial, shape (ntrials, 60).

ii.
```python
run_pos = beh['run_pos']
speed_bins = speed_to_bins(run_pos, speed_edges)
```

iii. The AI uses `run_pos` (position-interpolated running speed) rather than `ft_RunSpeed` (frame-level running speed).

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 quartile bins. Quartile edges are computed globally across all 89 sessions from `run_pos` using percentiles at [25, 50, 75]. Each session's speeds are then discretized using `np.digitize` with these global edges.

ii.
```python
def discretize_speed_quartiles(run_pos_all):
    all_speeds = np.concatenate([rp.ravel() for rp in run_pos_all])
    all_speeds = all_speeds[~np.isnan(all_speeds)]
    edges = np.percentile(all_speeds, [25, 50, 75])
    return edges

def speed_to_bins(run_pos, edges):
    result = np.digitize(run_pos, edges)
    return result.astype(np.int64)
```

iii. Global quartiles ensure consistent bin definitions across sessions, though the reference uses per-session rank-based quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is thresholded using global percentile edges at [25, 50, 75] computed across all sessions. `np.digitize` maps each speed value to bins 0-3.

ii.
```python
edges = np.percentile(all_speeds, [25, 50, 75])
result = np.digitize(run_pos, edges)
```

iii. The thresholds are value-based (percentile edges) rather than rank-based.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` is already in position-space (60 bins per trial), matching the neural data's spatial binning. No additional alignment is needed.

ii.
```python
speed_trial = speed_bins[trial, :].astype(np.int64)
```

iii. Both use the same 60-bin spatial grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN/Inf values in neural data are replaced with zeros. Sessions with too few valid neurons (<10) or trials (<2) are skipped. Lick positions are clipped to valid bin ranges. No other explicit error handling is described.

ii.
```python
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
lick_frame = beh['LickFr'].astype(int)
bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
```

iii. The AI applies defensive handling for numerical issues that may arise from spatial interpolation.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files (totaling ~400+ GB) is the most time-consuming step. The spatial interpolation step is also computationally expensive as it involves interpolation for each neuron.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
interp_spk = get_interpPos_spk(spk_filtered[:, VRmove], ft_AcumPos[VRmove], ntrials, ...)
```

iii. I/O cost dominates, but the spatial interpolation adds significant computation compared to simply indexing frames.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `make_lick_spatial_bins` function loops over individual lick events rather than using vectorized operations. The `spk_pos_interp` function loops over neurons individually for interpolation.

ii.
```python
# Lick loop
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
    if 0 <= tr < ntrials:
        lick_arr[tr, bin_idx] = 1

# Neuron interpolation loop
for s in range(raw_spk.shape[0]):
    spk_resh.append(np.reshape(interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos), ...))
```

iii. Both loops process individual elements/neurons and could potentially be vectorized.

## 12-c. What processing does the code repeat multiple times?

i. The behavior data is loaded twice: once in the first pass for running speed quartile computation, and again accessed in the second pass for the full conversion. However, the data is cached so it's not re-read from disk.

ii.
```python
# First pass
for sess_key in all_sessions_full:
    beh = beh_cache[exp_type][beh_key]
    run_pos_all.append(beh['run_pos'])

# Second pass
for sess_idx, sess_key in enumerate(all_sessions):
    beh = beh_cache[exp_type][beh_key]
```

iii. The two-pass approach requires accessing behavior data twice, though caching mitigates the I/O cost.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes gray space data (bins 40-59) in every trial, which are 20 out of 60 bins. The instructions specify position bins for the 4m corridor only, yet the AI includes the 2m gray space. Additionally, the AI collects all unique stimulus names across all sessions in a separate loop, which scans all behavior files redundantly.

ii.
```python
n_bins = 60  # spatial bins per trial (40 corridor + 20 gray)
# Gray space bins (40-59) are included but may not be informative for corridor-based decoding
```

iii. The gray space bins add data that may dilute corridor-specific signals.
