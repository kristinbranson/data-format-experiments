# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, which maps experiment types to lists of session metadata dicts. It builds a session-to-behavior mapping, then iterates over all 89 unique recording sessions (identified by `mname_datexp_blk` keys). For each session, it loads: (1) neural data from `data/spk/{mname}_{datexp}_{blk}_neural_data.npy`, (2) behavioral data from `data/beh/Beh_{exp_type}.npy`, and (3) retinotopy from `data/retinotopy/{mname}_{datexp}_trans.npz`. When a session appears in multiple experiment types, it selects the entry with the most non-NaN `stim_id` values.

ii.
```python
exp_info = np.load(os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))
all_sessions = sorted(session_map.keys())

# For each session:
spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
beh = beh_cache[exp_type][beh_key]
iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
```

iii. The AI based this on reading the reference code's `data_process_script.ipynb` and `utils.py`, which showed how data was loaded from `Exp_info`, neural spike files, and behavior files. The session-to-behavior mapping with preference for more stimuli handles the case where a recording appears across multiple experiment types.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the mouse name prefix of each session key (e.g., "DR10" from "DR10_2022_07_12_1"). All unique mouse names are collected, sorted, and assigned indices. Each session is mapped to its subject via `subject_idx`.

ii.
```python
all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
# ...
subject_idx_list.append(mouse_to_idx[mname])
```

iii. The AI correctly identified 19 unique mice from the session keys, matching the paper's stated "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Each unique recording (identified by `mname_datexp_blk`) is one session. The AI processes 89 sessions total, sorting them alphabetically for reproducibility. Sessions that fail to load neural data, retinotopy, or have fewer than 2 trials or fewer than 10 valid neurons are skipped.

ii.
```python
all_sessions = sorted(session_map.keys())
# ...
if ntrials < 2:
    skipped_sessions.append(sess_key)
    continue
if n_valid < 10:
    skipped_sessions.append(sess_key)
    continue
```

iii. The 89 session count matches the paper. No sessions were actually skipped in the final run.

## 1-d. How are the data split into trials?

i. Trial count per session comes from `beh['ntrials']`. Neural data is spatially interpolated into `(n_neurons, ntrials, 60)` arrays, then split by trial index. Each trial has exactly 60 spatial bins.

ii.
```python
ntrials = beh['ntrials']
interp_spk = get_interpPos_spk(spk_filtered[:, VRmove], ft_AcumPos[VRmove], ntrials, n_bins=int(CL), lengths=CL)
# interp_spk shape: (n_valid_neurons, ntrials, n_bins)
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The number of trials per session is taken from the behavior data. All trials within a session are included (no trial-level filtering).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All trials from each session are included as long as the session itself passes basic checks (≥2 trials, ≥10 valid neurons, data files exist). NaN/Inf values in neural data are replaced with 0.

ii.
```python
if ntrials < 2:
    skipped_sessions.append(sess_key)
    continue
if n_valid < 10:
    skipped_sessions.append(sess_key)
    continue
# ...
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The reference code does not appear to filter individual trials either, beyond excluding non-running frames from the interpolation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `'spks'` key in each neural data `.npy` file, which contains deconvolved fluorescence traces from Suite2p across multiple imaging planes.

ii.
```python
def load_spk(mname, datexp, blk, root='data/spk'):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. The AI correctly identified `spks` as the key containing neural data, matching the reference code in `utils.py`.

## 2-b. How is the `neural` data processed?

i. Neural data is processed by: (1) concatenating imaging planes, (2) filtering out non-visual-cortex neurons (iarea == -1 or 7), (3) selecting only running frames (`ft_move > 0`), and (4) spatially interpolating into 60 position bins per trial using `spk_pos_interp` from the reference code.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
valid_neuron_mask = (iarea != -1) & (iarea != 7)
spk_filtered = spk[valid_neuron_mask]
VRmove = beh['ft_move'][:nfr] > 0
ft_AcumPos = beh['ft_PosCum'][:nfr]
interp_spk = get_interpPos_spk(spk_filtered[:, VRmove], ft_AcumPos[VRmove], ntrials, n_bins=int(CL), lengths=CL)
```

iii. The AI copied the spatial interpolation functions (`interp_value`, `spk_pos_interp`, `get_interpPos_spk`) directly from the reference `utils.py`. The paper states "We only considered timepoints during running for analysis."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` (unassigned) or `iarea == 7` (outside visual cortex) are excluded. No other neuron-level quality filtering (e.g., minimum firing rate) is applied. NaN/Inf values are replaced with 0.

ii.
```python
valid_neuron_mask = (iarea != -1) & (iarea != 7)
spk_filtered = spk[valid_neuron_mask]
# ...
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The filtering matches the reference code's `neu_area_ID` function which only maps iarea values 0-6, 8-9 to brain regions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) via spatial interpolation. Each trial's neural data is interpolated from raw frame-level data to 60 position bins starting at position 0 (corridor entry). Since VR speed is constant at 60 cm/s during running, spatial bins map directly to time bins.

ii.
```python
linPos = np.arange(0, new_shape[0], 1 / new_shape[1])
# linPos goes from 0 to ntrials in steps of 1/60, representing position within each trial
spk_resh.append(np.reshape(
    interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos),
    (int(new_shape[0]), int(new_shape[1]))
))
```

iii. Alignment is to corridor entry (position 0) at the start of each trial. The AI documented this as "Trial start (corridor entry)" in the metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 166.67 ms, derived from the spatial bin size (1 dm) divided by the constant VR speed (6 dm/s). No explicit temporal rebinning is applied; instead, spatial interpolation into 60 equal-width position bins acts as the binning mechanism.

ii.
```python
time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms
# ...
'time_bin_size': time_bin_sec * 1000,  # 166.67 ms
```

iii. The AI correctly computed 166.67 ms from the VR speed and bin size. Since the VR moves at constant speed when the mouse is running, spatial bins correspond to uniform time intervals.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundPos']` (the position in dm where the sound cue occurs in each trial) and the spatial bin index (0-59).

ii.
```python
SoundPos = beh['SoundPos']
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. The AI found `SoundPos` in the reference code's `data_process_script.ipynb`, where it's described as the position of the sound cue randomly drawn from uniform [5, 35] dm.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial and each spatial bin position `p` (0-59), `time_to_cue = (SoundPos[trial] - p) * (1/6)` seconds. This gives positive values before the cue position and negative values after.

ii.
```python
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. This represents the time until the sound cue is reached, computed from the spatial distance divided by VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to sound cue is computed at the same 60 spatial bin positions as the neural data, so alignment is inherent. Both are indexed by spatial position within the trial.

ii.
```python
positions = np.arange(n_bins)  # 0 to 59
time_to_cue = (sound_pos - positions) * time_bin_sec
```

iii. Since both neural data and time-to-cue are computed on the same 60-bin spatial grid, they are inherently aligned.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field in experiment metadata (which encodes the recording date as `YYYY_MM_DD`).

ii.
```python
def compute_day_of_training(exp_info):
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
            date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')
            mouse_dates[ndb['mname']].append(date)
            session_dates[key] = (ndb['mname'], date)
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
    for key, (mname, date) in session_dates.items():
        session_days[key] = (date - mouse_first_date[mname]).days
    return session_days
```

iii. The AI computes the number of days since each mouse's first recording session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date across all experiment types is found. Day of training is the number of days between the session date and the mouse's first date. This value is constant within a session (broadcast across all 60 spatial bins).

ii.
```python
day = session_days.get(sess_key, 0)
day_array = np.full(n_bins, day, dtype=np.float32)
```

iii. The AI reasoned that day of training should be relative to each mouse's first session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the spatial bin index (0-59) and the time per bin (1/6 seconds).

ii.
```python
positions = np.arange(n_bins)
time_since_start = positions * time_bin_sec  # shape (60,)
```

iii. Since each spatial bin corresponds to a fixed time interval at constant VR speed, time since trial start is simply the bin index times the time per bin.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `time_since_start[p] = p * (1/6)` seconds, for p = 0 to 59. Ranges from 0 to 9.83 seconds.

ii.
```python
time_since_start = positions * time_bin_sec  # shape (60,)
```

iii. This is a straightforward linear mapping from spatial position to time.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Inherently aligned: both are on the same 60-bin spatial grid, where bin 0 = corridor entry = trial start.

ii.
```python
positions = np.arange(n_bins)  # same grid as neural data
time_since_start = positions * time_bin_sec
```

iii. Same spatial grid ensures alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a per-trial binary variable indicating whether the trial is in a rewarded corridor.

ii.
```python
isRew = beh['isRew']
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. The AI identified `isRew` from the behavior data structure documented in `data_process_script.ipynb`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value is broadcast to all 60 spatial bins as a constant per-trial value. Values are 0 (unrewarded) or 1 (rewarded). Unsupervised mice have all zeros.

ii.
```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. No further processing; the raw `isRew` value is used directly.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which contains the name of the wall texture for each trial (e.g., "circle1", "leaf2").

ii.
```python
WallName = beh['WallName']
stim_idx = stim_to_idx[str(WallName[trial])]
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```

iii. Stimulus names were collected from all behavior files to build a global vocabulary of 15 categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` values across all sessions and experiment types are collected and sorted alphabetically, producing 15 categories. Each trial's stimulus is mapped to its index in this sorted list and broadcast as a constant across all 60 spatial bins.

ii.
```python
all_stim_names_set = set()
for exp_type_key in exp_info.keys():
    beh_data = np.load(beh_path, allow_pickle=True).item()
    for beh_key_inner in beh_data:
        for name in np.unique(beh_data[beh_key_inner]['WallName']):
            all_stim_names_set.add(str(name))
all_stim_names = sorted(all_stim_names_set)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```

iii. The 15 categories match the full stimulus set used across all experiment types in the paper.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickPos']` (position of each lick event in dm) and `beh['LickTrind']` (trial index for each lick event).

ii.
```python
lick_spatial = make_lick_spatial_bins(beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL))
```

iii. The AI found these variables in the reference code's behavior data documentation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each lick event, the position is floored to get a spatial bin index (clipped to 0-59), and the corresponding entry in a binary `(ntrials, 60)` array is set to 1. Multiple licks in the same bin are collapsed to a single 1.

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

iii. The AI verified that unsupervised mice have zero licks (correct behavior).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is binned into the same 60 spatial position bins as the neural data, so they are inherently aligned.

ii.
```python
lick_spatial = make_lick_spatial_bins(beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL))
lick_trial = lick_spatial[trial, :].astype(np.int64)  # shape (60,)
```

iii. Both use the same spatial binning scheme.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived deterministically from the spatial bin index (0-59), not from raw behavioral position data. The position category for each bin is fixed and identical across all trials.

ii.
```python
positions = np.arange(n_bins)
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. Since neural data is spatially interpolated to uniform bins, position is inherently known from the bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 60 spatial bins are classified into position categories using `np.digitize` with edges `[10, 20, 30, 40]`, producing 5 categories: bins 0-9 (0-1m), 10-19 (1-2m), 20-29 (2-3m), 30-39 (3-4m), 40-59 (gray space).

ii.
```python
position_bins_edges = [10, 20, 30, 40]
n_position_categories = 5  # 4 corridor bins + gray space
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. The AI added a 5th "gray_space" category for bins 40-59, beyond the 4 corridor bins specified in the instructions.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Uses `np.digitize` with edges `[10, 20, 30, 40]`:
   - Bins 0-9 (0-1m) -> category 0
   - Bins 10-19 (1-2m) -> category 1
   - Bins 20-29 (2-3m) -> category 2
   - Bins 30-39 (3-4m) -> category 3
   - Bins 40-59 (gray space) -> category 4

ii.
```python
position_bins_edges = [10, 20, 30, 40]
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. This creates 5 categories instead of the 4 specified in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is deterministic given the spatial bin index, which is the same grid used for neural data. No alignment step needed.

ii.
```python
pos_trial = pos_category.copy()  # shape (60,)
```

iii. Inherent alignment through shared spatial grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['run_pos']`, which contains running speed values at each spatial position for each trial. Shape is `(ntrials, 60)`.

ii.
```python
run_pos = beh['run_pos']
speed_bins = speed_to_bins(run_pos, speed_edges)
```

iii. The AI identified `run_pos` from the behavior data as position-resolved running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A two-pass approach: first, all `run_pos` values from all 89 sessions are collected and flattened. Global quartile edges (25th, 50th, 75th percentiles) are computed. Then each speed value is digitized into 4 bins using these edges.

ii.
```python
def discretize_speed_quartiles(run_pos_all):
    all_speeds = np.concatenate([rp.ravel() for rp in run_pos_all])
    all_speeds = all_speeds[~np.isnan(all_speeds)]
    edges = np.percentile(all_speeds, [25, 50, 75])
    return edges

def speed_to_bins(run_pos, edges):
    result = np.digitize(run_pos, edges)  # 0, 1, 2, 3
    return result.astype(np.int64)
```

iii. Global quartiles ensure each bin contains ~25% of all data. Edges: [16.96, 29.25, 44.25] cm/s.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with 3 edges produces 4 bins (0-3), where each bin corresponds to a quartile of the global running speed distribution.

ii.
```python
speed_edges = discretize_speed_quartiles(run_pos_all)
# edges: [16.96, 29.25, 44.25]
result = np.digitize(run_pos, edges)  # 0, 1, 2, 3
```

iii. Matches the instruction "4 bins, each corresponding to 25% of the data."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` from the behavior data is already on the same 60-bin spatial grid as the interpolated neural data. No additional alignment needed.

ii.
```python
speed_trial = speed_bins[trial, :].astype(np.int64)  # shape (60,)
```

iii. Both use the same 60-bin spatial grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN/Inf values in neural data are replaced with 0. NaN running speeds are removed before computing quartile edges. Lick events with invalid trial indices are skipped. Sessions missing data files are skipped entirely.

ii.
```python
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
# ...
all_speeds = all_speeds[~np.isnan(all_speeds)]
# ...
if 0 <= tr < ntrials:
    lick_arr[tr, bin_idx] = 1
```

iii. The AI handles edge cases defensively but does not discard trials with partial issues.

## 12-a. What are the most time-consuming steps of the code?

i. The spatial interpolation of neural data (`get_interpPos_spk`) is by far the most time-consuming step, as it must interpolate tens of thousands of neurons across hundreds of trials per session. The full conversion with 89 sessions took several hours. Loading large neural data files (50K-90K neurons per session) is also slow.

ii.
```python
def get_interpPos_spk(spk, spk_culm_pos, ntrial, n_bins=60, lengths=60):
    interp_spk = np.zeros((spk.shape[0], ntrial, n_bins))
    step_size = 10000
    i = 0
    while i <= spk.shape[0]:
        interp_spk[i:i+step_size, :] = spk_pos_interp(...)
        i += step_size
    return interp_spk
```

iii. The AI batches neurons in groups of 10,000 for interpolation to manage memory, following the reference code pattern.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `spk_pos_interp` function loops over each neuron individually calling `interp_value` with `scipy.interpolate.interp1d`. This could potentially be vectorized using multidimensional interpolation. The `make_lick_spatial_bins` function loops over individual lick events, which could be vectorized with numpy indexing.

ii.
```python
# In spk_pos_interp - loops over neurons:
for s in range(raw_spk.shape[0]):
    spk_resh.append(np.reshape(interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos), ...))

# In make_lick_spatial_bins - loops over lick events:
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
    lick_arr[tr, bin_idx] = 1
```

iii. The neuron loop in `spk_pos_interp` was copied directly from the reference code. Vectorization would require significant refactoring.

## 12-c. What processing does the code repeat multiple times?

i. Behavior data files are loaded multiple times: once during the first pass for speed quartile computation, and again during the second pass for building the dataset. However, results are cached in `beh_cache` to avoid redundant file I/O. Stimulus names are also scanned separately from the main loop.

ii.
```python
# First pass loads behavior:
beh = beh_cache[exp_type][beh_key]
run_pos_all.append(beh['run_pos'])

# Separate stimulus scan:
for exp_type_key in exp_info.keys():
    beh_data = np.load(beh_path, allow_pickle=True).item()
    for beh_key_inner in beh_data:
        for name in np.unique(beh_data[beh_key_inner]['WallName']):
            all_stim_names_set.add(str(name))
```

iii. The caching mitigates repeated I/O, but the stimulus name collection could have been integrated into the first pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes all 60 spatial bins (40 corridor + 20 gray space) in the output. The gray space bins are included as a 5th position category, even though the instructions only specify 4 corridor position bins. The gray space neural activity and behavioral data are processed but may be less informative for the decoder. Additionally, the code stores neural data as float32 arrays for every neuron (up to 78K per session), which creates very large file sizes (~410 GB), though all neurons may not be needed for downstream analysis.

ii.
```python
n_bins = 60  # spatial bins per trial (40 corridor + 20 gray)
n_position_categories = 5  # 4 corridor bins + gray space
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
```

iii. The reference code uses all 60 bins (including gray space), so the AI followed this convention. However, the instructions only mention 4 position bins for the corridor.
