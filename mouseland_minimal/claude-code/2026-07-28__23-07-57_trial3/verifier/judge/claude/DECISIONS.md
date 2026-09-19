# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded from three directories under `data`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. `beh/Imaging_Exp_info.npy` is loaded first as the master index. A mapping from session key to behavior file/key is built via `get_session_beh_mapping()`. Behavior files are cached by experiment type. Neural data and retinotopy are loaded per-session. The AI performs two passes: first to collect running speed data for global quartile computation, then to build the dataset.

ii.
```python
exp_info = np.load(os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))

# Load behavior
beh_cache[exp_type] = np.load(beh_path, allow_pickle=True).item()

# Load neural data
spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
# which does:
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)

# Load retinotopy
iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
# which does:
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The AI recognized the master index structure and built a session-to-behavior mapping. It chose to cache behavior files by experiment type to avoid redundant loads.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from the session key by splitting on `_` (first element). All unique mice are sorted and indexed. `subject_idx` maps each session to its mouse index.

ii.
```python
all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
# Per session:
subject_idx_list.append(mouse_to_idx[mname])
```

iii. The AI used the session key naming convention to extract mouse names.

## 1-c. How are the data split into sessions?

i. A session is identified by the unique key `mname_datexp_blk`. When a session appears in multiple experiment types, `get_session_beh_mapping` keeps one entry, preferring the experiment type with more stimuli (based on `stim_id`). This yields the same 89 unique sessions.

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
                # Prefer experiment type with more stimuli
                old_nstim = np.sum(~np.isnan(old_ndb['stim_id'].astype(float)))
                new_nstim = np.sum(~np.isnan(ndb['stim_id'].astype(float)))
                if new_nstim > old_nstim:
                    session_map[key] = (exp_type, beh_key, ndb)
```

iii. The AI decided to prefer the experiment type with more stimuli when duplicates occur, rather than just keeping the first one encountered.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the behavior data using `beh['ntrials']`. Each trial corresponds to one corridor traversal. The AI does NOT use frame-level trial indices (`ft_trInd`) or corridor-space masks (`ft_CorrSpc`). Instead, the AI spatially interpolates neural data into 60 position bins per trial (40 corridor + 20 gray space bins) using accumulated position and only running frames (`ft_move > 0`).

ii.
```python
ntrials = beh['ntrials']
# ...
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :]  # shape (n_neurons, 60)
```

The spatial interpolation:
```python
VRmove = beh['ft_move'][:nfr] > 0
ft_AcumPos = beh['ft_PosCum'][:nfr]
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove], ft_AcumPos[VRmove], ntrials, n_bins=int(CL), lengths=CL
)
```

iii. The AI followed the reference paper's spatial interpolation approach, which bins neural activity by position rather than by time. This is how the paper's original analyses were conducted.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters sessions with fewer than 2 trials or fewer than 10 valid neurons. It does NOT filter individual trials based on length or other quality criteria. No 99th percentile trial length filter is applied.

ii.
```python
if ntrials < 2:
    print(f"    Skipping: only {ntrials} trials")
    skipped_sessions.append(sess_key)
    continue

if n_valid < 10:
    print(f"    Skipping: only {n_valid} valid neurons")
    skipped_sessions.append(sess_key)
    continue
```

iii. The AI noted that sessions need at least 2 trials for decoder evaluation and applied a minimum neuron count threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` (deconvolved calcium traces), concatenated across imaging planes. Brain region assignment comes from `iarea` in `retinotopy/<mouse>_<datexp>_trans.npz`.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
```

iii. The AI correctly identified the deconvolved traces as the neural data source.

## 2-b. How is the `neural` data processed?

i. The neural data is spatially interpolated into 60 position bins per trial. Only running frames (`ft_move > 0`) are included. The accumulated position (`ft_PosCum`) is used to map temporal frames to spatial positions. The interpolation is done using `scipy.interpolate.interp1d` with extrapolation. The result is stored as float32. All trials have the same fixed length of 60 bins.

ii.
```python
VRmove = beh['ft_move'][:nfr] > 0
ft_AcumPos = beh['ft_PosCum'][:nfr]
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove], ft_AcumPos[VRmove], ntrials, n_bins=int(CL), lengths=CL
)

def spk_pos_interp(raw_spk, accum_pos, corridorLen, new_shape):
    linPos = np.arange(0, new_shape[0], 1 / new_shape[1])
    spk_resh = []
    for s in range(raw_spk.shape[0]):
        spk_resh.append(np.reshape(
            interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos),
            (int(new_shape[0]), int(new_shape[1]))
        ))
    return np.array(spk_resh)
```

iii. The AI followed the reference paper's spatial interpolation approach, which was used in the paper's analyses. The AI noted: "Spatial interpolation: 60 bins per trial (40 corridor + 20 gray), each 1 dm" and "Only running frames included (ft_move > 0), consistent with paper."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` or `iarea == 7` are excluded (outside visual cortex). The remaining neurons are assigned to V1, mHV, lHV, or aHV based on `iarea` codes.

ii.
```python
valid_neuron_mask = (iarea != -1) & (iarea != 7)
spk_filtered = spk[valid_neuron_mask]

def neu_area_ID(iarea):
    idx = {}
    idx['V1'] = iarea == 8
    idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
    idx['lHV'] = (iarea == 5) | (iarea == 6)
    idx['aHV'] = (iarea == 3) | (iarea == 4)
    return idx
```

iii. The AI stated it excluded neurons outside visual cortex, consistent with the reference code's area assignments.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to trial start (corridor entry) via the spatial interpolation. Each trial starts at position 0 (corridor entry) and the 60 spatial bins span the full corridor (40 dm) plus gray space (20 dm). All trials have the same fixed length of 60 bins.

ii.
```python
neural_trial = interp_spk[:, trial, :].astype(np.float32)
# interp_spk shape: (n_valid_neurons, ntrials, n_bins=60)
```

iii. The AI's alignment is spatial rather than temporal. The corridor entry is position 0, and all trials are aligned to this spatial reference point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI sets the time bin size to 166.67 ms, computed as 1 dm / 6 dm/s at the constant VR speed of 60 cm/s. This is a spatial bin size (1 decimeter) converted to time assuming constant virtual speed, not the actual imaging frame rate. Each trial has exactly 60 bins.

ii.
```python
time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms
# In metadata:
'time_bin_size': time_bin_sec * 1000,  # 166.67 ms
```

iii. The AI reasoned that at a constant VR speed of 60 cm/s, each 1 dm spatial bin corresponds to 166.67 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos`, the position (in dm) at which the sound cue was played in each trial.

ii.
```python
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. The AI used spatial position of the sound cue rather than the frame number, consistent with its spatial binning approach.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundPos - position_index) * time_bin_sec`, where `positions = np.arange(60)` is the spatial bin index and `time_bin_sec = 1/6`. This gives positive values before the cue and negative values after.

ii.
```python
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. The AI converted spatial distance to the sound cue into time using the constant VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both the neural data and the time-to-cue input are on the same 60-bin spatial grid, so they are inherently aligned.

ii.
```python
input_trial = np.stack([
    time_to_cue.astype(np.float32),  # shape (60,)
    ...
], axis=0)  # shape (4, 60)
```

iii. Alignment is through the shared spatial binning scheme.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date string in `datexp` field of each session entry in the experiment info. The AI parses dates using `datetime.strptime` and computes calendar day offsets from each mouse's first session.

ii.
```python
def compute_day_of_training(exp_info):
    mouse_dates = defaultdict(list)
    session_dates = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
            date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')
            mouse_dates[ndb['mname']].append(date)
            session_dates[key] = (ndb['mname'], date)
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
    session_days = {}
    for key, (mname, date) in session_dates.items():
        session_days[key] = (date - mouse_first_date[mname]).days
    return session_days
```

iii. The AI computed actual calendar day offsets rather than ordinal session counts.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is the number of calendar days between the session date and the mouse's first session date. This differs from counting ordinal recording sessions. The value is broadcast as a constant across all 60 spatial bins of every trial.

ii.
```python
session_days[key] = (date - mouse_first_date[mname]).days
# Per trial:
day_array = np.full(n_bins, day, dtype=np.float32)
```

iii. The AI chose calendar days rather than ordinal session count, reasoning that it represents the actual elapsed training time.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the spatial bin index (0 to 59), converted to time using the constant VR speed.

ii.
```python
positions = np.arange(n_bins)  # 0 to 59
time_since_start = positions * time_bin_sec  # shape (60,)
```

iii. The AI used spatial position as a proxy for time, assuming constant VR speed.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed as `position_index * time_bin_sec`, where `time_bin_sec = 1/6` seconds. This gives a linear ramp from 0 to ~9.83 seconds across the 60 bins.

ii.
```python
time_since_start = positions * time_bin_sec  # shape (60,)
```

iii. The AI assumed constant VR speed to convert spatial position to elapsed time.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Both are on the same 60-bin spatial grid, so they are inherently aligned.

ii.
```python
input_trial = np.stack([
    ...
    time_since_start.astype(np.float32),
    ...
], axis=0)
```

iii. Alignment is through the shared spatial binning scheme.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether the trial was in a rewarded corridor.

ii.
```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. The AI correctly identified `isRew` as the source.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The binary reward flag is broadcast as a constant across all 60 spatial bins of each trial.

ii.
```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. No additional processing needed beyond broadcasting.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name on the corridor walls of each trial.

ii.
```python
WallName = beh['WallName']
stim_idx = stim_to_idx[str(WallName[trial])]
```

iii. The AI correctly identified `WallName` as the source variable.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects ALL unique wall names across ALL sessions (15 total) and uses each as a separate category. It does NOT group them into base texture categories (circle, leaf, rock, wood). The stimulus index maps each of the 15 names to a unique integer.

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

# Per trial:
stim_idx = stim_to_idx[str(WallName[trial])]
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```

iii. The AI enumerated all unique wall names as separate stimulus categories rather than grouping variants into base textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickPos` (lick position in spatial dm) and `LickTrind` (trial index of each lick).

ii.
```python
lick_spatial = make_lick_spatial_bins(
    beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL)
)
```

iii. The AI used the spatial lick variables rather than the frame-based `LickFr`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Licks are binned into spatial bins based on `LickPos`. Each spatial bin is set to 1 if at least one lick occurred in it, 0 otherwise.

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

iii. The AI spatially binned lick events to match the spatial binning of neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Both are on the same 60-bin spatial grid. Licks are assigned to spatial bins matching the neural data's spatial bins.

ii.
```python
lick_trial = lick_spatial[trial, :].astype(np.int64)  # shape (60,)
```

iii. Alignment is through the shared spatial binning scheme.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From the spatial bin index itself (0 to 59), NOT from `ft_Pos`. Since the data is spatially interpolated, position is deterministic from the bin index.

ii.
```python
positions = np.arange(n_bins)
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
# position_bins_edges = [10, 20, 30, 40]
```

iii. The AI derived position from the spatial bin index since the neural data is already in spatial coordinates.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 60 spatial bins are categorized into 5 groups: 4 corridor bins of 10 dm (1 m) each, plus gray space (bins 40-59).

ii.
```python
position_bins_edges = [10, 20, 30, 40]
n_position_categories = 5  # 4 corridor bins + gray space
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. The AI used `np.digitize` to bin positions into 5 categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Positions 0-9 -> bin 0 (0-1m), 10-19 -> bin 1 (1-2m), 20-29 -> bin 2 (2-3m), 30-39 -> bin 3 (3-4m), 40-59 -> bin 4 (gray_space). This creates 5 categories instead of the 4 requested in the instructions.

ii.
```python
position_bins_edges = [10, 20, 30, 40]
pos_category = np.digitize(positions, position_bins_edges)
# Results: 0, 1, 2, 3, 4
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
```

iii. The AI included the gray space as a 5th category because the spatial interpolation includes 20 gray space bins (bins 40-59).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Both are on the same 60-bin spatial grid. Position is deterministic from the bin index.

ii.
```python
pos_trial = pos_category.copy()  # shape (60,)
```

iii. Alignment is inherent in the spatial binning scheme.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `run_pos`, the running speed at each spatial position for each trial (shape: ntrials x 60).

ii.
```python
run_pos = beh['run_pos']  # (ntrials, 60)
```

iii. The AI used the spatially-binned running speed variable rather than the per-frame `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speeds are discretized into 4 quartile bins using global percentile thresholds computed across ALL sessions. The thresholds are the 25th, 50th, and 75th percentiles of all `run_pos` values across the entire dataset.

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

iii. The AI computed global quartile edges from all sessions' `run_pos` data. This differs from computing quartiles per-session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` is used with the global quartile edges, producing values 0-3. Bins are labeled with the quartile edge values.

ii.
```python
speed_bins = speed_to_bins(run_pos, speed_edges)
speed_labels = [
    f'Q1(<{speed_edges[0]:.1f})',
    f'Q2({speed_edges[0]:.1f}-{speed_edges[1]:.1f})',
    f'Q3({speed_edges[1]:.1f}-{speed_edges[2]:.1f})',
    f'Q4(>{speed_edges[2]:.1f})'
]
```

iii. Using percentile-based thresholds ensures approximately equal numbers of data points per bin globally, though not necessarily per session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Both are on the same 60-bin spatial grid. `run_pos` is already in the spatial format matching the neural data.

ii.
```python
speed_trial = speed_bins[trial, :].astype(np.int64)  # shape (60,)
```

iii. Alignment is through the shared spatial binning scheme.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles NaN/Inf in neural data by replacing with 0. Sessions with fewer than 10 valid neurons or fewer than 2 trials are skipped. Exceptions during neural/retinotopy loading cause the session to be skipped. The behavior is cut to the number of imaged frames (`nfr = spk.shape[1]`).

ii.
```python
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)

if n_valid < 10:
    skipped_sessions.append(sess_key)
    continue

VRmove = beh['ft_move'][:nfr] > 0
```

iii. The AI added defensive checks for NaN/Inf values and applied minimum thresholds for session validity.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the large spike files (total ~405 GB across all sessions) and the spatial interpolation of neural data (per-neuron interpolation loop using `scipy.interpolate.interp1d`).

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)

# Spatial interpolation - loops over each neuron
for s in range(raw_spk.shape[0]):
    spk_resh.append(np.reshape(
        interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos), ...
    ))
```

iii. The I/O cost of loading neural data is unavoidable. The spatial interpolation adds significant computation.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The spatial interpolation loops over neurons one at a time (`for s in range(raw_spk.shape[0])`), calling `interp1d` for each. This could potentially be vectorized. The lick spatial binning also loops over individual licks.

ii.
```python
# Per-neuron interpolation loop
for s in range(raw_spk.shape[0]):
    spk_resh.append(np.reshape(
        interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos), ...
    ))

# Per-lick loop
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
```

iii. The per-neuron loop is partially mitigated by processing in batches of 10,000 neurons via `get_interpPos_spk`.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded twice: once in the first pass for running speed collection, and again implicitly used in the second pass (cached). The stimulus name collection iterates over all behavior files separately from the main processing.

ii.
```python
# First pass: speed collection
for sess_key in all_sessions_full:
    beh = beh_cache[exp_type][beh_key]
    run_pos_all.append(beh['run_pos'])

# Separate stimulus name collection
for exp_type_key in exp_info.keys():
    beh_data = np.load(beh_path, allow_pickle=True).item()
    for beh_key_inner in beh_data:
        for name in np.unique(beh_data[beh_key_inner]['WallName']):
            all_stim_names_set.add(str(name))
```

iii. The stimulus name collection redundantly loads behavior files that were already cached.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes gray space bins (positions 40-59) in the spatial interpolation, resulting in 60 bins per trial. The reference solution only includes corridor space (positions 0-39). The gray space data and the 5th position category ("gray_space") are unnecessary for the decoder task which specifies 4 equal-length 1-m-long spatial bins. Additionally, the spatial interpolation itself is unnecessary if temporal frames are used directly.

ii.
```python
n_bins = 60  # spatial bins per trial (40 corridor + 20 gray)
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove], ft_AcumPos[VRmove], ntrials, n_bins=int(CL), lengths=CL
)
```

iii. The AI included gray space to capture the full corridor traversal, but the decoder task only asks for 4 position bins of 1 m each (the texture corridor).
