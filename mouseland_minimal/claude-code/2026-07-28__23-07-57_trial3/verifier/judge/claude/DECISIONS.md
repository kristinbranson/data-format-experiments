# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the experiment metadata from `Imaging_Exp_info.npy`, which is a dictionary keyed by experiment type (e.g., `unsup_train1_before_learning`, `sup_test1`, etc.), each containing a list of session metadata dicts. It then builds a mapping from unique session keys (`mname_datexp_blk`) to their behavior data and experiment type. Behavior data is loaded from per-experiment-type files (`Beh_{exp_type}.npy`). Neural data is loaded per-session from `data/spk/{mname}_{datexp}_{blk}_neural_data.npy`. Retinotopy data is loaded from `data/retinotopy/{mname}_{datexp}_trans.npz`.

ii.
```python
exp_info = np.load(os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))
# ...
spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
beh = beh_cache[exp_type][beh_key]
```

iii. The AI documented this in CONVERSION_NOTES.md: "Neural data (data/spk/): 89 .npy files ... Behavior data (data/beh/): .npy files organized by experiment type ... Retinotopy (data/retinotopy/): .npz files with brain area assignments."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by extracting the mouse name from the session key (first component of `mname_datexp_blk`). All unique mouse names are collected and sorted. A `subject_idx` array maps each session to its subject index.

ii.
```python
all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
# ...
subject_idx_list.append(mouse_to_idx[mname])
```

iii. The AI noted 19 unique mice were found, matching the paper's reported 19 mice.

## 1-c. How are the data split into sessions?

i. Each unique recording identified by `mname_datexp_blk` is treated as one session. When a recording appears in multiple experiment types in the metadata, the AI selects the experiment type with the most stimuli (most non-NaN `stim_id` entries) to avoid duplicates.

ii.
```python
all_sessions = sorted(session_map.keys())
# In get_session_beh_mapping:
if key not in session_map:
    session_map[key] = (exp_type, beh_key, ndb)
else:
    old_nstim = np.sum(~np.isnan(old_ndb['stim_id'].astype(float)))
    new_nstim = np.sum(~np.isnan(ndb['stim_id'].astype(float)))
    if new_nstim > old_nstim:
        session_map[key] = (exp_type, beh_key, ndb)
```

iii. CONVERSION_NOTES.md: "Each unique recording (identified by mouse_date_block) = one session... We use each unique recording exactly once."

## 1-d. How are the data split into trials?

i. Trials are determined by the `ntrials` field in the behavior data for each session. All trials from index 0 to `ntrials-1` are included. The neural data is spatially interpolated into 60 bins per trial, and each trial produces one entry in the session's trial list.

ii.
```python
ntrials = beh['ntrials']
# ...
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :]
    # ... build input/output for each trial
    neural_trials.append(neural_trial)
```

iii. The AI follows the reference code structure where `ntrials` from the behavior data defines the number of trials per session.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter individual trials. Sessions are skipped if they have fewer than 2 trials, fewer than 10 valid neurons, or if neural/retinotopy data cannot be loaded. But no trial-level quality filtering (e.g., based on running behavior) is applied.

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

iii. The output shows 0 sessions were skipped, so all 89 sessions passed the quality controls. No trial-level filtering is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key in the neural data files (`{mname}_{datexp}_{blk}_neural_data.npy`). These are deconvolved fluorescence traces from Suite2p, stored as a list of arrays (one per imaging plane).

ii.
```python
def load_spk(mname, datexp, blk, root='data/spk'):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. CONVERSION_NOTES.md: "Neural data: deconvolved fluorescence traces (Suite2p), concatenated across planes."

## 2-b. How is the `neural` data processed?

i. Neural data processing follows these steps: (1) Concatenate across imaging planes, (2) Filter to keep only neurons in visual cortex (exclude `iarea == -1` and `iarea == 7`), (3) Spatially interpolate into 60 position bins per trial using only running frames (`ft_move > 0`), using `get_interpPos_spk()` which applies `scipy.interpolate.interp1d` with accumulated position.

ii.
```python
spk_filtered = spk[valid_neuron_mask]
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

iii. CONVERSION_NOTES.md: "Concatenate imaging planes... Filter out non-visual-cortex neurons... Spatial interpolation into 60 bins per trial using only running frames (ft_move > 0)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered based on brain region assignment using retinotopy data (`iarea`). Neurons with `iarea == -1` (unassigned) or `iarea == 7` (outside visual cortex) are excluded. No other quality filtering (e.g., based on signal-to-noise ratio or firing rate) is applied.

ii.
```python
valid_neuron_mask = (iarea != -1) & (iarea != 7)
spk_filtered = spk[valid_neuron_mask]
```

iii. CONVERSION_NOTES.md: "Neurons with iarea == -1 or iarea == 7 are excluded (~10% of neurons per session)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) via spatial interpolation. The accumulated position (`ft_PosCum`) during running frames is used to interpolate neural activity into 60 equally-spaced position bins (0 to 60 dm), where position 0 corresponds to corridor entry. This means the first bin = trial start.

ii.
```python
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove],
    ft_AcumPos[VRmove],
    ntrials,
    n_bins=int(CL),
    lengths=CL
)
# interp_spk shape: (n_valid_neurons, ntrials, n_bins)
```

iii. CONVERSION_NOTES.md: "Alignment: Trial start = corridor entry (position 0)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 166.67 ms (1 dm / 6 dm/s at constant VR speed of 60 cm/s). No temporal rebinning is applied; the data is spatially binned, not temporally binned. Each of the 60 spatial bins corresponds to 1 dm of corridor position, and at constant VR speed this translates to ~166.67 ms per bin.

ii.
```python
n_bins = 60  # spatial bins per trial (40 corridor + 20 gray)
time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms
```

iii. CONVERSION_NOTES.md: "Time bin size: 166.67 ms (1 dm / 6 dm/s at constant VR speed)... 60 bins per trial: 40 corridor bins (0-4 m) + 20 gray space bins (4-6 m)."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Time to sound cue is derived from `beh['SoundPos']`, which gives the position (in dm) where the sound cue occurs for each trial, and the spatial bin position index.

ii.
```python
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. CONVERSION_NOTES.md: "Time to sound cue (seconds): (SoundPos - position) * (1/6). SoundPos is the position (in dm) where the sound cue occurs in each trial."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, `SoundPos[trial]` gives the position of the sound cue in dm. The time to sound cue at each spatial bin is computed as `(SoundPos - bin_position) * time_bin_sec`, where `time_bin_sec = 1/6` seconds. This produces a continuous time-varying signal that is positive before the cue position and negative after.

ii.
```python
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. CONVERSION_NOTES.md: "Positive before cue, negative after. SoundPos is the position (in dm) where the sound cue occurs in each trial, randomly drawn from uniform [5, 35] dm (0.5-3.5 m)."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both neural data and the time-to-cue input are indexed by the same 60 spatial bins (0-59 dm positions), so they are inherently aligned. The neural data is interpolated to these bins, and the time-to-cue is computed at these same bin positions.

ii.
```python
positions = np.arange(n_bins)  # 0 to 59
time_to_cue = (sound_pos - positions) * time_bin_sec
# Both neural_trial and time_to_cue have 60 entries aligned by spatial bin
```

iii. The alignment is implicit via shared spatial binning.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day of training is derived from the `datexp` field in the experiment metadata (session date strings in `YYYY_MM_DD` format) for each mouse.

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
    session_days[key] = (date - mouse_first_date[mname]).days
```

iii. CONVERSION_NOTES.md: "Day of training: Days since the mouse's first recording session, computed from dates in the experiment metadata."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is found. For each session, the day of training is computed as the number of days between the session date and the mouse's earliest date. This is a constant value per session, replicated across all 60 time bins.

ii.
```python
day = session_days.get(sess_key, 0)
day_array = np.full(n_bins, day, dtype=np.float32)
```

iii. CONVERSION_NOTES.md: "Day of training: Days since the mouse's first recording session."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI did **not** include an "Environment type" input. The instructions specified 4 inputs: Time to sound cue, Day of training, Time since trial start, and Reward availability. Environment type was not one of the specified decoder inputs.

ii. N/A - not implemented.

iii. No justification needed; the variable was not part of the specified decoder inputs.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not applicable - Environment type was not included as an input variable by the AI, and was not specified in the instructions.

ii. N/A.

iii. N/A.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Time since trial start is derived from the spatial bin index (0 to 59), converted to seconds using the time bin size.

ii.
```python
time_since_start = positions * time_bin_sec  # shape (60,)
```

iii. CONVERSION_NOTES.md: "Time since trial start (seconds): position * (1/6). Ranges from 0 to 9.83 s."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each spatial bin index (0 to 59) is multiplied by the time per bin (1/6 seconds = 166.67 ms) to get the time since trial start in seconds. This assumes constant VR speed.

ii.
```python
positions = np.arange(n_bins)  # 0, 1, ..., 59
time_since_start = positions * time_bin_sec  # 0, 0.167, 0.333, ..., 9.83
```

iii. The computation directly follows from the spatial binning and constant VR speed assumption.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Like all other variables, time since trial start is indexed by the same 60 spatial bins as the neural data. Bin 0 = trial start (corridor entry), so alignment is inherent.

ii.
```python
# Both use n_bins = 60, aligned by spatial position
time_since_start = positions * time_bin_sec
neural_trial = interp_spk[:, trial, :]  # also 60 bins
```

iii. Alignment via shared spatial binning.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived from `beh['isRew']`, a per-trial binary array indicating whether each trial is in a rewarded corridor.

ii.
```python
isRew = beh['isRew']
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. CONVERSION_NOTES.md: "Reward availability: Binary per trial from beh['isRew']. 1 = rewarded corridor, 0 = unrewarded."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The binary value `isRew[trial]` is converted to float and replicated across all 60 spatial bins to create a constant per-trial, time-varying input.

ii.
```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. CONVERSION_NOTES.md: "1 = rewarded corridor, 0 = unrewarded. Unsupervised mice have all zeros."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Visual stimulus category is derived from `beh['WallName']`, which gives the name of the wall texture for each trial (e.g., 'circle1', 'leaf1', etc.).

ii.
```python
WallName = beh['WallName']
stim_idx = stim_to_idx[str(WallName[trial])]
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```

iii. CONVERSION_NOTES.md: "Visual stimulus category: Per-trial, constant across time bins. Index into the full stimulus list across all sessions."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique wall names across all sessions and experiment types are collected and sorted alphabetically. Each trial's `WallName` is mapped to an integer index into this global sorted list. The index is replicated across all 60 time bins as a constant per-trial value. This produces 15 unique categories.

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

iii. The 15 stimulus categories found match those in the paper's experimental paradigm.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `beh['LickPos']` (position of each lick in dm) and `beh['LickTrind']` (trial index of each lick).

ii.
```python
lick_spatial = make_lick_spatial_bins(
    beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL)
)
```

iii. CONVERSION_NOTES.md: "Licking: Binary, time-varying. Mapped from LickPos and LickTrind to spatial bins."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick events are mapped to spatial bins: for each lick, its position is floored to get the spatial bin index, and the corresponding entry in a `(ntrials, n_bins)` binary array is set to 1. Positions are clipped to [0, n_bins-1].

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

iii. CONVERSION_NOTES.md: "1 = lick occurred in that spatial bin, 0 = no lick."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is mapped to the same 60 spatial bins as the neural data, using `LickPos` (position in dm). Both neural and licking data share the same spatial bin framework.

ii.
```python
bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
lick_trial = lick_spatial[trial, :]  # shape (60,)
```

iii. Alignment is inherent through shared spatial binning.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position in corridor is derived deterministically from the spatial bin indices (0-59), not from any raw behavioral variable. It is a fixed mapping from spatial bin index to position category.

ii.
```python
positions = np.arange(n_bins)
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. The position is inherent in the spatial binning structure.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 60 spatial bins are categorized using `np.digitize` with edges at [10, 20, 30, 40] dm. This produces 5 categories: bins 0-9 -> category 0 (0-1m), bins 10-19 -> category 1 (1-2m), bins 20-29 -> category 2 (2-3m), bins 30-39 -> category 3 (3-4m), bins 40-59 -> category 4 (gray space).

ii.
```python
position_bins_edges = [10, 20, 30, 40]
n_position_categories = 5  # 4 corridor bins + gray space
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
```

iii. CONVERSION_NOTES.md: "Position in corridor: 5 categories... 4 equal 1-m corridor bins... plus gray space."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 5 categories using bin edges at 10, 20, 30, 40 dm (1, 2, 3, 4 meters). The instructions specified "4 equal-length, 1-m-long spatial bins" but the AI added a 5th category for the gray space (bins 40-59).

ii.
```python
position_bins_edges = [10, 20, 30, 40]
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
pos_category = np.digitize(positions, position_bins_edges)
```

iii. The AI chose to include gray space as a 5th bin since the neural data spans 60 bins including the gray space region.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position categories are a deterministic function of the spatial bin index, which is the same indexing used for neural data. Alignment is inherent.

ii.
```python
pos_trial = pos_category.copy()  # same 60 bins as neural data
```

iii. No explicit alignment needed.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `beh['run_pos']`, a `(ntrials, 60)` array giving running speed at each spatial position bin.

ii.
```python
run_pos = beh['run_pos']
speed_bins = speed_to_bins(run_pos, speed_edges)
```

iii. CONVERSION_NOTES.md: "Running speed: 4 quartile bins... from beh['run_pos']."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speeds from all sessions are first concatenated to compute global quartile edges (25th, 50th, 75th percentiles). Then each speed value is discretized into 4 bins using `np.digitize`.

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

iii. CONVERSION_NOTES.md: "Quartile edges computed globally across all 89 sessions from beh['run_pos']. Each bin contains ~25% of the global data."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all running speed data across all 89 sessions. `np.digitize` with these 3 edges produces 4 bins (0-3), where each bin contains approximately 25% of all speed observations.

ii.
```python
edges = np.percentile(all_speeds, [25, 50, 75])
# edges = [16.96, 29.25, 44.25]
result = np.digitize(run_pos, edges)  # values 0, 1, 2, 3
```

iii. The quartile edges [16.96, 29.25, 44.25] are reported in the output.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed from `beh['run_pos']` is already organized as `(ntrials, 60)` matching the same 60 spatial bins as the interpolated neural data.

ii.
```python
speed_trial = speed_bins[trial, :]  # shape (60,), same bins as neural
```

iii. The `run_pos` variable is pre-computed in the behavior data at the same spatial resolution as the interpolated neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN/Inf values in neural data are replaced with zeros. Missing lick data for unsupervised mice (empty LickPos arrays) naturally produces all-zero lick arrays. Sessions with loading errors are skipped. No other missing data handling is implemented.

ii.
```python
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. CONVERSION_NOTES.md: "No NaN or Inf values in final neural arrays (cleaned during conversion)."

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Spatial interpolation of neural data via `get_interpPos_spk()`, which loops over neurons in batches of 10,000 and applies `scipy.interpolate.interp1d` for each neuron. (2) Loading the large neural data files (up to 89,577 neurons per recording). (3) The first pass collecting all running speed data for global quartile computation.

ii.
```python
def get_interpPos_spk(spk, spk_culm_pos, ntrial, n_bins=60, lengths=60):
    interp_spk = np.zeros((spk.shape[0], ntrial, n_bins))
    step_size = 10000
    i = 0
    while i <= spk.shape[0]:
        interp_spk[i:i+step_size, :] = spk_pos_interp(...)
        i += step_size
```

iii. The interpolation is the computational bottleneck, processing 20K-90K neurons per session.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `spk_pos_interp` function loops over individual neurons to apply `interp1d`, which could potentially be vectorized using batch interpolation. The `make_lick_spatial_bins` function loops over individual lick events, which could use `np.add.at` or histogram-based binning instead.

ii.
```python
# In spk_pos_interp - loops per neuron:
for s in range(raw_spk.shape[0]):
    spk_resh.append(np.reshape(
        interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos), ...))

# In make_lick_spatial_bins - loops per lick:
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
    lick_arr[tr, bin_idx] = 1
```

iii. The reference code also uses per-neuron loops for interpolation, so this follows the same pattern.

## 12-c. What processing does the code repeat multiple times?

i. The code makes two passes over all sessions: first to collect running speed data for global quartile computation, and second to build the full dataset. Additionally, it loads and scans all behavior files to collect unique stimulus names separately from the main processing loop. The behavior files are loaded and cached in the first pass, so they are not re-read from disk in the second pass.

ii.
```python
# First pass: collecting running speeds
for sess_key in all_sessions_full:
    beh = beh_cache[exp_type][beh_key]
    run_pos_all.append(beh['run_pos'])

# Separate loop for stimulus names
for exp_type_key in exp_info.keys():
    beh_data = np.load(beh_path, allow_pickle=True).item()
    for beh_key_inner in beh_data:
        for name in np.unique(beh_data[beh_key_inner]['WallName']):
            all_stim_names_set.add(str(name))
```

iii. The two-pass approach is necessary for global quartile computation before per-session processing.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `time_to_cue` and `time_since_start` for all 60 bins including the gray space (bins 40-59), even though the gray space has no visual stimulus and is arguably less relevant for decoding. The code also generates comprehensive metadata fields that are not used by the decoder. The `sample_data.pkl` creation includes all subjects even though only a subset of sessions are in the sample.

ii.
```python
# Sample data includes all subjects even though only 5 sessions used:
sample_data = {
    'subjects': data['subjects'],  # all 19 mice, not just those in 5 sessions
    ...
}
```

iii. No specific justification provided; these are minor inefficiencies that don't affect correctness.
