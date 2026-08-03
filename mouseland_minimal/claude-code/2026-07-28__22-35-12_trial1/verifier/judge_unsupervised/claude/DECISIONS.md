# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, which contains 23 experiment types, each listing recordings. For each experiment type, the corresponding behavior file `Beh_{exp_type}.npy` is loaded. Neural data is loaded per session from `{mname}_{datexp}_{blk}_neural_data.npy` files, and retinotopy from `{mname}_{datexp}_trans.npz`. Sessions are deduplicated by `(mname, datexp, blk)` to avoid processing the same physical recording multiple times.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
# ...
sessions = collect_all_sessions(exp_info, ROOT)
# In collect_all_sessions:
for exp_type in exp_order:
    db_list = exp_info[exp_type]
    beh_path = os.path.join(root, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_path, allow_pickle=True).item()
    for ndb in db_list:
        rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
        if rec_key in seen_recordings:
            continue
        # ... load behavior by key
# In process_session:
spk_fn = f"{mname}_{datexp}_{blk}_neural_data.npy"
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The AI followed the reference code's `data_process_script.ipynb` structure, which iterates over experiment types and loads behavior and neural data. The deduplication by `(mname, datexp, blk)` was added because the AI verified that 33 of 89 recordings appeared in multiple experiment types with identical behavior data. The neural loading via concatenating `spks` planes matches `load_spk()` from `utils.py`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field from experiment metadata. All unique mouse names are collected and sorted alphabetically. Each session is mapped to its subject via `all_subjects.index(mname)`.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions))
# ...
all_subject_idx.append(all_subjects.index(result['mname']))
```

iii. The `mname` field is the standard mouse identifier used throughout the reference code and data. 19 unique subjects were found, matching the paper's statement of "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Each unique `(mname, datexp, blk)` tuple defines a session. When a recording appears in multiple experiment types (e.g., both `sup_test1` and `sup_train1_after_learning`), only one instance is kept, with preference ordering: test > non-test, supervised > unsupervised, after-learning > before-learning.

ii.
```python
# In collect_all_sessions:
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
# Preference ordering:
exp_order = sorted(exp_info.keys(), key=lambda x: (
    0 if 'test' in x else 1,
    0 if 'sup_' in x else 1,
    0 if 'after' in x else 1,
    x
))
```

iii. The AI verified that behavior data was identical across experiment types for the same physical recording, so deduplication does not lose information. The preference ordering ensures that when behavior keys differ (e.g., test3 swap variants with `stimtype`), the more informative version is preferred.

## 1-d. How are the data split into trials?

i. Trial count is taken directly from `beh['ntrials']` for each session. Each trial corresponds to one corridor traversal. Position-interpolated neural data is reshaped into `(n_neurons, n_trials, n_bins)` and then split into per-trial matrices.

ii.
```python
n_trials = beh['ntrials']
# ...
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving],
                                       corridor_len, n_trials, n_bins=N_POS_BINS)
# ...
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The trial structure comes from the behavior data's `ntrials` field, which defines how many corridor traversals occurred. This matches the reference code's approach where `ntrials` from behavior determines the reshaping of position-interpolated data.

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filter is requiring at least 2 trials per session (`n_trials < 2`). No other trial quality filtering is applied (e.g., no filtering by running behavior, trial duration, or completeness).

ii.
```python
if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
```

iii. The AI's CONVERSION_NOTES do not mention any trial-level quality filtering beyond the minimum trial count. The reference code's basic position interpolation pipeline also does not filter trials -- trial filtering in the reference code (e.g., `fr_valid = VRmove & isCorridor`) is used for specific analyses like neuron selectivity, not for the basic processing pipeline.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `spks` in the `{mname}_{datexp}_{blk}_neural_data.npy` files. These are Suite2p deconvolved calcium fluorescence traces, concatenated across imaging planes.

ii.
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The AI followed `load_spk()` from `utils.py`, which concatenates planes from `spk_data['spks']`. The paper states "All our analyses were based on deconvolved fluorescence traces" with Suite2p deconvolution using tau=0.75s.

## 2-b. How is the `neural` data processed?

i. Neural data is position-interpolated from irregular frame-by-frame samples to evenly-spaced position bins (60 bins per corridor). Only running frames (`ft_move > 0`) are used. The interpolated data is then trimmed to the first 40 bins (texture corridor area). Interpolation uses `numpy.interp` (1D linear interpolation).

ii.
```python
VR_SPEED = 6.0
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
ft_pos_cum = beh['ft_PosCum'][:n_frames]
valid_spk = spk[valid_neurons][:, vr_moving]
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving],
                                       corridor_len, n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]  # Keep only first 40 bins

def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
    norm_pos = accum_pos / corridor_len
    for s in range(n_neurons):
        interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
    return interp_spk.reshape(n_neurons, n_trials, n_bins)
```

iii. This matches the reference code's `spk_pos_interp()` and `get_interpPos_spk()` from `utils.py`, which normalize cumulative position by corridor length and interpolate to a linear position grid. The AI used `numpy.interp` instead of `scipy.interpolate.interp1d` for speed, noting they are equivalent for 1D linear interpolation. The reference code used `interp1d(fill_value='extrapolate')`, while `numpy.interp` clamps to boundary values, but this should produce identical results within the data range.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only visual cortex neurons are included. Neurons with `iarea == -1` (not in visual cortex) or `iarea == 7` (undefined region) are excluded. No further neuron quality filtering (e.g., by firing rate, responsiveness, or SNR) is applied.

ii.
```python
def get_brain_region_idx(iarea):
    valid_mask = (iarea != -1) & (iarea != 7)
    region_idx = np.full(len(iarea), -1, dtype=int)
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return region_idx, valid_mask
```

iii. This matches `neu_area_ID()` and `load_retino()` from `utils.py`. The retinotopy data provides `iarea` values per neuron. The paper's selective neuron analysis (`Get_dprime_selective_neuron`) is for specific figures, not for the basic processing pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry). Position interpolation inherently aligns to corridor entry because position 0 corresponds to the start of each corridor. The first position bin (index 0) corresponds to corridor entry, and subsequent bins track progression through the corridor.

ii.
```python
# Position interpolation aligns to corridor traversal:
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
# Each trial starts at integer positions (0, 1, 2, ...), bins within trial are 0/n_bins to (n_bins-1)/n_bins
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The position interpolation naturally achieves this: cumulative position is normalized by corridor length, and the linear grid starts at trial 0. Each trial's data spans bins 0 to n_bins-1, representing the full corridor traversal from entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 166.67 ms, derived from the VR speed: each position bin is 1 dm, and at 6 dm/s constant VR speed, each bin takes 1/6 s = 166.67 ms. No temporal rebinning is applied -- the data is position-interpolated, and the time resolution is a consequence of the spatial binning and constant VR speed.

ii.
```python
VR_SPEED = 6.0        # dm/s (60 cm/s)
BIN_SIZE_SEC = 1.0 / VR_SPEED  # seconds per position bin = 1/6 s
BIN_SIZE_MS = BIN_SIZE_SEC * 1000  # 166.67 ms
```

iii. The paper states the VR moved at constant 60 cm/s (6 dm/s) when mice ran above the 6 cm/s threshold. Since data is position-interpolated (not time-binned), the temporal resolution is derived from the spatial bin size and VR speed. 40 bins per trial at 166.67 ms each gives ~6.67 seconds per trial, consistent with 4m at 60 cm/s.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundPos']`, which gives the position (in dm) within the corridor where the sound cue is programmed to occur, per trial.

ii.
```python
sound_pos = beh['SoundPos']
# ...
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The behavior data contains both `SoundPos` (corridor-relative position in dm) and `SoundDelPos` (cumulative position across corridors). The AI chose `SoundPos` because it provides within-corridor positions (values ~5-36 dm), which is appropriate for computing time relative to the sound cue within each trial. `SoundDelPos` contains cumulative positions (values in hundreds) that are not directly usable per-trial.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each position bin `i` (0-39), the time to sound cue is computed as `(i - SoundPos[trial]) * BIN_SIZE_SEC`, where `BIN_SIZE_SEC = 1/6`. This gives time in seconds, negative before the sound cue and positive after.

ii.
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. Each position bin represents 1 dm. At 6 dm/s VR speed, the time offset from the sound cue position is `(position_bin - sound_position) / speed = (i - SoundPos) * (1/6)` seconds. The resulting range across all sessions is approximately [-6.5, 6.3] seconds, consistent with the sound cue being placed between 0.5m and 3.5m (5-35 dm).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both time_to_sound_cue and neural data are indexed by the same position bins (0-39), so they are inherently aligned. Position bin `i` in the neural data corresponds to position bin `i` in the time_to_sound_cue computation.

ii.
```python
# Both use the same bin index i from 0 to N_TEXTURE_BINS-1
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail], axis=0)  # (4, 40)
```

iii. Since all variables are in position-bin space, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from `ndb['datexp']`, the date string (format `YYYY_MM_DD`) associated with each recording session.

ii.
```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')

# Compute first recording date per mouse
mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d

# Days since first recording
session_days[key] = float((d - mouse_first_date[s['mname']]).days)
```

iii. The AI computed day of training as the number of days since each mouse's first recording date. This makes the variable relative per mouse. Values range from 0 to 92 across all sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The recording date is parsed from `datexp`, the earliest recording date per mouse is found, and the day of training is the difference in days. This value is constant within a trial and broadcast across all 40 time bins.

ii.
```python
day_val = session_days[rec_key]
for trial_input in result['input']:
    trial_input[1, :] = day_val  # row 1 = day_of_training
# ...
day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)
```

iii. The instruction specifies "Day of training, continuous, time-varying." The AI makes it time-varying by broadcasting the same value across all 40 bins. The value is continuous (float) and per-session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the position bin index and the VR speed. No raw data variable is used directly -- it is computed from the bin structure.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. Since each position bin corresponds to 1 dm at 6 dm/s, the time since trial start at bin `i` is `i * (1/6)` seconds, ranging from 0 to ~6.5 seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Simple multiplication: bin_index * BIN_SIZE_SEC. The values are [0, 1/6, 2/6, ..., 39/6] seconds.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The time since trial start is the same for every trial (since VR speed is constant and the corridor is the same length), creating a linear ramp from 0 to ~6.5 seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned by position bin index. Bin 0 corresponds to corridor entry (trial start), and both neural data and time_since_start share the same indexing.

ii.
```python
# Same 40-bin structure as neural data
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. Since all data streams (neural, inputs, outputs) are in the same position-bin coordinate system, alignment is inherent.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a per-trial boolean array indicating whether the corridor is rewarded.

ii.
```python
is_rew = beh['isRew'].astype(float)
# ...
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The `isRew` field is a per-trial boolean. The AI confirmed that only supervised sessions have rewarded trials (isRew=1), while unsupervised/naive/grating sessions are all unrewarded (isRew=0).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (0.0 or 1.0) and broadcast across all 40 time bins per trial.

ii.
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The instruction specifies "1 if in rewarded corridor, 0 if not, discrete, per-trial." The AI correctly broadcasts the per-trial value across time bins to match the (4, 40) input shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which gives the stimulus/texture name for each trial (e.g., 'circle1', 'leaf2', 'wood5').

ii.
```python
wall_names = beh['WallName']
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. `WallName` provides the actual stimulus name per trial. All unique names across all sessions are collected and sorted to create a consistent index mapping.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique stimulus names across all sessions are collected, sorted alphabetically, and each trial's stimulus is mapped to an integer index (0-14). The per-trial value is broadcast across all 40 time bins.

ii.
```python
all_stim_set = set()
for s in sessions:
    for wn in s['beh']['UniqWalls']:
        all_stim_set.add(str(wn))
all_stim_names = sorted(all_stim_set)
# ...
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. 15 unique stimuli were found: circle1/2/3, leaf1/1_swap1/1_swap2/2/3, rock1/2, wood1/1_swap1/1_swap2/2/5. These are mapped to integers 0-14.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickPos']` (position of each lick event in dm) and `beh['LickTrind']` (trial index of each lick event).

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
```

iii. Lick events are stored as discrete events with position and trial information, which are then binned.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick events are converted to binary (0/1) per position bin per trial. Each lick's position is floored to an integer bin index, and if any lick falls in a given bin, that bin is set to 1.

ii.
```python
def lick_to_position_bins(lick_pos, lick_trind, n_trials, n_bins=40):
    lick_binary = np.zeros((n_trials, n_bins), dtype=int)
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        bin_idx = int(pos)
        if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
            lick_binary[tr, bin_idx] = 1
    return lick_binary
```

iii. The instruction specifies "Licking, binary, time-varying. 0 = not licking, 1 = licking." The AI creates a binary array per position bin, which is time-varying across the 40 bins. Licking is very sparse (1.9% of bin-trial combinations have licks), concentrated in supervised sessions.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick positions (in dm) are directly mapped to position bin indices, which share the same coordinate system as the position-interpolated neural data.

ii.
```python
bin_idx = int(pos)  # LickPos is in dm, bin index = floor(position in dm)
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

iii. Since `LickPos` is in dm and each position bin represents 1 dm, the integer conversion directly maps lick positions to bins, aligned with the neural data's position bins.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position in corridor is not derived from raw data but is deterministically constructed from the position bin index. Each of the 40 bins maps to one of 4 spatial bins.

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)  # 0-9 -> 0, 10-19 -> 1, 20-29 -> 2, 30-39 -> 3
```

iii. Since the data is position-interpolated, position is known by construction. No raw data variable is needed.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 position bins are grouped into 4 spatial bins of 10 bins each (1m each): bins 0-9 -> category 0, 10-19 -> 1, 20-29 -> 2, 30-39 -> 3.

ii.
```python
pos_bins[i] = min(i // 10, 3)
```

iii. Since each bin is 1 dm and there are 40 bins (4m), grouping by 10 gives 4 equal 1m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Integer division by 10 creates 4 equal bins: 0-1m (category 0), 1-2m (category 1), 2-3m (category 2), 3-4m (category 3). Each category has exactly 25% of bins by construction.

ii.
```python
pos_bins[i] = min(i // 10, 3)
# output_values: ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The instruction specifies "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins." The AI's implementation produces exactly this.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is deterministic per bin index, sharing the same position-bin coordinate system as neural data.

ii.
```python
pos_tr = pos_bins.astype(np.int64)  # Same for every trial
```

iii. Alignment is inherent in the shared position-bin indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['run_pos']`, a pre-computed array of shape `(n_trials, 60)` containing running speed already interpolated to position bins.

ii.
```python
run_pos = beh['run_pos']  # (n_trials, 60)
run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)
```

iii. `run_pos` is pre-computed in the behavior data, providing running speed at each position bin for each trial. This is the same variable used in the reference code for speed-related analyses.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed quartile edges are computed globally across all 89 sessions (using all position bins from all trials). The 25th, 50th, and 75th percentiles are computed, then `np.digitize` assigns each speed value to one of 4 bins (0-3).

ii.
```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
# Edges: [16.58, 28.72, 43.29] cm/s
# ...
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. The instruction specifies "Running speed discretized into 4 bins, each corresponding to 25% of the data." Global quartile computation ensures each bin contains ~25% of all data across all sessions. The edges [16.58, 28.72, 43.29] cm/s divide the data into 4 quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four quartile bins: Q1 (< 16.58 cm/s), Q2 (16.58-28.72 cm/s), Q3 (28.72-43.29 cm/s), Q4 (> 43.29 cm/s). The overall distribution is exactly 25% per bin by construction of global quartiles.

ii.
```python
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
# output_values: ['Q1', 'Q2', 'Q3', 'Q4']
```

iii. `np.digitize` with 3 edges returns values 0-3, mapping directly to the 4 quartile bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` is already in position-bin space (shape `(n_trials, 60)`), matching the neural data's position-interpolated coordinate system. The first 40 bins are used.

ii.
```python
run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)
```

iii. Both neural data and running speed are indexed by the same position bins, ensuring alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Session-level errors are caught with try/except, and the failing session is skipped with an error message. Behavior key mismatches are warned about and skipped. Missing behavior files are warned about and skipped. Lick events outside valid bin/trial ranges are silently ignored.

ii.
```python
# Session-level error handling:
try:
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
except Exception as e:
    print(f"  ERROR processing session: {e}")
    traceback.print_exc()
    continue

# Behavior key not found:
if beh_key not in beh_data:
    print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
    continue

# Lick bounds checking:
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1

# Frame count mismatch handling:
ft_move = beh['ft_move'][:n_frames]  # Truncate to neural frame count
```

iii. The AI's approach is defensive: skip problematic sessions rather than crash, validate array bounds for lick events, and truncate behavior arrays to match neural frame counts. All 89 sessions processed successfully.

## 12-a. What are the most time-consuming steps of the code?

i. Position interpolation is the most time-consuming step. It requires iterating over each neuron (up to ~78,000 per session) and performing 1D interpolation. The reference code noted this step took ~8 hours. Loading large neural data files is also time-consuming.

ii.
```python
# Per-neuron interpolation loop (most expensive):
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

iii. The AI initially encountered memory issues (>100GB) when running the conversion with scipy, and switched to numpy.interp for better performance. Memory management via `gc.collect()` and deletion of intermediate arrays was added.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron interpolation loop (iterating over up to ~78K neurons) could potentially be vectorized, though numpy.interp only supports 1D inputs. The `lick_to_position_bins` function iterates over individual lick events and could be vectorized using numpy indexing.

ii.
```python
# Neuron loop (could use scipy's interp1d on full matrix or custom vectorized approach):
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])

# Lick event loop (could use np.add.at or similar):
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    bin_idx = int(pos)
    if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
        lick_binary[tr, bin_idx] = 1
```

iii. The neuron loop is the same structure as the reference code's `spk_pos_interp()`, which also loops per neuron. The lick loop processes a relatively small number of events, so vectorization would provide minimal speedup.

## 12-c. What processing does the code repeat multiple times?

i. Date parsing via `compute_day_of_training()` is called multiple times for the same sessions (once when computing mouse_first_date, again when computing session_days). Behavior data is loaded and partially parsed in `collect_all_sessions` and then accessed again in `process_session`.

ii.
```python
# Date parsing repeated:
mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])  # First pass
    # ...
session_days = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])  # Second pass
```

iii. The repeated date parsing is trivial in cost. The behavior data stored in session dicts avoids re-loading behavior files.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code interpolates neural data to 60 position bins per corridor and then discards the last 20 (gray space). This means 33% of the interpolation work is wasted. The stimulus name collection iterates through `UniqWalls` for all sessions to build the global set, even though it could be done more efficiently.

ii.
```python
# Interpolate to 60 bins:
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving],
                                       corridor_len, n_trials, n_bins=N_POS_BINS)  # 60 bins
# Then discard last 20:
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]  # Keep only first 40
```

iii. The 60-bin interpolation followed by truncation to 40 matches the reference code's approach, which also interpolates to 60 bins and then uses only the texture area. Interpolating directly to 40 bins would require adjusting the corridor length parameter and could affect the interpolation quality at the boundary.
