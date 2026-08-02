# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, which contains database entries organized by experiment type (e.g., `sup_test1`, `unsup_train1_before_learning`, etc.). It iterates over all experiment types, loads corresponding behavior files (`Beh_{exp_type}.npy`), and builds a list of unique sessions. Neural data for each session is loaded from individual `.npy` files in the `spk/` directory, and retinotopy from `.npz` files in `retinotopy/`. Behavior data is accessed from the loaded `Beh_*.npy` dictionaries using keys constructed from mouse name, date, block, and optionally stimtype.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# ...
for exp_type in exp_order:
    db_list = exp_info[exp_type]
    beh_path = os.path.join(root, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_path, allow_pickle=True).item()
    for ndb in db_list:
        # ... build session info
# Neural data:
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
# Retinotopy:
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
```

iii. The AI followed the reference code's `load_spk()` and `load_exp_beh()` patterns, loading neural data with `np.concatenate` over `spks` planes and behavior from per-experiment-type files. This matches the reference code's approach.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in each database entry. All unique mouse names are collected and sorted alphabetically. Each session is assigned a `subject_idx` pointing into this sorted list.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions))
# ...
all_subject_idx.append(all_subjects.index(result['mname']))
```

iii. The AI correctly uses the `mname` field which uniquely identifies each mouse. The resulting 19 subjects match the paper's statement of "19 mice."

## 1-c. How are the data split into sessions?

i. Sessions are identified by the unique tuple `(mname, datexp, blk)`. When the same recording appears in multiple experiment types (e.g., both `sup_test1` and `sup_train1_after_learning`), only the first occurrence is kept, with a priority ordering: test > supervised > after-learning > alphabetical. This deduplication yielded 89 unique sessions.

ii.
```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
```

iii. The AI documented this in CONVERSION_NOTES.md. The 89 sessions match the paper's "89 recordings." The deduplication prioritizes test sessions which have more stimulus types, ensuring richer behavioral annotations. However, the deduplication key excludes `stimtype`, so recordings that appear with different stimtypes are collapsed to a single entry with one stimtype's behavior data.

## 1-d. How are the data split into trials?

i. The number of trials per session comes from `beh['ntrials']`. The position-interpolated neural data is reshaped into `(n_neurons, n_trials, n_bins)` where `n_trials` comes from this behavior variable. Each trial corresponds to one corridor traversal.

ii.
```python
n_trials = beh['ntrials']
# Position interpolation creates (n_neurons, n_trials, n_bins):
interp_spk = interp_spk.reshape(n_neurons, n_trials, n_bins)
```

iii. This matches the reference code, which uses the same `ntrials` field to determine trial count and reshapes interpolated data identically.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality filtering. All trials from each session are included. Sessions with fewer than 2 trials are skipped. Within each trial, only running frames (`ft_move > 0`) are used for position interpolation, effectively excluding stationary periods.

ii.
```python
if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
# Running frames only:
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
```

iii. The reference paper states "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The AI implements this by filtering for VR-moving frames. No explicit per-trial quality filtering is applied, consistent with the reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spks` (deconvolved fluorescence traces from Suite2p) stored in per-recording `.npy` files. The `spks` field contains spike-deconvolved calcium traces for all detected ROIs across multiple imaging planes.

ii.
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The paper states "All our analyses were based on deconvolved fluorescence traces" processed by Suite2p. The AI correctly uses the `spks` field, matching the reference code's `load_spk()` function.

## 2-b. How is the `neural` data processed?

i. Neural data is position-interpolated from frame-by-frame measurements to evenly-spaced position bins. First, only running frames (VR moving) are selected. Then, linear interpolation maps from cumulative position (`ft_PosCum`) to 60 bins per corridor. Finally, only the first 40 bins (texture area, 4m corridor) are retained.

ii.
```python
valid_spk = spk[valid_neurons][:, vr_moving]
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving], corridor_len, n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]  # Keep only texture area (first 40 bins)
```

iii. This matches the reference code's `spk_pos_interp()` and `get_interpPos_spk()` functions, which perform the same position interpolation. The reference code also uses only texture area bins (first 40 of 60). The AI uses `np.interp` instead of `scipy.interpolate.interp1d` with `fill_value='extrapolate'`, which clamps at boundaries rather than extrapolating - a minor difference unlikely to affect results.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only visual cortex neurons are included, filtering by retinotopy area assignment (`iarea`). Neurons with `iarea == -1` (not in visual cortex) or `iarea == 7` (undefined region) are excluded. No further quality filters (e.g., signal-to-noise, activity thresholds) are applied.

ii.
```python
def get_brain_region_idx(iarea):
    valid_mask = (iarea != -1) & (iarea != 7)
    # ...
valid_neurons = valid_mask
valid_spk = spk[valid_neurons][:, vr_moving]
```

iii. This matches the reference code's neuron filtering in `Get_density_map()`: `idx_neu = (arid!=-1) & (arid != 7)`. The resulting neuron counts (17,363-78,815) are lower than the paper's total neuron range (20,547-89,577), consistent with filtering out non-visual-cortex neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) through position interpolation. Since each corridor starts at position 0 and the data is binned by position from the start of each corridor, position bin 0 corresponds to corridor entry. The position-based representation naturally aligns all trials to corridor entry.

ii.
```python
linPos = np.arange(0, n_trials, 1.0 / n_bins)  # evenly spaced positions starting at corridor start
norm_pos = accum_pos / corridor_len
# Interpolation aligns to corridor start by construction
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." Since the VR moves at a constant speed (60 cm/s) when the mouse runs, position bins map directly to time from corridor entry. The AI correctly documents `temporal_alignment_event` as "Trial start (corridor entry)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is approximately 166.67 ms, derived from 1 dm (position bin size) / 6 dm/s (VR speed) = 1/6 second. No temporal rebinning is applied; the data is in position space (40 bins per corridor) which maps to time through the constant VR speed.

ii.
```python
VR_SPEED = 6.0        # dm/s (60 cm/s)
BIN_SIZE_SEC = 1.0 / VR_SPEED  # seconds per position bin = 1/6 s
BIN_SIZE_MS = BIN_SIZE_SEC * 1000  # ~166.67 ms
```

iii. The paper states "virtual corridors always moved at a constant speed (60 cm/s) as long as mice kept running faster than the threshold." With 40 bins across 4m, each bin represents 0.1m traversed at 0.6 m/s = 1/6 s = 166.67 ms. This is a position-based binning, not traditional temporal rebinning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Time to sound cue is derived from `beh['SoundPos']`, which contains the position (in dm) where the sound cue was delivered in each trial.

ii.
```python
sound_pos = beh['SoundPos']
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The AI uses `SoundPos` directly. The reference code uses `SoundDelPos` (with `np.mod`) in some functions and `SoundPos` in others. `SoundPos` represents the position bin where the sound cue occurred. The paper states the sound cue was randomly placed between 0.5m and 3.5m.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the time to sound cue at each position bin is computed as `(position_bin_index - sound_cue_position) * bin_size_in_seconds`. This yields negative values before the cue position and positive values after. The result ranges approximately from -6.5 to 6.3 seconds.

ii.
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. This provides a continuous, time-varying signal indicating temporal distance to the sound cue, which is relevant because the sound cue indicates the reward zone and mice show anticipatory licking relative to it.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to sound cue shares the same position-bin indexing as the neural data. Position bin `i` in the neural data corresponds to the same position bin `i` in the time-to-cue vector. Both have 40 bins.

ii.
```python
# Neural: interp_spk[:, tr, :] -> shape (n_neurons, 40)
# Input: time_to_cue -> shape (40,)
# Both indexed by position bin i in range(N_TEXTURE_BINS)
```

iii. Alignment is implicit through shared position indexing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day of training is derived from the `datexp` field (date string in format `YYYY_MM_DD`) of each session's database entry.

ii.
```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')
# First recording date per mouse:
mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d
```

iii. The `datexp` field is the recording date, parsed to compute relative training days.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is determined. Day of training for each session is computed as the number of days since that mouse's first recording. The value is constant within a trial (broadcast to all 40 time bins).

ii.
```python
days = (d - mouse_first_date[s['mname']]).days
session_days[key] = float(days)
# Later, broadcast to all time bins:
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. The instructions specify day of training as "continuous, time-varying." The AI makes it constant within each trial (same value for all 40 bins), which is correct since the day doesn't change within a trial. The "time-varying" aspect means it has a value at each time point.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is NOT included as a decoder input. The instructions specify 4 decoder inputs: time to sound cue, day of training, time since trial start, and reward availability. Environment type (supervised/unsupervised/grating) is not among them.

ii. N/A - no code for this variable.

iii. The AI correctly followed the decoder input specification, which does not include environment type. The experiment type is stored in session metadata but not as a decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. N/A - environment type is not a decoder input.

ii. N/A.

iii. See 4-a above.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Time since trial start is derived from the position bin index. Since the VR moves at constant speed and the data is position-binned starting from corridor entry, each position bin corresponds to a fixed time offset from trial start.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. No raw data variable is needed - it is computed from the position bin index and the known VR speed.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each position bin index `i` is multiplied by the time per bin (1/6 seconds), yielding a linear ramp from 0 to ~6.5 seconds across the 40 bins.

ii.
```python
BIN_SIZE_SEC = 1.0 / VR_SPEED  # = 1/6 s
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)], dtype=np.float32)
# Result: [0.0, 0.167, 0.333, ..., 6.5]
```

iii. This is a simple linear mapping from position to time, valid because the VR speed is constant.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same position-bin indexing as neural data - bin `i` in both time_since_start and neural data correspond to the same corridor position.

ii. Same indexing scheme as all other variables (40 position bins).

iii. Alignment is implicit.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived from `beh['isRew']`, a per-trial boolean/binary array indicating whether each trial was in a rewarded corridor.

ii.
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The `isRew` field matches what is used in the reference code for identifying rewarded versus unrewarded trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial binary value (0 or 1) is broadcast across all 40 time bins, creating a constant signal within each trial. For unsupervised/naive sessions, all trials have reward_availability = 0.

ii.
```python
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The instructions specify "1 if in rewarded corridor, 0 if not, discrete, per-trial." The AI correctly implements this as a per-trial value broadcast to all time bins.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Visual stimulus category is derived from `beh['WallName']`, a per-trial string array containing the name of the wall texture shown in each trial (e.g., 'circle1', 'leaf1', 'rock1'). The unique stimulus names are collected from `beh['UniqWalls']` across all sessions.

ii.
```python
wall_names = beh['WallName']
all_stim_set = set()
for s in sessions:
    for wn in s['beh']['UniqWalls']:
        all_stim_set.add(str(wn))
all_stim_names = sorted(all_stim_set)
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. The AI collects all unique stimulus names across all sessions and maps each to an integer index. 15 unique stimuli were found.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` is mapped to an integer index into the sorted global stimulus list. This per-trial integer is broadcast across all 40 time bins.

ii.
```python
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. The instructions specify visual stimulus as "per-trial." Broadcasting across time bins makes it time-varying in format while constant per trial, consistent with the requirement.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `beh['LickPos']` (position of each lick event in dm) and `beh['LickTrind']` (trial index of each lick event).

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
```

iii. These are the same variables used in the reference code's `get_lick_raster()` function.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick events are converted to binary per-position-bin per-trial. For each lick event, the position (in dm) is floored to get a bin index (0-39). If any lick falls in a given position bin for a given trial, that bin is set to 1; otherwise 0.

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

iii. The instructions specify licking as "binary, time-varying. 0 = not licking, 1 = licking." The AI implements this correctly. For unsupervised/naive sessions, no lick events exist (mice were not water restricted), so licking is all zeros.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick positions (in dm) are directly mapped to position bin indices (0-39), the same position bins as the neural data. A lick at position 5.3 dm goes into bin 5.

ii.
```python
bin_idx = int(pos)  # floor to position bin index
```

iii. Alignment is through shared position-bin indexing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position in corridor is derived directly from the position bin index (0-39), not from any raw data variable. Since the data is already position-binned, position is deterministic.

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. Position is a structural variable derived from the bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 position bins are grouped into 4 equal-length 1-meter bins: bins 0-9 -> category 0, bins 10-19 -> category 1, bins 20-29 -> category 2, bins 30-39 -> category 3.

ii.
```python
pos_bins[i] = min(i // 10, 3)  # 0-9 -> 0, 10-19 -> 1, 20-29 -> 2, 30-39 -> 3
```

iii. The instructions specify "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins." Each 1m = 10 dm = 10 position bins, so grouping by 10 is correct.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Integer division by 10 maps each position bin to one of 4 categories. The `min(..., 3)` clips to ensure maximum category is 3, though this is redundant since bin indices only go to 39.

ii.
```python
pos_bins[i] = min(i // 10, 3)
```

iii. This produces exactly 25% of data in each category (by construction), confirmed in the verification output.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is deterministic per position bin, sharing the same indexing as neural data. The same value is used for every trial at the same bin index.

ii.
```python
pos_tr = pos_bins.astype(np.int64)  # same for all trials
```

iii. Alignment is inherent in the position-bin structure.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `beh['run_pos']`, a pre-computed position-interpolated running speed array of shape `(n_trials, 60)` stored in the behavior data.

ii.
```python
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)
```

iii. `run_pos` contains the running speed at each position bin, already position-interpolated in the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The `run_pos` data is trimmed to the first 40 bins (texture area). Running speed is then discretized into 4 quartile bins using globally-computed quartile edges (25th, 50th, 75th percentiles) across all sessions.

ii.
```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
# Per session:
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. The instructions specify "Running speed discretized into 4 bins, each corresponding to 25% of the data." The AI computes global quartile edges [16.58, 28.72, 43.29] cm/s and uses `np.digitize` to assign bins 0-3.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` with 3 quantile edges maps speeds to bins 0-3: below Q25, Q25-Q50, Q50-Q75, above Q75. The result is clipped to [0, 3].

ii.
```python
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. The overall distribution is approximately 25% per bin, as confirmed in the verification output. Per-session distributions vary due to inter-session speed differences.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` is already position-interpolated to the same position bins as the neural data. The first 40 bins are used, matching the neural data's texture area bins.

ii.
```python
run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)
```

iii. Alignment is through shared position-bin indexing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data at the session level: if a behavior file is not found, a warning is printed and the session is skipped. If a behavior key is not found, the session is skipped. If processing a session throws an exception, it is caught and the session is skipped with an error message. For lick data, out-of-range lick positions or trial indices are silently ignored. No explicit handling of NaN values or other per-trial data issues is implemented.

ii.
```python
if not os.path.exists(beh_path):
    print(f"  Warning: behavior file not found for {exp_type}")
    continue
if beh_key not in beh_data:
    print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
    continue
try:
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
except Exception as e:
    print(f"  ERROR processing session: {e}")
    continue
# Lick bounds check:
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

iii. The AI's error handling is defensive but coarse-grained. All 89 sessions processed successfully with no errors or warnings, suggesting data completeness.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is position interpolation of neural data (`position_interpolate_spk`), which loops over every neuron individually to call `np.interp`. With 17,000-79,000 neurons per session and 89 sessions, this is computationally expensive. Loading large neural data files from disk is also significant.

ii.
```python
for s in range(n_neurons):  # Loop over each neuron
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

iii. The reference code's `get_interpPos_spk` also loops over neurons but processes them in batches of 10,000 (`step_size = 10000`), which is slightly more efficient for memory but similarly slow.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized:
1. The neuron loop in `position_interpolate_spk` - while `np.interp` is inherently 1D, the batch processing could use scipy's interpolation on multiple neurons at once.
2. The lick event loop in `lick_to_position_bins` could be replaced with numpy advanced indexing.

ii.
```python
# Neuron loop (position_interpolate_spk):
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])

# Lick loop (lick_to_position_bins):
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    bin_idx = int(pos)
    if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
        lick_binary[tr, bin_idx] = 1
```

iii. The lick loop could be vectorized as: `valid = (bin_idx >= 0) & (bin_idx < n_bins) & (tr >= 0) & (tr < n_trials); lick_binary[tr[valid], bin_idx[valid]] = 1`.

## 12-c. What processing does the code repeat multiple times?

i. The behavior data is effectively loaded twice for each session: once in `collect_all_sessions` (where it's stored in the session dict) and the behavior variables are accessed again during `process_session`. The running speed data is iterated twice: once for computing global quartile edges and once during per-session processing.

ii.
```python
# First pass: collect all speeds for quartile computation
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
# Second pass: per-session processing uses beh['run_pos'] again
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]
```

iii. The two-pass approach for speed quantiles is necessary (need global statistics before per-session binning), but the behavior data storage could be more memory-efficient.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. Position interpolation to 60 bins when only the first 40 are used - the last 20 (grey space) bins are computed then discarded.
2. The `session_date` datetime object is computed but not directly stored in the output (only `day_of_training` is used).
3. Running speed computation on the full `run_pos` array before trimming to 40 bins.

ii.
```python
# Interpolate to 60 bins, then discard 20:
interp_spk = position_interpolate_spk(..., n_bins=N_POS_BINS)  # 60 bins
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]  # Keep only 40
```

iii. Interpolating to 60 bins matches the reference code convention and ensures consistency with the position normalization. While slightly wasteful, modifying the interpolation to only produce 40 bins would require changing the normalization scheme, so this is a reasonable tradeoff.
