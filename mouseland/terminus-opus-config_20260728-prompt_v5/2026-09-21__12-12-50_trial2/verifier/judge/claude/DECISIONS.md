# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories under `/app/data`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for brain area assignments. It first loads `Imaging_Exp_info.npy` to enumerate all sessions, then loads behavioral files (`Beh_<exp_type>.npy`), spike files (`<session_id>_neural_data.npy`), and retinotopy files (`<mname>_<datexp>_trans.npz`). A global cache (`_beh_cache`) is used to avoid reloading behavioral files.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in spk_data['spks']], 0)

dtrans = np.load(ret_path, allow_pickle=True)
iarea = dtrans['iarea']
```

iii. The AI loads the same source files as the reference code. The caching strategy avoids reloading large behavioral files.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) from the experiment info is used as the subject identifier. A `subject_map` dictionary tracks unique subjects and assigns sequential indices.

ii.
```python
if mname not in subject_map:
    subject_map[mname] = len(unique_subjects)
    unique_subjects.append(mname)
```

iii. Subjects are split by mouse name from the experiment info, same as the reference approach.

## 1-c. How are the data split into sessions?

i. A session is defined by the unique key `{mname}_{datexp}_{blk}`. The `get_unique_sessions` function deduplicates sessions that appear under multiple experiment types.

ii.
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in sessions:
    sessions[key] = { ... }
sessions[key]['exp_types'].append(exp_type)
```

iii. Same deduplication logic as the reference -- sessions appearing under multiple experiment types are kept only once.

## 1-d. How are the data split into trials?

i. The AI does NOT split trials by frame indices like the reference. Instead, it uses position-interpolated data where each trial is represented as 60 fixed position bins (covering the 4m corridor + 2m grey space). The number of trials per session comes from `beh['ntrials']`, and all trials are included (no filtering).

ii.
```python
n_trials = beh['ntrials']
# Position interpolation creates fixed 60 bins per trial
interp_spk = position_interpolate_spk(
    spk[:, vr_moving], ft_pos_cum[vr_moving],
    corridor_len, n_trials, N_POS_BINS
)
neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
```

iii. The AI followed the reference paper's analysis code which uses position interpolation. The CONVERSION_NOTES state "Position-based binning (60 bins): Matches reference code exactly."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter any trials. All `ntrials` from each session are included.

ii. No filtering code exists in the AI's script. All trials from `range(n_trials)` are processed.

iii. The CONVERSION_NOTES do not discuss trial filtering. The reference solution filters trials longer than the 99th percentile (which removes 382 trials where mice stopped for extended periods) and trials with zero frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike files, which contains deconvolved fluorescence traces per imaging plane, concatenated across planes.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in spk_data['spks']], 0)
```

iii. Same source as the reference.

## 2-b. How is the `neural` data processed?

i. The AI applies position interpolation: it filters to only frames where the VR is moving (`ft_move > 0`), then interpolates the neural traces from their temporal sampling to 60 evenly-spaced position bins using `np.interp`. This transforms the data from a temporal representation to a spatial representation.

ii.
```python
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0

interp_spk = position_interpolate_spk(
    spk[:, vr_moving], ft_pos_cum[vr_moving],
    corridor_len, n_trials, N_POS_BINS
)

# Inside position_interpolate_spk:
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
src_pos = accum_pos / corridor_len
for s in range(batch.shape[0]):
    interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
```

iii. The CONVERSION_NOTES state this matches the reference code's `spk_pos_interp` function. However, the instructions specify temporal alignment, not spatial interpolation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered to visual cortex only, keeping those with `iarea` in V1 (8), mHV (0,1,2,9), lHV (5,6), or aHV (3,4). Neurons with iarea=-1 or 7 are excluded.

ii.
```python
AREA_MAPPING = {
    'V1': [8], 'mHV': [0, 1, 2, 9], 'lHV': [5, 6], 'aHV': [3, 4],
}

for r_idx, (region, areas) in enumerate(AREA_MAPPING.items()):
    for area_val in areas:
        mask = iarea == area_val
        region_idx[mask] = r_idx
valid_mask = region_idx >= 0
```

iii. Matches the reference code's `neu_area_ID` function exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI uses position-based binning rather than temporal alignment. Each trial is represented as 60 position bins (0-6m), not as variable-length temporal segments aligned to corridor entry. The position interpolation uses cumulative position to map neural activity to spatial locations.

ii.
```python
interp_spk = position_interpolate_spk(
    spk[:, vr_moving], ft_pos_cum[vr_moving],
    corridor_len, n_trials, N_POS_BINS
)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI instead uses spatial alignment to position bins, following the reference paper's analysis pipeline rather than the decoder task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses ~166.67 ms bins, derived from position bins: each bin is 0.1m, divided by 0.6 m/s VR speed. This is a spatial resolution converted to time assuming constant speed, not a true temporal resolution.

ii.
```python
BIN_SIZE_M = TOTAL_LENGTH_M / N_POS_BINS  # 0.1m per bin
TIME_PER_BIN = BIN_SIZE_M / VR_SPEED  # ~0.1667s per bin
TIME_BIN_MS = TIME_PER_BIN * 1000  # ~166.67 ms
```

iii. The reference solution uses the native imaging frame rate of 3.17 Hz (~315 ms per bin) with no rebinning. The AI's approach assumes constant VR speed to derive a time bin size from spatial bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos`, the position of the sound cue in each trial, expressed in position units (0-60).

ii.
```python
sound_pos = beh['SoundPos']  # position units
time_to_cue = make_time_to_sound_cue(sound_pos, N_POS_BINS)
```

iii. The reference uses `SoundFr` (the frame at which the sound cue was played) and `ft` (frame timestamps). The AI uses the position-domain variable instead.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes the distance in position units from the current bin center to the sound cue position, then converts to time by dividing by VR speed (0.6 m/s). This assumes constant traversal speed.

ii.
```python
def make_time_to_sound_cue(sound_pos, n_bins=60):
    positions = np.arange(n_bins) + 0.5  # center of each bin
    dist = sound_pos[:, np.newaxis] - positions[np.newaxis, :]
    time_to_cue = (dist * BIN_SIZE_M / VR_SPEED).astype(np.float32)
    return time_to_cue
```

iii. The reference computes time to cue from actual frame timestamps, which captures the real temporal relationship. The AI's approach is an approximation based on assumed constant speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both are indexed by the same 60 position bins, so alignment is inherent in the position-interpolation framework.

ii.
```python
inp = np.stack([
    time_to_cue[t],          # (n_bins,)
    ...
], axis=0)
```

iii. Alignment is consistent within the AI's position-based framework, but both neural and input data are spatially rather than temporally aligned.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `sess_num` (session number) stored in the experiment info metadata, accessed via `sess['sess_num']`.

ii.
```python
sessions[key] = {
    ...
    'sess_num': s.get('sess#', 0),
    ...
}
day_of_training = np.full(n_trials, sess_meta['sess_num'], dtype=np.float32)
```

iii. The reference counts the number of unique recording dates per mouse (0-indexed). The AI uses `sess#` from the experiment info, which may represent something different.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The value is taken directly from `sess_num` in the experiment info and broadcast to all bins. No counting or ordering is performed.

ii.
```python
day_of_training = np.full(n_trials, sess_meta['sess_num'], dtype=np.float32)
np.full(N_POS_BINS, day_of_training[t], dtype=np.float32)
```

iii. The reference counts sorted sessions per mouse to derive the training day. The AI uses a metadata field directly, which may not correspond to the same concept.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived deterministically from the position bin index and the assumed time per bin (BIN_SIZE_M / VR_SPEED).

ii.
```python
def make_time_since_trial_start(n_trials, n_bins=60):
    time_per_bin = BIN_SIZE_M / VR_SPEED
    times = (np.arange(n_bins) * time_per_bin).astype(np.float32)
    return np.tile(times, (n_trials, 1))
```

iii. No raw data variable is used. The reference derives this from `StartFr` and `ft` (actual frame timestamps).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes a deterministic time ramp: `bin_index * (0.1m / 0.6 m/s)`, producing identical time values for every trial. This assumes constant speed and that every trial takes the same amount of time.

ii.
```python
times = (np.arange(n_bins) * time_per_bin).astype(np.float32)
return np.tile(times, (n_trials, 1))
```

iii. The reference computes actual elapsed time from frame timestamps, which varies per trial based on real animal behavior.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Both use the same 60 position bins, so alignment is inherent in the framework.

ii.
```python
time_since_start[t],     # (n_bins,)
```

iii. Consistent within the position-based framework.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which indicates whether each trial was in a rewarded corridor.

ii.
```python
reward_avail = beh['isRew'].astype(np.float32)
np.full(N_POS_BINS, reward_avail[t], dtype=np.float32)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float and broadcast to all position bins. No further processing.

ii.
```python
reward_avail = beh['isRew'].astype(np.float32)
```

iii. Same approach as the reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (the texture name for each trial) via `UniqWalls` (unique wall names in the session).

ii.
```python
wall_name = beh['WallName']
uniq_walls = beh['UniqWalls']
stim_categories, stim_names = get_stimulus_category(wall_name, uniq_walls)
```

iii. Same source variable as the reference, but different categorization.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses each individual wall name as its own category, producing 15 distinct categories (e.g., circle1, circle2, leaf1, leaf1_swap1, etc.) across the dataset. Category indices are mapped to a global list of all unique wall names.

ii.
```python
def get_stimulus_category(wall_name, uniq_walls):
    category_names = list(uniq_walls)
    categories = np.array([category_names.index(wn) for wn in wall_name], dtype=np.int64)
    return categories, category_names

all_stim_names = collect_all_stim_names(exp_info)  # 15 unique names
output_values = [
    [str(s) for s in all_stim_names],  # 15 stimulus categories
    ...
]
```

iii. The reference groups the 15 wall names into 4 base texture categories (circle, leaf, rock, wood), matching the instruction "Visual stimulus category. e.g. circle, leaf, etc." The AI uses all 15 individual names.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickPos` (position of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
lick_raster = make_lick_raster(
    beh['LickPos'], beh['LickTrind'], n_trials, N_POS_BINS, corridor_len
)
```

iii. The reference uses `LickFr` (frame number of each lick). The AI uses position-domain lick variables consistent with its spatial binning approach.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick events are binned into position bins using `np.searchsorted`. Each position bin is marked 1 if any lick falls within it, 0 otherwise.

ii.
```python
def make_lick_raster(lick_pos, lick_trind, n_trials, n_bins=60, corridor_len=60.0):
    lick_raster = np.zeros((n_trials, n_bins), dtype=np.float32)
    bin_edges = np.linspace(0, corridor_len, n_bins + 1)
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        bin_idx = np.searchsorted(bin_edges, pos, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        lick_raster[tr, bin_idx] = 1.0
    return lick_raster
```

iii. The reference creates a per-frame binary lick indicator using `LickFr`. The AI creates a position-binned version.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Both use the same 60 position bins.

ii.
```python
lick_raster[t].astype(np.int64)  # (n_bins,)
```

iii. Consistent within the position-based framework.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is NOT derived from a raw data variable. It is deterministically generated from the position bin index.

ii.
```python
def make_position_output(n_trials, n_bins=60):
    pos_output = np.zeros((n_trials, n_bins), dtype=np.float32)
    for b in range(n_bins):
        if b < 10: pos_output[:, b] = 0
        elif b < 20: pos_output[:, b] = 1
        elif b < 30: pos_output[:, b] = 2
        else: pos_output[:, b] = 3
    return pos_output
```

iii. The reference uses `ft_Pos` (actual position at each frame). The AI's approach makes position output entirely deterministic -- the same pattern for every trial -- which means the decoder cannot learn anything meaningful from it relative to neural activity.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Bin indices 0-9 map to category 0 (0-1m), 10-19 to category 1 (1-2m), 20-29 to category 2 (2-3m), and 30-59 to category 3 (3-4m and grey space). This is identical for every trial.

ii.
```python
for b in range(n_bins):
    if b < 10: pos_output[:, b] = 0
    elif b < 20: pos_output[:, b] = 1
    elif b < 30: pos_output[:, b] = 2
    else: pos_output[:, b] = 3
```

iii. The reference divides `ft_Pos` by 10 (decimeters to meters) and clips to 0-3, producing actual per-frame position that varies with the animal's real location.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The 4 categories correspond to ranges of position bin indices: bins 0-9 (0-1m), 10-19 (1-2m), 20-29 (2-3m), 30-59 (3-4m + grey). Grey space bins (40-59) are all assigned to category 3.

ii. Same code as 9-b above.

iii. The reference clips `ft_Pos // 10` to [0, 3]. The AI's assignment of grey space (bins 40-59) to category 3 is a consequence of including grey space in the trial.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Both use the same 60 position bins.

ii.
```python
pos_output[t].astype(np.int64)  # (n_bins,)
```

iii. Consistent within the position-based framework but the position output is deterministic, identical for every trial.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `run_pos`, which contains position-interpolated running speed (shape: n_trials x 60).

ii.
```python
run_pos = beh['run_pos']  # (n_trials, 60)
speed_output = make_speed_output(run_pos, speed_bin_edges)
```

iii. The reference uses `ft_RunSpeed` (running speed at each imaging frame). The AI uses the pre-interpolated position-domain speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartiles are computed across all sessions. Speeds <= 0 and NaN are excluded before computing percentile edges. Then `np.digitize` maps each speed value to one of 4 bins.

ii.
```python
def compute_global_speed_quartiles(exp_info):
    ...
    valid = np.isfinite(all_speeds) & (all_speeds > 0)
    all_speeds = all_speeds[valid]
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    bin_edges = np.array([0, quartiles[0], quartiles[1], quartiles[2], np.inf])
    return bin_edges

def make_speed_output(run_pos, speed_bin_edges):
    speed_output = np.digitize(run_pos, speed_bin_edges[1:-1]).astype(np.float32)
    return speed_output
```

iii. The reference computes rank-based quartiles per session (ensuring exactly 25% per bin even with tied values). The AI computes global percentile-based edges excluding non-positive speeds, which may not produce equal-sized bins.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by global percentile edges: [0, Q25, Q50, Q75, inf]. Values are assigned to bins using `np.digitize`.

ii. Same code as 10-b.

iii. The reference uses rank-based quartiles per session to handle tied zero-speed values. The AI uses global percentile edges, which may produce unequal bins when many frames share the same speed value.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Both use the same 60 position bins. The `run_pos` variable is already in position-bin space.

ii.
```python
speed_output[t].astype(np.int64)  # (n_bins,)
```

iii. Consistent within the position-based framework.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavioral data to match the number of neural frames (`n_frames = spk.shape[1]`). The VR movement filter (`ft_move > 0`) excludes stationary frames from position interpolation. No explicit handling of missing lick data or anomalous trials.

ii.
```python
n_neurons_total, n_frames = spk.shape
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
```

iii. The reference also truncates behavior to neural frame count and additionally handles licks beyond the last imaged frame and empty/too-long trials.

## 12-a. What are the most time-consuming steps of the code?

i. Loading spike files (large .npy files) and the position interpolation per neuron are the most time-consuming steps.

ii.
```python
spk = load_spk(mname, datexp, blk)
# and
interp_spk = position_interpolate_spk(...)
```

iii. The CONVERSION_NOTES estimate ~50s for 2 sessions, ~40 minutes for all 89.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron interpolation loop in `position_interpolate_spk` iterates over each neuron individually. The lick raster construction loops over individual lick events.

ii.
```python
for s in range(batch.shape[0]):
    interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])

for i in range(len(lick_pos)):
    ...
```

iii. These loops could potentially be vectorized, though `np.interp` only operates on 1D arrays.

## 12-c. What processing does the code repeat multiple times?

i. The AI loads all behavioral files twice: once in `compute_global_speed_quartiles` to compute speed edges, once in `collect_all_stim_names` to collect stimulus names, and then again during session processing. The `get_unique_sessions(exp_info)` is called multiple times (once in main, once per session in `load_beh_for_session`).

ii.
```python
speed_bin_edges = compute_global_speed_quartiles(exp_info)  # loads all beh files
all_stim_names = collect_all_stim_names(exp_info)  # loads all beh files again
```

iii. This repeated loading is inefficient and could be combined into a single pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes grey space bins (bins 40-59) in every trial, producing 60 bins instead of 40 corridor bins. The grey space contributes 20 bins per trial where position is deterministically category 3 and neural activity may not be meaningful for the discrimination task. Additionally, the position output is entirely deterministic (same for every trial), making it uninformative for a decoder.

ii.
```python
N_POS_BINS = 60  # position bins per trial (corridor + grey)
# Grey space bins always assigned to category 3
pos_output[:, b] = 3  # for b >= 30
```

iii. The reference only includes frames inside the corridor (using `ft_CorrSpc` filter), excluding grey space entirely.
