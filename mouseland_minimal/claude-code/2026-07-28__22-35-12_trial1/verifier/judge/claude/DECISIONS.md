# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, then iterates over experiment types. For each experiment type it loads the corresponding `Beh_<exp_type>.npy` behavior file and collects all unique recordings keyed by `(mname, datexp, blk)`. It also loads neural data from `spk/<session_id>_neural_data.npy` and retinotopy from `retinotopy/<mname>_<datexp>_trans.npz`. Behavior data for all sessions from a given experiment type is loaded at once and stored in session dicts.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

# Per experiment type:
beh_data = np.load(beh_path, allow_pickle=True).item()

# Per session:
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)

ret_fn = f"{mname}_{datexp}_trans.npz"
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
iarea = ret_data['iarea']
```

iii. The AI reasoned: "For each unique recording, I should use the behavior data that has the most stimuli/trials" and sorted experiment types to prefer test sessions, supervised sessions, and after-learning sessions. It loads behavior data for all sessions within each experiment type at once during the `collect_all_sessions` phase, then processes neural data per-session.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) is extracted from each entry in the experiment info. A sorted set of unique mouse names forms the `subjects` list, and `subject_idx` maps each session to its index in that list.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions))
all_subject_idx.append(all_subjects.index(result['mname']))
```

iii. The AI used the `mname` field directly from the experiment info entries without further derivation.

## 1-c. How are the data split into sessions?

i. A session is defined by the unique tuple `(mname, datexp, blk)`. When the same recording appears in multiple experiment types, only the first occurrence is kept (with preference given to test/supervised/after-learning types due to the sorting of experiment types).

ii.
```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
```

iii. The AI stated: "each unique (mname, datexp, blk) recording = one session" and recognized that recordings can appear under multiple experiment types with identical underlying data.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the behavior data's `ntrials` count. Each trial gets 40 position bins (the texture area of the corridor), derived from position interpolation rather than from raw temporal frames. All trials are included without temporal frame-based splitting.

ii.
```python
n_trials = beh['ntrials']
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The AI decided to use the paper's position-interpolation approach, which inherently defines trials by their corridor traversals. Each trial gets exactly 40 spatial bins (the texture area).

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: sessions with fewer than 2 trials are skipped. No individual trial filtering is applied (no outlier length removal, no quality checks on individual trials).

ii.
```python
if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
```

iii. The AI did not discuss trial filtering based on length or quality. The position-interpolation approach inherently handles variable-length trials by binning them into a fixed number of spatial bins.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains lists of per-plane neuron-by-frame arrays. These are concatenated across planes. The visual area of each neuron comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)

ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
iarea = ret_data['iarea']
```

iii. The AI identified these as the correct source variables from examining the reference code and data files.

## 2-b. How is the `neural` data processed?

i. The AI applies position interpolation to the neural data, matching the paper's processing pipeline. Only running frames (`ft_move > 0`) are used. The cumulative position (`ft_PosCum`) is used to interpolate neural activity into 60 evenly spaced position bins per corridor, then only the first 40 bins (texture area) are kept. Data is stored as float32.

ii.
```python
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
ft_pos_cum = beh['ft_PosCum'][:n_frames]

valid_spk = spk[valid_neurons][:, vr_moving]
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving], corridor_len, n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

```python
def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
    norm_pos = accum_pos / corridor_len
    for s in range(n_neurons):
        interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
    return interp_spk.reshape(n_neurons, n_trials, n_bins)
```

iii. The AI reasoned: "I'll use position interpolation matching the paper, with the full 60 position bins but only keeping the 40 texture-area bins for the decoder." This follows the paper's analysis pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area: only neurons where `iarea != -1` and `iarea != 7` are kept. This corresponds to V1 (iarea=8), mHV (iarea in {0,1,2,9}), lHV (iarea in {5,6}), aHV (iarea in {3,4}).

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

iii. The AI followed the paper's brain region definitions and excluded non-visual-cortex neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned by position rather than by time. Position interpolation maps neural activity to 40 equally-spaced spatial bins along the corridor (0-4m), so each trial starts at corridor entry (position 0) and ends at position 4m. All trials have exactly 40 time bins.

ii.
```python
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving], corridor_len, n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
# Each trial: neural_trial = interp_spk[:, tr, :].astype(np.float32)  # shape (n_neurons, 40)
```

iii. The AI decided to use position interpolation to align trials, which is the paper's standard processing approach. This means every trial has a fixed number of bins (40), regardless of how long the animal took to traverse the corridor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses spatial bins rather than temporal bins. The time bin size is computed as `BIN_SIZE_MS = (1/6) * 1000 = 166.67 ms`, based on the VR speed of 60 cm/s and 1 dm position bins. This is not temporal rebinning in the usual sense; it is a position-to-time conversion assuming constant speed.

ii.
```python
VR_SPEED = 6.0        # dm/s (60 cm/s)
BIN_SIZE_SEC = 1.0 / VR_SPEED  # seconds per position bin = 1/6 s
BIN_SIZE_MS = BIN_SIZE_SEC * 1000  # 166.67 ms
```

iii. The AI computed the time bin size from the VR treadmill speed, treating each 1-dm position bin as a fixed time interval.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos`, the position where the sound cue was delivered in each trial, and the position bin index within the corridor.

ii.
```python
sound_pos = beh['SoundPos']
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The AI used `SoundPos` (position-based) rather than `SoundFr` (frame-based) because the data was position-interpolated.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The position difference between each bin and the sound cue position is multiplied by the time per position bin (`BIN_SIZE_SEC = 1/6 s`). The result is `(bin_index - sound_pos) * BIN_SIZE_SEC`, so it is negative before the sound cue position and positive after.

ii.
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The AI converted from spatial to temporal domain using constant VR speed. Note the sign convention is opposite to the reference (positive after cue, negative before).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed per position bin, matching the position-interpolated neural data. Each of the 40 bins has a corresponding time-to-cue value.

ii.
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail], axis=0)
```

iii. Alignment is implicit through shared position bins.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date string in `datexp` for each session, parsed into a datetime object.

ii.
```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')

mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d
```

iii. The AI used the date strings from the experiment info entries.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the number of calendar days between the session's date and the mouse's first recording date. This is different from counting recording sessions. The value is broadcast to all bins of all trials in that session.

ii.
```python
session_days = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    days = (d - mouse_first_date[s['mname']]).days
    key = (s['mname'], s['datexp'], s['blk'])
    session_days[key] = float(days)

# Applied later:
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. The AI chose calendar days from first recording rather than counting the number of recording sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the position bin index within each trial and the constant time per bin (`BIN_SIZE_SEC`).

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The AI derived time from position assuming constant VR speed.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each position bin index is multiplied by the time per bin (1/6 s), producing a linearly increasing time series from 0 to ~6.5 seconds. This assumes the mouse runs at constant speed (60 cm/s).

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The AI used a constant-speed approximation rather than actual frame timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned through shared position bins. The time-since-start is identical for every trial (always 0, 1/6, 2/6, ..., 39/6 seconds), since it is derived from bin index rather than actual timestamps.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. Alignment is implicit through position bins.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks trials in the rewarded corridor.

ii.
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The AI identified `isRew` as the correct source variable.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float and broadcast across all 40 position bins for each trial. No further processing.

ii.
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. Straightforward conversion from boolean to float.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name on the corridor walls for each trial, and `UniqWalls` to collect all unique stimulus names.

ii.
```python
all_stim_set = set()
for s in sessions:
    for wn in s['beh']['UniqWalls']:
        all_stim_set.add(str(wn))
all_stim_names = sorted(all_stim_set)

stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. The AI used `WallName` for per-trial assignment and `UniqWalls` to build the global set of stimulus names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps all 15 individual texture names as separate categories (e.g., `circle1`, `circle2`, `leaf1`, `leaf1_swap1`, etc.) rather than grouping them into 4 base texture categories. Each trial's `WallName` is mapped to its index in the sorted list of all unique stimulus names.

ii.
```python
all_stim_names = sorted(all_stim_set)
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. The AI used individual wall names rather than grouping them into base texture categories, resulting in ~15 stimulus categories instead of 4.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickPos` (position of each lick event in dm) and `LickTrind` (trial index of each lick event).

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
```

iii. The AI used position-based lick data rather than frame-based (`LickFr`).

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick event is assigned to a position bin based on its position (integer part of `LickPos`). A binary array `(n_trials, n_bins)` is created where 1 indicates at least one lick occurred in that position bin for that trial.

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

iii. The AI converted lick events from continuous positions to binary per-position-bin indicators.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Aligned through position bins. Both licking and neural data are mapped to the same 40 position bins.

ii.
```python
lick_tr = lick_binary[tr, :].astype(np.int64)
```

iii. Alignment is implicit through shared position bins.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From the position bin index within the corridor (0-39), which is deterministic given the position-interpolation approach.

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)  # 0-9 -> 0, 10-19 -> 1, 20-29 -> 2, 30-39 -> 3
```

iii. The AI derived position bins directly from the spatial bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 position bins are grouped into 4 equal 1-m bins: bins 0-9 map to category 0 (0-1m), 10-19 to category 1 (1-2m), 20-29 to category 2 (2-3m), 30-39 to category 3 (3-4m).

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. Since data is position-interpolated, position bins are deterministic and identical for every trial.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Integer division of the position bin index by 10, clamped to 0-3. This produces 4 equal-length 1-m spatial bins as specified in the instructions.

ii.
```python
pos_bins[i] = min(i // 10, 3)
```

iii. Straightforward spatial binning matching the instruction's 4 equal-length 1-m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is deterministic given the position-interpolation approach: bin index `i` always corresponds to position `i * 0.1 m`. It is perfectly aligned with neural data through the shared position bins.

ii.
```python
pos_tr = pos_bins.astype(np.int64)  # same for every trial
```

iii. Alignment is inherent in the position-interpolated framework.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `run_pos`, the position-interpolated running speed already available in the behavior data, with shape `(n_trials, 60)`.

ii.
```python
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)
```

iii. The AI used the pre-computed position-interpolated running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The running speed is discretized into 4 quartile bins using globally pre-computed quantile edges (25th, 50th, 75th percentiles across all sessions). `np.digitize` is used to assign each speed value to a bin.

ii.
```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])

speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. The AI computed global quartile edges across all sessions and applied them uniformly. This differs from per-session quartile computation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three threshold edges at the 25th, 50th, and 75th percentiles of all running speeds across all sessions are computed. `np.digitize` assigns values to bins 0-3 based on these thresholds.

ii.
```python
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. Using `np.percentile` with threshold-based binning rather than rank-based binning means that ties (e.g., many zero-speed values) can cause uneven bin sizes.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The pre-computed `run_pos` is already position-interpolated, so it aligns with the position-interpolated neural data through shared position bins.

ii.
```python
run_speed = run_pos[:, :N_TEXTURE_BINS]
speed_tr = speed_bins[tr, :].astype(np.int64)
```

iii. Alignment through shared position bins.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data by: (1) skipping sessions where behavior keys are not found, (2) skipping sessions with fewer than 2 trials, (3) catching and printing exceptions during session processing. For licking, out-of-range position bins and trial indices are silently skipped.

ii.
```python
if beh_key not in beh_data:
    print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
    continue

if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue

try:
    result = process_session(...)
except Exception as e:
    print(f"  ERROR processing session: {e}")
    continue

# In lick processing:
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

iii. The AI used defensive programming with try/except blocks and bounds checking to handle potential data issues.

## 12-a. What are the most time-consuming steps of the code?

i. The position interpolation of neural data is the most time-consuming step. For each session with 30k-64k neurons, every neuron must be individually interpolated across all frames. The AI noted this took about 30 minutes for the full dataset.

ii.
```python
def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    for s in range(n_neurons):
        interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

iii. The AI acknowledged this bottleneck in the trajectory: "The conversion is taking a very long time due to the position interpolation of large neural datasets (30-90k neurons per session)."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The position interpolation loop over neurons could potentially be vectorized. The lick-to-position-bin conversion also uses a Python loop over individual lick events.

ii.
```python
# Neuron-by-neuron interpolation:
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])

# Lick event loop:
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(pos)
```

iii. The neuron interpolation loop is the main performance bottleneck. The lick loop is minor in comparison.

## 12-c. What processing does the code repeat multiple times?

i. The code loads behavior data twice: once in `collect_all_sessions` (storing the full behavior dict in session info objects) and again implicitly when processing sessions. Running speed quantiles are computed in a separate pass over all sessions before the main processing loop.

ii.
```python
# First pass: collect all sessions and store beh data
sessions.append({...  'beh': beh_data[beh_key], ...})

# Second pass: compute speed quantiles
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
```

iii. The two-pass approach was necessary to compute global speed quantiles before processing sessions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code interpolates neural data to 60 position bins per corridor, then discards the last 20 (grey area). This means 33% of the interpolation work is wasted. The code also stores all behavior data in memory during the collection phase, which is unnecessary for sessions that haven't been processed yet.

ii.
```python
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving], corridor_len, n_trials, n_bins=N_POS_BINS)  # 60 bins
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]  # keep only 40
```

iii. The full 60-bin interpolation was done to match the paper's convention, even though only 40 bins were used.
