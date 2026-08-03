# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is read from three directories under `data`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. `beh/Imaging_Exp_info.npy` is loaded as the master index of all recordings grouped by experiment type. Each experiment type has a corresponding `Beh_<exp_type>.npy` behavior file. For each unique recording, the behavior file is loaded, neural data (`spk/<session>_neural_data.npy`) and retinotopy (`retinotopy/<mouse>_<date>_trans.npz`) are loaded per session.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
# ...
beh_path = os.path.join(root, 'beh', f'Beh_{exp_type}.npy')
beh_data = np.load(beh_path, allow_pickle=True).item()
# ...
spk_fn = f"{mname}_{datexp}_{blk}_neural_data.npy"
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
# ...
ret_fn = f"{mname}_{datexp}_trans.npz"
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
iarea = ret_data['iarea']
```

iii. The AI loads all behavior files for each experiment type upfront in `collect_all_sessions()`, and stores the behavior dict for each session. Neural and retinotopy data are loaded per-session in `process_session()`. The trajectory shows the agent identified that the same recording can appear under multiple experiment types and should only be processed once.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `mname` in each experiment info entry. All unique subjects are collected with `sorted(set(s['mname'] for s in sessions))`, and a `subject_idx` array maps each session to its index in the sorted subject list.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions))
# ...
all_subject_idx.append(all_subjects.index(result['mname']))
```

iii. The mouse name is directly available in the experiment info entries, so no derivation is needed.

## 1-c. How are the data split into sessions?

i. A session is defined by the unique tuple `(mname, datexp, blk)`. When a recording appears under multiple experiment types, only the first match is used, with a preference ordering: test > non-test, supervised > unsupervised, after-learning > before-learning.

ii.
```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
```

The experiment type ordering:
```python
exp_order = sorted(exp_info.keys(), key=lambda x: (
    0 if 'test' in x else 1,
    0 if 'sup_' in x else 1,
    0 if 'after' in x else 1,
    x
))
```

iii. The trajectory shows the agent recognized that recordings repeat across experiment types and chose to prefer test sessions over training sessions for the behavior annotations. This results in 89 unique sessions.

## 1-d. How are the data split into trials?

i. The number of trials is taken directly from `beh['ntrials']`. Each trial runs through the corridor. The AI position-interpolates the neural data into 60 position bins per corridor (matching the paper's `spk_pos_interp`), then keeps only the first 40 bins (texture area, 4m corridor). All trials have the same number of timepoints (40 bins).

ii.
```python
n_trials = beh['ntrials']
# ...
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The AI followed the paper's position-interpolation approach, where each corridor traversal becomes one trial with exactly 40 position bins for the texture area. This means trial boundaries are implicit in the cumulative position data rather than explicitly defined by frame indices.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal filtering: sessions with fewer than 2 trials are skipped. No per-trial quality filtering is applied beyond what the position interpolation handles implicitly (only running frames with `ft_move > 0` are used).

ii.
```python
if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
```

iii. The CONVERSION_NOTES.md states all 89 recordings are processed successfully with no sessions skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains deconvolved calcium traces as a list of one array per imaging plane. These are concatenated across planes. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The AI correctly identifies `spks` as deconvolved fluorescence data from Suite2p.

## 2-b. How is the `neural` data processed?

i. The AI performs position interpolation of the neural data. Only running frames (`ft_move > 0`) are used. The cumulative position (`ft_PosCum`) is used to interpolate each neuron's trace into 60 evenly-spaced position bins per corridor, then only the first 40 bins (texture area) are kept. This follows `spk_pos_interp()` / `get_interpPos_spk()` from the reference code repository.

ii.
```python
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
ft_pos_cum = beh['ft_PosCum'][:n_frames]

valid_spk = spk[valid_neurons][:, vr_moving]
interp_spk = position_interpolate_spk(
    valid_spk, ft_pos_cum[vr_moving], corridor_len, n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

The interpolation function:
```python
def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
    norm_pos = accum_pos / corridor_len
    n_neurons = raw_spk.shape[0]
    n_target = len(lin_pos)
    interp_spk = np.zeros((n_neurons, n_target), dtype=np.float32)
    for s in range(n_neurons):
        interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
    return interp_spk.reshape(n_neurons, n_trials, n_bins)
```

iii. The agent's trajectory reasoning shows it decided to follow the paper's position-interpolation approach (from `utils.py`) rather than using the raw frame-based data. The agent notes: "The paper interpolates neural data into position bins (60 bins per corridor, 1 dm each)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only visual cortex neurons are kept: neurons with `iarea != -1` and `iarea != 7`. This maps to V1 (iarea=8), mHV (iarea in {0,1,2,9}), lHV (iarea in {5,6}), aHV (iarea in {3,4}).

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

iii. The AI follows the `neu_area_ID()` function from the reference code's `utils.py`, which defines the same brain region mapping. Only running frames are used for interpolation, following the paper's convention.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is trial start (corridor entry). The AI uses position interpolation to create 40 evenly-spaced position bins per trial, starting from corridor entry (position 0). Each trial has exactly 40 timepoints corresponding to 40 dm of texture corridor.

ii.
```python
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving],
    corridor_len, n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]  # Keep only texture area
```

iii. The position-interpolation approach inherently aligns data to corridor entry, since position 0 corresponds to the start of the corridor texture.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 166.67 ms, derived from the VR speed: 1 dm / 6 dm/s = 1/6 second. This is a position-based binning rather than a temporal binning. Each trial has 40 bins corresponding to 40 dm of texture corridor.

ii.
```python
VR_SPEED = 6.0        # dm/s (60 cm/s)
BIN_SIZE_SEC = 1.0 / VR_SPEED  # seconds per position bin = 1/6 s
BIN_SIZE_MS = BIN_SIZE_SEC * 1000  # 166.67 ms
```

iii. The AI derives the bin size from the VR speed (60 cm/s = 6 dm/s), assuming the mouse runs at constant speed. This is a position-based rather than time-based binning. The raw imaging rate is 3.17 Hz (315 ms per frame).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundPos']`, the position (in dm) where the sound cue was delivered in each trial.

ii.
```python
sound_pos = beh['SoundPos']
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The AI uses position-based sound cue location rather than the frame-based `SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each position bin, the time to the sound cue is computed as `(position_bin_index - sound_cue_position) * bin_size_seconds`. This gives negative values before the cue and positive values after. Note: the sign convention is opposite to the reference -- the reference computes `cue_time - current_time` (positive before cue), while the AI computes `current_position - cue_position` (negative before cue).

ii.
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The AI converts from position-space difference to time difference using the constant VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both the neural data and the time-to-cue are computed on the same position-bin grid (40 bins per trial), so they are inherently aligned.

ii.
```python
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail], axis=0)
```

iii. All variables are computed on the same position-bin axis.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date string `datexp` in each session's metadata (e.g., '2022_08_17').

ii.
```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')
```

iii. The date is parsed from the experiment date string.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is computed as the number of calendar days since the first recording for each mouse. The first recording date per mouse is identified, then each session's day is `(session_date - first_recording_date).days`. This value is broadcast across all 40 position bins of each trial.

ii.
```python
mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d

session_days = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    days = (d - mouse_first_date[s['mname']]).days
    key = (s['mname'], s['datexp'], s['blk'])
    session_days[key] = float(days)
```

iii. The AI counts calendar days rather than recording session ordinals.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the position bin index and the constant VR speed (BIN_SIZE_SEC).

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. Time since trial start is computed as a linear function of position, assuming constant VR speed.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each position bin `i` maps to time `i * BIN_SIZE_SEC` seconds. Since BIN_SIZE_SEC = 1/6 s, the time ranges from 0 to ~6.5 seconds across 40 bins. This assumes the mouse runs at constant VR speed (60 cm/s).

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The constant-speed assumption means every trial has identical time-since-start values, which removes the natural variability in how fast the mouse traverses the corridor.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Aligned via the shared position-bin axis. Both neural and input data use the same 40-bin position grid.

ii. Same position-bin grid as neural data.

iii. Inherent alignment through position interpolation.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, which marks whether each trial is in a rewarded corridor.

ii.
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. Directly available from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float and broadcast across all 40 position bins of each trial. 1 if rewarded corridor, 0 otherwise.

ii.
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. No complex processing needed; it is a per-trial binary variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']`, which names the texture on the corridor walls for each trial. Also from `beh['UniqWalls']` to collect all unique stimulus names.

ii.
```python
wall_names = beh['WallName']
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. WallName gives the specific stimulus variant (e.g., 'circle1', 'leaf2').

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps all 15 individual stimulus names as separate categories (e.g., circle1, circle2, circle3, leaf1, leaf2, etc.) rather than grouping them into 4 base textures. The stimulus index is broadcast across all 40 position bins.

ii.
```python
all_stim_set = set()
for s in sessions:
    for wn in s['beh']['UniqWalls']:
        all_stim_set.add(str(wn))
all_stim_names = sorted(all_stim_set)
# ...
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

Output values:
```python
output_values = [
    all_stim_names,  # 15 individual stimulus names
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['Q1', 'Q2', 'Q3', 'Q4'],
]
```

iii. The AI chose to keep all 15 stimulus variants as separate categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickPos']` (position of each lick in dm) and `beh['LickTrind']` (trial index of each lick).

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
```

iii. The AI uses position-based lick data rather than frame-based (`LickFr`).

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick event is assigned to a position bin based on its position (`LickPos`). If any lick falls in a given position bin for a trial, that bin is set to 1, otherwise 0.

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

iii. The AI bins licks into position bins rather than imaging frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Both licking and neural data are on the same position-bin grid (40 bins), so they are inherently aligned.

ii. The lick_binary array has shape (n_trials, 40), matching the neural data's 40 position bins per trial.

iii. Alignment through shared position-bin axis.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is deterministic in the position-interpolated representation: bin `i` corresponds to position `i` dm. No raw variable is needed since position is the axis itself.

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)  # 0-9 -> 0, 10-19 -> 1, 20-29 -> 2, 30-39 -> 3
```

iii. In the position-interpolated representation, the position is deterministic and the same for every trial.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 position bins are grouped into 4 equal 1-meter bins (10 dm each): bins 0-9 -> category 0, bins 10-19 -> category 1, bins 20-29 -> category 2, bins 30-39 -> category 3.

ii.
```python
pos_bins[i] = min(i // 10, 3)
```

iii. Simple integer division by 10 gives the 1-meter bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins. Each bin covers 10 dm (10 position bins). The categories are: '0-1m', '1-2m', '2-3m', '3-4m'.

ii.
```python
pos_bins[i] = min(i // 10, 3)
# output_values for position: ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. Matches the instruction's requirement for "4 equal-length, 1-m-long spatial bins".

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is deterministic on the position-bin axis, so it is always aligned with the neural data.

ii. Same 40-bin position grid.

iii. Inherent alignment through position interpolation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['run_pos']`, which contains position-interpolated running speed (n_trials x 60 bins per corridor).

ii.
```python
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)
```

iii. The AI uses the already position-interpolated running speed from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 quartile bins using globally-computed quartile edges across all sessions. The quartile edges are computed from the `run_pos` data across all 89 sessions.

ii.
```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
# ...
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. Global quartile edges ensure consistent speed categories across sessions. The edges are [16.58, 28.72, 43.29] cm/s.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. 4 bins using `np.digitize` with 3 global quartile edges (Q25, Q50, Q75). Values below Q25 -> 0, Q25-Q50 -> 1, Q50-Q75 -> 2, above Q75 -> 3. Categories: ['Q1', 'Q2', 'Q3', 'Q4'].

ii.
```python
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. Using global edges means the distribution of speed categories can vary across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Both running speed and neural data are on the same position-bin grid (40 bins per trial).

ii. `run_speed = run_pos[:, :N_TEXTURE_BINS]` gives one speed value per position bin, matching the neural data.

iii. Inherent alignment through position interpolation.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data by: (1) skipping sessions that fail to process with a try/except; (2) skipping sessions with fewer than 2 trials; (3) only using running frames (`ft_move > 0`) for position interpolation; (4) clipping lick positions to valid bin range.

ii.
```python
try:
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
except Exception as e:
    print(f"  ERROR processing session: {e}")
    continue

if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
```

```python
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

iii. The broad try/except ensures the conversion continues even if individual sessions fail.

## 12-a. What are the most time-consuming steps of the code?

i. Two main bottlenecks: (1) Loading the large spike files (~405 GB total across all sessions); (2) Position interpolation of neural data, which loops over each neuron individually with `np.interp`. The trajectory shows this was a significant performance issue.

ii.
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
# ...
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

iii. The trajectory shows the agent had to optimize the interpolation code (switching from scipy to numpy.interp) due to the large number of neurons per session (30k-90k).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The position interpolation loop over neurons could potentially be vectorized. The lick-to-position-bin conversion also loops over individual lick events rather than using vectorized operations.

ii.
```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

```python
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(pos)
```

iii. The neuron interpolation loop is the most impactful, but `np.interp` doesn't support batch interpolation natively.

## 12-c. What processing does the code repeat multiple times?

i. The code loads all behavior files twice: once in `collect_all_sessions()` to build the session list (storing beh dicts), and the behavior data remains in memory. Running speed quantile computation iterates over all sessions' `run_pos` data separately from the main processing loop.

ii.
```python
# First pass: collect_all_sessions loads all behavior
beh_data = np.load(beh_path, allow_pickle=True).item()
sessions.append({..., 'beh': beh_data[beh_key], ...})

# Second pass: speed quantiles
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
```

iii. The behavior data is kept in memory from the first pass through the second, trading memory for avoiding re-reads.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Position is deterministic in the position-interpolated representation (every trial has the exact same position values), so decoding position from neural activity is trivial/meaningless in this framework. The position output provides no information about neural coding since it is constant across all trials. Additionally, the time_since_trial_start input is also identical across all trials (linear ramp from 0 to ~6.5s).

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
# pos_tr is the same for every trial
pos_tr = pos_bins.astype(np.int64)
```

iii. Because position is deterministic (identical for every trial), a decoder cannot learn anything meaningful about neural representations of position. This is a consequence of the position-interpolation approach.
