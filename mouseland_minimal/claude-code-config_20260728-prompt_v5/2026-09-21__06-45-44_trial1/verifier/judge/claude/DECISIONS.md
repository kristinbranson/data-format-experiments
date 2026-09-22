# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories under `data`: `beh` for behavior, `spk` for deconvolved calcium traces, and `retinotopy` for visual area assignments. `beh/Imaging_Exp_info.npy` is loaded first as the master index. The AI iterates over experiment types, collecting unique sessions keyed by `(mname, datexp, blk)`, keeping only the first occurrence of each. Behavior files (`Beh_<exp_type>.npy`) are cached by experiment type, and spike files and retinotopy files are loaded per session. A two-pass approach is used: first pass collects global statistics (stimuli, speed values, mouse dates), second pass processes each session.

ii.
```python
exp_info = np.load(
    os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
    allow_pickle=True
).item()

unique_sessions = {}
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = (ndb['mname'], ndb['datexp'], ndb['blk'])
        if key not in unique_sessions:
            unique_sessions[key] = (exp_type, ndb)
```

```python
spk = load_spk(mname, datexp, blk)
# which calls:
fn = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
d = np.load(fn, allow_pickle=True).item()
return np.concatenate(d['spks'], axis=0)
```

iii. The AI's trajectory shows it explored the experiment info structure, understood sessions can appear under multiple experiment types, and decided to deduplicate by `(mname, datexp, blk)` tuple. The two-pass approach was chosen to compute global speed quartiles before processing individual sessions.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from `mname` in the experiment info entries. Subjects are collected as sorted unique mouse names, and `subject_idx` maps each session to its index in the subjects list.

ii.
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
subjects = sorted(mouse_first_date.keys())
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

```python
data_subject_idx.append(subject_to_idx[mname])
```

iii. The AI recognized that mouse names are directly available in the experiment info entries and used them to build the subject list.

## 1-c. How are the data split into sessions?

i. A session is identified by the tuple `(mname, datexp, blk)`. The AI iterates over all experiment types in exp_info, and for each unique tuple keeps only the first occurrence. Sessions with the same recording but different experiment types are deduplicated. This yields 89 unique sessions.

ii.
```python
unique_sessions = {}
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = (ndb['mname'], ndb['datexp'], ndb['blk'])
        if key not in unique_sessions:
            unique_sessions[key] = (exp_type, ndb)

session_keys = sorted(unique_sessions.keys())
```

iii. The AI recognized that the same recording can appear under multiple experiment types and deduplicated appropriately.

## 1-d. How are the data split into trials?

i. Trials are defined by `ntrials` in the behavior data. However, instead of using raw frame-based trial boundaries, the AI uses **position interpolation** to resample neural data into 40 uniform position bins per trial (covering the 4m texture area). Each trial therefore has exactly 40 time bins.

ii.
```python
ntrials = int(beh['ntrials'])
# ...
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS)
# ...
for t in range(ntrials):
    neural_t = texture_spk[:, t, :]  # (n_neurons, 40)
```

iii. The AI's trajectory shows extensive deliberation between raw frame-based data and position-interpolated data. The AI ultimately chose position interpolation because it matches the paper's canonical processing pipeline (`get_interpPos_spk`), gives uniform trial lengths (40 bins), and naturally handles the running-only frame filter.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only if they contain NaN or Inf values in the neural data after position interpolation. Sessions with fewer than 2 valid trials or fewer than 2 total trials are skipped. There is no filtering based on trial length (since all trials are resampled to 40 bins).

ii.
```python
if ntrials < 2:
    print(f" -> SKIP: {ntrials} trials")
    continue
# ...
if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
    continue
# ...
if len(session_neural) < 2:
    print(f" -> SKIP: {len(session_neural)} valid trials")
    continue
```

iii. The AI's trajectory does not discuss trial length filtering because position interpolation produces fixed-length trials, making length-based filtering unnecessary. The NaN/Inf check handles edge cases from the interpolation process.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains one neurons-by-frames array per imaging plane, concatenated into a single array. The visual area of each neuron comes from `iarea` in `retinotopy/<mouse>_<datexp>_trans.npz`. Additionally, `ft_move` (running indicator) and `ft_PosCum` (cumulative position) from the behavior data are used for the position interpolation.

ii.
```python
spk = load_spk(mname, datexp, blk)
# loads spk/<mname>_<datexp>_<blk>_neural_data.npy, concatenates d['spks']

iarea = load_iarea(mname, datexp)
# loads retinotopy/<mname>_<datexp>_trans.npz, returns iarea

VRmove = beh['ft_move'][:n_fr] > 0
pos_cum = beh['ft_PosCum'][:n_fr]
```

iii. The AI recognized that the paper's standard processing uses running frames and cumulative position for interpolation.

## 2-b. How is the `neural` data processed?

i. The neural data undergoes **position interpolation**: only running frames (`ft_move > 0`) are kept, then neural activity is linearly interpolated from cumulative position space onto a uniform grid of 40 position bins per trial (covering the 4m texture area). This matches the paper's `get_interpPos_spk` / `spk_pos_interp` functions. The result is stored as float32.

ii.
```python
VRmove = beh['ft_move'][:n_fr] > 0
pos_cum = beh['ft_PosCum'][:n_fr]
spk_running = spk[:, :n_fr][:, VRmove]
pos_cum_running = pos_cum[VRmove]

texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS)
```

```python
def position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS):
    bin_offsets = np.arange(n_bins, dtype=np.float64) / CORRIDOR_LENGTH
    trial_starts = np.arange(ntrials, dtype=np.float64)
    target = (trial_starts[:, None] + bin_offsets[None, :]).ravel()
    source = pos_cum_running / CORRIDOR_LENGTH
    n_neurons = spk_running.shape[0]
    result = np.empty((n_neurons, len(target)), dtype=np.float32)
    for i in range(n_neurons):
        result[i] = np.interp(target, source, spk_running[i])
    return result.reshape(n_neurons, ntrials, n_bins)
```

iii. The AI explicitly chose position interpolation to match the paper's methodology, noting "position-interpolated deconvolved spikes (Suite2p with 0.75s decay). 60 bins per 6m corridor, first 40 bins (texture area) retained. Only running frames (VR moving) used for interpolation."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by excluding those with `iarea == -1` (unassigned) or `iarea == 7` (non-visual). This differs slightly from the reference, which keeps only neurons in V1, mHV, lHV, and aHV (area codes 8, 0,1,2,9, 5,6, 3,4). The AI's filter `(iarea != -1) & (iarea != 7)` would also include area codes not mapped to any of the four named regions (if any exist).

ii.
```python
vc_mask = (iarea != -1) & (iarea != 7)  # visual cortex mask
brain_region = get_brain_region_idx(iarea[vc_mask])
spk = spk[vc_mask]
```

```python
def get_brain_region_idx(iarea):
    region = np.empty(len(iarea), dtype=np.int32)
    region[:] = -1  # temporary
    region[iarea == 8] = 0  # V1
    for a in [0, 1, 2, 9]:
        region[iarea == a] = 1  # mHV
    for a in [5, 6]:
        region[iarea == a] = 2  # lHV
    for a in [3, 4]:
        region[iarea == a] = 3  # aHV
    return region
```

iii. The AI cited the paper's convention `idx_neu = (arid!=-1) & (arid != 7)` from `Get_density_map` as its justification for this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment is to corridor entry (trial start). Since position interpolation is used, each trial's position bin 0 corresponds to position 0 (corridor entry), so alignment is implicit. All trials have exactly 40 bins covering 0-4m of the texture area.

ii.
```python
# Target positions start at 0 for each trial
bin_offsets = np.arange(n_bins, dtype=np.float64) / CORRIDOR_LENGTH
trial_starts = np.arange(ntrials, dtype=np.float64)
target = (trial_starts[:, None] + bin_offsets[None, :]).ravel()
```

iii. The AI noted that "corridor entry corresponds to position 0, position interpolation actually already satisfies the alignment requirement."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is approximately 166.67 ms, derived from the VR speed of 60 cm/s and position bin size of 10 cm (1 decimeter): `1/6 s = 166.67 ms`. This is fundamentally different from the reference's 315 ms imaging frame rate. The data is rebinned via position interpolation rather than using raw imaging frames.

ii.
```python
VR_SPEED_DM_S = 6.0     # VR speed in dm/s (60 cm/s)
TIME_PER_BIN = 1.0 / VR_SPEED_DM_S  # seconds per position bin (1/6 s)
TIME_PER_BIN_MS = TIME_PER_BIN * 1000  # ~166.67 ms
```

iii. The AI computed the bin size from the VR movement speed, treating each 1-decimeter position bin as a constant-time interval.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos`, the position (in decimeters) at which the sound cue was played in each trial.

ii.
```python
sound_pos = beh['SoundPos'][t]  # in decimeters
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
```

iii. The AI used `SoundPos` (position-based) rather than `SoundFr` (frame-based) because the data is position-interpolated. The sound cue position is converted to time using the VR speed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue at each position bin is computed as `(SoundPos - bin_index) * TIME_PER_BIN`. This gives positive values before the cue position and negative values after, in seconds.

ii.
```python
sound_pos = beh['SoundPos'][t]  # in decimeters
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
```

iii. The AI converted the spatial distance to the sound cue into a temporal measure using the constant VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both are on the same position-bin grid. The neural data is interpolated to 40 position bins, and the time to sound cue is computed at each of those same 40 bins.

ii.
```python
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
# time_to_cue has shape (40,), same as neural bins
```

iii. Alignment is implicit since both use the same position-bin axis.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field of each session, which contains the recording date as a string in `YYYY_MM_DD` format. The first recording date for each mouse is used as the reference.

ii.
```python
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_dates.setdefault(mname, []).append(date)
# ...
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
```

iii. The AI chose to derive day of training from calendar dates rather than counting recording sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Day of training is computed as the number of **calendar days** elapsed between the session's date and the mouse's first recording date. This is broadcast across all 40 bins of each trial.

ii.
```python
session_date = datetime.strptime(datexp, '%Y_%m_%d')
day_of_training = float((session_date - mouse_first_date[mname]).days)
# ...
day_arr = np.full(N_BINS, day_of_training, dtype=np.float32)
```

iii. The AI reasoned that computing days elapsed from each mouse's first recording date gives a uniform scale across all experiment types.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the position bin index and the constant VR speed. Each bin is 1 decimeter, and at 6 dm/s, each bin represents 1/6 second.

ii.
```python
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
```

iii. Since position interpolation produces uniform position bins, the time since trial start is computed deterministically from the bin index.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is simply `bin_index * TIME_PER_BIN`, where `TIME_PER_BIN = 1/6 s`. This gives a linearly increasing time from 0 to approximately 6.5 seconds across the 40 bins.

ii.
```python
TIME_PER_BIN = 1.0 / VR_SPEED_DM_S  # 1/6 seconds
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
```

iii. The AI treated time as deterministic given position interpolation at constant VR speed.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Both are on the same 40-bin position grid. `time_since_start` is a fixed array `[0, 1/6, 2/6, ..., 39/6]` seconds that is identical for every trial.

ii.
```python
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. Alignment is inherent in the position-interpolated representation.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavior data, which marks whether the trial was in a rewarded corridor.

ii.
```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The AI recognized `isRew` as the direct indicator of reward availability.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[t]` is converted to float (0.0 or 1.0) and broadcast across all 40 bins of the trial.

ii.
```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. No additional processing needed; the field directly indicates reward availability.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` in the behavior data, which names the texture on the walls of each trial's corridor. Also from `UniqWalls` during the first pass to collect all unique stimulus names.

ii.
```python
# First pass:
for wn in beh['UniqWalls']:
    all_stimuli.add(str(wn))
# ...
stim_list = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}

# Second pass:
stim_name = str(beh['WallName'][t])
stim_cat = stim_to_idx[stim_name]
stim_arr = np.full(N_BINS, stim_cat, dtype=np.int32)
```

iii. The AI used individual wall names (e.g., "circle1", "circle2", "leaf1") as separate stimulus categories rather than grouping them into base textures.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each unique wall name is treated as a separate stimulus category. The sorted list of all unique wall names across all sessions forms the output categories (15 categories total). Each trial's stimulus is mapped to its index in this sorted list and broadcast across all 40 bins.

ii.
```python
stim_list = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
# ...
stim_name = str(beh['WallName'][t])
stim_cat = stim_to_idx[stim_name]
stim_arr = np.full(N_BINS, stim_cat, dtype=np.int32)
```

iii. The AI did not group wall names into base texture categories (circle, leaf, rock, wood). Instead, each variant (e.g., circle1, circle2, circle3) is treated as a distinct category, yielding 15 categories instead of 4.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickPos` (position of each lick in decimeters) and `LickTrind` (trial index of each lick) in the behavior data.

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
```

iii. The AI used position-based lick data (`LickPos`/`LickTrind`) rather than frame-based lick data (`LickFr`) because the neural data is position-interpolated.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks belonging to that trial (identified by `LickTrind == t`) are selected. Their positions are filtered to valid range (0 to 40 dm), floored to integer bin indices, and marked as 1 in a binary array of length 40.

ii.
```python
lick_binary = np.zeros(N_BINS, dtype=np.int32)
trial_mask = lick_trind == t
trial_lick_pos = lick_pos[trial_mask]
valid_licks = trial_lick_pos[(trial_lick_pos >= 0) & (trial_lick_pos < N_BINS)]
if len(valid_licks) > 0:
    lick_bins = np.clip(np.floor(valid_licks).astype(int), 0, N_BINS - 1)
    lick_binary[lick_bins] = 1
```

iii. The AI converted from continuous lick positions to a binary per-position-bin representation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Both licking and neural data are on the same 40-bin position grid. Lick positions are binned into the same decimeter bins used for the neural position interpolation.

ii.
```python
lick_binary = np.zeros(N_BINS, dtype=np.int32)
# ...
lick_bins = np.clip(np.floor(valid_licks).astype(int), 0, N_BINS - 1)
lick_binary[lick_bins] = 1
```

iii. Alignment is implicit through shared position-bin indexing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is **not derived from any raw data variable**. Since position interpolation resamples data to uniform position bins (0-39), position is deterministically known from the bin index.

ii.
```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
# ...
pos_arr = position_bins.copy()
```

iii. The AI recognized that under position interpolation, position is trivially determined by the bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 position bins are divided into 4 equal groups of 10 bins each, corresponding to 0-1m, 1-2m, 2-3m, 3-4m. This is a static array computed once and reused for every trial.

ii.
```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
# Result: [0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3] (10 each)
```

iii. This is a direct consequence of the position-interpolation approach.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Each of the 4 categories spans exactly 10 position bins (1 meter each). Bins 0-9 = category 0 (0-1m), bins 10-19 = category 1 (1-2m), bins 20-29 = category 2 (2-3m), bins 30-39 = category 3 (3-4m).

ii.
```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
```

iii. The categorization follows directly from the 4 equal-length bins requested in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is on the same 40-bin grid as the neural data. Alignment is implicit.

ii.
```python
pos_arr = position_bins.copy()
output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], axis=0)
```

iii. Both share the position-bin axis.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `run_pos` in the behavior data, which contains position-interpolated running speed per trial per position bin.

ii.
```python
speed = beh['run_pos'][t, :N_BINS].copy()
```

iii. The AI used the pre-computed position-interpolated speed from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using **global quartile thresholds** computed across all sessions in the first pass. `np.digitize` is used with the 25th, 50th, and 75th percentile values. NaN values are replaced with 0.

ii.
```python
# First pass:
speed = beh['run_pos'][:, :N_BINS]
valid = ~np.isnan(speed)
all_speed_values.append(speed[valid].ravel())
# ...
speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])

# Second pass:
speed = beh['run_pos'][t, :N_BINS].copy()
speed[np.isnan(speed)] = 0.0
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The AI computed global quartile thresholds across all sessions to ensure consistent categorization, noting that the instructions specify "4 bins, each corresponding to 25% of the data."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three threshold values (25th, 50th, 75th percentiles) are computed globally across all sessions' texture-area speed values. `np.digitize` assigns each speed value to one of 4 bins: below 25th percentile, 25th-50th, 50th-75th, above 75th.

ii.
```python
speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The AI chose global thresholds rather than per-session thresholds to maintain consistency across the dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` is already position-interpolated in the behavior data, so taking the first 40 columns aligns with the 40-bin neural data grid.

ii.
```python
speed = beh['run_pos'][t, :N_BINS].copy()
```

iii. Alignment is through the shared position-bin axis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN values in speed are replaced with 0. Trials with NaN/Inf in neural data after interpolation are skipped. Sessions with fewer than 2 valid trials are skipped. The behavior and spike data are truncated to the minimum of their respective lengths (`n_fr = min(n_frames_spk, len(beh['ft_move']))`).

ii.
```python
n_fr = min(n_frames_spk, len(beh['ft_move']))
# ...
speed[np.isnan(speed)] = 0.0
# ...
if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
    continue
```

iii. The AI handled length mismatches between spike and behavior data, and used NaN/Inf checks as a safety net for the interpolation process.

## 12-a. What are the most time-consuming steps of the code?

i. The position interpolation of neural data is the most time-consuming step. It loops over each neuron individually calling `np.interp`, and with 50-70K neurons per session this takes ~18 seconds per session. The total conversion across 89 sessions took over 40 minutes according to the trajectory.

ii.
```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```

iii. The AI noted in the trajectory that "the large neuron counts (60k-80k) make interpolation slow."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron-by-neuron interpolation loop in `position_interpolate` is the main candidate for vectorization. `np.interp` only handles 1D, so each neuron requires a separate call. Using `scipy.interpolate.interp1d` with an `axis` parameter could potentially vectorize this.

ii.
```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```

iii. The AI considered using `scipy.interpolate.interp1d` for vectorization but opted for the simpler `np.interp` loop.

## 12-c. What processing does the code repeat multiple times?

i. The code does two full passes over the data: the first pass loads all behavior files to collect global statistics (stimuli names, speed values, mouse dates), and the second pass reloads them for actual processing. This means every behavior file is loaded twice.

ii.
```python
# First pass:
for key in session_keys:
    beh_cache[exp_type] = np.load(...)
# ...
del beh_cache

# Second pass:
for sess_i, key in enumerate(session_keys):
    beh_cache[exp_type] = np.load(...)
```

iii. The two-pass approach was chosen to compute global speed quartiles before processing, but the behavior data could have been cached from the first pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The position interpolation itself is arguably unnecessary processing, as it transforms the raw frame-based neural data into a different representation. The reference solution uses raw frames directly, which is simpler and preserves more information. Additionally, the `gc.collect()` calls add overhead that is unlikely to be necessary given the available memory.

ii.
```python
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS)
del spk_running, pos_cum_running
gc.collect()
```

iii. The AI spent considerable time on the position interpolation approach, which adds significant computational cost compared to simply extracting raw frame data.
