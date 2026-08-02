# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `data/beh/Imaging_Exp_info.npy`, which contains experiment info organized by experiment type. It iterates over all experiment types to collect session entries (mname, datexp, blk), deduplicates by session_id (`mname_datexp_blk`), keeping only the first occurrence when a neural recording appears under multiple experiment types. Neural data is loaded per-session from `.npy` files in `data/spk/` using `load_spk()` which concatenates across imaging planes. Behavior data is loaded from `data/beh/Beh_{exp_type}.npy` files.

ii.
```python
def get_all_sessions():
    info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    session_map = {}
    for exp_type in info.keys():
        for entry in info[exp_type]:
            session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            beh_key = session_id if 'stimtype' not in entry else f"{session_id}_{entry['stimtype']}"
            if session_id not in session_map:
                session_map[session_id] = (exp_type, entry, beh_key)
    ...

def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. The AI documented in CONVERSION_NOTES.md that it identified 89 unique sessions matching the paper ("We performed 89 recordings in 19 mice"). The deduplication strategy uses the first experiment type encountered for each neural recording. The `load_spk` function matches the reference code's `utils.load_spk()`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in the session metadata. A list of unique subjects is built as sessions are processed, preserving encounter order. Each session is assigned a `subject_idx` pointing into this list.

ii.
```python
all_subjects = []
...
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. The AI noted 19 unique subjects matching the paper. The subject assignment is straightforward from the `mname` field in the experiment info.

## 1-c. How are the data split into sessions?

i. Each unique neural recording file (identified by `mname_datexp_blk`) constitutes one session. When the same neural recording appears under multiple experiment types (e.g., with different `stimtype` values), only the first occurrence is kept. This results in 89 unique sessions.

ii.
```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. The AI reasoned (trajectory step 22-23) that since each neural recording file is unique, each file should be one session. Sessions with `stimtype` variants share the same neural data but have different behavioral annotations; the AI chose to use only the first behavioral variant encountered during iteration over experiment types.

## 1-d. How are the data split into trials?

i. Trials are defined by the `ft_trInd` behavioral variable, which assigns a trial index (0 to ntrials-1) to each neural frame. The AI iterates over trial indices and extracts frames belonging to each trial.

ii.
```python
ntrials = beh['ntrials']
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    if len(valid_frame_indices) < 2:
        continue
```

iii. The AI uses `beh['ntrials']` as the total trial count and `beh['ft_trInd']` for frame-to-trial assignment, both standard variables documented in the data_process_script.ipynb notebook.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring at least 2 valid (VR-moving) frames. Sessions with fewer than 2 valid trials are excluded entirely. No other trial quality filtering is applied.

ii.
```python
if len(valid_frame_indices) < 2:
    continue
...
if valid_count < 2:
    return None
```

iii. The AI set the minimum at 2 frames per trial and 2 trials per session. The reference code does not impose explicit trial quality filters beyond the running-frame requirement (ft_move > 0). The paper states "we only considered timepoints during running for analysis."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `spks` key in the neural data `.npy` files (e.g., `{mname}_{datexp}_{blk}_neural_data.npy`). These contain deconvolved fluorescence traces from Suite2p processing, organized as a list of arrays per imaging plane.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
```

iii. CONVERSION_NOTES.md documents: "Neural data: Suite2p deconvolved fluorescence traces" and "All analyses based on deconvolved fluorescence traces" (quoting the paper). The reference code `utils.load_spk()` uses the same approach.

## 2-b. How is the `neural` data processed?

i. Neural data is loaded by concatenating across imaging planes, then filtered by brain region (excluding neurons with `iarea == -1` or `iarea == 7`). Only frames where `ft_move > 0` (VR is moving) are selected. Per trial, the neural data is extracted as `spk[:, valid_frame_indices]` and cast to float32. No additional normalization, z-scoring, or position interpolation is applied.

ii.
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md that the neuron filter matches the reference code: `(iarea != -1) & (iarea != 7)` excludes non-visual cortex neurons. The reference code's `Get_density_map` function uses the same filter: `idx_neu = (arid!=-1) & (arid != 7)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural data filter is the brain region filter: neurons with `iarea == -1` (unassigned) or `iarea == 7` (non-visual cortex) are excluded. No additional quality filters (e.g., based on signal-to-noise, firing rate, or stimulus selectivity) are applied.

ii.
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
iarea_filtered = iarea[neuron_mask]
```

iii. The AI documented "Neuron curation: Exclude iarea==-1 and iarea==7 (outside visual cortex)" matching the reference code pattern. The reference code uses additional selectivity filters (dprime thresholds) for specific analyses, but these are for figure generation, not for general data loading.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI aligns neural data to the first VR-moving frame of each trial. For each trial, frames are selected where `ft_trInd == trial_idx` AND `ft_move > 0`. The first such frame becomes timepoint 0 (as reflected in `time_since_trial_start` starting at 0).

ii.
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The AI reasoned (trajectory step 24) that "trial start" corresponds to corridor entry, and used the frame-level trial index combined with the running filter as the alignment mechanism.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate of 3.17 Hz, corresponding to ~315.5 ms per frame. No temporal rebinning is applied; each imaging frame becomes one time bin.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. CONVERSION_NOTES.md states "Frame rate: 3.17 Hz" and the metadata stores `time_bin_size: 315.5 ms`. The reference code operates at the native frame rate without rebinning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundFr']` (the frame index when the sound cue was delivered, per trial) and `beh['ft']` (timestamps for each neural frame, in MATLAB datenum format).

ii.
```python
s_fr = sound_fr[trial_idx]
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
else:
    s_fr_int = int(np.floor(s_fr))
    s_fr_frac = s_fr - s_fr_int
    if 0 <= s_fr_int < len(ft) - 1:
        sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
    ...
    time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The AI used `SoundFr` as a fractional frame index and interpolated the corresponding time from `ft` timestamps. It handled NaN SoundFr values by setting time_to_sound to zero.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI interpolates the sound cue time from the fractional `SoundFr` index: it takes the floor of `SoundFr` and linearly interpolates between `ft[floor]` and `ft[floor+1]`. Then for each valid frame in the trial, it computes `(frame_time - sound_time) * 86400` to convert from MATLAB datenum (days) to seconds. When `SoundFr` is NaN, the time-to-sound is set to all zeros.

ii. (See code snippet in 3-a)

iii. The agent's trajectory (step 28) discusses converting frame timestamps to real time using the MATLAB datenum format (days since epoch), multiplied by 86400 to get seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to sound cue is computed at each valid (VR-moving) frame within the trial, using the same frame indices as the neural data. This ensures perfect temporal alignment.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
input_trial[0] = time_to_sound
```

iii. The same `valid_frame_indices` are used for neural, input, and output data, guaranteeing alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from `datexp` (the session date string in `YYYY_MM_DD` format) from the experiment info metadata.

ii.
```python
def compute_day_of_training(sessions):
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((i, s))
    day_of_training = np.zeros(len(sessions))
    for mname, msessions in mouse_sessions.items():
        dates = [(idx, datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))) for idx, s in msessions]
        dates.sort(key=lambda x: x[1])
        first_date = dates[0][1]
        for idx, date in dates:
            day_of_training[idx] = (date - first_date).days
    return day_of_training
```

iii. The AI computes day of training as days since each mouse's first session date. This is per-mouse, starting from 0 for the earliest session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, all session dates are sorted chronologically. The first date becomes day 0, and subsequent sessions are assigned their day offset. The value is constant across all frames within a trial (broadcast as a scalar).

ii.
```python
input_trial[1] = np.float32(day_val)  # broadcast scalar to all timepoints
```

iii. The AI fills the day_of_training row with a constant value for the entire trial, consistent with the instruction specifying it as "continuous, per-trial."

## 4-a (Environment type). What variables in the raw data is `input` *Environment type* derived from?

i. The AI did NOT include an "Environment type" input variable. The decoder instruction specifies only 4 inputs: Time to sound cue, Day of training, Time since trial start, and Reward availability. "Environment type" is not listed as a decoder input.

ii. Not applicable - no code implements this variable.

iii. The AI followed the decoder input specification from the instructions exactly, which does not include "Environment type."

## 4-b (Environment type). What processing is involved in computing `input` *Environment type*?

i. Not implemented. See 4-a (Environment type) above.

ii. Not applicable.

iii. Not applicable.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `beh['ft']` (frame timestamps in MATLAB datenum format) for the valid frames within each trial.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The AI computes elapsed time from the first valid frame of the trial, converting from MATLAB datenum days to seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the time of the first valid (VR-moving) frame is subtracted from all valid frame times, and the result is converted from days to seconds by multiplying by 86400. This produces a monotonically increasing time series starting at 0.

ii. (See code snippet in 5-a)

iii. The AI chose actual elapsed time (which can include pauses when the mouse stops running, since non-running frames are excluded but time still passes) rather than counting frames. This means time gaps appear when the mouse pauses.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned by using the same `valid_frame_indices` for both neural data and time computation, ensuring each time value corresponds exactly to the same neural frame.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
input_trial[2] = time_since_start
```

iii. Same alignment mechanism as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a boolean array indexed by trial, indicating whether each trial is in a rewarded corridor.

ii.
```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The AI directly uses `isRew` per trial, which matches the instruction "1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[trial_idx]` is converted to float32 (1.0 or 0.0) and broadcast as a constant across all timepoints in the trial.

ii. (See code snippet in 6-a)

iii. Straightforward boolean-to-float conversion. Per-trial constant matching the instruction.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which contains the stimulus name (e.g., 'leaf1', 'circle1') for each trial.

ii.
```python
all_stim_names = get_all_stim_names(sessions)  # sorted unique names from all sessions
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
...
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
output_trial[0] = stim_idx  # constant across all timepoints
```

iii. The AI collects all unique wall names across all sessions via `beh['UniqWalls']`, sorts them alphabetically, and assigns integer indices. This results in 15 categories matching the verification output.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each unique stimulus name is assigned an integer index (0-14) based on alphabetical sorting of all stimulus names across all sessions. The index is constant for all timepoints within a trial (per-trial variable). 15 total categories: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

ii. (See code snippet in 7-a)

iii. The AI chose to use all unique stimulus names globally rather than per-session stimulus IDs. The reference code uses `stim_id` fields (0-6) within sessions, but the AI's global naming approach captures the actual stimulus identity across sessions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']`, which contains the neural frame indices at which licks occurred.

ii.
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    if len(lick_fr) == 0:
        return lick_binary
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
    return lick_binary
```

iii. The AI creates a binary lick trace at the neural frame level, marking frames where licks occurred.

## 8-b. What processing is involved in computing `output` *Licking*?

i. `LickFr` values are rounded to integer frame indices and used to set a binary array to 1 at those frames. The binary array covers all neural frames. Then, for each trial, the lick values at the valid (VR-moving) frame indices are extracted.

ii.
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
...
lick_trial = lick_binary[valid_frame_indices]
output_trial[1] = lick_trial
```

iii. The AI handles missing LickFr (using `beh.get('LickFr', np.array([]))`) gracefully, producing all-zero lick traces when no lick data exists.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary array is built for all neural frames, then indexed by the same `valid_frame_indices` used for neural data, ensuring exact frame-level alignment.

ii. (See code snippet in 8-b)

iii. Same alignment mechanism as all other time-varying variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, which contains the VR position (in decimeters) for each neural frame.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The AI uses the per-frame VR position and discretizes it into 4 bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The VR position (in decimeters, range 0-60) is divided by 15 and floored to produce 4 bins covering the full 6m corridor (0-1.5m, 1.5-3m, 3-4.5m, 4.5-6m). Values are clipped to range [0, 3].

ii. (See code snippet in 9-a)

iii. The AI initially tried 4 bins of 10dm (1m) over the 4m texture area but found a highly skewed distribution (50% in bin 3 because grey space frames were included). It then switched to 4 bins of 15dm over the full 6m corridor for even distribution (trajectory steps 64-66).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-width spatial bins of 15 decimeters (1.5 meters) each:
- Bin 0: 0-1.5m (0-15dm)
- Bin 1: 1.5-3m (15-30dm)
- Bin 2: 3-4.5m (30-45dm)
- Bin 3: 4.5-6m (45-60dm)

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The verification output confirms roughly equal distribution: 0-1.5m (25.0%), 1.5-3m (25.1%), 3-4.5m (25.2%), 4.5-6m (24.6%).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted from `ft_Pos` at the same `valid_frame_indices` used for neural data, ensuring per-frame alignment.

ii. (See code snippet in 9-a)

iii. Same alignment mechanism as all other time-varying variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, which contains the running speed for each neural frame.

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The AI uses the per-frame running speed variable from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartile boundaries are computed from all sessions' VR-moving frames. Running speed is then discretized using `np.digitize` with these quartile boundaries into 4 bins (Q1-Q4), each containing approximately 25% of the data.

ii.
```python
def compute_global_speed_quartiles(sessions):
    all_speeds = []
    for i, session in enumerate(sessions):
        ...
        speeds = beh['ft_RunSpeed'][vr_move]
        if len(speeds) > 5000:
            speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
```

iii. The AI subsamples up to 5000 speed values per session for efficiency when computing quartiles, using a fixed random seed for reproducibility.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins based on global quartile boundaries:
- Q1: speed <= 25th percentile
- Q2: 25th < speed <= 50th percentile
- Q3: 50th < speed <= 75th percentile
- Q4: speed > 75th percentile

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The verification output shows roughly equal global distribution: Q1 (23.0%), Q2 (24.2%), Q3 (25.9%), Q4 (26.9%). The slight imbalance arises from subsampling during quartile computation.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted from `ft_RunSpeed` at the same `valid_frame_indices` used for neural data, ensuring per-frame alignment.

ii. (See code snippet in 10-a)

iii. Same alignment mechanism as all other time-varying variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- **Missing LickFr**: Uses `beh.get('LickFr', np.array([]))` to handle sessions without lick data, producing all-zero lick traces.
- **NaN SoundFr**: Sets time_to_sound to all zeros when the sound frame is NaN.
- **SoundFr out of range**: Clamps to first or last frame time when SoundFr is outside the frame array bounds.
- **Trials with too few frames**: Skips trials with fewer than 2 VR-moving frames.
- **Sessions with too few trials**: Skips sessions with fewer than 2 valid trials.
- **Neural/behavior length mismatch**: Truncates behavior arrays to neural frame count: `ft = beh['ft'][:n_frames_neural]`.

ii.
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
...
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
...
ft = beh['ft'][:n_frames_neural]
```

iii. The AI's CONVERSION_NOTES.md notes "Edge cases: Sessions with stimtype handled (use first occurrence)" but does not extensively document other edge case handling.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the neural data files (`load_spk`), which can be several GB each. The AI reported individual sessions taking 5-200+ seconds to load. The full conversion took 73.1 minutes for 89 sessions. Memory management (temp file approach) was also a bottleneck that the AI optimized.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```

iii. Trajectory steps 83-108 document the long loading times and memory issues. The AI implemented a temp-file-per-session approach to avoid memory accumulation, which significantly improved performance (from 188s to 12s per session).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial loop iterates over trials within each session, extracting frame indices and computing per-trial variables. This could potentially be vectorized using `np.split` or groupby operations on `ft_trInd`. The `build_lick_per_frame` function is already reasonably efficient.

ii.
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    ...
```

iii. The trial loop creates a boolean mask for each trial individually, which involves scanning the full frame array `ntrials` times. A single `np.unique(ft_trInd, return_index=True)` or similar approach could identify trial boundaries more efficiently.

## 12-c. What processing does the code repeat multiple times?

i. The code loads behavior data multiple times across different functions:
1. `get_all_stim_names()` loads all behavior files to collect stimulus names.
2. `compute_global_speed_quartiles()` loads all behavior files to compute speed quartiles.
3. `process_session()` loads each session's behavior file again for the actual conversion.

The behavior files are loaded at least 2-3 times each.

ii.
```python
# In compute_global_speed_quartiles:
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
# In get_all_stim_names:
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
# In process_session:
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
```

iii. The AI uses caching within `get_all_stim_names` and `compute_global_speed_quartiles` (using `beh_cache` dict), but doesn't cache across these functions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes global speed quartiles using a subsampling strategy (5000 samples per session) even though the full data is available. The code also saves intermediate session pickle files to a `cache/` directory which are then re-loaded and deleted. Additionally, the `plot_processing` function generates diagnostic plots that are not used in the final dataset. The code loads the full behavior data dictionary for each session but only uses a subset of the variables.

ii.
```python
# Loading entire behavior dict when only a few fields are needed:
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
beh = beh_all[beh_key]
del beh_all
```

iii. The temp-file approach was an optimization for memory management, not unnecessary processing per se. The behavior files contain dozens of variables but only ~10 are used for the conversion.
