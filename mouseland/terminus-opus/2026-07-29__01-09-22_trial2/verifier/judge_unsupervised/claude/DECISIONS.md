# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three sources: (1) `Imaging_Exp_info.npy` for session metadata, (2) per-session neural data files (`{mname}_{datexp}_{blk}_neural_data.npy`) from `data/spk/`, and (3) per-experiment-type behavior files (`Beh_{exp_type}.npy`) from `data/beh/`. Retinotopy data is loaded from `data/retinotopy/{mname}_{datexp}_trans.npz`. Sessions are identified by unique `{mname}_{datexp}_{blk}` keys derived from the experiment info file.

ii. ```python
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

iii. The AI documented this in CONVERSION_NOTES.md Step 1, noting that `load_spk` concatenates spike data across imaging planes (matching the reference code's `load_spk` function in `utils.py`). The session discovery approach iterates through all experiment types in `Imaging_Exp_info.npy`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in the experiment info entries. Each unique `mname` becomes a subject. A list `all_subjects` is built in order of first encounter, and `subject_idx` maps each session to its subject index.

ii. ```python
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that there are 19 unique subjects, matching the paper's report of "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Each unique combination of `{mname}_{datexp}_{blk}` constitutes one session. When the same neural recording appears across multiple experiment types (e.g., different `stimtype` entries), only the first occurrence is kept via the `session_map` deduplication. This yields 89 unique sessions.

ii. ```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. CONVERSION_NOTES.md Step 5: "All 89 sessions included: Each unique neural recording = one session." The deduplication is necessary because the same recording can appear under multiple experiment types.

## 1-d. How are the data split into trials?

i. Trials are defined by the `ft_trInd` variable in the behavior data, which assigns each frame to a trial index (0 to ntrials-1). The code iterates over trial indices from 0 to `ntrials-1` and extracts frames belonging to each trial.

ii. ```python
ntrials = beh['ntrials']
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    if len(valid_frame_indices) < 2:
        continue
```

iii. The AI uses the behavior data's `ntrials` field and `ft_trInd` to identify trial boundaries. NaN values in `ft_trInd` (inter-trial intervals) are naturally excluded since NaN != any integer.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring at least 2 valid (VR-moving) frames. Within each trial, only frames where `ft_move > 0` are included. Sessions with fewer than 2 valid trials are excluded entirely.

ii. ```python
valid_mask = (ft_trInd == trial_idx) & vr_move  # vr_move = ft_move > 0
valid_frame_indices = np.where(valid_mask)[0]
if len(valid_frame_indices) < 2:
    continue
...
if valid_count < 2:
    return None
```

iii. CONVERSION_NOTES.md Step 1: "Frame filter: ft_move > 0 only running/VR-moving frames" and the paper states "only considered timepoints during running." No trials were actually filtered out (all 38,110 trials passed).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `{mname}_{datexp}_{blk}_neural_data.npy` files in `data/spk/`. These contain Suite2p deconvolved fluorescence traces stored under the `'spks'` key, organized as a list of arrays per imaging plane.

ii. ```python
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
```

iii. CONVERSION_NOTES.md Step 3: "Neural data: Deconvolved fluorescence" and "All analyses based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. Neural data from multiple imaging planes is concatenated along the neuron axis. Neurons are filtered to include only those in visual cortex areas (excluding `iarea == -1` and `iarea == 7`). Per-trial neural data is extracted by selecting frames that are both within the trial and VR-moving (`ft_move > 0`). The data is cast to float32.

ii. ```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 1: "Neuron filter: (iarea != -1) & (iarea != 7) excludes non-visual cortex neurons" matching reference code. No additional normalization, z-scoring, or position interpolation is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered based on retinotopy area assignment: neurons with `iarea == -1` (no retinotopic map) or `iarea == 7` (outside visual cortex) are excluded. No additional quality filtering (e.g., by firing rate, signal-to-noise, or d-prime selectivity) is applied.

ii. ```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
```

iii. CONVERSION_NOTES.md Step 10: "Neuron filtering: Same as reference (iarea!=-1 & iarea!=7)." The reference code also has `Get_dprime_selective_neuron` for selectivity-based filtering, but this was not applied as it appears to be for specific analyses rather than general data loading.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry). For each trial, frames are selected where `ft_trInd == trial_idx` AND `ft_move > 0`. The first valid frame of each trial is effectively the trial start. There is no fixed temporal offset or windowing; trial length varies based on the number of valid frames.

ii. ```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5: alignment event is "Trial start (corridor entry)." The metadata confirms `off_start: 0.0` and `off_end: None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate of 3.17 Hz, corresponding to ~315.5 ms per time bin. No temporal rebinning is applied.

ii. ```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. CONVERSION_NOTES.md Step 3: "Frame rate: 3.17 Hz" matching the paper. The metadata stores `time_bin_size: 315.5 ms`.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (sound cue frame number per trial) and `ft` (frame timestamps in MATLAB datenum format).

ii. ```python
s_fr = sound_fr[trial_idx]
...
sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "SoundFr, ft -> input[0]: time_to_sound_cue, Time from frame to sound cue (seconds)."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound frame `SoundFr[trial_idx]` is a fractional frame number. The code interpolates to find the exact sound time by splitting into integer and fractional parts, then linearly interpolating between adjacent frame timestamps. The time difference between each frame's timestamp and the sound time is computed and converted from days to seconds by multiplying by 86400.

ii. ```python
s_fr_int = int(np.floor(s_fr))
s_fr_frac = s_fr - s_fr_int
if 0 <= s_fr_int < len(ft) - 1:
    sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
elif s_fr_int >= len(ft) - 1:
    sound_time = ft[-1]
else:
    sound_time = ft[0]
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. For NaN sound frames, time_to_sound is set to zeros. The computation produces negative values before the cue and positive after. The range across sessions is [-723.5, 1767.7] seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to sound cue is computed at the same frame indices as the neural data (the valid VR-moving frames), so they are inherently aligned frame-by-frame.

ii. ```python
ft_trial = ft[valid_frame_indices]
...
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
input_trial[0] = time_to_sound
```

iii. The same `valid_frame_indices` are used for both neural and input extraction, ensuring temporal alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from `datexp` (date of experiment) in the session metadata, which is formatted as `YYYY_MM_DD`.

ii. ```python
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
```

iii. CONVERSION_NOTES.md Step 5: "datexp -> input[1]: day_of_training, Days since first session per mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted chronologically. The day of training is the number of days since that mouse's first recording session. This produces a per-trial scalar (same value for all timepoints within a trial and session). Values range from 0 to 92 days.

ii. ```python
first_date = dates[0][1]
for idx, date in dates:
    day_of_training[idx] = (date - first_date).days
...
input_trial[1] = np.float32(day_val)  # broadcast scalar to all timepoints
```

iii. The computation is per-mouse, so day 0 is different for each mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `ft` (frame timestamps) for the valid frames within each trial.

ii. ```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5: "ft -> input[2]: time_since_trial_start, Elapsed time from trial start."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first valid frame's timestamp is subtracted from all frame timestamps in the trial, then multiplied by 86400 to convert from MATLAB datenum (days) to seconds. This produces a time series starting at 0 and increasing throughout the trial.

ii. ```python
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. Range across sessions: [0.0, 1769.4] seconds. Most trials have much shorter durations (typical ~10s at 3.17 Hz with ~33 frames).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same `valid_frame_indices` as neural data, ensuring frame-by-frame alignment.

ii. ```python
ft_trial = ft[valid_frame_indices]  # same indices as neural data
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
input_trial[2] = time_since_start
```

iii. Alignment is inherent since the same frame selection is used for all data streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `isRew` (boolean array per trial indicating whether the corridor is rewarded).

ii. ```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. CONVERSION_NOTES.md Step 5: "isRew -> input[3]: reward_availability, 1=rewarded, 0=not."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A simple boolean-to-float conversion: 1.0 if the trial is rewarded, 0.0 if not. This is a per-trial scalar broadcast to all timepoints. The verification output shows many sessions have reward_availability [0.0, 0.0] (no rewarded trials), with some having [0.0, 1.0] (mix of rewarded and unrewarded).

ii. ```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The AI correctly identified `isRew` as the reward indicator. The distribution pattern (many all-zero sessions) reflects the experimental design where unsupervised/naive conditions have no reward.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `WallName` (per-trial string indicating the wall texture) and `UniqWalls` (list of unique wall names across all sessions).

ii. ```python
def get_all_stim_names(sessions):
    all_stim = set()
    beh_cache = {}
    for session in sessions:
        ...
        for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
            all_stim.add(str(wn))
    return sorted(all_stim)

stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. CONVERSION_NOTES.md Step 5: "WallName -> output[0]: visual_stimulus_category, Map to index." 15 unique stimuli identified.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique wall names across all sessions are collected, sorted alphabetically, and assigned integer indices. Each trial gets a single integer index that is broadcast to all timepoints. The 15 categories are: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

ii. ```python
all_stim_names = get_all_stim_names(all_sessions)  # sorted alphabetically
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
output_trial[0] = stim_idx  # broadcast scalar to all timepoints
```

iii. This is a per-trial value; the same stimulus index is assigned to every timepoint in a trial.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` (array of fractional frame indices where licks occurred).

ii. ```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
```

iii. CONVERSION_NOTES.md Step 5: "LickFr -> output[1]: licking, Binary per frame."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame indices (fractional) are rounded to nearest integer and used to set corresponding elements in a binary array to 1. Out-of-range indices are clipped. If `LickFr` is empty (no licks), the entire array is zeros.

ii. ```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    if len(lick_fr) == 0:
        return lick_binary
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
    return lick_binary
```

iii. The verification output shows very sparse licking: 97.3% no_lick, 2.7% lick overall. Many sessions have 100% no_lick (unsupervised/naive conditions without lick training).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick array is computed for all frames in the session, then indexed at the same `valid_frame_indices` as neural data.

ii. ```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
...
lick_trial = lick_binary[valid_frame_indices]
output_trial[1] = lick_trial
```

iii. Alignment is ensured by using the same frame indices for licking and neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos` (position in the corridor in decimeters, range 0-60).

ii. ```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. CONVERSION_NOTES.md Step 5: "ft_Pos -> output[2]: position_bin, 4 bins of 15dm over 60dm."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values (in decimeters, 0-60) are divided by 15.0 and floored to create 4 bins over the full 6m corridor (0-1.5m, 1.5-3m, 3-4.5m, 4.5-6m). Values are clipped to [0, 3].

ii. ```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
output_trial[2] = pos_bins
```

iii. The output_values labels confirm: `['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m']`. The distribution is approximately uniform (~25% each).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. 4 bins of 1.5m each over the full 6m corridor: bin 0 = [0, 1.5m), bin 1 = [1.5, 3m), bin 2 = [3, 4.5m), bin 3 = [4.5, 6m].

ii. ```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The instructions specify "4 equal-length, 1-m-long spatial bins." The corridor texture is 4m (40dm) with 2m (20dm) grey. The AI chose to bin the full 6m corridor into 4 x 1.5m bins rather than binning only the 4m texture region into 4 x 1m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted at the same `valid_frame_indices` used for neural data.

ii. ```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. Same frame-level alignment as all other variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed` (running speed per frame).

ii. ```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. CONVERSION_NOTES.md Step 5: "ft_RunSpeed -> output[3]: running_speed_bin, 4 quartile bins."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartile boundaries are computed across all sessions (only for frames where `ft_move > 0`). Each session subsamples up to 5000 speed values for efficiency. The quartiles are then used to bin all running speed values into 4 bins using `np.digitize`.

ii. ```python
def compute_global_speed_quartiles(sessions):
    all_speeds = []
    for i, session in enumerate(sessions):
        ...
        vr_move = beh['ft_move'] > 0
        speeds = beh['ft_RunSpeed'][vr_move]
        if len(speeds) > 5000:
            speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
```

iii. Speed quartile boundaries: [11.79, 24.43, 40.44]. Distribution is approximately 25% per bin (Q1: 23.0%, Q2: 24.2%, Q3: 25.9%, Q4: 26.9%).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. 4 bins based on global quartiles: Q1 (speed < 11.79), Q2 (11.79-24.43), Q3 (24.43-40.44), Q4 (> 40.44). `np.digitize` assigns bin indices 0-3.

ii. ```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The instructions specify "4 bins, each corresponding to 25% of the data." The global quartile approach achieves this approximately.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted at the same `valid_frame_indices` used for neural data.

ii. ```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. Same frame-level alignment as all other variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Missing lick data**: If `LickFr` is absent or empty, all-zeros array is used.
- **NaN sound frames**: If `SoundFr` is NaN, time_to_sound is set to zeros for that trial.
- **Sound frame out of range**: Clamped to first or last frame timestamp.
- **Short trials**: Trials with fewer than 2 valid frames are skipped.
- **Sessions with too few trials**: Sessions with fewer than 2 valid trials are excluded.
- **Behavior/neural length mismatch**: Behavior arrays are truncated to match neural data length (`[:n_frames_neural]`).
- **NaN in ft_trInd**: Handled implicitly since NaN != any integer in the mask comparison.

ii. ```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
...
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
...
ft = beh['ft'][:n_frames_neural]
ft_trInd = beh['ft_trInd'][:n_frames_neural]
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Step 10, noting "Sessions with stimtype handled (use first occurrence)."

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural data files is the most time-consuming step, taking 4-15 seconds per session. The full conversion took 73.1 minutes for 89 sessions. The neural data files are large (each containing tens of thousands of neurons across thousands of frames).

ii. From the conversion output:
```
[1/89] DR10_2022_07_12_1 (unsup_train1_before_learning)
  Neurons: 58224 -> 52246 (8.3s load)
  Trials: 503/503 (3.7s process)
```

iii. CONVERSION_NOTES.md Step 7: "Run Time: 179.6s for 2 sessions." The loading step dominates, as it involves reading large `.npy` files and concatenating arrays.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level processing loop iterates over each trial sequentially, extracting frame indices and computing per-trial variables. This could potentially be vectorized using `np.split` or `np.unique`-based grouping to process all trials at once.

ii. ```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    ...
```

iii. However, since trials have variable lengths, full vectorization is complex. The trial loop is relatively fast compared to data loading (3-6s processing vs 4-15s loading per session).

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded multiple times:
1. Once during `get_all_stim_names()` to collect stimulus names
2. Once during `compute_global_speed_quartiles()` to compute quartiles
3. Once per session during `process_session()` for actual data extraction

The full behavior files are loaded each time without caching across these three phases.

ii. ```python
# In get_all_stim_names:
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
# In compute_global_speed_quartiles:
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
# In process_session:
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
```

iii. Each of these functions maintains its own local cache within the function, but the cache is not shared between functions. Behavior files are thus loaded 3 times each.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads full behavior dictionaries (containing many fields like VRpos, VRposCum, VRposTime, etc.) when only a subset of fields is needed. Additionally, the `plot_processing` function generates diagnostic plots that are not used in the final output. The session data is saved to temporary pickle files and then re-loaded for combining, adding unnecessary I/O.

ii. ```python
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
beh = beh_all[beh_key]
del beh_all  # loads entire file then deletes most of it

# Temp file round-trip
temp_fn = f'cache/session_{i:03d}.pkl'
with open(temp_fn, 'wb') as f:
    pickle.dump({...}, f, protocol=4)
...
with open(tf, 'rb') as f:
    sd = pickle.load(f)
```

iii. The temp file approach was chosen for memory management (to avoid holding all sessions in memory simultaneously), which is a reasonable trade-off for a 211 GB dataset but adds I/O overhead.
