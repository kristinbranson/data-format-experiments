# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, which maps experiment types to lists of session dicts. It iterates over all experiment types and sessions to build a `session_meta` dictionary keyed by `{mname}_{datexp}_{blk}`. Neural data is loaded from individual `*_neural_data.npy` files in `data/spk/`. Behavioral data is loaded from `Beh_{exp_type}.npy` files in `data/beh/`. Retinotopy data is loaded from `*_trans.npz` files in `data/retinotopy/`.

ii.
```python
def load_experiment_info():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_meta = {}
    for exp_type, sessions in exp_info.items():
        for db in sessions:
            sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            ...
            session_meta[sid] = { ... }

def load_spk(mname, datexp, blk):
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    return spk

def load_beh(session_id, beh_files, beh_keys):
    for beh_file in beh_files:
        beh_all = np.load(beh_path, allow_pickle=True).item()
        if session_id in beh_all:
            return beh_all[session_id]
        for key in beh_keys:
            if key in beh_all:
                return beh_all[key]
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that the reference code uses `load_spk()` from `utils.py` which concatenates spks across imaging planes, and `load_exp_beh()` for behavioral data. The AI's approach matches the reference code's loading pattern. The handling of `stimtype` suffix in behavioral data keys matches the reference code pattern `'%s_%s_%s_%s'%(ndb['mname'], ndb['datexp'], ndb['blk'], ndb['stimtype'])`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in the experiment info. The AI collects unique mouse names from all sessions and creates a sorted list. A `subject_to_idx` mapping is created. Each session is associated with its subject via `session_subject_map`.

ii.
```python
subjects_set = set()
for sid in session_ids:
    subjects_set.add(session_meta[sid]['mname'])
subjects_list = sorted(subjects_set)
subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
```

iii. The AI noted 19 unique subjects matching the paper's "19 mice". The approach is straightforward extraction from metadata.

## 1-c. How are the data split into sessions?

i. Each unique combination of `mname_datexp_blk` is treated as one session. The AI deduplicates sessions that appear under multiple experiment types in `Imaging_Exp_info.npy`, resulting in 89 unique sessions. All sessions from all experiment types are included.

ii.
```python
session_meta = {}
for exp_type, sessions in exp_info.items():
    for db in sessions:
        sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        if sid not in session_meta:
            session_meta[sid] = { ... }
        session_meta[sid]['exp_types'].append(exp_type)
```

iii. CONVERSION_NOTES.md documents 89 sessions matching the paper's "89 recordings in 19 mice". The deduplication logic ensures each physical recording session appears only once.

## 1-d. How are the data split into trials?

i. Trials are split using the `ft_trInd` behavioral variable, which assigns each neural frame to a trial index. For each trial index from 0 to `ntrials-1`, the AI selects frames where `ft_trInd == trial_idx`.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
    if len(frame_indices) < 2:
        skipped += 1
        continue
```

iii. This matches the reference code pattern of using `ft_trInd` for frame-to-trial mapping, as seen in functions like `Get_coding_direction`.

## 1-e. How are trials filtered based on quality controls?

i. The AI only filters out trials with fewer than 2 frames. No filtering based on running status, corridor vs. gray space, or other quality criteria is applied at the trial level.

ii.
```python
if len(frame_indices) < 2:
    skipped += 1
    continue
frame_indices = frame_indices[frame_indices < n_frames_spk]
if len(frame_indices) < 2:
    skipped += 1
    continue
```

iii. CONVERSION_NOTES.md states "Trial curation rules: Include all trials. Skip trials with < 2 frames." The reference paper states "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The reference code extensively uses `VRmove = beh['ft_move'][:nfr]>0` to filter to running-only frames. The AI does NOT implement this filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `spks` key in the `*_neural_data.npy` files. These are deconvolved fluorescence traces from Suite2p processing.

ii.
```python
def load_spk(mname, datexp, blk):
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    return spk
```

iii. CONVERSION_NOTES.md correctly notes: "Neural data is deconvolved fluorescence traces (Suite2p output), NOT raw dF/F" and "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. Neural data is concatenated across imaging planes, filtered to visual cortex neurons (excluding `iarea == -1` and `iarea == 7`), then subsampled to a maximum of 2000 neurons per session using stratified sampling by brain region. The data is then sliced per trial using frame indices from `ft_trInd` and cast to float32.

ii.
```python
def filter_and_subsample_neurons(spk, iarea, max_neurons=MAX_NEURONS, rng=None):
    valid_mask = (iarea != -1) & (iarea != 7)
    valid_indices = np.where(valid_mask)[0]
    if n_valid <= max_neurons:
        selected_indices = valid_indices
    else:
        # Stratified subsampling by brain region
        for region in BRAIN_REGIONS:
            n_sample = max(1, int(np.round(max_neurons * n_region / total_in_regions)))
            sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
            ...

# Per trial:
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The AI justified subsampling in CONVERSION_NOTES.md: "Max 2000 neurons/session, stratified by brain region. Justified because decoder uses PCA to 100 components and SVD with max 2000 neurons." The reference code does NOT subsample neurons; it uses all valid neurons within visual cortex.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by brain region assignment: neurons with `iarea == -1` (unassigned) or `iarea == 7` (outside visual cortex) are excluded. No other quality filtering (e.g., based on firing rate, signal quality, or selectivity) is applied.

ii.
```python
valid_mask = (iarea != -1) & (iarea != 7)
```

iii. This matches the reference code's neuron filtering: `(arid!=-1) & (arid != 7)` seen in functions like `Get_density_map` and `Get_dprime_selective_neuron`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by selecting frames where `ft_trInd == trial_idx`. The `ft_trInd` variable assigns each neural frame to its corresponding trial, with the first frame of each trial corresponding to corridor entry. `off_start` is set to 0.0.

ii.
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. CONVERSION_NOTES.md states: "Trial alignment: Align to corridor entry (trial start). Use ft_trInd for frame-to-trial mapping." The instructions specify "Temporally aligned based on trial start (corridor entry)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate of 3.17 Hz, yielding ~315.5 ms time bins. No temporal rebinning is applied.

ii.
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

iii. CONVERSION_NOTES.md confirms: "Frame rate: 3.17 Hz (time bin ~315 ms)" and "Time bin: Native frame rate (~315 ms). No resampling." This matches the paper's stated frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` (the neural frame index when the sound cue was delivered) and the frame indices of the current trial.

ii.
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. The AI derives this from `beh['SoundFr']`, which is documented in the data as "neural frame when sound cue was delivered."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the time to sound cue is computed as `(SoundFr - frame_index) / FS`, giving time in seconds. Positive values mean the sound cue is in the future, negative values mean it has already occurred.

ii.
```python
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. This produces a time-varying signal as required by the instructions ("continuous, time-varying"). The sign convention is: positive = sound cue is ahead, negative = sound cue has passed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is computed using the same `frame_indices` used for the neural data, ensuring frame-by-frame alignment. Each frame in the neural data has a corresponding time-to-sound-cue value.

ii.
```python
frame_indices = np.where(trial_mask)[0]
trial_neural = spk[:, frame_indices].astype(np.float32)
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. Both neural and input data use the same `frame_indices`, ensuring temporal alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the `days` or `sess#` field in the experiment info metadata (`Imaging_Exp_info.npy`).

ii.
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
...
day_of_training = float(meta['day_of_training'])
trial_input[1, :] = day_of_training
```

iii. CONVERSION_NOTES.md mapping table shows: "days/sess# -> input[1]: day_of_training, Direct value."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day value is extracted directly from the experiment metadata and broadcast as a constant across all timepoints in each trial. It is converted to float. The range in the converted data is [0, 15].

ii.
```python
trial_input[1, :] = day_of_training  # broadcast per-trial
```

iii. The instructions specify "continuous, per-trial" which is implemented by broadcasting the scalar value.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does NOT include "Environment type" as a decoder input. The experiment type information (`exptype`, `rewType`) is available in the session metadata but is not used as a decoder input variable.

ii. No code for this - not implemented.

iii. "Environment type" is not listed in the Decoder Inputs specification in the instructions, so the AI's decision to omit it is consistent with the task requirements.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not applicable - not included as a decoder input.

ii. No code for this.

iii. See 4-a above.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` (the neural frame when the animal entered each corridor) and the frame indices of the current trial.

ii.
```python
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. The AI derives this from `beh['StartFr']`, documented as "neural frame when animal enter each corridor."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame in a trial, the time since trial start is computed as `(frame_index - StartFr) / FS`, giving time in seconds. Values start near 0 and increase as the trial progresses.

ii.
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. This produces a time-varying signal as required. The range in the data is [0, 1773] seconds, with very large values indicating trials where the mouse stopped running for extended periods.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame indices are used for both neural and input data, ensuring alignment.

ii.
```python
frame_indices = np.where(trial_mask)[0]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. Both are indexed from the same `frame_indices` array.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the `isRew` boolean variable in the behavioral data, which indicates whether each trial occurs in a rewarded corridor.

ii.
```python
is_rew = beh['isRew']
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. CONVERSION_NOTES.md mapping: "isRew -> input[3]: reward_availability, Boolean to 0/1."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value for the trial is converted to float (0.0 or 1.0) and broadcast across all timepoints in the trial.

ii.
```python
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The instructions specify "discrete, per-trial" which is implemented by broadcasting the scalar value.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` (the stimulus name for each trial), `UniqWalls` (unique stimulus names in the session), and `stim_id` (numerical ID mapping stimuli to canonical categories).

ii.
```python
wall_name = beh['WallName']
uniq_walls = beh['UniqWalls']
stim_id_map = beh.get('stim_id', None)
STIM_NAMES = {0: 'circle1', 1: 'circle2', 2: 'leaf1', 3: 'leaf2', 4: 'leaf3', 5: 'leaf1_swap1', 6: 'leaf1_swap2'}
# Maps WallName -> stim_id -> canonical name
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
```

iii. The AI uses the `stim_id` mapping from the reference code to canonicalize stimulus names (e.g., mapping numeric IDs to names like 'circle1', 'leaf1').

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's wall name is mapped through `stim_id` to a canonical stimulus name (e.g., 'leaf1', 'circle1'). A global mapping from stimulus names to integer indices is built across all sessions. The integer index is broadcast across all timepoints in the trial. The final dataset has 12 unique stimulus categories.

ii.
```python
def build_stimulus_mapping(all_output_trials):
    all_stim_names = set()
    for session_outputs in all_output_trials:
        for _, stim_name in session_outputs:
            all_stim_names.add(stim_name)
    sorted_stim = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(sorted_stim)}
    return stim_to_idx, sorted_stim

# Per trial:
trial_output[0, :] = stim_idx  # broadcast per-trial
```

iii. The instructions specify "per-trial" and "e.g. circle1, leaf2, etc." The AI correctly implements per-trial categorical encoding.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` (neural frame indices of lick events) and `LickTrind` (trial index of each lick event).

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
```

iii. CONVERSION_NOTES.md mapping: "LickFr, LickTrind -> output[1]: licking, Binary per frame."

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events are identified by filtering `LickTrind == trial_idx`. The corresponding `LickFr` values are rounded and matched to the trial's frame indices. A binary array is created: 1 where a lick frame matches a trial frame, 0 otherwise.

ii.
```python
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
lick_binary = np.zeros(n_tp, dtype=np.int64)
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
    elif pos > 0 and frame_indices[pos-1] == lf_int:
        lick_binary[pos-1] = 1
```

iii. The lick fraction is very low (2.7% of timepoints have licking), which is partly because at 3.17 Hz frame rate, individual lick events are sparse. The reference code's `lickCount()` function returns a per-trial binary lick response, not per-frame. The AI creates a per-frame binary signal instead.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames (`LickFr`) are matched to the trial's `frame_indices` using `searchsorted` to find the corresponding neural frame. This ensures lick events are placed at the correct timepoint in the neural data.

ii.
```python
pos = np.searchsorted(frame_indices, lf_int)
if pos < n_tp and frame_indices[pos] == lf_int:
    lick_binary[pos] = 1
```

iii. The alignment relies on the fact that both neural data and lick events are indexed by neural frame number.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, which gives the position inside the VR corridor for each neural frame.

ii.
```python
ft_Pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_Pos[frame_indices]
```

iii. CONVERSION_NOTES.md mapping: "ft_Pos -> output[2]: position_bin, 4 bins of 10 VR units."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position values (in VR units, range 0-60) are clipped to [0, 39.999] and divided into 4 bins of 10 VR units each. Each VR unit represents 10 cm (since the 4m corridor = 40 VR units).

ii.
```python
trial_pos = ft_Pos[frame_indices]
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The instructions specify "4 equal-length, 1-m-long spatial bins" for the 4m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 4 bins: bin 0 = 0-1m (0-10 VR units), bin 1 = 1-2m (10-20), bin 2 = 2-3m (20-30), bin 3 = 3-4m (30-40). Positions beyond 40 VR units (gray space, 4-6m) are clipped to bin 3.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The bin 3 (3-4m) contains 48.6% of data, much more than the expected ~25%. This is because gray space frames (positions 40-60, i.e., 4-6m) are clipped into bin 3, inflating its count. Frames in gray space should ideally not be included in corridor position bins, or should be assigned a separate category.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position values are extracted using the same `frame_indices` as the neural data, ensuring temporal alignment.

ii.
```python
trial_pos = ft_Pos[frame_indices]
```

iii. Both neural and position data use the same `frame_indices` array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`, which gives the running speed for each neural frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_RunSpeed[frame_indices]
```

iii. CONVERSION_NOTES.md mapping: "ft_RunSpeed -> output[3]: running_speed_bin, 4 global quartile bins."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed quartiles are computed globally across all sessions' `ft_RunSpeed` values (all frames, including non-running frames and gray space). The quartile boundaries at the 25th, 50th, and 75th percentiles are used to bin per-trial speeds.

ii.
```python
def compute_running_speed_quartiles(session_meta, session_ids):
    all_speeds = []
    for sid in session_ids:
        beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
```

iii. The instructions specify "4 bins, each corresponding to 25% of the data." However, the output distribution shows Q1=9.2%, Q2=40.9%, Q3=25.0%, Q4=24.9%, which does not match the expected ~25% each. This is because quartiles are computed on ALL frames (including non-running and gray space) but the output only includes frames assigned to trials via `ft_trInd`.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed values are binned using `np.digitize` with the global quartile boundaries: bin 0 = below 25th percentile, bin 1 = 25th-50th, bin 2 = 50th-75th, bin 3 = above 75th.

ii.
```python
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The clipping to [0, 3] is redundant since `np.digitize` with 3 boundaries produces values in [0, 3].

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed values are extracted using the same `frame_indices` as neural data, ensuring temporal alignment.

ii.
```python
trial_speed = ft_RunSpeed[frame_indices]
```

iii. Both use the same `frame_indices`.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- Frame count mismatch between neural and behavioral data: uses `min(n_frames_spk, n_frames_beh)`
- Trials with fewer than 2 frames: skipped
- Missing `stimtype` field: falls back to session ID without suffix
- NaN `stim_id` values: uses raw wall name
- Missing behavioral data keys: tries multiple key formats

ii.
```python
n_frames = min(n_frames_spk, n_frames_beh)
ft_trInd = beh['ft_trInd'][:n_frames]
...
if len(frame_indices) < 2:
    skipped += 1
    continue
...
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
```

iii. CONVERSION_NOTES.md Step 10 documents several edge cases handled, including stimtype suffix handling and NaN stim_id fallback. However, the code does not handle NaN values in `ft_trInd` (frames outside behavioral recording), which could lead to incorrect trial assignments.

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural data files is the most time-consuming step, taking 1-30 seconds per session depending on file size. The full conversion takes ~18 minutes for 89 sessions, with neural data loading dominating.

ii.
```python
# Timing logged:
print(f"    Raw: {n_neurons_raw} neurons x {n_frames_spk} frames ({time.time()-t_load:.1f}s)")
```

iii. CONVERSION_NOTES.md Step 7 estimates: "Load neural data: 1-30s (varies by file size), ~15 min total."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick binary computation uses a Python for-loop over individual lick events, which could be vectorized using numpy array operations.

ii.
```python
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
```

iii. This loop iterates over individual lick events per trial. A vectorized approach using `np.searchsorted` on the entire array of lick frames would be more efficient.

## 12-c. What processing does the code repeat multiple times?

i. The behavioral data is loaded twice for each session: once during `compute_running_speed_quartiles()` and once during `process_session()`. This doubles the I/O for behavioral data files.

ii.
```python
# First load in compute_running_speed_quartiles:
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
speeds = beh['ft_RunSpeed']

# Second load in process_session:
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

iii. Caching the behavioral data from the first pass would avoid redundant file I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The neuron subsampling to 2000 neurons per session is unnecessary processing that reduces the data. The reference code uses all valid visual cortex neurons, and the decoder's PCA step would handle dimensionality reduction. Additionally, the `--show-processing` plotting code creates concatenated arrays across trials which is memory-intensive and only used for visualization.

ii.
```python
MAX_NEURONS = 2000
def filter_and_subsample_neurons(spk, iarea, max_neurons=MAX_NEURONS, rng=None):
    ...
    if n_valid <= max_neurons:
        selected_indices = valid_indices
    else:
        # Stratified subsampling
        ...
```

iii. The subsampling reduces data from 20,000-90,000 neurons to 2,000, which discards significant neural information that the decoder could potentially use.
