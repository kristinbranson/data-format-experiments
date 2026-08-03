# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, which maps experiment types to lists of session dicts. It iterates over all experiment types and builds a `session_meta` dict keyed by `{mname}_{datexp}_{blk}`. Neural data is loaded from `data/spk/{mname}_{datexp}_{blk}_neural_data.npy` via `load_spk()`, which concatenates across imaging planes. Behavioral data is loaded from `data/beh/Beh_{exp_type}.npy` files via `load_beh()`, which tries multiple key formats. Retinotopy is loaded from `data/retinotopy/{mname}_{datexp}_trans.npz`.

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

def load_spk(mname, datexp, blk):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
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

iii. The AI documented in CONVERSION_NOTES.md that the reference code `load_spk()` concatenates `spks` arrays across imaging planes, and that behavioral data uses different key formats depending on whether `stimtype` is present. The approach matches the reference code's `load_spk(db, root)` and `load_exp_beh(root, exp_type)` functions.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field from experiment info. A unique list of mouse names is built across all sessions. Each session is mapped to a subject index via `subject_to_idx`.

ii.
```python
subjects_set = set()
for sid in session_ids:
    subjects_set.add(session_meta[sid]['mname'])
subjects_list = sorted(subjects_set)
subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
```

iii. The AI identified 19 unique mouse names, matching the paper's statement of "19 mice." Subject identifiers come from the experiment metadata's `mname` field.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique combinations of `{mname}_{datexp}_{blk}` from the experiment info. All 89 unique sessions across all experiment types (supervised, unsupervised, naive, grating) are included. A session may appear in multiple experiment types but is only processed once.

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

iii. The AI noted 142 total entries across experiment types but only 89 unique sessions (matching "89 recordings" from the paper). Some sessions appear in multiple experiment types but are deduplicated.

## 1-d. How are the data split into trials?

i. Trials are split using the `ft_trInd` behavioral variable, which assigns each imaging frame to a trial index. For each trial index from 0 to `ntrials-1`, frames matching that index are extracted.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
```

iii. The AI documented that `ft_trInd` maps frames to trial indices, consistent with the reference code's approach. Total of 38,110 trials across all sessions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by a minimum frame count criterion: trials with fewer than 2 frames are skipped. Frame indices beyond the neural data range are also excluded. No other trial quality filtering is applied (no filtering by running status, corridor space, or behavioral criteria).

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

iii. The CONVERSION_NOTES state: "Trial curation rules: Include all trials. Skip trials with < 2 frames." The reference paper states "We only considered timepoints during running for analysis" but this was not applied as a trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key in the neural data files (`{mname}_{datexp}_{blk}_neural_data.npy`). These are deconvolved fluorescence traces from Suite2p processing.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

iii. The AI noted in CONVERSION_NOTES: "Neural data is deconvolved fluorescence traces (Suite2p output), NOT raw dF/F" and "All our analyses were based on deconvolved fluorescence traces" from the paper.

## 2-b. How is the `neural` data processed?

i. Neural data is concatenated across imaging planes, then neurons outside visual cortex are filtered out (iarea==-1 or iarea==7). If more than 2000 neurons remain after filtering, they are subsampled to 2000 using stratified sampling by brain region. Per-trial neural data is extracted by indexing columns with frame indices and cast to float32. No further processing (e.g., normalization, z-scoring) is applied.

ii.
```python
def filter_and_subsample_neurons(spk, iarea, max_neurons=MAX_NEURONS, rng=None):
    valid_mask = (iarea != -1) & (iarea != 7)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    if n_valid <= max_neurons:
        selected_indices = valid_indices
    else:
        # Stratified subsampling by brain region
        ...
    return spk[selected_indices], iarea[selected_indices], selected_indices

# Per trial:
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The AI justified subsampling to 2000 neurons because "decoder uses PCA to 100 components and SVD with max 2000 neurons." The reference code does not subsample neurons; it uses all neurons in visual cortex.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` or `iarea == 7` are excluded (those outside visual cortex regions). No other neuron-level quality filtering is applied (e.g., no minimum firing rate, no signal-to-noise filtering).

ii.
```python
valid_mask = (iarea != -1) & (iarea != 7)
```

iii. The AI states this matches the reference code filter: `(arid!=-1) & (arid != 7)`. This is confirmed by the reference code's `neu_area_ID()` function which only defines regions for iarea values 0-6, 8, 9.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by extracting frames where `ft_trInd == trial_idx`. The first frame of each trial corresponds to the trial start. All frames belonging to the trial (including gray space) are included.

ii.
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI uses `ft_trInd` to extract frame indices for each trial, which aligns to corridor entry. The metadata sets `off_start: 0.0`, confirming alignment to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of 3.17 Hz, corresponding to ~315.5 ms time bins. No temporal rebinning is applied.

ii.
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

iii. The AI noted: "Frame rate: 3.17 Hz (time bin ~315 ms)" and "Time bin: Native frame rate (~315 ms). No resampling." This matches the reference paper's frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (sound cue frame index per trial) and the per-frame indices within each trial.

ii.
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The AI planned this mapping in CONVERSION_NOTES Step 5: "SoundFr - frame_idx | input[0]: time_to_sound_cue | (SoundFr - frame) / FS".

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Computed as `(SoundFr - frame_index) / FS`, expressed in seconds. This gives positive values before the sound cue (when frame < SoundFr) and negative values after. No clipping or normalization is applied.

ii.
```python
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. The AI chose the sign convention where positive = before cue, negative = after cue. The range observed is [-1771.1, 725.0] seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both use the same `frame_indices` array (from `ft_trInd == trial_idx`), so the time-to-sound-cue value at each timepoint corresponds exactly to the same frame as the neural data.

ii.
```python
# Same frame_indices used for both:
trial_neural = spk[:, frame_indices].astype(np.float32)
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. Consistent frame indexing ensures alignment. The AI verified this in sanity checks: "time_to_sound_cue: Expected 2.5546s, got 2.5546s (np.allclose passed)."

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `days` or `sess#` field in the experiment info metadata for each session.

ii.
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
...
day_of_training = float(meta['day_of_training'])
trial_input[1, :] = day_of_training
```

iii. The AI noted in Step 5 mapping: "days/sess# | input[1]: day_of_training | Direct value | exp_info." The range is 0-15 across all sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The raw integer value is converted to float and broadcast as a constant across all timepoints of every trial in that session. No normalization or transformation is applied.

ii.
```python
day_of_training = float(meta['day_of_training'])
trial_input[1, :] = day_of_training
```

iii. Day of training is per-session metadata, so it's constant within a session. The AI broadcasts it to all timepoints to match the (n_input, n_timepoints) format.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `StartFr` (start frame per trial) and the per-frame indices within each trial.

ii.
```python
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The AI mapped: "frame_idx - StartFr | input[2]: time_since_trial_start | (frame - StartFr) / FS."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(frame_index - StartFr) / FS` in seconds. This gives 0 at trial start and increases over time. No clipping is applied.

ii.
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. The range is [0.0, 1773.0] seconds, with some very long trials (max 5621 frames = ~1773 seconds). These extreme values suggest some trials include extended gray space or non-running periods.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same `frame_indices` array is used for both neural data and time-since-start computation, ensuring perfect frame-level alignment.

ii.
```python
trial_neural = spk[:, frame_indices]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The AI verified: "time_since_trial_start: Expected 0.0641s, got 0.0641s (np.allclose passed)."

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from the `isRew` array in the behavioral data, which is a per-trial indicator of whether the trial occurs in a rewarded corridor.

ii.
```python
is_rew = beh['isRew']
...
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The AI mapped: "isRew | input[3]: reward_availability | Boolean to 0/1."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[trial_idx]` is cast to float (0.0 or 1.0) and broadcast as a constant across all timepoints of the trial.

ii.
```python
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The verification output shows reward_availability range is [0.0, 1.0] per session, with many sessions being all-0 (unrewarded cohorts) and some having mixed values.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `WallName` (per-trial wall texture name), `stim_id` (mapping from wall to standard stimulus ID), and `UniqWalls` (unique wall names). The `stim_id` mapping converts wall names to standard categories (circle1, leaf1, etc.).

ii.
```python
uniq_walls = beh['UniqWalls']
stim_id_map = beh.get('stim_id', None)
wall_to_stim = {}
if stim_id_map is not None:
    for i, wall in enumerate(uniq_walls):
        if i < len(stim_id_map):
            sid_val = stim_id_map[i]
            if not np.isnan(sid_val):
                wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
            else:
                wall_to_stim[wall] = wall
        else:
            wall_to_stim[wall] = wall
```

iii. The AI used the `stim_id` mapping from the data notebook: "0=circle1, 1=circle2, 2=leaf1, 3=leaf2, 4=leaf3, 5=leaf1_swap1, 6=leaf1_swap2." When stim_id is NaN, the raw wall name is used directly, resulting in names like "circle3", "rock2", "wood1_swap2", etc.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are mapped to standard stimulus names via `stim_id`, then a global categorical index is assigned. The stimulus category is constant for the entire trial (broadcast to all timepoints). After collecting all sessions, a sorted list of all unique stimulus names is built and each is assigned an integer index.

ii.
```python
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
trial_output[0, :] = -1  # placeholder, filled later

# After all sessions:
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
for output_arr, stim_name in session_outputs:
    stim_idx = stim_to_idx[stim_name]
    output_arr[0, :] = stim_idx
```

iii. This results in 12 stimulus categories: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock2, wood1_swap2, wood2, wood5. The distribution is heavily dominated by circle1 (32.9%) and leaf1 (34.3%).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` (frame indices of lick events) and `LickTrind` (trial indices for each lick event).

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
```

iii. The AI documented: "LickFr, LickTrind | output[1]: licking | Binary per frame | lickCount()."

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events are identified by matching `LickTrind == trial_idx`. Each lick frame is matched to the nearest imaging frame using `np.searchsorted`. A binary array (0=no lick, 1=lick) is constructed per imaging frame.

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

iii. The licking is very sparse: only 2.7% of all timepoints have lick=1. The matching approach requires exact frame index matches, which means many lick events may not be matched (lick events between imaging frames are lost).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick events are matched to imaging frame indices via `np.searchsorted` and exact index comparison. The resulting binary array has the same length as the neural data timepoints for each trial.

ii.
```python
# lf_int is the rounded lick frame
pos = np.searchsorted(frame_indices, lf_int)
if pos < n_tp and frame_indices[pos] == lf_int:
    lick_binary[pos] = 1
```

iii. The approach requires exact frame matching. At 3.17 Hz (~315ms bins), lick events occurring between frames will only be captured if they happen to fall on an exact frame index. A more robust approach would assign any lick event to the nearest frame or bin lick events within each inter-frame interval.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos`, which gives the VR position (0-60 units) at each imaging frame.

ii.
```python
ft_Pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_Pos[frame_indices]
```

iii. The AI mapped: "ft_Pos | output[2]: position_bin | 4 bins of 10 VR units."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values are clipped to [0, 39.999] (the 4m corridor range in VR units), then divided by 10 and floored to produce bin indices 0-3.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The corridor is 4m long (40 VR units). Each bin represents 1m (10 VR units), matching the instruction for "4 equal-length, 1-m-long spatial bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided into 4 bins: 0-1m (bin 0), 1-2m (bin 1), 2-3m (bin 2), 3-4m (bin 3). Positions in gray space (40-60 VR units) are clipped to 39.999 and assigned to bin 3. This causes bin 3 to be overrepresented (48.6% of all frames).

ii.
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The 4 bins correspond to: 0-1m (19.4%), 1-2m (15.9%), 2-3m (16.1%), 3-4m (48.6%). The heavy skew toward bin 3 is because gray space frames (positions 40-60) are all clipped to bin 3. The instructions only specify "4 equal-length, 1-m-long spatial bins" for the corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position values are extracted from `ft_Pos` using the same `frame_indices` as neural data, ensuring frame-level alignment.

ii.
```python
trial_pos = ft_Pos[frame_indices]
```

iii. Same frame indexing as neural data ensures alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed`, which gives the running speed at each imaging frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
...
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The AI mapped: "ft_RunSpeed | output[3]: running_speed_bin | 4 global quartile bins."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global quartile boundaries are computed across all sessions by concatenating all `ft_RunSpeed` values. Then per-frame speeds are binned using `np.digitize`. The quartile boundaries are [0.0, 9.67, 31.46].

ii.
```python
def compute_running_speed_quartiles(session_meta, session_ids):
    all_speeds = []
    for sid in session_ids:
        ...
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles

# Per trial:
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The AI noted that speed quartiles were computed globally. However, the 25th percentile is at 0.0 (many stationary frames), causing highly uneven bins: Q1=9.2%, Q2=40.9%, Q3=25.0%, Q4=24.9%.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 4 bins using global quartile boundaries: Q1 (speed <= 0), Q2 (0 < speed <= 9.67), Q3 (9.67 < speed <= 31.46), Q4 (speed > 31.46). The bins do not contain equal amounts of data due to many speeds at exactly 0.

ii.
```python
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The instructions say "4 bins, each corresponding to 25% of the data." The actual distribution is highly uneven (Q1=9.2%, Q2=40.9%, Q3=25.0%, Q4=24.9%), violating the equal-quartile requirement. This is because the 25th percentile falls at 0.0 and many frames have speed=0 or negative.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are extracted from `ft_RunSpeed` using the same `frame_indices` as neural data, ensuring frame-level alignment.

ii.
```python
trial_speed = ft_RunSpeed[frame_indices]
```

iii. Same frame indexing as neural data ensures alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Frame count mismatch between neural and behavioral data: use minimum of both. (2) Trials with < 2 frames: skipped. (3) Missing stim_id (NaN): use raw wall name. (4) Stimtype suffix in behavioral data keys: try multiple key formats. (5) Neural frame indices beyond behavioral data range: clipped.

ii.
```python
n_frames = min(n_frames_spk, n_frames_beh)
ft_trInd = beh['ft_trInd'][:n_frames]

if len(frame_indices) < 2:
    skipped += 1
    continue

frame_indices = frame_indices[frame_indices < n_frames_spk]

# stim_id NaN handling:
if not np.isnan(sid_val):
    wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
else:
    wall_to_stim[wall] = wall
```

iii. The AI documented handling of stimtype suffix issues and NaN stim_id values in CONVERSION_NOTES Step 10.

## 12-a. What are the most time-consuming steps of the code?

i. Neural data loading is by far the most time-consuming step, taking 1-30 seconds per session depending on file size. The full conversion takes approximately 18 minutes, dominated by loading 89 large neural data files (each 20k-90k neurons).

ii. From conversion output:
```
[1/89] DR10_2022_07_12_1
  Loading neural data...
    Raw: 58224 neurons x 31707 frames (14.1s)
```

iii. The AI noted in Step 7: "Load neural data: 1-30s (varies by file size), estimated 15 min total. Full conversion: ~18 min actual."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking binary computation uses a Python loop over individual lick frames with `np.searchsorted` inside:

ii.
```python
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
    elif pos > 0 and frame_indices[pos-1] == lf_int:
        lick_binary[pos-1] = 1
```

iii. This loop could be vectorized by calling `np.searchsorted` on the entire array of lick frames at once, then using boolean indexing to set the lick_binary values. However, the number of lick events per trial is small, so the performance impact is minimal.

## 12-c. What processing does the code repeat multiple times?

i. Behavioral data is loaded twice: once during `compute_running_speed_quartiles()` (which loads all sessions to compute global quartiles) and again during `process_session()`. This means every behavioral .npy file is loaded and parsed twice.

ii.
```python
def compute_running_speed_quartiles(session_meta, session_ids):
    for sid in session_ids:
        meta = session_meta[sid]
        beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)

# Then later, in process_session:
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

iii. The behavioral data files are loaded from disk twice for every session. This could be optimized by caching the behavioral data or computing quartiles during the main processing loop.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `ft_CorrSpc` array is loaded but never used. (2) The `stim_id_map` lookup with NaN handling creates stimulus names like "circle3", "rock2", "wood1_swap2", "wood2", "wood5" that may not be meaningful. (3) Neuron subsampling to 2000 per session is unnecessary since the decoder already applies PCA dimensionality reduction. (4) The `show_processing` plotting code includes a plot for `ic[1]` (day_of_training), which is constant per session and not informative.

ii.
```python
ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]  # loaded but unused
```

iii. The AI loaded `ft_CorrSpc` (corridor space indicator) but never used it for filtering or processing. This variable could have been useful for excluding gray space frames from position binning.
