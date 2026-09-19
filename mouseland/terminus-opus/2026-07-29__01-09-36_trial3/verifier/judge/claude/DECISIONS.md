# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` (behavior), `spk/` (neural), and `retinotopy/` (brain regions). It reads `Imaging_Exp_info.npy` as the master index to enumerate all sessions. For each session, it loads the neural data file, retinotopy, and behavioral data separately. Behavioral data is loaded per-session by trying multiple behavior files and keys, rather than reading each behavior file once for all sessions it contains.

ii. Loading experiment info and building session metadata:
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
session_meta = {}
for exp_type, sessions in exp_info.items():
    for db in sessions:
        sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        if sid not in session_meta:
            session_meta[sid] = { ... }
```

Loading neural and behavioral data per session:
```python
spk = load_spk(mname, datexp, blk)  # np.concatenate(spk_data['spks'], axis=0)
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
iarea = load_retinotopy(mname, datexp)
```

iii. The AI documented that data is organized by experiment type in the master index, with 89 unique sessions across 19 mice. The session metadata is collected in a single pass over the experiment info.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) is extracted from the experiment info for each session. All unique mouse names are sorted to form the subjects list, and a mapping from mouse name to index is built.

ii.
```python
subjects_set = set()
for sid in session_ids:
    subjects_set.add(session_meta[sid]['mname'])
subjects_list = sorted(subjects_set)
subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
```

iii. The AI noted 19 unique mice across 89 sessions, consistent with the paper.

## 1-c. How are the data split into sessions?

i. A session is identified by the composite key `mname_datexp_blk`. The AI deduplicates sessions by only storing the first occurrence in `session_meta` when a session appears under multiple experiment types.

ii.
```python
sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if sid not in session_meta:
    session_meta[sid] = { ... }
```

iii. The AI noted 89 unique sessions matching the paper's count.

## 1-d. How are the data split into trials?

i. The AI splits trials using `ft_trInd`, which labels each frame with its trial index. For each trial, ALL frames with that trial index are included -- the AI does NOT filter by `ft_CorrSpc` (corridor space flag). This means frames in the gray space between corridors are included.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
```

iii. The AI's CONVERSION_NOTES documented using `ft_trInd` for frame-to-trial mapping. No mention of filtering by corridor space.

## 1-e. How are trials filtered based on quality controls?

i. The AI only skips trials with fewer than 2 frames. It does NOT filter out excessively long trials (e.g., mice that stopped mid-corridor for extended periods).

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

iii. The AI's CONVERSION_NOTES state "Include all trials. Skip trials with < 2 frames."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spks` in `spk/<session_id>_neural_data.npy`, which contains deconvolved calcium traces as a list of arrays per imaging plane. These are concatenated across planes. The brain region of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
iarea = load_retinotopy(mname, datexp)
```

iii. The AI correctly identified these as Suite2p-processed deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI filters neurons to visual cortex areas and then SUBSAMPLES to a maximum of 2000 neurons per session, stratified by brain region. This is a significant deviation from the reference, which keeps all neurons in visual areas.

ii.
```python
MAX_NEURONS = 2000

def filter_and_subsample_neurons(spk, iarea, max_neurons=MAX_NEURONS, rng=None):
    valid_mask = (iarea != -1) & (iarea != 7)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    if n_valid <= max_neurons:
        selected_indices = valid_indices
    else:
        # stratified subsampling by brain region
        for region in BRAIN_REGIONS:
            n_sample = max(1, int(np.round(max_neurons * n_region / total_in_regions)))
            sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
            ...
```

iii. The AI justified this by noting the decoder uses PCA to 100 components and SVD with max 2000 neurons, so subsampling shouldn't hurt. This is documented in CONVERSION_NOTES Step 5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside visual cortex are filtered out by excluding `iarea == -1` and `iarea == 7`. After filtering, neurons are subsampled to max 2000 per session.

ii.
```python
valid_mask = (iarea != -1) & (iarea != 7)
```

iii. The AI noted this matches the reference code's `(arid!=-1) & (arid != 7)` filter. However, the reference actually filters by positive membership in the four known visual areas (V1, mHV, lHV, aHV), which is functionally equivalent for this dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial start (corridor entry). The AI uses all frames with the matching trial index, starting from the first frame of each trial. Neural data is extracted as `spk[:, frame_indices]`.

ii.
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The alignment event is corridor entry, consistent with the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of 3.17 Hz is used, giving a time bin of ~315.5 ms.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

iii. The AI correctly identified that the imaging frame is the native resolution and no resampling is needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (the frame number of the sound cue for each trial) and the trial's `frame_indices` (the frame numbers in the trial).

ii.
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The AI identified SoundFr as the source for the sound cue timing.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes time to sound cue as `(SoundFr - frame_index) / FS`, converting a frame difference directly to seconds by dividing by the frame rate. This assumes uniform frame spacing. The reference instead interpolates frame times from the `ft` timestamp array and computes real time differences.

ii.
```python
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The AI documented computing time_to_sound_cue in seconds, positive before the cue and negative after.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The same `frame_indices` are used for both neural data and the time_to_sound_cue computation, so they are aligned.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. Aligned by shared frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives day_of_training from the `days` or `sess#` field in the experiment info entries.

ii.
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
session_meta[sid]['day_of_training'] = day_val
```

iii. The AI documented using the experiment info's `days`/`sess#` field directly.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI directly uses the `days` or `sess#` value from the experiment info with no further processing. This is broadcast across all time bins of each trial. The reference instead counts the number of prior recording sessions for each mouse (0-indexed), giving values from 0 to 7.

ii.
```python
day_of_training = float(meta['day_of_training'])
trial_input[1, :] = day_of_training
```

iii. The verification output shows day_of_training values like 0, 1, 2, 6, 7, 8, 9, 10, 12, 13, 15 -- these are non-contiguous and go up to 15, unlike the reference which ranges 0-7. This indicates the AI is using actual day counts rather than session indices.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `StartFr` (the corridor entry frame for each trial) and the trial's `frame_indices`.

ii.
```python
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The AI identified StartFr as the corridor entry frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes `(frame_index - StartFr) / FS`, converting frame differences to seconds by dividing by frame rate. Same approach as time_to_sound_cue -- assumes uniform frame spacing rather than using actual timestamps from `ft`.

ii.
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The AI documented this as computing time in seconds since trial start.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same `frame_indices` as the neural data, so aligned.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. Aligned by shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
is_rew = beh['isRew']
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The AI correctly identified isRew as the source.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (0.0 or 1.0) and broadcast across all time bins of the trial. No further processing.

ii.
```python
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. Straightforward binary per-trial variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives visual stimulus from `WallName` and `stim_id` fields. It uses a mapping through `stim_id` values when available, falling back to `WallName` directly.

ii.
```python
STIM_NAMES = {0: 'circle1', 1: 'circle2', 2: 'leaf1', 3: 'leaf2', 4: 'leaf3', 5: 'leaf1_swap1', 6: 'leaf1_swap2'}
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
```

iii. The AI documented using stim_id when available and wall name otherwise.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps wall names to individual texture names (e.g., circle1, circle2, leaf1, leaf2, wood2, etc.), resulting in 12 distinct stimulus categories. The reference groups these into 4 broad categories (circle, leaf, rock, wood) using a hardcoded mapping. The instruction says "Visual stimulus category. e.g. circle, leaf, etc." suggesting the broad categories.

ii.
```python
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
# Results in 12 categories: ['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1',
#   'leaf1_swap2', 'leaf2', 'leaf3', 'rock2', 'wood1_swap2', 'wood2', 'wood5']
```

iii. The AI's output shows 12 stimulus categories with very uneven distributions (e.g., rock2=0.3%, wood5=0.7% vs circle1=32.9%, leaf1=34.3%).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` (lick frame numbers) and `LickTrind` (lick trial indices).

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
```

iii. The AI uses both LickFr and LickTrind to associate licks with trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds each lick frame to the nearest integer, then uses searchsorted to find the matching frame in the trial's frame indices. If a match is found, that time bin is marked as 1. The reference instead creates a session-wide binary lick array by truncating LickFr to int and setting those positions to 1, then indexes by the trial's window.

ii.
```python
lick_binary = np.zeros(n_tp, dtype=np.int64)
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
    elif pos > 0 and frame_indices[pos-1] == lf_int:
        lick_binary[pos-1] = 1
```

iii. The approach is more complex but functionally similar, though using `round` instead of `int` truncation could occasionally place a lick in a different frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary is computed for the same `frame_indices` used for the neural data.

ii.
```python
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
# matched against frame_indices
```

iii. Aligned by shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos`, the position at each imaging frame (in decimeters, 0-60).

ii.
```python
trial_pos = ft_Pos[frame_indices]
```

iii. The AI correctly identified ft_Pos as the source.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips position to [0, 39.999] then divides by 10 and floors to get 4 bins of 1m each.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. This is similar to the reference approach. However, since the AI includes gray space frames (ft_Pos > 40), those get clipped to bin 3 (3-4m), inflating the proportion in that bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four 1-meter bins: 0-1m, 1-2m, 2-3m, 3-4m. Position values above 40 dm (gray space) are clipped to bin 3.

ii.
```python
pos_bin_names = ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The verification output shows position distribution: 0-1m=19.4%, 1-2m=15.9%, 2-3m=16.1%, 3-4m=48.6%. The 3-4m bin is heavily overrepresented because gray space frames are included and clipped to this bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same `frame_indices` as the neural data.

ii.
```python
trial_pos = ft_Pos[frame_indices]
```

iii. Aligned by shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The AI correctly identified ft_RunSpeed as the source.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes GLOBAL speed quartiles across all sessions using `np.percentile([25, 50, 75])`, then uses `np.digitize` to bin each frame's speed. The reference computes PER-SESSION rank-based quartiles, ensuring each bin contains exactly 25% of frames within that session.

ii.
```python
def compute_running_speed_quartiles(session_meta, session_ids):
    all_speeds = []
    for sid in session_ids:
        beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
        all_speeds.append(beh['ft_RunSpeed'])
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles

speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
```

iii. The AI documented using global quartile bins. The resulting speed distribution is very uneven: Q1=9.2%, Q2=40.9%, Q3=25.0%, Q4=24.9%, showing the global percentile approach doesn't produce equal bins per session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins based on global speed quartile boundaries: [0, 9.67, 31.46] cm/s. Bin assignment via `np.digitize`.

ii.
```python
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The very uneven distribution (Q1=9.2% vs Q2=40.9%) is problematic -- the instruction says "4 bins, each corresponding to 25% of the data."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same `frame_indices` as the neural data.

ii.
```python
trial_speed = ft_RunSpeed[frame_indices]
```

iii. Aligned by shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI clips all behavioral arrays to the minimum of neural and behavioral frame counts. Trials with fewer than 2 frames after clipping are skipped. If a session fails to process entirely, it is caught by a try/except and skipped.

ii.
```python
n_frames = min(n_frames_spk, n_frames_beh)
ft_trInd = beh['ft_trInd'][:n_frames]
```
The reference does the same clipping:
```python
n_frames = spikes.shape[1]
frame_time = (beh['ft'][:n_frames] - beh['ft'][0]) * SEC_PER_DAY
```

iii. The AI documented handling frame count mismatches between neural and behavioral data.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files (spk files), which are large numpy arrays. The full conversion takes ~1081 seconds, dominated by file I/O.

ii.
```python
spk = load_spk(mname, datexp, blk)  # 14s per large session
```

iii. The AI logged per-session timing, showing neural data loading as the main bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking computation loops over individual lick frames using Python for-loop and searchsorted, which could be vectorized. The trial processing loop also reloads behavioral data per session rather than batching.

ii.
```python
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    ...
```

iii. The reference vectorizes licking by creating a session-wide binary array in one step.

## 12-c. What processing does the code repeat multiple times?

i. The AI reloads behavioral data files multiple times -- once during `compute_running_speed_quartiles()` for all sessions, and again during `process_session()` for each session. Each behavioral file is loaded independently per session rather than once per file.

ii.
```python
# In compute_running_speed_quartiles:
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
# Then again in process_session:
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

iii. The reference loads each behavior file once and processes all sessions it contains together.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes gray space frames in every trial (frames outside the corridor texture). These frames contain position values that get clipped to bin 3, inflating that bin. The reference excludes these frames using `ft_CorrSpc`, so they are unnecessary data.

ii. The AI does not filter by `ft_CorrSpc`:
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
```

iii. Including gray space frames adds unnecessary data that distorts the position distribution.
