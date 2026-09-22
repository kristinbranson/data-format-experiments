# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories under `data`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. It reads the master index `Imaging_Exp_info.npy` to enumerate all unique recordings, then loads behavior files per experiment type (with caching), spike files per session, and retinotopy files per mouse/date.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# ...
spk = load_spk(mname, datexp, blk)
# in load_spk:
data = np.load(fn, allow_pickle=True).item()
return np.concatenate(data['spks'], 0)
# retinotopy:
iarea = load_retino(mname, datexp)
# in load_retino:
return np.load(fn, allow_pickle=True)['iarea']
```

iii. The AI uses the same data sources and loading approach as described in the reference code. Behavior files are cached to avoid re-reading.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info entries. A subject-to-index mapping is built dynamically as sessions are processed. 19 subjects are found.

ii.
```python
mname = result['mname']
if mname not in subject_to_idx:
    subject_to_idx[mname] = len(subjects)
    subjects.append(mname)
```

iii. The AI identifies subjects from the same field (`mname`) as the reference. Subject ordering depends on iteration order rather than being sorted.

## 1-c. How are the data split into sessions?

i. A session is identified by the composite key `{mname}_{datexp}_{blk}`. Duplicate recordings across experiment types are deduplicated by keeping only the first occurrence. 89 unique sessions are found.

ii.
```python
def get_all_unique_recordings():
    exp_info = np.load(...)
    seen = set()
    recordings = []
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in seen:
                seen.add(key)
                recordings.append((key, s, exp_type))
    return recordings
```

iii. The deduplication logic matches the reference approach.

## 1-d. How are the data split into trials?

i. Trials are defined by iterating over trial indices 0 to `ntrials-1`. For each trial, frames are selected where `ft_trInd == trial` AND `ft_CorrSpc` (corridor space) AND `ft_move > 0` (running). The running mask filter is an additional condition not present in the reference solution.

ii.
```python
trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
trial_frames = np.where(trial_mask)[0]
```
where:
```python
corridor_mask = ft_CorrSpc.copy()
running_mask = beh['ft_move'][:n_use] > 0
```

iii. The AI applies a running filter (`ft_move > 0`) citing the reference paper's statement "We only considered timepoints during running." The reference solution does NOT apply this filter, keeping all corridor frames regardless of whether the mouse is running.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 frames (after applying corridor and running masks) are dropped. No upper bound on trial length is applied.

ii.
```python
if len(trial_frames) < 2:
    continue
```

iii. The AI's CONVERSION_NOTES do not discuss filtering long outlier trials. The reference solution applies a 99th percentile trial length limit to remove extremely long trials where mice stopped mid-corridor.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/{session_id}_neural_data.npy`, which contains deconvolved calcium traces per imaging plane. Planes are concatenated into a single neurons-by-frames matrix.

ii.
```python
data = np.load(fn, allow_pickle=True).item()
return np.concatenate(data['spks'], 0)
```

iii. Same source as reference.

## 2-b. How is the `neural` data processed?

i. The neural data is extracted for each trial's frames and stored as float32. No additional processing (dF/F, deconvolution) is applied since the data is already deconvolved.

ii.
```python
neural = spk[:, trial_frames].astype(np.float32)
```

iii. The AI correctly recognizes that the data is already deconvolved. Uses float32 instead of float16 (reference uses float16).

## 2-c. How is the `neural` data filtered based on quality controls?

i. ALL neurons are included, including those in unassigned brain regions. The AI defines 5 brain regions (V1, mHV, aHV, lHV, unassigned) and keeps all neurons regardless of their region assignment. The reference filters to only 4 visual areas (V1, mHV, lHV, aHV), dropping unassigned neurons.

ii.
```python
BRAIN_REGION_NAMES = ['V1', 'mHV', 'aHV', 'lHV', 'unassigned']

def iarea_to_region_idx(iarea):
    region_idx = np.full(len(iarea), 4, dtype=np.int64)  # 4 = unassigned
    for ia_code, reg_idx in IAREA_TO_REGION.items():
        region_idx[iarea == ia_code] = reg_idx
    return region_idx
```
No filtering step removes unassigned neurons.

iii. The AI's CONVERSION_NOTES state "All neurons included - no d-prime filtering" but do not mention that unassigned neurons outside the four visual areas should be dropped.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (trial start). Each trial's neural data starts at the first frame satisfying the corridor + running mask conditions and runs to the last such frame within that trial.

ii.
```python
trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
trial_frames = np.where(trial_mask)[0]
neural = spk[:, trial_frames].astype(np.float32)
```

iii. The alignment event is corridor entry, matching the reference. However, the running mask means not all contiguous frames from corridor entry are included.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed as the median inter-frame interval converted to milliseconds: approximately 314.69 ms (~3.18 Hz).

ii.
```python
dt = float(np.median(np.diff(ft)) * 86400)
# ...
median_dt_ms = float(np.median(dt_values) * 1000)
```

iii. The reference uses a fixed constant `FRAME_RATE = 3.17` giving ~315.46 ms. The AI computes it empirically from the data, arriving at a very similar value.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and `trial_frames` (the frame indices of each trial). A constant `dt` (median frame interval in seconds) is used for time conversion.

ii.
```python
sound_fr = beh['SoundFr']
# ...
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. The AI uses frame index differences multiplied by a constant dt, rather than interpolating actual frame timestamps as the reference does.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(trial_frames - sound_fr[trial]) * dt`, which gives the time elapsed since the sound cue (negative before cue, positive after). The sign convention is **reversed** compared to the reference, which computes `cue_time - frame_time` (positive before cue, negative after). The instruction name "Time to sound cue" implies the reference convention (time remaining until the cue).

ii.
```python
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. The AI uses a constant dt approximation rather than proper time interpolation, and has the sign reversed relative to the reference's convention for "time TO sound cue."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Computed from the same `trial_frames` indices used for the neural data, so alignment is inherent.

ii.
```python
trial_frames = np.where(trial_mask)[0]
neural = spk[:, trial_frames].astype(np.float32)
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. Alignment is correct by construction.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `days` or `sess#` fields in the experiment info entries (`db_info`), not from counting sessions per mouse chronologically.

ii.
```python
def get_training_day(db_info):
    if 'days' in db_info:
        return float(db_info['days'])
    if 'sess#' in db_info:
        return float(db_info['sess#'])
    return 0.0
```

iii. The AI reads a pre-existing field from the data rather than computing the training day from session ordering. This produces a range of 0-15, while the reference counts sessions per mouse giving 0-7.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The value is read directly from `db_info['days']` or `db_info['sess#']` with no further processing. It is broadcast as a constant across all time bins of each trial.

ii.
```python
training_day = get_training_day(db_info)
# ...
day_arr = np.full(n_t, training_day, dtype=np.float32)
```

iii. The reference computes training day by counting the number of prior recorded sessions for each mouse (0-indexed). The AI's approach yields different numerical values.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `trial_frames` (the frame indices surviving corridor + running filtering) and a constant `dt`.

ii.
```python
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. The reference uses `StartFr` (the actual trial start frame) and `ft` (frame timestamps) with interpolation. The AI uses the first surviving frame index instead of the actual start frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(trial_frames - trial_frames[0]) * dt`, where `trial_frames[0]` is the first frame passing the corridor + running filter, and dt is the median frame interval. This starts at exactly 0 for the first surviving frame.

ii.
```python
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. Two differences from reference: (1) uses constant dt rather than actual frame timestamps, and (2) starts from the first surviving frame after running-mask filtering rather than the actual corridor entry frame (StartFr). Since running filtering can remove the first few frames, the alignment origin may differ.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same `trial_frames` used for neural data, so alignment is inherent.

ii.
```python
trial_frames = np.where(trial_mask)[0]
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. Alignment is correct by construction.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
is_rew = beh['isRew']
# ...
reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value for the trial is converted to float (0.0 or 1.0) and broadcast across all time bins.

ii.
```python
reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)
```

iii. Matches the reference approach.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
wall_name = beh['WallName']
# ...
stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
```

iii. Same source as reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each of the 15 individual wall texture names (e.g., circle1, circle2, leaf1_swap1) to a unique index (0-14), resulting in 15 output categories. The reference maps the 15 names to 4 broad categories (circle, leaf, rock, wood).

ii.
```python
ALL_STIMULI = sorted(['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2',
                      'leaf2', 'leaf3', 'rock1', 'rock2', 'wood1', 'wood1_swap1',
                      'wood1_swap2', 'wood2', 'wood5'])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
# ...
stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
stim_arr = np.full(n_t, stim_idx, dtype=np.int64)
```

iii. The instructions say "Visual stimulus category. e.g. circle, leaf, etc." which implies the 4 broad categories used by the reference. The AI's use of 15 individual stimuli does not match this specification.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame number of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
# Pre-compute lick lookup: trial -> list of frame indices
lick_by_trial = defaultdict(list)
for li in range(len(lick_fr)):
    lick_by_trial[int(lick_trind[li])].append(lick_fr[li])
```

iii. The reference uses only `LickFr` (not `LickTrind`) and creates a session-wide binary lick array.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI finds the closest trial frame to each lick frame (from that trial's lick list) and marks it as 1 if within 1.0 frame distance. Because trial frames are filtered by the running mask, licks on non-running frames may be missed.

ii.
```python
lick_binary = np.zeros(n_t, dtype=np.int64)
if trial in lick_by_trial:
    for lf in lick_by_trial[trial]:
        diffs = np.abs(trial_frames - lf)
        closest = np.argmin(diffs)
        if diffs[closest] < 1.0:
            lick_binary[closest] = 1
```

iii. The reference creates a simpler session-wide binary array by truncating `LickFr` to int and directly indexing: `licking[lick_frame.astype(int)] = 1`, then extracts per-trial with `licking[window]`. The AI's approach is more complex but achieves a similar result for licks that fall on running frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick events are mapped to the same `trial_frames` used for neural data, so alignment is inherent.

ii.
```python
diffs = np.abs(trial_frames - lf)
```

iii. Alignment is correct by construction.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position along the corridor at each imaging frame, in decimeters.

ii.
```python
ft_Pos = beh['ft_Pos'][:n_use]
# ...
pos = ft_Pos[trial_frames]
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position (in decimeters) is divided by 10 and floored to get meter bins, then clipped to 0-3.

ii.
```python
pos = ft_Pos[trial_frames]
pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Matches the reference's approach: `np.clip(beh['ft_Pos'][:n_frames] // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four 1-meter bins: 0-1m, 1-2m, 2-3m, 3-4m. Position values are divided by 10 (converting decimeters to meters) and floored.

ii.
```python
pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Matches reference discretization.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted for the same `trial_frames` used for neural data.

ii.
```python
pos = ft_Pos[trial_frames]
```

iii. Alignment is correct by construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_use]
# ...
speed = ft_RunSpeed[trial_frames]
```

iii. Same source as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartile boundaries across ALL recordings (using running corridor frames), then applies `np.digitize` with those fixed boundaries to assign each frame to a speed bin. The reference computes rank-based quartiles per session (using only kept trial frames), ensuring exactly 25% per bin within each session.

ii.
```python
def compute_running_speed_quartiles(recordings):
    # collects all running corridor frame speeds globally
    all_speeds = np.concatenate(all_speeds)
    bins = np.percentile(all_speeds, [25, 50, 75])
    return bins

# Per trial:
speed_binned = np.clip(np.digitize(speed, speed_bins), 0, 3).astype(np.int64)
```

iii. The global quantile approach means individual sessions can have very uneven speed bin distributions. The verification output confirms some sessions have >95% of frames in one speed bin. The reference's per-session rank-based approach guarantees 25% per bin per session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed values are assigned to bins 0-3 using `np.digitize` with three global percentile boundaries (Q25=12.39, Q50=25.33, Q75=40.83 cm/s).

ii.
```python
speed_binned = np.clip(np.digitize(speed, speed_bins), 0, 3).astype(np.int64)
```

iii. The reference uses a rank-based split per session that is immune to tied values at zero speed.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted for the same `trial_frames` used for neural data.

ii.
```python
speed = ft_RunSpeed[trial_frames]
```

iii. Alignment is correct by construction.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The number of usable frames is capped at `min(n_frames_neural, len(beh['ft']))` to handle cases where behavior data extends beyond imaging data. Missing retinotopy files result in all neurons being assigned to 'unassigned'. Sessions with fewer than 2 valid trials are skipped. Exceptions during session processing are caught and the session is skipped.

ii.
```python
n_use = min(n_frames, len(beh['ft']))
# ...
iarea = load_retino(mname, datexp)
brain_region_idx = iarea_to_region_idx(iarea) if iarea is not None else np.full(n_neurons, 4, dtype=np.int64)
# ...
if len(neural_trials) < 2:
    return None
```

iii. The reference handles the behavior/imaging mismatch similarly. The AI adds exception handling for robustness.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files is the bottleneck, as each file can be several GB. The AI reports load times of 1-7 seconds per session.

ii.
```python
spk = load_spk(mname, datexp, blk)  # t_load reported per session
```

iii. Consistent with reference assessment.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick assignment loop iterates per-lick per-trial, computing distances to all trial frames for each lick. This could be vectorized.

ii.
```python
for lf in lick_by_trial[trial]:
    diffs = np.abs(trial_frames - lf)
    closest = np.argmin(diffs)
    if diffs[closest] < 1.0:
        lick_binary[closest] = 1
```

iii. The reference avoids this entirely by creating a session-wide binary lick array with vectorized indexing.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded twice: once to compute global speed quartiles (`compute_running_speed_quartiles`), and again during session processing. The cache is cleared between these passes.

ii.
```python
speed_bins = compute_running_speed_quartiles(all_recordings)
_beh_cache.clear()
# ... later, behavior is loaded again during process_session calls
```

iii. The reference avoids this by computing speed quartiles per session during the single processing pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Including unassigned neurons (585,641 across all sessions) adds significant data that is outside the four visual areas studied in the paper and may add noise rather than signal for decoding. Additionally, the global speed quartile computation over all recordings is an extra pass that could be avoided.

ii.
```python
BRAIN_REGION_NAMES = ['V1', 'mHV', 'aHV', 'lHV', 'unassigned']
# unassigned neurons are included in neural data
```

iii. The reference drops neurons outside the four visual areas, keeping only relevant neurons.
