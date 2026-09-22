# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories under `/app/data`: `beh/` for behavior, `spk/` for neural (deconvolved calcium) traces, and `retinotopy/` for visual area assignments. It first reads `Imaging_Exp_info.npy` as a master index, then for each session loads the behavior file (`Beh_{exp_type}.npy`), neural data (`{mname}_{datexp}_{blk}_neural_data.npy`), and retinotopy (`{mname}_{datexp}_trans.npz`). The behavior file is re-read for every session via `get_session_beh()` rather than being cached per file.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# Per session:
spk = load_spk(mname, datexp, blk)  # np.concatenate(dat['spks'], axis=0)
iarea = load_retino(mname, datexp)   # dat['iarea']
beh, exp_type, db_entry = get_session_beh(key, exp_info)  # loads Beh_{exp_type}.npy each time
```

iii. The AI identified the same three data sources (behavior, spikes, retinotopy) and master index as the reference. The behavior loading is inefficient (re-reads file per session) but functionally correct.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info entries. The AI tracks unique subjects via a `subjects_seen` dictionary during processing, assigning each new mouse name an incremental index. 19 unique mice are found.

ii.
```python
subjects_seen = {}
# per session:
mname = result['mname']
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subj_idx = subjects_seen[mname]
```

iii. The subject split follows directly from the `mname` field in the data, same as the reference.

## 1-c. How are the data split into sessions?

i. A session is a unique combination of `mname`, `datexp`, and `blk`. The AI deduplicates across experiment types by tracking seen session keys in a set. 89 unique sessions are found.

ii.
```python
def get_all_unique_sessions(exp_info):
    seen = set()
    for exp_type, session_list in exp_info.items():
        for s in session_list:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in seen:
                seen.add(key)
                sessions.append({...})
    return sessions
```

iii. Same approach as the reference: session = mouse + date + block, deduplicated across experiment types.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd` (frame-to-trial index) and `ft_CorrSpc` (in-corridor flag), but the AI also filters by `ft_move > 0` (running). Only frames satisfying all three conditions are included. The reference does NOT filter by running.

ii.
```python
trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
frame_indices = np.where(trial_mask)[0]
```

iii. The AI noted from the reference code that `VRmove = ft_move > 0` was used as a frame mask in the original analyses. However, applying this filter removes non-running frames within trials, creating temporal gaps that could disrupt decoder training.

## 1-e. How are trials filtered based on quality controls?

i. The AI only filters out trials with fewer than 2 valid frames (after the running filter). There is no trial length outlier filter. The reference applies a 99th percentile trial length filter to remove extremely long trials (animals that stopped for extended periods).

ii.
```python
if len(frame_indices) < 2:
    continue  # skip trials with too few frames
# ...
if len(neural_trials) < 2:
    print(f"  WARNING: {key} has {len(neural_trials)} valid trials, skipping")
    return None
```

iii. The AI does not document a rationale for omitting a trial length filter. The verification output shows max T of 178 and time_since_trial_start up to 1765 seconds (29 minutes), indicating extremely long outlier trials are kept.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/{session_id}_neural_data.npy` (list of arrays per imaging plane, concatenated) and `iarea` in `retinotopy/{mname}_{datexp}_trans.npz`.

ii.
```python
spk = np.concatenate(dat['spks'], axis=0)  # (n_neurons, n_frames)
iarea = load_retino(mname, datexp)
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. The traces are not further processed (no dF/F, no deconvolution). Columns for valid trial frames are extracted and stored as float32. The reference uses float16.

ii.
```python
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The AI correctly noted the data already contains deconvolved traces. Using float32 instead of float16 doubles memory usage but preserves more precision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea` in `{-1, 7}` (outside visual cortex / unassigned) are excluded. The remaining neurons are mapped to V1, mHV, lHV, aHV using the same iarea codes as the reference. 4,105,393 neurons are kept.

ii.
```python
EXCLUDED_IAREA = {-1, 7}
BRAIN_REGION_MAP = {8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV', 5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV'}
mask = np.ones(len(iarea), dtype=bool)
for exc in EXCLUDED_IAREA:
    mask &= (iarea != exc)
```

iii. Same filtering as the reference. The neuron count matches (4,105,393), and brain region distribution matches exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start), the first frame satisfying `ft_CorrSpc & ft_move > 0` for that trial. The reference aligns to corridor entry using `ft_CorrSpc` only (no running filter). Both use variable-length trials with `off_start = 0.0` and `off_end = None`.

ii.
```python
trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The alignment event is corridor entry, consistent with instructions. However, by filtering for running, the actual first frame may differ from the corridor entry frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frames are used directly at ~3.178 Hz (~315 ms per frame). The time_bin_size is set to 315.0 ms.

ii.
```python
'time_bin_size': 315.0,  # approximate, in ms (1/3.178 Hz * 1000)
```

iii. Same as reference (no rebinning, native frame rate). The reference computes 1000.0/3.17 = 315.46 ms; the AI uses 315.0 ms -- a minor rounding difference.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (sound cue frame per trial) and `ft` (frame timestamps in MATLAB datenum format).

ii.
```python
sound_fr = SoundFr[trial_idx]
sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
frame_times = ft[frame_indices] * 86400
time_to_sound = sound_time_s - frame_times
```

iii. Same source variables as reference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound cue frame is interpolated to a timestamp (since SoundFr is fractional), then subtracted from each frame's timestamp. The sign convention is `sound_time - frame_time`, which is positive before the cue and negative after.

ii.
```python
sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
time_to_sound = sound_time_s - frame_times  # positive before cue, negative after
```

iii. Same processing as reference: interpolation of fractional frame, then time difference. Same sign convention (positive before cue).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Computed at the same frame indices used for neural data extraction.

ii.
```python
frame_times = ft[frame_indices] * 86400
time_to_sound = sound_time_s - frame_times
```

iii. Aligned by shared frame indices, same as reference.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field (date string) of each session, parsed into a datetime object.

ii.
```python
def compute_day_of_training(sessions):
    for s in sessions:
        date = datetime.strptime(s['datexp'], '%Y_%m_%d')
        mouse_dates[mname].append((date, s['key']))
    for mname, dates in mouse_dates.items():
        dates.sort(key=lambda x: x[0])
        first_date = dates[0][0]
        for date, key in dates:
            day_map[key] = (date - first_date).days
```

iii. The AI uses the datexp field to determine session dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes calendar days since the first session for each mouse: `(date - first_date).days`. This gives values up to 92. The reference counts session ordinals (0, 1, 2, ...), giving values up to 7.

ii.
```python
day_map[key] = (date - first_date).days  # calendar days since first session
```

iii. The AI interpreted "day of training" as actual calendar days elapsed, not session index. This is a reasonable interpretation of "day of training" but differs from the reference which counts recording day index.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` (frame timestamps in MATLAB datenum format), using the timestamps of the selected frame indices.

ii.
```python
frame_times = ft[frame_indices] * 86400
time_since_start = frame_times - frame_times[0]
```

iii. The AI uses frame timestamps but references the first included frame (first running frame in corridor) rather than the corridor entry frame (`StartFr`).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes `frame_times - frame_times[0]`, where `frame_times[0]` is the first running frame in the corridor. The reference interpolates `StartFr` (the corridor entry frame) onto the time axis and subtracts that. These differ because the first running frame may not be the corridor entry frame.

ii.
```python
time_since_start = frame_times - frame_times[0]  # relative to first included frame
```

iii. The AI's approach always starts time at 0 for the first frame. The reference measures time from corridor entry, which is conceptually the "trial start" event.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same `frame_indices` used for neural data.

ii.
```python
frame_times = ft[frame_indices] * 86400
```

iii. Aligned by shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether the trial was in a rewarded corridor.

ii.
```python
isRew = beh['isRew']
np.full((1, n_tp), float(isRew[trial_idx]), dtype=np.float32)
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Per-trial boolean is cast to float and broadcast across timepoints. No further processing.

ii.
```python
np.full((1, n_tp), float(isRew[trial_idx]), dtype=np.float32)
```

iii. Same as reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
wall_name = str(WallName[trial_idx])
stim_cat = STIM_CATEGORY_MAP.get(wall_name, None)
```

iii. Same source as reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 wall texture names are mapped to 4 categories (circle, leaf, rock, wood) using a hardcoded mapping. The category index is broadcast across timepoints.

ii.
```python
STIM_CATEGORY_MAP = {}
for s in ['circle1', 'circle2', 'circle3']:
    STIM_CATEGORY_MAP[s] = 'circle'
# ... (leaf, rock, wood similarly)
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'wood']
stim_cat_idx = STIM_CATEGORIES.index(stim_cat)
np.full((1, n_tp), stim_cat_idx, dtype=np.int64)
```

iii. Same mapping and categories as reference.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (lick frame numbers) and `LickTrind` (lick trial indices).

ii.
```python
LickFr = beh['LickFr']
LickTrind = beh['LickTrind']
trial_lick_mask = (LickTrind == trial_idx)
trial_lick_frs = LickFr[trial_lick_mask]
```

iii. The AI uses both `LickFr` and `LickTrind`. The reference only uses `LickFr` (creating a per-frame binary array).

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI filters licks to the current trial via `LickTrind`, then uses `searchsorted` to assign each lick to the nearest included frame (among running frames). This is more complex than the reference, which simply creates a per-frame binary array and indexes it.

ii.
```python
trial_lick_mask = (LickTrind == trial_idx)
trial_lick_frs = LickFr[trial_lick_mask]
lick_binary = np.zeros(n_tp, dtype=np.float32)
if len(trial_lick_frs) > 0:
    fi_sorted = frame_indices.astype(float)
    lick_bin_idx = np.searchsorted(fi_sorted, trial_lick_frs)
    for li in range(len(trial_lick_frs)):
        idx = lick_bin_idx[li]
        if idx >= n_tp:
            idx = n_tp - 1
        elif idx > 0:
            if abs(trial_lick_frs[li] - fi_sorted[idx-1]) < abs(trial_lick_frs[li] - fi_sorted[idx]):
                idx = idx - 1
        lick_binary[idx] = 1.0
```

iii. Because the AI filters out non-running frames, licks that occur during non-running periods are reassigned to the nearest running frame. The reference simply checks if any lick falls in a given frame. This could lead to lick signals being shifted to different frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are assigned to the nearest included frame index, so licking is aligned to the same frame set as neural data.

ii. See 8-b code.

iii. Due to the running filter, alignment may differ from the reference.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
positions = ft_Pos[frame_indices]
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions are clipped to [0, 39.999] and binned using `np.digitize` with edges at [10, 20, 30, 40] dm, giving 4 bins of 1 m each.

ii.
```python
positions = np.clip(positions, 0, CORRIDOR_LENGTH_DM - 0.001)
pos_bin = np.digitize(positions, POSITION_BIN_EDGES_DM[1:])  # 0,1,2,3
pos_bin = np.clip(pos_bin, 0, N_POSITION_BINS - 1)
```

iii. Functionally equivalent to the reference's `np.clip(ft_Pos // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. 4 equal-length spatial bins of 1 m each: 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
POSITION_BIN_EDGES_DM = np.array([0, 10, 20, 30, 40])
pos_bin = np.digitize(positions, POSITION_BIN_EDGES_DM[1:])
```

iii. Same as reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read from `ft_Pos` at the same frame indices used for neural data.

ii.
```python
positions = ft_Pos[frame_indices]
```

iii. Aligned by shared frame indices, same approach as reference.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speeds = ft_RunSpeed[frame_indices]
```

iii. Same source as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartile edges across all sessions (using running+corridor frames, subsampled to 10000 per session), then applies `np.digitize` to bin speeds. The reference computes rank-based quartiles per session over kept frames only.

ii.
```python
def compute_speed_quartile_edges(sessions, exp_info):
    for s in sessions:
        mask = ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
        speeds = ft_RunSpeed[mask]
        if len(speeds) > 10000:
            speeds = rng.choice(speeds, 10000, replace=False)
        all_speeds.append(speeds)
    edges = np.percentile(all_speeds, [0, 25, 50, 75, 100])
    return edges
# Per trial:
speed_bin = np.digitize(speeds, speed_edges[1:-1])
```

iii. The AI uses global percentile edges rather than per-session rank-based splitting. This means speed bins may not contain exactly 25% of data per session. Additionally, filtering for running frames before computing quartiles changes the speed distribution (excludes stationary frames).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. 4 bins based on global quartile edges: [-19.2, 12.2], [12.2, 25.0], [25.0, 40.5], [40.5, 321.9] cm/s. The reference uses per-session rank splitting into 4 equal bins.

ii.
```python
speed_edges = np.percentile(all_speeds, [0, 25, 50, 75, 100])
speed_bin = np.digitize(speeds, speed_edges[1:-1])
```

iii. The global edge approach can produce very unequal bin sizes per session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read from `ft_RunSpeed` at the same frame indices used for neural data.

ii.
```python
speeds = ft_RunSpeed[frame_indices]
```

iii. Aligned by shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles neuron count mismatches between spk and retinotopy by truncating to the minimum. Behavior and neural frame counts are reconciled by taking `min(n_frames, nfr_beh)`. Trials with fewer than 2 valid frames are skipped. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
if len(iarea) != n_neurons_raw:
    min_n = min(n_neurons_raw, len(iarea))
    spk = spk[:min_n]
    iarea = iarea[:min_n]
# ...
nfr = min(n_frames, nfr_beh)
# ...
if len(frame_indices) < 2:
    continue
```

iii. The reference handles similar edge cases: truncating behavior to neural frame count, dropping licks past the last imaged frame, dropping trials with no frames.

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural spike files (the largest are ~10+ seconds per session) dominates processing time. Total conversion takes ~1387 seconds (~23 minutes) for 89 sessions.

ii.
```python
spk = load_spk(mname, datexp, blk)  # 2-14 seconds per session
```

iii. Same bottleneck as reference: I/O for large neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking assignment loop iterates over each lick with a Python for-loop rather than vectorizing the nearest-frame assignment.

ii.
```python
for li in range(len(trial_lick_frs)):
    idx = lick_bin_idx[li]
    # ... nearest frame logic
    lick_binary[idx] = 1.0
```

iii. This loop could be vectorized with numpy operations.

## 12-c. What processing does the code repeat multiple times?

i. The behavior file is re-loaded for every session via `get_session_beh()`, even though multiple sessions share the same behavior file. This causes redundant file I/O. Additionally, speed quartile edges computation loads all behavior files a second time.

ii.
```python
def get_session_beh(session_key, exp_info):
    # For each session, iterates through exp_info and loads the behavior file
    beh_path = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
    beh_all = np.load(beh_path, allow_pickle=True).item()
```

iii. The reference groups sessions by behavior file and reads each file once.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `compute_speed_quartile_edges()` function loads and processes all behavior files to compute global speed edges before the main conversion loop, adding ~6 seconds. The reference computes speed bins per-session inside the main loop, avoiding this extra pass. Additionally, the AI stores `n_neurons_raw` in results but it's not used in the final output.

ii.
```python
speed_edges = compute_speed_quartile_edges(speed_sessions, exp_info)
```

iii. The extra pass is a design choice for global quartiles, not strictly unnecessary given the AI's approach.
