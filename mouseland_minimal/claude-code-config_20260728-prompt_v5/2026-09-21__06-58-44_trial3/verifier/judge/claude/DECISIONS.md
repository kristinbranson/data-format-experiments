# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, iterates over all experiment types, and collects unique sessions keyed by `(mname, datexp, blk)`. For each session, it loads the behavior from `Beh_<exp_type>.npy`, neural data from `spk/<session_id>_neural_data.npy`, and retinotopy from `retinotopy/<mname>_<datexp>_trans.npz`. Behavior files are cached by experiment type.

ii.
```python
exp_info = np.load(
    os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
).item()
# ...
for exp_type, sessions in exp_info.items():
    for s in sessions:
        key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
        if key not in unique_sessions:
            # ...
            unique_sessions[key] = { ... }
```

```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. The AI explored the data structure in the trajectory, discovered that sessions appear under multiple experiment types, and decided to deduplicate by the physical recording key `(mname, datexp, blk)`, keeping only the first encountered entry for each.

## 1-b. How are the data split into subjects?

i. The mouse name `mname` is taken from each session's entry. Subjects are collected into a list as they are encountered (not sorted). `subject_idx` maps each session to its position in that list.

ii.
```python
if mname not in subjects_set:
    subjects_set.append(mname)
# ...
subject_idx_all.append(subjects_set.index(mname))
```

iii. The AI groups sessions by mouse name from the experiment info. Subject ordering depends on iteration order of the dictionary rather than being explicitly sorted.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` combination. Sessions appearing under multiple experiment types are deduplicated, keeping only the first encountered. Sessions with fewer than 2 valid trials or no valid neurons are skipped.

ii.
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in unique_sessions:
    unique_sessions[key] = { ... }
```

```python
if len(neural_trials) < 2:
    print(f"  Only {len(neural_trials)} valid trials, skipping session")
    skipped_sessions += 1
    continue
```

iii. The AI investigated shared sessions across experiment types and confirmed that the behavior data is effectively the same for the same physical recording, so deduplication is correct.

## 1-d. How are the data split into trials?

i. Trials are defined by the data's `ft_trInd` field. For each trial (0 to `ntrials`), frames where `ft_trInd == trial` AND `ft_CorrSpc == True` are selected. This extracts only the corridor (texture) frames. Trials with fewer than `MIN_FRAMES_PER_TRIAL = 3` corridor frames are excluded.

ii.
```python
for t in range(ntrials):
    mask = (ft_trInd == t) & ft_CorrSpc
    frames = np.where(mask)[0]
    if len(frames) < MIN_FRAMES_PER_TRIAL:
        continue
```

iii. The AI decided to use `ft_CorrSpc` to select corridor frames, matching the approach of keeping only texture-corridor traversal frames aligned to corridor entry.

## 1-e. How are trials filtered based on quality controls?

i. The only filter is a minimum of 3 corridor frames per trial (`MIN_FRAMES_PER_TRIAL = 3`). Trials with fewer frames are excluded. There is no upper bound on trial length (no outlier removal for extremely long trials).

ii.
```python
MIN_FRAMES_PER_TRIAL = 3

# ...
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
```

iii. The AI did not implement any long-trial filtering. The trajectory does not show reasoning about removing outlier-length trials where the mouse stopped for extended periods.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which is a list of per-plane arrays concatenated along axis 0. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```

iii. The AI correctly identified the neural data source from the spike files and retinotopy files.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed. For each trial, the columns corresponding to the corridor frames are extracted. The data is stored as the original float64 (no dtype cast to reduce size).

ii.
```python
neural_trial = spk_filtered[:, frames]
# ...
neural_trials.append(neural_trial)
```

iii. The AI recognized that the data already contains deconvolved traces from Suite2p and no additional processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area. The `neu_area_ID` function assigns neurons to V1 (iarea==8), mHV (iarea in {0,1,2,9}), lHV (iarea in {5,6}), aHV (iarea in {3,4}). Only neurons in one of these four areas are kept.

ii.
```python
def neu_area_ID(iarea):
    idx = {}
    idx['V1'] = iarea == 8
    idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
    idx['lHV'] = (iarea == 5) | (iarea == 6)
    idx['aHV'] = (iarea == 3) | (iarea == 4)
    return idx

valid_neurons = area_idx['V1'] | area_idx['mHV'] | area_idx['lHV'] | area_idx['aHV']
spk_filtered = spk[neuron_indices, :]
```

iii. The AI matched the reference code's `neu_area_ID` function to determine which areas to include.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Trials are extracted using `ft_CorrSpc` frames, which start at corridor entry. Trials are variable length (no padding or truncation to a fixed window).

ii.
```python
mask = (ft_trInd == t) & ft_CorrSpc
frames = np.where(mask)[0]
neural_trial = spk_filtered[:, frames]
```

iii. The AI used `ft_CorrSpc` to identify frames inside the corridor, naturally aligning to corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed empirically from the median inter-frame interval of the first session's `ft` timestamps, converted to milliseconds. This yields approximately 315 ms.

ii.
```python
ft = first_beh['ft']
dt_days = np.median(np.diff(ft))
time_bin_ms = dt_days * 24 * 3600 * 1000  # convert days to ms
```

iii. The AI computed the bin size from the data rather than using a hardcoded constant like `1000/3.17`.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr`, the sound cue frame for each trial, and the trial's corridor frame indices.

ii.
```python
sound_fr = SoundFr[t]
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The AI used `SoundFr` as the source of sound cue timing.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundFr[trial] - frame_index) * time_bin_s`, where `time_bin_s` is the median frame interval. This gives positive values before the cue and negative after. The computation treats frame indices as uniformly spaced, multiplying by a constant bin size rather than using actual timestamps.

ii.
```python
sound_fr = SoundFr[t]
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The AI computed frame differences and multiplied by a constant time bin size, rather than interpolating onto the actual time axis from `ft`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same `frames` array used for the neural data of that trial, ensuring alignment.

ii.
```python
frames = np.where(mask)[0]
time_to_cue = (sound_fr - frames) * time_bin_s
neural_trial = spk_filtered[:, frames]
```

iii. Same frame indices are used for both neural and input data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date string in `datexp` for each session. The AI parses `datexp` as a calendar date.

ii.
```python
def get_date(datexp):
    return datetime.strptime(datexp, '%Y_%m_%d')

def compute_days_per_mouse(unique_sessions):
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
    for key, info in unique_sessions.items():
        d = get_date(info['datexp'])
        first_d = mouse_first_date[info['mname']]
        days[key] = (d - first_d).days
    return days
```

iii. The AI decided to compute calendar days since the mouse's first recording session, rather than counting session ordinals.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is computed as the number of calendar days between the session date and the mouse's first recording date. This differs from counting the ordinal number of recording sessions. For example, if a mouse was recorded on days 0, 3, and 7, the reference would give 0, 1, 2 while the AI gives 0, 3, 7.

ii.
```python
days[key] = (d - first_d).days
```

iii. The AI interpreted "day of training" literally as calendar days elapsed, not as ordinal session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the corridor frame indices. The first corridor frame of the trial is taken as the trial start.

ii.
```python
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. The AI used the frame index difference from the first corridor frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is `(frame_index - first_frame_index) * time_bin_s`. It starts at 0 for the first corridor frame and increases by `time_bin_s` per frame. This uses a constant bin size rather than actual timestamps, and uses the first corridor frame (from `ft_CorrSpc`) rather than `StartFr`.

ii.
```python
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. The AI approximated time using uniform frame spacing rather than interpolating `StartFr` onto actual frame times.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Computed from the same `frames` array used for neural data, ensuring alignment.

ii.
```python
time_since_start = (frames - frames[0]) * time_bin_s
neural_trial = spk_filtered[:, frames]
```

iii. Same frame indices used throughout.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks trials in rewarded corridors.

ii.
```python
reward_avail = np.full(n_frames, float(isRew[t]), dtype=np.float32)
```

iii. Directly used from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The binary value is broadcast across all frames of the trial. No further processing.

ii.
```python
reward_avail = np.full(n_frames, float(isRew[t]), dtype=np.float32)
```

iii. Straightforward conversion from per-trial boolean to per-frame float.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name for each trial, and `UniqWalls`, the set of unique wall names in the session.

ii.
```python
for name in beh['UniqWalls']:
    all_stim_names.add(name)
stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(stim_names)}
# ...
stim_cat = stim_to_idx[WallName[t]]
```

iii. The AI collected all unique stimulus names from `UniqWalls` across all sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` is mapped to an index in the sorted list of all unique wall names across the dataset. **This preserves individual texture variants** (e.g., circle1, circle2, circle3 are separate categories) rather than grouping them into base textures (circle, leaf, rock, wood). The resulting number of categories equals the number of unique wall names (15) rather than 4.

ii.
```python
stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(stim_names)}
stim_cat = stim_to_idx[WallName[t]]
stim_arr = np.full(n_frames, stim_cat, dtype=int)
```

iii. The AI used individual texture names as categories. The trajectory does not show reasoning about grouping textures into broader categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (lick frame numbers) and `LickTrind` (lick trial indices).

ii.
```python
LickFr = beh['LickFr']
LickTrind = beh['LickTrind']
```

iii. The AI used both `LickFr` and `LickTrind` to identify which licks belong to which trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks are identified by matching `LickTrind == trial`. The lick frame numbers are rounded to the nearest integer (`np.round`). A binary array is created per trial, marking frames with at least one lick as 1.

ii.
```python
trial_lick_mask = (LickTrind.astype(int) == t)
trial_lick_frames = np.round(LickFr[trial_lick_mask]).astype(int)
frame_to_local = {f: i for i, f in enumerate(frames)}
for lf in trial_lick_frames:
    if lf in frame_to_local:
        lick_arr[frame_to_local[lf]] = 1
```

iii. The AI used a per-trial approach, matching licks to trials via `LickTrind`, and rounding rather than truncating frame numbers.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are matched to the corridor frames of each trial. Only licks falling on corridor frames are included, ensuring alignment with neural data.

ii.
```python
frame_to_local = {f: i for i, f in enumerate(frames)}
for lf in trial_lick_frames:
    if lf in frame_to_local:
        lick_arr[frame_to_local[lf]] = 1
```

iii. Frame-level alignment through the shared `frames` array.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in decimeters.

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr]
pos_bins = discretize_position(ft_Pos[frames])
```

iii. Direct use of the frame-level position variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (integer division) and clipped to [0, 3], giving four 1-meter bins.

ii.
```python
def discretize_position(positions):
    bins = np.clip(positions // 10, 0, 3).astype(int)
    return bins
```

iii. Matches the instruction to discretize into 4 equal-length 1-m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 gives bins [0,10) -> 0, [10,20) -> 1, [20,30) -> 2, [30,40] -> 3. Values above 30 are clipped to 3.

ii.
```python
bins = np.clip(positions // 10, 0, 3).astype(int)
```

iii. Same approach as the reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same `frames` array used for neural data, ensuring alignment.

ii.
```python
pos_bins = discretize_position(ft_Pos[frames])
```

iii. Frame-level alignment through shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
speed_bins = discretize_speed(ft_RunSpeed[frames], speed_quartiles)
```

iii. Direct use of the frame-level running speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using **global** quartile boundaries computed across all corridor frames from all sessions. The boundaries are computed using `np.percentile` at [25, 50, 75], and `np.digitize` assigns each speed to a bin. This differs from the reference which computes rank-based quartiles per session.

ii.
```python
def compute_speed_quartiles(unique_sessions):
    all_speeds = []
    for key, info in unique_sessions.items():
        corridor_speeds = ft_RunSpeed[ft_CorrSpc]
        all_speeds.append(corridor_speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles

def discretize_speed(speeds, quartiles):
    bins = np.digitize(speeds, quartiles)
    return bins
```

iii. The AI chose global percentile-based quartiles rather than per-session rank-based quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with three boundaries (25th, 50th, 75th percentiles computed globally). Values below Q25 -> 0, Q25-Q50 -> 1, Q50-Q75 -> 2, above Q75 -> 3.

ii.
```python
bins = np.digitize(speeds, quartiles)
```

iii. Percentile-based thresholding rather than rank-based quartering.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is indexed by the same `frames` array used for neural data.

ii.
```python
speed_bins = discretize_speed(ft_RunSpeed[frames], speed_quartiles)
```

iii. Frame-level alignment through shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior data is truncated to the number of neural frames (`nfr = spk.shape[1]`). Trials with fewer than 3 corridor frames are skipped. Sessions that fail to load (neural or retinotopy errors) are skipped with a warning. Licks are matched per-trial via `LickTrind` and only assigned to corridor frames.

ii.
```python
nneu, nfr = spk.shape
ft_trInd = beh['ft_trInd'][:nfr]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
# ...
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
```

iii. The AI handles mismatches between behavior and neural frame counts by truncating to neural frames, similar to the reference approach.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files for each session, which are large numpy files. Additionally, the AI loads spike data twice: once during `compute_speed_quartiles` (to get `nfr`) and again during main processing.

ii.
```python
# In compute_speed_quartiles:
spk_data = np.load(spk_path, allow_pickle=True).item()['spks']
nfr = spk_data[0].shape[1]

# In main processing:
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```

iii. The I/O cost of loading large neural data files dominates the runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial lick assignment loop, which iterates over individual lick frames and uses a dictionary lookup to map to local indices, could be vectorized using array operations.

ii.
```python
for lf in trial_lick_frames:
    if lf in frame_to_local:
        lick_arr[frame_to_local[lf]] = 1
```

iii. Using a Python loop over lick frames is slower than vectorized array indexing.

## 12-c. What processing does the code repeat multiple times?

i. The code loads spike data twice: once in `compute_speed_quartiles` (to determine `nfr` for truncating behavior) and again in the main processing loop. Behavior files are also loaded redundantly - the stimulus name collection loop loads all behavior files, and they are loaded again (though cached) during main processing.

ii.
```python
# First pass in compute_speed_quartiles:
spk_data = np.load(spk_path, allow_pickle=True).item()['spks']
nfr = spk_data[0].shape[1]

# Second pass in main:
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```

iii. The double loading of spike files is particularly expensive given their size.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes global speed quartiles across all sessions in a separate first pass, loading all spike files just to get `nfr`. This could be deferred or combined with the main processing pass. The stimulus name collection is also a separate pass that could be merged.

ii.
```python
speed_quartiles = compute_speed_quartiles(unique_sessions)
```

iii. The separate passes for speed quartiles and stimulus names add overhead that could be avoided.
