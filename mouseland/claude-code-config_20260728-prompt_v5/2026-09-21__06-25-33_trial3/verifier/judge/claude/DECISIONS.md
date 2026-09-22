# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index to discover all sessions grouped by experiment type. It builds a dictionary of unique sessions keyed by `mname_datexp_blk`. For each session, it loads the behavior from `Beh_{exp_type}.npy`, neural data from `spk/{session_key}_neural_data.npy`, and retinotopy from `retinotopy/{mname}_{datexp}_trans.npz`.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
# ...
spk = load_neural_data(mname, datexp, blk)
# in load_neural_data:
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate(data['spks'], axis=0)
# retinotopy:
data = np.load(path, allow_pickle=True)
return data['iarea']
```

iii. The AI documented that `Imaging_Exp_info.npy` is the master index and that sessions can appear under multiple experiment types. It prefers experiment types without `stimtype` for behavior loading.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) is extracted from the session info. All unique mouse names are sorted to form the subjects list, and each session is assigned its mouse's index.

ii.
```python
all_subjects = sorted(set(r['mname'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The AI identified 19 unique subjects from the master index, consistent with the paper.

## 1-c. How are the data split into sessions?

i. A session is defined as a unique combination of `mname`, `datexp`, and `blk`. The AI de-duplicates sessions that appear under multiple experiment types, keeping 89 unique sessions.

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in sessions:
    sessions[key] = { ... }
sessions[key]['exp_types'].append(exp_type)
```

iii. The AI noted that 142 session entries across experiment types reduce to 89 unique sessions, matching the paper's reported count.

## 1-d. How are the data split into trials?

i. Trials are indexed by `ft_trInd` in the behavior data, with `ntrials` giving the total count. The AI extracts frames for each trial where `ft_trInd == trial_idx`, `ft_CorrSpc == True`, AND `ft_move > 0` (mouse is running). Trials with fewer than 2 valid frames are skipped.

ii.
```python
trial_mask = valid & (ft_trInd.astype(float) == trial_idx)
corr_mask = ft_CorrSpc.astype(bool)
move_mask = ft_move > 0
frame_mask = trial_mask & corr_mask & move_mask
return np.where(frame_mask)[0]
```

iii. The AI justified the `ft_move > 0` filter by citing the paper: "We only considered timepoints during running for analysis" and the reference code's use of `ft_move > 0`. The AI also noted that the corridor frames (`ft_CorrSpc`) restrict to the 4m texture area.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 valid frames after filtering are dropped. Sessions with fewer than 2 valid trials are dropped. There is no upper-bound trial length filter.

ii.
```python
if len(frame_idx) < 2:
    continue  # Skip trials with too few frames
# ...
if len(neural_trials) < 2:
    print(f"  WARNING: Session {session_key} has {len(neural_trials)} valid trials, skipping")
    return None
```

iii. The AI did not apply the 99th-percentile trial length filter that the reference uses to remove outlier-long trials (e.g., mice that stopped for extended periods).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (one per session), which contains lists of per-plane neuron-by-frame arrays. These are concatenated across planes. The brain region assignment comes from `iarea` in the retinotopy files.

ii.
```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate(data['spks'], axis=0)  # (n_neurons, n_frames)
```

iii. The AI correctly identified the deconvolved fluorescence traces as the neural data source.

## 2-b. How is the `neural` data processed?

i. The neural data is filtered to keep only neurons in visual cortex regions, then the columns corresponding to valid trial frames are extracted and stored as float32 arrays.

ii.
```python
spk_filtered = spk[neuron_mask]
neural = spk_filtered[:, frame_idx].astype(np.float32)
```

iii. The AI noted that the data is already deconvolved, so no further processing (dF/F, deconvolution) is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered based on brain region. The AI excludes neurons with `iarea == -1` (unassigned) or `iarea == 7` (not in the 4 defined visual areas). Valid neurons are mapped to V1, mHV, lHV, or aHV.

ii.
```python
mask = (iarea != -1) & (iarea != 7)
region_idx = np.zeros(mask.sum(), dtype=np.int64)
valid_iarea = iarea[mask]
for iarea_val, region_name in AREA_MAP.items():
    region_idx[valid_iarea == iarea_val] = BRAIN_REGIONS.index(region_name)
```

iii. The AI documented the mapping: V1(8), mHV(0,1,2,9), lHV(5,6), aHV(3,4), and cited the reference code's `neu_area_ID()` function. The resulting neuron counts (V1: 1,833,035; mHV: 1,108,860; lHV: 495,318; aHV: 668,180) match the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). The AI extracts frame indices from the behavior data for each trial, and the neural data at those frames forms the trial's neural activity. Variable-length trials are preserved.

ii.
```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
```

iii. The AI set `temporal_alignment_event` to "Corridor entry (trial start)" and `off_start` to 0.0, `off_end` to None (variable trial length).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame rate (~3.17 Hz, ~315 ms per frame). No rebinning is applied. The bin size is computed as the mean inter-frame interval across sessions.

ii.
```python
frame_dt_days = np.nanmedian(np.diff(ft))
frame_dt_sec = frame_dt_days * 24 * 3600
# ...
'time_bin_size': mean_frame_dt * 1000,  # in ms
```

iii. The AI noted that the imaging frame is the native temporal resolution at ~315 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame index of the sound cue for each trial) and the current frame indices.

ii.
```python
sound_fr = beh['SoundFr'][t]
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. The AI documented using `SoundFr` to compute the time difference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(frame_idx - sound_fr) * frame_dt_sec`, where `frame_dt_sec` is the median inter-frame interval. This gives a value that is **negative before the sound cue and positive after**, which is "time since sound cue" rather than "time to sound cue."

ii.
```python
sound_fr = beh['SoundFr'][t]
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. The AI described this as "negative before, positive after" in its mapping plan. The variable name in the instructions is "Time to sound cue," and the reference computes `cue_time - frame_time` (positive before cue, negative after). The AI's sign convention is reversed relative to the reference.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same `frame_idx` indices used for the neural data of that trial, multiplied by the frame interval.

ii.
```python
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. Aligned by frame index, same as neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field in the session metadata, which contains the date string (e.g., `2022_07_12`).

ii.
```python
def compute_day_of_training(sessions_info):
    mouse_dates = defaultdict(list)
    for key, info in sessions_info.items():
        mouse_dates[info['mname']].append((key, parse_date(info['datexp'])))
```

iii. The AI uses the session date to compute training days.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the actual number of calendar days since the first session for each mouse. For example, if the first session is July 12 and the next is July 19, the day of training is 7, not 1. This is different from the reference, which counts the sequential recording session number (0, 1, 2, ...).

ii.
```python
for mname, date_list in mouse_dates.items():
    date_list.sort(key=lambda x: x[1])
    first_date = date_list[0][1]
    for key, dt in date_list:
        day_map[key] = (dt - first_date).days
```

iii. The AI justified this as computing from dates relative to first session date. The verification output shows day_of_training ranges up to 92, while the reference ranges up to 7 (since no mouse has more than 8 sessions).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (the frame index of corridor entry for each trial) and the current frame indices.

ii.
```python
start_fr = beh['StartFr'][t]
time_since_start = (frame_idx - start_fr) * frame_dt_sec
```

iii. Documented as using StartFr for corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes `(frame_idx - start_fr) * frame_dt_sec`, using the median inter-frame interval as an approximation. The reference interpolates `StartFr` onto the actual timestamp axis for more precise timing.

ii.
```python
time_since_start = (frame_idx - start_fr) * frame_dt_sec
```

iii. The AI uses a constant frame interval approximation rather than actual timestamps.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Same frame indices as the neural data, so aligned by construction.

ii.
```python
time_since_start = (frame_idx - start_fr) * frame_dt_sec
```

iii. Aligned by frame index.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavior data, which is a boolean per trial.

ii.
```python
reward_avail = float(beh['isRew'][t])
```

iii. Directly from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (0.0 or 1.0) and broadcast across all timepoints of the trial.

ii.
```python
inp[3, :] = reward_avail  # per-trial, broadcast
```

iii. No complex processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` in the behavior data, which gives the texture name for each trial.

ii.
```python
stim_name = wall_names[t]
# ...
all_stimuli = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
out_filled[0, :] = stim_to_idx[stim_name]
```

iii. The AI uses WallName directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses each individual WallName (e.g., circle1, circle2, leaf1, leaf2, rock1, wood1, etc.) as a separate category, resulting in **15 unique stimulus categories**. The reference groups these into 4 broad categories (circle, leaf, rock, wood) using a mapping table. The instructions say "Visual stimulus category. e.g. circle, leaf, etc." which suggests the grouped categories.

ii.
```python
all_stimuli = sorted(all_stimuli)  # 15 unique names
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
```

iii. The AI documented using WallName directly and producing 15 categories. In its Step 5 mapping plan, it noted "Use WallName directly" and "Pool all unique stimuli across sessions."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame indices of licks) and `LickTrind` (trial index per lick).

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
trial_lick_mask = lick_trind == trial_idx
trial_lick_fr = lick_fr[trial_lick_mask]
```

iii. The AI uses both LickFr and LickTrind to get per-trial licks.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI filters licks by `LickTrind`, rounds lick frames to the nearest integer, then creates a binary array (1 if a lick falls on that frame, 0 otherwise). The reference instead creates a global licking array for the entire session by truncating (not rounding) `LickFr` to int, then slices it per trial.

ii.
```python
lick_int_frames = np.round(trial_lick_fr).astype(int)
lick_set = set(lick_int_frames)
binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices], dtype=np.float32)
```

iii. The AI uses per-trial filtering and rounding, while the reference uses global truncation. The functional difference is minor.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames index the same imaging frames as the neural data. The binary licking signal is computed for the same frame indices used for neural data.

ii.
```python
licking = make_lick_binary(beh, frame_idx, t)
```

iii. Aligned by frame index.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position within the corridor at each imaging frame, in decimeters (0-60).

ii.
```python
positions = beh['ft_Pos'][:n_use]
trial_positions = positions[frame_idx[frame_idx < n_use]]
```

iii. Directly from the frame-level position data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position values (in decimeters, 0-40 in the texture area) are discretized into 4 bins using `np.digitize` with bin edges at [0, 10, 20, 30, 40].

ii.
```python
pos_clipped = np.clip(positions, 0, TEXTURE_LENGTH - 1e-6)
bins = np.digitize(pos_clipped, bin_edges) - 1
bins = np.clip(bins, 0, N_POS_BINS - 1)
```

iii. Each bin is 10 decimeters = 1 meter, giving 4 equal-length bins as required.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The AI uses `np.digitize` with edges [0, 10, 20, 30, 40] to create bins [0-10), [10-20), [20-30), [30-40]. The reference uses `ft_Pos // 10` clipped to [0,3], which produces the same bins.

ii.
```python
POS_BIN_EDGES = np.array([0, 10, 20, 30, 40])
bins = np.digitize(pos_clipped, bin_edges) - 1
```

iii. Both approaches produce equivalent 1-meter bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken from `ft_Pos` at the same frame indices used for the neural data.

ii.
```python
trial_positions = positions[frame_idx[frame_idx < n_use]]
```

iii. Aligned by frame index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
run_speeds = beh['ft_RunSpeed'][:n_use_speed]
trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
```

iii. Directly from the frame-level running speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartile thresholds (25th, 50th, 75th percentiles) across all sessions' running+corridor frames, with subsampling (max 10,000 frames per session). Speeds are then discretized using `np.digitize` against these thresholds. The reference computes per-session rank-based quartiles ensuring exactly 25% in each bin.

ii.
```python
# Global quartile computation:
quartiles = np.percentile(all_speeds, [25, 50, 75])
# Per-trial discretization:
bins = np.digitize(speeds, quartile_edges)
```

iii. The AI justified global quartiles for "consistent binning" across sessions. However, the reference uses per-session quartiles with rank-based splitting for exactly equal bins, handling ties (e.g., many zero-speed frames) more robustly.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using global percentile-based thresholds: speeds below the 25th percentile are bin 0, between 25th-50th are bin 1, etc. The edges were [12.2, 25.0, 40.5] based on the data.

ii.
```python
bin_edges = np.array([-np.inf, quartile_edges[0], quartile_edges[1], quartile_edges[2], np.inf])
bins = np.digitize(speeds, quartile_edges)
```

iii. Global thresholds vs. per-session rank-based splitting is a meaningful difference, especially for sessions with unusual speed distributions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is taken from `ft_RunSpeed` at the same frame indices used for the neural data.

ii.
```python
trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
```

iii. Aligned by frame index.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- Frame count mismatch between neural and behavior data: clips to minimum
- Neuron count mismatch between spike data and retinotopy: clips to minimum
- Trials with fewer than 2 frames: skipped
- Sessions with fewer than 2 valid trials: skipped
- Position/speed arrays shorter than neural frames: pads with last valid value

ii.
```python
n_frames = min(n_neural_frames, n_beh_frames)
# ...
if len(iarea) != n_neurons_total:
    min_n = min(len(iarea), n_neurons_total)
    iarea = iarea[:min_n]
    spk = spk[:min_n]
```

iii. The AI documented handling frame count mismatches (off by 1 is expected) and padded missing values.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files, which total ~434 GB of spike data across 89 sessions. Each file requires `np.load` with `allow_pickle=True` and concatenation of planes.

ii.
```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate(data['spks'], axis=0)
```

iii. The AI noted total spike data is ~434 GB and estimated ~5 s/GB for processing.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `make_lick_binary` function uses a Python list comprehension to check each frame against a set of lick frames, instead of using vectorized NumPy operations. The behavior loading re-loads behavior files for each session rather than caching them.

ii.
```python
binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices], dtype=np.float32)
```

iii. This is a Python loop over frames that could be vectorized with `np.isin` or direct array indexing.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded multiple times: once during `collect_all_running_speeds` and again during `process_session` for each session. The `find_behavior_for_session` function re-loads the behavior `.npy` file each time it is called. The reference code caches behavior files by grouping sessions sharing the same behavior file.

ii.
```python
# In collect_all_running_speeds:
beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)
# In process_session (called separately):
beh, exp_type = find_behavior_for_session(session_key, sessions_info, exp_info)
```

iii. Each behavior file is loaded at least twice: once for speed quartile computation and once for actual processing.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `collect_all_running_speeds` function subsamples 10,000 frames per session when computing global speed quartiles, introducing unnecessary approximation. The processing plots (`--show-processing`) re-load all data for the session a second time.

ii.
```python
if len(speeds) > 10000:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(speeds), 10000, replace=False)
    speeds = speeds[idx]
```

iii. The subsampling introduces randomness into an otherwise deterministic computation.
