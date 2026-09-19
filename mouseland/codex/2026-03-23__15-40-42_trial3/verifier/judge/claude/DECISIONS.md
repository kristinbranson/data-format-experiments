# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` (behavior), `spk/` (deconvolved calcium traces), and `retinotopy/` (visual area assignments). It reads `Imaging_Exp_info.npy` as the master index, iterates over experiment groups, deduplicates recordings by `<mouse>_<date>_<blk>`, and collects all unique sessions. For each session it loads the behavior from `Beh_<exp_type>.npy`, the neural data from `<session_id>_neural_data.npy`, and the retinotopy from `<mouse>_<date>_trans.npz`.

ii.
```python
def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()

def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()

# In convert_dataset:
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
```

iii. The AI uses the same data sources as the reference code. It additionally validates that duplicate behavior entries across experiment groups are consistent before choosing a canonical one.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mname` field in the experiment index entries. Unique subject names are collected and assigned indices. The AI maintains subject ordering through `subject_to_idx`.

ii.
```python
subjects = []
subject_to_idx = {}
for sess in sessions:
    if sess.subject not in subject_to_idx:
        subject_to_idx[sess.subject] = len(subjects)
        subjects.append(sess.subject)
```

iii. The subject identity comes directly from the raw metadata. 19 unique subjects are identified, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is defined by the unique recording ID `<mouse>_<date>_<blk>`. Duplicate entries across experiment groups are merged. After deduplication, 89 unique sessions remain, matching the paper. Sessions are sorted by subject, date, and block number.

ii.
```python
per_rec: dict[str, list[tuple[str, dict]]] = defaultdict(list)
for exp_type, dbs in exp_info.items():
    for db in dbs:
        rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        per_rec[rec_id].append((exp_type, db))
```

iii. The deduplication approach is the same as the reference. The AI additionally validates that duplicate references are consistent.

## 1-d. How are the data split into trials?

i. Trials are defined by the behavior field `ntrials` and frame-level `ft_trInd`. For each trial, the AI retains frames where `ft_CorrSpc` is true (corridor space), `ft_move > 0` (mouse is running), and `ft_trInd` is finite and matches the trial index. This is stricter than the reference, which only filters on `ft_CorrSpc`.

ii.
```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
    ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
    valid = finite_trial & ft_corr & ft_move
    masks = []
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
        masks.append(frame_idx)
    return masks
```

iii. The AI justified the running filter by citing the paper: "We only considered timepoints during running for analysis" and the reference code's use of `ft_move > 0` in selectivity analyses.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial length filtering. Every trial with at least one valid (running corridor) frame is retained. This differs from the reference, which drops trials longer than the 99th percentile of trial lengths. However, the AI's running-frame filter (`ft_move > 0`) removes stationary frames that cause long trials in the reference.

ii.
```python
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
    if len(frame_idx) == 0:
        raise ValueError(f"Trial {trial} has no retained running corridor frames")
    masks.append(frame_idx)
```

iii. The AI does not explicitly discuss filtering outlier trials. The `ft_move > 0` filter indirectly handles the issue by removing stationary frames from long trials, but does not remove the trials themselves.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data file (list of neuron-by-frame arrays per imaging plane, concatenated) and `iarea` from the retinotopy file.

ii.
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
iarea = np.asarray(ret["iarea"])
```

iii. Same raw sources as the reference.

## 2-b. How is the `neural` data processed?

i. The AI applies a complex selectivity-based neuron selection before extracting trial data. Selected neurons are stored as float16. No additional processing (e.g., normalization, smoothing) is applied to the traces themselves.

ii.
```python
kept_idx, region_idx, kept_stats = compute_selected_neurons(spk_chunks, beh, iarea, sess.rec_id)
# ...
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The AI justified storing as float16 for memory efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies a complex selectivity-based neuron selection inspired by the reference code's analysis functions:
- mHV neurons are selected based on familiar-stimulus d' (top/bottom 5% among corridor-responsive neurons on odd running trials).
- aHV neurons are selected based on reward-prediction d' (>= 0.3 among positively stimulus-selective neurons) using position-interpolated activity.
- V1 and lHV neurons are entirely excluded.
- This yields ~102,541 neurons (mean 1,152/session) compared to the reference's ~4,105,393 neurons across all 4 visual areas.

ii.
```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
# ...
keep_mask = mhv_mask | ahv_mask
```

iii. The AI argued this was necessary for tractability with the decoder trainer, which concatenates all sessions in memory. The CONVERSION_NOTES.md states: "Exporting all visual-area neurons would be intractable."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). Each trial contains only the frames from corridor entry through the corridor traversal (running frames in corridor space). Trials are variable length.

ii.
```python
for trial_idx, frame_idx in enumerate(trial_masks):
    # frame_idx contains running corridor frames for this trial
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The alignment event is corridor entry, matching the instructions and reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of 3.17 Hz (~315 ms per frame) is used. The AI computes `time_bin_size` from the median inter-frame interval across sessions.

ii.
```python
"time_bin_size": float(np.median([np.median(np.diff(np.asarray(
    load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0
    for s in sessions]) * 1000.0),
```

iii. Same approach as the reference (native frame rate, no rebinning).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the absolute time of the sound cue for each trial, in MATLAB datenum days) and `ft` (the frame timestamps).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=float)
ft = np.asarray(beh["ft"], dtype=float)
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The AI uses `SoundTime` directly rather than the reference's approach of interpolating `SoundFr` onto the frame time axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue time is taken from `SoundTime[trial]` (already in MATLAB datenum units). For each frame, the time-to-cue is `(SoundTime[trial] - ft[frame]) * 86400`, giving seconds. Positive values are before the cue, negative after.

ii.
```python
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The sign convention (positive before cue, negative after) matches the reference.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`frame_idx`) used for the neural data of that trial.

ii.
```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. All data streams share the same frame indices per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date strings (`datexp`) in the session metadata, parsed into datetime objects.

ii.
```python
def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    for subject, sess_list in by_subject.items():
        first_date = min(parse_date(s.date_str) for s in sess_list)
        for sess in sess_list:
            day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```

iii. The AI derives training day from the calendar date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest recording date is found. The day of training for each session is the number of calendar days since that first recording, plus 1 (so the first session is day 1). Values range from 1 to 93. This differs from the reference, which counts recording sessions (0-indexed, values 0 to 7).

ii.
```python
day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```

iii. The AI interpreted "day of training" as calendar days rather than session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the absolute time of trial start for each trial) and `ft` (frame timestamps), both in MATLAB datenum units.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The AI uses `Trial_start_time` rather than the reference's `StartFr` (corridor entry frame), though both represent trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame, the time since trial start is `(ft[frame] - Trial_start_time[trial]) * 86400`, giving seconds. Values start near 0 and increase.

ii.
```python
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The computation is straightforward time difference in seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame indices used for the neural data of that trial.

ii.
```python
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. All data streams share the same frame indices per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial boolean/integer indicating whether the trial is in the rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=float)
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The `isRew` value is cast to float and broadcast across all frames of the trial.

ii.
```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. No processing beyond type conversion and broadcasting. Same as reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which gives the name of the wall texture for each trial.

ii.
```python
wall_name = as_str_array(beh["WallName"])
stim_code = np.full(len(frame_idx), category_to_idx[str(wall_name[trial_idx])], dtype=np.int16)
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps all 15 individual wall names (circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5) as separate categories, resulting in a 15-way classification. This differs from the reference, which maps all names to 4 base texture categories (circle, leaf, rock, wood).

ii.
```python
categories = sorted(
    {
        str(v)
        for sess in sessions
        for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
    }
)
category_to_idx = {name: idx for idx, name in enumerate(categories)}
```

iii. The AI chose to preserve all individual texture names. The instructions say "Visual stimulus category. e.g. circle, leaf, etc." which suggests grouping into base texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
```

iii. The AI uses both `LickFr` and `LickTrind` to assign licks to specific trials, whereas the reference only uses `LickFr` (which already indexes frames directly).

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frames belonging to that trial (identified via `LickTrind`) are found, and a binary vector is created: 1 if the frame matches a lick frame, 0 otherwise.

ii.
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The result is a binary time-varying vector per trial, same as the reference.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is computed on the same `frame_idx` used for the neural data, so it is inherently aligned.

ii.
```python
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. Frame-level alignment, same as reference.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to [0, 39.999] (corridor range in decimeters), then divided by 10 and floored to get 4 bins of 1 meter each.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Equivalent to the reference's `clip(ft_Pos // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The position is divided by 10 dm (= 1 m) and floored, yielding bins [0-1m), [1-2m), [2-3m), [3-4m]. Values are clipped to the range [0, 3].

ii.
```python
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Same 4-bin scheme as the reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same `frame_idx` used for neural data.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. Frame-level alignment, same as reference.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Same source as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global quartile edges across all retained frames from all sessions using `np.quantile` at [0.25, 0.5, 0.75]. Then each frame's speed is binned using `np.digitize`. This differs from the reference, which computes per-session rank-based quartiles ensuring exactly 25% of frames per bin within each session.

ii.
```python
# Global edges computation:
speed_values = np.concatenate(speed_values).astype(np.float32)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

# Per-frame binning:
def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The AI justified global quartiles as ensuring 25% bins across the full dataset. However, this leads to very uneven per-session distributions (e.g., one session has 95.6% in Q1).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global threshold edges are computed at the 25th, 50th, and 75th percentiles of all retained speed values. `np.digitize` assigns each value to one of 4 bins.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Global thresholds rather than per-session rank-based quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed by the same `frame_idx` used for neural data.

ii.
```python
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Frame-level alignment, same as reference.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behavior and neural data lengths may differ, so the neural frame count is used as the authoritative length; (2) `ft_trInd` may contain NaN values, which are handled by checking `np.isfinite`; (3) trials with zero retained frames raise an error (though this doesn't occur in practice); (4) behavior files are loaded for validation of duplicate entries.

ii.
```python
ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
finite_trial = np.isfinite(ft_trial)
ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
```

iii. The approach is similar to the reference's handling of behavior-neural length mismatches.

## 12-a. What are the most time-consuming steps of the code?

i. Two main bottlenecks: (1) loading the large neural data files (~405 GB total), and (2) the d'-based neuron selection computation, which requires computing stimulus selectivity across all neurons in each session. The full conversion takes ~17 minutes.

ii.
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
# ...
kept_idx, region_idx, kept_stats = compute_selected_neurons(spk_chunks, beh, iarea, sess.rec_id)
```

iii. The neuron selection step is an additional bottleneck not present in the reference.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `compute_trial_masks` scans valid frames for each trial index sequentially, similar to the reference's `trial_frames` function. This could be vectorized with `np.unique` or grouping.

ii.
```python
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
```

iii. Minor inefficiency relative to I/O costs.

## 12-c. What processing does the code repeat multiple times?

i. The AI loads behavior files multiple times: once in `collect_sessions` (for canonical behavior selection), once in `build_global_metadata` (for speed quartiles), and once in `convert_dataset` (for actual conversion). This is redundant I/O.

ii.
```python
# In collect_sessions:
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
# In build_global_metadata:
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
# In convert_dataset:
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
```

iii. The behavior files are relatively small compared to neural files, so the overhead is modest.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The complex d'-based neuron selection (`compute_selected_neurons`) is extensive processing that the reference does not perform. It computes stimulus selectivity, corridor-responsiveness, and reward-prediction metrics to select a small subset of neurons. The reference simply keeps all neurons in the 4 visual areas. This is unnecessary for the conversion task and adds both computational cost and complexity.

ii.
```python
def compute_selected_neurons(spk_chunks, beh, iarea, session_id):
    # ~90 lines of d' computation, percentile filtering, reward prediction analysis
    stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
    mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
    # ...
```

iii. The AI justified this as necessary for memory tractability with the decoder trainer, but this is not what the reference does.
