# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories under `data/`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. `Imaging_Exp_info.npy` is loaded first as the master index. Each unique recording is identified by `<mouse>_<date>_<blk>`. Behavior files `Beh_<exp_type>.npy` are loaded per experiment type. Neural data and retinotopy are loaded per session. The AI also imports `utils.py` from the reference `code/` directory for helper functions like `neu_area_ID`.

ii.
```python
def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()

def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()

spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
```

iii. The AI follows the same data loading approach as the reference code, reading from the same three directories and the same file naming conventions.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info entries. Each unique mouse name becomes a subject. The AI builds a `subject_to_idx` mapping in order of first appearance (sorted by subject then date).

ii.
```python
subjects = []
subject_to_idx = {}
for sess in sessions:
    if sess.subject not in subject_to_idx:
        subject_to_idx[sess.subject] = len(subjects)
        subjects.append(sess.subject)
```

iii. The AI correctly identifies 19 unique subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `<mouse>_<date>_<blk>`. The AI deduplicates the 142 experiment entries in `Imaging_Exp_info.npy` down to 89 unique recordings. When a recording appears under multiple experiment types, the AI selects a "canonical" behavior entry by choosing the one with the most non-NaN `stim_id` values and most unique wall names, after validating that duplicates agree on key fields.

ii.
```python
per_rec: dict[str, list[tuple[str, dict]]] = defaultdict(list)
for exp_type, dbs in exp_info.items():
    for db in dbs:
        rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        per_rec[rec_id].append((exp_type, db))
```

iii. The AI documents that duplicate experiment entries are validated for consistency before choosing one canonical entry. This is more thorough than the reference which simply takes the first occurrence.

## 1-d. How are the data split into trials?

i. Trials are defined by iterating over `range(beh['ntrials'])`. For each trial, the AI computes a frame mask using `ft_CorrSpc & (ft_move > 0)` and `ft_trInd == trial` to get only running corridor frames. Trials with no retained frames raise an error rather than being silently dropped. Trials have variable length (no fixed window or padding).

ii.
```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
    ...
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
    ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
    valid = finite_trial & ft_corr & ft_move
    masks = []
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
        ...
        masks.append(frame_idx)
    return masks
```

iii. The AI justifies filtering to running frames by citing the paper: "We only considered timepoints during running for analysis." The reference does NOT filter by `ft_move > 0`.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trial frames to only running corridor frames (`ft_CorrSpc & (ft_move > 0)`). If a trial has zero retained frames after this filter, it raises a `ValueError`. No other trial-level quality filter is applied.

ii.
```python
valid = finite_trial & ft_corr & ft_move
...
if len(frame_idx) == 0:
    raise ValueError(f"Trial {trial} has no retained running corridor frames")
```

iii. The AI applies `ft_move > 0` as a quality filter, which the reference does not apply to trial frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` (a list of neuron-by-frame arrays per imaging plane, concatenated along axis 0), and `iarea` from `retinotopy/<mouse>_<date>_trans.npz` for visual area assignment.

ii.
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
```

iii. Same raw data sources as the reference.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed (no dF/F). They are sliced to the retained frame indices per trial and stored as float16. Trials have variable length (no padding to a fixed window).

ii.
```python
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The AI states the `spks` arrays are already deconvolved traces from Suite2p, consistent with the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies a complex reference-style neuron selection procedure rather than simply keeping all neurons in the four visual areas. It computes stimulus selectivity d-prime on odd running corridor frames, selects the top/bottom 5% of mHV neurons by d-prime (the `corr_neu` criterion is also applied), and selects aHV reward-prediction neurons using interpolated position activity. Only mHV and aHV neurons are kept (V1 and lHV are excluded). This reduces from ~4.7M total neurons to ~102k selected neurons across 89 sessions.

ii.
```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
corr_neu = (spk[:, stim_pos_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)) | (
    spk[:, stim_neg_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)
)
mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
...
keep_mask = mhv_mask | ahv_mask
```

iii. The AI justified this as a "reference-style subset for tractability" because the full dataset of ~4.7M neurons would be impractical for the decoder. The reference solution keeps ALL neurons in V1, mHV, lHV, and aHV (~4.1M neurons).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). The AI uses `Trial_start_time` for computing `time_since_trial_start`. Neural frames are taken from the running corridor portion of each trial. Trials have variable length.

ii.
```python
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The alignment event is corridor entry, matching the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of 3.17 Hz is preserved, giving a bin size of ~315 ms. The AI computes the actual median frame interval from the data.

ii.
```python
"time_bin_size": float(np.median([np.median(np.diff(np.asarray(
    load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0
    for s in sessions]) * 1000.0),
```

iii. Consistent with the reference approach.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the absolute timestamp of the sound cue for each trial) and `ft` (the timestamp of every imaging frame).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=float)
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The AI uses `SoundTime` (an absolute time in days) rather than `SoundFr` (a frame number) used by the reference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound time and frame time are both in MATLAB datenum format (days). The difference `SoundTime[trial] - ft[frame]` is converted to seconds by multiplying by 86400. The result is positive before the cue and negative after.

ii.
```python
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The reference computes cue time by interpolating `SoundFr` (frame number) onto the frame time axis with `np.interp`, then computing `cue[trial] - time`. Both approaches yield signed seconds to the cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`frame_idx`) used for the neural data of that trial.

ii.
```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. Aligned by using the same retained frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date strings in `datexp` field of the experiment info entries.

ii.
```python
def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    ...
    for subject, sess_list in by_subject.items():
        first_date = min(parse_date(s.date_str) for s in sess_list)
        for sess in sess_list:
            day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
    return day_map
```

iii. Derived from the date metadata in the experiment info.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the calendar-day offset from each mouse's first recording date, plus 1 (so the first day is 1, not 0). This uses actual date differences, so if recordings are on days 1, 3, 5, the values would be 1, 3, 5 rather than 0, 1, 2.

ii.
```python
day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```

iii. The reference counts session ordinal (0-indexed: 0, 1, 2, ...) rather than actual calendar-day offsets. The AI's approach uses real date differences with 1-indexing.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (absolute timestamp of trial start) and `ft` (frame timestamps).

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The reference uses `StartFr` (frame number of corridor entry) interpolated onto the time axis. The AI uses `Trial_start_time` directly.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The difference `ft[frame] - Trial_start_time[trial]` is converted to seconds by multiplying by 86400. The result starts near zero and increases through the trial.

ii.
```python
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Simple time difference conversion to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same `frame_idx` used for neural data.

ii.
```python
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Aligned by shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks rewarded vs unrewarded trials.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=float)
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The raw `isRew` value (0 or 1) is broadcast across all time points of the trial.

ii.
```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. No processing needed; direct use of the raw field.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which gives the wall texture name for each trial.

ii.
```python
wall_name = as_str_array(beh["WallName"])
stim_code = np.full(len(frame_idx), category_to_idx[str(wall_name[trial_idx])], dtype=np.int16)
```

iii. Same raw variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses ALL 15 individual wall name strings as separate categories (e.g., `circle1`, `circle2`, `leaf1`, `leaf1_swap1`, etc.), sorted alphabetically and mapped to integer indices 0-14. The value is per-trial, broadcast across all time points.

ii.
```python
categories = sorted({
    str(v) for sess in sessions
    for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
})
category_to_idx = {name: idx for idx, name in enumerate(categories)}
```

iii. The reference groups the 15 names into 4 broad texture categories (circle, leaf, rock, wood). The AI keeps all 15, resulting in a 15-way classification task instead of 4-way. The instructions say "Visual stimulus category. e.g. circle1, leaf2, etc." which could be interpreted either way.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame number of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The reference uses only `LickFr` (not `LickTrind`), converting it to a session-wide binary vector and then slicing per trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI finds lick frames belonging to that trial (via `LickTrind`), then checks which retained frame indices match those lick frames using `np.isin`. The result is binary: 1 if a lick occurred at that frame, 0 otherwise.

ii.
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. Functionally equivalent to the reference approach but uses `LickTrind` for trial matching rather than a session-wide binary vector.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Uses the same `frame_idx` as neural data. The lick frame numbers index imaging frames, so they are already on the same grid.

ii.
```python
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. Aligned by shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame, in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Same raw variable as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is clipped to [0, 39.999] (corridor range) and divided by 10 to get meter bins (0-3), producing 4 bins of 1 m each.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Similar to reference which uses `np.clip(beh['ft_Pos'][:n_frames] // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four 1-meter bins: [0,10), [10,20), [20,30), [30,40] in decimeters, corresponding to 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Matches the 4 equal-length 1-m bins specified in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same `frame_idx` as neural data.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. Aligned by shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Same raw variable as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes GLOBAL quartile edges across ALL retained frames in ALL sessions during a preprocessing pass, then applies `np.digitize` to bin each frame's speed. This produces 4 bins based on fixed threshold values.

ii.
```python
speed_values = np.concatenate(speed_values).astype(np.float32)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The reference computes per-session rank-based quartiles, ensuring exactly 25% of each session's data falls in each bin. The AI uses global quantile thresholds, which means individual sessions may have uneven bin occupancy.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges from all retained frames are used as thresholds. `np.digitize` assigns each speed value to one of 4 bins.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The instructions say "4 bins, each corresponding to 25% of the data." The AI achieves this globally (25% overall) but not per-session. The reference achieves it per-session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same `frame_idx` as neural data.

ii.
```python
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Aligned by shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles NaN values in `ft_trInd` by checking for finite values before converting to int. It validates that duplicate behavior entries across experiment types agree on key fields. Trials with no retained running corridor frames raise an error. The behavior is clipped to the number of neural frames (`nfr = spk.shape[1]`) implicitly by only using `frame_idx` values that index into the neural array.

ii.
```python
ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
finite_trial = np.isfinite(ft_trial)
ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
```

iii. The AI is more defensive than the reference about NaN handling and data validation.

## 12-a. What are the most time-consuming steps of the code?

i. The neuron selection procedure (`compute_selected_neurons`) is a major time cost, involving d-prime computation, interpolated position activity, and reward-prediction analysis. Reading the large spike files (~405 GB total) is also a bottleneck.

ii.
```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
interp_spk = utils.get_interpPos_spk(...)
```

iii. Full conversion took 1013.7 seconds (~17 minutes). The neuron selection adds significant overhead compared to the reference's simple area-based filtering.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that builds neural, input, and output arrays could potentially be partially vectorized. The chunk-local row computation iterates over chunks.

ii.
```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        ...
```

iii. The inner chunk loop is needed due to the chunked storage format, but the trial loop is inherent to the variable-length trial structure.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded multiple times: once in `build_global_metadata` for speed quartiles, once in `collect_sessions` for sample selection, and once per session during conversion. The `compute_trial_masks` function is also called both in `build_global_metadata` and during conversion.

ii.
```python
# In build_global_metadata:
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
trial_masks = compute_trial_masks(beh)

# Again in convert_dataset:
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
trial_masks = compute_trial_masks(beh)
```

iii. Behavior files are relatively small compared to neural data, so this redundancy has minor impact.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The complex neuron selection procedure (d-prime computation, interpolated position activity, reward-prediction analysis) is substantial processing that the reference solution does not perform. This is extra computation to select a small subset of neurons, whereas the reference simply keeps all neurons in the four visual areas. The `choose_canonical_behavior` function with its multi-field validation is also unnecessary overhead compared to simply taking the first occurrence.

ii.
```python
def compute_selected_neurons(...):
    # ~100 lines of d-prime, interpolation, reward-prediction analysis
    ...
```

iii. The neuron selection logic replicates analysis-specific code from the reference paper's figures, not the data loading/conversion pipeline.
