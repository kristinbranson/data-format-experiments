# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three sources per session: (1) `Imaging_Exp_info.npy` as a master session index mapping experiment types to session metadata, (2) `Beh_<exp_type>.npy` behavior dictionaries per experiment type, and (3) `<mouse>_<date>_<blk>_neural_data.npy` neural data files. It deduplicates the 142 experiment entries in `Imaging_Exp_info.npy` down to 89 unique recording IDs (`<mouse>_<date>_<blk>`), then iterates over all unique sessions to load and process data.

ii.
```python
def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()

def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()

# In convert_dataset:
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
```

iii. The AI documented in CONVERSION_NOTES.md that `Imaging_Exp_info.npy` contains 23 experiment groups with 142 entries mapping to 89 unique recordings. The reference code functions `load_spk`, `load_exp_beh`, and `load_retino` load the same files. The AI reuses the reference `utils` module (imported at the top) for area ID mapping.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `mname` field in the experiment metadata dictionaries. Unique subject names are collected across all sessions and stored as a list. A `subject_idx` array maps each session to its subject index.

ii.
```python
# In build_global_metadata:
subjects = []
subject_to_idx = {}
for sess in sessions:
    if sess.subject not in subject_to_idx:
        subject_to_idx[sess.subject] = len(subjects)
        subjects.append(sess.subject)

# In convert_dataset:
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int64),
```

iii. The AI identified 19 unique subjects matching the paper's statement of "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique recording IDs: `<mouse>_<date>_<blk>`. The AI deduplicates experiment entries from `Imaging_Exp_info.npy` by grouping all entries with the same recording ID together, then validates that duplicate behavior references agree on key fields before choosing a canonical behavior entry.

ii.
```python
def collect_sessions(sample: bool) -> list[SessionRef]:
    exp_info = load_exp_info()
    per_rec: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            per_rec[rec_id].append((exp_type, db))

    sessions = []
    for rec_id, refs in per_rec.items():
        first_db = refs[0][1]
        exp_type, beh_key = choose_canonical_behavior(rec_id, refs)
        sessions.append(SessionRef(...))
    sessions.sort(key=lambda s: (s.subject, parse_date(s.date_str), int(s.blk)))
```

iii. The AI documented that 142 experiment entries reduce to 89 unique recording IDs, matching the paper's count. Sessions are sorted by subject and date for deterministic ordering.

## 1-d. How are the data split into trials?

i. Trials are defined by the `ft_trInd` frame-level trial index in the behavior data. The code iterates over trial indices 0 to `ntrials-1`, finding all frames belonging to each trial that also pass the corridor/running filter.

ii.
```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
    ft_trial_int = np.full(ft_trial.shape, -1, dtype=np.int64)
    finite_trial = np.isfinite(ft_trial)
    ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
    ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
    valid = finite_trial & ft_corr & ft_move
    masks = []
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
        if len(frame_idx) == 0:
            raise ValueError(f"Trial {trial} has no retained running corridor frames")
        masks.append(frame_idx)
    return masks
```

iii. The AI noted that trial splitting follows the reference code's pattern of using `ft_trInd` to identify which trial each frame belongs to, combined with quality filters.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered at the frame level: only frames where `ft_CorrSpc` is True (corridor space), `ft_move > 0` (animal is running), and `ft_trInd` is finite (valid trial assignment) are retained. All trials are kept (no trial-level rejection), but each trial's data consists only of the subset of frames passing these filters. If a trial has zero retained frames, an error is raised.

ii.
```python
valid = finite_trial & ft_corr & ft_move
```

iii. The AI justified this filtering by citing the paper's statement "We only considered timepoints during running for analysis" and the reference code's use of `ft_CorrSpc` and `ft_move > 0` masks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `<mouse>_<date>_<blk>_neural_data.npy` files, specifically the `spks` key which contains a list of neuron-block arrays (chunks). These are Suite2p deconvolved fluorescence traces. Retinotopy data from `<mouse>_<date>_trans.npz` (`iarea` field) is used for brain region assignment and neuron selection.

ii.
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
```

iii. The AI documented that the `spks` arrays are deconvolved activity traces from Suite2p, matching the paper's statement "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The `spks` chunks are concatenated along axis 0 to form a full neuron-by-frame matrix. A subset of neurons is then selected using a reference-style criterion: mHV neurons with top/bottom 5% d' for familiar stimulus selectivity (computed on odd running corridor frames), plus aHV reward-prediction neurons (d' >= 0.3 for late vs early cue position using position-interpolated activity). For each trial, only the selected neurons' activity at retained frames is extracted. Neural data is stored as float16.

ii.
```python
# Neuron selection in compute_selected_neurons:
spk = np.concatenate(spk_chunks, axis=0)
# ... d' computation on odd-trial corridor frames ...
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
# ... aHV reward-prediction selection ...
keep_mask = mhv_mask | ahv_mask

# Per-trial extraction:
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The AI stated this matches the reference `Get_coding_direction` function's neuron selection logic. The subsetting was motivated by memory constraints of the decoder training code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data filtering has two components: (1) neuron-level: only neurons in mHV or aHV brain regions that pass stimulus-selectivity or reward-prediction criteria are retained; (2) frame-level: only running corridor frames (`ft_CorrSpc & ft_move > 0`) are retained. The `corr_neu` criterion further requires that a neuron's mean activity in a stimulus condition exceeds its mean gray-space activity.

ii.
```python
corr_neu = (spk[:, stim_pos_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)) | (
    spk[:, stim_neg_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)
)
mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
```

iii. The AI documented that the reference code uses task-specific neuron curation rather than a global quality filter, and that the paper does not describe additional neuron-quality thresholds beyond Suite2p processing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). For each trial, the retained frames are the running corridor frames identified by `ft_CorrSpc & ft_move > 0` within that trial's `ft_trInd` assignment. The first retained frame of each trial corresponds to the earliest running corridor frame, which is approximately corridor entry. The `off_start` metadata is set to 0.0 and `off_end` to None.

ii.
```python
# Trial masks define frame indices per trial:
frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
# Neural is sliced at these frames:
trial_chunks.append(chunk[local_rows][:, frame_idx])
```

iii. The AI set `temporal_alignment_event` to "corridor entry (trial start)" and `off_start` to 0.0, consistent with the instructions specifying alignment to trial start (corridor entry).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate, approximately 314.69 ms per frame (~3.17 Hz). No temporal rebinning is applied; the data retains the original frame-level resolution.

ii.
```python
"time_bin_size": float(np.median([
    np.median(np.diff(np.asarray(
        load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float
    ))) * 86400.0
    for s in sessions
]) * 1000.0),
```

iii. The AI documented the frame rate as 3.17 Hz from the reference notebook, and chose to preserve native frame resolution rather than rebinning, noting this is consistent with the reference frame-based deconvolved traces.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from the `ft` (frame timestamps in days) and `SoundTime` (per-trial sound cue time in days) fields of the behavior dictionary.

ii.
```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The AI identified `SoundTime` as the cue timing variable and `ft` as the frame timestamp, converting from days to seconds by multiplying by 86400.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the signed time difference `SoundTime[trial] - ft[frame]` is computed and converted from days to seconds (multiply by 86400). This produces positive values before the cue and negative values after.

ii.
```python
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The AI documented this as a straightforward time difference, noting the sign convention: positive before cue, negative after.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to sound cue is computed at each retained frame index, so it is naturally aligned with the neural data frame-by-frame. Both share the same `frame_idx` array for each trial.

ii.
```python
current_ft = ft[frame_idx]  # same frame_idx used for neural
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. Alignment is implicit through shared frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session date string (`datexp` field in metadata) for each mouse. The first recording date per mouse serves as day 1.

ii.
```python
def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    by_subject: dict[str, list[SessionRef]] = defaultdict(list)
    for sess in sessions:
        by_subject[sess.subject].append(sess)
    day_map: dict[str, dict[str, float]] = defaultdict(dict)
    for subject, sess_list in by_subject.items():
        first_date = min(parse_date(s.date_str) for s in sess_list)
        for sess in sess_list:
            day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
    return day_map
```

iii. The AI computed day of training as calendar-day offset from each mouse's first recording date plus 1 (so day 1 is the first session).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date string is parsed, the earliest date per subject is found, and the day offset is `(session_date - first_date).days + 1`. This value is constant per session and broadcast across all frames within each trial.

ii.
```python
day_value = np.float32(day_map[sess.subject][sess.rec_id])
day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)
```

iii. The AI noted this is a derived quantity not directly present in the reference code but required by the decoder task specification.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `ft` (frame timestamps) and `Trial_start_time` (per-trial start time) fields in the behavior dictionary.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The AI identified `Trial_start_time` as the trial start event and `ft` as frame timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the elapsed time `ft[frame] - Trial_start_time[trial]` is computed and converted from days to seconds. Values are always non-negative since frames occur after trial start.

ii.
```python
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Verified in sanity checks that `time_since_trial_start` is monotonic within every converted trial.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at the same retained frame indices used for neural data, so alignment is automatic.

ii.
```python
current_ft = ft[frame_idx]  # shared with neural
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Same frame-level alignment as all other time-varying variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from the `isRew` field in the behavior dictionary, which is a per-trial boolean/binary array indicating whether reward was available in each trial.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=float)
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The AI noted that reward fraction varies across sessions since many sessions are entirely unrewarded, and that per-trial `isRew` must be used rather than assuming a fixed fraction.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value is simply broadcast across all retained frames of each trial. No additional processing or transformation is applied.

ii.
```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. Straightforward per-trial constant replicated across time.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from the `WallName` field in the behavior dictionary, which contains per-trial string labels for the visual stimulus (e.g., "circle1", "leaf1", "rock1").

ii.
```python
wall_name = as_str_array(beh["WallName"])
stim_code = np.full(
    len(frame_idx),
    category_to_idx[str(wall_name[trial_idx])],
    dtype=np.int16,
)
```

iii. The AI chose `WallName` over `stim_id` because `stim_id` has NaN values in some sessions, while `WallName` is consistently available. This is documented as a deliberate decision.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` strings across all sessions are collected and sorted alphabetically. Each string is mapped to an integer index. The per-trial string label is converted to its integer code and broadcast across all retained frames. 15 unique categories are present in the full dataset.

ii.
```python
categories = sorted({
    str(v)
    for sess in sessions
    for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
})
category_to_idx = {name: idx for idx, name in enumerate(categories)}
```

iii. Per-trial constant, replicated across timepoints. The 15 categories include familiar, novel, swap, and grating stimuli.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` (frame indices of lick events) and `LickTrind` (trial index for each lick event) in the behavior dictionary.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
```

iii. The AI identified these as the raw lick event fields used across the reference code's lick-related utility functions.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frame indices belonging to that trial are extracted. A binary vector is created: 1 if a retained frame index appears in the trial's lick frames, 0 otherwise. This produces a time-varying binary licking signal.

ii.
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The AI documented this as matching the reference code's pattern of using `LickFr` and `LickTrind` for lick detection.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is computed at the same retained frame indices as neural data. A lick is registered at a frame if any lick event's frame index matches that retained frame.

ii.
```python
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. Frame-level alignment through shared indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos`, a frame-level position variable in the behavior dictionary, representing the animal's position in the corridor in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. The AI identified `ft_Pos` as the raw position variable and noted positions are in decimeters (0-40 dm for the 4 m corridor).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values at retained frames are clipped to [0, 39.999] dm, then discretized into 4 equal 10-dm (1 m) bins via integer division by 10.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The AI documented the 4 bins as matching the instruction's "4 equal-length, 1-m-long spatial bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The corridor (0-40 dm = 0-4 m) is divided into four 10-dm bins: [0,10), [10,20), [20,30), [30,40]. Bin assignment is via floor division by 10, clipped to range [0,3]. Bins are labeled "0-1m", "1-2m", "2-3m", "3-4m".

ii.
```python
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
# output_values: ["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. Position bin distribution is approximately uniform (~25% each), which is expected since only running corridor frames are retained.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted at the same retained frame indices as neural data, providing frame-level alignment.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. Same frame-level alignment as all other variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed`, a frame-level running speed variable in the behavior dictionary.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The AI identified `ft_RunSpeed` as the raw speed variable from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartile edges are computed across all retained frames from all sessions in a preliminary pass. Then each frame's speed value is digitized into one of 4 bins using these edges.

ii.
```python
# Global quartile computation in build_global_metadata:
speed_values = np.concatenate(speed_values).astype(np.float32)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The AI documented that global quartiles ensure each bin contains ~25% of retained samples, matching the instruction's "4 bins, each corresponding to 25% of the data."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three quantile edges at 25%, 50%, 75% define four bins. `np.digitize` assigns each speed value to a bin (0-3). Bins are labeled "q1", "q2", "q3", "q4".

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
# output_values: ["q1", "q2", "q3", "q4"]
```

iii. The distribution is verified to be approximately 25% per bin in the full dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is extracted at the same retained frame indices as neural data.

ii.
```python
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Frame-level alignment through shared indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled: (1) NaN values in `ft_trInd` are excluded by the `np.isfinite` check; (2) NaN values in `stim_id` are avoided by using `WallName` instead; (3) behavior entries with `stimtype` suffixes are resolved by validating duplicates agree on key fields; (4) if a trial has zero retained frames after filtering, an error is raised; (5) if neuron selection yields zero neurons, a fallback selects the strongest mHV neurons by absolute d'; (6) if fewer than 128 mHV neurons pass the 5% percentile threshold, the top 128 by absolute d' are taken instead.

ii.
```python
# NaN handling in ft_trInd:
finite_trial = np.isfinite(ft_trial)
ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)

# Fallback for too few mHV neurons:
if int(np.sum(mhv_mask)) < 128:
    mhv_candidates = np.where(corr_neu & areas["mHV"])[0]
    order = np.argsort(np.abs(stim_dp[mhv_candidates]))[::-1]
    take = mhv_candidates[order[: min(128, len(order))]]

# Zero-neuron fallback:
if int(np.sum(keep_mask)) == 0:
    mhv_candidates = np.where(areas["mHV"])[0]
    order = np.argsort(np.abs(stim_dp[mhv_candidates]))[::-1]
    take = mhv_candidates[order[: min(128, len(order))]]
    keep_mask[take] = True
```

iii. The AI documented the chunk-local indexing bug found in Step 10 and its fix, plus edge cases like position clipping and empty trial handling.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the reward-prediction neuron selection for sessions with rewarded trials. This involves position interpolation (`get_interpPos_spk`) of all aHV neurons across the full session, which took up to ~47 seconds for a single session (VR2_2021_04_11_1). The full conversion took 1013.7 seconds (~17 minutes) across 89 sessions.

ii.
```python
# Expensive aHV interpolation:
interp_spk = utils.get_interpPos_spk(
    spk[ahv_idx][:, move_idx],
    poscum_move,
    int(beh["ntrials"]),
    n_bins=60,
    lengths=float(beh["Corridor_Length"]),
)
```

iii. The AI identified this in Step 7 and noted that restricting interpolation to aHV neurons only (rather than all neurons) was a speedup.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_dataset` iterates over all trials, extracting neural data and computing inputs/outputs for each. While some parts (like `np.isin` for licking) are vectorized within each trial, the outer trial loop and the chunk-based neural extraction could potentially be vectorized. The `compute_trial_masks` function also loops over trials.

ii.
```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        if len(local_rows) == 0:
            continue
        trial_chunks.append(chunk[local_rows][:, frame_idx])
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The AI noted this loop structure but did not vectorize it, likely because variable-length trials make full vectorization impractical.

## 12-c. What processing does the code repeat multiple times?

i. The code loads and parses behavior data multiple times: once in `build_global_metadata` to compute speed quartiles and subject metadata, and again per session in `convert_dataset`. `compute_trial_masks` is called both in `build_global_metadata` (for speed value collection) and again in the main conversion loop. The `WallName` categories are also collected twice (once for category list, once per session).

ii.
```python
# In build_global_metadata:
trial_masks = compute_trial_masks(beh)
speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])

# In convert_dataset (again per session):
trial_masks = compute_trial_masks(beh)
```

iii. The AI acknowledged the two-pass structure (metadata pass + conversion pass) but did not eliminate the duplication, likely prioritizing correctness over efficiency.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The neuron selection procedure (d' computation, reward-prediction interpolation, percentile thresholding) is specific to the AI's chosen subset approach and is not required by the instructions. The reference code's neuron selection (`Get_coding_direction`, `Get_dprime_rewPred_neuron`) was designed for specific analyses in the paper, not for a general-purpose decoder. The instructions do not specify neuron selection beyond matching the reference processing, and the reference `Get_dprime_selective_neuron` simply computes d' for all neurons without selecting a subset. Additionally, the `choose_canonical_behavior` function's validation of duplicate experiment entries, while defensive, performs redundant checks that don't affect the output.

ii.
```python
# Entire compute_selected_neurons function is custom neuron selection
# not directly required by instructions
def compute_selected_neurons(spk_chunks, beh, iarea, session_id):
    ...
```

iii. The AI justified neuron selection as necessary for memory constraints of the decoder training code, but this selective subsetting was not required by the instructions and introduces a deviation from the reference processing.
