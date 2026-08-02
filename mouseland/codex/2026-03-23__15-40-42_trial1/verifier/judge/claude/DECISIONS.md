# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all `Beh_*.npy` files in `data/beh/`, extracting session records keyed by `<mouse>_<date>_<blk>` identifiers. It deduplicates to 89 unique recording bases (matching the paper's "89 recordings in 19 mice"). For each session, it loads neural data from `data/spk/<base>_neural_data.npy` and retinotopy from `data/retinotopy/<mouse>_<date>_trans.npz`. Behavior-only pretraining data in `Unsupervised_pretraining_behavior/` is excluded since there are no matching neural recordings. Experiment metadata from `Imaging_Exp_info.npy` is used to annotate experiment types.

ii.
```python
def select_representative_sessions(root: Path) -> list[SessionSpec]:
    beh_dir = root / "data" / "beh"
    base_to_experiments = load_experiment_type_map(root)
    selected: dict[str, SessionSpec] = {}
    for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
        exp_type = beh_path.stem.replace("Beh_", "")
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for key, record in beh_dict.items():
            base = "_".join(key.split("_")[:5])
            ...
```

```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```

iii. The AI documented in CONVERSION_NOTES.md that there are 142 experiment-session records across 23 experiment types, collapsing to 99 unique behavior session keys and 89 unique neural recordings. It chose to use 89 unique recording bases as sessions to match the paper's count and avoid data duplication.

## 1-b. How are the data split into subjects (mice)?

i. Subject names are parsed from the first component of the recording base string (e.g., `TX108` from `TX108_2023_03_25_1`). Unique subjects are collected and sorted alphabetically to create a deterministic `subjects` list and `subject_idx` mapping.

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    blk = parts[4]
    return subject, date, blk
```

```python
subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
```

iii. The AI noted that 19 unique imaging mice are expected from the paper and data, and confirmed this count in its validation. The approach matches how the reference code identifies mice via `db['mname']`.

## 1-c. How are the data split into sessions?

i. Each unique recording base `<mouse>_<date>_<blk>` is treated as one session. When multiple behavior keys share the same recording base (e.g., plain key and swap variants), a scoring function selects the best representative key, preferring keys without "swap" and with more valid stimulus IDs.

ii.
```python
def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    stim_id = np.asarray(record.get("stim_id", []), dtype=float)
    non_nan_stim = int(np.isfinite(stim_id).sum()) if stim_id.size else 0
    unique_walls = int(len(np.unique(record.get("WallName", []))))
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)
```

iii. The AI justified this by noting that different experiment labels can reference the same neural recording, and that choosing one representative behavior key per base avoids data leakage. This yields 89 sessions matching the paper.

## 1-d. How are the data split into trials?

i. Trials are identified using the frame-level `ft_trInd` array, which assigns each imaging frame to a trial index. For each of the `ntrials` trials in a session, frame indices belonging to that trial are collected, filtered to running corridor frames.

ii.
```python
def build_trial_frame_indices(record: dict, nfr: int) -> list[np.ndarray]:
    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    valid_idx = np.flatnonzero(valid)
    trial_ids = ft_tr[valid].astype(int)
    keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
    keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
    ntrials = int(record["ntrials"])
    frame_indices: list[np.ndarray] = []
    for trial in range(ntrials):
        trial_frames = valid_idx[keep & (trial_ids == trial)]
        frame_indices.append(trial_frames.astype(np.int32, copy=False))
    return frame_indices
```

iii. The AI documented that it uses `ft_trInd` for trial assignment rather than `StartFr`/`EndFr`, matching the reference code convention. Only running corridor frames (`ft_CorrSpc & ft_move > 0`) are retained per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must have at least 5 retained timepoints (`MIN_TRIAL_TIMEPOINTS = 5`), (2) retained duration since trial start must be < 60 seconds, (3) maximum interframe gap must be < 10 seconds. Sessions with fewer than 2 valid trials after filtering are excluded entirely.

ii.
```python
def trial_passes_quality_filters(record, trial_idx, frame_idx, nfr):
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    frame_idx = frame_idx[frame_idx < nfr]
    if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
        return False, "too_few_timepoints"
    frame_times = np.asarray(record["ft"][:nfr], dtype=float)[frame_idx]
    retained_duration_s = float((frame_times[-1] - float(record["Trial_start_time"][trial_idx])) * SEC_PER_DAY)
    if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
        return False, "retained_duration_gt_60s"
    if frame_idx.size > 1:
        max_gap_s = float(np.max(np.diff(frame_times)) * SEC_PER_DAY)
        if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
            return False, "interframe_gap_gt_10s"
    return True, None
```

iii. The AI documented that 2,217 trials were removed by the duration and gap filters (reducing from 38,110 to 35,893 trials). These were described as "decoder-pathological stalled trials" with sparse late running segments. The corridor + running filters match the reference code; the duration and gap filters are additional quality controls not present in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` arrays stored in `<base>_neural_data.npy` files. Each file contains a dict with key `spks`, which is a list of arrays (one per imaging plane) that are concatenated along the neuron dimension.

ii.
```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```

iii. The AI confirmed in CONVERSION_NOTES.md that these are deconvolved fluorescence traces from Suite2p processing and that no delta-F/F computation is needed, consistent with the paper and reference code.

## 2-b. How is the `neural` data processed?

i. After loading and concatenating planes, the AI applies a neuron selection step: it computes stimulus d-prime on running corridor frames, identifies corridor-responsive neurons (activity > gray space activity), then selects the top 5% positive and 5% negative d' neurons per visual area (V1, mHV, lHV, aHV). This reduces the neuron count from ~50K to ~3.4K per session on average.

ii.
```python
def select_decoder_neurons(spk, record, region_idx):
    nfr = spk.shape[1]
    stim_pair = choose_reference_stimuli(record)
    if stim_pair is None:
        return np.arange(spk.shape[0], dtype=np.int32)
    ...
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (
        (np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
        | (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
    )
    selected = np.zeros(spk.shape[0], dtype=bool)
    for area in range(4):
        candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
        ...
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))
    return np.flatnonzero(selected).astype(np.int32)
```

iii. The AI justified this as following the reference coding-direction analysis (`Get_coding_direction`, `Get_sort_spk`) which uses top 5% positive/5% negative selective neurons per area. However, this selection is analysis-specific in the reference code and is not a general neuron filtering step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two quality filters are applied: (1) only corridor-responsive neurons (mean activity in corridor > mean activity in gray space), and (2) the top 5%/bottom 5% d' selection per visual area. Additionally, neurons not assignable to V1/mHV/lHV/aHV are excluded.

ii. (Same `select_decoder_neurons` code as 2-b above)

iii. The AI noted in CONVERSION_NOTES.md that Suite2p already classifies cells and the reference code has no additional global neuron quality filter, but applied the analysis-specific selectivity filter for "tractable full-dataset decoding."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by selecting frames belonging to each trial via `ft_trInd`. Only running frames inside the textured corridor (`ft_CorrSpc & ft_move > 0`) are retained. The first retained frame of each trial is near the corridor entry point.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The AI set `temporal_alignment_event` to "corridor entry / trial start" and `off_start` to 0.0, indicating alignment starts at trial onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame rate (~314.7 ms median) without any temporal rebinning. The `time_bin_size` metadata field is set to the median frame interval across sessions.

ii.
```python
metadata = {
    ...
    "time_bin_size": float(np.median(frame_dt_medians)),
    ...
    "frame_bin_source": "native imaging frame timestamps; no temporal resampling",
}
```

iii. The AI documented that the paper does not state a fixed ms bin size, and that keeping native frame timestamps is more faithful than interpolation to a synthetic time grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundTime` (per-trial sound cue timestamp in MATLAB datenum format) and `ft` (per-frame timestamps in MATLAB datenum format).

ii.
```python
def session_trial_info(record, trial, frame_idx):
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
    ...
    return time_to_cue.astype(np.float32), ...
```

iii. The AI documented that `SoundTime` contains the cue timing for each trial and that continuous time-to-cue is appropriate for the decoder input.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, computes `(SoundTime[trial] - frame_time) * 86400` to convert from MATLAB datenum difference to seconds. Positive values indicate frames before the cue; negative values indicate frames after.

ii. (Same `session_trial_info` code as 3-a)

iii. The AI's sign convention (positive before cue, negative after) is documented in the metadata.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Computed at exactly the same retained imaging frames as the neural data, producing one value per timepoint.

ii.
```python
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
input_trial = np.vstack([time_to_cue, training_day, time_since_start, reward_available])
```

iii. Both neural and input arrays index into the same `frame_idx`, ensuring perfect temporal alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session date parsed from the recording base name (e.g., `2023_03_25` from `TX108_2023_03_25_1`). The first session date for each mouse is tracked to compute relative days.

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    ...
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    ...

specs = sorted(selected.values(), key=lambda s: (s.subject, s.date, int(s.blk)))
first_date_by_subject: dict[str, datetime] = {}
for spec in specs:
    first_date_by_subject.setdefault(spec.subject, spec.date)
    spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. The AI noted this variable is not in the paper but is required by the decoder task. It uses elapsed days since the mouse's first retained imaging session as a reproducible continuous proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Computes `(session_date - first_session_date_for_mouse).days` as a float. This per-trial scalar is broadcast to all timepoints.

ii.
```python
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. Values range from 0.0 to 92.0 across the dataset, reflecting the span of recording dates.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does not have a separate "Environment type" input. Instead, it has `reward_available` as input[3], derived from the per-trial `isRew` array in the behavior data. This binary variable indicates whether the trial is in a rewarded corridor (1) or not (0), effectively encoding the environment type relevant to the animal's task.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. The AI followed the decoder task specification which lists "Reward availability: 1 if in rewarded corridor, 0 if not" as the fourth decoder input, rather than a separate "Environment type" variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The `isRew` per-trial boolean is converted to float (0.0 or 1.0) and broadcast across all timepoints of the trial. No additional processing.

ii. (Same code as 4-a above)

iii. The AI documented that unsupervised and naive sessions have `isRew.sum() == 0`, meaning all trials in those sessions have `reward_available = 0`.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `ft` (per-frame timestamps) and `Trial_start_time` (per-trial start timestamps), both in MATLAB datenum format.

ii.
```python
def session_trial_info(record, trial, frame_idx):
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    ...
    time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
    return ..., time_since_start.astype(np.float32)
```

iii. The AI noted this should always be non-negative since frames come after trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame: `(frame_time - Trial_start_time[trial]) * 86400` to convert datenum difference to seconds. Values are always non-negative.

ii. (Same code as 5-a)

iii. Range is [0.0, 54.3] seconds across the full dataset after quality filtering.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at exactly the same retained imaging frames as neural data, ensuring frame-level alignment.

ii.
```python
input_trial = np.vstack([time_to_cue, training_day, time_since_start, reward_available])
```

iii. Same frame indices used for neural, input, and output arrays.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from the per-trial `isRew` boolean array in the behavior data.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. The AI documented that `isRew` indicates whether reward is available in that corridor on that trial. Unsupervised sessions have all zeros.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial boolean `isRew[trial_idx]` is converted to float (0.0 or 1.0) and repeated for every timepoint in the trial.

ii. (Same code as 6-a)

iii. Per the decoder task specification: "1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from the per-trial `WallName` array and the session-level `UniqWalls` array. A global list of all unique wall names across all retained sessions is constructed.

ii.
```python
def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories

visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}
stim_idx = np.full(frame_idx.size, visual_to_idx[str(spec.record["WallName"][trial_idx])], dtype=np.int16)
```

iii. The AI found 15 unique visual stimulus categories across the imaging data. Each trial is assigned a single categorical index based on its wall name.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` is mapped to an integer index in the global sorted list of unique wall names. This per-trial index is repeated across all timepoints. 15 categories are used.

ii. (Same code as 7-a)

iii. The AI documented the 15 categories: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` (frame indices of lick events) and `LickTrind` (trial index for each lick event).

ii.
```python
def build_lick_frame_lookup(record, nfr):
    lick_frames = np.asarray(record["LickFr"], dtype=float)
    lick_trial = np.asarray(record["LickTrind"], dtype=float)
    valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
    lick_frames = lick_frames[valid].astype(int)
    lick_trial = lick_trial[valid].astype(int)
    ...
    lookup: dict[int, np.ndarray] = {}
    for trial in np.unique(lick_trial):
        lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
    return lookup
```

iii. The AI documented that lick events are matched to imaging frames using `LickFr` indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each retained frame in a trial, checks if that frame index appears in the trial's lick frame set. Produces a binary vector (0 = no lick, 1 = lick) at imaging frame resolution.

ii.
```python
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The overall lick rate across the full dataset is ~3.8% of retained frames, which is sparse as expected since retained frames are running corridor frames and mice mainly lick around the reward zone.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick events are identified at the same imaging frame indices used for neural data. The `np.isin(frame_idx, lick_frames)` call ensures exact frame-level alignment.

ii. (Same code as 8-b)

iii. Both neural and licking data use the same `frame_idx` array per trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos`, the per-frame position in the corridor in decimeters.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Raw positions are in decimeters; the corridor is 40 decimeters (4 meters) long.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Raw `ft_Pos` values are read at retained frame indices. No interpolation or smoothing.

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Position values are clipped to [0, 39.999] decimeters before binning.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Positions are divided into 4 equal 1-meter bins by integer-dividing decimeter values by 10: bin 0 = 0-1m, bin 1 = 1-2m, bin 2 = 2-3m, bin 3 = 3-4m. Clipped to valid range.

ii. (Same `digitize_position` code as 9-b)

iii. The resulting distribution is near-uniform (~25% each) across the full dataset, consistent with expected corridor traversal patterns.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position values are read at the same retained frame indices as neural data, ensuring frame-level alignment.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Same `frame_idx` used for neural, input, and output.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed`, the per-frame running speed.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Running speed is a behavioral variable recorded alongside imaging frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Raw `ft_RunSpeed` values are read at retained frame indices. No smoothing or interpolation.

ii.
```python
def digitize_speed(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. The AI computes global quartile edges from all retained running frames across all sessions before applying binning.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed at [25%, 50%, 75%] from all retained running frames across all retained sessions. Speeds are binned into 4 quartile categories (q1, q2, q3, q4) using `np.searchsorted`.

ii.
```python
speed_values = np.concatenate(all_speeds).astype(np.float32, copy=False)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```

iii. The resulting distribution is exactly [0.25, 0.25, 0.25, 0.25] by construction, matching the decoder task specification of "4 bins, each corresponding to 25% of the data."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are read at the same retained frame indices as neural data.

ii. (Same code as 10-a)

iii. Same `frame_idx` for all data streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Behavior frame arrays are truncated to the neural frame count (`[:nfr]`), since behavior arrays are 1-3 frames longer
- NaN values in `ft_trInd` are excluded via `np.isfinite` check
- NaN values in `LickFr` and `LickTrind` are filtered out
- Trials with fewer than 5 retained timepoints are excluded
- Trials with retained duration > 60s are excluded (sparse stalled trials)
- Trials with interframe gaps > 10s are excluded
- Sessions with fewer than 2 valid trials are excluded
- When no valid stimulus pair is found for neuron selection, all mapped neurons are kept as fallback

ii.
```python
ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
valid = np.isfinite(ft_tr)
...
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
...
if stim_pair is None:
    return np.arange(spk.shape[0], dtype=np.int32)
```

iii. The AI documented that 2,217 trials were removed by the additional quality filters and that no sessions dropped below 2 valid trials. The behavior-neural frame mismatch is handled by the standard `[:nfr]` truncation matching the reference code.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the full neural recording files (up to 89,577 neurons x ~29K frames per session). The neuron selection step (`select_decoder_neurons`) also requires loading the full spike matrix to compute d-prime, adding significant computation. The full conversion takes ~635 seconds for 89 sessions.

ii.
```python
spk = load_spike_matrix(root, spec.base)  # loads full neuron x frame matrix
...
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)  # computes d' on full matrix
```

iii. The AI estimated ~10.4 seconds per session and noted that I/O is the primary bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_sessions` iterates over each trial to construct neural, input, and output arrays. Some of this (e.g., computing time_to_cue and time_since_start for all frames at once, then splitting) could be vectorized. The per-area loop in `select_decoder_neurons` (4 iterations) is minor.

ii.
```python
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...
    neural_trial = spk[:, frame_idx]
    time_to_cue, time_since_start = session_trial_info(...)
    ...
```

iii. The AI focused optimization on minimizing I/O and doing retinotopy/lick processing once per session rather than per trial.

## 12-c. What processing does the code repeat multiple times?

i. Trial quality checks are performed twice: once in `prepare_session_specs` (to compute speed quartiles) and again in the main `process_sessions` loop. Running speed values are also collected twice for the same reason. The frame index clipping (`frame_idx < nfr`) is done in `build_trial_frame_indices` and again in the main loop.

ii.
```python
# In prepare_session_specs:
if keep_trial and trial_passes_quality_filters(...)[0]:
    all_speeds.append(run_speed[frame_idx])

# In process_sessions:
keep_trial, remove_reason = trial_passes_quality_filters(...)
```

iii. The AI noted this redundancy but prioritized correctness over micro-optimization.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The primary unnecessary processing is the neuron selection step (`select_decoder_neurons`), which computes d-prime and selectivity metrics for all ~50K neurons per session, then discards ~93% of them. This is computationally expensive and not required by the decoder format or the instructions. The reference code only applies this filter for specific paper analyses (coding direction, sequence sorting), not for general data export.

Additionally, metadata about removed sessions/trials, experiment types, and source behavior keys is computed and stored but not used by the decoder.

ii.
```python
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
```

iii. The AI justified neuron selection as necessary for "tractable full-dataset decoding" to reduce the pickle file size, but the decoder `train_decoder.py` could handle larger datasets.
