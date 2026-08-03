# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories under `data/`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. It first loads `Imaging_Exp_info.npy` to get experiment metadata, then iterates over all `Beh_*.npy` files in the behavior directory to collect session records. For each session, the spike file and retinotopy file are loaded individually. The AI also loads all behavior files upfront during the `select_representative_sessions` phase to pick the best behavior key per recording base.

ii.
```python
# Loading experiment info
exp_info = np.load(root / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()

# Loading behavior files
for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
    beh_dict = np.load(beh_path, allow_pickle=True).item()
    for key, record in beh_dict.items():
        ...

# Loading spike data
spk_item = np.load(path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)

# Loading retinotopy
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=float)
```

iii. The AI loads all behavior files during session selection to pick the best representative key per recording base (preferring non-swap keys with more unique walls and non-NaN stim_id values). This approach ensures each unique recording base gets the best behavior key before processing begins.

## 1-b. How are the data split into subjects?

i. The mouse name is parsed from the recording base string (first component of `mouse_YYYY_MM_DD_blk`). Subjects are collected as sorted unique names across all session specs, and `subject_idx` maps each session to its index in this sorted list.

ii.
```python
subject, date, blk = parse_base(base)
# where parse_base splits: parts = base.split("_"); subject = parts[0]

subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
subject_idx.append(subject_to_idx[spec.subject])
```

iii. The subject is directly available from the recording base identifier. 19 subjects are found.

## 1-c. How are the data split into sessions?

i. A session corresponds to a unique recording base (`mouse_YYYY_MM_DD_blk`), matching the paper's "89 recordings." When multiple behavior keys exist for the same recording base (e.g., swap variants), the AI selects one representative key using a scoring function that prefers non-swap keys with more stim_id coverage.

ii.
```python
base = "_".join(key.split("_")[:5])
# Deduplication: one SessionSpec per base
if base not in selected:
    selected[base] = SessionSpec(...)
else:
    current = selected[base]
    if session_key_score(key, record) < session_key_score(current.key, current.record):
        current.key = key
        current.record = record
```

iii. The AI explicitly deduplicates to 89 unique recording bases, matching the paper's statement of "89 recordings in 19 mice."

## 1-d. How are the data split into trials?

i. Trials are split using the frame-level `ft_trInd` array. For each trial (0 to `ntrials-1`), the AI finds frames where `ft_trInd == trial`, `ft_CorrSpc` is true, AND `ft_move > 0` (running frames only). The resulting frame indices are variable-length per trial.

ii.
```python
def build_trial_frame_indices(record: dict, nfr: int) -> list[np.ndarray]:
    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    valid_idx = np.flatnonzero(valid)
    trial_ids = ft_tr[valid].astype(int)
    keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
    keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
    ...
    for trial in range(ntrials):
        trial_frames = valid_idx[keep & (trial_ids == trial)]
        frame_indices.append(trial_frames.astype(np.int32, copy=False))
```

iii. The AI uses both corridor space mask AND running mask (`ft_move > 0`), restricting to running frames in the textured corridor, matching what the paper describes for its analyses.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three quality criteria: (1) minimum 5 retained timepoints, (2) retained trial duration must be <= 60 seconds, (3) maximum inter-frame gap must be <= 10 seconds. Sessions with fewer than 2 valid trials are also excluded.

ii.
```python
MIN_TRIAL_TIMEPOINTS = 5
MAX_RETAINED_TRIAL_DURATION_S = 60.0
MAX_RETAINED_INTERFRAME_GAP_S = 10.0

def trial_passes_quality_filters(record, trial_idx, frame_idx, nfr):
    ...
    if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
        return False, "too_few_timepoints"
    retained_duration_s = float((frame_times[-1] - float(record["Trial_start_time"][trial_idx])) * SEC_PER_DAY)
    if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
        return False, "retained_duration_gt_60s"
    if frame_idx.size > 1:
        max_gap_s = float(np.max(np.diff(frame_times)) * SEC_PER_DAY)
        if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
            return False, "interframe_gap_gt_10s"
    return True, None
```

iii. These filters were added during Step 10 to handle extreme outliers (e.g., trials spanning 1,765 seconds). 2,217 trials were removed. The reference solution does not apply these filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` (deconvolved calcium traces), concatenated across imaging planes. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```

iii. The neural data source matches the reference: concatenated deconvolved fluorescence traces from Suite2p.

## 2-b. How is the `neural` data processed?

i. The AI applies a selective neuron subset using the paper's d-prime selectivity logic: for each session, it computes stimulus d' on running corridor frames, identifies corridor-responsive neurons, and keeps the top 5% positive and 5% negative d' neurons per visual area. The selected neurons' traces are then stored as float32 for the retained frame indices (variable length per trial, no padding).

ii.
```python
def select_decoder_neurons(spk, record, region_idx):
    # ... compute d-prime between two reference stimuli
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (
        (np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
        | (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
    )
    selected = np.zeros(spk.shape[0], dtype=bool)
    for area in range(4):
        candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))
    ...

# Per-trial extraction:
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The AI justifies the neuron selection as following the paper's coding-direction analysis methodology. This reduces the dataset from ~4.7M to ~304K neurons, making decoding tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) neurons outside the four visual areas (V1, mHV, lHV, aHV) are excluded via retinotopy, and (2) a further d-prime-based selectivity filter keeps only the top/bottom 5% of stimulus-selective, corridor-responsive neurons per area.

ii.
```python
# Region assignment
region_idx = np.full(nneurons, 4, dtype=np.int16)
region_idx[iarea == 8] = 0       # V1
region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1  # mHV
region_idx[np.isin(iarea, [5, 6])] = 2         # lHV
region_idx[np.isin(iarea, [3, 4])] = 3         # aHV

# Then selectivity filter in select_decoder_neurons()
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
```

iii. The reference solution only applies the area filter (keeping all neurons in the 4 visual areas). The AI adds an additional selectivity-based neuron filter not required by the instructions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Trials contain only the running frames within the corridor space (`ft_CorrSpc & ft_move > 0`). The trial length is variable (no fixed window or padding). The AI stores each trial's frames as-is.

ii.
```python
# Frame indices are built from corridor+running mask
trial_frames = valid_idx[keep & (trial_ids == trial)]

# Neural data extracted per trial without padding
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The reference solution uses a fixed window of N_FRAMES=32, padding shorter trials and truncating longer ones. The AI instead stores variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native imaging frame rate (~3.17 Hz, ~314.7 ms per frame) is preserved. The AI reports the median frame dt across sessions as the `time_bin_size` in metadata.

ii.
```python
"time_bin_size": float(np.median(frame_dt_medians)),
```

iii. Both the AI and reference preserve native frame timestamps without resampling.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (the timestamp of every imaging frame).

ii.
```python
def session_trial_info(record, trial, frame_idx):
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
    ...
```

iii. The AI uses `SoundTime` (absolute timestamp) rather than `SoundFr` (frame number) used by the reference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, compute `(SoundTime[trial] - frame_time) * SEC_PER_DAY` in seconds. Positive values mean before cue, negative values mean after cue, matching the "time TO" semantics.

ii.
```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. The reference uses `SoundFr` interpolated onto frame time axis, then computes `cue_time - frame_time`. Both approaches compute the same quantity but from different source variables.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the frame times of the same retained frame indices used for the neural data of that trial.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. All data streams use the same frame indices per trial, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date parsed from the recording base string, relative to the first session date for that mouse.

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    blk = parts[4]
    return subject, date, blk

# In select_representative_sessions:
first_date_by_subject: dict[str, datetime] = {}
for spec in specs:
    first_date_by_subject.setdefault(spec.subject, spec.date)
    spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. The AI uses elapsed calendar days since the mouse's first session, not session ordinal.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Compute elapsed calendar days between the session date and the mouse's first session date. This gives a continuous value (e.g., 0, 7, 9, 16, ...) rather than a session ordinal (0, 1, 2, 3, ...). The value is broadcast across all timepoints of each trial.

ii.
```python
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
# Then in processing:
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. The reference counts session ordinals (0, 1, 2, ...), while the AI counts calendar days (0, 7, 9, ...). The AI's range is [0, 92] vs the reference's [0, 7].

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the absolute timestamp of trial start) and `ft` (the timestamp of every imaging frame).

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. The AI uses `Trial_start_time` rather than `StartFr` used by the reference.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, compute `(frame_time - Trial_start_time[trial]) * SEC_PER_DAY` in seconds. Values are always non-negative (starting near zero at trial start).

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. The reference uses `StartFr` interpolated onto the frame time axis. Both approaches should give equivalent results since `Trial_start_time` and `StartFr` encode the same event.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. It is computed from the frame times of the same retained frame indices used for the neural data of that trial.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. Same alignment mechanism as all other data streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. Same source variable as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast `isRew[trial]` to boolean then to float (0.0 or 1.0), broadcast across all retained timepoints of the trial.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. Same approach as the reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial, and `UniqWalls`, which lists the unique wall names in the session.

ii.
```python
stim_idx = np.full(
    frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])],
    dtype=np.int16,
)
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses 15 individual stimulus categories (circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5) collected from all `UniqWalls` across sessions. Each trial's `WallName` is mapped to its index in this sorted global list. The value is broadcast across all timepoints.

ii.
```python
def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories

# Visual categories (15): ['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', ...]
```

iii. The reference groups the 15 names into 4 broad categories (circle, leaf, rock, wood). The AI keeps all 15 individual names as separate categories, resulting in a 15-class classification problem instead of 4.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
def build_lick_frame_lookup(record: dict, nfr: int) -> dict[int, np.ndarray]:
    lick_frames = np.asarray(record["LickFr"], dtype=float)
    lick_trial = np.asarray(record["LickTrind"], dtype=float)
    valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
    lick_frames = lick_frames[valid].astype(int)
    lick_trial = lick_trial[valid].astype(int)
    ...
    for trial in np.unique(lick_trial):
        lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
    return lookup
```

iii. The reference uses only `LickFr` directly as frame indices. The AI additionally uses `LickTrind` to build a per-trial lookup.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary flag is computed per retained frame: 1 if the frame index appears in the lick lookup for that trial, 0 otherwise.

ii.
```python
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The reference builds a full-session lick array (`licking[lick_frame] = 1`) and slices it per trial. Both approaches should produce the same result.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is computed for the same retained frame indices as the neural data, ensuring alignment.

ii.
```python
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Same alignment mechanism as all other data streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)

position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Same source variable as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is clipped to [0, 39.999] and integer-divided by 10 to produce four 1-m bins (0-1m, 1-2m, 2-3m, 3-4m), clipped to [0, 3].

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Same binning logic as the reference (`// 10` then clip to [0, 3]).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins: 0-10 dm -> bin 0, 10-20 dm -> bin 1, 20-30 dm -> bin 2, 30-40 dm -> bin 3.

ii.
```python
POSITION_OUTPUT_VALUES = ["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. Same as the reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read from `ft_Pos` at the same retained frame indices as the neural data.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Same alignment mechanism as all other data streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Same source variable as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized into 4 bins using global quartile edges computed across all retained frames from all sessions. The edges are computed using `np.quantile` at [0.25, 0.5, 0.75], then `np.searchsorted` assigns each frame's speed to a bin.

ii.
```python
# Global quartile computation:
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

# Per-frame assignment:
def digitize_speed(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. The reference uses per-session rank-based quartiles, while the AI uses global quantile edges. The reference approach guarantees exactly 25% in each bin per session, while the global approach gives exactly 25% across all sessions but may not be balanced within individual sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four quartile bins labeled q1, q2, q3, q4 based on global speed edges [13.83, 26.68, 41.84].

ii.
```python
RUNNING_SPEED_OUTPUT_VALUES = ["q1", "q2", "q3", "q4"]
```

iii. The reference uses per-session quartiles based on rank ordering.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read from `ft_RunSpeed` at the same retained frame indices as the neural data.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Same alignment mechanism as all other data streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior arrays are truncated to the neural frame count (behavior can run a few frames past imaging). Lick frames past the neural frame count are excluded. Trials with NaN in `ft_trInd` frames are handled by the `np.isfinite` check. Trials with fewer than 5 retained frames, duration > 60s, or inter-frame gaps > 10s are excluded.

ii.
```python
# Truncate to neural frame count
nfr = spk.shape[1]
# Handle NaN in ft_trInd
ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
valid = np.isfinite(ft_tr)
# Handle lick frames past neural data
valid = (lick_frames >= 0) & (lick_frames < nfr)
```

iii. The reference handles the same behavior-neural length mismatch. The AI adds additional quality filters not present in the reference.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files (each file is several GB) and the neuron selection computation (d-prime calculation on full spike matrices).

ii.
```python
spk = load_spike_matrix(root, spec.base)  # Loading full spike matrix
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)  # d-prime computation
```

iii. The AI's neuron selection step adds significant computation per session beyond what the reference does.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_lick_frame_lookup` function loops over unique trial indices to build a per-trial lick dictionary, which could be replaced with a single vectorized groupby-style operation. The `select_decoder_neurons` loops over 4 brain areas for the percentile calculation.

ii.
```python
for trial in np.unique(lick_trial):
    lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
```

iii. These loops are minor compared to I/O cost.

## 12-c. What processing does the code repeat multiple times?

i. The `prepare_session_specs` function calls `trial_passes_quality_filters` for every trial during the speed-edge computation phase. Then during the main processing loop, the same quality filter is called again for every trial. This means quality filtering runs twice per trial.

ii.
```python
# In prepare_session_specs:
if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
    all_speeds.append(run_speed[frame_idx])

# In process_sessions:
keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
```

iii. The double quality check is intentional to ensure speed quartiles are computed only on valid trials.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `select_decoder_neurons` function performs d-prime computation, corridor-vs-gray responsiveness checks, and percentile-based selection on the full spike matrix. This complex neuron selection reduces the dataset significantly but is not required by the instructions or reference -- the reference simply keeps all neurons in the 4 visual areas.

ii.
```python
def select_decoder_neurons(spk, record, region_idx):
    # d-prime computation, corridor responsiveness, percentile selection...
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (...)
    for area in range(4):
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))
```

iii. This processing is computationally expensive and discards the vast majority of neurons that could be useful for decoding position and running speed.
