# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all behavior data by globbing `data/beh/Beh_*.npy` files. Each file is loaded with `np.load(..., allow_pickle=True).item()` to get a dictionary of session records. It also loads experiment metadata from `data/beh/Imaging_Exp_info.npy` to map experiment types. Neural data is loaded per-session from `data/spk/<base>_neural_data.npy`, and retinotopy from `data/retinotopy/<mouse>_<date>_trans.npz`. The behavior-only pretraining cohort files in `data/beh/Unsupervised_pretraining_behavior/` are not loaded since they lack matching neural recordings.

ii.
```python
def load_experiment_type_map(root: Path) -> dict[str, set[str]]:
    exp_info = np.load(root / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
    ...

def select_representative_sessions(root: Path) -> list[SessionSpec]:
    beh_dir = root / "data" / "beh"
    ...
    for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
        exp_type = beh_path.stem.replace("Beh_", "")
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for key, record in beh_dict.items():
            ...

def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES.md Step 1-2: The AI identified that `load_spk` in the reference code loads neural data by concatenating stored `spks` arrays. The AI mirrored this approach. All behavior files are loaded to discover sessions, then deduplicated to 89 unique recording bases matching the paper's "89 recordings in 19 mice."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by parsing the recording base name (e.g., `TX108_2023_03_25_1`) to extract the mouse name (first component). Unique subjects are collected, sorted, and indexed.

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    blk = parts[4]
    return subject, date, blk

# In process_sessions:
subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
subject_idx.append(subject_to_idx[spec.subject])
```

iii. From CONVERSION_NOTES.md Step 5: "Session unit will be unique recording base `<mouse>_<date>_<blk>` (89 sessions), not duplicated metadata entries." The mouse name is the first underscore-separated component of the recording base identifier, consistent with the reference code convention.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique recording bases (`<mouse>_<date>_<blk>`). When multiple behavior keys map to the same base (e.g., `TX108_2023_04_07_1` vs `TX108_2023_04_07_1_swap2`), the "best" key is selected using a scoring function that prefers non-swap keys with more non-NaN stim_id values and more unique wall names.

ii.
```python
def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    stim_id = np.asarray(record.get("stim_id", []), dtype=float)
    non_nan_stim = int(np.isfinite(stim_id).sum()) if stim_id.size else 0
    unique_walls = int(len(np.unique(record.get("WallName", []))))
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)
```

iii. From CONVERSION_NOTES.md Step 5 Key Decisions: "Use 89 unique recording bases as sessions: This matches the paper's '89 recordings in 19 mice' and avoids leakage from duplicated experiment labels and swap1/swap2 aliases that share the same raw trials." And: "Prefer a plain behavior key when multiple keys share one recording base."

## 1-d. How are the data split into trials?

i. Trials are identified using the frame-level `ft_trInd` array which maps each imaging frame to its trial index. The code iterates over `range(ntrials)` (from the record's `ntrials` field), collects frame indices for each trial, and filters frames to those that are in the corridor (`ft_CorrSpc`) and during movement (`ft_move > 0`).

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

iii. From CONVERSION_NOTES.md Step 5: "Align trials using frame-level `ft_trInd` rather than `StartFr` / `EndFr`: This matches the reference code and avoids boundary ambiguities." The reference code uses `ft_trInd` for trial assignment consistently.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages: (1) Trials with fewer than 5 retained (running + corridor) frames are excluded. (2) Trials with retained duration > 60 seconds or inter-frame gap > 10 seconds are excluded. Sessions with fewer than 2 remaining valid trials are dropped entirely.

ii.
```python
MIN_TRIAL_TIMEPOINTS = 5
MAX_RETAINED_TRIAL_DURATION_S = 60.0
MAX_RETAINED_INTERFRAME_GAP_S = 10.0

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

# Sessions:
if len(session_neural) < 2:
    removed_sessions.append(...)
    continue
```

iii. From CONVERSION_NOTES.md Step 10: The AI found extreme `time_to_sound_cue_s` and `time_since_trial_start_s` ranges caused by "raw trials with sparse late running segments long after nominal trial start." These filters were added during Step 10 (Critical Review 1) to handle edge cases. The reference code/paper does not explicitly describe these duration/gap filters; these are the AI's own quality control additions. The AI noted 2,217 trials were removed by these filters (from 38,110 to 35,893).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` key in `<base>_neural_data.npy` files in the `data/spk/` directory. These are deconvolved calcium fluorescence traces from Suite2p processing.

ii.
```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES.md Step 1: "The core raw neural loader is `load_spk`; there is no delta-F/F computation in the reference code. The saved `spks` arrays are treated as the neural signal directly." From Step 3: "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The raw deconvolved traces are loaded and concatenated across imaging planes, then a selective neuron subset is chosen using stimulus-selectivity d-prime analysis. For each session, the code computes d-prime between two reference stimuli on running corridor frames, identifies corridor-responsive neurons, and keeps the top 5% positive and bottom 5% negative d-prime neurons per brain region. The selected neurons' raw deconvolved traces are kept without further normalization.

ii.
```python
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]

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
        if candidates.sum() < 20:
            selected[candidates] = True
            continue
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))
    ...
```

iii. From CONVERSION_NOTES.md Step 5: "Select neurons using the paper's 5% positive / 5% negative per-area logic... This follows the reference coding-direction analysis and reduces the export size enough to make full decoding practical." The AI argued this is consistent with `Get_coding_direction` and `Get_sort_spk` in the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Beyond the stimulus-selectivity neuron selection, there is no additional neuron quality filtering. The AI noted the reference code/paper do not describe global neuron-quality filters beyond Suite2p processing. The only frame-level filtering is the running + corridor mask (`ft_CorrSpc` and `ft_move > 0`).

ii.
```python
# In build_trial_frame_indices:
keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
```

iii. From CONVERSION_NOTES.md Step 3: "No additional global neuron-quality filter is described in the text beyond Suite2p processing." From Step 1: "There is no electrophysiology-style unit quality curation in the reference code."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry). Frames are assigned to trials via `ft_trInd`, and only running corridor frames are retained. The temporal alignment is at native imaging frame resolution — no resampling to a fixed time grid.

ii.
```python
# Per trial, the neural data is simply the spike matrix columns at the retained frame indices:
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. From metadata in the code: `"temporal_alignment_event": "corridor entry / trial start"` and `"off_start": 0.0`. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI's alignment uses frame indices assigned by `ft_trInd` to each trial, which correspond to corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate, with a median of ~314.7 ms across sessions. No temporal rebinning or resampling is applied.

ii.
```python
# In metadata:
"time_bin_size": float(np.median(frame_dt_medians)),
"frame_bin_source": "native imaging frame timestamps; no temporal resampling",

# Frame dt computed as:
ft = np.asarray(record["ft"][:spec.estimated_nfr], dtype=float)
dt_ms = np.diff(ft) * MS_PER_DAY
spec.median_frame_dt_ms = float(np.median(dt_ms))
```

iii. From CONVERSION_NOTES.md Step 5: "Use raw frame timestamps instead of resampling to a new clock: Median imaging frame spacing is very consistent across sessions (~314.7 ms), so keeping native frame bins is more faithful than interpolation to a synthetic time grid."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `record["SoundTime"]` (per-trial sound cue time) and `record["ft"]` (per-frame timestamps).

ii.
```python
def session_trial_info(record, trial, frame_idx):
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
    ...
    return time_to_cue.astype(np.float32), ...
```

iii. From CONVERSION_NOTES.md Step 5: "For each retained frame, compute `SoundTime - frame_time` in seconds; positive before cue, negative after cue."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the difference between the trial's `SoundTime` and the frame's timestamp is computed, then converted from days to seconds by multiplying by `SEC_PER_DAY` (86400). The result is positive before the cue and negative after.

ii.
```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. The AI documented this as a time-varying continuous input. The convention (positive = before cue) is a reasonable choice representing countdown to cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to sound cue is computed at the exact same retained frame indices as the neural data, so it is inherently aligned frame-by-frame.

ii.
```python
# Both use the same frame_idx:
neural_trial = spk[:, frame_idx]
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
```

iii. From CONVERSION_NOTES.md Step 5: "Time-varying continuous input" derived from the same retained frame timestamps used for neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session date parsed from the recording base name (e.g., `TX108_2023_03_25_1` gives date `2023-03-25`).

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    ...

# In select_representative_sessions:
first_date_by_subject: dict[str, datetime] = {}
for spec in specs:
    first_date_by_subject.setdefault(spec.subject, spec.date)
    spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. From CONVERSION_NOTES.md Step 5: "Continuous per-trial scalar = elapsed days since the mouse's first retained imaging session." The paper does not define this variable directly, but the decoder task requires it.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest session date is identified. The training day for each session is the number of elapsed days from that first session. This value is constant within a session and is broadcast across all timepoints.

ii.
```python
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. Justification: The decoder task requires "Day of training, continuous, per-trial." Since the paper doesn't provide explicit training day numbers, elapsed days from first session is a reasonable proxy.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `record["ft"]` (frame timestamps) and `record["Trial_start_time"]` (per-trial start times).

ii.
```python
def session_trial_info(record, trial, frame_idx):
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
    return ..., time_since_start.astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 5: "`frame_time - Trial_start_time` in seconds for each retained frame."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the difference between the frame's timestamp and the trial's `Trial_start_time` is computed and converted from days to seconds. The result is always non-negative.

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. This produces a monotonically increasing time-varying signal starting from ~0 at trial start.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at the same retained frame indices as neural data, ensuring frame-by-frame alignment.

ii.
```python
# Same frame_idx used for both neural and input:
neural_trial = spk[:, frame_idx]
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
```

iii. Inherently aligned since both use the same frame index array.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `record["isRew"]`, a per-trial array indicating whether the trial is in a rewarded corridor.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. From CONVERSION_NOTES.md Step 5: "Trial-level binary repeated across timepoints; 1 only for rewarded-corridor task trials, 0 otherwise."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value is cast to bool then float (0.0 or 1.0) and broadcast across all timepoints in the trial.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. From CONVERSION_NOTES.md: "In unsupervised / naive / grating sessions this is expected to be all 0."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `record["WallName"]`, a per-trial array of stimulus/wall texture names (e.g., "circle1", "leaf2").

ii.
```python
stim_idx = np.full(
    frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])],
    dtype=np.int16,
)
```

iii. From CONVERSION_NOTES.md Step 5: "Global categorical mapping over all unique stimulus names across retained sessions."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique wall/stimulus names across all sessions are collected and sorted to create a global category list (15 categories). Each trial's `WallName` is mapped to its index in this sorted list. The index is repeated across all timepoints in the trial.

ii.
```python
def collect_visual_categories(specs):
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories

visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}
```

iii. The 15 categories found: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `record["LickFr"]` (frame indices of lick events) and `record["LickTrind"]` (trial assignment of each lick).

ii.
```python
def build_lick_frame_lookup(record, nfr):
    lick_frames = np.asarray(record["LickFr"], dtype=float)
    lick_trial = np.asarray(record["LickTrind"], dtype=float)
    valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
    lick_frames = lick_frames[valid].astype(int)
    lick_trial = lick_trial[valid].astype(int)
    valid = (lick_frames >= 0) & (lick_frames < nfr)
    ...
    for trial in np.unique(lick_trial):
        lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
    return lookup
```

iii. From CONVERSION_NOTES.md Step 5: "Build a binary imaging-frame vector (1 if any lick occurs on that retained frame, else 0)."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A lookup dictionary mapping trial index to lick frame indices is built. For each retained trial, a binary vector is created where 1 indicates a retained frame that matches a lick frame index.

ii.
```python
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Binary output: 0 = no lick, 1 = lick at that imaging frame. This is time-varying and aligned to neural frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is determined by checking whether each retained frame index appears in the lick frame lookup for that trial. Since the same `frame_idx` array is used for neural data and licking, they are inherently aligned.

ii.
```python
# Same frame_idx for neural and licking:
neural_trial = spk[:, frame_idx]
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Frame-level alignment is automatic since both use the same retained frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `record["ft_Pos"]`, the per-frame position in the corridor in decimeters.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. From CONVERSION_NOTES.md Step 4: "Raw data / code are in decimeters: 60 = 6 m total, 40 = 4 m texture."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw position (in decimeters, 0-40 range for the 4 m corridor) is clipped to [0, 39.999] and divided by 10 to get bin indices (floor division), yielding bins 0-3.

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Each bin covers 1 m (10 decimeters): 0-1m, 1-2m, 2-3m, 3-4m. This matches the instruction "4 equal-length, 1-m-long spatial bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 decimeters (1 meter) creates 4 bins: bin 0 = [0,10) dm = 0-1m, bin 1 = [10,20) dm = 1-2m, bin 2 = [20,30) dm = 2-3m, bin 3 = [30,40) dm = 3-4m.

ii.
```python
pos = np.clip(pos_decimeters, 0.0, 39.999)
return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The output values are labeled `["0-1m", "1-2m", "2-3m", "3-4m"]`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read at the same retained frame indices as neural data, ensuring frame-by-frame alignment.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Same `frame_idx` is used for all neural, input, and output variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `record["ft_RunSpeed"]`, the per-frame running speed.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. `ft_RunSpeed` provides the instantaneous running speed at each imaging frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global quartile edges are computed from all retained running-corridor speed values across all sessions. Then per-trial speed values are digitized using `np.searchsorted` with those edges.

ii.
```python
# Global quartile computation in prepare_session_specs:
speed_values = np.concatenate(all_speeds).astype(np.float32, copy=False)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

def digitize_speed(speed, edges):
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. From CONVERSION_NOTES.md Step 5: "Discretize into global quartiles (25% each) across all retained timepoints in all retained sessions."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three quartile edge values are computed from the pooled retained speed data. Speed values are assigned to 4 bins using `np.searchsorted`: q1 (below 25th percentile), q2 (25th-50th), q3 (50th-75th), q4 (above 75th).

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
# Edges: [13.83, 26.68, 41.84]
```

iii. Output values labeled `["q1", "q2", "q3", "q4"]`. The instruction says "4 bins, each corresponding to 25% of the data," and the global distribution confirms [0.250, 0.250, 0.250, 0.250].

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read at the same retained frame indices as neural data.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Same `frame_idx` ensures alignment with neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Behavior arrays are truncated to neural frame count (`[:nfr]`) since behavior arrays are 1-3 frames longer.
- NaN values in `ft_trInd`, `LickFr`, and `LickTrind` are filtered with `np.isfinite`.
- Lick frames outside valid range are excluded.
- Trials with fewer than 5 retained frames, > 60s duration, or > 10s inter-frame gaps are excluded.
- Sessions with fewer than 2 valid trials are removed.
- If stimulus selectivity can't be computed (< 10 frames per stimulus), all mapped visual-area neurons are kept.

ii.
```python
# NaN filtering:
valid = np.isfinite(ft_tr)
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
valid = (lick_frames >= 0) & (lick_frames < nfr)

# Behavior truncation:
ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)

# Fallback neuron selection:
if stim1_mask.sum() < 10 or stim2_mask.sum() < 10:
    mapped = region_idx < 4
    return np.flatnonzero(mapped).astype(np.int32)
```

iii. From CONVERSION_NOTES.md Step 10: The AI documented handling of edge cases including extreme timing outliers and sparse stalled trials. From Step 4: "Behavior vs neural frame length: Code truncates behavior arrays with `[:nfr]` before combining with neural data."

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural data files is the dominant cost. Each session's neural file contains 20,000-90,000 neurons x ~29,000 frames. The full conversion took ~635 seconds (~10.4 s/session on average), with I/O being the bottleneck.

ii.
```python
# Per-session timing reported:
session_start = time.perf_counter()
spk = load_spike_matrix(root, spec.base)  # This is the bottleneck
elapsed = time.perf_counter() - session_start
print(f"    ... session time {elapsed:.1f}s")
```

iii. From CONVERSION_NOTES.md Step 6: "The current implementation still has to load each neural recording file in full before selecting the compact neuron subset, so full conversion will still be I/O-heavy."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial loop in `process_sessions` iterates over trials within each session. However, each trial has different frame indices and lengths, making full vectorization difficult. The `build_lick_frame_lookup` uses a Python loop over unique lick trial indices to build the lookup dictionary. The `select_decoder_neurons` loops over 4 brain regions.

ii.
```python
# Trial loop:
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...

# Lick lookup loop:
for trial in np.unique(lick_trial):
    lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)

# Region loop:
for area in range(4):
    candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
    ...
```

iii. The trial loop is inherently difficult to vectorize due to variable trial lengths. The lick lookup could potentially use `np.split` or `pandas.groupby` but the current approach is adequate.

## 12-c. What processing does the code repeat multiple times?

i. The `prepare_session_specs` function computes trial frame indices and applies quality filters to determine which trials to keep for speed quartile computation. Then in `process_sessions`, the same quality filters are applied again per trial. Frame indices are built once in `prepare_session_specs` but quality filters are checked twice.

ii.
```python
# In prepare_session_specs:
for trial_idx, (keep_trial, frame_idx) in enumerate(zip(spec.kept_trial_mask, spec.trial_frame_indices)):
    if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
        all_speeds.append(run_speed[frame_idx])

# In process_sessions (called again):
keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
```

iii. The AI documented this in Step 6: "Session preparation currently uses behavior-derived frame counts before the exact neural frame count is known; the second pass clips frame indices to the true neural frame count." The double-pass is partly necessary because `nfr` may differ slightly between passes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The neuron selectivity computation (`select_decoder_neurons`) computes d-prime and corridor responsiveness for all neurons, but only retains ~5-10% of them. The full spike matrix is loaded into memory before selecting the compact subset, which means most loaded neural data is discarded. Additionally, the `session_key_score` function computes scoring for all behavior keys even though only one per base is used.

ii.
```python
# Full spike matrix loaded, then subset selected:
spk = load_spike_matrix(root, spec.base)  # All neurons loaded
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]  # Most neurons discarded

# All behavior keys scored:
for key, record in beh_dict.items():
    ...
    if session_key_score(key, record) < session_key_score(current.key, current.record):
        ...
```

iii. From CONVERSION_NOTES.md Step 6: The AI acknowledged the I/O overhead of loading full neural recordings. The neuron selection is a deliberate design choice to match the paper's analysis, but loading all neurons when only ~3,000-6,000 are retained from 20,000-90,000 is wasteful.
