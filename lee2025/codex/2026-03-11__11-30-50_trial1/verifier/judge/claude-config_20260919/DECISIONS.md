# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset from the **joblib per-animal files** in `/app/data` (`QLAK-CA1-08`, `QLAK-CA1-30`, ..., i.e. the extension-less files), not the parallel `.mat` files. These are the files the reference repository's own `load_dat(..., format='joblib')` reads. Animals are discovered by a filename regex, and each animal file is loaded one at a time ("streamed") so only one animal's dict is in memory at once. Each animal dict contains `trace` `(n_days, n_cells, n_frames)`, `position` `(n_days, 2, n_frames)`, `envs` `(n_days, 1)`, `blocked` (list of length `n_days`), plus unused `maps`, `SFPs`, `centroids`. Sessions are the day index, trials are created by the conversion (no native trial structure).

ii.
```python
def get_animals(data_dir: str) -> list[str]:
    animals = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and re.fullmatch(r"QLAK-CA1-\d+", name):
            animals.append(name)
    return animals

def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]
```
```python
for animal in animals:
    animal_data = load_animal(data_dir, animal)
    ndays = animal_data["trace"].shape[0]
    print(f"Loaded {animal}: {ndays} sessions")
    for day_idx in range(ndays):
        ...
        neural_trials, input_trials, output_trials, brain_region_idx, debug_info = convert_session(...)
```
```python
    trace_day = animal_data["trace"][day]
    position_day = animal_data["position"][day]
    env_label = animal_data["envs"].reshape(-1)[day]
    open_mask = blocked_to_open_vector(animal_data["blocked"][day])
```

iii. From CONVERSION_NOTES.md Step 1: *"The joblib data files are the reference loading path used throughout `main.py`."* The AI identified `load_dat` / `save_dat` / `mat2joblib` in `code/georepca1/src/utils.py` and concluded the joblib copies are the canonical analysis input. It verified the loaded totals against the paper: 7 mice, 207 sessions, 5,413 registered cells, 69,744 valid cell-by-session registrations, mean cells/animal 773.29 ± 68.50 SE — all exactly matching the paper's reported "5,413 unique neurons across 207 sessions … 69,744 rate maps" and "773 ± 68 SE". Per-animal streaming was chosen explicitly to limit peak memory.

## 1-b. How are the data split into subjects (mice)?

i. One subject per data file. The subject ID is the file name (`QLAK-CA1-<nn>`), matched by `re.fullmatch(r"QLAK-CA1-\d+", name)` so that the `.mat` duplicates, `behav_dict`, and `precomputed_results/` are excluded. `subjects` is the sorted animal list; `subject_idx` records the animal index for every emitted session. Result: 7 subjects.

ii.
```python
if os.path.isfile(path) and re.fullmatch(r"QLAK-CA1-\d+", name):
    animals.append(name)
...
subjects = animals
subject_to_idx = {animal: idx for idx, animal in enumerate(subjects)}
...
converted["subject_idx"].append(subject_to_idx[animal])
...
converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)
```

iii. The dataset is organized as one file per animal, keyed internally by the animal ID (`joblib.load(path)[animal]`), which the AI confirmed in Step 2. The regex was needed because `data/` also contains `.mat` twins of every animal plus `behav_dict`; the AI documented these as duplicates of the same recordings.

## 1-c. How are the data split into sessions?

i. One session per recording day per animal: the leading axis of `trace`/`position`/`envs`/`blocked`. Sessions are emitted in sorted-animal order, then native day order. Six mice contribute 31 days, `QLAK-CA1-51` contributes 21, giving **207 sessions**. Each session gets a `session_id` of the form `QLAK-CA1-08_day07` and session-level metadata (geometry label, original/usable frame counts, trial count, valid neuron count).

ii.
```python
@dataclass(frozen=True)
class SessionRecord:
    animal: str
    day_idx: int
    @property
    def session_id(self) -> str:
        return f"{self.animal}_day{self.day_idx:02d}"
```
```python
ndays = animal_data["trace"].shape[0]
for day_idx in range(ndays):
    session = SessionRecord(animal=animal, day_idx=day_idx)
    ...
    converted["metadata"]["session_ids"].append(session.session_id)
    converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
```

iii. The paper states one 40-min session per day, with a sequence of 10 geometries repeated up to three times (31 days). The AI checked that the number of days per animal in the data (31, 31, 31, 21, 31, 31, 31 = 207) matches the paper's "207 sessions", and resolved the 21-day animal as a released-data fact consistent with "up to three total repetitions".

## 1-d. How are the data split into trials?

i. Each session is cut into **40 consecutive non-overlapping 1-minute trials** measured from session start (frame 0), i.e. trial *k* = frames `[k*1800, (k+1)*1800)`. Two edge rules are applied: (a) frames beyond the nominal 40-min session (72,000 frames) are discarded — this affects `QLAK-CA1-51` (72,219 frames, 219 frames dropped); (b) if a recording is slightly *shorter* than 40 min, the last trial is **truncated rather than dropped** — the three 71,866-frame animals get a final trial of 1,666 frames → 555 time bins instead of 600. A slice is kept only if it contains at least one full 3-frame bin. Result: 207 × 40 = **8,280 trials**, trial length 600 bins (min 555, max 600).

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
TRIAL_FRAMES = FPS * TRIAL_SECONDS            # 1800
NOMINAL_SESSION_SECONDS = 40 * 60
NOMINAL_SESSION_FRAMES = NOMINAL_SESSION_SECONDS * FPS   # 72000

def get_trial_slices(n_frames_session: int) -> list[tuple[int, int]]:
    usable_frames = min(n_frames_session, NOMINAL_SESSION_FRAMES)
    slices = []
    for trial_idx in range(NOMINAL_SESSION_SECONDS // TRIAL_SECONDS):   # 40
        start = trial_idx * TRIAL_FRAMES
        end = min(start + TRIAL_FRAMES, usable_frames)
        if end - start >= TEMPORAL_BIN_FRAMES:
            slices.append((start, end))
    return slices
```
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 3: *"Trialization: impose 40 one-minute trials per session using session start as time zero. Rationale: paper sessions are nominally 40 min. For recordings shorter than 72,000 frames, the 40th trial is shorter; for recordings longer than 72,000 frames, discard the small tail beyond 40 min. This preserves the experimental session duration while minimizing data loss."* The task instructions specify 1-minute trials; the source has no native trial structure. The AI verified the resulting trial lengths (600 bins everywhere, 555 for the final trial of the three short recordings) against raw frame counts with `np.allclose` spot-checks in Step 10.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality filtering is applied.** The only rejection rule is structural: a trial slice shorter than one 3-frame bin is not emitted (never triggered in practice — every session yields exactly 40 trials). Notably, the AI did *not* port the velocity filter (`v_thresh=5` on Gaussian-smoothed speed) that the reference repository's `decode_position_within` applies inside its decoder, nor the `cell_threshold=5` activity filter.

ii.
```python
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```

iii. CONVERSION_NOTES.md Step 3 "Trial curation rules" records that there is no native trial structure and that the reference's velocity/activity thresholds live *inside* the decoding function, not in the dataset. The AI's Step 10 reference-code comparison treats these as decoder-internal choices rather than data curation, so all timepoints are retained and left to the downstream decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively the `trace` field: the authors' pre-processed, **binarized rising-phase calcium event** matrix, indexed per day as `trace[day]` with shape `(n_registered_cells, n_frames)` and values in {0, 1} (NaN for cells not registered that day). No dF/F is computed, no raw fluorescence is used, and the precomputed `maps`, `SFPs`, `centroids` are not used for the neural stream.

ii.
```python
trace_day = animal_data["trace"][day]
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md Step 1: *"The `trace` field is already a rise-extracted event matrix where `1` marks significant calcium events. No code computes `dF/F`; for this conversion the reference neural signal is the provided event trace."* Step 3 reproduces the Methods pipeline (derivative → Gaussian smoothing sd=5 frames → noise estimate → z-score → threshold z>2.5 → binary vector) and notes *"The resulting binary event vector is treated as the firing rate in all later analyses."* Key Decision 1 states the released binarized trace is used directly because re-deriving a fluorescence signal would be inconsistent with the paper.

## 2-b. How is the `neural` data processed?

i. Processing is: (1) drop unregistered (NaN) cells for that day; (2) cast to `float32`; (3) slice into the 1-minute trials; (4) **average non-overlapping 3-frame windows** so each trial becomes `(n_valid_cells, n_bins)` at 100 ms resolution, with values in {0, 1/3, 2/3, 1}. Array orientation is already `(neurons, time)` in the source, so no transpose is needed. The reference repo's extra `gaussian_filter1d(traces, sigma=3)` smoothing (applied inside `fit_decoder` before pooling) is **not** applied. `brain_regions = ['CA1']` and `brain_region_idx` is a zero vector of length `n_valid_cells` per session.

ii.
```python
def temporal_bin_mean(arr: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (arr.shape[-1] // bin_size) * bin_size
    if usable <= 0:
        raise ValueError("Segment is too short to create at least one temporal bin.")
    trimmed = arr[..., :usable]
    new_shape = arr.shape[:-1] + (usable // bin_size, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```
```python
neural_raw = trace_valid[:, start:end]
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
neural_trials.append(neural_binned)
...
brain_region_idx = np.zeros(valid_cells.sum(), dtype=np.int64)
```

iii. Step 5 Variable Mapping: *"Keep only session-valid cells … Convert each chunk from 30 Hz binary events to 100 ms bins via non-overlapping 3-frame averaging … No dF/F computation. No place-cell filter."* The 3-frame averaging is justified as reproducing the reference decoder's `AvgPool1d(kernel_size=3, stride=3)` pooling in `fit_decoder`/`test_decoder`. A Step 10 sanity check reloaded raw joblib data for 8 sessions spanning all 7 animals and confirmed with `np.allclose` that converted neural values equal the mean of the corresponding raw 3-frame segments.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is removal of **cells not registered on that day**, detected as NaN in the trace. The mask is computed from the **first frame only** (`~np.isnan(trace_day[:, 0])`). No place-cell / reliability filter, no event-count threshold, no SNR filter. Mean 336.93 neurons/session (min 113, max 564), summing to 69,744 session-neuron entries.

ii.
```python
def get_valid_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])
...
valid_cells = get_valid_cell_mask(trace_day)
trace_valid = trace_day[valid_cells].astype(np.float32, copy=False)
```

iii. Step 5 Key Decision 2: *"Neuron inclusion: keep all valid registered cells for each session and remove only unregistered (NaN) cells on that day. Rationale: the paper explicitly says the observed reliability motivated inclusion of all cells in subsequent analyses; the within-session decoder in `main.py` also uses all cells."* Step 1 notes *"Registration quality is encoded by NaN traces/maps for cells absent on a given day."* The resulting per-session counts were cross-checked against the paper's 69,744 rate maps and per-animal cell counts `[515, 875, 942, 554, 862, 713, 952]`. (Independent check performed for this review: in `QLAK-CA1-08` day 0, NaN cells are NaN at *every* frame — 330 all-NaN columns, 330 first-frame-NaN columns — so the first-frame test is exactly equivalent to an all-frames test on this dataset.)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the recordings are continuous free-exploration sessions. The AI defines the alignment event as **the start of each consecutive 1-minute chunk within a session** and records it, with `off_start = 0.0` and `off_end = 60.0` s, in `metadata`. All trials for a session are cut from the same frame indices for neural, input, and output, so the three streams are aligned by construction.

ii.
```python
"temporal_alignment_event": "start of each consecutive 1-minute chunk within a session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
"nominal_session_duration_s": float(NOMINAL_SESSION_SECONDS),
"trial_duration_s": float(TRIAL_SECONDS),
```
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
```

iii. Step 3 notes *"Neural and behavioral streams were acquired simultaneously at 30 Hz and timestamped for post-hoc alignment"*, and Step 10 check 3(c): *"Temporal alignment: conversion keeps the released neural and position streams aligned frame-for-frame and applies a common 3-frame binning to both."* Since the task defines trials artificially, the AI documents the chunk boundary as the nominal alignment event rather than leaving the metadata fields empty.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — rebinned from the native 30 Hz (33.3 ms) to 100 ms.** Three consecutive frames are averaged (non-overlapping) for both the neural and position streams; `metadata['time_bin_size'] = 100.0` ms. A full trial is therefore 600 bins. Trailing frames that do not fill a whole 3-frame bin inside a trial are dropped by `temporal_bin_mean` (only relevant for the 1,666-frame short final trials: 1666 → 555 bins, 1 frame dropped).

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TIME_BIN_MS = 100.0
...
neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
...
"time_bin_size": TIME_BIN_MS,
```

iii. Step 5 Key Decision 4: *"Temporal bin size: store data at 100 ms resolution using non-overlapping 3-frame bins. Rationale: the reference decoder temporally bins data in 3-frame windows. Using 100 ms bins matches this scale, reduces computation substantially for the validator, and preserves alignment across neural and behavior streams."* This mirrors `fit_decoder(..., temporal_bin_size=3)` / `test_decoder(..., temporal_bin_size=3)` in `utils.py`, which the AI quoted in Step 1.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field, one entry per day: a list containing an array of blocked 3×3 partition indices, with the sentinel `[-1]` meaning "nothing blocked" (the plain square). The `envs` string label (`square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`) is deliberately kept only as metadata, **not** used to build the input via the repo's `get_env_mat` helper.

ii.
```python
open_mask = blocked_to_open_vector(animal_data["blocked"][day])
...
env_label = animal_data["envs"].reshape(-1)[day]      # metadata only
...
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
```

iii. Step 4 discrepancy row: the AI compared `blocked` against `get_env_mat(env)` for all 207 sessions and found they disagree for `t`, `l`, `bit donut` (orientation) and `glenn` (template differs), with no single flip/transpose reconciling all of them. Resolution: *"For the decoder input, use raw `blocked` as canonical geometry because it directly encodes blocked partitions per session. Keep `envs` as metadata / human-readable labels. This preserves actual session geometry and avoids helper-template simplifications."*

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked` is converted to a **length-9 open-mask** (1 = open, 0 = blocked): start from all-ones, set blocked indices to 0, unless the entry is the `[-1]` sentinel. The 9-vector is then **reshaped to 3×3, transposed, and re-flattened** so that its index convention (`x_bin * 3 + y_bin`) matches the output position-class convention — the raw `blocked` indices are stored y-major (`y * 3 + x`). The vector is static within a session and stored as a 1-D `(9,)` array per trial (the validator tiles 1-D inputs over time). `input_names = ['open_partition_0' ... 'open_partition_8']`.

ii.
```python
def blocked_to_open_vector(blocked_entry) -> np.ndarray:
    open_mask = np.ones(9, dtype=np.float32)
    blocked_values = np.array(blocked_entry[0], dtype=np.float32).reshape(-1)
    if not (blocked_values.size == 1 and blocked_values[0] == -1):
        open_mask[blocked_values.astype(int)] = 0.0
    # Raw blocked indices are stored in a y-major 3x3 layout, while output classes use x_bin * 3 + y_bin.
    return open_mask.reshape(3, 3).T.reshape(-1)
...
input_trials.append(open_mask.copy())
...
"input_names": [f"open_partition_{idx}" for idx in range(9)],
```

iii. Step 5 Key Decision 8: static geometry is stored as a compact 1-D length-9 vector because it is constant within a trial. The transpose was **not** in the first version: Step 10 found it as a real bug. The AI measured the fraction of output time bins that fall in a partition the geometry vector marks as blocked; it was **17.15 %** before the fix and **4.15e-05 (206 / 4,963,815)** after, which it attributed to negligible bin-boundary effects. The fix was re-validated with `np.allclose` against raw `blocked` entries for 8 sessions across all 7 animals. (Independent check performed for this review, animal `QLAK-CA1-08`: day 4 has `blocked = [0, 3, 6]` and raw position `min x = 25`, confirming raw index = `y*3 + x`, so the transpose is correct.)

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, per day of shape `(2, n_frames)` = (x, y) head-tracking coordinates sampled at 30 Hz in register with `trace`. Independent check for this review: values are already calibrated in cm, spanning 0 → 75 (arena is 75 × 75 cm), with no NaNs.

ii.
```python
position_day = animal_data["position"][day]
position_valid = position_day.astype(np.float32, copy=False)
...
position_raw = position_valid[:, start:end]
```

iii. Step 2 documents `position` as *"dense array of shape (n_days, 2, n_frames) with x-y position sampled at the same frame rate as trace"*; Step 3 notes *"Position is derived from DeepLabCut head tracking."* The reference repo's `decode_position_within` and `get_rate_maps` take exactly this array as the behavioral input.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is sliced into the same 1-minute trials as the neural data, **temporally averaged over the same non-overlapping 3-frame windows** (100 ms), then discretized to a single 9-class categorical variable per time bin. The result is stored as `(1, n_bins)` int64. `output_names = ['position_bin_3x3']`, `output_values = [['bin_0' ... 'bin_8']]`. One combined 9-class variable is used, not two separate x/y marginals.

ii.
```python
position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
output_trials.append(output_binned)
...
"output_names": ["position_bin_3x3"],
"output_values": [[f"bin_{idx}" for idx in range(9)]],
```

iii. Step 5 Key Decision 7: *"Output representation: use one categorical, time-varying output dimension (`position_bin_3x3`) rather than two separate x/y outputs. Rationale: the task explicitly requests 9 spatial bins. A single 9-class output matches the decoder objective directly and avoids reconstructing classes from separate marginals."* Averaging position before discretizing mirrors `fit_decoder`, which pools the scaled behavioral signal with `AvgPool1d` and *then* casts to integer bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is divided into 3 bins by **integer flooring against the session-wise per-axis maximum**: `bin = floor(coord / ((session_max_axis + 1e-5) / 3))`, clipped to `[0, 2]`, then combined as `class = x_bin * 3 + y_bin` (values 0–8). `session_max_xy` is computed once per session over the usable (≤40 min) frames with `np.nanmax`; a guard replaces a non-positive denominator with 1.0. Because the session max is used rather than a fixed 75 cm, bin edges shift slightly per session (e.g. `QLAK-CA1-08` day 4 has `max x = 72.94`, giving edges at 24.31/48.63 cm instead of 25/50 cm).

ii.
```python
POSITION_BINS = 3
BUFFER = 1e-5

def discretize_position_3x3(position_xy_by_time, session_max_xy) -> np.ndarray:
    if position_xy_by_time.shape[0] != 2:
        raise ValueError(f"Expected position shape (2, T), got {position_xy_by_time.shape}")
    denom = (session_max_xy + BUFFER) / POSITION_BINS
    denom = np.where(denom <= 0, 1.0, denom)
    binned = np.floor(position_xy_by_time / denom[:, np.newaxis]).astype(np.int64)
    binned = np.clip(binned, 0, POSITION_BINS - 1)
    classes = binned[0] * POSITION_BINS + binned[1]
    return classes[np.newaxis, :]
```
```python
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
```

iii. Step 5 Key Decision 6: *"Position discretization: use session-wise maximum-based spatial binning, matching the reference code's flooring rule, but with 3 bins instead of 15. Rationale: the paper's code bins position by dividing by `(session_max + buffer) / n_bins`. Reusing the same logic preserves coordinate handling while adapting to the requested 3 × 3 output grid."* This copies `get_rate_maps`: `position_binned = (position // ((np.nanmax(position, axis=0) + buffer) / n_bins)).astype(int)` (and the repo's `plots.py` likewise rescales position by its max). The discretization was validated indirectly by the blocked-partition occupancy check (4.15e-05 of bins land in a partition marked blocked) and directly by `np.allclose` spot-checks against raw data.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame. `trace` and `position` are released on a common 30 Hz clock with equal frame counts; both are sliced with the **same** `(start, end)` trial indices and passed through the **same** `temporal_bin_mean(·, 3)`, so every trial's neural and output arrays have identical time dimensions. The static geometry input is replicated once per trial.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)

    neural_trials.append(neural_binned)
    input_trials.append(open_mask.copy())
    output_trials.append(output_binned)
```

iii. Step 5 planned check: *"Alignment check: within a spot-checked trial, neural, input, and output arrays must have identical time dimensions after 100 ms binning."* Step 10 confirms the streams stay aligned frame-for-frame with a common binning, and the `--show-processing` plots overlay trial boundaries, raw trajectory, binned neural and binned position classes for two sessions. The full-data validator reported `T: mean 599.49, median 600, min 555, max 600` with no dimension errors or warnings.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled: (1) **unregistered cells** — NaN rows are dropped per session (2-c); (2) **`blocked = [-1]` sentinel** — treated as "no partitions blocked", producing an all-open vector rather than indexing with −1; (3) **sessions longer than the nominal 40 min** — truncated at 72,000 frames (219 frames dropped in `QLAK-CA1-51`); (4) **sessions shorter than 40 min** — the final trial is kept but shortened (1,666 frames → 555 bins) rather than discarded, and frames not filling a whole 3-frame bin are dropped. `np.nanmax` is used for the position maxima. There is no explicit NaN guard on `position` itself — a NaN sample would silently floor/cast to bin 0 — but the AI checked NaN counts in Step 4, and an independent check for this review confirms `position` contains no NaNs.

ii.
```python
if not (blocked_values.size == 1 and blocked_values[0] == -1):
    open_mask[blocked_values.astype(int)] = 0.0
```
```python
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
...
end = min(start + TRIAL_FRAMES, usable_frames)
if end - start >= TEMPORAL_BIN_FRAMES:
    slices.append((start, end))
```
```python
usable = (arr.shape[-1] // bin_size) * bin_size
if usable <= 0:
    raise ValueError("Segment is too short to create at least one temporal bin.")
```

iii. Step 2 recorded the five distinct session frame counts `{71866, 72060, 72071, 72091, 72219}` (39.93–40.12 min) and the `-1` sentinel in `blocked`. Step 10 Check 5 (edge cases) verified that *"final trial length is 555 bins for the three slightly short 39.93 min recordings and 600 bins elsewhere"*, with `np.allclose` comparisons against raw data for sessions of both kinds, spanning all 7 animals.

## 6-a. What are the most time-consuming steps of the code?

i. **Reading and decompressing the per-animal joblib files** dominates. In the full run (`conversion_full_out.txt`), per-animal wall time is 45–190 s while the actual per-session conversion is only 0.11–0.36 s (207 sessions ≈ 50 s total); total elapsed 467 s (~7.8 min). The load is expensive partly because `joblib.load` materializes the *entire* animal dict, including `maps` (15×15×n_cells×n_days, float64), `SFPs` (35×35×n_cells×n_days) and `centroids`, none of which the conversion uses. The second cost is writing the 6.3 GB output pickle.

ii.
```python
def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]
...
animal_start = time.perf_counter()
animal_data = load_animal(data_dir, animal)
...
print(f"Finished {animal} in {animal_elapsed:.2f}s")
...
with open(out_path, "wb") as f:
    pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6: *"Code inefficiencies identified: Full-data runtime will be dominated by decompressing the 7 large joblib animal files."* Step 7 estimated 3–4 min, conservatively <10 min including decompression — the actual 7.8 min was within that, and within the 15-minute budget, so no further optimization was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate is the **per-trial loop in `convert_session`**: `temporal_bin_mean` is called 40× per session (80× counting position) on slices of an array that could be binned **once** for the whole session and then sliced, since the trial boundaries (1800 frames) are exact multiples of the 3-frame bin except in the short final trial. Likewise `discretize_position_3x3` is invoked per trial instead of once per session. The animal/day loops are inherently serial I/O and could only be parallelized across processes, which the AI chose not to do. These are minor: the whole per-session conversion is ~0.2 s, so the achievable saving is a few tens of seconds against a 467 s run whose cost is I/O.

ii.
```python
for start, end in trial_slices:
    neural_raw = trace_valid[:, start:end]
    position_raw = position_valid[:, start:end]
    neural_binned = temporal_bin_mean(neural_raw, TEMPORAL_BIN_FRAMES).astype(np.float32, copy=False)
    position_binned = temporal_bin_mean(position_raw, TEMPORAL_BIN_FRAMES)
    output_binned = discretize_position_3x3(position_binned, session_max_xy).astype(np.int64, copy=False)
```

iii. Step 6 "Code speedups added": *"Per-animal streaming instead of loading all animals at once. Vectorized 3-frame temporal binning with reshape/mean. Static per-trial inputs stored as 1D arrays."* The AI vectorized the inner binning operation itself (reshape + mean, no Python loop over frames) and, having measured the conversion at ~0.2 s/session, judged further vectorization of the outer trial loop unnecessary.

## 6-c. What processing does the code repeat multiple times?

i. (a) `min(n_frames, NOMINAL_SESSION_FRAMES)` is computed twice per session — once as `usable_frames` and again inside `get_trial_slices`. (b) `temporal_bin_mean` is re-entered 80× per session (see 6-b) with its `usable` trimming re-derived each time. (c) `open_mask.copy()` is made once per trial (8,280 tiny copies) for a vector that is constant within a session. (d) `debug_info` (including `trial_slices` and `session_max_xy`) is assembled for every session even when plotting is off. (e) During development, the full conversion was run more than once — Step 10 found the geometry-orientation bug and rebuilt `converted_data.pkl` from scratch. (f) An earlier version did a redundant full pre-pass over all animal files; this was removed.

ii.
```python
usable_frames = min(trace_valid.shape[1], NOMINAL_SESSION_FRAMES)
session_max_xy = np.nanmax(position_valid[:, :usable_frames], axis=1)
trial_slices = get_trial_slices(trace_valid.shape[1])   # recomputes the same min() internally
```
```python
input_trials.append(open_mask.copy())
```
```python
debug_info = {
    "env_label": str(env_label),
    "open_mask": open_mask,
    "usable_frames": usable_frames,
    "original_frames": int(trace_valid.shape[1]),
    "trial_slices": trial_slices,
    "session_max_xy": session_max_xy,
}
```

iii. Step 10 "Issues Found and Resolved": *"Full-mode inefficiency: The first full conversion attempt did an unnecessary preload over all animal files before conversion. Resolution: removed the redundant `select_sessions()` full-data pass and reran the full conversion. Runtime returned to the expected few-minute range."* The remaining repetitions are cheap bookkeeping the AI did not flag.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (a) **Loading unused fields**: `joblib.load` materializes `maps['sampling'/'smoothed'/'unsmoothed']`, `SFPs` and `centroids` for every animal even though only `trace`, `position`, `envs`, `blocked` are used — this is the bulk of the dominant load cost, and a lazy reader (e.g. `h5py` on the `.mat` twins) would read only what is needed. (b) `debug_info` is built unconditionally but consumed only when `--show-processing` is set. (c) `env_label` and the six `session_*` metadata lists are stored but not used by the decoder. (d) `import math` is unused. (e) `first_trial_neural_binned` / `first_trial_output` are tracked on every session for plots that are produced at most twice. (f) The static 9-vector geometry input is duplicated 8,280 times instead of once per session. (g) Neural values take only four distinct values (0, 1/3, 2/3, 1) but are stored as `float32`, producing a 6.3 GB pickle; storing 3-frame *counts* as `int8` would be 4× smaller.

ii.
```python
import math          # never used
...
def load_animal(path: str, animal: str) -> dict:
    return joblib.load(os.path.join(path, animal))[animal]   # pulls maps/SFPs/centroids too
...
debug_info = {...}    # built every session, used only when do_plot
...
converted["metadata"]["session_ids"].append(session.session_id)
converted["metadata"]["session_geometry_labels"].append(debug_info["env_label"])
converted["metadata"]["session_original_frames"].append(debug_info["original_frames"])
converted["metadata"]["session_usable_frames"].append(debug_info["usable_frames"])
converted["metadata"]["session_trial_counts"].append(len(neural_trials))
converted["metadata"]["session_valid_neuron_counts"].append(int(brain_region_idx.shape[0]))
```

iii. The AI did not flag these; Step 6 only records the joblib decompression cost as the bottleneck, and Step 13 notes that the extra per-session metadata was deliberately retained *"for inspection"* / auditability. Since total runtime (467 s) was comfortably inside the 15-minute budget and the validator accepted the 6.3 GB file, the AI chose not to optimize further.
