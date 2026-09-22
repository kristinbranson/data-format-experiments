# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven per-animal **joblib** files in `/app/data` (the extension-less files, not the parallel `.mat` files), one animal at a time, using `joblib.load`. Each file is a dict `{animal_id: fields}` from which only four fields are extracted — `trace` (day × cell × frame), `position` (day × 2 × frame), `blocked` (per-day blocked-partition indices) and `envs` (per-day geometry name). The remaining native fields (`SFPs`, `centroids`, `maps`) are dropped immediately and `gc.collect()` is called so that each animal is loaded exactly once and released before the next. The animal list is hard-coded in the order used by the reference `main.py`. Day counts are cross-checked across `trace`, `position`, `blocked` and `envs` before any conversion. All 7 animals / 207 animal-days / 69,744 day-cell instances / 8,187 one-minute trials are loaded, and the script asserts these exact totals at the end of a `--full` run.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
        bundle = joblib.load(DATA_DIR / animal)
        if list(bundle.keys()) != [animal]:
            raise ValueError(f"Unexpected top-level keys in {animal}: {list(bundle.keys())}")
        record = bundle[animal]
        trace = record["trace"]
        position = record["position"]
        blocked = record["blocked"]
        envs = np.asarray(record["envs"]).reshape(-1)
        del record, bundle
        gc.collect()
...
        if not (trace.shape[0] == position.shape[0] == len(blocked) == len(envs)):
            raise ValueError(f"{animal}: day counts disagree across source fields")
...
        if nsessions != 207 or ntrials != 8187 or nneurons != 69744:
            raise AssertionError("Full native-count check failed: ...")
```

iii. From CONVERSION_NOTES Step 2: "Conversion will use joblib, as does the reference code, because it contains the same already-converted arrays and loads substantially faster." The reference `utils.load_dat` defaults to `format="joblib"`, and `mat2joblib` shows the joblib files are a verbatim conversion of the `.mat` files. The hard-coded animal list follows the repository README / `main.py` ("The repository README identifies seven animals ... `main.py` analyzes all seven"). Unused fields are released because "holding derivative `SFPs` and `maps` while processing would waste many GiB".

## 1-b. How are the data split into subjects?

i. One subject per source file / animal ID. `subjects` is the fixed seven-element `ANIMALS` list, and every session appends the enclosing animal's index to `subject_idx` (stored as an `int16` array). Session ordering is animal-list order, then zero-based day, so `subject_idx` is monotonically non-decreasing and blocks of 31/31/31/21/31/31/31 sessions.

ii.
```python
    data = {..., "subjects": ANIMALS.copy(), "subject_idx": [], ...}
...
    for subject_index, animal in enumerate(ANIMALS):
        ...
            data["subject_idx"].append(subject_index)
...
    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int16)
```

iii. Each joblib file holds all recording days of one mouse, keyed by the animal ID, so the file/animal ID is the natural subject identifier (CONVERSION_NOTES Step 2/Step 5 "Animal ID → `subjects`, `subject_idx`; seven fixed IDs; one subject index for each day-session"). The paper's 7 mice (4 male, 3 female) are matched exactly.

## 1-c. How are the data split into sessions?

i. One target session = one animal-**day** (the first axis of `trace`/`position`/`blocked`/`envs`). 6 animals contribute 31 days and `QLAK-CA1-51` contributes 21, giving 207 sessions, matching the paper's "207 sessions". Each session has a single arena geometry and a fixed neuron population (cells registered on that day), and per-session provenance (`session_id`, subject, source day index, environment name, blocked indices, source/used/discarded frame counts, trial start frames, retained source-neuron indices) is stored in `metadata['session_info']`.

ii.
```python
        for day in range(trace.shape[0]):
            ...
            result = convert_session(
                trace[day], position[day], blocked[day],
                animal, subject_index, day, str(envs[day]),
            )
            neural_trials, input_trials, output_trials, valid_cells, session_info = result
            data["neural"].append(neural_trials)
            data["input"].append(input_trials)
            data["output"].append(output_trials)
            ...
            data["metadata"]["session_info"].append(session_info)
```
```python
    session_info = {
        "session_id": f"{animal}_day{day:02d}", "subject": animal,
        "subject_idx": subject_index, "source_day_index": day,
        "environment": environment, "blocked_partition_indices": blocked_indices,
        "source_n_cells": int(trace_day.shape[0]),
        "retained_n_neurons": int(valid_cells.size), ...}
```

iii. CONVERSION_NOTES Step 5, Key decision 1: "Each source animal-day is one target session. Neuron identity is stable across all trials within that day, while NaN registration varies across days." The methods state one 40-min session was recorded per day, so day = session.

## 1-d. How are the data split into trials?

i. The recordings are continuous (no native trial structure). Following the decoder-task instruction, each session is cut from source frame 0 into consecutive, non-overlapping 1,800-frame (60 s at 30 Hz) blocks; the final incomplete block is discarded without padding or overlap. Sessions are 71,866–72,219 frames, yielding 39 trials for animals 08/30/50 and 40 for the rest → 8,187 trials total (93 sessions × 39, 114 × 40). After 3-frame pooling each trial is 600 bins of 100 ms. The number of discarded tail frames is recorded per session.

ii.
```python
SOURCE_FPS = 30
TRIAL_SECONDS = 60
SOURCE_FRAMES_PER_TRIAL = SOURCE_FPS * TRIAL_SECONDS   # 1800
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES  # 600
...
    n_source_frames = trace_day.shape[1]
    n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
    if n_trials < 2:
        raise ValueError(f"{animal} day {day}: fewer than two complete trials")
    n_used_frames = n_trials * SOURCE_FRAMES_PER_TRIAL
...
    for trial in range(n_trials):
        start = trial * SOURCE_FRAMES_PER_TRIAL
        stop = start + SOURCE_FRAMES_PER_TRIAL
        _, labels = pool_position(position_day[:, start:stop])
```

iii. CONVERSION_NOTES Step 5, Key decision 2: "Starting at source frame 0, take consecutive 1,800-frame (60-s) blocks. Exclude only a final incomplete block; never pad or overlap." Step 4 notes the 39.93–40.12 min session-length variation is normal acquisition jitter around the documented 40-min sessions, so only the ragged tail is lost.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied: every complete 1,800-frame block of every session is kept. The only exclusions are (a) the final incomplete tail block and (b) a guard that raises an error if a session would yield fewer than two complete trials (never triggered; the decoder spec requires ≥2 trials/session). The AI explicitly decided **not** to port the reference position decoder's speed filter (smoothed speed > 5 cm/s), because removing scattered frames would break the fixed-length contiguous-minute trial definition.

ii.
```python
    if n_trials < 2:
        raise ValueError(f"{animal} day {day}: fewer than two complete trials")
...
    for s in range(nsessions):
        if len(data["neural"][s]) < 2:
            raise AssertionError(f"Session {s} has fewer than two trials")
```

iii. CONVERSION_NOTES Step 5, Key decision 5 ("No speed-frame deletion"): "Complete one-minute trials and fixed time bins are explicit downstream requirements. Removing scattered low-speed frames would turn elapsed time into irregular samples and violate the trial definition." Step 3/4 note no trial curation is described in the paper because sessions are continuous free exploration.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively the native `trace` field — the rising-phase-extracted, binarized calcium event raster (1 = significant event, NaN for a cell not registered on that day), indexed as `trace[day][cell, frame]` at 30 Hz. The derived `maps` (smoothed/unsmoothed rate maps), `SFPs` and `centroids` are explicitly *not* used.

ii.
```python
        trace = record["trace"]
...
            result = convert_session(trace[day], position[day], ...)
...
    selected = np.asarray(trace_day[valid_cells, :n_used_frames], dtype=np.float32)
```

iii. CONVERSION_NOTES Step 1: "Native `trace` is already a rise-extracted calcium-event series (1 = significant event); delta-F/F must **not** be recomputed." Step 5: `maps` are "Not copied — derived spatial summaries ... Avoid circularity: decode framewise position from synchronized trace, not a trace-derived rate map."

## 2-b. How is the `neural` data processed?

i. Per session: select the cells registered that day, truncate to complete trials, reshape to (cell, trial, 1800 frames), Gaussian-smooth along the within-trial frame axis with **sigma = 3 frames** (`mode="reflect"`, so smoothing never crosses a trial boundary), then **mean-pool non-overlapping groups of 3 frames**, producing `(n_neurons, 600)` float32 per trial at 10 Hz. No ΔF/F, no deconvolution, no z-scoring, no normalization across neurons. This reproduces exactly the temporal processing inside the reference `fit_decoder`/`test_decoder` (`gaussian_filter1d(traces, sigma=temporal_bin_size=3)` followed by `AvgPool1d(kernel_size=3, stride=3)`), the only deliberate difference being that the AI applies it per trial rather than across the whole concatenated session.

ii.
```python
NEURAL_SMOOTH_SIGMA_FRAMES = 3.0
POOL_FRAMES = 3
...
    selected = np.asarray(trace_day[valid_cells, :n_used_frames], dtype=np.float32)
    selected = selected.reshape(valid_cells.size, n_trials, SOURCE_FRAMES_PER_TRIAL)
    smoothed = gaussian_filter1d(selected, sigma=NEURAL_SMOOTH_SIGMA_FRAMES,
                                 axis=2, mode="reflect")
    pooled_neural = smoothed.reshape(
        valid_cells.size, n_trials, POOLED_BINS_PER_TRIAL, POOL_FRAMES
    ).mean(axis=3, dtype=np.float32)
...
        neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5, Key decision 3: "Match the paper decoder's `temporal_bin_size=3` and neural Gaussian sigma 3 frames. Process trials independently so random train/validation trial splitting cannot leak neural samples across a boundary." Step 10 Check 3 (Binning): "Reference `fit_decoder`/`test_decoder` Gaussian-smooth traces with sigma 3 frames and apply non-overlapping `AvgPool1d(3)`. Conversion matches this numerically using SciPy plus reshape/mean." The binary rising-phase vector is preserved as-is ("Exact agreement; no delta-F/F or deconvolution") because the paper treats it as the firing rate in all analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filter is removal of cells that are NaN on that day, i.e. cells not registered by CellReg for that session; validity is tested on the day's **first frame** (`np.isfinite(trace_day[:, 0])`), which is the same test used by the reference code (`p_vals[np.isnan(trace[0, :])] = np.nan` in `get_place_cells`). This retains 69,744 of the day-cell instances, matching the paper's 69,744 rate maps. The AI deliberately does **not** apply the reference decoder's ">5 events" activity threshold and does **not** restrict to place cells. A per-trial finiteness assertion guarantees no NaN survives into the output (and confirms the frame-0 test is equivalent to an all-frames test on this dataset).

ii.
```python
    # Native data represent an absent CellReg cell as NaN for its entire day.
    valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
    if valid_cells.size == 0:
        raise ValueError(f"{animal} day {day}: no registered cells")
...
        if not np.isfinite(neural_trial).all():
            raise ValueError(f"{animal} day {day}: retained neural data contain NaN/Inf")
```
```python
        "cell_filter": ("All manually curated cells registered (finite) on the source day; "
                        "no place-cell or activity threshold."),
```

iii. CONVERSION_NOTES Step 5, Key decision 4: "Retain every cell finite for the entire source day. These cells already passed motion/manual footprint QC. Do not place-cell-filter or activity-filter the stored dataset: the paper motivates all-cell inclusion, and the code's >5-event selection is an internal decoder feature-selection step tied to its deletion of low-speed frames." Step 4 also records that 69,632/69,744 instances would pass the >5-event criterion anyway, so the filter would be nearly inert while breaking the fixed neuron axis within a session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event to align to — free exploration of a continuous 40-min session. Each trial is aligned to its own segment onset: trial *i* spans source frames `[1800*i, 1800*(i+1))`, so `temporal_alignment_event = "start of each non-overlapping 1-minute session segment"`, `off_start = 0.0 s`, `off_end = 60.0 s`. Neural, input and output streams are cut with the identical frame indices, and each trial's source start frame is recorded in `session_info`. Position and trace share one 30-Hz frame axis in the released data (the DAQ timestamped both streams), so no offset or resampling is inferred.

ii.
```python
            "temporal_alignment_event": "start of each non-overlapping 1-minute session segment",
            "off_start": 0.0,
            "off_end": 60.0,
```
```python
        "trial_source_start_frames": [int(i * SOURCE_FRAMES_PER_TRIAL) for i in range(n_trials)],
```

iii. CONVERSION_NOTES Step 5, Key decision 11: "Trials align to their own segment start; `off_start=0.0`, `off_end=60.0`, `time_bin_size=100.0` ms, and per-trial start times are recorded in `session_info`." Step 4: "Simultaneous, timestamped 30-Hz streams ... Exact agreement; no resampling or inferred offset is needed."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the source 30 Hz (33.33 ms) stream is rebinned to **100 ms bins (10 Hz)** by Gaussian smoothing (sigma 3 frames) plus non-overlapping 3-frame average pooling, giving 600 bins per 60-s trial. `metadata['time_bin_size'] = 100.0` ms. The rebinning is identical for neural and position streams (same 3-frame windows), so the streams remain sample-for-sample aligned.

ii.
```python
POOL_FRAMES = 3
POOLED_BINS_PER_TRIAL = SOURCE_FRAMES_PER_TRIAL // POOL_FRAMES   # 600
...
            "time_bin_size": 100.0,
            "source_sampling_rate_hz": SOURCE_FPS,
            "temporal_pool_frames": POOL_FRAMES,
...
    print(f"Source: {SOURCE_FPS} Hz; trial: {TRIAL_SECONDS} s; "
          f"pool: {POOL_FRAMES} frames -> {1000*POOL_FRAMES/SOURCE_FPS:.1f} ms")
```

iii. CONVERSION_NOTES Step 5 mapping table: "per 1-min trial Gaussian-smooth at sigma 3 source frames and non-overlapping mean-pool 3 frames ... Matches the reference decoder's 3-frame smoothing/pooling while preventing smoothing leakage across held-out trial boundaries." Trajectory step 67: "100-ms bins match the reference decoder's 3-frame smoothing/pooling." The verifier confirms all trials are exactly T = 600 bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The native `blocked` field, which gives, per day, the indices of the occluded partitions of the 3×3 arena in the canonical order `[[0,1,2],[3,4,5],[6,7,8]]`, or `-1` when nothing is blocked. The `envs` geometry name is kept only as provenance metadata, and `get_env_mat` was used only as a cross-check.

ii.
```python
        blocked = record["blocked"]
...
            result = convert_session(trace[day], position[day], blocked[day], ...)
...
def blocked_vector(blocked_entry) -> tuple[np.ndarray, list[int]]:
    """Return a 9-vector (1=blocked) from the native nested blocked field."""
    values = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
```

iii. CONVERSION_NOTES Step 4: "Use `blocked` directly to create 9 binary 'is blocked' inputs; avoids plotting-orientation transforms." Step 5, Key decision 6: "Use nine blocked/not-blocked flags directly from `blocked`, rather than a 10-condition one-hot, because the requested input is compositional arena geometry and the source explicitly defines index order."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The per-day blocked indices are unwrapped from their nested MATLAB structure, validated (integer-valued, within 0–8, with `-1` meaning "none blocked"), and converted to a 9-element float32 binary vector (1 = blocked). The same static `(9,)` vector is attached to every trial of the session (`input_names = blocked_partition_0 … _8`). Across the dataset all partitions except index 7 take both values (index 7 is never blocked in the 10 geometries), which the AI verified against the data.

ii.
```python
def blocked_vector(blocked_entry):
    values = np.asarray(blocked_entry[0], dtype=float).reshape(-1)
    if values.size == 1 and values[0] == -1:
        indices: list[int] = []
    else:
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"Non-integer blocked indices: {values}")
        indices = [int(v) for v in values]
    if any(v < 0 or v > 8 for v in indices):
        raise ValueError(f"Blocked index outside 0..8: {indices}")
    vector = np.zeros(9, dtype=np.float32)
    vector[indices] = 1.0
    return vector, indices
...
        input_trials.append(geometry.copy())
```

iii. CONVERSION_NOTES Step 5 mapping: "Nine float32 binary flags in canonical source partition order; 1=blocked, 0=accessible; repeated as a static `(9,)` value for each trial ... Meets explicit static geometry-input task." Step 9 consistency table confirms "each source blocked index represented; index 7 never blocked" matches the converted per-partition ranges.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The native `position` field: DeepLabCut head-tracking x–y coordinates in cm, shape (day, 2, frame), on the same 30-Hz frame axis as `trace`. Row 0 is x, row 1 is y; the values span 0–75 cm and contain no NaNs (verified over all animals).

ii.
```python
        position = record["position"]
...
            result = convert_session(trace[day], position[day], ...)
...
    if not np.isfinite(position_day).all():
        raise ValueError(f"{animal} day {day}: position contains NaN/Inf")
```

iii. CONVERSION_NOTES Step 2: "`position` is finite throughout, ranges from 0 to 75 cm, and shares exactly the same day/frame axes as `trace`." Step 3: "Head position was obtained with DeepLabCut." The `maps['sampling']` occupancy product was rejected as a derived summary.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. For each trial the (2, 1800) position segment is **mean-pooled over the same non-overlapping 3-frame windows** used for the neural data, producing continuous (2, 600) coordinates, which are then discretized (see 4-c). No smoothing, interpolation, speed filtering or outlier rejection is applied to the position stream; the continuous pooled coordinates themselves are used only for plotting.

ii.
```python
def pool_position(position_trial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool (2, 1800) position and return coordinates plus 9-class labels."""
    pooled = position_trial.reshape(2, POOLED_BINS_PER_TRIAL, POOL_FRAMES).mean(axis=2)
    xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
    np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
    labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
    return pooled, labels
```

iii. CONVERSION_NOTES Step 5, Key decision 3: "Mean position coordinates before spatial discretization, keeping neural and output windows identical." Step 10: "Reference code mean-pools continuous position then encodes 15x15 bins. The requested task overrides only spatial resolution to 3x3." This mirrors `fit_decoder`'s `AvgPool1d` on `behav` followed by `.astype(int)`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each pooled coordinate is divided by the fixed 25 cm partition width and floored, then clipped into 0–2 (so the exact 75 cm boundary sample falls into the outer bin), giving a 3×3 grid over the fixed 75×75 cm arena. The two axis bins are flattened **row-major with y as the row and x as the column**: `label = y_bin*3 + x_bin`, producing one time-varying categorical output `position_3x3` of shape (1, 600), dtype uint8, values 0–8, with value names `y0_x0 … y2_x2`. The y-major order was chosen so that class indices coincide with the source `blocked` partition indices; the AI tested all eight transpose/flip mappings and found this one leaves only 152/4,912,200 (0.0031%) samples inside blocked partitions versus 13.6–21.7% for alternatives. Resulting global class fractions: [0.100, 0.098, 0.135, 0.075, 0.057, 0.077, 0.116, 0.141, 0.200].

ii.
```python
SPATIAL_BIN_CM = 25.0
N_SPATIAL_BINS = 3
...
    xy_bin = np.floor(pooled / SPATIAL_BIN_CM).astype(np.int16)
    np.clip(xy_bin, 0, N_SPATIAL_BINS - 1, out=xy_bin)
    # Native position is (x, y), whereas the source blocked-partition field is
    # row-major as [[0,1,2],[3,4,5],[6,7,8]]. Empirically and geometrically,
    # rows correspond to y and columns to x, so transpose into that canonical
    # geometry order before flattening.
    labels = (xy_bin[1] * N_SPATIAL_BINS + xy_bin[0]).astype(np.uint8)[None, :]
...
        if labels.min() < 0 or labels.max() > 8:
            raise ValueError(f"{animal} day {day}: output outside 0..8")
```
```python
        "output_names": ["position_3x3"],
        "output_values": [[f"y{y}_x{x}" for y in range(3) for x in range(3)]],
```

iii. CONVERSION_NOTES Step 5, Key decisions 7 and 8: "Use one categorical output with nine values, not two three-class outputs, because the task says 3x3 = 9 spatial bins ... exhaustive testing of all eight axis/flip transforms across 207 sessions identifies `(row,column)=(y,x)`"; "Clip exact 75-cm coordinates to bin 2. They are valid physical edge samples; dropping them would desynchronize streams." Step 10 documents that this axis order was a bug found and fixed in iteration 1 (17.14% → 0.0031% blocked-bin occupancy), and that the residual 152 samples are present in the raw pose stream itself.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Perfect index-for-index alignment: `position` and `trace` share the same day/frame axes in the source data (simultaneously acquired, timestamped 30-Hz DAQ streams), the same `[1800*i, 1800*(i+1))` slices define each trial for both, and the same non-overlapping 3-frame pooling windows are applied to both, so neural `(N, 600)` bin *k* and output `(1, 600)` bin *k* cover the identical 100 ms of source time. No lag, shift or interpolation is introduced. The `--show-processing` figure overlays raw and pooled position with the pooled time axis and the neural raster to demonstrate the 3:1 correspondence.

ii.
```python
    if trace_day.shape[1] != position_day.shape[1]:
        raise ValueError(f"{animal} day {day}: trace and position frame counts differ")
...
    for trial in range(n_trials):
        start = trial * SOURCE_FRAMES_PER_TRIAL
        stop = start + SOURCE_FRAMES_PER_TRIAL
        _, labels = pool_position(position_day[:, start:stop])
        neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
        ...
        neural_trials.append(neural_trial)
        output_trials.append(labels)
```

iii. CONVERSION_NOTES Step 10 Check 3 (Temporal alignment): "Reference functions index position and traces with the same frame mask. Conversion uses their identical 30-Hz frame axes and the same three-frame windows, with no inferred offset or resampling." Independent `np.allclose` spot checks at (session, trial, cell) = (0,5,3), (102,20,7) and (206,39,11) reproduced both neural and label values exactly from the raw joblib files.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) **Unregistered cells** (NaN for a whole day) are dropped per session, so the neuron axis is fixed within but not across sessions; a hard assertion guarantees no NaN/Inf reaches the output. (b) **Ragged session lengths** (71,866–72,219 frames) are handled by keeping only complete 1,800-frame trials and recording the discarded tail (1,666 / 219 / 91 / 60 / 71 frames depending on animal); nothing is padded. (c) **Boundary coordinates** exactly at 75 cm are clipped into the outer bin rather than dropped, keeping streams synchronized. (d) **Structural problems fail loudly**: mismatched day counts, mismatched trace/position frame counts, non-finite position, non-integer or out-of-range blocked indices, zero registered cells, or <2 complete trials all raise. (e) **Rare behavioural oddities** — 152 pooled samples (0.0031%) that fall inside a blocked partition — were investigated and deliberately left untouched because 150 of them are already in blocked bins in the raw pose stream and 2 arise from averaging three valid boundary samples.

ii.
```python
    if not np.isfinite(position_day).all():
        raise ValueError(f"{animal} day {day}: position contains NaN/Inf")
    valid_cells = np.flatnonzero(np.isfinite(trace_day[:, 0]))
    if valid_cells.size == 0:
        raise ValueError(f"{animal} day {day}: no registered cells")
    n_trials = n_source_frames // SOURCE_FRAMES_PER_TRIAL
    if n_trials < 2:
        raise ValueError(f"{animal} day {day}: fewer than two complete trials")
...
        "discarded_tail_frames": int(n_source_frames - n_used_frames),
```

iii. CONVERSION_NOTES Step 4: "Small acquisition-length variation is normal. Form complete 1,800-frame trials from frame zero and exclude only the final incomplete tail"; "For categorical output, clip `floor(position/25)` into 0–2 ... retaining it preserves aligned frames." Step 10: "They are source pose/boundary observations, not a conversion shift. Editing or deleting them would modify synchronized behavior without a supplied validity flag, so they are retained."

## 6-a. What are the most time-consuming steps of the code?

i. Decompressing/loading the seven joblib archives dominates: 8.9 + 14.8 + 15.8 + 6.5 + 14.6 + 12.0 + 16.0 ≈ **88.6 s of the 167.7 s** full run. The remaining cost is the per-session `gaussian_filter1d` + reshape-mean on (cells × trials × 1800) float32 arrays, ≈0.16–0.6 s per session (≈60–70 s over 207 sessions), plus 5.9 s to pickle the 6.17 GiB output. The AI instrumented and printed all of these (per-animal load time, per-session conversion time, write time, total).

ii.
```python
        load_start = time.perf_counter()
        bundle = joblib.load(DATA_DIR / animal)
        ...
        print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f}s: ...")
...
            session_start = time.perf_counter()
            result = convert_session(...)
            print(f"  {session_info['session_id']}: {valid_cells.size} neurons, "
                  f"{len(neural_trials)} trials in {time.perf_counter()-session_start:.2f}s")
...
    print(f"Write time: {time.perf_counter()-write_start:.2f}s")
    print(f"Total conversion time: {elapsed:.2f}s")
```

iii. CONVERSION_NOTES Step 7 run-time table attributes the cost to "Native joblib load (sample animal) 9.02 s / animal" and estimates "<60 s" for the write, concluding "Conservative full estimate ~4.8 min, well below 15 min"; the realized full run was 167.7 s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The expensive inner work is already vectorized: one batched `gaussian_filter1d` call per session over the whole (neuron, trial, frame) array and a single reshape-mean for pooling, rather than 8,187 per-trial SciPy calls. What remains is a small per-trial Python loop in `convert_session` that (i) calls `pool_position` on one trial's (2, 1800) slice, (ii) copies the neural slice with `np.ascontiguousarray`, and (iii) re-runs `np.isfinite(...).all()` and min/max checks per trial. The position pooling and label computation could have been done once per session on the full (2, n_used_frames) array (reshape to (2, n_trials, 600, 3)), exactly as was done for the neural stream, and the finiteness check could be a single session-level call. There is also a second per-trial validation loop at the end of `main` over all 8,187 trials. These loops are cheap relative to I/O, so the impact is small.

ii.
```python
    for trial in range(n_trials):
        start = trial * SOURCE_FRAMES_PER_TRIAL
        stop = start + SOURCE_FRAMES_PER_TRIAL
        _, labels = pool_position(position_day[:, start:stop])
        neural_trial = np.ascontiguousarray(pooled_neural[:, trial, :], dtype=np.float32)
        if not np.isfinite(neural_trial).all():
            ...
        if labels.min() < 0 or labels.max() > 8:
            ...
```
```python
    for s in range(nsessions):
        ...
        for neural, inp, out in zip(data["neural"][s], data["input"][s], data["output"][s]):
            if neural.shape != (n, POOLED_BINS_PER_TRIAL):
                raise AssertionError(...)
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Naive per-neuron/per-frame Python loops and per-trial SciPy calls would add substantial overhead"; "Code speedups added: One vectorized SciPy call filters `(neuron, trial, frame)` batches per session while preserving independent trial boundaries; reshape/mean performs pooling." The residual per-trial loop is not called out in the notes.

## 6-c. What processing does the code repeat multiple times?

i. Small, bounded repetitions: (1) `blocked_vector(blocked[day])` is recomputed in `main` for the plotting path even though `convert_session` already computed the identical vector; (2) in `--show-processing` mode `plot_processing` re-runs `gaussian_filter1d` on the raw traces and re-runs `pool_position` for trial 0, duplicating work already done (at most twice per run); (3) shape/finiteness validation is performed inside `convert_session` per trial and then again in the final validation loop over all sessions and trials; (4) `np.ascontiguousarray(..., dtype=np.float32)` copies data that is already contiguous-compatible float32. None of these are in the hot path. Each animal file is loaded exactly once, which is the repetition that would have mattered.

ii.
```python
            if args.show_processing and n_plotted < 2:
                geometry, _ = blocked_vector(blocked[day])   # recomputed
                plot_path = plot_processing(...)
```
```python
    raw = np.asarray(trace_day[valid_cells[:20], :SOURCE_FRAMES_PER_TRIAL], dtype=np.float32)
    smooth = gaussian_filter1d(raw, sigma=NEURAL_SMOOTH_SIGMA_FRAMES, axis=1)
    pooled_xy, labels = pool_position(position_day[:, :SOURCE_FRAMES_PER_TRIAL])
```

iii. CONVERSION_NOTES Step 6 states the optimization principle — "each animal is loaded only once", "avoids unnecessary file I/O" — but does not flag the duplicated `blocked_vector`/plot-path recomputation or the duplicated validation passes, presumably because they are negligible and the validation duplication is intentional defence-in-depth.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `joblib.load` deserializes the **entire** animal archive — including `SFPs`, `centroids` and the `maps` (smoothed/unsmoothed/sampling) products — which are then immediately deleted; these unused fields account for a large share of the ~88 s of load time (an h5py/lazy reader on the `.mat` files would read only `trace`, `position`, `blocked`). (2) `pool_position` always computes and returns the continuous pooled x–y coordinates, but `convert_session` discards them (`_, labels = ...`); only the plotting path uses them. (3) `session_info` stores `source_neuron_indices` for every session (69,744 integers in total) and other provenance the decoder never reads. (4) `geometry.copy()` is stored separately for all 8,187 trials instead of sharing one array. (5) The final whole-dataset validation loop re-walks every trial. All of these are minor next to the I/O cost, and the largest one (loading unused fields) is a consequence of the joblib format choice rather than of the conversion logic.

ii.
```python
        bundle = joblib.load(DATA_DIR / animal)      # also materializes SFPs, centroids, maps
        record = bundle[animal]
        trace = record["trace"]; position = record["position"]
        blocked = record["blocked"]; envs = np.asarray(record["envs"]).reshape(-1)
        del record, bundle
        gc.collect()
```
```python
        _, labels = pool_position(position_day[:, start:stop])   # pooled coords discarded
...
        "source_neuron_indices": valid_cells.astype(int).tolist(),
```

iii. CONVERSION_NOTES Step 6: "Release unused maps/SFPs after joblib load — Reduces resident memory before processing", i.e. the AI recognized these fields are unnecessary and minimized their *memory* footprint but still paid to deserialize them. Step 5 justifies keeping the provenance metadata: `session_info` "Retains provenance needed for independent spot checks", which the AI did in fact use for its Step 10 `np.allclose` verification.
