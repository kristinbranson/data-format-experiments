# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All scientific data are read exclusively through the AllenSDK `VisualBehaviorOphysProjectCache`, constructed with `from_s3_cache(cache_dir="/app/data")`. The experiment table (`get_ophys_experiment_table()`) is the master listing. Because the local cache holds only a subset of the released experiments, the AI enumerated which experiments are physically present by scanning `/app/data` for `*.nwb` **file names only** (never opening them) and intersecting the numeric tokens with the experiment-table index. The resulting 284 local experiments were then restricted to *active behavior* experiments (`behavior_type == "active_behavior"` and `passive == False`), giving 202 candidate experiments. No `project_code` filter was applied, so both `VisualBehavior` (168 exps) and `VisualBehaviorMultiscope` (34 exps) are included. Each experiment is then loaded with `cache.get_behavior_ophys_experiment(eid)`, which yields `dff_traces`, `ophys_timestamps`, `cell_specimen_table`, `trials`, `stimulus_presentations`, `running_speed` and `eye_tracking` from one synchronized session object. In full mode the 202 experiments are processed by a `ProcessPoolExecutor` with 4 workers (each worker builds its own cache).

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
table = cache.get_ophys_experiment_table()
ids = local_experiment_ids(table)
local = table.loc[ids]
active = local[(local["behavior_type"] == "active_behavior") & (~local["passive"].astype(bool))]
ids = list(map(int, active.index))
```
```python
def local_experiment_ids(table: pd.DataFrame) -> list[int]:
    """Identify cached experiment resources by filename; do not open NWB files."""
    valid = set(map(int, table.index))
    found = set()
    for path in DATA_DIR.rglob("*.nwb"):
        for token in re.findall(r"\d{8,}", path.name):
            value = int(token)
            if value in valid:
                found.add(value)
```
```python
def prepare_experiment(cache, eid, row, verbose=True):
    exp = cache.get_behavior_ophys_experiment(int(eid))
    ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    cell_ids, dff = stack_dff(exp)
```
```python
def prepare_experiment_worker(eid):
    """Process-safe wrapper; each worker accesses data through its own SDK cache."""
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
    table = cache.get_ophys_experiment_table()
    return prepare_experiment(cache, int(eid), table.loc[int(eid)], verbose=False)
```

iii. From CONVERSION_NOTES Step 1/Step 4: "`/app/code` is the AllenSDK repository. Its Visual Behavior Ophys documentation explicitly recommends loading and interacting with NWB-backed data through AllenSDK; conversion will not directly open NWB files." Passive experiments (82 of 284) were dropped because "Paper states passive was not analyzed; decoder requires task trials" and trial outcomes are undefined without licking. Multiscope data were kept because "Decoder says collect the Visual Behavior task; paper behavioral scope includes all active levels… A familiar-multiscope restriction would discard requested task data and most sessions without being required." All 202 active experiments loaded with zero SDK failures (293.5 s with 4 workers).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` strings taken from the experiment-table row of each retained experiment. They are sorted to give a deterministic ordering, and `subject_idx` maps every session (experiment) to its mouse. 38 mice result, matching the census of the active local subset.

ii.
```python
subjects = sorted({s["mouse_id"] for s in prepared})
subject_map = {x: i for i, x in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array([subject_map[s["mouse_id"]] for s in prepared], dtype=np.int16),
```
(`mouse_id` is captured per experiment in `prepare_experiment`: `"mouse_id": str(row.mouse_id)`.)

iii. Step 5 mapping table: "`mouse_id` → `subjects`, `subject_idx`; Sorted unique string IDs and integer lookup; 38 expected subjects in full active subset." `mouse_id` is the SDK's canonical animal identifier, and the count was cross-checked against the metadata census (`active_mice=38`) and confirmed in the validator output ("Number of subjects: 38").

## 1-c. How are the data split into sessions?

i. **One target "session" = one `ophys_experiment_id` (one imaging plane)**, not one `ophys_session_id`. Simultaneously acquired planes of a multiscope session are *not* merged; each plane becomes its own session entry with its own cells, its own `ophys_timestamps`, and (necessarily) a duplicate copy of that behavioral session's trials. Sessions are emitted in the order of the (sorted) local experiment-id list; 199 of the 202 candidates are retained. Per-session provenance (ophys_session_id, behavior_session_id, depth, cre line, experience level, session type, native rate, cell/trial counts, exclusions) is written to `metadata['session_info']`.

ii.
```python
result = {
    "eid": int(eid),
    "mouse_id": str(row.mouse_id),
    "region": str(row.targeted_structure),
    "cell_ids": cell_ids,
    "records": records,
    ...
    "info": {"ophys_experiment_id": int(eid), "ophys_session_id": int(row.ophys_session_id),
             "behavior_session_id": int(row.behavior_session_id), ...,
             "native_ophys_rate_hz": native_rate, "n_cells": int(len(cell_ids)), ...},
}
```
```python
for sess in prepared:          # one `sess` == one ophys experiment == one output session
    sn, si, so = [], [], []
    for rec in sess["records"]:
        ...
    neural.append(sn); inputs.append(si); outputs.append(so)
```

iii. Step 4 discrepancy table: "202 active experiments map to 174 ophys sessions; simultaneous planes have identical trials but plane-specific timestamps… Use each ophys experiment/imaging plane as one target session. This gives internally native cell matrices and avoids artificial cross-plane interpolation." Step 4 also reports a measured plane-to-plane timestamp offset of up to ~23 ms in a 7-plane example, and notes the reference paper "reports decoder summaries over planes". (Consequence visible in the validator output: runs of 7 identical trial counts, e.g. `207, 207, 207, 207, 207, 207, 207`, i.e. the same behavioral session appearing as 7 sessions.)

## 1-d. How are the data split into trials?

i. Trials come from the SDK `exp.trials` table. A trial is eligible if `(go OR catch) AND NOT aborted AND NOT auto_rewarded` (nullable booleans filled with `False`). The temporal extent is the experimental trial window `start_time → stop_time`, resampled onto 100 ms bin centers anchored at `start_time + 0.05 s`; trial length is therefore variable (validator: T = 70–125 bins, mean 84.1). 51,992 trials are eligible before stream QC; 48,112 are retained.

ii.
```python
trials = exp.trials.copy()
eligible = (
    (trials["go"].fillna(False) | trials["catch"].fillna(False))
    & ~trials["aborted"].fillna(False)
    & ~trials["auto_rewarded"].fillna(False)
)
trials = trials.loc[eligible]
```
```python
def trial_grid(start, stop):
    """100 ms bin centers anchored to the experimental trial start."""
    n = int(np.floor((float(stop) - float(start)) / BIN_SEC + 1e-9))
    if n < 1:
        return np.empty(0, dtype=np.float64)
    return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)
```

iii. Step 5 Key Decision 3: "Variable trial lengths are preserved: trials are segmented by experimental SDK `start_time`/`stop_time`, not forced into an arbitrary change-centered window. Metadata therefore uses `off_start=None`, `off_end=None`." Step 4: "Eligible mask is `(go OR catch) AND NOT aborted AND NOT auto_rewarded`", matching the instruction to include Go and Catch and exclude Aborted and Auto-rewarded. Keeping the full window lets image identity and image change vary within the trial.

## 1-e. How are trials filtered based on quality controls?

i. Beyond the go/catch eligibility mask, each trial must pass, with a counted exclusion reason for every rejection:
- exactly one of `hit/miss/false_alarm/correct_reject` true (`ambiguous_outcome`),
- a non-empty 100 ms grid (`empty_grid`),
- grid entirely inside the ophys timestamp support (`outside_ophys_support`),
- every bin has a finite interpolated running speed (`missing_running`),
- every bin has a finite pupil value, where blinks are invalidated and interpolation across gaps > 0.5 s is refused (`missing_pupil`),
- the whole experiment is dropped if the eye-tracking table is absent (`missing_eye_table`).
Sessions with fewer than 2 retained trials are dropped (3 experiments, 276 cells). Net: 48,112/51,992 = 92.5 % of eligible trials retained, and 199/202 sessions.

ii.
```python
for trial_id, tr in trials.iterrows():
    outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
    if sum(outcome_flags) != 1:
        exclusion["ambiguous_outcome"] += 1;  continue
    grid = trial_grid(tr.start_time, tr.stop_time)
    if grid.size == 0:
        exclusion["empty_grid"] += 1;  continue
    if grid[0] < ophys_t[0] or grid[-1] > ophys_t[-1]:
        exclusion["outside_ophys_support"] += 1;  continue
    run = interp_vector(running["timestamps"], running["speed"], grid)
    if not np.isfinite(run).all():
        exclusion["missing_running"] += 1;  continue
    pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
    if not np.isfinite(pup).all():
        exclusion["missing_pupil"] += 1;  continue
```
```python
if len(sess["records"]) >= 2:
    prepared.append(sess)
else:
    failed.append((eid, "fewer_than_two_retained_trials"))
```

iii. Step 5 Key Decisions 9–12: "mark likely blinks invalid; interpolate only between nearest valid samples when the gap is <=0.5 s. Exclude a trial if any requested 100 ms pupil sample remains invalid, because inventing a sixth 'missing' category violates the requested five bins… Record all exclusions." and "Running gaps: interpolate within source timestamp support only; reject trials with missing aligned running samples rather than extrapolating." Step 10 reconciles the counts exactly (3 excluded experiments listed by id; "No trials were silently lost").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.dff_traces['dff']` — the SDK's released, pre-computed ΔF/F traces — ordered by `cell_specimen_table.index`, together with `ophys_timestamps` as the time base. Detected calcium events were considered and deliberately not used.

ii.
```python
def stack_dff(experiment):
    cells = experiment.cell_specimen_table
    dff = experiment.dff_traces
    if not cells.index.equals(dff.index):
        dff = dff.loc[cells.index]
    traces = np.stack(dff["dff"].to_numpy()).astype(np.float32, copy=False)
    if traces.shape[0] != len(cells):
        raise ValueError("cell table and dF/F row count differ")
    if not np.isfinite(traces).all():
        raise ValueError("nonfinite values in released dF/F traces")
    return cells.index.to_numpy(), traces
```

iii. Step 1: "Neural signal choice: use the released `dff_traces`. The SDK reads a precomputed `DfOverF` stream and aligns it to valid ROI/cell IDs. Recomputing delta-F/F from fluorescence would diverge from released/reference processing." Step 3/4: the whitepaper documents the released ΔF/F pipeline (motion correction, neuropil correction, 600 s median baseline, 3.33 s detrend); the paper's own analyses used detected events, but "the decoder specification asks for neural activity generally… dF/F preserves dense framewise activity and is preferable for time-varying decoding" — logged as an intentional, decoder-driven difference.

## 2-b. How is the `neural` data processed?

i. No signal processing is applied beyond (a) stacking the per-cell ΔF/F arrays into an `(n_cells, T_native)` float32 matrix in `cell_specimen_table` order, (b) asserting finiteness and index alignment, and (c) resampling onto the 100 ms trial grid by vectorized linear interpolation from `ophys_timestamps`. No normalization, smoothing, z-scoring, deconvolution, or re-derivation of ΔF/F. Planes are never merged.

ii.
```python
def linear_sample_matrix(source_t, matrix, target_t):
    """Vectorized linear interpolation of rows in matrix onto target_t."""
    ...
    hi = np.searchsorted(source_t, target_t, side="left")
    hi = np.clip(hi, 1, source_t.size - 1)
    lo = hi - 1
    den = source_t[hi] - source_t[lo]
    alpha = ((target_t - source_t[lo]) / den).astype(np.float32)
    return matrix[:, lo] * (1.0 - alpha)[None, :] + matrix[:, hi] * alpha[None, :]
```
```python
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
```

iii. Step 5 Key Decision 11: "Cell curation: retain exactly SDK cells and dF/F rows, after asserting index equality and finite usable traces. No manual re-curation or dF/F recomputation." Step 6 lists "Vectorized neural interpolation for all neurons in a trial" as the main speed-up; float32 storage is used to control the 2.47 GiB pickle size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level QC. Every cell returned in `cell_specimen_table` / `dff_traces` is kept (29,168 cells over 199 retained sessions; per-session counts 4–666). The only neuron-level code is defensive: re-index ΔF/F to the cell table order if they disagree, error on row-count mismatch, error on non-finite traces. Planes with as few as 4 cells are kept.

ii. See `stack_dff` above (index/row-count/finiteness assertions only) and:
```python
if not np.isfinite(rec["neural"]).all():
    raise AssertionError("nonfinite converted neural data")
```

iii. Step 3 Curation rules: "Use only cells/ROIs returned in the SDK `cell_specimen_table` and aligned `dff_traces`; these have passed the released ROI filtering pipeline… Do not independently redo segmentation, ROI filtering, neuropil correction, dF/F, or session matching." Step 10: "Low-cell multiplane sessions (minimum four cells) are retained because cells are SDK-valid and the target imposes no minimum neuron count."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instruction is "temporally align based on ophys timestamp". Alignment is to **trial start** on the absolute ophys clock: the trial grid is `start_time + 0.05 + 0.1·k` seconds (absolute, not frame indices), and ΔF/F is linearly interpolated from `ophys_timestamps` onto exactly those instants. Every other stream (image identity, change, running, pupil) is evaluated on the same grid, so all streams are aligned by construction. Trials whose grid falls outside the ophys timestamp support are dropped rather than extrapolated. `metadata['temporal_alignment_event'] = "trial start; 100 ms bin centers represented in absolute ophys timestamp coordinates"`, `off_start = off_end = None`.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
if grid[0] < ophys_t[0] or grid[-1] > ophys_t[-1]:
    exclusion["outside_ophys_support"] += 1
    continue
...
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
image_names, changes = visible_images_and_changes(stim, grid)
```
```python
if target_t.size == 0 or target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
    raise ValueError("target grid outside ophys timestamp support")
```

iii. Step 5 Key Decision 4: "Ophys-based alignment: the common trial grid is defined in absolute seconds and all streams are sampled onto it; neural interpolation uses native `ophys_timestamps`, satisfying the explicit temporal-alignment requirement." Step 10's independent check (`critical_review.py`, nine trials from first/middle/last sessions, reloaded straight from the SDK) reproduced the converted ΔF/F to `np.allclose` with max |Δ| 4.6e-8–1.3e-7.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — everything is rebinned to a uniform 100 ms (10 Hz) grid** (`BIN_SEC = 0.100`, `metadata['time_bin_size'] = 100.0`). Native rates in the retained data span 10.726–30.950 Hz (single plane ~31 Hz, 7-plane multiscope ~10.7 Hz). Rebinning is done by *point-sampling with linear interpolation* at the bin centers, not by averaging the frames within each bin, so ~2/3 of the frames of a 31 Hz plane are not used for the value of any bin.

ii.
```python
BIN_SEC = 0.100
...
return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)
...
"time_bin_size": 100.0,
```

iii. Step 5 Key Decision 2: "Uniform temporal bin is 100 ms (10 Hz): required format says all sessions need the same bin size. This is just below the slowest observed native plane rate (~10.7 Hz), avoids information-free upsampling, and supports all time-varying outputs." Step 4 adds: "The target requires alignment based on ophys timestamp, and upsampling 10.7 Hz data to 30 Hz adds no information." Step 10: "Native rates range 10.726-30.950 Hz, but every converted bin is exactly 100 ms by construction."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `exp.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection`, using the columns `image_name`, `start_time`, `end_time` and `omitted`. It is **not** taken from the trials table's `initial_image_name` / `change_image_name`. Because grey inter-stimulus intervals and omitted flashes carry no image, they receive a dedicated category `none/gray` (code 0); the 16 real images of sets A+B occupy codes 1–16.

ii.
```python
stim = exp.stimulus_presentations
stim = stim[stim["stimulus_block_name"].str.contains("change_detection", na=False)].copy()
stim = stim.sort_values("start_time")
```
```python
def visible_images_and_changes(stim, grid):
    image = np.empty(grid.size, dtype=object); image[:] = None
    ...
    relevant = stim[(stim["start_time"] <= grid[-1]) & (stim["end_time"] > grid[0])]
    for row in relevant.itertuples():
        name = str(row.image_name)
        omitted = bool(row.omitted) if pd.notna(row.omitted) else False
        if not omitted and name not in {"omitted", "nan", "None"}:
            mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
            image[mask] = name
```

iii. Step 5 Key Decision 5: "Image category during gray/omission: use `none/gray` because decoder output asks for identity 'during the non-grey screen'. Omitted flashes present no image and belong to the same no-image category." Step 4 justifies the block filter: the SDK itself warns that the modern table contains several stimulus blocks and directs users to `stimulus_block_name.str.contains('change_detection')`. Validator result: grey 0.665 / images 0.335, matching the 500 ms grey + 250 ms image cycle (Step 9: "Excellent").

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per trial, an object array of per-bin image names (or `None`) is built, then mapped to integer codes with a single **global** dictionary built from the sorted union of all real image names across all retained sessions, prefixed by `none/gray` at index 0. Codes are stored as `int16` in row 0 of the `(5, T)` output array; `output_values[0]` carries the human-readable names and `metadata['image_encoding']` the full mapping.

ii.
```python
real_images = sorted({str(x) for s in prepared for r in s["records"] for x in r["image_names"] if x is not None})
image_values = ["none/gray"] + real_images
image_to_code = {name: i for i, name in enumerate(image_values)}
...
image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]], dtype=np.int16)
```

iii. Step 5: "Real global image labels are the union of image sets A/B and are encoded consistently across sessions" — a global, sorted mapping keeps class identity comparable across sessions/mice and is deterministic. Observed: 17 categories (`none/gray` + im000…im106), each real image ~2 % of bins.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the same 100 ms bin centers used for the neural data: a bin gets an image name iff its center lies in `[presentation.start_time, presentation.end_time)`. No separate resampling, no index shifting — the same `grid` array drives `linear_sample_matrix` for ΔF/F and the image labelling, so the two are aligned by construction.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
neural      = linear_sample_matrix(ophys_t, dff, grid)
image_names, changes = visible_images_and_changes(stim, grid)   # same `grid`
...
mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
image[mask] = name
```

iii. Step 10 comparison table: "Image output: Real identity only during SDK visible interval; `none/gray` otherwise… Exact task semantics; omissions correctly no-image." Step 10 Check 2 verified per-bin image codes against an independent SDK reload for nine trials (`np.allclose` pass). Decoding image identity at 6.33× chance is offered as further evidence that no temporal misalignment exists.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` (and `omitted`, `start_time`) columns of the same `change_detection` stimulus-presentation table. `trials.change_time` was checked against presentation onsets and found to agree exactly, but the presentation table is what the code uses. Catch (sham-change) trials have no `is_change` flash and therefore stay all-zero.

ii.
```python
is_change = bool(row.is_change) if pd.notna(row.is_change) else False
if is_change and not omitted:
    k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
    if 0 <= k < grid.size:
        change[k] = 1
```

iii. Step 4: "Representative go-trial change times equal nearest change-presentation onset exactly (100% within 20 ms; observed delta 0)… Use SDK trial boundaries for segmentation and presentation onsets for time-varying image/change labels." Using `is_change` also automatically excludes omitted-flash rows from being labelled a real change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A single `int8`/`int16` binary vector of length T per trial, zero everywhere except the one bin whose 100 ms interval contains the change-flash onset. The bin index is computed arithmetically from the grid origin (`grid[0] - BIN_SEC/2` is the left edge of bin 0). No smoothing, no widening to the flash or response window. Result: 1.03 % of all bins are labelled `change`.

ii.
```python
change = np.zeros(grid.size, dtype=np.int8)
...
k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
if 0 <= k < grid.size:
    change[k] = 1
...
out = np.vstack([image, rec["changes"].astype(np.int16), ...])
```

iii. Step 5 Key Decision 6: "Change is an onset event: exactly one bin immediately containing/after a real identity change is 1. This avoids labeling the entire changed-image presentation as an event." Step 12 revisits this after the decoder reached only 1.29× chance and declines to widen it: "Sparse framewise change representation is required by 'right after a change'; broadening it merely to increase accuracy would violate semantics."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Already binary — no thresholding of a continuous quantity is involved. Two declared categories, `["no_change", "change"]`, with values 0/1 only; the whole-dataset assertion loop verifies every code lies inside the declared value list.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
...
if trial[j].min() < 0 or trial[j].max() >= len(values):
    raise AssertionError(f"output {j} code outside declared values")
```

iii. The instruction defines the variable as binary ("Have value of 1 right after a change in image identity, otherwise 0"), so the AI keeps it binary and only documents the resulting imbalance (`image_change counts [4024483, 42086] fractions [0.9897, 0.0103]`).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same grid, same trial window, same function call as image identity — `visible_images_and_changes(stim, grid)` returns image and change together, so the change bin is indexed in exactly the frame coordinates used for ΔF/F. Changes occurring outside the trial grid (`k` out of range) are simply not marked.

ii.
```python
image_names, changes = visible_images_and_changes(stim, grid)
...
records.append({"grid": grid, "neural": neural, ..., "changes": changes, ...})
```

iii. Step 10: "Change output: One 100 ms event bin at SDK `is_change` onset… SDK `change_time` equals presentation onset; target says 'right after a change' — Exact; direct checks passed." Step 12's targeted raw check reloaded sessions 0/99/198 from the SDK and confirmed "each raw trial had one image-change event and each converted vector had the same one event at the same bin (`np.allclose=True`)."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `exp.running_speed`, using its `timestamps` and `speed` columns (the SDK's processed wheel-encoder speed, in cm/s, which can be negative). Sorted by timestamp before use.

ii.
```python
running = exp.running_speed.sort_values("timestamps")
...
run = interp_vector(running["timestamps"], running["speed"], grid)
```

iii. Step 1: "Running speed is an SDK-produced timestamped table" — the standard SDK interface, so no re-derivation from encoder voltage is needed. Step 5 notes "Negative SDK speeds are retained before binning."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the finite samples onto the trial's 100 ms grid, with **no extrapolation** (bins outside the source support stay NaN and cause the trial to be excluded). No smoothing, no rectification, no clipping of negative speeds. The interpolated values then go to global quintile discretization (5-c).

ii.
```python
def interp_vector(source_t, values, target_t, max_gap=None):
    good = np.isfinite(source_t) & np.isfinite(values)
    st, sv = source_t[good], values[good]
    out = np.full(target_t.shape, np.nan, dtype=np.float32)
    if st.size < 2:
        return out
    inside = (target_t >= st[0]) & (target_t <= st[-1])
    out[inside] = np.interp(target_t[inside], st, sv).astype(np.float32)
    ...
```
```python
run = interp_vector(running["timestamps"], running["speed"], grid)
if not np.isfinite(run).all():
    exclusion["missing_running"] += 1
    continue
```

iii. Step 3: "Running: paper event analyses linearly interpolated running to 30 Hz. For conversion, SDK running speed will likewise be interpolated to ophys timestamps before percentile discretization." Step 10: "Running output: SDK speed interpolated, global quintiles… Same alignment principle; common 10 Hz grid required by slowest imaging planes."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count percentile bins. Edges are the 0/20/40/60/80/100th percentiles of **all** finite, aligned running samples pooled over every retained trial of every retained session (a single global edge set, computed after all experiments are processed and before output assembly). Assignment uses `np.searchsorted` on the four interior edges with `side="right"`, so values equal to an interior edge fall in the higher bin and the maximum stays in bin 4. Edges are asserted strictly increasing. Observed edges `[-23.47, -0.0030, 0.322, 16.47, 33.81, 99.88]` cm/s and occupancy exactly 0.200 per class.

ii.
```python
def percentile_edges(values):
    x = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
    x = x[np.isfinite(x)]
    edges = np.percentile(x, [0, 20, 40, 60, 80, 100]).astype(np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"duplicate/non-increasing percentile edges: {edges}")
    return edges

def discretize(x, edges):
    # Interior edges divide values into categories 0..4; max remains category 4.
    return np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)
```
```python
run_edges = percentile_edges([r["running"] for s in prepared for r in s["records"]])
...
discretize(rec["running"], run_edges),
```

iii. Step 5 Key Decision 7: "Continuous-output bins are global quintiles: percentile edges are learned in a first pass over finite aligned samples from all included sessions, then fixed in a second conversion pass. Global rather than per-session bins preserve comparable class semantics. Duplicate percentile edges are checked and would be handled explicitly." Step 7 notes the deliberate consequence: per-session occupancy is not 20 % each, but pooled occupancy is exact.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly onto the same trial grid as the ΔF/F, i.e. onto absolute times on the ophys clock; row 2 of the `(5, T)` output array therefore shares bin-for-bin correspondence with the neural matrix. No nearest-frame snapping or index-based matching is used anywhere.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
run    = interp_vector(running["timestamps"], running["speed"], grid)
neural = linear_sample_matrix(ophys_t, dff, grid)
...
out = np.vstack([image, rec["changes"], discretize(rec["running"], run_edges), ...])
```

iii. Step 5 Key Decision 4 ("the common trial grid is defined in absolute seconds and all streams are sampled onto it"). Step 10 Check 2 independently re-interpolated running speed from a fresh SDK load and re-applied the saved percentile edges for nine trials: `np.allclose` passed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `exp.eye_tracking`, using `timestamps`, `pupil_width`, `pupil_height` and `likely_blink`. Diameter is defined as the geometric mean of the ellipse axes, `sqrt(width × height)` (equivalent to `2·sqrt(area/π)`). Frames flagged `likely_blink` (NaNs treated as blink) are set to NaN before any interpolation. If an experiment has no eye-tracking table at all, all its trials are excluded.

ii.
```python
eye = exp.eye_tracking
if eye is None:
    exclusion["missing_eye_table"] += len(trials)
else:
    eye = eye.sort_values("timestamps")
    pupil = np.sqrt(
        eye["pupil_width"].to_numpy(float) * eye["pupil_height"].to_numpy(float)
    )
    blink = eye["likely_blink"].fillna(True).to_numpy(bool)
    pupil[blink] = np.nan
```

iii. Step 5 Key Decision 8: "Pupil diameter: use geometric mean of SDK full ellipse width and height, an orientation-invariant equivalent-area diameter. SDK `pupil_area` is equivalent but widths/heights make the diameter definition explicit." Step 1: "blink-invalid pupil values are represented as missing by SDK and must not be treated as real diameters."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After blink invalidation, the remaining finite samples are linearly interpolated onto the 100 ms grid, but **only across gaps ≤ 0.5 s** in the *valid* source samples: for each target bin the bracketing valid source samples are found and the bin is re-set to NaN if they are more than 0.5 s apart. No extrapolation. Any trial with a residual NaN bin is dropped (the dominant exclusion reason, ~3.9 k trials, and the reason 3 whole experiments were dropped). Surviving values are then globally quintile-binned.

ii.
```python
MAX_PUPIL_GAP_SEC = 0.500
...
if max_gap is not None and inside.any():
    pos = np.searchsorted(st, target_t[inside], side="left")
    pos = np.clip(pos, 1, st.size - 1)
    gap_ok = (st[pos] - st[pos - 1]) <= max_gap
    ii = np.flatnonzero(inside)
    out[ii[~gap_ok]] = np.nan
```
```python
pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1
    continue
```

iii. Step 5 Key Decision 9: "interpolate only between nearest valid samples when the gap is <=0.5 s. Exclude a trial if any requested 100 ms pupil sample remains invalid, because inventing a sixth 'missing' category violates the requested five bins. Exclude an experiment only if fewer than two fully valid eligible trials remain. Record all exclusions." Step 10: the three dropped experiments (795953296, 806456687, 833631914) and the 276 associated cells are reconciled exactly.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: one global set of 0/20/40/60/80/100th-percentile edges over all finite aligned pupil samples in the retained dataset, `np.searchsorted(edges[1:-1], x, side="right")` → classes 0–4, monotonic edges asserted. Observed edges `[7.31, 35.35, 40.16, 44.49, 50.36, 140.0]` px and occupancy exactly 0.200 per class.

ii.
```python
pupil_edges = percentile_edges([r["pupil"] for s in prepared for r in s["records"]])
...
discretize(rec["pupil"], pupil_edges),
...
"output_values": [..., ["q1_smallest", "q2", "q3", "q4", "q5_largest"], ...],
"pupil_percentile_edges": pupil_edges.tolist(),
```

iii. Same rationale as running (Step 5 Key Decision 7): global edges keep class semantics comparable across sessions and mice, and satisfy the instruction's "five equal percentile bins" exactly at the pooled level. Edges are saved in metadata so the binning is invertible/auditable.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same trial grid, same absolute ophys-clock instants as the ΔF/F; row 3 of the `(5, T)` output array. Because a bin only survives if it has valid, non-extrapolated, short-gap pupil support, every retained pupil label corresponds to genuinely measured data at that neural time bin.

ii.
```python
pup    = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
neural = linear_sample_matrix(ophys_t, dff, grid)
...
out = np.vstack([image, rec["changes"], discretize(rec["running"], run_edges),
                 discretize(rec["pupil"], pupil_edges), np.full(T, rec["outcome"], dtype=np.int16)])
```

iii. Step 10: "Pupil output: SDK blink-masked ellipse diameter, short-gap interpolation, global quintiles… Consistent and conservative." Step 10 Check 2 recomputed the diameter and interpolation from a fresh SDK load for nine trials and matched the converted quintiles (`np.allclose`).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK trials table: `hit`, `miss`, `false_alarm`, `correct_reject` (constant `OUTCOMES` list, giving codes 0–3). Missing/NA flags are treated as False, and the trial is discarded unless exactly one flag is true.

ii.
```python
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
if sum(outcome_flags) != 1:
    exclusion["ambiguous_outcome"] += 1
    continue
...
"outcome": int(np.flatnonzero(outcome_flags)[0]),
```

iii. Step 4: "Trial categories… Eligible flags are mutually consistent: hit+miss are go, false alarm+correct reject are catch"; Step 3 lists the same four categories from the whitepaper and notes aborted/auto-rewarded are distinct flags (already excluded). Requiring exactly one true flag is the AI's explicit guard against ambiguous rows.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single integer code is broadcast across all T bins of the trial as row 4 of the `(5, T)` output array, i.e. stored as a constant time series rather than a scalar, while metadata documents it as static per trial. Bin-weighted distribution: hit 0.307, miss 0.567, false alarm 0.017, correct reject 0.108 (trial-weighted: 14,979 / 27,107 / 878 / 5,148).

ii.
```python
np.full(T, rec["outcome"], dtype=np.int16),
...
"output_names": ["image_identity", "image_change", "running_speed_bin", "pupil_diameter_bin", "trial_outcome"],
"output_values": [..., OUTCOMES],
```

iii. Step 5 Output Schema: "Time-varying outputs are represented together as a `(5,T)` integer array for validator compatibility; the static trial outcome is repeated across T while metadata marks it static. This is semantically equivalent to a per-trial scalar and avoids ragged mixed output containers." Step 12 verified outcome constancy over time and exact agreement with SDK flags on twelve independently reloaded trials.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered handling, all counted and reported rather than silently patched:
- **Nullable/absent trial flags**: `.fillna(False)` on `go`, `catch`, `aborted`, `auto_rewarded`; `pd.notna` guards on outcome, `is_change`, `omitted`.
- **Ambiguous outcomes**: trial dropped (`ambiguous_outcome`).
- **Degenerate/too-short trial windows**: `empty_grid` → dropped.
- **Trials extending past the imaging record**: `outside_ophys_support` → dropped (no clipping, no extrapolation).
- **Missing behavioral samples**: NaN after interpolation → trial dropped (`missing_running` / `missing_pupil`); blinks invalidated; pupil gaps > 0.5 s refused; missing eye table → whole experiment excluded.
- **Cell-table / trace disagreements**: ΔF/F re-indexed to the cell table; row-count mismatch or non-finite ΔF/F raises.
- **Session-level failures**: any exception during an experiment is caught, recorded in `failed`, and the run continues; sessions with < 2 retained trials are dropped.
- **Provenance**: every exclusion reason and count is stored per session in `metadata['session_info']` and `metadata['failed_or_excluded_sessions']`.

ii.
```python
eligible = ((trials["go"].fillna(False) | trials["catch"].fillna(False))
            & ~trials["aborted"].fillna(False) & ~trials["auto_rewarded"].fillna(False))
```
```python
try:
    sess = future.result()
    if len(sess["records"]) >= 2:
        by_id[eid] = sess
    else:
        failed.append((eid, "fewer_than_two_retained_trials"))
except Exception as exc:
    failed.append((eid, f"{type(exc).__name__}: {exc}"))
    print(f"FAILED experiment {eid}: {type(exc).__name__}: {exc}", flush=True)
```
```python
"trial_exclusions": dict(exclusion),
...
"failed_or_excluded_sessions": failed,
```

iii. Step 5 Key Decision 9 explains the refusal to invent values: assigning a quintile to a blink-corrupted bin "violates the requested five bins". Step 10: "51,992 initially eligible trials and 48,112 retained (92.54%). Exclusions are recorded per session in metadata and principally reflect blink/missing pupil gaps >0.5 s. No trials were silently lost." Non-finite ΔF/F is treated as a hard error because it would indicate a corrupted release product rather than an expected gap.

## 9-a. What are the most time-consuming steps of the code?

i. Documented and measured: SDK experiment construction / NWB-backed I/O inside `cache.get_behavior_ophys_experiment()` dominates (~1–2 s per experiment in the census; 5.45 s per session end-to-end in serial sample mode). Secondary costs: the per-trial neural interpolation over all cells, the per-trial Python loops over trials and stimulus presentations, returning ~2.5 GB of trial arrays from worker processes to the parent, and pickling the 2,466 MiB output. Mitigation: a 4-process pool. Full run: 293.5 s for 202 experiments, 299.4 s end-to-end — under the 15-minute budget.

ii.
```python
workers = min(4, os.cpu_count() or 1)
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(prepare_experiment_worker, eid): eid for eid in ids}
```
```python
st = time.time()
exp = cache.get_behavior_ophys_experiment(int(eid))
...
print(f"experiment {eid}: cells={len(cell_ids)} eligible={len(trials)} "
      f"retained={len(records)} rate={native_rate:.2f}Hz time={time.time()-st:.2f}s ...")
```

iii. Step 6: "AllenSDK experiment construction dominates runtime (about 1-2 seconds/session in the exploration census)." Step 7 gives the extrapolation (5.45 s/session serial → ~18.5 min; 4 workers → ~5–7 min) and Step 9 confirms the realized 293.5 s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops, in rough order of cost:
- `for trial_id, tr in trials.iterrows()` — per-trial `iterrows` (slow pandas row objects); the grid construction, support checks and interpolation calls could be batched across trials of a session.
- `for row in relevant.itertuples()` in `visible_images_and_changes`, plus the boolean re-filtering `stim[(stim.start_time <= grid[-1]) & (stim.end_time > grid[0])]` executed once per trial over the whole presentation table — the whole session's image/change labelling could be computed once on a session-wide grid with `np.searchsorted`.
- `image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]])` — a per-bin Python loop over ~4 M bins in total; could be a vectorized factorization.
- `outcome_flags = [bool(tr[x]) ... for x in OUTCOMES]` per trial; could be one vectorized `to_numpy()` over the four columns.
- The final assembly loop over sessions/trials and the class-count accumulation loop.
The AI did vectorize the expensive inner operation (interpolation over all neurons at once) and acknowledges the rest.

ii.
```python
for trial_id, tr in trials.iterrows():
    ...
    relevant = stim[(stim["start_time"] <= grid[-1]) & (stim["end_time"] > grid[0])]
    for row in relevant.itertuples():
        ...
```
```python
image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]], dtype=np.int16)
```
```python
# vectorized part:
return matrix[:, lo] * (1.0 - alpha)[None, :] + matrix[:, hi] * alpha[None, :]
```

iii. Step 6: "Python loops remain over sessions/trials/presentations because trial lengths and presentation intervals are ragged; heavy neural interpolation is vectorized over cells" and "Code speedups added: Vectorized neural interpolation for all neurons in a trial." The AI judged the remaining loops acceptable because I/O dominates, which the timing data support.

## 9-c. What processing does the code repeat multiple times?

i. Mostly avoided, but not entirely:
- **Cache and experiment-table construction per task**: `prepare_experiment_worker` builds a fresh `VisualBehaviorOphysProjectCache` and calls `get_ophys_experiment_table()` on *every* invocation, i.e. 202 times in a full run (the parent process already loaded the table once). Reusing a module-level per-process cache via a pool initializer would remove this.
- **Per-trial re-filtering of the stimulus table** (`relevant = stim[...]`) — a full-table boolean scan repeated ~48 k times.
- **Per-trial interpolation setup** over the full running / eye arrays (searchsorted on the full source arrays once per trial rather than once per session).
- `discretize` is recomputed in `make_plot` for values already available.
- Each experiment is nonetheless loaded from the SDK exactly once, and the percentile edges are computed from in-memory trial records so there is no second disk pass.

ii.
```python
def prepare_experiment_worker(eid):
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)   # per task
    table = cache.get_ophys_experiment_table()                                  # per task
    return prepare_experiment(cache, int(eid), table.loc[int(eid)], verbose=False)
```
```python
relevant = stim[(stim["start_time"] <= grid[-1]) & (stim["end_time"] > grid[0])]   # per trial
```

iii. Step 6: "Two-pass disk loading was avoided by retaining compact converted trial records in memory until global percentile edges are computed"; "One SDK load per experiment and one in-memory percentile/construction pass"; "No redundant raw-file access, dF/F recomputation, or cross-plane resampling." The per-worker cache rebuild is the one repetition the notes do not mention.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts of work whose products never reach the pickle or the decoder:
- `bool_col()` is defined and never called (dead code).
- Per-trial `grid`, `trial_id`, `start_time`, `stop_time` are computed and carried in every record but only used by `--show-processing` plots; they are not saved.
- `native_rate` is computed per experiment and only recorded in metadata (the converted bin size is fixed at 100 ms).
- `cell_ids` arrays are retained in full although only their length is used (`brain_region_idx` is a constant fill per session).
- `np.isfinite(traces).all()` over the entire native ΔF/F matrix, plus a second `np.isfinite(rec["neural"]).all()` per trial, and the whole-dataset `bincount` class-count pass — validation only.
- `inp = np.empty((0, T))` is allocated per trial for a task with no inputs.
- Image names are materialized as object arrays and only afterwards mapped to codes.
- Verbose per-session `session_info` (including exclusion dictionaries) is stored in metadata; useful for provenance but unused by the decoder.
None of these is expensive relative to SDK loading. The one substantive cost is storing dense 100 ms ΔF/F for 48,112 trials (2.47 GiB), which *is* consumed downstream.

ii.
```python
def bool_col(df, name):          # never called
    return df[name].fillna(False).astype(bool).to_numpy()
```
```python
records.append({"trial_id": int(trial_id), "grid": grid, ...,
                "start_time": float(tr.start_time), "stop_time": float(tr.stop_time)})
```
```python
inp = np.empty((0, T), dtype=np.float32)
...
if not np.isfinite(rec["neural"]).all():
    raise AssertionError("nonfinite converted neural data")
```

iii. Step 6 frames the retained per-trial records as the deliberate alternative to a second disk pass; Step 5/Step 10 justify the assertion and metadata overhead as the audit trail demanded by the instructions ("Include sanity checks", "Validate data shapes and types at each step"). The notes do not flag `bool_col` or the unused per-trial timing fields.
