# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object API. It found that the environment's `pynwb`/`hdmf` stack cannot instantiate these NWB 2.6.0 files through `BehaviorOphysExperiment.from_nwb_path` (an `external_resources` abstract-method mismatch), so it reads the released NWB/HDF5 files directly with `h5py`, using the SDK source in `/app/code` only as the schema/semantics guide.

The index of what to load is the local metadata CSV `project_metadata/ophys_experiment_table.csv`, intersected with the NWB files actually present on disk (284 files). From those it keeps rows with `passive == False` (202 experiments) and then drops 3 experiments whose NWB has no `acquisition/EyeTracking` group, leaving **199 "sessions"**. Notably it does **not** filter on `project_code`, so the converted set mixes 168 `VisualBehavior` (single-plane, ~31 Hz) experiments with 34 `VisualBehaviorMultiscope` (multi-plane, ~11 Hz) experiments.

Per NWB file it reads: `intervals/trials` (go/catch/aborted/auto_rewarded/hit/miss/false_alarm/correct_reject/change_time/start_time/stop_time), the active change-detection stimulus-presentations interval table, `processing/ophys/event_detection` (data + timestamps), `processing/running/speed`, and `acquisition/EyeTracking/pupil_tracking`.

Conversion is two-pass: pass 1 (`collect_global_statistics`, `load_events=False`) reads every session to build the global image vocabulary and the global running/pupil quintile edges; pass 2 (`convert_sessions`) re-reads every session with the event traces and builds the output arrays.

ii.
```python
DATA_ROOT = Path("/app/data")
RELEASE_ROOT = DATA_ROOT / "visual-behavior-ophys-1.1.0"
EXPERIMENT_DIR = RELEASE_ROOT / "behavior_ophys_experiments"
METADATA_DIR = RELEASE_ROOT / "project_metadata"

def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    exp_table = exp_table.sort_values("ophys_experiment_id")
    sessions = [SessionInfo(...) for row in exp_table.itertuples(index=False)]
    filtered_sessions = [s for s in sessions if has_required_eye_tracking(s.path)]
    ...
    return sessions
```

```python
def read_session_raw(session, load_events=True):
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
        trials = read_interval_table(trial_group, ["go","catch","aborted","auto_rewarded",
            "hit","miss","false_alarm","correct_reject","change_time","start_time","stop_time"])
        keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
        stim_group = choose_task_presentation_group(h5f)
        ...
        ophys_timestamps = np.asarray(h5f["processing"]["ophys"]["event_detection"]["timestamps"], ...)
        if load_events:
            events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
        pupil_width  = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], ...)
```

iii. From CONVERSION_NOTES Step 5, Key Decision 9: *"Load NWB content with `h5py` rather than the SDK session object … the current environment's `pynwb/hdmf` stack cannot instantiate these NWB 2.6.0 files … Direct HDF5 reads will therefore mirror the SDK field definitions explicitly."* Step 4 also records that the local disk is *"a curated subset of the release"* and that statistics must be compared to the local subset, not the full public release. Step 10's reference-code comparison asserts "field semantics match the SDK objects." The absence of a `project_code` filter is never justified explicitly; the notes only observe that the local NWBs are a mixture and that multi-plane sessions can hold up to 8 planes.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` values from `ophys_experiment_table.csv`, stored as strings. A subject is registered lazily the first time a session belonging to it is converted, so `subjects` is ordered by first appearance (which, because the experiment table is sorted by `ophys_experiment_id`, is essentially chronological). Result: 38 mice.

ii.
```python
mouse_id=str(int(row.mouse_id)),
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
...
data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
```

iii. Step 5 variable mapping: *"`NWB general/subject/subject_id` / metadata `mouse_id` → `subjects`, `subject_idx`; convert to global subject list and per-session index … use mouse identifier strings."* The `mouse_id` field is the canonical animal identifier in the SDK metadata tables.

## 1-c. How are the data split into sessions?

i. **One NWB `ophys_experiment` file = one decoder "session".** The AI explicitly chose *not* to group imaging planes by `ophys_session_id`. For the 168 single-plane `VisualBehavior` experiments this is a 1:1 mapping with the recording session. For the 34 `VisualBehaviorMultiscope` planes it splits a single simultaneous recording into up to 7 separate "sessions", each of which carries the *identical* trials table and therefore identical behavioural/stimulus outputs (visible in `verification_full_out.txt` as runs of repeated trial counts: `209, 209, 209, 209, 209, 209, 209`, `287 ×7`, `309 ×7`, …). Sessions are ordered by `ophys_experiment_id`, not by acquisition date.

ii.
```python
exp_table = exp_table.sort_values("ophys_experiment_id")
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    if int(raw["keep_mask"].sum()) < 2:
        continue
    ...
    region_idx = brain_region_to_idx.setdefault(session.targeted_structure, len(brain_region_to_idx))
    ...
    data["neural"].append(session_neural)
    data["brain_region_idx"].append(np.full(raw["n_neurons"], region_idx, dtype=np.int64))
```

iii. CONVERSION_NOTES Step 4 (Discrepancies, "Session definition"): *"Treat each NWB experiment file as one decoder session because neural traces are experiment-specific; preserve subject/session metadata so multiple experiments from one behavior session remain linked through subject/session fields."* The `ophys_session_id` linkage is in fact **not** preserved in the output — only `session_ophys_experiment_ids` is written to metadata.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. The trial window is the full `start_time` → `stop_time` interval (variable length; 211–377 bins at 30 Hz, mean 254.7). Total: 51,075 trials over 199 sessions (mean 256.7/session).

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)

def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. Step 4 (Trial taxonomy): *"Trial inclusion for decoder should follow the common interpretation across sources: include `go` and `catch`, exclude `aborted` and `auto_rewarded`"*, referencing `Trial._get_trial_data` in the SDK where aborted trials force `go = catch = auto_rewarded = False`. This directly implements the instruction "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." Using the full `start_time`→`stop_time` window keeps both the pre-change flash sequence and the post-change response window inside the trial, which is what makes time-varying image identity/image change meaningful.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, at three levels:
- **Session level (experiment type)**: all `passive == True` experiments are dropped (82 local files), because passive sessions have the lick spout retracted and therefore produce degenerate trial outcomes (all go trials = miss, all catch trials = correct_reject).
- **Session level (data availability)**: 3 active sessions with no `acquisition/EyeTracking` group are dropped entirely (`795953296`, `806456687`, `833631914`).
- **Session level (size)**: sessions with `< 2` valid trials are skipped, checked twice (before and after conversion).
- **Trial level**: the `(go|catch) & ~aborted & ~auto_rewarded` mask above. No trial is dropped for missing `change_time`, short duration, or being clipped at the recording edge (verified: no valid trial falls outside the ophys timestamp range in this dataset).

ii.
```python
exp_table = exp_table[~exp_table["passive"]].copy()

def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session ... {int(raw['keep_mask'].sum())} valid trials")
    continue
...
if len(session_neural) < 2:
    print(f"[pass2] skipping session ... {len(session_neural)} trials")
    continue
```

iii. Step 5 Key Decision 1: *"Use only active sessions: Local data include 82 passive experiment files, but the decoder task requires trial outcome and the strategy paper's behavioral analyses focus on active sessions. Passive sessions also produce degenerate outcomes (sample passive file: all go trials are misses and all catch trials are correct rejects). Excluding passive sessions keeps the mapping aligned to the operant task."* Key Decision 8: *"Require at least two valid trials per session after filtering"* (the format spec requires ≥2 trials per session to evaluate the decoder). The eye-tracking exclusion was added in Step 10 after the first full run crashed: *"Missing pupil data in 3 active sessions caused full-conversion failure … exclude those sessions before both passes so global bin edges and converted sessions are computed on the same valid subset."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the Allen pipeline's **precomputed detected calcium events** (the SDK's `BehaviorOphysExperiment.events`), shape `(n_timepoints, n_rois)`, with `processing/ophys/event_detection/timestamps` as the ophys timebase. dF/F (`processing/ophys/dff`) is present in the same NWB files but is deliberately **not** used.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
if load_events:
    events = np.asarray(
        h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
    )
```

iii. Step 4 (Neural signal choice): *"SDK exposes both `dff_traces` and `events`… Whitepaper defines dF/F processing; strategy paper states analyses used 'detected calcium events'. → Use precomputed `events` as the neural signal for conversion because that best matches the analysis paper and avoids diverging from reference processing."* Step 5 Key Decision 2 repeats this. The metadata field records `"neural_signal": "precomputed calcium events"`.

## 2-b. How is the `neural` data processed?

i. The only processing is **linear interpolation of the event traces from the native ophys timestamps onto the trial's 30 Hz grid**, then a transpose to `(n_neurons, n_timepoints)` and a cast to `float32`. No dF/F recomputation, no smoothing, no normalisation, no z-scoring, no baseline subtraction, no merging of imaging planes (planes are separate sessions — see 1-c). Interpolation is vectorised across all neurons at once via gather-and-blend index arrays.

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    """Linear interpolation for 2D source arrays of shape (T, N)."""
    n_src = source_t.shape[0]
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, n_src - 1)
    left = np.clip(right - 1, 0, n_src - 1)
    same = right == left
    t0 = source_t[left]; t1 = source_t[right]
    denom = np.where(np.abs(t1 - t0) < 1e-12, 1.0, t1 - t0)
    w = np.where(same, 0.0, (query_t - t0) / denom)
    out = source_values[left] * (1.0 - w[:, None]) + source_values[right] * w[:, None]
    return out.astype(np.float32)
...
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. Step 5 mapping table: *"Use precomputed event traces; transpose to `(n_neurons, n_timepoints)` per trial after resampling/interpolation to common 30 Hz trial grid."* Step 6: *"Trial-level interpolation is vectorized over neurons within each trial."* The rationale for interpolating rather than indexing frames is Key Decision 3 (see 2-e). No further processing is applied because, per Step 4, the released `events` are already the end product of the Allen QC/segmentation/demixing/event-detection pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering of the AI's own. It defensively applies the SDK's `exclude_invalid_rois` rule: if `processing/ophys/image_segmentation/cell_specimen_table` has a `valid_roi` column whose width matches the event trace, and not all ROIs are valid, the invalid columns are dropped. In the released NWB files `valid_roi` is all-True (I checked 8 files: `nvalid == ncells` in every case), so this is a no-op in practice — all released cells are kept. Result: 29,168 neurons, 4–666 per session, mean 146.6. No activity-based filter, no removal of the 2,467 trials (4.8%) whose event matrix is all zeros.

ii.
```python
if load_events:
    cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cell_table = len(cell_table["id"])
    if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
        valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
        if valid_roi.sum() != events.shape[1]:
            events = events[:, valid_roi]
    n_neurons = events.shape[1]
```

iii. Step 4 (ROI / neuron filtering): *"Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc neuron filtering beyond reference QC/filtering already reflected in the files"*, citing `CellSpecimens.__init__(..., exclude_invalid_rois=True)` and the whitepaper's description of union/duplicate/dim-ROI exclusion already performed upstream. On the all-zero trials, Step 10 Check 1: *"the warnings reflect genuine sparsity of the released calcium-event signal, not conversion corruption. They are therefore not fixable without changing the referenced neural representation."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. The per-trial time grid begins exactly at the trials-table `start_time` and steps in 1/30 s increments until `stop_time`; every stream (neural, running, pupil, image identity, image change, outcome) is evaluated on that same `grid` array, so all streams are aligned by construction. Metadata records `temporal_alignment_event: "trial start"`, `off_start: 0.0`, `off_end: None` (trials are variable-length).

ii.
```python
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)

neural_trial  = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont  = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont    = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
image_codes   = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
image_change  = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Step 10 (Temporal alignment comparison): *"all trial streams are aligned in absolute experiment time and resampled to a common 30 Hz grid from raw trial `start_time` / `stop_time`."* Because the Allen pipeline hardware-syncs all clocks onto one timebase, absolute times are directly comparable across streams. Step 12's alignment check re-examined the `--show-processing` plots and found *"no visible offsets."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **33.333 ms (30 Hz), uniform for all trials and sessions.** Yes — every stream, including neural, is **resampled** off its native clock onto this synthetic grid by linear interpolation. The native ophys rates in the converted set are ~32.3 ms (31 Hz, single-plane `VisualBehavior`) and ~93.2 ms (11 Hz, `VisualBehaviorMultiscope`); the AI logged this range explicitly (`[setup] native ophys dt range: 0.0323 … 0.0932`). So for 83% of sessions 30 Hz is a mild (31→30 Hz) resample that blends adjacent frames, and for the 34 multiscope planes it is a **3× upsample**, in which roughly two of every three neural samples are interpolated rather than measured. No temporal smoothing or bin-averaging is applied; this is pure point interpolation, not rebinning in the spike-count sense.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0

def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
"resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
```

iii. Step 5 Key Decision 3: *"Use a common 30 Hz time base for all sessions: The target format requires one shared bin size across sessions, while local data mix ~31 Hz single-plane and ~11 Hz multi-plane ophys sampling. The strategy paper linearly interpolates calcium event responses onto common 30 Hz timestamps for neural and running analyses, and behavior/eye tracking are naturally 30 Hz, so 30 Hz is the most defensible common grid."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **active change-detection stimulus-presentations interval table** (in these files `intervals/Natural_Images_Lum_Matched_set_TRAINING_2017_presentations`), columns `start_time`, `stop_time`, `image_name`, `omitted`. The correct table is selected by scanning all `*_presentations` groups and picking the one that has an `active` column and an `image_name` column, scoring first on whether `stimulus_block_name` mentions `change_detection` and then on the number of active presentations. This is flash-level ground truth rather than the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
def choose_task_presentation_group(h5f):
    for name in intervals.keys():
        if not name.endswith("_presentations") or name == "trials": continue
        grp = intervals[name]
        if "active" not in grp or "image_name" not in grp: continue
        n_active = int(np.asarray(h5_array(grp["active"])).astype(bool).sum())
        if n_active == 0: continue
        has_change_detection = ... "change_detection" in stimulus_block_name ...
        score = (has_change_detection, n_active)
        if score > best_score: best_score, best_name = score, name
    return intervals[best_name]
```

iii. Step 5 mapping: *"stimulus-presentation `image_name`, `start_time`, `stop_time`, `omitted`, active task block only"*, citing `BehaviorSession.stimulus_presentations` / `get_stimulus_presentations`. Step 1 notes *"`stimulus_presentations` contains more than just the active task block in newer SDK releases; for VBO change-detection analyses, the active block is identified via `stimulus_block_name` containing `change_detection`."*

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A piecewise-constant per-bin categorical signal. For each 30 Hz bin the preceding presentation is found with `searchsorted`; if the bin falls inside `[start_time, stop_time)` of a non-omitted presentation, the bin gets that image's integer code, **otherwise it gets the code for a dedicated `"gray"` class**. The 500 ms inter-stimulus grey gaps and the 5% omitted flashes therefore both map to `gray`. The vocabulary is global across all sessions: `image_values = ["gray"] + sorted(image_names)` → 17 classes (`gray` + im000, im031, im035, im045, im054, im061, im062, im063, im065, im066, im069, im073, im075, im077, im085, im106).

The consequence in the converted data is that **`gray` occupies 66.9% of all time bins** and each of the 16 real images gets only 1.9–2.2% (250 ms flash / 750 ms cycle = 1/3 non-grey).

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    idx = np.searchsorted(stimulus["start_time"], query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = (idx >= 0) & (idx < len(starts))
    in_interval = query_t[valid] < stops[idx[valid]]
    if np.any(in_interval):
        sub_idx = idx[valid][in_interval]
        target = np.full(sub_idx.shape, image_to_code["gray"], dtype=np.int64)
        for i, (name, is_omitted) in enumerate(zip(names, omit)):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        codes[np.flatnonzero(valid)[in_interval]] = target
    return codes
...
image_values = ["gray"] + sorted(image_names)
image_to_code = {name: idx for idx, name in enumerate(image_values)}
```

iii. Step 5 Key Decision 6: *"Use `gray` as an explicit image-identity class: Because the task includes 500 ms gray periods and omissions extend gray instead of showing an image, a `gray` category is needed for a complete time-varying identity signal."* Key Decision 7 covers the global vocabulary: *"Discretize continuous outputs globally, not per session … so class definitions are shared across sessions."*

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on the *same* `grid` array as the neural trial, so it is aligned bin-for-bin by construction. Bin boundaries follow the actual flash `start_time`/`stop_time` in absolute session time, so the on/off transitions land on the correct bin to within one 33 ms bin.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes  = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Same rationale as 2-d: one grid per trial shared by every stream guarantees alignment. Step 7's plot review states the identity trace *"show[s] the expected alternation of flashed images and gray intervals"* with no visible offset.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` and `omitted` columns of the same active stimulus-presentations table (plus its `start_time`/`stop_time`). It is **not** derived from the trials table's `change_time`/`go`. Because the SDK marks catch-trial flashes as `is_sham_change` (not `is_change`), catch trials automatically get an all-zero image-change trace.

This is a revision: the first implementation used a one-bin impulse at the trials-table `change_time` for `go` trials only, and was replaced during Step 7/8.

ii.
```python
stim = read_interval_table(stim_group, ["start_time","stop_time","image_name",
    "is_change","omitted","trials_id","active","flashes_since_change"])
...
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Step 5 mapping (revised): *"stimulus-presentation `is_change`, `start_time`, `stop_time` → binary time-varying label: 1 during the changed-image presentation window, else 0 … Marks the post-change flashed image itself rather than a one-bin impulse; catch trials remain 0."* Step 10 Issues: *"The initial `image_change` target was too sparse because it used a single-bin impulse at `change_time`. Resolution: relabeled `image_change` as the changed-image presentation window using raw stimulus `is_change` intervals; sample validation accuracy moved from below chance to above chance."*

## 4-b. What processing is involved in computing `output` *Image change*?

i. Same interval lookup as image identity: for each 30 Hz bin, find the containing presentation; the bin is 1 iff that presentation has `is_change == True` and `omitted == False`. Since a flash lasts 250 ms, this marks roughly **7–8 consecutive bins** per go trial (the flash only, not the following 500 ms grey). Empirically 2.59% of all bins are 1 (`{no_change: 0.974, change: 0.026}`).

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    idx = np.searchsorted(stimulus["start_time"], query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    valid = (idx >= 0) & (idx < len(starts))
    in_interval = query_t[valid] < stops[idx[valid]]
    if np.any(in_interval):
        sub_idx = idx[valid][in_interval]
        changed = is_change[sub_idx] & (~omitted[sub_idx])
        codes[np.flatnonzero(valid)[in_interval]] = changed.astype(np.int64)
    return codes
```

iii. Step 6: *"`image_change` is constructed from the stimulus table's `is_change` presentation interval instead of a single-bin impulse at `change_time`, which yields a less degenerate decoder target while staying aligned to the task structure."* Step 12 confirmed the class is *"sparse but not degenerate"* at 2.59% and spot-checked 3 positive trials against raw data (7–8 positive bins each, `np.allclose == True`).

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is intrinsically binary (`is_change` is a boolean column), so it is emitted directly as `{0, 1}` with `output_values = ["no_change", "change"]`. The only implicit "threshold" is the choice of window width: the 250 ms change-flash interval rather than a single bin or flash+grey.

ii.
```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
    [f"bin_{i}" for i in range(5)],
    [f"bin_{i}" for i in range(5)],
    outcome_values,
],
```

iii. Implicit in the instruction ("Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0"). The AI's window choice is justified in Step 6 as giving a *"less degenerate decoder target"* than a one-bin impulse.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — same `grid`, same absolute-time interval lookup, so it is bin-for-bin aligned with the neural trial. The rising edge lands on the first 30 Hz bin at/after the change flash's `start_time`.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Same as 2-d/3-c. Step 12's temporal-alignment check: *"the saved processing plots from Step 7 show image identity, image change, running, pupil, and neural traces aligned on the same 30 Hz grid without visible offsets."*

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — the SDK's default **filtered** running speed (`BehaviorSession.running_speed`), i.e. the encoder signal after transient removal, artifact rejection and a 10 Hz low-pass Butterworth filter. The unfiltered alternative `processing/running/speed_unfiltered` present in the same files is not used.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. Step 5 mapping: *"running speed timeseries → `output[running_speed_bin]` … Uses filtered running speed, matching SDK default"*, citing `RunningSpeed.from_stimulus_file`. Step 1 records that the SDK builds this stream with monitor delay forced to 0.0, so it sits on the sync/stimulus clock without display-lag compensation.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the native running timestamps onto the trial's 30 Hz grid (`np.interp`, which holds the endpoint value outside the source range rather than producing NaN), then discretisation. Values are collected across **all trials of all included sessions in pass 1** and the quintile edges are computed once, globally, so the bins are comparable across sessions. Verified: the resulting distribution is exactly uniform, `{bin_0…bin_4: 0.200 each}`.

ii.
```python
def interpolate_vector(source_t, source_values, query_t):
    return np.interp(query_t, source_t, source_values).astype(np.float32)
...
# pass 1
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
...
running_all = np.concatenate(running_values).astype(np.float32)
running_edges = robust_quintile_edges(running_all)
# pass 2
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 5 mapping + Key Decision 7: *"Discretize continuous outputs globally, not per session: Running-speed and pupil-diameter bin edges will be computed from all valid included timepoints across the converted dataset so class definitions are shared across sessions."*

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins. The 20th/40th/60th/80th percentiles of the pooled running values become four interior edges, and `np.digitize(..., right=False)` maps each value to `{0,1,2,3,4}`. A small monotonicity guard nudges tied percentiles apart by 1e-6 so that degenerate edges cannot collapse bins. The edges are saved to `metadata["running_speed_bin_edges"]`.

ii.
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles

def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Directly implements the instruction "Running speed, discretized into five equal percentile bins." The monotonicity guard is not separately justified in the notes; it is an engineering safeguard against a degenerate (e.g. mostly-stationary) distribution collapsing two edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly onto the trial's `grid`, the same array used for the neural data, so it is bin-for-bin aligned. Because both are evaluated in absolute session time on a hardware-synced clock, no offset correction is applied.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Same rationale as 2-d. Step 7's plot review: *"Running-speed and pupil traces are smooth after interpolation onto the common 30 Hz grid, with discretized bins tracking the continuous signals without obvious temporal offsets."*

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height` (the fitted pupil ellipse axes), timestamped by `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as the **element-wise maximum of width and height** (i.e. the ellipse's major axis), not `width` alone and not `area`. Blink frames are excluded by dropping non-finite samples — and in these NWB files the NaN mask is exactly the `likely_blink` mask (I verified `np.array_equal(np.isnan(width), likely_blink) == True` on 8 files), so this is equivalent to the SDK's blink filter.

ii.
```python
pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_timestamps = np.asarray(h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. Step 5 mapping: *"eye-tracking pupil width/height + blink mask → Compute pupil diameter as `max(width, height)`; use blink-masked values"*, citing `EyeTrackingTable` / `filter_on_blinks`. Step 1 records the SDK behaviour: *"blink frames are set to `NaN` for derived pupil/eye area signals"*, which is what makes the `isfinite` test equivalent to reading `likely_blink`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite (blink) samples are removed, then the remaining samples are linearly interpolated onto the trial's 30 Hz grid — which means blink gaps are bridged by interpolation across the gap, and the signal is held constant (not NaN) outside the tracked range. Degenerate cases are handled: zero valid samples raises (and those sessions were pre-excluded), one valid sample fills a constant. Then the same global-quintile discretisation as running speed, from pass-1 pooled values. Resulting distribution is exactly uniform.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    if valid.sum() == 1:
        return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Step 5 mapping: *"interpolate across valid timestamps onto 30 Hz trial grid; discretize with global quintile bins … Small blink-related gaps are filled by interpolation after applying reference invalid-frame masking."*

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: `robust_quintile_edges` on the pooled pass-1 pupil values gives the 20/40/60/80 percentiles as four interior edges, `np.digitize` maps to `{0..4}`, edges saved to `metadata["pupil_diameter_bin_edges"]`, labels `bin_0 … bin_4`. Because NaNs never survive `interpolate_pupil`, no NaN-to-bin fallback is needed.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
"pupil_diameter_bin_edges": pupil_edges.astype(float).tolist(),
```

iii. Directly implements "Pupil diameter, discretized into five equal percentile bins", with the same global-edges rationale as Key Decision 7.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated directly onto the shared trial `grid`, so bin-for-bin aligned with the neural trial; eye-tracking timestamps and ophys timestamps are on the same hardware-synced clock, so absolute-time interpolation is valid.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 3 records that *"Temporal synchronization was performed by recording experimental clocks on a single NI PCI-6612 digital IO board sampled at 100 kHz"*, which justifies cross-stream interpolation in absolute time.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually-exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that fixed order. There is no `other` fallback — a kept trial with none of the four flags set raises a `ValueError`.

ii.
```python
def trial_outcome_code(trials, idx, mapping) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. Step 5 mapping: *"trial outcome flags (`hit`, `miss`, `false_alarm`, `correct_reject`) → Static trial label"*, citing `Trial._get_trial_data`, which Step 1 identifies as the SDK code that *"Defines behavioral trial classes and outcomes."* These are the canonical labels of the change-detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A fixed mapping `hit→0, miss→1, false_alarm→2, correct_reject→3` (the order of `outcome_values`, which is also written to `output_values[4]`), then **broadcast constant across every time bin of the trial** so that it fits the time-varying output convention. Observed distribution in the converted data: `hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108`.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Step 5 Key Decision 5: *"Represent all outputs as time-varying: To satisfy the decoder format and simplify training, even static trial outcome will be repeated across all bins in a trial."* This follows the format spec's preference: "Can be time-varying or discrete values per trial. If at all possible, make it time-varying."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Five mechanisms, all pre-emptive rather than recovery-based:
- **Missing eye tracking**: a pre-scan opens every candidate NWB and drops any lacking `acquisition/EyeTracking/{pupil_tracking, eye_tracking}` (3 sessions). This was reactive — the first full run crashed on these files.
- **Blinks / NaN pupil samples**: dropped before interpolation; the gap is bridged by linear interpolation (and held constant outside the tracked range), so no NaN and no synthetic bin-0 label ever enters the output.
- **Running/pupil outside the sampled range**: `np.interp`'s default endpoint-hold, so the boundary value is repeated rather than becoming NaN.
- **Degenerate pupil streams**: 0 valid samples → raise (pre-excluded); exactly 1 valid sample → constant fill.
- **Degenerate percentile edges**: `robust_quintile_edges` nudges tied percentiles apart so the 5 bins cannot collapse.
- **Too-few-trial sessions**: `< 2` valid trials → skipped, checked both before and after conversion.
- **Malformed trial labels**: a kept trial with no outcome flag raises rather than being silently coded.

What is *not* handled: there is **no try/except around session processing**, so any unanticipated failure aborts the whole ~6-minute run; a trial whose `stop_time` overran the recording would be silently extrapolated (via the `np.clip` in `interpolate_matrix`) rather than clipped — though I confirmed no valid trial in this dataset falls outside the ophys timestamp range, so this path is never exercised.

ii.
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye
...
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
if valid.sum() == 1:
    return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
...
for i in range(1, len(percentiles)):
    if percentiles[i] <= percentiles[i - 1]:
        percentiles[i] = percentiles[i - 1] + 1e-6
...
right = np.clip(right, 0, n_src - 1)
left = np.clip(right - 1, 0, n_src - 1)
```

iii. Step 10 Issues: *"Missing pupil data in 3 active sessions caused full-conversion failure on the first Step 9 attempt. Resolution: exclude those sessions before both passes so global bin edges and converted sessions are computed on the same valid subset."* Step 10 Edge-case review also confirms *"Raw start/stop-derived `T` matched converted trial lengths exactly in the sanity checks"* and that the all-zero-neural warnings are genuine data sparsity rather than a conversion fault.

## 9-a. What are the most time-consuming steps of the code?

i. The AI instrumented timing per session and per pass. From `conversion_full_out.txt` the full run took **364.3 s**, split as:
- eye-tracking pre-scan (opens all 202 NWBs): ~16 s
- **pass 1** (`collect_global_statistics`, no event traces loaded): **106.4 s** (29%)
- **pass 2** (`convert_sessions`, event traces + interpolation): **241.7 s** (66%)

Within pass 2 the dominant cost is `interpolate_matrix`, which materialises a `(n_bins, n_neurons)` float array per trial (~2.05×10⁹ neuron-bins overall) plus the HDF5 read of the full `(T, N)` event array per session. The AI's own diagnosis matches: *"Neural interpolation is still the dominant expected cost because every kept trial needs event traces resampled onto the common grid."* Secondary cost: writing the 8.1 GB pickle.

ii.
```python
t0 = time.perf_counter()
raw = read_session_raw(session, load_events=False)
...
print(f"[pass1] {idx:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{valid_trial_counts[...]} valid trials in {elapsed:.2f}s")
...
elapsed = time.perf_counter() - t0
total_bins = sum(trial.shape[1] for trial in session_neural)
print(f"[pass2] {sess_num:03d}/{len(sessions)} session ...: "
      f"{len(session_neural)} trials, {raw['n_neurons']} neurons, {total_bins} bins in {elapsed:.2f}s")
```

iii. Step 6 lists neural interpolation as the expected bottleneck; Step 7 extrapolated a ~18.7 min full run from the 2-session sample by scaling on neuron-bins rather than session count, and flagged that this exceeded the 15-minute guidance. The actual run came in under that estimate at 6.1 min.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops remain scalar:
- **The per-trial loops in both passes.** Each trial separately calls `interpolate_matrix` / `interpolate_vector` / `interpolate_pupil` / `stimulus_identity_codes` / `stimulus_change_codes`. The natural vectorisation (and what the human reference does) is to interpolate each stream **once per session** onto a whole-session grid and then slice trials out by index — turning ~51,000 interpolation calls into 199 and eliminating the repeated `searchsorted` over the full timestamp array.
- **The inner Python loop in `stimulus_identity_codes`**, which walks presentation-by-presentation to map names to codes. This could be a single `np.array([...])` lookup or a pre-computed per-presentation code array built once per session.
- **The per-neuron interpolation is already vectorised** (`interpolate_matrix` does a gather-and-blend across all neurons at once), which is the one optimisation the AI did make and did document.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
    ...
# and inside stimulus_identity_codes:
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. Step 6 claims only *"Trial-level interpolation is vectorized over neurons within each trial. Reduces Python-loop overhead in the dominant resampling step."* The larger session-level opportunity is not identified anywhere in CONVERSION_NOTES. The implicit justification is that the achieved 6-minute runtime was acceptable, so no further optimisation was pursued.

## 9-c. What processing does the code repeat multiple times?

i. **Pass 1 and pass 2 duplicate a substantial amount of work on every session:**
- Every NWB is opened and its trials table, stimulus table, ophys timestamps, running stream and pupil stream are read **twice** (three times counting the `has_required_eye_tracking` pre-scan, which opens each file a third time). Only the event traces are skipped in pass 1 (`load_events=False`).
- The `keep_mask` is recomputed in both passes.
- `session_grid`, `interpolate_vector` (running) and `interpolate_pupil` are computed **per trial in both passes** on identical inputs, producing identical arrays. Pass 1 discards the interpolated traces after pooling them for percentile edges; pass 2 recomputes them from scratch.

This duplicated behaviour work is essentially the whole 106 s of pass 1, i.e. ~29% of total runtime.

- Two further computations are performed and then only printed, never used: `native_dt_by_session` and `valid_trial_counts`.

ii.
```python
# pass 1
raw = read_session_raw(session, load_events=False)
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
    pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# pass 2 — same reads, same grids, same interpolations recomputed
raw = read_session_raw(session)
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The AI frames the two-pass structure purely as a memory optimisation, never as duplicated work: Step 6, *"Two-pass design avoids storing all neural arrays while computing global bin edges. Keeps memory bounded and avoids a large temporary accumulation cost."* That rationale is sound for the 8 GB of neural data (which pass 1 correctly declines to load), but it does not explain why the small running/pupil traces are not cached between passes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly minor:
- **Unused columns read from the stimulus table**: `trials_id` and `flashes_since_change` are read, decoded and cast on every session and never referenced. `catch` is read from the trials table and only used inside `keep` (fine), but `change_time` is read, cast to float64, and never used at all after the `image_change` implementation was switched to the stimulus table.
- **Computed-then-printed-only values**: `native_dt_by_session` and `valid_trial_counts` are built across all 199 sessions in pass 1 and only appear in two `print` lines.
- **`RNG = np.random.default_rng(0)`** is created at module scope and never used (the conversion is deterministic).
- **The redundant pass-1 running/pupil interpolation** described in 9-c.
- **Materially: 3× neural upsampling of multiscope planes.** For the 34 `VisualBehaviorMultiscope` sessions, resampling 11 Hz traces onto a 30 Hz grid stores roughly two interpolated (information-free) samples for every measured one, inflating storage and training cost with no added information. This is a real contributor to the 8.1 GB output size.
- Corresponding metadata inaccuracy: `"resampling_reference": "common 30 Hz grid derived from source ophys timestamps"` — the grid is actually derived from trial `start_time`/`stop_time`, not from ophys timestamps.

ii.
```python
RNG = np.random.default_rng(0)  # never used
...
stim["trials_id"] = stim["trials_id"].astype(np.int64)            # never used
stim["flashes_since_change"] = stim["flashes_since_change"].astype(np.int64)  # never used
trials["change_time"] = trials["change_time"].astype(np.float64)  # never used after the is_change rewrite
...
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
...
print("[setup] native ophys dt range:", min(native_dt.values()), max(native_dt.values()))
print("[setup] valid trial count range:", min(valid_counts.values()), max(valid_counts.values()))
```

iii. CONVERSION_NOTES does not discuss discarded computation at all. The diagnostic prints are consistent with the instructions' request to "Print timing information to find bottlenecks" and with Step 4's cross-source consistency checks (the dt range print is what revealed the mixed 31/11 Hz rates). The leftover `change_time`/`trials_id`/`flashes_since_change` reads are vestigial from the first `image_change` implementation, which was replaced in Step 7/8 and not cleaned up.
