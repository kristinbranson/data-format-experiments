# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK `VisualBehaviorOphysProjectCache` API. It discovered `/app/data/.../behavior_ophys_experiments/behavior_ophys_experiment_<id>.nwb` files by globbing, joined them to `project_metadata/ophys_experiment_table.csv` on `ophys_experiment_id`, and then read the NWB/HDF5 contents directly with `h5py`, mirroring the SDK field paths (`intervals/trials`, `intervals/<image>_presentations`, `processing/ophys/event_detection`, `processing/running/speed`, `acquisition/EyeTracking`). Three inclusion filters are applied at this stage: (1) only experiments whose NWB file is present locally, (2) only **active** (non-`passive`) experiments, (3) only experiments that contain an `acquisition/EyeTracking` group with both `pupil_tracking` and `eye_tracking` (3 sessions excluded: 795953296, 806456687, 833631914). Notably, **no `project_code` filter is applied**, so both `VisualBehavior` (single-plane) and `VisualBehaviorMultiscope` experiments are included. The pipeline is two-pass: pass 1 re-reads every session without neural data to build global quintile edges and the image vocabulary; pass 2 re-reads every session with neural data and builds the trials. Final scope: 199 sessions, 38 mice, 51,075 trials, 29,168 neurons.

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
    sessions = [SessionInfo(ophys_experiment_id=int(row.ophys_experiment_id), path=file_map[...],
                            mouse_id=str(int(row.mouse_id)), targeted_structure=str(row.targeted_structure),
                            session_type=str(row.session_type), project_code=str(row.project_code),
                            passive=bool(row.passive))
                for row in exp_table.itertuples(index=False)]
    filtered_sessions = [s for s in sessions if has_required_eye_tracking(s.path)]
    ...
    return filtered_sessions
```
```python
def read_session_raw(session, load_events=True):
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
        trials = read_interval_table(trial_group, ["go","catch","aborted","auto_rewarded",
            "hit","miss","false_alarm","correct_reject","change_time","start_time","stop_time"])
        stim_group = choose_task_presentation_group(h5f)
        ...
        ophys_timestamps = np.asarray(h5f["processing"]["ophys"]["event_detection"]["timestamps"], ...)
        events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], ...)
        pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], ...)
```

iii. From CONVERSION_NOTES Step 5 Key Decision 9: *"Load NWB content with `h5py` rather than the SDK session object … the current environment's `pynwb/hdmf` stack cannot instantiate these NWB 2.6.0 files through `BehaviorOphysExperiment.from_nwb_path` because of an `external_resources` abstract-method mismatch. Direct HDF5 reads will therefore mirror the SDK field definitions explicitly."* Key Decision 1 justifies excluding passive sessions: *"the decoder task requires trial outcome and the strategy paper's behavioral analyses focus on active sessions. Passive sessions also produce degenerate outcomes (sample passive file: all go trials are misses and all catch trials are correct rejects)."* Step 10 Check 5 justifies excluding the 3 eye-tracking-less sessions: they have no `acquisition/EyeTracking` group at all, and excluding them before both passes keeps the global bin edges and the converted sessions on the same subset. The AI validated the h5py path against the SDK semantics with `np.allclose` raw-vs-converted spot checks (Step 10 Check 2).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` strings from `ophys_experiment_table.csv`. They are registered lazily in the order sessions are processed (sessions are ordered by `ophys_experiment_id`), and `subject_idx` stores the index of the owning mouse for each session. Result: 38 mice.

ii.
```python
mouse_id=str(int(row.mouse_id)),
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
...
data["subject_idx"].append(subject_idx)
...
data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: *"NWB `general/subject/subject_id` / metadata `mouse_id` → `subjects`, `subject_idx` … Use mouse identifier strings"*. `mouse_id` is the SDK's canonical unique animal identifier, and the AI cross-checked the count (38 mice) against the local metadata tables in Step 2 and Step 9.

## 1-c. How are the data split into sessions?

i. One **NWB experiment file = one decoder session**. The AI deliberately did not group imaging planes by `ophys_session_id`. For the single-plane `VisualBehavior` experiments this is a 1:1 mapping, but the local subset also contains `VisualBehaviorMultiscope` experiments (34 active experiment files belonging to only 6 real `ophys_session_id`s from one mouse). Those 34 planes are emitted as 34 independent "sessions" that share identical trials/behavior and each contain only 4–20 neurons. The verification log accordingly reports *"Subject 457841: 34 sessions"*. Sessions are ordered by `ophys_experiment_id` rather than `date_of_acquisition`.

ii.
```python
exp_table = exp_table.sort_values("ophys_experiment_id")
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    data["neural"].append(session_neural)
    data["brain_region_idx"].append(np.full(raw["n_neurons"], region_idx, dtype=np.int64))
```
```python
"session_ophys_experiment_ids": [int(s.ophys_experiment_id) for s in sessions],
```

iii. CONVERSION_NOTES Step 4 discrepancy table: *"Treat each NWB experiment file as one decoder session because neural traces are experiment-specific; preserve subject/session metadata so multiple experiments from one behavior session remain linked through subject/session fields."* The stated rationale is that the neural traces (and their ROI tables) are per-experiment, so an experiment is the natural neural unit. The AI recorded the 284-experiments-vs-247-sessions discrepancy in Step 2/Step 4 but chose not to merge planes.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept when `(go or catch) and not aborted and not auto_rewarded`. Each kept trial spans the full `start_time` → `stop_time` window, so trials are variable length (211–377 bins, mean 254.7 at 30 Hz ≈ 8.5 s), covering both the pre-change flash sequence and the post-change response window.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```
```python
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: *"Segment trials from `start_time` to `stop_time`: Trial boundaries will come directly from the NWB `intervals/trials` table, after filtering to keep only `(go or catch) and not aborted and not auto_rewarded`."* Step 4 records that this matches the SDK's `Trial._get_trial_data` taxonomy and the instruction to *"Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."*

## 1-e. How are trials filtered based on quality controls?

i. Three curation rules: (1) the `(go|catch) & ~aborted & ~auto_rewarded` mask above; (2) sessions with fewer than 2 kept trials are skipped, checked both before conversion (on the raw mask) and again after building the trial list; (3) sessions lacking eye-tracking data are excluded at the session level (see 1-a). No explicit `change_time.notna()` requirement is applied, and no clipping of trials that would run past the end of the recording. In practice neither omission bites: on the local files every `(go|catch) & ~aborted & ~auto_rewarded` trial has a finite `change_time`, and every trial's `stop_time` falls ~600 s before the last ophys timestamp. Minimum kept-trial count across included sessions is 39.

ii.
```python
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session ... because it has {int(raw['keep_mask'].sum())} valid trials")
    continue
...
if len(session_neural) < 2:
    print(f"[pass2] skipping session ... after conversion because it has {len(session_neural)} trials")
    continue
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: *"Require at least two valid trials per session after filtering: The active-session subset already satisfies this in a spot/global check (202 active sessions; minimum 39 valid trials after filtering), so no expected extra session loss from this rule."* Step 3 Curation Steps records the reference rationale: *"Aborted trials are defined by licks before the scheduled change and are excluded from hit/false-alarm/d-prime calculations in the SDK metrics. Free-reward / auto-reward trials occur for the first 5 trials of sessions and after 10 consecutive misses; these should not be treated as standard go/catch contingencies."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is the pre-computed **detected calcium event** traces, `processing/ophys/event_detection/data` (shape `(T, n_rois)`) with `processing/ophys/event_detection/timestamps`. dF/F (`processing/ophys/dff`) is present in the files but deliberately **not** used.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```
```python
"neural_signal": "precomputed calcium events",
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: *"Use `events` instead of `dff_traces`: The whitepaper explains dF/F generation, but the strategy paper explicitly states its neural analyses used detected calcium events. Events are already present in the NWB files and best match the reference analyses."* This is directly supported by the paper text the AI extracted: *"For all analysis of neural data we used the detected calcium events … This process produces, for each cell, a set of calcium events each with a time and magnitude"* and *"We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."*

## 2-b. How is the `neural` data processed?

i. The only processing is per-trial **linear interpolation of the event traces onto a common 30 Hz grid** spanning `start_time`→`stop_time`, followed by a transpose to `(n_neurons, n_timepoints)` and a cast to `float32`. No smoothing, normalisation, z-scoring or baseline correction is applied. Planes are not merged (each plane is its own session), so there is no cross-plane stacking. The interpolation is hand-written and vectorised over neurons; it clamps before the first timestamp but linearly extrapolates past the last one (never triggered on this data).

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, n_src - 1)
    left = np.clip(right - 1, 0, n_src - 1)
    same = right == left
    t0 = source_t[left]; t1 = source_t[right]
    denom = np.where(np.abs(t1 - t0) < 1e-12, 1.0, t1 - t0)
    w = np.where(same, 0.0, (query_t - t0) / denom)
    out = source_values[left] * (1.0 - w[:, None]) + source_values[right] * w[:, None]
    return out.astype(np.float32)
```
```python
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. CONVERSION_NOTES Step 3: *"The strategy paper states its neural analyses use detected calcium events rather than raw dF/F, and for event-triggered analyses it linearly interpolates onto common 30 Hz timestamps relative to behavioral events."* The paper text confirms: *"We compute the behavioral event triggered response by isolating the calcium events around the triggering behavioral event, then linearly interpolating onto a consistent set of 30hz timestamps."* Step 6 notes *"Trial-level interpolation is vectorized over neurons"* as the main speed-up.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ad-hoc neuron filtering is added. The AI relies on the released ROI curation and adds a defensive `valid_roi` filter: if the event array's column count equals the `cell_specimen_table` row count and the table has a `valid_roi` column that is not all-True, the columns are subset to the valid ROIs. On the local files the released event traces already contain only valid ROIs, so this is a no-op. Resulting counts: 29,168 neurons, mean 146.6/session, range 4–666.

ii.
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
n_neurons = events.shape[1]
```

iii. CONVERSION_NOTES Step 4: *"Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc neuron filtering beyond reference QC/filtering already reflected in the files."* Step 1 records that the SDK default is `exclude_invalid_rois=True` in `CellSpecimens.__init__`, which the AI mirrors. Step 10 Check 3 confirms *"Resulting neuron counts match raw released valid-ROI counts."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. The 30 Hz query grid begins exactly at `trials['start_time']` and steps by 1/30 s until `stop_time`; the event traces (and running, pupil, stimulus labels) are all evaluated at those same absolute times, so every stream shares one index axis. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (trials are variable length).

ii.
```python
grid = session_grid(start, stop)                                  # absolute times, 1/30 s steps from start_time
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
image_codes  = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
image_change = stimulus_change_codes(raw["stimulus"], grid)
```
```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES Step 10 Check 3 (Temporal alignment): *"all trial streams are aligned in absolute experiment time and resampled to a common 30 Hz grid from raw trial `start_time` / `stop_time`."* The AI verified this with raw-data `np.allclose` checks on sessions 775614751 (trial 0) and 939327156 (trial 10), and visually in `processing_<id>.png`: *"No visual sign of cross-stream temporal misalignment in the reviewed plots."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **33.33 ms (30 Hz), uniform across all trials and sessions.** Yes — every stream is rebinned/resampled: the native ophys rate is ~31 Hz for the single-plane experiments and ~11 Hz for the multiscope planes, and both are linearly interpolated onto the common 30 Hz grid. Running (~60 Hz) and eye tracking (~30 Hz) are likewise interpolated to the same grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
"resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: *"Use a common 30 Hz time base for all sessions: The target format requires one shared bin size across sessions, while local data mix ~31 Hz single-plane and ~11 Hz multi-plane ophys sampling. The strategy paper linearly interpolates calcium event responses onto common 30 Hz timestamps for neural and running analyses, and behavior/eye tracking are naturally 30 Hz, so 30 Hz is the most defensible common grid."* The format spec's requirement that *"Time bins should be the same size for all trials and sessions"* is the driving constraint, since the AI's session set mixes two acquisition rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentations table** (`intervals/<image_set>_presentations`), using `image_name`, `start_time`, `stop_time` and `omitted` — not from the trials table's `initial_image_name`/`change_image_name`. The presentations group is chosen automatically as the `*_presentations` group that has `active`/`image_name` columns and the most active rows preferring a `change_detection` stimulus block.

ii.
```python
def choose_task_presentation_group(h5f):
    for name in h5f["intervals"].keys():
        if not name.endswith("_presentations") or name == "trials": continue
        grp = h5f["intervals"][name]
        if "active" not in grp or "image_name" not in grp: continue
        ...
        score = (has_change_detection, n_active)
```
```python
stim = read_interval_table(stim_group, ["start_time","stop_time","image_name",
        "is_change","omitted","trials_id","active","flashes_since_change"])
```

iii. CONVERSION_NOTES Step 5 variable mapping: *"stimulus-presentation `image_name`, `start_time`, `stop_time`, `omitted`, active task block only → `output[image_identity]` … use actual image name during image display; use `gray` during gray-screen or omission periods."* Step 1 notes that *"`stimulus_presentations` contains more than just the active task block in newer SDK releases; for VBO change-detection analyses, the active block is identified via `stimulus_block_name` containing `change_detection`."*

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A piecewise-constant per-bin categorical signal. For each 30 Hz bin the code finds the last presentation whose `start_time` ≤ t; if t also falls before that presentation's `stop_time` (flashes are 250 ms) and the flash is not `omitted`, the bin gets that image's code, otherwise it gets the code for an explicit **`gray`** class. The vocabulary is global across all sessions, built in pass 1, with `gray` forced to index 0: 17 classes total. The resulting distribution is `gray` 0.669 and each of 16 images 0.019–0.022 — i.e. two-thirds of all bins are the `gray` class, reflecting the 250 ms image / 500 ms grey cadence.

ii.
```python
image_values = ["gray"] + sorted(image_names)
image_to_code = {name: idx for idx, name in enumerate(image_values)}
```
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = (idx >= 0) & (idx < len(starts))
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        target = np.full(sub_idx.shape, image_to_code["gray"], dtype=np.int64)
        for i, (name, is_omitted) in enumerate(zip(names, omit)):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        codes[np.flatnonzero(valid)[in_interval]] = target
    return codes
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: *"Use `gray` as an explicit image-identity class: Because the task includes 500 ms gray periods and omissions extend gray instead of showing an image, a `gray` category is needed for a complete time-varying identity signal."*

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at exactly the same absolute-time 30 Hz `grid` array used to interpolate the neural events, so index *k* of the image row corresponds to index *k* of the neural matrix by construction. No separate resampling or offsetting is done.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
image_codes  = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
...
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 10 Check 2 (Output sanity check 1/2): *"Session 775614751, trial 0: converted `image_identity`, `image_change`, `running_speed_bin`, `pupil_diameter_bin`, and `trial_outcome` each matched independently reconstructed raw-data equivalents exactly"* with `np.allclose == True`. Step 12 adds the visual alignment check from the processing plots.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the stimulus presentations table's `is_change` flag (together with `omitted`, `start_time`, `stop_time`). It does **not** use the trials table's `change_time`/`go` columns. Because the SDK sets `is_change = False` for sham (catch-trial) changes — they are flagged separately as `is_sham_change` — this yields 1 only for genuine image changes.

ii.
```python
stim["is_change"] = stim["is_change"].astype(bool)
stim["omitted"] = stim["omitted"].astype(bool)
```
```python
changed = is_change[sub_idx] & (~omitted[sub_idx])
```

iii. CONVERSION_NOTES Step 5 variable mapping: *"stimulus-presentation `is_change`, `start_time`, `stop_time` → `output[image_change]` … Marks the post-change flashed image itself rather than a one-bin impulse; catch trials remain 0."* Step 1 records `compute_is_sham_change` as the SDK function that separates catch-trial flashes from real changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary per-bin indicator: 1 for every 30 Hz bin that falls inside the presentation interval of a non-omitted `is_change` flash, i.e. the 250 ms of the changed image (~7–8 bins), 0 everywhere else including the following grey period. Global distribution: 0.974 no-change / 0.026 change. This replaced an earlier single-bin impulse implementation.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    valid = (idx >= 0) & (idx < len(starts))
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        changed = is_change[sub_idx] & (~omitted[sub_idx])
        codes[np.flatnonzero(valid)[in_interval]] = changed.astype(np.int64)
    return codes
```

iii. CONVERSION_NOTES Step 10 Issues Found and Resolved: *"The initial `image_change` target was too sparse because it used a single-bin impulse at `change_time`. Resolution: relabeled `image_change` as the changed-image presentation window using raw stimulus `is_change` intervals; sample validation accuracy moved from below chance to above chance."* Step 6 repeats this: *"`image_change` is constructed from the stimulus table's `is_change` presentation interval instead of a single-bin impulse at `change_time`, which yields a less degenerate decoder target while staying aligned to the task structure."*

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the variable is natively binary. Categories are `["no_change", "change"]` = `[0, 1]`, produced directly by casting the boolean `is_change & ~omitted` to int.

ii.
```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
    ...
]
...
codes[assign] = changed.astype(np.int64)
```

iii. The Decoder Task spec defines the variable as binary (*"Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0"*), so the only design freedom the AI exercised was the width of the 1-window (the 250 ms change flash), justified in 4-b.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — evaluated on the same absolute-time 30 Hz `grid` as the neural interpolation, so the two share an index axis exactly.

ii.
```python
grid = session_grid(start, stop)
neural_trial  = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_change  = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Step 12 explicitly re-checked this because `image_change` had the smallest above-chance margin: *"Raw-value checks on 3 positive trials from `ophys_experiment_id 775614751`: trial 0: 8 positive bins, `np.allclose == True`; trial 1: 8 positive bins …; trial 2: 7 positive bins …"* and concluded *"the modest but above-chance `image_change` performance appears to be a property of the required whole-trial multitask formulation and sparse event representation, not a conversion bug."*

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — the SDK's default **filtered** running speed (the NWB also contains `speed_unfiltered` and `dx`, which are not used).

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 5 variable mapping: *"running speed timeseries → `output[running_speed_bin]` … Uses filtered running speed, matching SDK default"*, referencing `BehaviorSession.running_speed` / `RunningSpeed.from_stimulus_file`. Step 3 records the whitepaper's filtering chain (*"unwrap encoder voltage, remove transients/spikes, remove z-score >= 10 artifacts, smooth with a 10 Hz low-pass Butterworth filter"*) as already applied to this stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. `np.interp` linear interpolation from the native running timestamps onto the trial's 30 Hz grid, cast to float32, then quintile-binned. `np.interp` clamps to the endpoint values outside the sampled range rather than producing NaN, so no missing-value substitution is needed.

ii.
```python
def interpolate_vector(source_t, source_values, query_t):
    return np.interp(query_t, source_t, source_values).astype(np.float32)
...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 5 variable mapping: *"Interpolate running speed onto 30 Hz trial grid; discretize with global quintile bins across included active-session timepoints."* Directly grounded in the paper methods the AI extracted: *"Running speed traces were processed in the same manner as calcium event traces … isolating running timepoints around the triggering behavioral event then linearly interpolating onto a common 30hz timeseries."*

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **equal-percentile (quintile) bins** with edges at the 20th/40th/60th/80th percentiles, computed **globally in pass 1** over the interpolated running values of *every* kept trial in *every* included session (so the categories are shared across sessions). Ties are broken by nudging non-increasing edges apart by 1e-6. `np.digitize(..., right=False)` maps values to bins 0–4. Realised edges: `[-0.0041, 0.303, 15.78, 33.44]` cm/s, giving an exactly uniform 0.200/0.200/0.200/0.200/0.200 distribution.

ii.
```python
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles

def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```
```python
running_all = np.concatenate(running_values).astype(np.float32)
running_edges = robust_quintile_edges(running_all)
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: *"Discretize continuous outputs globally, not per session: Running-speed and pupil-diameter bin edges will be computed from all valid included timepoints across the converted dataset so class definitions are shared across sessions."* This follows the Decoder Task requirement *"Running speed, discretized into five equal percentile bins."* The `robust_quintile_edges` tie-breaking exists because running speed is exactly 0 for long stationary stretches, which would otherwise collapse bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Same `grid` as the neural data — interpolated at the identical absolute times, so bin *k* of the running row is simultaneous with bin *k* of the neural matrix.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 10 Check 3 (Temporal alignment): *"all trial streams are aligned in absolute experiment time and resampled to a common 30 Hz grid."* The raw-vs-converted `np.allclose` output sanity checks on two sessions covered `running_speed_bin` specifically, and the `processing_*.png` panels overlay the continuous trace and the discretised bin to show no offset.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height`, timestamped by `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as the element-wise **maximum of width and height** (the major axis of the ellipse fit). In these NWB files those width/height arrays are already NaN exactly on `likely_blink` frames (verified: the NaN mask equals `likely_blink` exactly, ~9% of frames), and the AI drops non-finite samples before interpolating — so blink frames are excluded.

ii.
```python
pupil_width  = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    if valid.sum() == 1:
        return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 variable mapping: *"eye-tracking pupil width/height + blink mask → `output[pupil_diameter_bin]` — Compute pupil diameter as `max(width, height)`; use blink-masked values; interpolate across valid timestamps … Small blink-related gaps are filled by interpolation after applying reference invalid-frame masking"*, citing `EyeTrackingTable` / `filter_on_blinks`. The `max(width, height)` convention mirrors the SDK's own `compute_circular_area`, which the AI read in Step 1: *"Calculate the area of the pupil as a circle using the max of the height/width as radius."*

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: drop non-finite (blink) samples, `np.interp` onto the trial's 30 Hz grid (which bridges blink gaps), then global quintile binning. Endpoint clamping means no NaN reaches the discretiser.

ii.
```python
pupil_cont  = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins  = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Step 5: *"Compute pupil diameter as `max(width, height)`; use blink-masked values; interpolate across valid timestamps onto 30 Hz trial grid; discretize with global quintile bins."* Same rationale as running speed — interpolation onto the shared ophys/behaviour grid keeps all streams on one index axis; blink removal before interpolation prevents blink artefacts from contaminating neighbouring bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global equal-percentile bins, same machinery as running speed, with edges computed in pass 1 from every kept trial of every included session. Realised edges `[36.91, 41.90, 46.44, 52.72]` px, distribution exactly 0.200 per bin.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
"pupil_diameter_bin_edges": pupil_edges.astype(float).tolist(),
```

iii. Step 5 Key Decision 7 (global, not per-session, discretisation) plus the Decoder Task requirement *"Pupil diameter, discretized into five equal percentile bins."* The edges are written into `metadata` so the binning is auditable and reproducible (Step 10 Check 5).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same `grid` as the neural data; interpolation is evaluated at the identical absolute times, so the rows are index-aligned with the neural matrix.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Step 10 Check 3 and Check 2: the raw-data reconstruction of `pupil_diameter_bin` for session 775614751 trial 0 and session 939327156 trial 10 matched the converted rows with `np.allclose == True`. Step 1 notes the eye-tracking and ophys clocks are hardware-synced (*"experimental clocks on a single NI PCI-6612 digital IO board sampled at 100 kHz"*, Step 3), so direct interpolation onto the ophys-derived grid is valid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, read for the kept trial index.

ii.
```python
trials = read_interval_table(trial_group, ["go","catch","aborted","auto_rewarded",
    "hit","miss","false_alarm","correct_reject","change_time","start_time","stop_time"])
for key in ("go","catch","aborted","auto_rewarded","hit","miss","false_alarm","correct_reject"):
    trials[key] = trials[key].astype(bool)
```

iii. Step 5 variable mapping: *"trial outcome flags (`hit`, `miss`, `false_alarm`, `correct_reject`) → `output[trial_outcome]` … Categories: `hit`, `miss`, `false_alarm`, `correct_reject`"*, referencing the SDK's `Trial._get_trial_data`, which Step 1 identifies as the function that *"Defines behavioral trial classes and outcomes: `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true flag in the fixed order `hit, miss, false_alarm, correct_reject` is mapped to code 0–3, and that scalar is broadcast across all time bins of the trial so the output is time-varying in shape but constant within a trial. If a kept trial has none of the four flags set, the code **raises** rather than falling back to an "other" class. Realised distribution: hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

ii.
```python
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
...
outcome_code  = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Step 5 Key Decision 5: *"Represent all outputs as time-varying: To satisfy the decoder format and simplify training, even static trial outcome will be repeated across all bins in a trial."* This follows the format spec's *"If at all possible, make it time-varying."* The four labels are mutually exclusive and exhaustive for non-aborted, non-auto-rewarded trials, which is why the AI treated a missing label as an error condition rather than a category.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, plus some notable gaps:
- **Missing eye tracking**: every candidate NWB is pre-screened by opening it and checking for `acquisition/EyeTracking` with `pupil_tracking` and `eye_tracking`; 3 active sessions (795953296, 806456687, 833631914) are dropped *before both passes* so the global bin edges and the converted set stay consistent. This was added after the first full run crashed on them.
- **Blink / invalid pupil frames**: non-finite pupil samples are excluded before interpolation, and the gap is bridged by linear interpolation. A session with zero valid pupil samples raises.
- **Out-of-range interpolation**: `np.interp` clamps at the endpoints for running and pupil, so no NaN propagates into `np.digitize`. The hand-written `interpolate_matrix` clamps before the first ophys timestamp but linearly extrapolates past the last one (not triggered — every trial's `stop_time` is ~600 s inside the recording).
- **Degenerate sessions**: sessions with <2 kept trials are skipped, checked both before and after conversion.
- **Variable stimulus-table naming**: `choose_task_presentation_group` scores the available `*_presentations` groups rather than hard-coding a name, and raises if none qualifies.
- **Gaps**: there is no per-session `try/except`, so an unexpected failure in any one session aborts the whole conversion; and `trial_outcome_code` raises on an unlabeled trial rather than assigning a fallback class.

ii.
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye

filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
excluded = len(sessions) - len(filtered_sessions)
if excluded:
    print(f"[setup] excluded {excluded} active sessions missing eye-tracking pupil data")
```
```python
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
...
if int(raw["keep_mask"].sum()) < 2: continue
...
if best_name is None:
    raise RuntimeError(f"Could not find active task stimulus table in {h5f.filename}")
```

iii. CONVERSION_NOTES Step 10 Issues Found and Resolved: *"Missing pupil data in 3 active sessions caused full-conversion failure on the first Step 9 attempt. Resolution: exclude those sessions before both passes so global bin edges and converted sessions are computed on the same valid subset."* Step 5 mapping: *"Small blink-related gaps are filled by interpolation after applying reference invalid-frame masking."* Step 10 Check 5 also documents the off-by-one review: *"Raw start/stop-derived `T` matched converted trial lengths exactly in the sanity checks."* The 2,467 all-zero-neural trials flagged by the verifier were investigated and attributed to genuine event sparsity (*"matched the raw 30 Hz-interpolated event matrix exactly with `np.allclose(...) == True` and `max_abs_diff == 0.0`"*), not missing data.

## 9-a. What are the most time-consuming steps of the code?

i. The full run took 364 s for 199 sessions. Per the logs, pass 1 (metadata + trials + stimulus + running + pupil reads, plus per-trial behaviour interpolation) took 106 s (~0.53 s/session), and pass 2 (the same reads *plus* the full `(T, n_rois)` event array read and per-trial neural interpolation) took ~258 s (~1.3 s/session). So the dominant costs are (1) reading the large `event_detection/data` array out of HDF5 and (2) the per-trial `interpolate_matrix` gather, which for every trial fancy-indexes the whole-session event array twice. A third, smaller cost is the pre-screening pass that opens all 284 NWB files just to test for an `EyeTracking` group. Writing the 8.1 GB pickle is also non-trivial.

ii.
```python
t0 = time.perf_counter()
raw = read_session_raw(session)
...
elapsed = time.perf_counter() - t0
total_bins = sum(trial.shape[1] for trial in session_neural)
print(f"[pass2] {sess_num:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{len(session_neural)} trials, {raw['n_neurons']} neurons, {total_bins} bins in {elapsed:.2f}s")
```
```python
left_vals = source_values[left]      # (n_query, N) gather, per trial
right_vals = source_values[right]
out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
```

iii. CONVERSION_NOTES Step 6: *"Neural interpolation is still the dominant expected cost because every kept trial needs event traces resampled onto the common grid."* Step 7 extrapolated from the 2-session sample (*"Sample pass 2: 1.35 s / session on 7,557,084 neuron-bins → ~18.1 min for 2,052,381,553 neuron-bins"*) and flagged that the estimate exceeded the 15-minute guideline; the actual full run came in well under that at 6 minutes. Per-session timing prints were added specifically for bottleneck detection.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
- **The per-trial loop in both passes.** Every trial independently calls `searchsorted` + a two-sided gather against the *whole-session* arrays. Because all trials share one uniform 30 Hz grid, the session could have been interpolated **once** onto a single session-wide 30 Hz timeline and then sliced per trial, turning ~250 gathers per session into one. This is the single largest missed speed-up and it sits directly on the dominant cost path.
- **The Python `for` loop inside `stimulus_identity_codes`**, which looks up `image_to_code[name]` one presentation at a time. Presentation image codes could be precomputed once per session as an integer array and gathered.
- **The `has_required_eye_tracking` loop**, which opens 284 HDF5 files serially; this could be folded into pass 1 (or parallelised).

What the AI *did* vectorise: interpolation across neurons within a trial (`interpolate_matrix` handles all neurons at once instead of looping), and the stimulus label assignment across time bins via `searchsorted`.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):      # ~250 iterations/session
    grid = session_grid(start, stop)
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
    running_cont = interpolate_vector(...)
    pupil_cont   = interpolate_pupil(...)
```
```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):   # Python loop over presentations
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. CONVERSION_NOTES Step 6 Code speedups added: *"Trial-level interpolation is vectorized over neurons within each trial"* — the AI recognised and removed the neuron-level loop but did not identify the trial-level loop or the presentation-name loop as remaining vectorisation targets. Step 7 treated the runtime estimate as acceptable-ish (*"The current full-run estimate still exceeds 15 minutes, so Step 9 should treat optimization as live work rather than a closed issue"*) but no further vectorisation was implemented.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read **three times**:
1. `has_required_eye_tracking` — opens the file purely to test for an `EyeTracking` group;
2. `collect_global_statistics` → `read_session_raw(..., load_events=False)` — reads trials, the stimulus table, ophys timestamps, running, and pupil;
3. `convert_sessions` → `read_session_raw(...)` — reads all of the above **again**, plus the event array.

Consequently the per-trial **running-speed and pupil interpolation is computed twice** for every trial in the dataset: once in pass 1 to accumulate values for the quintile edges, and again in pass 2 to produce the saved bins. The trial mask, the stimulus-group selection, and `session_native_dt` are likewise recomputed in both passes. Within pass 2, `stimulus_identity_codes` and `stimulus_change_codes` each independently redo the same `searchsorted`/interval-membership computation over the same grid.

ii.
```python
# pass 1
raw = read_session_raw(session, load_events=False)
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
    pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# pass 2 — same file, same trials, same interpolation redone
raw = read_session_raw(session)
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. CONVERSION_NOTES Step 6 frames the two-pass design as a **memory** optimisation, not an oversight: *"Two-pass design avoids storing all neural arrays while computing global bin edges"* and *"pass 1 computes global image vocabulary and global quintile edges for running speed and pupil diameter; pass 2 builds per-trial neural/output arrays and metadata."* The trade-off — recomputing cheap behaviour interpolation instead of holding it in RAM — is deliberate; the notes do not, however, acknowledge that the pass-1 running/pupil results could simply have been cached (they are small: ~13 M floats total) to avoid the duplication entirely, nor that the eye-tracking pre-screen adds a third full pass of file opens.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items are computed or read and then never used in the saved output:
- **Unused columns read from HDF5 every session**: `flashes_since_change` and `trials_id` from the stimulus table are parsed and type-cast but never referenced; `stim["active"]` is stored and never used (the `active` check happens only inside `choose_task_presentation_group` on a separate read); `trials["change_time"]` is read and cast but never used anywhere in the conversion.
- **Diagnostics-only computation**: `valid_trial_counts` and `native_dt_by_session` are built for all 199 sessions and only printed as a min/max range; `session_native_dt` is recomputed inside the plotting path.
- **Dead code**: the module-level `RNG = np.random.default_rng(0)` is never used; `interpolate_matrix`'s `n_src == 1` branch is unreachable in practice.
- **The eye-tracking pre-screen** opens all 284 NWB files, information that pass 1 would have obtained anyway.
- **Data inflation**: resampling the ~11 Hz multiscope planes up to 30 Hz roughly triples their sample count without adding information, and the full pickle is **8.1 GB** — the interpolated neural array is `float32` at 30 Hz for 51,075 trials. Much of the pipeline's time and all of the disk cost flow from this.

ii.
```python
RNG = np.random.default_rng(0)          # never used
```
```python
stim["trials_id"] = stim["trials_id"].astype(np.int64)            # never used
stim["active"] = stim["active"].astype(bool)                      # never used downstream
stim["flashes_since_change"] = stim["flashes_since_change"].astype(np.int64)   # never used
trials["change_time"] = trials["change_time"].astype(np.float64)  # never used
```
```python
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
...
print("[setup] native ophys dt range:", min(native_dt.values()), max(native_dt.values()))
print("[setup] valid trial count range:", min(valid_counts.values()), max(valid_counts.values()))
```

iii. The CONVERSION_NOTES do not identify any of these as waste. The extra columns were read as part of a deliberately broad schema mirror of the SDK tables (Step 5 lists `image_name`, `start_time`, `stop_time`, `omitted`, `is_change` as the fields actually needed), and the diagnostic prints were added under Step 6's instruction to *"Print timing information to find bottlenecks"* and to support Step 9/10 consistency checks (*"[setup] native ophys dt range"* was used to justify the 30 Hz common grid in Key Decision 3). The 30 Hz resampling itself is justified in Key Decision 3 rather than treated as overhead.
