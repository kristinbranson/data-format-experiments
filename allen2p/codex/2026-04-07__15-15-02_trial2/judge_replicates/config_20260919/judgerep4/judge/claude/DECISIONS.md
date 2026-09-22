# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object model. It reads the local release directly:

- It reads `project_metadata/ophys_experiment_table.csv` and globs every
  `behavior_ophys_experiment_*.nwb` file in `behavior_ophys_experiments/`, then intersects the two
  (284 experiment files present locally).
- It filters out **passive** experiments (`~exp_table["passive"]`), leaving 202 active experiments.
- It then opens every remaining NWB file once more and drops any experiment that lacks
  `acquisition/EyeTracking/{pupil_tracking, eye_tracking}` (3 sessions dropped: `795953296`,
  `806456687`, `833631914`), leaving **199** experiments.
- It does **not** filter on `project_code`, so both `VisualBehavior` (single-plane, VISp) and
  `VisualBehaviorMultiscope` (multi-plane, VISp + VISl) experiments are included.
- Each NWB is then read with `h5py` field-by-field: `intervals/trials`, the active image
  presentations interval table, `processing/ophys/event_detection/{data,timestamps}`,
  `processing/running/speed/{data,timestamps}`, and
  `acquisition/EyeTracking/pupil_tracking/{width,height}` + `eye_tracking/timestamps`.
- Data are read twice: pass 1 (`collect_global_statistics`, `load_events=False`) to build the global
  image vocabulary and the global running/pupil quintile edges; pass 2 (`convert_sessions`) to build
  the actual arrays.

ii.
```python
def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    exp_table = exp_table.sort_values("ophys_experiment_id")
    ...
    filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

```python
def read_session_raw(session, load_events=True):
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
        trials = read_interval_table(trial_group, ["go", "catch", "aborted", "auto_rewarded",
            "hit", "miss", "false_alarm", "correct_reject", "change_time", "start_time", "stop_time"])
        ...
        stim_group = choose_task_presentation_group(h5f)
        ophys_timestamps = np.asarray(h5f["processing"]["ophys"]["event_detection"]["timestamps"], ...)
        events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
        pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], ...)
```

iii. From CONVERSION_NOTES Step 5 decision 9: *"Load NWB content with `h5py` rather than the SDK
session object … the current environment's `pynwb/hdmf` stack cannot instantiate these NWB 2.6.0
files through `BehaviorOphysExperiment.from_nwb_path` because of an `external_resources`
abstract-method mismatch. Direct HDF5 reads will therefore mirror the SDK field definitions
explicitly."* Step 1 of the notes documents the SDK functions whose field semantics were mirrored.
Passive sessions were excluded (Step 5 decision 1) because *"the decoder task requires trial outcome
… Passive sessions also produce degenerate outcomes (sample passive file: all go trials are misses
and all catch trials are correct rejects)."* The 3 eye-tracking-less sessions were excluded in Step
10 after the first full run crashed on them.

## 1-b. How are the data split into subjects?

i. Subjects are the `mouse_id` column of `ophys_experiment_table.csv`, stored as strings, registered
in first-encountered order (which, because the table is sorted by `ophys_experiment_id`, is roughly
chronological). `subject_idx` is the index of the mouse for each converted session. 38 subjects
result.

ii.
```python
mouse_id=str(int(row.mouse_id)),
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
...
data["subject_idx"].append(subject_idx)
```

iii. Step 5 variable-mapping table: *"NWB `general/subject/subject_id` / metadata `mouse_id` →
`subjects`, `subject_idx`; convert to global subject list and per-session index … Use mouse
identifier strings."* `mouse_id` is the SDK's canonical animal identifier.

## 1-c. How are the data split into sessions?

i. **One NWB `ophys_experiment` file = one decoder "session".** The AI does not group experiments by
`ophys_session_id`, so a multi-plane (Multiscope) behavioral session appears as up to ~8 separate
converted sessions that share identical behavior but contain different neurons. Sessions are ordered
by `ophys_experiment_id`, not by acquisition date. Passive experiments are dropped, so the converted
set is 199 sessions (168 single-plane `VisualBehavior` active minus 3 without eye tracking, plus 34
`VisualBehaviorMultiscope` active experiments coming from only ~6 distinct behavioral sessions).

ii.
```python
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    data["neural"].append(session_neural)
```
```python
"session_ophys_experiment_ids": [int(s.ophys_experiment_id) for s in sessions],
```

iii. CONVERSION_NOTES Step 4 discrepancy table: *"SDK distinguishes `behavior_session`,
`ophys_session`, and `ophys_experiment`; one NWB file corresponds to one `ophys_experiment` … Local
data are 284 experiment NWBs but only 247 unique behavior/ophys sessions … **Resolution:** Treat each
NWB experiment file as one decoder session because neural traces are experiment-specific; preserve
subject/session metadata so multiple experiments from one behavior session remain linked through
subject/session fields."* In practice only the experiment-id list is stored, not `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if
`(go or catch) and not aborted and not auto_rewarded`. Each kept trial spans the full trial window
`start_time → stop_time`, resampled onto a uniform 30 Hz grid anchored at `start_time`. Trial lengths
are therefore variable (211–377 bins; mean ≈ 257 bins ≈ 8.5 s). 51,075 trials result.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
```
```python
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. Step 5 decision 4: *"Segment trials from `start_time` to `stop_time`: Trial boundaries will come
directly from the NWB `intervals/trials` table, after filtering to keep only `(go or catch) and not
aborted and not auto_rewarded`."* Step 4 notes the SDK's `Trial._get_trial_data` taxonomy and the
instruction to include Go and Catch and exclude Aborted and Auto-rewarded.

## 1-e. How are trials filtered based on quality controls?

i. Quality controls applied:
- Trial-level: aborted and auto-rewarded trials dropped; only `go`/`catch` kept.
- Session-level: passive sessions dropped; sessions with no eye-tracking group dropped; sessions with
  `< 2` kept trials dropped (checked both before and after conversion).
- No further trial-level rejection: no `change_time` validity requirement, no clipping of trials that
  run past the last ophys frame (`np.interp`/`interpolate_matrix` clamp to the endpoint value
  instead), and no exclusion of "all-zero neural" trials (2,467 / 51,075 = 4.8 % of trials, which the
  verifier flags as warnings).

ii.
```python
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session {session.ophys_experiment_id} because it has ...")
    continue
...
if len(session_neural) < 2:
    print(f"[pass2] skipping session {session.ophys_experiment_id} after conversion ...")
    continue
```
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye
```

iii. Step 5 decision 8: *"Require at least two valid trials per session after filtering. The
active-session subset already satisfies this … (202 active sessions; minimum 39 valid trials after
filtering), so no expected extra session loss from this rule."* Step 10 edge-case review:
*"Missing eye tracking: Found 3 active sessions with no `acquisition/EyeTracking` group. Fix: exclude
… before both conversion passes."* On the all-zero trials, Step 10 concludes: *"the warnings reflect
genuine sparsity of the released calcium-event signal, not conversion corruption"* (verified against
raw NWB with `np.allclose`).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` (precomputed/detected calcium **events**, shape
`(T, n_rois)`) with `processing/ophys/event_detection/timestamps` as the ophys timebase. dF/F
(`processing/ophys/dff`) is present in every file but is **not** used.

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

iii. Step 5 decision 2: *"Use `events` instead of `dff_traces`: The whitepaper explains dF/F
generation, but the strategy paper explicitly states its neural analyses used detected calcium
events. Events are already present in the NWB files and best match the reference analyses."* Step 4
records the same resolution: *"Use precomputed `events` as the neural signal for conversion because
that best matches the analysis paper and avoids diverging from reference processing."*

## 2-b. How is the `neural` data processed?

i. Minimal processing: the `(T, n_rois)` event matrix is linearly interpolated (per neuron,
vectorised over neurons) from the native ophys timestamps onto the per-trial 30 Hz grid, transposed
to `(n_neurons, n_timepoints)`, and cast to `float32`. No normalisation, smoothing, z-scoring or
baseline subtraction is applied. Planes from the same behavioral session are **not** merged (see
1-c). Outside the native timestamp range the interpolation clamps to the nearest sample.

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
...
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. Step 3/Step 5: *"The strategy paper states its neural analyses use detected calcium events
rather than raw dF/F, and for event-triggered analyses it linearly interpolates onto common 30 Hz
timestamps relative to behavioral events."* The AI therefore treats "interpolate events onto a common
30 Hz grid" as the reference-matched processing and adds nothing else.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron curation beyond what the released NWB already contains. The code reads
`cell_specimen_table/valid_roi` and, only if the trace width equals the cell-table length and the
valid count differs from the trace width, subsets the traces to valid ROIs. In the released files
every ROI is already `valid_roi == True`, so this branch never fires and the filter is a no-op. All
29,168 ROIs across the 199 sessions are kept (4–666 per session).

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

iii. Step 4: *"SDK defaults to `exclude_invalid_rois=True` and filters to `valid_roi` … **Resolution:**
Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc neuron filtering
beyond reference QC/filtering already reflected in the files."* Step 3 notes the whitepaper's ROI
curation (union/duplicate/edge ROIs, ~1 % loss) is already applied upstream in the released data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. The 30 Hz grid begins exactly at the trials-table `start_time`
(`off_start = 0.0`) and runs in fixed `1/30 s` steps until `stop_time` (`off_end = None`, since trial
length varies). Neural, running, pupil, image identity and image change are all evaluated on that
same `grid` array, so every stream shares one absolute-time axis; no per-stream offsets or monitor
delay corrections are applied (consistent with the SDK, which builds running speed with monitor
delay 0).

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T...
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

iii. Step 5 decision 4 and Step 10: *"all trial streams are aligned in absolute experiment time and
resampled to a common 30 Hz grid from raw trial `start_time` / `stop_time`."* Step 10 sanity checks
reconstructed session `775614751` trial 0 and session `939327156` trial 10 directly from raw NWB and
report `np.allclose == True` for the neural matrix and every output row.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Time bin size = **33.33 ms (30 Hz)**, uniform for every trial and session
(`metadata['time_bin_size'] = 33.333…`). This is a **resampling**, not a rebinning/averaging: every
stream is linearly interpolated onto the 30 Hz grid. Native ophys rates are ~30.95 Hz for
single-plane experiments (slight downsampling) and ~11 Hz for the Multiscope experiments (3× upsampling
by interpolation).

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
"resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
```

iii. Step 5 decision 3: *"Use a common 30 Hz time base for all sessions: The target format requires
one shared bin size across sessions, while local data mix ~31 Hz single-plane and ~11 Hz multi-plane
ophys sampling. The strategy paper linearly interpolates calcium event responses onto common 30 Hz
timestamps for neural and running analyses, and behavior/eye tracking are naturally 30 Hz, so 30 Hz
is the most defensible common grid."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **active image stimulus-presentations interval table** (here
`intervals/Natural_Images_Lum_Matched_set_training_2017_presentations`), using its `start_time`,
`stop_time`, `image_name` and `omitted` columns — not from the trials table's
`initial_image_name`/`change_image_name`. The presentation group is auto-selected by preferring a
group that has `active` and `image_name` columns, a `change_detection` stimulus block, and the most
active flashes.

ii.
```python
def choose_task_presentation_group(h5f):
    for name in intervals.keys():
        if not name.endswith("_presentations") or name == "trials": continue
        grp = intervals[name]
        if "active" not in grp or "image_name" not in grp: continue
        ...
        score = (has_change_detection, n_active)
```
```python
stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name",
    "is_change", "omitted", "trials_id", "active", "flashes_since_change"])
```

iii. Step 5 mapping table: *"stimulus-presentation `image_name`, `start_time`, `stop_time`,
`omitted`, active task block only → `output[image_identity]`"*, referencing SDK
`BehaviorSession.stimulus_presentations` / `get_stimulus_presentations`. Step 1 notes
*"`stimulus_presentations` contains more than just the active task block in newer SDK releases; for
VBO change-detection analyses, the active block is identified via `stimulus_block_name` containing
`change_detection`."*

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A piecewise-constant categorical time series on the 30 Hz grid. For each grid time the code finds
the last flash whose `start_time <= t`; if `t < stop_time` of that flash and the flash is not
omitted, the label is that flash's image code, otherwise the label is an explicit **`gray`** class
(code 0). The vocabulary is global across sessions: `["gray"] + sorted(all image names)` = 17
categories. Resulting distribution: `gray` 0.6695, each of the 16 images 0.0189–0.0224.

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
        for i, (name, is_omitted) in enumerate(zip(names, omit)):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        codes[np.flatnonzero(valid)[in_interval]] = target
    return codes
```

iii. Step 5 decision 6: *"Use `gray` as an explicit image-identity class: Because the task includes
500 ms gray periods and omissions extend gray instead of showing an image, a `gray` category is
needed for a complete time-varying identity signal."*

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same `grid` array used for the neural interpolation, so index `t`
of the image-identity row corresponds to the same absolute time as column `t` of the neural matrix.
Flash boundaries are resolved with `np.searchsorted` on absolute presentation `start_time`/`stop_time`.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T...
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 10 output sanity check: for sessions `775614751` (trial 0) and `939327156` (trial 10) the
converted `image_identity` row *"matched independently reconstructed raw-data equivalents exactly"*
(`np.allclose == True`). Step 7 plot review: *"Stimulus identity traces show the expected alternation
of flashed images and gray intervals … No visual sign of cross-stream temporal misalignment."*

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the same active stimulus-presentations table: the `is_change` boolean column together with
`omitted`, `start_time`, `stop_time`. Not from the trials-table `change_time`. In the released data
`is_change` is True only for genuine image changes (sham/catch changes carry `is_sham_change`
instead), so catch trials contribute no positive bins.

ii.
```python
stim["is_change"] = stim["is_change"].astype(bool)
stim["omitted"] = stim["omitted"].astype(bool)
```

iii. Step 5 mapping table: *"stimulus-presentation `is_change`, `start_time`, `stop_time` →
`output[image_change]` … Marks the post-change flashed image itself rather than a one-bin impulse;
catch trials remain 0."* Step 10 "Issues Found and Resolved": *"The initial `image_change` target was
too sparse because it used a single-bin impulse at `change_time`. Resolution: relabeled
`image_change` as the changed-image presentation window using raw stimulus `is_change` intervals;
sample validation accuracy moved from below chance to above chance."*

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series that is 1 for all 30 Hz bins falling inside the 250 ms presentation window of
a non-omitted change flash, and 0 everywhere else (including the following gray period). Implemented
fully vectorised with the same last-flash `searchsorted` lookup as image identity. Resulting
distribution over all bins: `[0.9741 no_change, 0.0259 change]` (~7–8 positive bins per go trial).

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

iii. Step 6: *"`image_change` is constructed from the stimulus table's `is_change` presentation
interval instead of a single-bin impulse at `change_time`, which yields a less degenerate decoder
target while staying aligned to the task structure."* Step 12 confirms the class is *"sparse but not
degenerate"* (2.59 % of bins) and verified 3 positive trials against raw data.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the variable is natively binary. `output_values[1] =
["no_change", "change"]`, values 0/1.

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

iii. Implicit in Step 5's mapping: *"Binary time-varying label: 1 during the changed-image
presentation window, else 0."* The instruction itself specifies image change as a binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identical alignment to image identity: computed on the same `grid` array as the neural matrix, so
row index `t` corresponds to neural column `t`.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
output_trial = np.vstack([image_codes.astype(np.int64), image_change, running_bins,
                          pupil_bins, trial_outcome])
```

iii. Step 12: *"Raw-value checks on 3 positive trials from `ophys_experiment_id 775614751`: trial 0:
8 positive bins, `np.allclose == True`; trial 1: 8 positive bins …"*, plus the Step 7 processing plots
showing change impulses coincident with the identity transitions.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` and `processing/running/speed/timestamps` — the SDK's default
filtered running speed (10 Hz low-pass Butterworth, monitor delay 0), not `speed_unfiltered` and not
`dx`.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. Step 5 mapping table: *"running speed timeseries → `output[running_speed_bin]` … Uses filtered
running speed, matching SDK default"*, referencing `BehaviorSession.running_speed` /
`RunningSpeed.from_stimulus_file`. Step 1: *"`BehaviorSession.running_speed` is sampled on timestamps
with monitor delay `0.0`, so running is aligned to sync/stimulus time without display-lag
compensation."*

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`, which clamps to the end values outside the sampled range) from
the native running timestamps onto each trial's 30 Hz grid, then discretisation into 5 global
quintile bins. The quintile edges are computed in pass 1 over the concatenation of the interpolated
running values of *every kept trial in every included session*, so the bins are global rather than
per-session.

ii.
```python
def interpolate_vector(source_t, source_values, query_t):
    return np.interp(query_t, source_t, source_values).astype(np.float32)
...
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
...
running_all = np.concatenate(running_values).astype(np.float32)
running_edges = robust_quintile_edges(running_all)
...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 5 decision 7: *"Discretize continuous outputs globally, not per session: Running-speed and
pupil-diameter bin edges will be computed from all valid included timepoints across the converted
dataset so class definitions are shared across sessions."* Verified in Step 9/10 to give exactly
uniform class fractions `[0.2, 0.2, 0.2, 0.2, 0.2]`.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four inner edges at the 20th/40th/60th/80th percentiles of the pooled interpolated values
(`np.nanpercentile`), with a monotonicity guard that nudges any non-increasing edge up by 1e-6 (needed
because running speed is ≈0 for a large fraction of the time, which can make the 20th and 40th
percentiles identical). `np.digitize(..., right=False)` then yields codes 0–4, labelled
`bin_0 … bin_4`.

ii.
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles

def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The instructions require *"discretized into five equal percentile bins"*; the AI records the
resulting edges in `metadata['running_speed_bin_edges']` so the mapping is auditable, and Step 10
re-derives the bins from raw data with `np.allclose`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated at the same `grid` times as the neural matrix, so bin-for-bin aligned with no offset.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 10: *"all trial streams are aligned in absolute experiment time and resampled to a common
30 Hz grid from raw trial `start_time` / `stop_time`"*; the running row of two trials was
reconstructed from raw NWB and matched with `np.allclose == True`. Step 7 plots show the discretised
bins tracking the continuous trace *"without obvious temporal offsets"*.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height` (the fitted pupil-ellipse axes),
with `acquisition/EyeTracking/eye_tracking/timestamps` as the timebase. Diameter is defined as the
element-wise **maximum of width and height**. Blink frames are already stored as `NaN` in these
arrays (they correspond exactly to `likely_blink == True`), and the code discards non-finite samples
before interpolating.

ii.
```python
pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_timestamps = np.asarray(h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], ...)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. Step 5 mapping table: *"eye-tracking pupil width/height + blink mask → `output[pupil_diameter_bin]`;
Compute pupil diameter as `max(width, height)`; use blink-masked values … Small blink-related gaps are
filled by interpolation after applying reference invalid-frame masking"*, referencing
`EyeTrackingTable` / `filter_on_blinks`. Step 1: *"blink frames are set to `NaN` for derived
pupil/eye area signals."*

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite (blink) samples are dropped; the remaining samples are linearly interpolated onto the
30 Hz trial grid (which bridges blink gaps and clamps at the record boundaries); the result is then
discretised into 5 global quintile bins computed in pass 1 over all kept trials of all included
sessions. A degenerate case (0 or 1 valid sample) raises or fills a constant.

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
pupil_all = np.concatenate(pupil_values).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same rationale as running speed (Step 5 decision 7, global quintiles). Blink handling follows
Step 1's reading of `EyeTrackingTable` / `filter_on_blinks`; sessions whose eye-tracking group is
absent entirely are excluded up front rather than imputed (Step 10).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: 20/40/60/80th percentiles of the pooled interpolated
diameters with the monotonicity guard, then `np.digitize` → codes 0–4 labelled `bin_0 … bin_4`. Edges
saved to `metadata['pupil_diameter_bin_edges']`. Realised fractions are exactly `[0.2]*5`.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
"pupil_diameter_bin_edges": pupil_edges.astype(float).tolist(),
```

iii. Required by the instructions (*"discretized into five equal percentile bins"*); Step 9 reports
the uniform `[0.2, 0.2, 0.2, 0.2, 0.2]` distribution as the consistency check that the quintiles were
computed correctly.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Evaluated on the same `grid` array as the neural matrix, so bin-for-bin aligned. Blink gaps are
bridged by interpolation rather than shifting the series.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T...
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Step 10 output sanity check 1/2: converted `pupil_diameter_bin` for the two spot-checked trials
*"matched independently reconstructed raw-data equivalents exactly"*, using the raw blink-masked
width/height and the saved global edges.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`,
`correct_reject`, checked in that order.

ii.
```python
def trial_outcome_code(trials, idx, mapping) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. Step 5 mapping table: *"trial outcome flags (`hit`, `miss`, `false_alarm`, `correct_reject`) →
`output[trial_outcome]`"*, referencing `Trial._get_trial_data`. Step 1 documents that this is the
SDK's trial-outcome taxonomy for go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to integer codes 0–3 with the fixed ordering `["hit", "miss", "false_alarm",
"correct_reject"]` and **broadcast constant across all time bins of the trial**, so the static
per-trial variable is stored as a time-varying row. If a kept trial matched none of the four flags
the code raises (no fallback class). Realised distribution:
`[0.303 hit, 0.571 miss, 0.017 false_alarm, 0.108 correct_reject]`.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Step 5 decision 5: *"Represent all outputs as time-varying: To satisfy the decoder format and
simplify training, even static trial outcome will be repeated across all bins in a trial."* The
target-format spec likewise says *"If at all possible, make it time-varying."*

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling that is present:
- **Missing eye tracking (whole session)**: the session is excluded before both passes
  (`has_required_eye_tracking`), affecting 3 active sessions.
- **Blink / NaN pupil samples**: dropped, then bridged by interpolation; a session with 1 valid
  sample is filled with a constant; 0 valid samples raises.
- **Degenerate quintile edges**: non-increasing percentiles are nudged apart by 1e-6 so
  `np.digitize` still produces 5 distinct bins (matters for running speed, which is ≈0 much of the
  time).
- **Out-of-range query times**: `np.interp` and `interpolate_matrix` clamp to the first/last sample
  rather than producing NaN, so a trial whose `stop_time` runs past the last ophys/running/pupil
  sample gets a constant hold instead of being clipped or dropped.
- **Sessions with too few trials**: skipped if `< 2` valid trials, checked both before and after
  conversion.
- **Stimulus-table ambiguity**: `choose_task_presentation_group` scores candidate interval tables and
  picks the change-detection/active one instead of hard-coding a name.
- **All-zero neural trials** (2,467 trials, 4.8 %): kept, after verifying against raw NWB that the
  zeros are genuine.

Handling that is **absent**: there is no `try/except` around session processing, so an unexpected
file aborts the entire run (this is exactly what happened on the first full-conversion attempt);
`trial_outcome_code` raises rather than falling back to an "other" class; a session registered in
`subjects`/`brain_regions` but then skipped by the post-conversion `< 2 trials` check would leave an
orphan entry in `subjects`.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
excluded = len(sessions) - len(filtered_sessions)
if excluded:
    print(f"[setup] excluded {excluded} active sessions missing eye-tracking pupil data")
```
```python
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
```
```python
for i in range(1, len(percentiles)):
    if percentiles[i] <= percentiles[i - 1]:
        percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. Step 10 "Issues Found and Resolved": *"Missing pupil data in 3 active sessions caused full-conversion
failure on the first Step 9 attempt. Resolution: exclude those sessions before both passes so global
bin edges and converted sessions are computed on the same valid subset."* And: *"the remaining
verify-only warnings are genuine properties of the sparse event signal and cannot be removed without
changing the referenced neural representation."*

## 9-a. What are the most time-consuming steps of the code?

i. Measured total runtime was 356 s for 199 sessions (~1.8 s/session). The dominant costs are:
1. **HDF5 I/O**: pass 2 reads the full `(T, n_rois)` event array into memory for every session
   (up to 140k × 666 floats), plus the running/pupil/stimulus/trials tables.
2. **Per-trial neural interpolation** (`interpolate_matrix`), which allocates two `(n_bins, n_rois)`
   gather arrays per trial — the AI's own Step 7 estimate attributed ~18 min of the projected runtime
   to "neuron-bin work".
3. **Reading every file three times**: once in `has_required_eye_tracking`, once in pass 1, once in
   pass 2.
4. **Pickling the 8.1 GB output**.

ii.
```python
t0 = time.perf_counter()
raw = read_session_raw(session)
...
elapsed = time.perf_counter() - t0
print(f"[pass2] {sess_num:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{len(session_neural)} trials, {raw['n_neurons']} neurons, {total_bins} bins in {elapsed:.2f}s")
```

iii. Step 6: *"Neural interpolation is still the dominant expected cost because every kept trial needs
event traces resampled onto the common grid."* Step 7 tabulates 0.62 s/session for pass 1 and
1.35 s/session for pass 2 and extrapolates to ~18.7 min for the full subset (the actual run was much
faster, 5.9 min).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Candidates:
1. **The per-trial loop in `convert_sessions`/`collect_global_statistics`** — running and pupil could
   be interpolated once onto a whole-session 30 Hz grid and then sliced per trial, instead of calling
   `np.interp`/`searchsorted` 51,075 times. The same applies to the neural interpolation, though
   memory would need care.
2. **The Python `for` loop inside `stimulus_identity_codes`** — it iterates over every in-interval
   *grid point* (≈250 per trial × 51,075 trials ≈ 5–13 M Python iterations) to look up an image code
   in a dict. Precomputing an integer code array per stimulus presentation once per session would
   make this a single fancy-index (`codes[assign] = presentation_codes[sub_idx]`), exactly as the
   already-vectorised `stimulus_change_codes` does.
3. **`decode_strings`** builds a Python list comprehension over every element of every string column
   of the stimulus table (tens of thousands of flashes) on each of the two passes.

ii.
```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Step 6 lists *"Trial-level interpolation is vectorized over neurons within each trial"* as the
speed-up that was implemented; the AI did not identify the remaining per-trial and per-grid-point
loops as further vectorisation opportunities.

## 9-c. What processing does the code repeat multiple times?

i. Substantial duplication between the two passes:
- Every NWB file is **opened three times**: `has_required_eye_tracking`, pass 1
  (`read_session_raw(load_events=False)`), pass 2 (`read_session_raw(load_events=True)`).
- The trials table, the stimulus presentation table (including `decode_strings` on `image_name`), the
  ophys timestamps, the running stream and the pupil streams are **parsed twice**.
- The per-trial `session_grid`, running interpolation and pupil interpolation are computed **twice
  for every one of the 51,075 trials** — once in pass 1 purely to accumulate values for the quintile
  edges, then discarded, and again in pass 2. Caching the pass-1 results (or a subsample of them)
  would remove an entire pass.
- `session_native_dt` and the valid-trial count are recomputed per session although the values are
  only printed.

ii.
```python
# pass 1
raw = read_session_raw(session, load_events=False)
...
grid = session_grid(start, stop)
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# pass 2 — identical work repeated
raw = read_session_raw(session)
...
grid = session_grid(start, stop)
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Step 6 frames the two-pass design as a *memory* optimisation: *"Two-pass design avoids storing
all neural arrays while computing global bin edges."* That is true for the neural data (which pass 1
skips via `load_events=False`), but the notes do not acknowledge that the behavioural interpolation
and all table parsing are done twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Discarded / unused work:
- `stim["trials_id"]` and `stim["flashes_since_change"]` are read from HDF5 and cast to `int64` for
  every session in both passes but are never used.
- `stim["active"]` is read in `read_session_raw` and never used (the active-block selection is done
  separately inside `choose_task_presentation_group`, which re-reads its own copies).
- `trials["change_time"]` is read and cast but never used (the change label comes from the stimulus
  table instead).
- `native_dt_by_session` and `valid_trial_counts` are accumulated across all sessions only to print a
  min/max line.
- `RNG = np.random.default_rng(0)` is created and never used; `interpolate_matrix`'s `n_src == 1`
  branch is unreachable for real sessions.
- `pupil_height` is read in full even though only `np.maximum(width, height)` is retained.
- Most significantly, **resampling the ~11 Hz Multiscope sessions up to 30 Hz** triples their stored
  size without adding information, and resampling ~30.95 Hz single-plane data to 30 Hz is a
  near-identity operation that still costs a full interpolation of every neuron × every bin. The
  resulting pickle is 8.1 GB.

ii.
```python
stim["trials_id"] = stim["trials_id"].astype(np.int64)
stim["active"] = stim["active"].astype(bool)
stim["flashes_since_change"] = stim["flashes_since_change"].astype(np.int64)
```
```python
RNG = np.random.default_rng(0)
...
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
```

iii. The notes do not discuss any of these; Step 6's "Code inefficiencies identified" section only
mentions the `pynwb` incompatibility and the cost of neural interpolation. The 30 Hz grid is
justified in Step 5 decision 3 as necessary for a single shared `time_bin_size` across the mixed
11 Hz / 31 Hz session pool the AI chose to include.
