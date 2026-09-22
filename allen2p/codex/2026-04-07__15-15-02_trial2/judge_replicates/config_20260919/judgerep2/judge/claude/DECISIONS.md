# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK object API. It found that the environment's `pynwb`/`hdmf` stack could not instantiate the release's NWB 2.6.0 files through `BehaviorOphysExperiment.from_nwb_path` (an `external_resources` abstract-method mismatch), so it read the NWB/HDF5 files directly with `h5py`, using the SDK source only as a schema/semantics guide. The set of recordings is discovered from the release metadata CSV `project_metadata/ophys_experiment_table.csv`, intersected with the NWB files actually present on disk (284 files), then filtered to `passive == False` (202 files) and to files that contain an `acquisition/EyeTracking` group with both `pupil_tracking` and `eye_tracking` (199 files). Note that **no `project_code` filter is applied**: the local subset contains 239 `VisualBehavior` experiments plus 45 `VisualBehaviorMultiscope` experiments, and the AI keeps the active ones from both (168 VB + 34 Multiscope = 202 before the eye-tracking filter). Within each file, everything is read from fixed HDF5 paths: `intervals/trials`, the active stimulus-presentations interval table, `processing/ophys/event_detection/{data,timestamps}`, `processing/running/speed/{data,timestamps}`, and `acquisition/EyeTracking/*`. The whole dataset is traversed twice (pass 1 for global statistics without the neural array, pass 2 for the actual conversion).

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
    sessions = [SessionInfo(ophys_experiment_id=int(row.ophys_experiment_id),
                            path=file_map[int(row.ophys_experiment_id)],
                            mouse_id=str(int(row.mouse_id)),
                            targeted_structure=str(row.targeted_structure),
                            session_type=str(row.session_type),
                            project_code=str(row.project_code),
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
        trials = read_interval_table(trial_group, ["go", "catch", "aborted", "auto_rewarded",
            "hit", "miss", "false_alarm", "correct_reject", "change_time", "start_time", "stop_time"])
        stim_group = choose_task_presentation_group(h5f)
        stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name",
            "is_change", "omitted", "trials_id", "active", "flashes_since_change"])
        ophys_timestamps = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
        if load_events:
            events = np.asarray(
                h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
        ...
```

iii. From CONVERSION_NOTES Step 5, Key Decision 9: "Load NWB content with `h5py` rather than the SDK session object — The reference SDK is still the guide for field semantics and processing, but the current environment's `pynwb/hdmf` stack cannot instantiate these NWB 2.6.0 files through `BehaviorOphysExperiment.from_nwb_path` ... Direct HDF5 reads will therefore mirror the SDK field definitions explicitly." The AI backed this with Step 10 sanity checks that re-read the raw NWB independently and compared with `np.allclose` (all passed). In Step 4 it noted the local files are "a curated subset of the release", so statistics are validated against the local subset rather than the published totals. It did not discuss restricting to `project_code == 'VisualBehavior'`; it explicitly kept `VISp` and `VISl` because "Local NWBs contain only `VISp` and `VISl` ... Paper describes V1 and LM datasets."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` values from `ophys_experiment_table.csv`, stored as strings and registered lazily in the order in which their first surviving experiment is converted. `subject_idx` holds one index per output session. The result is 38 subjects (37 from `VisualBehavior` plus 1 extra mouse that only appears in the `VisualBehaviorMultiscope` files).

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

iii. Step 5 variable mapping: "NWB `general/subject/subject_id` / metadata `mouse_id` → `subjects`, `subject_idx`; convert to global subject list and per-session index; use mouse identifier strings." The count (38 mice) was cross-checked in Step 2 against the local NWB subset and reported again in Step 9/10 as matching the raw data exactly.

## 1-c. How are the data split into sessions?

i. **Each NWB file (i.e. each `ophys_experiment_id`, one imaging plane) is treated as one decoder "session."** The AI explicitly noticed that the local subset contains 284 experiment files but only 247 unique `ophys_session_id`s and chose not to merge planes recorded simultaneously within one `ophys_session_id`. In practice this is 1:1 for the 239 single-plane `VisualBehavior` experiments, but the 34 active `VisualBehaviorMultiscope` experiments come from only 8 behavioural sessions, so those 8 sessions are emitted as 34 separate "sessions" that each repeat the identical trial structure and identical behavioural outputs (visible in the verification log as trial counts repeated 7, 7, 7, 5, 5, 3 times). Sessions are ordered by `ophys_experiment_id`. Output: 199 sessions.

ii.
```python
exp_table = exp_table.sort_values("ophys_experiment_id")
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    data["neural"].append(session_neural)
    data["input"].append(session_input)
    data["output"].append(session_output)
    data["subject_idx"].append(subject_idx)
    data["brain_region_idx"].append(np.full(raw["n_neurons"], region_idx, dtype=np.int64))
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "SDK distinguishes `behavior_session`, `ophys_session`, and `ophys_experiment`; one NWB file corresponds to one `ophys_experiment` ... Local data are 284 experiment NWBs but only 247 unique behavior/ophys sessions ... **Resolution**: Treat each NWB experiment file as one decoder session because neural traces are experiment-specific; preserve subject/session metadata so multiple experiments from one behavior session remain linked through subject/session fields." (`metadata['session_ophys_experiment_ids']` is stored so the mapping is recoverable.)

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if `(go or catch) and not aborted and not auto_rewarded`. The trial window is the full `start_time` → `stop_time` interval of the trials table (variable length; ~211–377 bins at 30 Hz, mean 255), resampled onto a per-trial 30 Hz grid anchored at `start_time`. No fixed window around `change_time` is used, so each trial contains several pre-change flashes and the post-change response period.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. Step 5, Key Decision 4: "Segment trials from `start_time` to `stop_time` — Trial boundaries will come directly from the NWB `intervals/trials` table, after filtering to keep only `(go or catch) and not aborted and not auto_rewarded`." Step 1 notes cite `Trial._get_trial_data` / `_get_trial_timing` / `Trials._get_trial_bounds` as the SDK definitions of these fields and note that consecutive trial bounds are adjusted so there is no dead time between trials.

## 1-e. How are trials filtered based on quality controls?

i. Trial level: only `(go | catch) & ~aborted & ~auto_rewarded` are kept, matching the instruction to include Go and Catch and exclude Aborted and Auto-rewarded. (I verified on several raw NWB files that this mask is *identical* to the reference's `~aborted & ~auto_rewarded & change_time.notna()` mask.) Session level, the AI adds three filters the reference does not have: (a) passive sessions are dropped (82 of 284 files); (b) 3 active sessions with no eye-tracking group at all are dropped (`795953296`, `806456687`, `833631914`); (c) sessions with fewer than 2 valid trials are dropped (checked both before and after conversion; no session actually hit this). No trial-level rejection for missing behaviour or short windows exists, because the trial grid is always at least one bin and running/pupil are always interpolable in the retained sessions.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
exp_table = exp_table[~exp_table["passive"]].copy()
...
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session ... {int(raw['keep_mask'].sum())} valid trials"); continue
...
if len(session_neural) < 2:
    print(f"[pass2] skipping session ... {len(session_neural)} trials"); continue
```

iii. Step 4: "Trial inclusion for decoder should follow the common interpretation across sources: include `go` and `catch`, exclude `aborted` and `auto_rewarded`." Step 5, Key Decision 1: "Use only active sessions — ... the decoder task requires trial outcome and the strategy paper's behavioral analyses focus on active sessions. Passive sessions also produce degenerate outcomes (sample passive file: all go trials are misses and all catch trials are correct rejects)." Key Decision 8 covers the ≥2-trial rule. The eye-tracking exclusion came out of Step 10 edge-case review: "Missing pupil data in 3 active sessions caused full-conversion failure on the first Step 9 attempt. Resolution: exclude those sessions before both passes so global bin edges and converted sessions are computed on the same valid subset."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` (detected calcium events, shape `(T, n_rois)`) with `processing/ophys/event_detection/timestamps` as the ophys timebase — **not** `dff_traces`. ROI identity/validity is taken from `processing/ophys/image_segmentation/cell_specimen_table` (`valid_roi`).

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
if load_events:
    events = np.asarray(
        h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
...
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cell_table = len(cell_table["id"])
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
n_neurons = events.shape[1]
```

iii. Step 4 discrepancy table: "SDK exposes both `dff_traces` and `events`... Whitepaper defines dF/F processing; strategy paper states analyses used 'detected calcium events'. **Resolution**: Use precomputed `events` as the neural signal for conversion because that best matches the analysis paper and avoids diverging from reference processing." Step 5, Key Decision 2 repeats this, quoting the paper's Methods ("For all analysis of neural data we used the detected calcium events as described in Garrett et al."). Step 1 also records that dF/F does not need to be recomputed because both signals are precomputed in the NWB.

## 2-b. How is the `neural` data processed?

i. Three operations: (1) optional `valid_roi` selection of columns (a no-op on this release — I confirmed the stored event matrices already contain only valid ROIs); (2) linear interpolation of every neuron's event trace from the native ophys timestamps onto the trial's 30 Hz grid, vectorised across neurons; (3) transpose to `(n_neurons, n_timepoints)` and cast to `float32`. No normalisation, smoothing, z-scoring, deconvolution or baseline correction is applied. Total 29,168 neurons, mean 146.6/session (min 4, max 666).

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    """Linear interpolation for 2D source arrays of shape (T, N)."""
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

iii. Step 5 mapping table: "Use precomputed event traces; transpose to `(n_neurons, n_timepoints)` per trial after resampling/interpolation to common 30 Hz trial grid ... Neural signal will be calcium events, not dF/F." The interpolation itself is justified by the paper's Methods, which interpolate calcium events "onto a consistent set of 30hz timestamps relative to the triggering behavioral event." Step 10 verified two sessions/trials against an independent raw re-interpolation with `np.allclose == True`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional curation beyond what the Allen pipeline already applied. The code will subset to `valid_roi == True` if the stored trace width disagrees with the valid-ROI count, but on this release the released event traces already contain only valid ROIs, so nothing is ever dropped. No firing-rate, SNR, or activity thresholds are applied, and trials whose event matrix is entirely zero (2,467 of 51,075) are deliberately retained.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. Step 4: "SDK defaults to `exclude_invalid_rois=True` and filters to `valid_roi` ... Whitepaper describes exclusion of non-cell ROIs, duplicate/union ROIs, and additional problematic demixed traces. **Resolution**: Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc neuron filtering beyond reference QC/filtering already reflected in the files." On the all-zero trials, Step 10: "the warnings reflect genuine sparsity of the released calcium-event signal, not conversion corruption ... not fixable without changing the referenced neural representation" (verified by re-deriving one flagged trial from the raw NWB, `max_abs_diff == 0.0`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**: the per-trial grid begins exactly at the trials-table `start_time` and steps by 1/30 s until `stop_time`; the neural events, running speed, pupil, image identity, image change and trial outcome are all evaluated on that same grid in absolute session time, so all streams are aligned by construction. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (variable trial length). The change event sits mid-trial rather than at t=0.

ii.
```python
grid = session_grid(start, stop)                     # start = trials['start_time']
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont  = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont    = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
image_codes   = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
image_change  = stimulus_change_codes(raw["stimulus"], grid)
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Step 10 reference-code comparison: "Reference: synchronized timestamps across behavior/ophys streams; strategy paper interpolates activity to common 30 Hz timestamps. Conversion: all trial streams are aligned in absolute experiment time and resampled to a common 30 Hz grid from raw trial `start_time`/`stop_time`." Step 1 notes that the Allen sync system records all clocks on one 100 kHz DIO board, so the NWB timestamps of the different streams are directly comparable. The `--show-processing` plots were reviewed for cross-stream offsets ("No visual sign of cross-stream temporal misalignment").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. A single global bin size of **33.33 ms (30 Hz)** is used for every trial and session, and yes — every stream is resampled onto it by linear interpolation. Native rates in the local subset are heterogeneous: the AI's own log reports `native ophys dt range: 0.0323 … 0.0932` s, i.e. ~31 Hz for the single-plane `VisualBehavior` files and ~11 Hz for the `VisualBehaviorMultiscope` files. Single-plane data are therefore slightly *down*sampled and multiscope data are ~3× *up*sampled onto interpolated bins. No averaging/binning of events into bins is performed — the value in each bin is a linear interpolation between neighbouring event samples.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
"resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
```

iii. Step 5, Key Decision 3: "Use a common 30 Hz time base for all sessions — The target format requires one shared bin size across sessions, while local data mix ~31 Hz single-plane and ~11 Hz multi-plane ophys sampling. The strategy paper linearly interpolates calcium event responses onto common 30 Hz timestamps for neural and running analyses, and behavior/eye tracking are naturally 30 Hz, so 30 Hz is the most defensible common grid."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus-presentations interval table** (not the trials table): `start_time`, `stop_time`, `image_name`, `omitted`. The presentation table is chosen by `choose_task_presentation_group`, which prefers a `*_presentations` group that has `active` and `image_name` columns and mentions `change_detection` in `stimulus_block_name`. The global vocabulary is built in pass 1 from all non-omitted `image_name` values across all included sessions, giving 16 images; `"gray"` is prepended as code 0, so there are 17 categories.

ii.
```python
stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name",
        "is_change", "omitted", "trials_id", "active", "flashes_since_change"])
...
image_names.update(str(x) for x, omitted in zip(stim["image_name"], stim["omitted"])
                   if (not omitted) and str(x) not in ("", "None", "nan"))
...
image_values = ["gray"] + sorted(image_names)
```

iii. Step 5 mapping table: "stimulus-presentation `image_name`, `start_time`, `stop_time`, `omitted`, active task block only → `output[image_identity]` ... Global categories = `gray` + all unique image names in included sessions", referencing `BehaviorSession.stimulus_presentations` / `get_stimulus_presentations` in the SDK. Step 1 notes that "`stimulus_presentations` contains more than just the active task block in newer SDK releases; for VBO change-detection analyses, the active block is identified via `stimulus_block_name` containing `change_detection`."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 30 Hz bin, the code finds the presentation interval containing the bin time (`searchsorted` on presentation `start_time`, then a check that the bin is before that presentation's `stop_time`) and emits that presentation's image code. Bins that fall in the 500 ms inter-stimulus grey period, or inside an omitted flash, or outside any presentation, get the explicit `"gray"` code 0. Image names are mapped to integers through the global `image_to_code` dictionary. The resulting distribution is heavily dominated by grey: `gray` 0.669 and each of the 16 images ~0.019–0.022 of all bins.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    idx = np.searchsorted(stimulus["start_time"], query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = idx >= 0
    valid &= idx < len(starts)
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        names = np.asarray(image_names[sub_idx], dtype=object)
        omit = omitted[sub_idx]
        target = np.full(sub_idx.shape, image_to_code["gray"], dtype=np.int64)
        for i, (name, is_omitted) in enumerate(zip(names, omit)):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        codes[np.flatnonzero(valid)[in_interval]] = target
    return codes
```

iii. Step 5, Key Decision 6: "Use `gray` as an explicit image-identity class — Because the task includes 500 ms gray periods and omissions extend gray instead of showing an image, a `gray` category is needed for a complete time-varying identity signal." Step 5 mapping: "Piecewise-constant categorical signal on 30 Hz grid; use actual image name during image display; use `gray` during gray-screen or omission periods."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same per-trial 30 Hz `grid` array as the neural data, using absolute session times, so row *t* of `image_identity` and column *t* of the neural matrix refer to the same instant. No lag or shift is introduced (in particular, no monitor-delay compensation is added, consistent with the SDK's `running_speed`/stimulus timestamps).

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes  = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 10 output sanity checks 1 and 2 reconstructed `image_identity` for session `775614751` trial 0 and session `939327156` trial 10 directly from the raw stimulus intervals and confirmed `np.allclose == True`; the `--show-processing` plots overlay image identity, change, running, pupil and neural activity on the same trial-relative time axis and were reviewed for offsets.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The stimulus-presentations table columns `is_change`, `omitted`, `start_time` and `stop_time`. It is **not** derived from the trials-table `change_time`/`go` columns (an earlier version did use those and was replaced). Because `is_change` is False for the sham change of a catch trial, catch trials are all-zero.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]; stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]; omitted = stimulus["omitted"]
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    ...
    changed = is_change[sub_idx] & (~omitted[sub_idx])
    codes[np.flatnonzero(valid)[in_interval]] = changed.astype(np.int64)
    return codes
```

iii. Step 5 mapping: "stimulus-presentation `is_change`, `start_time`, `stop_time` → `output[image_change]`; Binary time-varying label: 1 during the changed-image presentation window, else 0 ... Marks the post-change flashed image itself rather than a one-bin impulse; catch trials remain 0." Step 1 records `compute_is_sham_change` as the SDK function that distinguishes catch-trial sham changes from real changes in the presentation table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Purely the interval lookup above: a bin is 1 if it falls inside a presentation whose `is_change` is True and which is not omitted, else 0. Since a flash lasts 250 ms, the label is 1 for ~7–8 consecutive 30 Hz bins per go trial, giving an overall positive rate of 2.59% of bins. No smoothing, dilation into the following grey period, or per-trial masking is applied.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Step 6: "`image_change` is constructed from the stimulus table's `is_change` presentation interval instead of a single-bin impulse at `change_time`, which yields a less degenerate decoder target while staying aligned to the task structure." Step 10 Issues: "The initial `image_change` target was too sparse because it used a single-bin impulse at `change_time`. Resolution: relabeled `image_change` as the changed-image presentation window using raw stimulus `is_change` intervals; sample validation accuracy moved from below chance to above chance."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is needed — the variable is natively binary, `{0: "no_change", 1: "change"}`, obtained from the boolean `is_change & ~omitted`. Verified distribution: `no_change` 0.974, `change` 0.026.

ii.
```python
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
...
"output_values": [..., ["no_change", "change"], ...]
```

iii. Step 5/Step 10: the only design question the AI documented here was the *width* of the positive window (single-bin impulse vs. full changed-image presentation), resolved in favour of the full 250 ms flash because the impulse target was degenerate for the decoder.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: evaluated on the identical trial `grid`, in absolute session time, and stacked as row 1 of the trial's output matrix, so it shares timepoints one-for-one with the neural matrix.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
output_trial = np.vstack([image_codes, image_change, running_bins, pupil_bins, trial_outcome])
```

iii. Step 12: "Raw-value checks on 3 positive trials from `ophys_experiment_id 775614751`: trial 0: 8 positive bins, `np.allclose == True`; trial 1: 8 positive bins ...; trial 2: 7 positive bins ...", plus "the saved processing plots from Step 7 show image identity, image change, running, pupil, and neural traces aligned on the same 30 Hz grid without visible offsets."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — i.e. the SDK's default *filtered* running speed (10 Hz low-pass Butterworth, monitor delay 0), not `speed_unfiltered` and not the raw `dx` encoder signal.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. Step 5 mapping: "running speed timeseries → `output[running_speed_bin]` ... Uses filtered running speed, matching SDK default", citing `BehaviorSession.running_speed` / `RunningSpeed.from_stimulus_file`. Step 1 notes "`BehaviorSession.running_speed` is sampled on timestamps with monitor delay `0.0`, so running is aligned to sync/stimulus time without display-lag compensation", and Step 3 records the whitepaper's filtering chain (unwrap, despike, z≥10 artifact removal, 10 Hz low-pass).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`) of the filtered speed from its native ~60 Hz timestamps onto the trial's 30 Hz grid (constant extrapolation at the edges, since `np.interp` clamps), then discretisation into 5 global quintile bins. The continuous value itself is not stored. Signed negative speeds are kept (bin 0 spans backwards running).

ii.
```python
def interpolate_vector(source_t, source_values, query_t):
    return np.interp(query_t, source_t, source_values).astype(np.float32)
...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 5 mapping: "Interpolate running speed onto 30 Hz trial grid; discretize with global quintile bins across included active-session timepoints." Step 10 output sanity checks re-derived `running_speed_bin` for two trials from the raw NWB with `np.allclose == True`.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-population quintile bins. In pass 1 the interpolated running speed is accumulated over *exactly the trial windows that will be kept* across all included sessions, the 20/40/60/80th percentiles are taken as bin edges (with a 1e-6 nudge to enforce strict monotonicity against ties), and `np.digitize(..., right=False)` maps each bin to 0–4. Edges are global (identical for all sessions) and stored in metadata: `[-0.0041, 0.303, 15.78, 33.44]` cm/s. Resulting distribution is exactly uniform: 0.200 per bin.

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
...
running_all = np.concatenate(running_values).astype(np.float32)
running_edges = robust_quintile_edges(running_all)
...
"running_speed_bin_edges": running_edges.astype(float).tolist(),
```

iii. Step 5, Key Decision 7: "Discretize continuous outputs globally, not per session — Running-speed and pupil-diameter bin edges will be computed from all valid included timepoints across the converted dataset so class definitions are shared across sessions." This follows the Decoder Task instruction "discretized into five equal percentile bins". The `robust_quintile_edges` tie-nudge is an explicit guard against degenerate (duplicate) percentile edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly onto the same per-trial 30 Hz `grid` used for the neural matrix, in absolute session time; row 2 of the output matrix. No lead/lag correction is applied.

ii.
```python
grid = session_grid(start, stop)
neural_trial  = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont  = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins  = digitize_with_edges(running_cont, running_edges)
```

iii. Step 10: "all trial streams are aligned in absolute experiment time and resampled to a common 30 Hz grid"; the running panel of `processing_<id>.png` overlays the continuous trace, the discretised bin trace and the bin edges on the trial-relative axis, and was reviewed: "discretized bins tracking the continuous signals without obvious temporal offsets."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height` (the pupil ellipse-fit semi-axes), timestamped by `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as the elementwise **maximum of width and height**. The `likely_blink` array is not read explicitly — but I verified on the raw files that blink frames are already stored as `NaN` in `pupil_tracking/width`/`height` (100% of `likely_blink` frames are NaN, 9.2% of frames in the file I checked), and the AI's interpolation drops non-finite samples, so blinks are excluded exactly as the SDK's `filter_on_blinks` would.

ii.
```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. Step 5 mapping: "eye-tracking pupil width/height + blink mask → `output[pupil_diameter_bin]`; Compute pupil diameter as `max(width, height)`; use blink-masked values; interpolate across valid timestamps onto 30 Hz trial grid ... Small blink-related gaps are filled by interpolation after applying reference invalid-frame masking", citing `EyeTrackingTable` / `filter_on_blinks`. Step 1 records: "blink frames are set to `NaN` for derived pupil/eye area signals."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite (blink/untracked) samples are removed, the remaining samples are linearly interpolated onto the trial's 30 Hz grid (so blink gaps are bridged; constant extrapolation at the session edges), and the result is discretised into the 5 global quintile bins. Degenerate cases are guarded: zero valid samples raises (and those sessions were pre-excluded), one valid sample yields a constant trace.

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
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. As 6-a: blink-masked values, gaps filled by interpolation, then global quintile discretisation. Step 10 output sanity checks re-derived `pupil_diameter_bin` from the raw blink-masked width/height for two trials with `np.allclose == True`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: global 20/40/60/80th percentiles of the interpolated pupil diameter accumulated over all kept trial windows in pass 1, `np.digitize` into bins 0–4, edges stored in metadata (`[36.91, 41.90, 46.44, 52.72]` pixels). Distribution is exactly uniform (0.200 per bin).

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
"pupil_diameter_bin_edges": pupil_edges.astype(float).tolist(),
```

iii. Step 5, Key Decision 7 (global, not per-session, quintiles) and the Decoder Task requirement of "five equal percentile bins". Step 7/9/10 confirm the realised fractions are `[0.200, 0.200, 0.200, 0.200, 0.200]`.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same per-trial 30 Hz `grid` in absolute session time; row 3 of the output matrix. Because pupil samples are dropped only where they are NaN, the interpolant is defined on the same absolute clock as the ophys timestamps and needs no realignment.

ii.
```python
grid = session_grid(start, stop)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
output_trial = np.vstack([image_codes, image_change, running_bins, pupil_bins, trial_outcome])
```

iii. Step 1 notes the SDK aligns eye-tracking frames to sync timestamps; Step 10 records the common-30 Hz-grid alignment for all streams, and the pupil panel of the processing plots was inspected for offsets.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that fixed order. Categories are `["hit", "miss", "false_alarm", "correct_reject"]` → codes 0–3. If a kept trial matches none of them the code raises rather than silently emitting a sentinel.

ii.
```python
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
...
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
```

iii. Step 5 mapping: "trial outcome flags (`hit`, `miss`, `false_alarm`, `correct_reject`) → `output[trial_outcome]`; Static trial label, broadcast across trial timepoints", citing `Trial._get_trial_data`, which Step 1 identifies as the SDK code that "Defines behavioral trial classes and outcomes: `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single integer code is broadcast to every time bin of the trial (`np.full(grid.shape, code)`) so that the nominally static variable is stored as a time-varying row, matching the target format's preference. Realised distribution: hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

ii.
```python
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
output_trial = np.vstack([image_codes, image_change, running_bins, pupil_bins, trial_outcome])
```

iii. Step 5, Key Decision 5: "Represent all outputs as time-varying — To satisfy the decoder format and simplify training, even static trial outcome will be repeated across all bins in a trial." This follows the target-format guidance "If at all possible, make it time-varying."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is mostly *pre-emptive exclusion* plus explicit failure, rather than per-trial repair:
- **Missing eye tracking**: sessions whose NWB lacks `acquisition/EyeTracking/{pupil_tracking,eye_tracking}` are detected in a pre-pass and dropped before either conversion pass (3 active sessions: 795953296, 806456687, 833631914), so that global bin edges and the converted data use the same session set.
- **Blink / untracked pupil frames**: NaN samples are dropped and bridged by linear interpolation; 0-valid-sample and 1-valid-sample sessions are special-cased (raise / constant).
- **Behaviour outside the recorded range**: `np.interp` clamps, i.e. constant extrapolation rather than NaN.
- **Degenerate quintiles**: duplicate percentile edges are nudged apart by 1e-6.
- **ROI table/trace mismatch**: traces are subset to `valid_roi` only if their widths disagree.
- **Too-few-trial sessions**: skipped (checked before and after conversion).
- **Unlabelled trial outcome**: raises `ValueError` (fail loud).
- There is **no** per-session `try/except`, so any unanticipated bad session aborts the whole run — which is exactly what happened on the first full-conversion attempt before the eye-tracking pre-filter was added.
- All-zero neural trials (2,467) are retained deliberately after being verified against the raw files.

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
if valid.sum() == 0: raise ValueError("No valid pupil samples available")
if valid.sum() == 1: return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
...
for i in range(1, len(percentiles)):
    if percentiles[i] <= percentiles[i - 1]:
        percentiles[i] = percentiles[i - 1] + 1e-6
...
if int(raw["keep_mask"].sum()) < 2: ... continue
if len(session_neural) < 2: ... continue
```

iii. Step 10 Edge-case review and Issues Found: "Missing pupil data in 3 active sessions caused full-conversion failure on the first Step 9 attempt. Resolution: exclude those sessions before both passes so global bin edges and converted sessions are computed on the same valid subset." And on the zero-activity trials: "Verified as genuine raw-data zeros, not an interpolation or indexing bug." The excluded session ids are listed in the notes, and `metadata['session_ophys_experiment_ids']` records exactly which sessions survived.

## 9-a. What are the most time-consuming steps of the code?

i. From the AI's own per-session timing prints: pass 1 costs ~0.3–0.5 s/session (opening the NWB, reading the trials/stimulus/running/eye tables, and interpolating running + pupil for every kept trial) and pass 2 costs ~0.5–1.5 s/session, dominated by (a) reading the full `(T, n_neurons)` event array out of HDF5 and (b) the per-trial `interpolate_matrix` call, which for each trial gathers two `(n_bins, n_neurons)` slices and does the blend. Total full run: 364 s for 199 sessions (pass 1 ≈ 85 s, pass 2 ≈ 275 s). Writing the 8.6 GB pickle is also a significant fixed cost, inflated because 11 Hz multiscope data are stored at 30 Hz. The AI's own prediction ("Neural interpolation is still the dominant expected cost because every kept trial needs event traces resampled onto the common grid") matches the measurements.

ii.
```python
t0 = time.perf_counter()
raw = read_session_raw(session)
...
print(f"[pass2] {sess_num:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{len(session_neural)} trials, {raw['n_neurons']} neurons, {total_bins} bins in {elapsed:.2f}s")
```

iii. Step 6 "Code inefficiencies identified" and Step 7 "Run Time Estimates" tables: the AI estimated ~18.7 min from the 2-session sample by scaling pass 2 on neuron-bins, noted "The current full-run estimate still exceeds 15 minutes, so Step 9 should treat optimization as live work", and the actual run came in well under that at ~6 min.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three clear candidates:
- **Per-trial resampling of the neural matrix.** `interpolate_matrix` is called once per trial (51,075 times), each call redoing `searchsorted`, fancy-indexing two `(n_bins, n_neurons)` gathers and a blend. Because all trials of a session share one monotone grid definition, the whole session could have been resampled once onto a session-wide 30 Hz grid and then sliced per trial — removing ~51k Python-level calls and ~2× the gather traffic.
- **The per-timepoint Python loop inside `stimulus_identity_codes`**, which iterates over every in-interval bin to look up `image_to_code`. This is the only genuinely scalar loop in the hot path; it could be a single `np.where(omit, gray, code_lut[name_codes])` after mapping presentation names to codes **once per session** instead of once per bin.
- **The per-trial behaviour interpolation in pass 1** (`interpolate_vector` / `interpolate_pupil` per trial): running and pupil could be interpolated once per session onto a session grid and then gathered per trial.

ii.
```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):     # scalar loop over time bins
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):            # ~250 trials/session
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
    running_cont = interpolate_vector(...)
    pupil_cont   = interpolate_pupil(...)
```

iii. The AI claims "Trial-level interpolation is vectorized over neurons within each trial" (Step 6), which is true but stops at the trial boundary; it did not identify the per-bin loop in `stimulus_identity_codes` or the session-level vectorisation opportunity, presumably because the realised runtime (~6 min) was already inside the 15-minute budget.

## 9-c. What processing does the code repeat multiple times?

i. The dataset is read **three times** end-to-end:
1. `has_required_eye_tracking` opens all 202 active NWB files just to test for two group names;
2. `collect_global_statistics` (pass 1) reopens every file, re-reads the trials table, the stimulus table, running and eye tracking, and interpolates running + pupil for **every kept trial**, solely to derive 8 percentile edges and the 16-name image vocabulary;
3. `convert_sessions` (pass 2) reopens every file and redoes all of that from scratch, plus the events.

So the trials table, the stimulus table, the running and pupil streams, and the per-trial running/pupil interpolation are each computed exactly twice, and the file open/close happens three times. Caching pass 1's per-trial running/pupil (or even just subsampling for the percentile estimate) would have removed roughly the entire 85 s of pass 1.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]   # read #1
...
raw = read_session_raw(session, load_events=False)     # pass 1, read #2
    running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
    pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))
...
raw = read_session_raw(session)                         # pass 2, read #3 — same interpolations again
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The AI framed the two passes as a *memory* optimisation rather than a redundancy: "Two-pass design avoids storing all neural arrays while computing global bin edges" (Step 6). That is a real benefit for the neural array (which pass 1 skips via `load_events=False`), but the behavioural re-interpolation is pure duplicated work and was not flagged.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- **Unused stimulus columns are read and decoded every session**: `trials_id`, `active`, and `flashes_since_change` are loaded (and `decode_strings`/`astype` applied) but never used anywhere; `choose_task_presentation_group` separately reads `active` and `stimulus_block_name` for every candidate presentation group.
- **`collect_global_statistics` materialises every kept trial's interpolated running and pupil trace** (~13 M float32 samples each) only to take four percentiles from each; a subsample or a streaming histogram would do.
- **`native_dt_by_session` and `valid_trial_counts`** are built for all 199 sessions but only used for two `min`/`max` print statements.
- **Up-sampling the 11 Hz multiscope sessions to 30 Hz** roughly triples the stored size of those sessions' neural matrices without adding information, contributing to the 8.6 GB pickle.
- The `input` field stores a `(0, n_bins)` zero array per trial — correct per the spec (no decoder inputs), but the array is allocated 51,075 times rather than shared.
- `make_processing_plot` recomputes trial-filtering counts from the raw table, but that is only active under `--show-processing`.
- Notably, the code does **not** waste time on dF/F: `load_events=False` in pass 1 and never reading `processing/ophys/dff` are both deliberate savings.

ii.
```python
stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name", "is_change",
        "omitted", "trials_id", "active", "flashes_since_change"])   # last three unused
...
running_values.append(interpolate_vector(...))   # full traces kept only for np.nanpercentile
pupil_values.append(interpolate_pupil(...))
...
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
...
print("[setup] native ophys dt range:", min(native_dt.values()), max(native_dt.values()))
print("[setup] valid trial count range:", min(valid_counts.values()), max(valid_counts.values()))
...
input_trial = np.zeros((0, len(grid)), dtype=np.float32)
```

iii. The AI did not document any of these as waste. Its efficiency discussion (Step 6, Step 7) is limited to the two-pass memory design, the choice of direct HDF5 reads over SDK object construction, and vectorising interpolation across neurons within a trial; since the full conversion finished in 364 s it never revisited the question.
