# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object model. It reads the local release
directly: the experiment listing comes from the CSV `project_metadata/ophys_experiment_table.csv`,
and the per-experiment data come from the NWB/HDF5 files in
`behavior_ophys_experiments/behavior_ophys_experiment_<id>.nwb`, opened with `h5py`.
The file glob is intersected with the CSV so only experiments actually present on disk are used
(284 NWB files). The table is then filtered to `passive == False` (202 experiments) and
sorted by `ophys_experiment_id`. A further pre-pass opens every remaining NWB and drops any
that lack the `acquisition/EyeTracking/{pupil_tracking,eye_tracking}` groups (3 dropped → 199).
Note that the AI does **not** filter on `project_code`, so both `VisualBehavior` (single-plane,
168 active files) and `VisualBehaviorMultiscope` (multi-plane, 34 active files) are included.

Per session, a single function `read_session_raw` pulls every stream out of one HDF5 file:
the trials interval table, the change-detection stimulus presentation table, the
`event_detection` traces + timestamps, running speed + timestamps, and pupil width/height +
timestamps. Trials are then enumerated inside each file from the trials table.

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
        ...
        ophys_timestamps = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
        events = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
        ...
```

iii. The AI documented (Step 5, Key Decision 9) that it deliberately bypassed the SDK:
*"The reference SDK is still the guide for field semantics and processing, but the current
environment's `pynwb/hdmf` stack cannot instantiate these NWB 2.6.0 files through
`BehaviorOphysExperiment.from_nwb_path` because of an `external_resources` abstract-method
mismatch. Direct HDF5 reads will therefore mirror the SDK field definitions explicitly."*
It cross-checked its raw reads against the SDK semantics in Step 10 ("Reference code comparison")
and validated them with `np.allclose` spot checks against independently re-loaded raw NWB data.
Restriction to active sessions is justified in Step 5 Key Decision 1 (passive sessions give
degenerate trial outcomes). The eye-tracking pre-filter is justified in Step 10 Check 5 as an
edge-case fix after the first full run crashed.

## 1-b. How are the data split into subjects?

i. Subjects are the `mouse_id` column of `ophys_experiment_table.csv`, cast to `str`.
Subjects are registered lazily, in the order sessions are processed (i.e. ascending
`ophys_experiment_id`), and `subject_idx` holds one index per emitted session. 38 subjects result.

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

iii. Step 5 variable-mapping table: *"NWB `general/subject/subject_id` / metadata `mouse_id` →
`subjects`, `subject_idx`; Convert to global subject list and per-session index; Use mouse
identifier strings."* The AI verified the subject count (38) against the raw metadata table in
Step 9/Step 10 Check 4.

## 1-c. How are the data split into sessions?

i. **One NWB experiment file = one decoder "session".** The AI does not group imaging planes by
`ophys_session_id`. For the 168 active single-plane (`VisualBehavior`) files this is 1:1, but the
34 active `VisualBehaviorMultiscope` files come from only 6 behavioral sessions of a single mouse
(457841, 3–7 planes each), so those 6 behavioral sessions appear as 34 separate decoder sessions
with duplicated trials/behavior and the simultaneously recorded neurons split apart. The
verification log shows the consequence: `Subject 457841: 34 sessions`, 17% of the 199 sessions
from one mouse. Passive sessions are excluded up front, and sessions with fewer than 2 kept
trials are skipped (never triggered).

ii.
```python
exp_table = exp_table[~exp_table["passive"]].copy()
...
for sess_num, session in enumerate(sessions, start=1):   # session == one ophys_experiment NWB
    raw = read_session_raw(session)
    if int(raw["keep_mask"].sum()) < 2:
        print(f"[pass2] skipping session {session.ophys_experiment_id} ...")
        continue
    ...
    region_idx = brain_region_to_idx.setdefault(session.targeted_structure, len(brain_region_to_idx))
    ...
    data["brain_region_idx"].append(np.full(raw["n_neurons"], region_idx, dtype=np.int64))
```

iii. Step 4 discrepancy table: *"Treat each NWB experiment file as one decoder session because
neural traces are experiment-specific; preserve subject/session metadata so multiple experiments
from one behavior session remain linked through subject/session fields."* Exclusion of passive
sessions is Step 5 Key Decision 1: *"the decoder task requires trial outcome and the strategy
paper's behavioral analyses focus on active sessions. Passive sessions also produce degenerate
outcomes (sample passive file: all go trials are misses and all catch trials are correct
rejects)."* The AI recorded the 284-files-vs-247-sessions mismatch in Step 2/Step 4 but never
revisited the duplication it creates, and never noted the resulting per-mouse imbalance.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if
`(go | catch) & ~aborted & ~auto_rewarded`. The trial window is the full
`start_time` → `stop_time` interval (mean ≈ 8.5 s), re-sampled on a fixed 30 Hz grid anchored at
`start_time`, giving variable-length trials (211–377 bins in the full run). 51,075 trials result.

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

iii. Step 5 Key Decision 4: *"Segment trials from `start_time` to `stop_time`: Trial boundaries
will come directly from the NWB `intervals/trials` table, after filtering to keep only
`(go or catch) and not aborted and not auto_rewarded`."* Step 4 grounds the taxonomy in
`Trial._get_trial_data` of the SDK and in the task instruction ("Include both the Go and Catch
trials, but exclude the Aborted and Auto-rewarded trials"). The full window is used so all
outputs can be time-varying over pre-change and post-change periods.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: aborted and auto-rewarded trials are dropped, and only `go` or `catch` trials are
kept (this is one boolean mask; no separate `change_time` non-null test is applied, which is
equivalent in this dataset — no kept trial has a NaN `change_time`). Session-level: passive
sessions dropped (82 files), sessions with no eye-tracking groups dropped (3 files: 795953296,
806456687, 833631914), and sessions with < 2 kept trials dropped (both before and after
conversion; neither triggered). No trial is dropped for truncation because no kept trial extends
beyond the ophys recording. No neuron-level trial rejection is applied; 2,467 trials (4.8%) with
all-zero event matrices are kept.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session {session.ophys_experiment_id} because it has "
          f"{int(raw['keep_mask'].sum())} valid trials")
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

iii. Step 4: *"Trial inclusion for decoder should follow the common interpretation across sources:
include `go` and `catch`, exclude `aborted` and `auto_rewarded`"*, with the whitepaper rationale
that aborted trials are early-lick resets and free-reward trials are not standard go/catch
contingencies. Step 5 Key Decision 8 justifies the ≥ 2-trial rule as a format requirement
("There needs to be at least two trials within each session"). Step 10 Check 5 documents the
eye-tracking exclusion: *"Found 3 active sessions with no `acquisition/EyeTracking` group. Fix:
exclude ... before both conversion passes"* so that global bin edges and converted sessions come
from the same subset. The all-zero-event trials were investigated in Step 10 Check 1 and kept as
*"genuine sparsity of the released calcium-event signal, not conversion corruption."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` (detected calcium event magnitudes, shape
`(T, n_rois)`) with `processing/ophys/event_detection/timestamps`. **Not** dF/F. A safety net
filters columns by `cell_specimen_table/valid_roi` if the widths disagree (a no-op here — the
released NWBs already contain only valid ROIs).

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
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

iii. Step 5 Key Decision 2: *"Use `events` instead of `dff_traces`: The whitepaper explains dF/F
generation, but the strategy paper explicitly states its neural analyses used detected calcium
events. Events are already present in the NWB files and best match the reference analyses."*
The AI quoted the paper's methods verbatim in Step 3: *"For all analysis of neural data we used
the detected calcium events as described in Garrett et al."* Metadata records
`"neural_signal": "precomputed calcium events"`.

## 2-b. How is the `neural` data processed?

i. Only one transformation: per trial, the `(T, N)` event matrix is linearly interpolated from
the native ophys timestamps onto the trial's 30 Hz grid and transposed to `(n_neurons, n_bins)`,
stored as `float32`. No normalization, smoothing, z-scoring, baseline subtraction or
cross-plane stacking (each session is one plane by construction). Interpolation is vectorized
over neurons; queries before the first ophys timestamp are held constant, queries after the last
are linearly extrapolated (never triggered in this dataset).

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

iii. Step 5 variable mapping: *"Use precomputed event traces; transpose to
`(n_neurons, n_timepoints)` per trial after resampling/interpolation to common 30 Hz trial grid."*
The events themselves are already the product of the Allen pipeline (motion correction,
demixing, neuropil correction, dF/F, event detection), so the AI added nothing. Step 6 notes
*"Trial-level interpolation is vectorized over neurons"* as an explicit speed-up.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional filtering. The AI relies on the Allen pipeline's ROI curation already baked into
the released NWB files, and adds only the defensive `valid_roi` mask described in 2-a. All
neurons in `event_detection` are kept (4–666 per session, 29,168 total). Trials whose event
matrix is entirely zero are kept, not discarded.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. Step 4: *"Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc
neuron filtering beyond reference QC/filtering already reflected in the files."* Step 10 Check 3
adds: *"Reference: `CellSpecimens(..., exclude_invalid_rois=True)` filters to valid ROIs.
Conversion: if `valid_roi` is present and differs from event-trace width, traces are filtered to
that mask. Resulting neuron counts match raw released valid-ROI counts."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the trials-table `start_time`: the time grid is
`start_time + k·(1/30 s)` for `k = 0 … ceil((stop_time − start_time)·30) − 1`, and every stream
(neural, running, pupil, image identity, image change) is evaluated on that same grid in absolute
session time. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`,
`off_end = None` (variable-length trials). Note that the grid is anchored on the behavioral trial
start rather than on an ophys frame, so ophys samples are interpolated onto it rather than
indexed.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
image_change = stimulus_change_codes(raw["stimulus"], grid)
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Step 10 Check 3: *"Reference: synchronized timestamps across behavior/ophys streams; strategy
paper interpolates activity to common 30 Hz timestamps. Conversion: all trial streams are aligned
in absolute experiment time and resampled to a common 30 Hz grid from raw trial `start_time` /
`stop_time`."* The AI verified alignment with `np.allclose` against raw re-interpolation for two
sessions (Step 10 Check 2) and by visual inspection of `processing_*.png`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 33.333 ms (30 Hz) uniformly for every trial and session. Yes — rebinning (strictly,
resampling by linear interpolation) is applied to every stream. Native ophys rates are ~31 Hz for
single-plane files (slight downsampling) and ~11 Hz for the multiscope files (≈3× upsampling).
`metadata['time_bin_size'] = 33.333…` ms and
`metadata['resampling_reference'] = "common 30 Hz grid derived from source ophys timestamps"`.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
```

iii. Step 5 Key Decision 3: *"Use a common 30 Hz time base for all sessions: The target format
requires one shared bin size across sessions, while local data mix ~31 Hz single-plane and ~11 Hz
multi-plane ophys sampling. The strategy paper linearly interpolates calcium event responses onto
common 30 Hz timestamps for neural and running analyses, and behavior/eye tracking are naturally
30 Hz, so 30 Hz is the most defensible common grid."* The AI quotes both the whitepaper
("31 Hz for single plane … 11 Hz for each plane in multi-plane experiments … eye tracking (30 Hz),
and behavior (30 Hz)") and the paper ("linearly interpolating onto a consistent set of 30hz
timestamps").

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The change-detection **stimulus presentation** table, not the trials table. The AI selects the
image-presentation interval group that has a `change_detection` `stimulus_block_name` and the most
`active` rows, then uses its `start_time`, `stop_time`, `image_name` and `omitted` columns. The
vocabulary is built globally across all included sessions from non-omitted image names, with an
extra `gray` class prepended (17 classes total: `gray` + 16 images).

ii.
```python
def choose_task_presentation_group(h5f):
    ...
    if "stimulus_block_name" in grp:
        block_names = decode_strings(np.asarray(h5_array(grp["stimulus_block_name"])))
        has_change_detection = int(any("change_detection" in str(x) for x in block_names.tolist()))
    score = (has_change_detection, n_active)
```
```python
image_names.update(
    str(x)
    for x, omitted in zip(stim["image_name"], stim["omitted"])
    if (not omitted) and str(x) not in ("", "None", "nan")
)
...
image_values = ["gray"] + sorted(image_names)
```

iii. Step 5 variable mapping: *"stimulus-presentation `image_name`, `start_time`, `stop_time`,
`omitted`, active task block only → `output[image_identity]`; Piecewise-constant categorical
signal on 30 Hz grid; use actual image name during image display; use `gray` during gray-screen or
omission periods."* Step 5 Key Decision 6: *"Use `gray` as an explicit image-identity class:
Because the task includes 500 ms gray periods and omissions extend gray instead of showing an
image, a `gray` category is needed for a complete time-varying identity signal."* Step 1 notes the
SDK function `get_stimulus_presentations` as the corresponding reference code.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 30 Hz bin, `searchsorted` finds the last presentation whose `start_time` ≤ bin time;
the bin gets that presentation's image code only if the bin also falls before its `stop_time`
(≈250 ms flash) and the presentation is not omitted; otherwise the bin is `gray`. Codes come from
a global, sorted `image_to_code` map so codes are consistent across sessions. Result: 66.9% of
bins are `gray`, the 16 images 1.9–2.2% each.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    starts, stops = stimulus["start_time"], stimulus["stop_time"]
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = idx >= 0
    valid &= idx < len(starts)
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stops[idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        names = np.asarray(stimulus["image_name"][sub_idx], dtype=object)
        omit = stimulus["omitted"][sub_idx]
        target = np.full(sub_idx.shape, image_to_code["gray"], dtype=np.int64)
        for i, (name, is_omitted) in enumerate(zip(names, omit)):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        codes[np.flatnonzero(valid)[in_interval]] = target
    return codes
```

iii. Same as 3-a: the AI wanted a faithful "what is on the screen right now" signal. It validated
the result in Step 10 Check 2 (*"converted `image_identity` … matched independently reconstructed
raw-data equivalents exactly"*) and checked the distribution against the expected 250/750 ms duty
cycle (Step 9: *"gray 0.6695, each image class 0.0189-0.0224"*).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same `grid` array used to interpolate the neural data, so bin
*k* of the image-identity row and bin *k* of the neural matrix refer to the same absolute time.
Alignment error is bounded by half a bin (16.7 ms) at the flash on/off transitions.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
...
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 10 Check 3 (temporal alignment): *"all trial streams are aligned in absolute experiment
time and resampled to a common 30 Hz grid."* Step 7: *"No visual sign of cross-stream temporal
misalignment in the reviewed plots."*

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The same stimulus presentation table: `is_change`, `omitted`, `start_time`, `stop_time`.
`is_change` is True only for real image changes; sham (catch-trial) changes carry
`is_sham_change` instead and so are never labeled 1.

ii.
```python
stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name", "is_change",
                                        "omitted", "trials_id", "active", "flashes_since_change"])
stim["is_change"] = stim["is_change"].astype(bool)
```

iii. Step 5 variable mapping: *"stimulus-presentation `is_change`, `start_time`, `stop_time` →
`output[image_change]`; Binary time-varying label: 1 during the changed-image presentation window,
else 0 … Marks the post-change flashed image itself rather than a one-bin impulse; catch trials
remain 0."* Step 1 lists `compute_is_sham_change` as the SDK function that distinguishes sham from
real changes. This replaced an earlier `change_time`-impulse version (Step 10 Issues Resolved:
*"The initial `image_change` target was too sparse because it used a single-bin impulse at
`change_time` … sample validation accuracy moved from below chance to above chance."*).

## 4-b. What processing is involved in computing `output` *Image change*?

i. The same interval lookup as image identity: each 30 Hz bin is 1 if it falls inside a
presentation interval flagged `is_change and not omitted`, else 0. Because a flash lasts
~250 ms, each change is 7–8 consecutive bins; the following 500 ms gray gap is 0. Overall
2.59% of bins are 1.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    idx = np.searchsorted(stimulus["start_time"], query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    valid = idx >= 0
    valid &= idx < len(stimulus["start_time"])
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stimulus["stop_time"][idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        changed = stimulus["is_change"][sub_idx] & (~stimulus["omitted"][sub_idx])
        codes[np.flatnonzero(valid)[in_interval]] = changed.astype(np.int64)
    return codes
```

iii. See 4-a. The AI verified the label against raw data on three trials in Step 12
(*"trial 0: 8 positive bins, `np.allclose == True`"*, etc.) and checked non-degeneracy
(*"positive `image_change` bins occupy 2.5872% of all bins, so the class is sparse but not
degenerate"*).

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary — no threshold on a continuous quantity. The "categorization" is the
choice of temporal window: 1 for the duration of the changed flash only (~250 ms, 7–8 bins),
0 everywhere else including the 500 ms gray period that follows the change. `output_values`
for this dimension are `["no_change", "change"]`.

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

iii. Step 6: *"`image_change` is constructed from the stimulus table's `is_change` presentation
interval instead of a single-bin impulse at `change_time`, which yields a less degenerate decoder
target while staying aligned to the task structure."* The AI treated the change flash interval
itself as the natural extent of "right after a change in image identity"; it did not consider
extending the window to cover the slower calcium transient.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identical to image identity: computed on the shared per-trial `grid`, so row 1 of the output
matrix is bin-for-bin aligned with the neural matrix. The change onset bin is the first grid bin
at or after `change_time` (the change flash `start_time` equals the trials-table `change_time`).

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Same justification as 3-c; the AI additionally re-checked the change-onset alignment in
Step 12 ("Temporal-alignment check") using the saved processing plots.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — the SDK's default
low-pass-filtered running speed (cm/s), not `speed_unfiltered`.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. Step 5 variable mapping: *"running speed timeseries → `output[running_speed_bin]` … Uses
filtered running speed, matching SDK default."* Step 1 records that
`RunningSpeed.from_stimulus_file` computes speed on stimulus/sync timestamps with monitor delay
forced to 0, and Step 3 records the whitepaper's filtering chain (unwrap, despike, z≥10 artifact
removal, 10 Hz low-pass).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`) of the filtered speed onto the trial's 30 Hz grid, cast to
`float32`. `np.interp` clamps outside the source range rather than producing NaN, so no missing
values arise. No smoothing, clipping or absolute value is applied. The interpolated values are
then discretized (5-c). The same interpolation is performed twice: once in pass 1 to accumulate
values for the global bin edges, once in pass 2 to build the output.

ii.
```python
def interpolate_vector(source_t, source_values, query_t):
    return np.interp(query_t, source_t, source_values).astype(np.float32)
...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Step 5: *"Interpolate running speed onto 30 Hz trial grid; discretize with global quintile
bins across included active-session timepoints."* Rationale is the common-30 Hz-grid decision
(Key Decision 3) plus the paper's own use of linear interpolation onto 30 Hz timestamps.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count (quintile) bins. Pass 1 concatenates the interpolated running speed of every
kept trial of every included session, then takes the 20/40/60/80th percentiles as interior edges
(with a 1e-6 nudge if percentiles tie, e.g. from the many zero-speed samples), and `np.digitize`
assigns bins 0–4. Edges are global — one set for the whole dataset — and are stored in
`metadata['running_speed_bin_edges']`. The realized distribution is exactly
[0.200, 0.200, 0.200, 0.200, 0.200].

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
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 5 Key Decision 7: *"Discretize continuous outputs globally, not per session:
Running-speed and pupil-diameter bin edges will be computed from all valid included timepoints
across the converted dataset so class definitions are shared across sessions."* This follows the
instruction "discretized into five equal percentile bins". The AI confirmed the uniform 20% class
fractions in Step 9/Step 10 Check 4.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly onto the shared per-trial `grid` in absolute session time, so it is
bin-for-bin aligned with the neural matrix (row 2 of the output array).

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The running timestamps are on the same hardware-synced session clock as the ophys timestamps
(Step 3: *"Temporal synchronization was performed by recording experimental clocks on a single
NI PCI-6612 digital IO board sampled at 100 kHz"*), so interpolating onto the common grid
guarantees alignment. Verified by `np.allclose` raw-vs-converted checks in Step 10 Check 2.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height` (ellipse-fit axes), with
timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`. Pupil diameter is defined as
the element-wise **maximum of width and height**. Blink frames are already stored as NaN in these
arrays (they coincide exactly with `acquisition/EyeTracking/likely_blink`, ~9% of frames), and the
AI excludes them via a finiteness mask rather than reading `likely_blink` explicitly.

ii.
```python
pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. Step 5 variable mapping: *"eye-tracking pupil width/height + blink mask →
`output[pupil_diameter_bin]`; Compute pupil diameter as `max(width, height)`; use blink-masked
values; interpolate across valid timestamps onto 30 Hz trial grid … Small blink-related gaps are
filled by interpolation after applying reference invalid-frame masking."* Step 1 records that the
SDK's `EyeTrackingTable` masks blink frames to NaN, which is what the released arrays contain.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite (blink) samples are dropped, then the remaining samples are linearly interpolated
onto the trial's 30 Hz grid — so blink gaps are bridged by a straight line between the last and
next valid samples. `np.interp` clamps at the ends, so no NaN survives into the output. Degenerate
cases raise (zero valid samples) or broadcast a constant (one valid sample). The result is then
discretized (6-c). As with running speed, this is computed twice (pass 1 and pass 2).

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    if valid.sum() == 0:
        raise ValueError("No valid pupil samples available")
    if valid.sum() == 1:
        return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. Step 5 mapping (quoted in 6-a): blink frames are masked before interpolation so blink
artifacts do not propagate; gaps are bridged rather than left missing so every bin has a valid
class label.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: global 20/40/60/80th-percentile edges computed in pass 1
over all interpolated pupil values from all kept trials of all included sessions, then
`np.digitize` into bins 0–4. Edges saved to `metadata['pupil_diameter_bin_edges']`. Realized
distribution is exactly uniform (0.200 each).

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Step 5 Key Decision 7 (global, not per-session, discretization) and the instruction
"discretized into five equal percentile bins". Verified uniform in Step 9 / Step 10 Check 4.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same per-trial `grid`, so row 3 of the output matrix is bin-for-bin
aligned with the neural matrix.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same hardware-sync argument as running speed; eye tracking is natively 30 Hz (whitepaper),
so resampling onto the 30 Hz grid is close to a phase shift only. Verified by the Step 10
`np.allclose` output sanity checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`,
`false_alarm`, `correct_reject`, checked in that order for the kept trial.

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

iii. Step 5 variable mapping: *"trial outcome flags (`hit`, `miss`, `false_alarm`,
`correct_reject`) → `output[trial_outcome]`; Static trial label, broadcast across trial
timepoints"*, sourced from `Trial._get_trial_data` in the SDK (Step 1). Realized distribution:
hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The label is mapped to an integer 0–3 and broadcast (`np.full`) across every bin of the trial,
so the static per-trial variable is stored as a constant time series (row 4). A trial matching
none of the four flags raises rather than being silently coded.

ii.
```python
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 5 Key Decision 5: *"Represent all outputs as time-varying: To satisfy the decoder format
and simplify training, even static trial outcome will be repeated across all bins in a trial."*
This follows the format instruction "If at all possible, make it time-varying".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, all pre-emptive rather than exception-based:
- **Sessions with no eye tracking**: detected by opening every candidate NWB before conversion and
  checking for the `pupil_tracking`/`eye_tracking` groups; 3 sessions excluded from *both* passes
  so bin edges and converted data use the same subset.
- **Blink / missing pupil samples**: NaN samples dropped and bridged by linear interpolation;
  a session with zero valid pupil samples raises.
- **Behavioral samples outside the trial window**: `np.interp` clamps to the endpoints instead of
  producing NaN, so there are no missing bins to impute.
- **Short sessions**: sessions with < 2 kept trials are skipped, both before and after conversion.
- **ROI table mismatch**: event columns are filtered by `valid_roi` if the counts disagree.
- **Unlabeled trial outcome**: raises `ValueError` (fail-fast) instead of emitting a sentinel.
- **Stimulus table ambiguity**: `choose_task_presentation_group` scores candidate interval tables
  and raises if no active change-detection table exists.
- There is **no** `try/except` around per-session processing: a bad session aborts the whole run.
  All-zero-event trials (4.8%) are kept deliberately.

ii.
```python
def has_required_eye_tracking(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        if "acquisition" not in h5f or "EyeTracking" not in h5f["acquisition"]:
            return False
        eye = h5f["acquisition"]["EyeTracking"]
        return "pupil_tracking" in eye and "eye_tracking" in eye
...
excluded = len(sessions) - len(filtered_sessions)
if excluded:
    print(f"[setup] excluded {excluded} active sessions missing eye-tracking pupil data")
```
```python
valid = np.isfinite(pupil_diameter)
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
...
if best_name is None:
    raise RuntimeError(f"Could not find active task stimulus table in {h5f.filename}")
...
raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. Step 10 Check 5 / Issues Found and Resolved: *"Missing pupil data in 3 active sessions caused
full-conversion failure on the first Step 9 attempt. Resolution: exclude those sessions before
both passes so global bin edges and converted sessions are computed on the same valid subset."*
Step 10 Check 1 justifies keeping the all-zero trials: *"the warnings reflect genuine sparsity of
the released calcium-event signal, not conversion corruption. They are therefore not fixable
without changing the referenced neural representation."*

## 9-a. What are the most time-consuming steps of the code?

i. The AI instrumented both passes with `time.perf_counter` and printed per-session timings. The
actual full run (364 s) splits as: session discovery + eye-tracking pre-scan (opens 202 NWBs),
pass 1 = 106 s (~0.5 s/session: HDF5 reads of trials/stimulus/running/pupil plus per-trial
behavioral interpolation), pass 2 = ~258 s (~1.3 s/session: reading the full `(T, N)` event array
plus the per-trial `interpolate_matrix` over all neurons, which dominates — 2.05 × 10⁹
neuron-bins in total). Writing the 8.1 GB pickle is also non-trivial.

ii.
```python
t0 = time.perf_counter()
raw = read_session_raw(session, load_events=False)
...
print(f"[pass1] {idx:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{valid_trial_counts[session.ophys_experiment_id]} valid trials in {elapsed:.2f}s")
...
print(f"[pass2] {sess_num:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{len(session_neural)} trials, {raw['n_neurons']} neurons, {total_bins} bins in {elapsed:.2f}s")
```

iii. Step 6: *"Neural interpolation is still the dominant expected cost because every kept trial
needs event traces resampled onto the common grid."* Step 7 extrapolated the sample timings by
neuron-bins to estimate ~18.7 min for the full run; the actual run took 6.1 min, so the estimate
was conservative and no further optimization was pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorized the expensive one (neural interpolation across neurons) but left three
Python-level loops that could be vectorized and did not document them:
- `stimulus_identity_codes` loops **per time bin** (`for i, (name, is_omitted) in enumerate(...)`)
  to map image names to codes; this could be a single precomputed per-presentation code array
  indexed with `sub_idx`.
- The per-trial loop in `convert_sessions`/`collect_global_statistics` re-runs `np.searchsorted`
  over the *whole-session* timestamp arrays for every trial; one session-level resample (or a
  single concatenated query vector) would do all trials at once.
- `has_required_eye_tracking` opens every NWB in a separate serial pass; this could be folded into
  pass 1, and sessions are independent so the whole conversion could be parallelized across
  processes (the AI considered but did not implement parallelism).

ii.
```python
        for i, (name, is_omitted) in enumerate(zip(names, omit)):   # per-time-bin Python loop
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
```
```python
        for trial_idx in np.flatnonzero(raw["keep_mask"]):
            ...
            neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid)
```

iii. Step 6 "Code speedups added": *"Trial-level interpolation is vectorized over neurons within
each trial."* The AI's own efficiency review stopped there — it judged the total runtime
acceptable and did not revisit the remaining loops.

## 9-c. What processing does the code repeat multiple times?

i. Every included NWB file is opened **three** times and the same work is redone:
1. `has_required_eye_tracking` opens the file just to test for two groups.
2. Pass 1 (`collect_global_statistics`) reads trials/stimulus/running/pupil, builds every trial's
   30 Hz grid, and interpolates running speed and pupil for every trial — purely to compute the
   global quintile edges and the image vocabulary.
3. Pass 2 (`convert_sessions`) re-reads the same file (now including events) and recomputes the
   identical grids and the identical running/pupil interpolations.

So the trial grids and both behavioral interpolations are computed exactly twice, and the trials
and stimulus tables are parsed twice; this accounts for the entire 106 s of pass 1 (~30% of the
runtime). Additionally, `valid_trial_counts` and `native_dt_by_session` are computed in pass 1 and
the keep-mask is recomputed in pass 2.

ii.
```python
# pass 1
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
    pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))

# pass 2 — same three lines again
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Step 6: *"Two-pass design avoids storing all neural arrays while computing global bin
edges."* The AI framed the two-pass structure purely as a memory optimization and never
acknowledged the duplicated I/O and interpolation it costs; caching the (small) per-trial
behavioral vectors from pass 1 would have removed the duplication at modest memory cost.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, none documented by the AI:
- Stimulus columns `trials_id`, `active`, and `flashes_since_change` are read and type-cast for
  every session but never used. Trials column `change_time` is read and cast but is unused in the
  final version (`image_change` now comes from the stimulus table).
- `collect_global_statistics` computes and returns `valid_trial_counts` and `native_dt_by_session`
  for every session; they are only printed as a min/max range.
- `RNG = np.random.default_rng(0)` is created and never used.
- The eye-tracking pre-scan opens 202 files to read nothing.
- Every output row is stored as `int64` although all values fit in `int8`; the five output rows are
  therefore ~8× larger than necessary (~0.5 GB of the 8.1 GB pickle). The static `trial_outcome`
  is materialized as a full-length constant vector per trial (this one is required by the chosen
  time-varying representation).
- `pupil_width`/`pupil_height` are both loaded and combined with `np.maximum`; only the combined
  signal is used.

ii.
```python
RNG = np.random.default_rng(0)          # never used
...
stim["trials_id"] = stim["trials_id"].astype(np.int64)                 # unused
stim["active"] = stim["active"].astype(bool)                           # unused
stim["flashes_since_change"] = stim["flashes_since_change"].astype(np.int64)  # unused
trials["change_time"] = trials["change_time"].astype(np.float64)       # unused in final version
...
return running_edges, pupil_edges, image_values, valid_trial_counts, native_dt_by_session
...
print("[setup] native ophys dt range:", min(native_dt.values()), max(native_dt.values()))
print("[setup] valid trial count range:", min(valid_counts.values()), max(valid_counts.values()))
...
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The AI gave no justification for these — they are residue of exploration and of the earlier
`change_time`-based `image_change` implementation. The `native_dt`/`valid_trial_counts` printouts
served as sanity diagnostics (Step 7/Step 9 statistics reporting), which is a legitimate if minor
use; the rest is dead work.
