# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object API. It documented (Step 5, decision 9) that the environment's `pynwb/hdmf` stack cannot instantiate these NWB 2.6.0 files through `BehaviorOphysExperiment.from_nwb_path` (`external_resources` abstract-method mismatch), so it reads the released NWB/HDF5 files directly with `h5py`, using the SDK source only as the schema/semantics guide.
The inventory of data comes from the local release metadata table `project_metadata/ophys_experiment_table.csv`, intersected with the NWB files actually present on disk (284 files). Two filters are then applied *before* loading anything: `passive == False` (drops 82 passive experiment files) and a pre-scan that opens every candidate NWB and requires `acquisition/EyeTracking/{pupil_tracking,eye_tracking}` to exist (drops 3 more). The result is 199 "sessions".
There is **no `project_code` filter**: the local subset contains 239 `VisualBehavior` experiments plus 45 `VisualBehaviorMultiscope` experiments (1 mouse), and the AI keeps the active ones from both (168 + 34 = 202 → 199 after the eye-tracking filter).
Loading is done in two passes over the same files: pass 1 (`collect_global_statistics`, `load_events=False`) reads trials/stimulus/running/pupil to build the global image vocabulary and the global running/pupil quintile edges; pass 2 (`convert_sessions`) re-reads each file including `event_detection` and builds the per-trial arrays.

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
    return filtered_sessions
```

```python
def read_session_raw(session, load_events=True):
    with h5py.File(session.path, "r") as h5f:
        trial_group = h5f["intervals"]["trials"]
        trials = read_interval_table(trial_group, ["go","catch","aborted","auto_rewarded",
            "hit","miss","false_alarm","correct_reject","change_time","start_time","stop_time"])
        stim_group = choose_task_presentation_group(h5f)
        stim = read_interval_table(stim_group, ["start_time","stop_time","image_name",
            "is_change","omitted","trials_id","active","flashes_since_change"])
        ophys_timestamps = np.asarray(
            h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
        if load_events:
            events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], ...)
        running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], ...)
        pupil_width  = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], ...)
```

iii. From CONVERSION_NOTES Step 5 decision 9: "The reference SDK is still the guide for field semantics and processing, but the current environment's `pynwb/hdmf` stack cannot instantiate these NWB 2.6.0 files ... Direct HDF5 reads will therefore mirror the SDK field definitions explicitly." Decision 1 justifies dropping passive sessions: "the decoder task requires trial outcome and the strategy paper's behavioral analyses focus on active sessions. Passive sessions also produce degenerate outcomes (sample passive file: all go trials are misses and all catch trials are correct rejects)." Step 10 justifies the eye-tracking pre-filter: 3 active sessions have no `acquisition/EyeTracking` group at all, which crashed the first full run; excluding them before *both* passes keeps the global quintile edges and the converted sessions computed on the same subset. The AI verified equivalence to the raw files with `np.allclose` spot checks (Step 10, check 2).

## 1-b. How are the data split into subjects?

i. Subjects are the `mouse_id` column of `ophys_experiment_table.csv`, cast to `str`, registered lazily in first-encountered order as sessions are converted. 38 mice result (37 `VisualBehavior` + 1 `VisualBehaviorMultiscope`). `subject_idx` is one integer per converted session.

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

iii. CONVERSION_NOTES Step 5: "NWB `general/subject/subject_id` / metadata `mouse_id` → `subjects`, `subject_idx` ... Use mouse identifier strings." The count (38) was cross-checked against the local subset in Steps 2 and 9 ("Subjects | ... | 38 | 38 | Yes for local subset").

## 1-c. How are the data split into sessions?

i. **One converted "session" = one `ophys_experiment_id` = one NWB file = one imaging plane.** The AI explicitly considered and rejected grouping planes by `ophys_session_id`. Consequently the single multiscope mouse in the local subset contributes 34 converted sessions that actually come from only 6 distinct behavioral sessions, so the same trials/behavior are duplicated 3–7 times, each with a different small subset (4–37) of that session's neurons. For single-plane `VisualBehavior` experiments (168 of 199) experiment and session are 1:1, so the split matches the reference there.

ii.
```python
@dataclass(frozen=True)
class SessionInfo:
    ophys_experiment_id: int
    path: Path
    mouse_id: str
    targeted_structure: str
    session_type: str
    project_code: str
    passive: bool
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    data["neural"].append(session_neural)
```
and in the metadata:
```python
"session_ophys_experiment_ids": [int(s.ophys_experiment_id) for s in sessions],
```

iii. CONVERSION_NOTES Step 4, discrepancy row "Session definition": "SDK distinguishes `behavior_session`, `ophys_session`, and `ophys_experiment`; one NWB file corresponds to one `ophys_experiment` ... Local data are 284 experiment NWBs but only 247 unique behavior/ophys sessions ... Whitepaper says single-plane has 1 experiment/session, multi-plane can have up to 8 experiments/session. **Resolution**: Treat each NWB experiment file as one decoder session because neural traces are experiment-specific; preserve subject/session metadata so multiple experiments from one behavior session remain linked through subject/session fields."

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table (the SDK-built trials table). Each kept trial spans `start_time` → `stop_time`, i.e. the full variable-length trial window (pre-change flashes + response window), not a fixed window around the change. The resampled grid gives 211–377 bins (≈7–12.5 s) at 30 Hz.

ii.
```python
trials = read_interval_table(trial_group, ["go","catch","aborted","auto_rewarded", ...,
                                           "change_time","start_time","stop_time"])
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop  = float(raw["trials"]["stop_time"][trial_idx])
    grid  = session_grid(start, stop)

def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. CONVERSION_NOTES Step 5 decision 4: "Segment trials from `start_time` to `stop_time`: Trial boundaries will come directly from the NWB `intervals/trials` table, after filtering to keep only `(go or catch) and not aborted and not auto_rewarded`." Step 1 notes that `Trials._get_trial_bounds` adjusts consecutive trial bounds so there is no dead time between trials, and Step 4 resolves the trial taxonomy: "Trial inclusion for decoder should follow the common interpretation across sources: include `go` and `catch`, exclude `aborted` and `auto_rewarded`" — which is exactly what the task instructions ask for.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, at three levels:
- **Trial level**: keep only `(go | catch) & ~aborted & ~auto_rewarded` (51,075 of ~171,887 raw trials survive).
- **Session level (before loading)**: `passive == False`; and the NWB must contain an `EyeTracking` group with `pupil_tracking` and `eye_tracking` (excludes 795953296, 806456687, 833631914).
- **Session level (after filtering)**: sessions with `< 2` valid trials are skipped — checked once on the keep-mask and again after conversion. In practice no session was dropped (min = 39 valid trials).
- No trial is dropped for being truncated or having a NaN `change_time`; the AI did not add a `change_time.notna()` guard (empirically a no-op for go/catch trials).

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
    print(f"[pass2] skipping session {session.ophys_experiment_id} after conversion "
          f"because it has {len(session_neural)} trials")
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

iii. Aborted/auto-rewarded exclusion is mandated by the task instructions and cross-checked against the SDK trial logic (Step 1: "aborted trials are detected from an `abort` event; if aborted, then `go = catch = auto_rewarded = False`") and the whitepaper ("Free-reward / auto-reward trials occur for the first 5 trials of sessions and after 10 consecutive misses; these should not be treated as standard go/catch contingencies"). Step 5 decision 8: "Require at least two valid trials per session after filtering ... so no expected extra session loss from this rule." Step 10 edge-case review documents the eye-tracking exclusion as a fix for a crash in the first full run.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` (detected calcium event magnitudes, shape `(T, n_rois)`) with `processing/ophys/event_detection/timestamps`. **Not** `dff_traces`, which is also present in the NWB files.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```

iii. CONVERSION_NOTES Step 4: "SDK exposes both `dff_traces` and `events`; no need to recompute dF/F at load time ... Whitepaper defines dF/F processing; strategy paper states analyses used 'detected calcium events'. **Resolution**: Use precomputed `events` as the neural signal for conversion because that best matches the analysis paper and avoids diverging from reference processing." Step 5 decision 2 repeats this. The paper quote the AI found (trajectory step 81) is: "We compute the behavioral event triggered response by isolating the calcium events around the triggering behavioral event, then linearly interpolating onto a consistent set of 30hz timestamps."

## 2-b. How is the `neural` data processed?

i. Per trial, the `(T, n_neurons)` event matrix is linearly interpolated from the native ophys timestamps onto the trial's uniform 30 Hz grid and transposed to `(n_neurons, n_bins)`. No smoothing, normalisation, z-scoring or trial-baseline subtraction is applied. Interpolation is hand-written and vectorised across neurons; values before the first ophys sample are clamped, values after the last are linearly extrapolated (never triggered in practice — no kept trial's `stop_time` exceeds the ophys record).

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, n_src - 1)
    left  = np.clip(right - 1, 0, n_src - 1)
    same  = right == left
    t0, t1 = source_t[left], source_t[right]
    denom = np.where(np.abs(t1 - t0) < 1e-12, 1.0, t1 - t0)
    w = np.where(same, 0.0, (query_t - t0) / denom)
    out = source_values[left] * (1.0 - w[:, None]) + source_values[right] * w[:, None]
    return out.astype(np.float32)
...
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. Step 5 variable-mapping table: "Use precomputed event traces; transpose to `(n_neurons, n_timepoints)` per trial after resampling/interpolation to common 30 Hz trial grid." Justification for the resampling is decision 3 (see 2-e). The AI verified the interpolation against an independent re-derivation from the raw NWB for two sessions with `np.allclose(..., atol=1e-6) == True` (Step 10, neural sanity checks 1 and 2), and confirmed that the 2,467 "all neural data is zero" verifier warnings are genuine sparsity of the released event traces, not conversion loss (`max_abs_diff == 0.0` against the raw interpolation).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional filtering beyond what the released NWB already contains. A defensive `valid_roi` mask is applied only if the event matrix still has one column per row of the `cell_specimen_table` *and* some ROI is invalid; in the released files the traces are already restricted to valid ROIs, so this branch is effectively inert. Total 29,168 neurons, 4–666 per session.

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

iii. Step 4: "SDK defaults to `exclude_invalid_rois=True` and filters to `valid_roi` ... Whitepaper describes exclusion of non-cell ROIs, duplicate/union ROIs, and additional problematic demixed traces. **Resolution**: Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc neuron filtering beyond reference QC/filtering already reflected in the files." Step 10 verified converted per-session neuron counts equal the raw event/cell-table ROI counts for session indices 0, 98 and 198.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**: the grid begins exactly at `trials['start_time']` and steps by 1/30 s. All streams (neural, image identity, image change, running, pupil) are evaluated on that identical grid in absolute experiment (sync) time, so alignment across streams is by construction. `metadata['temporal_alignment_event'] = 'trial start'`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii.
```python
start = float(raw["trials"]["start_time"][trial_idx])
stop  = float(raw["trials"]["stop_time"][trial_idx])
grid  = session_grid(start, stop)
neural_trial  = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont  = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont    = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
image_codes   = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
image_change  = stimulus_change_codes(raw["stimulus"], grid)
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Step 10, "Temporal alignment": "Reference: synchronized timestamps across behavior/ophys streams; strategy paper interpolates activity to common 30 Hz timestamps. Conversion: all trial streams are aligned in absolute experiment time and resampled to a common 30 Hz grid from raw trial `start_time` / `stop_time`." Step 7 reviewed the `--show-processing` plots and reported "No visual sign of cross-stream temporal misalignment"; Step 10 output sanity checks 1 and 2 reproduced every output row from raw data with `np.allclose == True`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — everything is rebinned/resampled onto a fixed 30 Hz grid** (`TIME_BIN_SIZE_S = 1/30`, `time_bin_size = 33.33 ms`). The native ophys rate in the included subset ranges from 0.0323 s (≈31 Hz, single plane) to 0.0932 s (≈10.7 Hz, multiscope) — reported by the script as "native ophys dt range: 0.0323 0.0932". So single-plane data is mildly downsampled and multiscope data is ~3× **up**sampled, in both cases by linear interpolation. Resulting trial lengths are 211–377 bins.

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
"resampling_reference": "common 30 Hz grid derived from source ophys timestamps",
```

iii. Step 5 decision 3: "Use a common 30 Hz time base for all sessions: The target format requires one shared bin size across sessions, while local data mix ~31 Hz single-plane and ~11 Hz multi-plane ophys sampling. The strategy paper linearly interpolates calcium event responses onto common 30 Hz timestamps for neural and running analyses, and behavior/eye tracking are naturally 30 Hz, so 30 Hz is the most defensible common grid." Step 3 records the whitepaper figures ("31 Hz for single plane ... 11 Hz for each plane in multi-plane experiments", "eye tracking (30 Hz), and behavior (30 Hz)").

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus presentations table** of the active change-detection block, not the trials table: `start_time`, `stop_time`, `image_name`, `omitted`. The active block is picked by `choose_task_presentation_group`, which scores every `*_presentations` interval table by (a) whether `stimulus_block_name` contains `change_detection` and (b) the number of `active == True` rows.

ii.
```python
def choose_task_presentation_group(h5f):
    for name in h5f["intervals"].keys():
        if not name.endswith("_presentations") or name == "trials": continue
        grp = h5f["intervals"][name]
        if "active" not in grp or "image_name" not in grp: continue
        n_active = int(np.asarray(h5_array(grp["active"])).astype(bool).sum())
        ...
        score = (has_change_detection, n_active)
        if score > best_score: best_score, best_name = score, name
    return h5f["intervals"][best_name]
```
```python
stim = read_interval_table(stim_group, ["start_time","stop_time","image_name","is_change",
                                        "omitted","trials_id","active","flashes_since_change"])
```

iii. Step 1: "`stimulus_presentations` contains more than just the active task block in newer SDK releases; for VBO change-detection analyses, the active block is identified via `stimulus_block_name` containing `change_detection`." Step 5 mapping row: "stimulus-presentation `image_name`, `start_time`, `stop_time`, `omitted`, active task block only → `output[image_identity]`", citing `BehaviorSession.stimulus_presentations` / `get_stimulus_presentations`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A piecewise-constant categorical time series on the 30 Hz grid. For each bin, the script finds the last presentation whose `start_time` ≤ bin time and checks the bin is still before that presentation's `stop_time`; if so the bin gets that image's code, otherwise (inter-stimulus grey, or an omitted flash) it gets the code for an explicit **`gray`** class. The vocabulary is global across sessions: `["gray"] + sorted(unique non-omitted image names)` = 17 classes (gray + 16 images from image sets A and B). Resulting distribution: `gray` 0.669, each image ≈0.019–0.022.

ii.
```python
image_values = ["gray"] + sorted(image_names)   # built in pass 1
image_to_code = {name: idx for idx, name in enumerate(image_values)}

def stimulus_identity_codes(stimulus, query_t, image_to_code):
    idx = np.searchsorted(stimulus["start_time"], query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    valid = (idx >= 0) & (idx < len(stimulus["start_time"]))
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stimulus["stop_time"][idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        target = np.full(sub_idx.shape, image_to_code["gray"], dtype=np.int64)
        for i, (name, is_omitted) in enumerate(zip(stimulus["image_name"][sub_idx],
                                                   stimulus["omitted"][sub_idx])):
            if (not is_omitted) and str(name) in image_to_code:
                target[i] = image_to_code[str(name)]
        codes[np.flatnonzero(valid)[in_interval]] = target
    return codes
```

iii. Step 5 decision 6: "Use `gray` as an explicit image-identity class: Because the task includes 500 ms gray periods and omissions extend gray instead of showing an image, a `gray` category is needed for a complete time-varying identity signal." Step 3 records the stimulus cadence from the paper ("250 ms stimulus duration ... 500 ms inter-stimulus duration") and that "omissions occur only on non-change stimuli". Step 10 output sanity check reconstructed the `image_identity` row from the raw stimulus intervals for two trials with `np.allclose == True`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on the *same* `grid` array used for the neural interpolation, in absolute sync time, so the two are aligned bin-for-bin by construction; no separate alignment step exists.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes  = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
...
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 10, "Temporal alignment": all trial streams share one 30 Hz grid derived from the trial's raw `start_time`/`stop_time`, so per-stream offsets cannot arise. The AI inspected the `--show-processing` plots (`processing_775614751.png`, `processing_788490510.png`) and reported the identity trace "show[s] the expected alternation of flashed images and gray intervals" with the change markers at the go-trial change times and "no visual sign of cross-stream temporal misalignment".

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The same active stimulus-presentations table: the boolean `is_change` column together with `omitted`, `start_time` and `stop_time`. It is **not** derived from the trials table's `change_time`/`go` columns (the AI's first draft used `change_time`; see 4-c).

ii.
```python
stim["is_change"] = stim["is_change"].astype(bool)
stim["omitted"]   = stim["omitted"].astype(bool)
...
def stimulus_change_codes(stimulus, query_t):
    idx = np.searchsorted(stimulus["start_time"], query_t, side="right") - 1
    codes = np.zeros(query_t.shape, dtype=np.int64)
    valid = (idx >= 0) & (idx < len(stimulus["start_time"]))
    idx_valid = idx[valid]
    in_interval = query_t[valid] < stimulus["stop_time"][idx_valid]
    if np.any(in_interval):
        sub_idx = idx_valid[in_interval]
        changed = stimulus["is_change"][sub_idx] & (~stimulus["omitted"][sub_idx])
        codes[np.flatnonzero(valid)[in_interval]] = changed.astype(np.int64)
    return codes
```

iii. Step 5 mapping row: "stimulus-presentation `is_change`, `start_time`, `stop_time` → `output[image_change]` ... Marks the post-change flashed image itself rather than a one-bin impulse; catch trials remain 0", citing `get_stimulus_presentations` and `Trial._get_trial_timing`. Step 1 also records `compute_is_sham_change`, which is what makes catch-trial (sham) flashes carry `is_change == False`, so catch trials are automatically 0.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series: 1 for every 30 Hz bin that falls inside a presentation with `is_change == True & ~omitted`, else 0. Since a flash lasts 250 ms this gives 7–8 consecutive 1-bins per go trial and 0 everywhere in catch trials. Final distribution: 0.974 no-change / 0.026 change.

ii. See the `stimulus_change_codes` snippet in 4-a; the row is then stacked into `output_trial`:
```python
output_trial = np.vstack([image_codes.astype(np.int64), image_change,
                          running_bins, pupil_bins, trial_outcome])
```

iii. Step 6 "Code speedups / changes": "`image_change` is constructed from the stimulus table's `is_change` presentation interval instead of a single-bin impulse at `change_time`, which yields a less degenerate decoder target while staying aligned to the task structure." Step 10 "Issues Found and Resolved": "The initial `image_change` target was too sparse because it used a single-bin impulse at `change_time`. Resolution: relabeled `image_change` as the changed-image presentation window using raw stimulus `is_change` intervals; sample validation accuracy moved from below chance to above chance."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is involved — the variable is natively binary (`no_change` = 0, `change` = 1). The only "threshold" is the temporal extent assigned to the event: the AI chose the duration of the changed flash itself (250 ms ≈ 8 bins), having rejected both a single-bin impulse (too sparse, below chance) and any longer post-change window.

ii.
```python
"output_names": ["image_identity", "image_change", "running_speed_bin",
                 "pupil_diameter_bin", "trial_outcome"],
"output_values": [list(image_values), ["no_change", "change"],
                  [f"bin_{i}" for i in range(5)], [f"bin_{i}" for i in range(5)],
                  outcome_values],
```

iii. Step 12: "Output variation check: positive `image_change` bins occupy 2.5872% of all bins, so the class is sparse but not degenerate"; three positive trials from session 775614751 were re-derived from the raw data (`np.allclose == True`, 7–8 positive bins each). The choice is anchored to the paper's stimulus cadence recorded in Step 3 ("250 ms image followed by 500 ms gray screen").

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: computed on the shared per-trial 30 Hz `grid` in absolute sync time, so it is bin-aligned with the neural matrix.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Same as 3-c: the single shared grid makes cross-stream alignment automatic; verified against raw data in Step 10 (output sanity checks) and Step 12 (three positive trials).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` and `processing/running/speed/timestamps` — the SDK's default **filtered** running speed (the NWB also contains `dx` and `speed_unfiltered`, which are not used).

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. Step 5 mapping row: "Uses filtered running speed, matching SDK default", citing `BehaviorSession.running_speed` / `RunningSpeed.from_stimulus_file`. Step 1 and Step 3 record the whitepaper/SDK processing chain (unwrap encoder voltage, remove transients, drop z ≥ 10 artifacts, 10 Hz low-pass Butterworth) and note "`BehaviorSession.running_speed` is sampled on timestamps with monitor delay `0.0`, so running is aligned to sync/stimulus time without display-lag compensation."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`, constant clamping outside the sampled range) from the running timestamps onto the trial's 30 Hz grid; then discretisation into 5 bins. No extra smoothing, no absolute value, no clipping.

ii.
```python
def interpolate_vector(source_t, source_values, query_t):
    return np.interp(query_t, source_t, source_values).astype(np.float32)
...
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 5 mapping row: "Interpolate running speed onto 30 Hz trial grid; discretize with global quintile bins across included active-session timepoints." Step 3 notes the paper treats running the same way as calcium events: "running speed traces were processed in the same manner as calcium event traces ... linearly interpolating onto a common 30hz timeseries."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile (quintile) bins. The edges are the 20/40/60/80th percentiles computed **globally** in pass 1 over the interpolated running trace of every kept trial of every included session (i.e. exactly the timepoints that end up in the dataset), with a monotonicity guard that nudges tied percentiles apart by 1e-6. `np.digitize(..., right=False)` then yields labels 0–4. Verified distribution: exactly 0.200 per bin.

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
...
running_all = np.concatenate(running_values).astype(np.float32)
running_edges = robust_quintile_edges(running_all)
...
"running_speed_bin_edges": running_edges.astype(float).tolist(),
```

iii. Task instruction: "Running speed, discretized into five equal percentile bins." Step 5 decision 7: "Discretize continuous outputs globally, not per session: Running-speed and pupil-diameter bin edges will be computed from all valid included timepoints across the converted dataset so class definitions are shared across sessions." Step 10 added the edges to `metadata` "so raw-to-converted checks are reproducible", and re-derived the bin labels from raw running speed + saved edges with `np.allclose == True`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly onto the shared per-trial 30 Hz `grid`, hence bin-aligned with the neural data. No monitor-delay correction is applied (running timestamps already carry monitor delay 0.0 in the SDK).

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Step 3: "Temporal synchronization was performed by recording experimental clocks on a single NI PCI-6612 digital IO board sampled at 100 kHz" — i.e. all streams already share one clock, so interpolating onto a common grid in absolute time is sufficient. Step 7 plot review: "Running-speed and pupil traces are smooth after interpolation onto the common 30 Hz grid, with discretized bins tracking the continuous signals without obvious temporal offsets."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height` (ellipse-fit axes), timestamped by `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as the element-wise **maximum of width and height**. Blink frames are handled implicitly: in these NWB files the released `pupil_tracking` width/height are already NaN exactly on `likely_blink` frames (verified: NaN fraction == blink fraction == 0.0919 in the file inspected), and `np.maximum` propagates the NaN, so the subsequent `np.isfinite` mask is equivalent to dropping `likely_blink` rows.

ii.
```python
pupil_width  = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"],  dtype=np.float32)
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. Step 5 mapping row: "eye-tracking pupil width/height + blink mask → Compute pupil diameter as `max(width, height)`; use blink-masked values; interpolate across valid timestamps onto 30 Hz trial grid", citing `EyeTrackingTable` / `filter_on_blinks`. Step 1: "`BehaviorSession.eye_tracking` / `EyeTrackingTable` use frame timestamps plus blink/outlier filtering; blink frames are set to `NaN` for derived pupil/eye area signals."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite (blink/outlier) samples are dropped, then the remaining samples are linearly interpolated onto the trial's 30 Hz grid — so short blink gaps are bridged rather than left missing. Degenerate cases are guarded: zero valid samples raises, one valid sample yields a constant trace. The result is then discretised (6-c). No smoothing or per-session normalisation.

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

iii. Step 5 mapping row: "Small blink-related gaps are filled by interpolation after applying reference invalid-frame masking." Step 10 reconstructed the `pupil_diameter_bin` row from raw blink-masked width/height plus the saved edges for two trials, `np.allclose == True`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: global 20/40/60/80th percentiles over the interpolated pupil trace of every kept trial of every included session (computed in pass 1), monotonicity guard, `np.digitize` → labels 0–4. Verified distribution exactly 0.200 per bin. Edges saved to `metadata['pupil_diameter_bin_edges']`.

ii.
```python
pupil_all   = np.concatenate(pupil_values).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
...
"pupil_diameter_bin_edges": pupil_edges.astype(float).tolist(),
```

iii. Task instruction: "Pupil diameter, discretized into five equal percentile bins." Step 5 decision 7 (global rather than per-session edges, so class definitions are shared across sessions).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated onto the shared per-trial 30 Hz `grid`, hence bin-aligned with the neural matrix.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same justification as 5-d: one hardware-synced clock, one shared grid. Step 7's processing plots were inspected for pupil/neural offsets and none were found.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that fixed order. Resulting distribution: hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

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

iii. Step 1: `Trial._get_trial_data` "Defines behavioral trial classes and outcomes: `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`". Step 5 mapping row: "trial outcome flags (`hit`, `miss`, `false_alarm`, `correct_reject`) → `output[trial_outcome]` ... Categories: `hit`, `miss`, `false_alarm`, `correct_reject`."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single integer code 0–3 is broadcast across every 30 Hz bin of the trial, so that the static variable is stored as a time-varying row and all five outputs share the shape `(5, n_bins)`. A trial with none of the four flags set raises rather than being silently labelled (never triggered — verified that all kept go/catch trials carry exactly one outcome flag).

ii.
```python
outcome_code  = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
output_trial  = np.vstack([image_codes.astype(np.int64), image_change,
                           running_bins, pupil_bins, trial_outcome])
```

iii. Step 5 decision 5: "Represent all outputs as time-varying: To satisfy the decoder format and simplify training, even static trial outcome will be repeated across all bins in a trial." This follows the target-format guidance "Can be time-varying or discrete values per trial. If at all possible, make it time-varying."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i.
- **Sessions with no eye tracking at all** (3 active sessions: 795953296, 806456687, 833631914) are detected by a pre-scan and dropped *before both passes*, so the global quintile edges and the converted set are computed on the same subset. This was a bug fix: the first full run crashed on them.
- **Blink / invalid pupil frames** are NaN in the release; they are dropped and bridged by interpolation, so no missing-value sentinel ever reaches the output. Zero-valid and one-valid-sample sessions are handled explicitly (raise / constant).
- **Grid points outside a behavioural stream's sampled range** get `np.interp`'s constant clamping (running, pupil); for the neural matrix the hand-written interpolator clamps at the start and linearly extrapolates past the end (never exercised — no kept trial's `stop_time` exceeds the last ophys timestamp).
- **Bins outside any stimulus presentation, and omitted flashes**, fall back to the explicit `gray` class rather than an undefined code.
- **Degenerate percentile edges** are nudged apart by 1e-6 so `np.digitize` stays monotonic.
- **Sessions with < 2 valid trials** are skipped (checked before and after conversion).
- **Trials with no outcome flag** raise `ValueError` (fail loud rather than mislabel).
- **All-zero neural trials** (2,467 / 51,075 = 4.8%) are *not* treated as missing data: the AI verified against the raw NWB that they are genuine sparsity of the event traces (`np.allclose == True`, `max_abs_diff == 0.0`) and kept them.
- **Variable stimulus-table layout across releases** is handled by `choose_task_presentation_group`, which scores candidate tables instead of hard-coding a name.

ii.
```python
def has_required_eye_tracking(path):            # missing eye tracking → drop session
    ...
filtered_sessions = [s for s in sessions if has_required_eye_tracking(s.path)]

valid = np.isfinite(pupil_diameter)             # blinks / outliers
if valid.sum() == 0: raise ValueError("No valid pupil samples available")
if valid.sum() == 1: return np.full(query_t.shape, float(pupil_diameter[valid][0]), np.float32)
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)

for i in range(1, len(percentiles)):            # degenerate percentiles
    if percentiles[i] <= percentiles[i - 1]:
        percentiles[i] = percentiles[i - 1] + 1e-6

if int(raw["keep_mask"].sum()) < 2: continue    # degenerate sessions
if len(session_neural) < 2:         continue

raise ValueError(f"Trial {idx} has no valid outcome label")   # fail loud
```

iii. Step 10 "Edge-case review" and "Issues Found and Resolved": "Missing pupil data in 3 active sessions caused full-conversion failure on the first Step 9 attempt. Resolution: exclude those sessions before both passes so global bin edges and converted sessions are computed on the same valid subset." And on the zero warnings: "the warnings reflect genuine sparsity of the released calcium-event signal, not conversion corruption. They are therefore not fixable without changing the referenced neural representation."

## 9-a. What are the most time-consuming steps of the code?

i. The script prints per-session timing for both passes, so the profile is measurable from `conversion_full_out.txt` (total 364 s for 199 sessions):
- **Pass 2 (~242 s, 66%)** — dominant. Per session it re-reads the NWB *including* the full `(T, n_rois)` event matrix (the single largest array), then runs one `interpolate_matrix` per trial. Cost scales with neuron-bins; the slowest session took 3.42 s.
- **Pass 1 (~106 s, 29%)** — re-reads every NWB (trials, stimulus, running, pupil) and interpolates running + pupil per trial, purely to obtain the global image vocabulary and the two sets of quintile edges.
- **Setup (~16 s, 4%)** — `has_required_eye_tracking` opens all 284 candidate NWB files once.
- Pickling the 8.1 GB result is also non-trivial I/O.
The AI's own estimate before the full run was ~18.7 min; the actual 6.1 min beat it.

ii.
```python
t0 = time.perf_counter()
raw = read_session_raw(session, load_events=False)
...
print(f"[pass1] {idx:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{valid_trial_counts[session.ophys_experiment_id]} valid trials in {elapsed:.2f}s")
...
elapsed = time.perf_counter() - t0
total_bins = sum(trial.shape[1] for trial in session_neural)
print(f"[pass2] {sess_num:03d}/{len(sessions)} session {session.ophys_experiment_id}: "
      f"{len(session_neural)} trials, {raw['n_neurons']} neurons, {total_bins} bins in {elapsed:.2f}s")
```

iii. Step 6: "Neural interpolation is still the dominant expected cost because every kept trial needs event traces resampled onto the common grid." Step 7 tabulates the extrapolation from the 2-session sample: "Sample pass 2 (2 sessions) | 1.35 s / session on 7,557,084 neuron-bins | ~18.1 min for 2,052,381,553 neuron-bins if scaling by neuron-bin work", and correctly scaled by neuron-bins rather than by session count.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
1. **The per-timepoint Python loop inside `stimulus_identity_codes`.** For every 30 Hz bin inside a presentation it does a Python-level dict lookup — roughly 13 M iterations over the full dataset. It could be replaced by precomputing one `int` code per presentation row once per session (vectorised `np.array([...])` or a `pd.Categorical`) and then a single fancy-index `codes[assign] = presentation_codes[sub_idx]`.
2. **The per-trial interpolation loops (both passes).** Neural, running and pupil are interpolated trial by trial. Because every trial's grid is a sub-grid of the same 1/30 s lattice, the whole session could be interpolated once onto a single session-wide 30 Hz grid and then sliced per trial — turning ~250 interpolation calls per session into 1, and removing all the repeated `np.searchsorted` over the full timestamp vector.
3. **The setup scan `has_required_eye_tracking`**, which opens 284 files serially; with `multiprocessing` (or by folding the check into pass 1) this would be near-free.
The AI did vectorise the inner, most important axis — `interpolate_matrix` handles all neurons at once — and `stimulus_change_codes` is fully vectorised.

ii. The remaining scalar loop:
```python
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```
The per-trial loops:
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    grid = session_grid(start, stop)
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
    running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
    pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Step 6 lists as a deliberate speedup: "Trial-level interpolation is vectorized over neurons within each trial" and "Direct HDF5 reads avoid SDK object-construction overhead". The AI did not flag the residual per-timepoint loop or the per-trial interpolation as further opportunities; it considered the budget met once the full run finished in 6 min (Step 7 target was 15 min).

## 9-c. What processing does the code repeat multiple times?

i.
- **Each NWB file is opened up to three times**: once in `has_required_eye_tracking`, once in pass 1, once in pass 2. Trials, stimulus, running and pupil arrays are therefore parsed twice, and the stimulus-table scoring in `choose_task_presentation_group` (which itself reads every candidate `*_presentations/active` array) runs twice per session.
- **Running-speed and pupil interpolation onto the trial grids is computed twice** — once in pass 1 to pool the values for the percentile edges, then discarded, and again in pass 2 to produce the bins.
- `session_grid` is recomputed identically in both passes for every trial.
- `session_native_dt` is computed in pass 1 and again inside the plotting path.
This costs roughly the 106 s of pass 1. It is avoidable: the reference approach (accumulate the continuous per-trial values during the single loading pass, compute edges afterwards, then digitise) needs only one pass, and it would not have increased peak memory here because pass 2 already holds the entire 8.1 GB dataset in RAM before pickling.

ii.
```python
# pass 1
raw = read_session_raw(session, load_events=False)
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
pupil_values.append(interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid))
...
# pass 2 — same two calls again on the same grids
raw = read_session_raw(session)
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
pupil_cont   = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Step 6 presents the duplication as an intentional memory trade-off: "Two-pass design avoids storing all neural arrays while computing global bin edges." The AI did not revisit whether the trade-off is real given that pass 2 accumulates the full dataset in memory anyway.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Up-sampling the ~11 Hz multiscope planes to 30 Hz** manufactures ~3× more samples than the imaging actually contains, and the interpolation of sparse deconvolved event traces smears each event over intermediate bins. This inflates `converted_data.pkl` to 8.1 GB without adding information (and the decoder just consumes the extra bins).
- **Columns read and then never used**: `trials_id`, `active` and `flashes_since_change` from the stimulus table are parsed and dtype-cast in `read_session_raw` but never referenced afterwards (`active` is used only inside `choose_task_presentation_group`, from a separate read).
- **Statistics computed only to be printed**: `valid_trial_counts` and `native_dt_by_session` are returned from `collect_global_statistics` and used solely for two `print` lines.
- **Dead code**: `RNG = np.random.default_rng(0)` is never used (nothing in the conversion is stochastic); `Iterable` is imported unused.
- **Empty input arrays**: a `(0, n_bins)` float32 array is allocated and stored for each of the 51,075 trials even though `input_names == []`.
- The `--show-processing` plotting path re-derives `session_native_dt` and builds an example-trial dict that is discarded.

ii.
```python
RNG = np.random.default_rng(0)                       # never used
from typing import Dict, Iterable, ...               # Iterable never used
...
stim["trials_id"] = stim["trials_id"].astype(np.int64)             # unused downstream
stim["active"] = stim["active"].astype(bool)                       # unused downstream
stim["flashes_since_change"] = stim["flashes_since_change"].astype(np.int64)  # unused
...
return running_edges, pupil_edges, image_values, valid_trial_counts, native_dt_by_session
print("[setup] native ophys dt range:", min(native_dt.values()), max(native_dt.values()))
print("[setup] valid trial count range:", min(valid_counts.values()), max(valid_counts.values()))
...
input_trial = np.zeros((0, len(grid)), dtype=np.float32)
```

iii. The AI did not document these as waste. It did document the 30 Hz choice (Step 5 decision 3) as required by the "one shared bin size across sessions" format rule, and reported the resulting 8.1 GB artifact in Step 9 without commenting on the up-sampling cost. The unused stimulus columns appear to be leftovers from the earlier `change_time`-based `image_change` design that Step 10 replaced.
