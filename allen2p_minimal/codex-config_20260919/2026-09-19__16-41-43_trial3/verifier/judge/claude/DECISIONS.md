# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the `VisualBehaviorOphysProjectCache` API at all. It enumerates the NWB files physically present in `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`, parses the `ophys_experiment_id` out of each filename, and intersects that set with the local `project_metadata/ophys_experiment_table.csv`. Each retained experiment is then read directly with `BehaviorOphysExperiment.from_nwb_path()`. From each NWB it pulls `events`, `ophys_timestamps`, `trials`, `stimulus_presentations`, `running_speed`, `eye_tracking` and `metadata`.

Two session-level filters are applied before anything is loaded:
- **No project-code filter.** All 284 locally present experiments are candidates, which includes 239 `VisualBehavior` (single-plane) experiments *and* 45 `VisualBehaviorMultiscope` experiments (all from one mouse, 457841).
- **Passive sessions are excluded** (`behavior_type == "active_behavior" & ~passive`), which removes the 82 `OPHYS_2_*_passive` / `OPHYS_5_*_passive` experiments.

This leaves 202 candidate experiments (168 VisualBehavior + 34 Multiscope, 38 mice); 199 survive conversion (3 dropped for unusable pupil), yielding 51,075 trials and 29,168 neurons.

ii.
```python
def _local_experiment_ids() -> set[int]:
    ids: set[int] = set()
    for path in NWB_DIR.glob("behavior_ophys_experiment_*.nwb"):
        match = re.search(r"_(\d+)\.nwb$", path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def select_experiments(max_sessions: int | None = None) -> pd.DataFrame:
    table = pd.read_csv(EXPERIMENT_TABLE, index_col="ophys_experiment_id")
    local_ids = _local_experiment_ids()
    table = table.loc[table.index.intersection(local_ids)].copy()
    active = table[table["behavior_type"].eq("active_behavior") & ~table["passive"]]
    active = active.sort_index()
    ...
```
```python
path = NWB_DIR / f"behavior_ophys_experiment_{experiment_id}.nwb"
dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
```

iii. From the code header and the agent's progress notes: *"Only active Visual Behavior experiments are used. Passive replay sessions have synthetic go/catch labels but no genuine choice/outcome."* The agent verified this empirically (step 22): for passive experiment 953659743 the trials table reports 353 go / 52 catch with **0 hits and 0 false alarms** — every go is mechanically a "miss" and every catch a "correct rejection" because the lick spout is retracted. It concluded these are not animal outcomes and would corrupt the required `trial_outcome` output. It also stated it would *"use every locally provided QC-passed active-behavior experiment (including familiar and novel active sessions)"* — i.e. it treated the supplied file set, rather than a project code, as the definition of the requested corpus. Reading NWBs directly (rather than through the S3-backed cache) was chosen because the container has a curated local subset and no network.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` strings taken from each loaded experiment's NWB `metadata`. They are sorted to give a deterministic ordering, and `subject_idx` indexes each session into that list. Result: 38 subjects (37 VisualBehavior mice + the one Multiscope mouse).

ii.
```python
"mouse_id": str(meta["mouse_id"]),
...
subjects = sorted({session["mouse_id"] for session in converted})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_lookup[session["mouse_id"]] for session in converted], dtype=np.int64
),
```

iii. Not discussed explicitly in the trajectory. `mouse_id` is the SDK's canonical animal identifier and is present both in the experiment table and in every NWB's metadata; the agent read it from the NWB so that the subject label is guaranteed to come from the same object as the neural data.

## 1-c. How are the data split into sessions?

i. **One decoder "session" = one `ophys_experiment_id` = one imaging plane.** Planes that were acquired simultaneously in the same `ophys_session_id` are *not* merged. For the 168 single-plane VisualBehavior experiments this is identical to grouping by `ophys_session_id`. For the 34 Multiscope planes it is not: those 34 planes come from only ~6 physical behavioral sessions of mouse 457841, so that one mouse contributes 34 of the 199 output sessions (every other mouse contributes 2–9), and the same behavioral trials/labels are repeated once per plane. `ophys_session_id`, `ophys_container_id` and `imaging_depth` are still recorded in `metadata['session_info']` so the grouping is recoverable.

ii.
```python
for number, experiment_id in enumerate(experiments.index, start=1):
    converted.append(convert_experiment(int(experiment_id)))
...
"session_info": {
    "ophys_experiment_id": int(experiment_id),
    "ophys_session_id": int(meta["ophys_session_id"]),
    "ophys_container_id": int(meta["ophys_container_id"]),
    ...
}
```
```python
"session_unit": "one QC-passed ophys experiment (imaging plane)",
```

iii. Code header: *"Each imaging plane (`ophys_experiment_id`) is one decoder session, matching the plane-wise decoding analysis and avoiding artificial interpolation among interleaved Multiscope planes."* Step 9: *"the key decision is the session unit: each file is one imaging plane, while simultaneous planes sharing a behavior session should not be treated as independent trials."* Step 48: *"all Multiscope planes are being retained separately with their own synchronized ophys timestamps."* The technical point is real — Multiscope planes are imaged in an interleaved fashion and therefore have different `ophys_timestamps`, so stacking them into one (n_neurons × n_timepoints) matrix would require resampling one plane onto another's clock.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. A trial is kept if `(go or catch) and not aborted and not auto_rewarded`. The trial window is the full experiment-defined window `[start_time, stop_time)` (variable length, ~7–12.5 s), which spans the pre-change flashes and the post-change response window. Each window is tiled with `floor(duration / 0.1)` consecutive 100 ms bins starting at `start_time`; the trailing partial bin (<100 ms) is discarded. Mean trial length is 84 bins (8.4 s), range 70–125 bins.

ii.
```python
trials = dataset.trials
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
trials = trials.loc[keep].copy()
```
```python
for _, row in trials.iterrows():
    duration = float(row["stop_time"] - row["start_time"])
    n_bins = int(np.floor(duration / BIN_SEC))
    if n_bins < 1:
        continue
    centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
    trial_centers.append(centers)
    outcomes.append(_trial_outcome(row))
```

iii. Code header: *"Experiment-defined go and catch trials are retained; aborted and auto-rewarded trials are excluded"* — a direct transcription of the instruction. The agent inspected the trials table (step 24) to confirm that `go`/`catch`/`aborted`/`auto_rewarded` are mutually exclusive flags and that `trial_length` is variable (7.3–11.8 s in the printed sample), then chose to keep the whole experiment-defined window rather than impose a fixed peri-change window.

## 1-e. How are trials filtered based on quality controls?

i. Beyond the go/catch/aborted/auto-rewarded selection, the filters are:
- trials shorter than one 100 ms bin are dropped (`n_bins < 1`);
- a trial whose flags do not give **exactly one** of hit/miss/false_alarm/correct_reject raises, which aborts (and therefore excludes) the whole session;
- a session with fewer than 2 retained trials — checked twice, before and after binning — raises and is excluded;
- a session with no QC-passed cells, or with a mismatch between the event matrix width and `ophys_timestamps`, raises and is excluded;
- a session with fewer than two finite pupil samples is excluded (3 sessions).

No per-trial *neural* quality control is applied: 4,925 of 51,075 trials (9.6%) contain no events at all and are retained (the validator warns "all neural data is zero" for each). There is no clipping of the trial window to the end of the ophys recording; out-of-range bins would silently reuse the last frame (this never triggers — in the eight experiments I sampled, the last trial ends ~600 s before the last ophys frame).

ii.
```python
if events.shape[1] != len(ophys_times):
    raise ValueError(f"event/timestamp length mismatch: ...")
if events.shape[0] == 0:
    raise ValueError("no QC-passed cells")
...
if len(trials) < 2:
    raise ValueError(f"only {len(trials)} retained trials")
...
if len(trial_centers) < 2:
    raise ValueError(f"only {len(trial_centers)} nonempty retained trials")
```
```python
def _trial_outcome(row: pd.Series) -> int:
    flags = np.asarray([bool(row[name]) for name in OUTCOME_NAMES])
    if flags.sum() != 1:
        raise ValueError(
            f"retained trial {row.name} does not have exactly one outcome: {flags.tolist()}"
        )
    return int(np.flatnonzero(flags)[0])
```

iii. Code header: *"NWB files and cells have already passed Allen ophys/ROI QC"* — the agent's position is that curation is already done upstream by the Allen pipeline, so the converter only enforces structural invariants and fails loudly rather than silently patching. On the all-zero trials (step 68): *"The validator emits 4,925 warnings for entirely silent trials; these are expected for sparse inferred-event traces, especially low-cell-count planes, and I'm retaining them because dropping valid Go/Catch trials would bias the outcome distribution."* The ≥2-trial rule comes straight from the instruction's "at least two trials within each session".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dataset.events["events"]` — the AllenSDK **inferred/detected calcium event** traces (L0-regularized deconvolution of the dF/F trace), one row per QC-passed cell, stacked into an (n_cells, n_frames) float32 matrix. `dff_traces` is deliberately *not* used. `dataset.ophys_timestamps` provides the time base.

ii.
```python
events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
ophys_times = np.asarray(dataset.ophys_timestamps, dtype=np.float64)
if events.shape[1] != len(ophys_times):
    raise ValueError(
        f"event/timestamp length mismatch: {events.shape[1]} != {len(ophys_times)}"
    )
```
```python
"neural_signal": "AllenSDK inferred calcium events",
```

iii. Code header: *"The neural signal is the inferred calcium-event trace, not dF/F. This avoids carrying the slow GCaMP decay into later stimulus epochs, as in the paper."* This is directly supported by the provided methods text, which the agent grepped in step 18: *"We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f"* and *"For all analysis of neural data we used the detected calcium events as described in Garrett et al."* (`/app/methods.txt` lines 179 and 208).

## 2-b. How is the `neural` data processed?

i. Essentially no signal processing is applied: no smoothing, no z-scoring, no baseline subtraction, no neuron-level normalization. The event rows of the single plane are stacked (no cross-plane merging is needed since a session is one plane), and the trial matrix is formed by **selecting, for each 100 ms bin center, the single nearest ophys sample** (`_nearest_indices`). Values are stored as float32.

Because event traces are extremely sparse (I measured 0.25% non-zero samples in experiment 1007107386) and Scientifica sessions are sampled at 31 Hz, selecting 1 of every ~3 samples discards most of the signal: in that experiment only **32% of the total event mass inside the trial windows survives**, and 49/365 trials become identically zero. Dataset-wide this produces the 4,925 all-zero trials.

ii.
```python
def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right)
```
```python
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
```

iii. Code header: *"All streams are placed on a common 100 ms grid anchored at each trial start. Neural samples are selected by nearest synchronized ophys timestamp."* The stated reason for the grid is cross-rig comparability: *"A common grid is required because Scientifica and Multiscope planes were acquired at about 31 Hz and 11 Hz, respectively. Ten Hz is still finer than the roughly 200 ms effective resolution of the inferred events."* No justification is offered anywhere for preferring nearest-sample selection over summing/averaging events within each bin.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is performed. Every cell in the NWB `events` table is kept; the AI relies entirely on the Allen pipeline's ROI/cell QC (the NWB only contains `valid_roi` cells). The only cell-related check is that the session has at least one cell. Sessions range from 4 to 666 neurons (mean 147), and 4-neuron planes are retained.

ii.
```python
if events.shape[0] == 0:
    raise ValueError("no QC-passed cells")
```
```python
"n_neurons": int(events.shape[0]),
```

iii. Code header: *"NWB files and cells have already passed Allen ophys/ROI QC."* The agent cross-checked the per-experiment cell counts against `ophys_cells_table.csv` (step 25: 202 experiments, 29,444 cells, min 4, max 666, no experiment with zero cells) before deciding that no extra filtering was warranted.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the **experiment-defined trial start** (`trials.start_time`) on the ophys clock. Bin centers are laid down at `start_time + 0.05, +0.15, ...`, and each bin takes the nearest ophys sample, so the maximum alignment error is half an ophys frame (~16 ms at 31 Hz, ~45 ms at 11 Hz). `metadata` records `temporal_alignment_event` and `off_start = 0.0`, `off_end = None` (variable trial length).

ii.
```python
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
...
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
```
```python
"temporal_alignment_event": (
    "experiment-defined trial start on the synchronized ophys clock; "
    "100 ms bin centers begin 50 ms after trial start"
),
"off_start": 0.0,
"off_end": None,
```

iii. The instruction says "Temporally align based on ophys timestamp" and "Segment each recording session into individual trials based on how they are defined in the experiment". The agent confirmed from the methods text (step 18) that all data streams are hardware-synchronized onto a single clock (*"Temporal synchronization of all data-streams ... was achieved by recording all experimental clocks on a single NI PCI-6612 digital IO board at 100 kHz"*), so it treats ophys, running and eye timestamps as directly comparable and resamples all three onto the same trial-anchored grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100 ms (`metadata['time_bin_size'] = 100.0`), uniform across every trial and every session. This *is* a rebinning: the native rates are 31 Hz (Scientifica, 32 ms) and ~11 Hz (Multiscope, ~90 ms). The rebinning is implemented as nearest-neighbour **resampling**, not aggregation — 31 Hz data is decimated ~3:1 (losing ~68% of event mass, see 2-b) and 11 Hz data is mildly upsampled (a source sample can be reused in two adjacent bins). Running speed and pupil are linearly interpolated onto the same centers, so all streams share one grid.

ii.
```python
BIN_SEC = 0.100
...
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
...
"time_bin_size": BIN_SEC * 1000.0,
"resampling": (
    "nearest ophys event sample at each 100 ms center; linear interpolation "
    "of running speed and blink-filtered pupil diameter"
),
```

iii. Code header: *"A common grid is required because Scientifica and Multiscope planes were acquired at about 31 Hz and 11 Hz, respectively. Ten Hz is still finer than the roughly 200 ms effective resolution of the inferred events."* The format requirement that "Time bins should be the same size for all trials and sessions" cannot be met with native frames once two acquisition rates are included, so a fixed grid was adopted. Step 29: *"a common 100 ms grid so the 31 Hz and 11 Hz recordings satisfy the same-bin-size requirement."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `dataset.stimulus_presentations`, using the `image_name`, `start_time` and `end_time` columns — i.e. the actual per-flash stimulus table, not the trials table. The label at each bin is the image whose 250 ms flash interval contains that bin center; bins in the inter-stimulus grey period, bins with `image_name` NaN, and bins during an **omitted** flash all get the dedicated class `"gray"` (class 0).

ii.
```python
overlap = presentations[
    (presentations["start_time"] < centers[-1] + BIN_SEC / 2)
    & (presentations["end_time"] > centers[0] - BIN_SEC / 2)
]
for _, stim in overlap.iterrows():
    name = stim["image_name"]
    if pd.notna(name) and name != "omitted":
        if name not in IMAGE_TO_CLASS:
            raise ValueError(f"unknown Visual Behavior image name {name!r}")
        on = (centers >= float(stim["start_time"])) & (
            centers < float(stim["end_time"])
        )
        image[on] = IMAGE_TO_CLASS[name]
```

iii. Code comment: *"Keep actual image flashes only. NaN is gray and 'omitted' is an expected flash deliberately replaced by continued gray."* And in the constants block: *"Class zero is the gray/no-image state, including omitted flashes (an omission is not an image)."* The agent is reading the instruction's "Image identity (of the image presented during the non-grey screen)" as licensing an explicit grey/no-image state for every non-flash bin.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Names are mapped to a **hard-coded global 17-class vocabulary**: `"gray"` plus the 8 images of set A and the 8 images of set B, in a fixed order, identical for all sessions (only 8 of the 16 images appear in any one session). An unrecognised name raises. Output dtype is uint8, stored as row 0 of the 5-row output matrix. The resulting dataset-wide distribution is 66.5% grey and ~2% for each of the 16 images.

ii.
```python
IMAGE_NAMES = [
    "gray",
    "im000", "im031", "im035", "im045", "im054", "im073", "im075", "im106",
    "im061", "im062", "im063", "im065", "im066", "im069", "im077", "im085",
]
IMAGE_TO_CLASS = {name: idx for idx, name in enumerate(IMAGE_NAMES)}
```
```python
image = np.zeros(len(centers), dtype=np.uint8)
...
"output_values": [IMAGE_NAMES, ...]
```

iii. Constants-block comment: *"Global labels make a category mean the same thing in every session. The two counterbalanced image sets are disjoint."* The agent verified the two image sets empirically in step 27 (session `OPHYS_1_images_A` → `im061…im085`; session `OPHYS_4_images_B` → `im000…im106`) before hard-coding them, and made the mapping raise rather than silently extend so that an unexpected image set fails loudly.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same 100 ms bin-center array used to select the neural samples, so it is aligned by construction. A bin is labelled with an image iff its center lies in `[flash.start_time, flash.end_time)`; `stimulus_presentations` times are already monitor-delay corrected by the SDK and live on the same synchronized clock as `ophys_timestamps`.

ii.
```python
for centers, outcome in zip(trial_centers, outcomes):
    n_bins = len(centers)
    nearest = _nearest_indices(ophys_times, centers)
    neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
    image, change = _stimulus_labels(centers, presentations)
    ...
    output_trials.append(np.vstack([image, change, run_class[...], pupil_class[...], outcome_row]))
```

iii. The agent's stated principle (code header) is that every stream is placed "on a common 100 ms grid anchored at each trial start"; once that grid exists, alignment of the stimulus labels is exact and needs no further reasoning. It relies on the hardware synchronization described in the methods text.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` boolean column of `dataset.stimulus_presentations`, together with that presentation's `start_time`. Catch (sham-change) trials have no `is_change` flash, so their image-change row stays all-zero; the trials table's `change_time`/`go` columns are not used for this output.

ii.
```python
if bool(stim.get("is_change", False)):
    idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
    if idx < len(changed):
        changed[idx] = 1
```

iii. No explicit narration in the trajectory, but the choice follows from the agent's decision to drive all stimulus labels from `stimulus_presentations` rather than the trials table; `is_change` is the SDK's own per-flash change flag and is computed as "image_name differs from the previous flash's image_name", which is exactly the instruction's "a change in image identity".

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary uint8 row, zero everywhere except for **one single 100 ms bin** — the first bin whose center is at or after the change flash onset. Dataset-wide the positive class is 1.03% of bins (roughly one bin in 97).

ii.
```python
changed = np.zeros(len(centers), dtype=np.uint8)
...
# A change is an event, not a 250 ms state: label the first common time
# bin whose center is on or after its display-lag-corrected onset.
if bool(stim.get("is_change", False)):
    idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
    if idx < len(changed):
        changed[idx] = 1
```

iii. The inline comment is the whole justification: *"A change is an event, not a 250 ms state."* The agent reads the instruction ("have value of 1 right after a change in image identity, otherwise 0") literally as a point event, and marks the single bin that follows the change onset rather than an extended post-change window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the variable is already binary. The only discretization decision is the width of the positive window (one 100 ms bin, see 4-b). `output_values` for this row is `["no_change", "change"]`.

ii.
```python
"output_values": [
    IMAGE_NAMES,
    ["no_change", "change"],
    ...
]
```

iii. Not separately discussed; it follows from the instruction "Image change, binary variable."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same grid as everything else: the positive bin is located by `np.searchsorted(centers, stim.start_time, side="left")` on the same bin-center array used for the neural samples, so the flagged bin is the first bin whose center is at or after the (monitor-delay-corrected) change onset. No response-latency offset is added.

ii.
```python
idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
if idx < len(changed):
    changed[idx] = 1
```

iii. As in 3-c, alignment is a consequence of the single common grid; the code comment notes the onset is the *"display-lag-corrected"* time, i.e. the agent checked that `stimulus_presentations.start_time` is already corrected for monitor delay by the SDK.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, using its `timestamps` and `speed` columns (the SDK's filtered running-disk speed in cm/s, ~60 Hz).

ii.
```python
running = dataset.running_speed
...
run_values = _interp_valid(
    running["timestamps"].to_numpy(dtype=np.float64),
    running["speed"].to_numpy(dtype=np.float64),
    all_centers,
)
```

iii. Not narrated; `running_speed` is the SDK's canonical locomotion accessor and the agent confirmed in step 17 that the NWB contains *"60 Hz running"* data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite samples are dropped, the remainder is sorted by time and **linearly interpolated** (`np.interp`) onto the concatenated bin centers of all retained trials in the session. The interpolated values are then converted to 5 classes using **that session's own** 20/40/60/80th percentiles (see 5-c). No smoothing, no absolute-value or sign handling, and no cross-session pooling.

ii.
```python
def _interp_valid(times, values, targets):
    valid = np.isfinite(times) & np.isfinite(values)
    if valid.sum() < 2:
        raise ValueError("fewer than two finite samples available for interpolation")
    x = times[valid]; y = values[valid]
    order = np.argsort(x, kind="stable")
    x = x[order]; y = y[order]
    return np.interp(targets, x, y).astype(np.float32, copy=False)
```
```python
all_centers = np.concatenate(trial_centers)
run_values = _interp_valid(running["timestamps"]..., running["speed"]..., all_centers)
run_class = _percentile_classes(run_values)
```

iii. Code header: running speed is *"linearly interpolated to that same grid"*. Code comment on the extrapolation behaviour: *"np.interp uses the nearest endpoint outside the measured range. Trial samples normally lie inside the behavior/eye recordings; endpoint use is retained as a safe treatment of a few synchronization-edge samples."* The per-session scope of the percentiles is recorded in metadata as *"separately within each session over retained trial bins"* but is not otherwise argued for in the trajectory.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five classes from the empirical quantiles at 0.2/0.4/0.6/0.8, computed over the retained trial bins of **each session separately**; `np.searchsorted(..., side="right")` assigns the class. Because the cuts are per-session, each session is exactly balanced at 20% per class, and so is the pooled dataset (measured fractions 0.2000/0.1999/0.2000/0.1999/0.2000). Class names are `["0-20%", ..., "80-100%"]`.

ii.
```python
def _percentile_classes(values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite value remained before percentile binning")
    cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    # side='right' gives classes 0..4 and a deterministic treatment of ties.
    return np.searchsorted(cuts, values, side="right").astype(np.uint8)
```

iii. The instruction asks for "five equal percentile bins"; the agent's smoke-test note (step 32) says *"the percentile outputs are exactly balanced within sessions"*, indicating that exact within-session balance was the property it was optimizing for. The `side="right"` choice is justified inline as giving "a deterministic treatment of ties".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The interpolation targets *are* the neural bin centers, so alignment is exact by construction. Interpolation is done once for the whole session over the concatenated centers of all trials, then sliced back into trials with a running `cursor`.

ii.
```python
cursor = 0
for centers, outcome in zip(trial_centers, outcomes):
    n_bins = len(centers)
    ...
    output_trials.append(np.vstack([
        image, change,
        run_class[cursor : cursor + n_bins],
        pupil_class[cursor : cursor + n_bins],
        outcome_row,
    ]).astype(np.uint8, copy=False))
    cursor += n_bins
```

iii. Same justification as 2-d/3-c: all clocks are hardware-synchronized per the methods text, so linear interpolation of the 60 Hz running trace onto the ophys-anchored 100 ms grid is valid and needs no further correction.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking`, column `pupil_area`, converted to a diameter as `2 * sqrt(area / pi)`. The session is skipped outright if `eye_tracking` is None. Blink handling is implicit: the SDK already sets `pupil_area` to NaN on `likely_blink` frames (I confirmed this — in experiment 1007107386, the NaN fraction of `pupil_area` is 0.0919, exactly equal to the `likely_blink` fraction), and `_interp_valid` drops all non-finite samples before interpolating. `pupil_width` is not used.

ii.
```python
eye = dataset.eye_tracking
if eye is None:
    raise ValueError("eye tracking is unavailable")

# SDK pupil_area is pi * max(ellipse radius)^2 after blink/outlier removal;
# converting it to a diameter is monotonic (and therefore preserves the
# requested percentile classes).
pupil_area = eye["pupil_area"].to_numpy(dtype=np.float64)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The agent grepped the SDK's `eye_tracking_processing.py` in step 19 specifically to settle *"how the SDK defines pupil size"* (step 17), and recorded the finding in the code comment: `pupil_area` is the blink/outlier-filtered fitted-ellipse area, and area→diameter is a monotonic transform. The instruction asks for "pupil diameter", so it converts rather than reporting area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: drop non-finite (= blink) samples, sort, linearly interpolate onto the session's concatenated bin centers (`np.interp`, endpoint-held outside the measured range — which also means blink gaps are bridged by a straight line with no maximum-gap limit), then bin into 5 per-session percentile classes.

ii.
```python
pupil_values = _interp_valid(
    eye["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    all_centers,
)
pupil_class = _percentile_classes(pupil_values)
```

iii. Same as 5-b — one shared `_interp_valid` / `_percentile_classes` path for both behavioural outputs, described in the code header as *"Running speed and blink-filtered pupil diameter are linearly interpolated to that same grid."*

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The same `_percentile_classes` helper: per-session 0.2/0.4/0.6/0.8 quantiles, `searchsorted(side="right")`, classes 0–4 named `["0-20%", ..., "80-100%"]`. Measured pooled fractions are 0.20 each. Note this means the classes are *relative to each session's own pupil distribution*, not absolute pupil sizes.

ii.
```python
pupil_class = _percentile_classes(pupil_values)
...
"percentile_scope": "separately within each session over retained trial bins",
```

iii. Only the metadata string documents the scope; the agent did not argue for per-session vs. global binning in the trajectory beyond noting that it makes the outputs "exactly balanced within sessions" (step 32).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated directly onto the neural bin centers, then sliced per trial with the shared `cursor`.

ii.
```python
pupil_values = _interp_valid(eye["timestamps"]..., pupil_diameter, all_centers)
pupil_class = _percentile_classes(pupil_values)
...
pupil_class[cursor : cursor + n_bins],
```

iii. As in 5-d: the eye camera clock is hardware-synchronized to the ophys clock (methods text), so resampling onto the ophys-anchored grid is sufficient.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — in that fixed order.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
def _trial_outcome(row: pd.Series) -> int:
    flags = np.asarray([bool(row[name]) for name in OUTCOME_NAMES])
    if flags.sum() != 1:
        raise ValueError(...)
    return int(np.flatnonzero(flags)[0])
```

iii. Step 17: the agent explicitly set out to resolve *"whether 'trial outcome' is encoded as four behavioral outcomes or as the two Go/Catch correctness labels"*, then verified in step 24 on a real session that go/catch trials partition cleanly into hit/miss/false_alarm/correct_reject. This four-way signal-detection labelling is also the convention used in the paper. The same check is what motivated dropping passive sessions (step 22), where the four labels degenerate.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome index (0–3) is computed once per trial and then **broadcast across every time bin of that trial** as row 4 of the output matrix, so the static per-trial variable is stored in the same time-varying (5, n_bins) array as the other four outputs. Measured distribution: 30.3% hit, 57.1% miss, 1.7% false alarm, 10.8% correct reject.

ii.
```python
outcome_row = np.full(n_bins, outcome, dtype=np.uint8)
output_trials.append(
    np.vstack([image, change, run_class[...], pupil_class[...], outcome_row]).astype(np.uint8, copy=False)
)
```
```python
"output_names": [
    "image_identity", "image_change", "running_speed_percentile",
    "pupil_diameter_percentile", "trial_outcome",
],
```

iii. The target format requires all outputs of a trial to share one array, and states "If at all possible, make it time-varying"; replicating the constant across bins is the standard way to satisfy both. The strict "exactly one flag" assertion is the agent's guard against the degenerate label combinations it found in passive sessions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The policy is **fail loudly, skip the session, and record why** — nothing is imputed:
- Any exception during a session's conversion is caught in `build_dataset`, the session is dropped, and `{ophys_experiment_id, reason}` is appended to `metadata['skipped_sessions']`. Exactly 3 of 202 sessions were dropped, all with `"ValueError: fewer than two finite samples available for interpolation"` (no usable pupil at all).
- Missing eye tracking → session skipped rather than pupil imputed.
- Blink frames (NaN pupil) are dropped and bridged by linear interpolation; running/pupil samples outside the measured range are held at the nearest endpoint by `np.interp`.
- `_percentile_classes` re-asserts that nothing non-finite survived into the labels.
- Structural mismatches (events vs. timestamps length, zero cells, <2 trials, ambiguous outcome flags) all raise.
- All-zero neural trials are *not* treated as missing data and are kept (see 1-e).

Residual weaknesses: linear interpolation bridges arbitrarily long blink/tracking dropouts with no maximum-gap check, and a trial window extending past the last ophys frame would silently repeat the final frame rather than being clipped (this never occurs in this dataset).

ii.
```python
try:
    converted.append(convert_experiment(int(experiment_id)))
except Exception as exc:
    # Missing pupil data prevents constructing a required decoder output;
    # such a session is unusable rather than safely imputable.  Preserve
    # the exact exclusion and reason in metadata.
    print(f"  skipped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    skipped.append({
        "ophys_experiment_id": int(experiment_id),
        "reason": f"{type(exc).__name__}: {exc}",
    })
```
```python
"skipped_sessions": skipped,
```

iii. Step 32: *"any session lacking usable pupil data will be explicitly skipped and recorded in metadata rather than silently imputed."* Step 39: *"Three early Scientifica experiments have all blink-filtered pupil measurements invalid and are being excluded, as required to avoid inventing a pupil target."* The agent's stated principle is that a fabricated value for a *decoder target* is worse than a missing session, and that every exclusion must be auditable from the saved artifact.

## 9-a. What are the most time-consuming steps of the code?

i. Dominated by I/O and SDK object construction: `BehaviorOphysExperiment.from_nwb_path()` on each of the 202 NWB files (230–240 MB each, ~50 GB total), plus the SDK's lazy construction of `events`, `stimulus_presentations` (which re-runs `is_change`/omission processing) and `eye_tracking`. The whole run took ~15 minutes wall clock (20:55→21:10). Secondary costs: `np.vstack` of the full (n_cells × ~140k) event matrix for every session, the per-trial Python loops (~250–370 trials/session × 199 sessions), and finally pickling a 2.53 GiB artifact.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
...
del dataset, events
gc.collect()
```

iii. The agent budgeted for this up front: in step 25 it estimated the full-resolution footprint (*"approx floats 2944400000 GB 11.7776"*), and in step 23 it checked free memory/disk before committing. It ran the conversion as a background process and polled it, and added an explicit `del` + `gc.collect()` per session so only trial slices accumulate.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops are vectorizable:
- `_stimulus_labels` iterates `overlap.iterrows()` (~10–15 flashes per trial, ~300 trials, ~199 sessions ≈ 10^6 row-object constructions) and builds a full boolean mask `(centers >= start) & (centers < end)` over the whole trial for *each* flash. Since presentations are sorted and non-overlapping, the entire session's image labels could be produced with a single `np.searchsorted(presentation_start_times, all_centers)` lookup, and all change bins with one vectorized `searchsorted`.
- The first `trials.iterrows()` loop that builds bin centers and outcomes; outcomes could be read as `np.argmax(trials[OUTCOME_NAMES].to_numpy(), axis=1)` in one shot, and centers built with a single ragged construction.
- `_nearest_indices` is called once per trial rather than once per session on `all_centers` (the running/pupil interpolation already uses the efficient concatenated form — the neural path does not).

ii.
```python
for _, stim in overlap.iterrows():
    ...
    on = (centers >= float(stim["start_time"])) & (centers < float(stim["end_time"]))
    image[on] = IMAGE_TO_CLASS[name]
```
```python
for centers, outcome in zip(trial_centers, outcomes):
    nearest = _nearest_indices(ophys_times, centers)   # per trial, not per session
```

iii. Not discussed by the agent. It is a defensible non-priority: runtime is dominated by NWB reading, so vectorizing these loops would not have changed total wall clock much.

## 9-c. What processing does the code repeat multiple times?

i. - The overlap mask `presentations["start_time"] < ... & presentations["end_time"] > ...` is recomputed over the **entire** stimulus-presentation table (~4,800 rows) once per trial, i.e. ~300× per session, to find the ~12 relevant rows.
- `_nearest_indices` re-does a `searchsorted` over the full `ophys_timestamps` array per trial instead of once per session.
- The trials table is iterated twice (once to build centers/outcomes, once to build the output rows).
- `n_bins`/`len(centers)` and the uint8 casts are recomputed at several points.
- `gc.collect()` is invoked once per session.
- The `mouse_id`/`n_neurons` fields are stored both at the top level of the per-session dict and again inside `session_info`.

ii.
```python
overlap = presentations[
    (presentations["start_time"] < centers[-1] + BIN_SEC / 2)
    & (presentations["end_time"] > centers[0] - BIN_SEC / 2)
]
```
```python
"mouse_id": str(meta["mouse_id"]),
...
"session_info": {..., "mouse_id": str(meta["mouse_id"]), "n_neurons": int(events.shape[0]), ...},
```

iii. Not discussed. Each repetition is cheap relative to NWB loading, and the duplicated metadata fields are deliberate convenience for the assembly step.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. - **The area→diameter conversion is a no-op for the saved output.** `2*sqrt(area/pi)` is strictly monotonic and the value is immediately replaced by a percentile class, so the sqrt over ~136k samples per session changes nothing. The code's own comment admits the transform is monotonic and "therefore preserves the requested percentile classes".
- **~68% of the loaded event data is read, stacked and cast to float32, then thrown away** by the nearest-sample decimation (31 Hz → 10 Hz). Building the full (n_cells × 140k) matrix before subsampling is the single largest piece of wasted work.
- The static trial outcome is materialized as `n_bins` identical uint8 values per trial (~4.3M redundant values dataset-wide) — required by the target format, but redundant as data.
- Empty `(0, n_bins)` input arrays are allocated and pickled for all 51,075 trials even though `n_input = 0`.
- `metadata['session_info']` stores 14 fields × 199 sessions that the decoder never reads.
- `_percentile_classes` re-validates finiteness that `_interp_valid` already guarantees.

ii.
```python
# ...converting it to a diameter is monotonic (and therefore preserves the
# requested percentile classes).
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```
```python
input_trials.append(np.empty((0, n_bins), dtype=np.float32))
```
```python
outcome_row = np.full(n_bins, outcome, dtype=np.uint8)
```

iii. The agent's justification for the pupil conversion is interpretability rather than necessity — the instruction asks for "pupil diameter", so it reports a diameter-derived quantity even though the discretization makes it equivalent to area. The `(0, n_bins)` inputs are a deliberate shape-consistency choice ("No inputs for this task"), and the verbose `session_info` was kept for auditability of which experiments were included and why others were skipped.
