# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file in the dataset is discovered with a single recursive glob over `/app/data` matching the release's file-naming convention `sub-<id>/sub-<id>_ses-<nn>_behavior+ophys.nwb`, sorted for determinism. Each file is opened once with `pynwb.NWBHDF5IO(..., load_namespaces=True)` inside a context manager, and all needed streams are pulled from two processing modules: `behavior/BehavioralTimeSeries` (position, speed, lick, environment, reward_zone, trial number, trial_start, teleport, Reward) and `ophys` (`Deconvolved` ROI response series, `ImageSegmentation/PlaneSegmentation` for `iscell`/`planeIdx`). There is no separate survey pass — each file is read exactly once and converted in place. The run reported 152 sessions / 11 subjects / 12,216 trials.

ii.
```python
DATA_ROOT = Path("/app/data")
...
def build_dataset() -> dict:
    session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not session_paths:
        raise RuntimeError(f"No NWB files found under {DATA_ROOT}")
```
```python
with NWBHDF5IO(str(session_path), "r", load_namespaces=True) as io:
    nwb = io.read()
    beh = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series
    seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    deconv_iface = nwb.processing["ophys"].data_interfaces["Deconvolved"]
```

iii. From the trajectory (step 18) the agent enumerated the files and confirmed the release contains 11 mice matching the paper's switch-task cohort ("I've confirmed the release contains the 11 switch-task mice from the main cohort, not the fixed-condition controls"). It first mapped the NWB layout on one file (steps 13–16) before writing the loader, and stated the NWB export "already contains curated ophys and synchronized behavior, but not a ready-made trial table".

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the parent directory name of each NWB file, stripping the `sub-` prefix. Subjects are appended to `data['subjects']` in order of first appearance while iterating the sorted file list, and `subject_idx` records the index for each session. The resulting list is `['m11','m12','m13','m14','m15','m17','m18','m19','m3','m4','m7']` (11 subjects), identical to the reference.

ii.
```python
for session_idx, session_path in enumerate(session_paths, start=1):
    subject = session_path.parent.name.replace("sub-", "")
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subjects)
        subjects.append(subject)
    ...
    data["subject_idx"].append(subject_to_idx[subject])
...
data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
```
The NWB's own `subject_id` is read but only recorded in metadata:
```python
"subject": str(nwb.subject.subject_id),
```

iii. The agent verified the directory/file naming convention up front (steps 7, 18: `Counter` over `sub-(m\d+)_ses-(\d+)`) and reported the subject/session counts before writing the converter.

## 1-c. How are the data split into sessions?

i. One session = one NWB file. Sessions are ordered by the sorted path list (so subject-major, then session number), and each session contributes one entry to `neural`, `input`, `output`, `subject_idx`, `brain_region_idx` and `metadata['session_info']`. No cross-day cell alignment is attempted; every session's neurons are treated independently.

ii.
```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
    neural_trials, input_trials, output_trials, plane_idx, session_info = _load_session(session_path)
    data["neural"].append(neural_trials)
    data["input"].append(input_trials)
    data["output"].append(output_trials)
```

iii. The agent's summary confirms the split ("152 sessions", 12–14 sessions per subject), which matches the `ses-<nn>` naming in the release. No explicit justification beyond the file layout was recorded.

## 1-d. How are the data split into trials?

i. Trials come from the framewise `trial_start` and `teleport` behavior channels, not from the stored `trial number` series. The frame indices where `trial_start > 0` are paired positionally with the frame indices where `teleport > 0`, with two edge corrections (a leading teleport before the first start is dropped; a trailing start with no teleport is dropped), and any residual excess is truncated with `min(...)`. Each trial is the half-open frame slice `[trial_start, teleport)`, i.e. the on-track lap excluding the teleport frame itself. Sessions yielding fewer than two paired trials raise an error.

ii.
```python
def _prepare_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)

    if teleports.size == starts.size + 1 and teleports[0] < starts[0]:
        teleports = teleports[1:]
    if starts.size == teleports.size + 1 and starts[-1] > teleports[-1]:
        starts = starts[:-1]

    n_pairs = min(starts.size, teleports.size)
    starts = starts[:n_pairs]
    teleports = teleports[:n_pairs]

    bounds = []
    for start, stop in zip(starts, teleports):
        if stop > start:
            bounds.append((int(start), int(stop)))
    return bounds
```
```python
bounds = _prepare_trial_bounds(trial_start, teleport)
if len(bounds) < 2:
    raise RuntimeError(f"{session_path.name}: fewer than two valid trials after trial_start/teleport pairing.")
```

iii. Step 39: "I've pinned down the right trial slicing rule: pair the `trial_start` and `teleport` event indices and keep frames from start up to, but not including, the teleport event, matching the repository's `keep_teleports=False` path." Before writing it the agent scanned every session for start/teleport count mismatches and `teleport <= start` inversions (steps 41/48/56), and checked that positions within those slices stay in `[-1, 460]` cm (step 42, "position issues count 0"). The trial-slicing decision is also recorded in metadata: `"trial_slicing": "Frames from trial_start inclusive to teleport exclusive, matching keep_teleports=False preprocessing."`

## 1-e. How are trials filtered based on quality controls?

i. Essentially no trial-level quality control is applied. The only trials discarded are degenerate pairs where `teleport <= trial_start` (zero/negative length) and any unpaired leading/trailing event. There is no minimum-duration filter, no speed/engagement filter, and no lick-artifact trial removal. All 12,216 paired trials are kept (shortest trial 96 frames ≈ 6.2 s, longest 3,359 frames).

ii.
```python
    for start, stop in zip(starts, teleports):
        if stop > start:
            bounds.append((int(start), int(stop)))
```

iii. No explicit justification was given for omitting a length filter. The agent did survey trial structure across all sessions before writing the code (steps 41–49) and found consistent 80–100 trial sessions with no pathological start/teleport pairs, so in practice nothing needed removing — the reference's `min_ntimepoints = 50` filter also removes nothing at this stage (both solutions report `T_min = 96`).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are the NWB `ophys/Deconvolved` ROI response series (Suite2p's own deconvolution of the raw fluorescence), read per imaging plane and concatenated across planes. The `Fluorescence` (F) and `Neuropil` (Fneu) arrays are **not** used; the paper's own dF/F + OASIS pipeline (`reward_relative.preprocessing.dff`) is not reproduced.

ii.
```python
deconv_iface = nwb.processing["ophys"].data_interfaces["Deconvolved"]
...
for series_name, rr in sorted(
    deconv_iface.roi_response_series.items(),
    key=lambda item: int(item[0].replace("plane", "")),
):
    plane_num = int(series_name.replace("plane", ""))
    plane_mask = plane_idx_all == plane_num
    ...
    keep_plane = iscell_all[plane_mask]
    deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
    plane_parts.append(np.full(int(np.sum(keep_plane)), plane_num, dtype=np.int64))

deconv = np.concatenate(deconv_parts, axis=1)
```
Metadata records the choice:
```python
"neural_signal": "Suite2p deconvolved activity from NWB, filtered to iscell ROIs.",
```

iii. The agent read the Methods (step 8), which describe the paper's maximin-baseline dF/F and OASIS deconvolution, and located `preprocessing.dff` in the repo (steps 22, 28). At step 32 it said it was "benchmarking two possible interneuron screens now: directly on the provided deconvolved traces versus recomputing `dF/F` with the repo's `maximin` baseline on a sample of cells. That will tell me whether the extra exclusion is worth reproducing for the full dataset." The `dff` benchmark script failed (step 34, process exited with code 1); the deconvolved-trace benchmark succeeded, and the agent proceeded with the stored `Deconvolved` array without retrying the repo pipeline or noting the failure.

## 2-b. How is the `neural` data processed?

i. Almost no processing. Per session: all ROI columns of each plane's `Deconvolved` series are read, restricted to the curated `iscell` ROIs of that plane, the planes are concatenated into one `(n_frames, n_cells)` matrix, the matrix is truncated to the common frame count shared with behavior, speed-correlated cells are dropped (2-c), and each trial slice is transposed to `(n_neurons, n_timepoints)` and cast to `float16` for storage. No neuropil subtraction, no maximin baseline, no dF/F normalization, no Gaussian smoothing, no OASIS deconvolution with `tau = 0.7`, and no per-session `keep_teleports` handling.

ii.
```python
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```
```python
if rr.data.shape[1] != int(np.sum(plane_mask)):
    raise RuntimeError(
        f"{session_path.name}: plane {plane_num} has {rr.data.shape[1]} ROIs in Deconvolved "
        f"but {int(np.sum(plane_mask))} entries in PlaneSegmentation."
    )
keep_plane = iscell_all[plane_mask]
deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
```

iii. Step 80: "The multi-plane issue is clear now: `m17` and `m18` store `plane0` and `plane1` separately, with a pooled segmentation table carrying `planeIdx`. I'm patching the loader to filter each plane against its own `iscell` slice and then concatenate them, which matches the paper's pooled-plane analysis." The agent framed the released deconvolved traces as the paper's signal ("neural data are the released deconvolved traces filtered to curated `iscell` ROIs", step 150). The `float16` cast is justified only on size grounds (step 113: "The pickle is 4.6 GB, which is in line with storing all curated neurons trial-by-trial in float16").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) ROIs are restricted to Suite2p's manually curated `iscell` column (`iscell[:,0] == 1`), applied per plane against that plane's slice of the pooled `PlaneSegmentation` table. (2) A putative-interneuron screen: Pearson correlation between each cell's trace and running speed, computed over on-track frames only, with cells at `r > 0.5` excluded. Critically, the correlation is computed on the **deconvolved** traces rather than the dF/F the paper specifies, and it excluded **0 of 138,678 cells across all 152 sessions** — the screen is a no-op as implemented (the paper reports 0.42 ± 0.85% of cells excluded; the reference solution removes ~2.5 cells/session).

ii.
```python
INTERNEURON_SPEED_CORR_THR = 0.5
...
def _speed_corr_filter(deconv, speed, bounds) -> np.ndarray:
    track_mask = np.zeros(deconv.shape[0], dtype=bool)
    for start, stop in bounds:
        track_mask[start:stop] = True

    x = deconv[track_mask]
    y = speed[track_mask].astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    ...
    corr[nonzero] = np.sum(x_centered[:, nonzero] * y_centered[:, None], axis=0) / denom[nonzero]
    return corr <= INTERNEURON_SPEED_CORR_THR
```
```python
non_interneuron = _speed_corr_filter(deconv, speed, bounds)
deconv = deconv[:, non_interneuron]
plane_idx = plane_idx[non_interneuron]
if deconv.shape[1] == 0:
    raise RuntimeError(f"{session_path.name}: no neurons remain after curation filters.")
```
Metadata is explicit that this is an approximation:
```python
"interneuron_filter": (
    "Excluded curated ROIs with deconvolved activity-speed correlation > 0.5 on on-track frames "
    "as a practical approximation to the paper's speed-correlated interneuron screen."
),
```

iii. Step 24: "The key remaining curation question is the cell mask. The NWB file still carries all ROI candidates, so I'm checking how `iscell` is encoded and whether I need to reproduce the paper's extra interneuron exclusion on top of that." The agent found the paper's `spatial.is_putative_interneuron` and the `int_thresh = 0.5` default in `dayData.py` (steps 29, 35), then measured speed correlations of the deconvolved traces on one session (step 33): "n_sampled 300 max 0.208 mean -0.002 n_gt_0.5 0". Its attempt to compare against repo-computed dF/F crashed (step 34) and was not retried; it kept the threshold at 0.5 on deconvolved traces anyway, and the converter's own summary reported "speed-correlated exclusions total: 0".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires nothing beyond the trial slicing: each trial's neural matrix starts exactly at the `trial_start` frame, so index 0 of every trial is the alignment event. Neural and behavior share the same frame index (both are sampled on the imaging clock), so the same `[start, stop)` slice is used for every stream. Metadata records `temporal_alignment_event = "trial start (entry to the 0 cm start of the virtual track)"`, `off_start = 0.0`, `off_end = None`.

ii.
```python
for trial_idx, (start, stop) in enumerate(bounds):
    ...
    time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
    neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```
```python
"temporal_alignment_event": "trial start (entry to the 0 cm start of the virtual track)",
"off_start": 0.0,
"off_end": None,
```

iii. No separate justification beyond the trial-slicing rationale (step 39). No pre-trial baseline window is included, consistent with the instruction to align on trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: data are kept at the native per-plane imaging rate. The frame rate is hardcoded as `FRAME_RATE_HZ = 15.5078125`, giving `time_bin_size = 64.4836 ms`, identical to the reference's computed value. This constant is correct for both single-plane sessions (stored NWB `rate` = 15.5078125) and the two-plane m17/m18 sessions (stored `rate` = 31.015625 is the scanner rate; the per-plane rate is 15.5078125, which is also the behavior timestamp rate). The code never cross-checks the constant against the per-file stored `rate`.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
"frame_rate_hz": FRAME_RATE_HZ,
```

iii. The paper's Methods (read at step 8) state the FOV was "collected at 1.0 magnification with unidirectional scanning at ~15.5 Hz" and that the two-plane mice were "imaged ... at ~31 Hz interleaved in the scan for a sampling rate of ~15.5 Hz per plane". The agent also printed the stored `rate` while surveying the ophys module (steps 16, 37), and used `frame_rate=15.5078125` in its dF/F benchmark at step 31.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attribute of the `position` behavior time series (all behavior series share one timestamp vector at 64.48 ms spacing).

ii.
```python
timestamps = np.asarray(beh["position"].timestamps[:], dtype=np.float64)
```

iii. The agent enumerated the behavior time series and their timestamps early (steps 15, 20) and confirmed that the behavior timestamps are the synchronized imaging-frame clock ("derives trial boundaries directly from the synchronized frame stream", step 14).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted from the trial's timestamp slice, so each trial's input row 0 starts at exactly 0 s and increases monotonically in seconds. It is stored as `float32` in row 0 of the `(4, T)` input array.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
input_trial = np.vstack([
    time_from_start,
    np.full(T, env, dtype=np.float32),
    np.full(T, tnum, dtype=np.float32),
    np.full(T, prev_reward, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. No explicit justification recorded; this is the direct reading of the instruction "Time from start of trial in seconds". The verifier reports the same range as the reference (0 to 216.5 s).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both come from the same frame index range `[start, stop)`, so they are aligned by construction. Before any slicing, every continuous stream (behavior channels, timestamps and the deconvolved matrix) is truncated to their common minimum length so that one-frame mismatches between ophys and behavior cannot shift the alignment.

ii.
```python
n_frames = min(
    timestamps.size, position.size, speed.size, lick.size, environment.size,
    reward_zone.size, trial_number.size, trial_start.size, teleport.size,
    deconv.shape[0],
)
timestamps = timestamps[:n_frames]
...
deconv = deconv[:n_frames]
```

iii. Step 85: "The second edge case is a one-frame mismatch between behavior and ophys in at least one multi-plane session. That's consistent with the repo's own 'one frame correction' logic, so I'm truncating every continuous stream in a session to the shared minimum length before building trials." This was triggered by a concrete crash at step 84 (`size of axis is 22794 but ... 22795`).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series (values 0 = ENV1, 1 = ENV2, with −1 marking invalid/inter-trial samples).

ii.
```python
environment = np.asarray(beh["environment"].data[:], dtype=np.float32)
...
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]
```

iii. Step 19: the agent verified "the exact framewise encodings for environment, reward, reward-zone, and trial boundaries so the saved labels line up with the original session objects", and at steps 48/54 confirmed the only valid values dataset-wide are 0 and 1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative (invalid) samples are discarded, the modal value over the remaining within-trial samples is taken, and that single value is broadcast across all timepoints of the trial as input row 1. Because the environment is constant within a trial, this is equivalent to using the raw framewise values. Sessions where the environment switches mid-session (day 8) therefore keep the switch at the correct trial.

ii.
```python
def _safe_mode(values: np.ndarray, fallback: int = 0) -> int:
    if values.size == 0:
        return fallback
    values = values.astype(int)
    counts = np.bincount(values)
    return int(np.argmax(counts))
...
env = float(_safe_mode(env_vals.astype(int), fallback=0))
...
np.full(T, env, dtype=np.float32),
```

iii. Step 53: "The day-8 environment transition could be within-session rather than between sessions, so I'm checking whether `environment` ever changes across trials inside a session before I lock in the per-trial labels." Step 55 confirmed on `sub-m12_ses-08` that trials 0–29 are env 0 and trials 30+ are env 1, and step 150 notes "Environment is derived per trial, so the within-session ENV1→ENV2 switch is preserved."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The stored `trial number` behavior time series, sampled at the trial's first frame, with a fallback to the loop index if that sample is negative. In this dataset the two are identical: `trial_number[trial_start_frames]` is exactly `0, 1, ..., n-1` in every session checked, and the converted range (0–99) matches the reference's loop-index range.

ii.
```python
trial_number = np.asarray(beh["trial number"].data[:], dtype=np.float32)
...
tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
```

iii. The agent inspected the `trial number` encoding early (step 27, grouping frames by `trial number` and cross-checking that each group contains exactly one `trial_start`), so it knew the stored series is a within-session sequential index with −1 for inter-trial frames.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the sample-and-fallback above; the scalar is broadcast constant across all timepoints of the trial into input row 2, as `float32`.

ii.
```python
input_trial = np.vstack([
    time_from_start,
    np.full(T, env, dtype=np.float32),
    np.full(T, tnum, dtype=np.float32),
    np.full(T, prev_reward, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. No further justification recorded; the instruction asks for a continuous per-trial trial number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavior time series — specifically its own `timestamps` (reward deliveries are sparse events with their own clock, not a framewise channel). Reward presence is computed once per trial into `reward_outcome_by_trial`, and the previous trial's entry is reused as the input.

ii.
```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
...
reward_outcome_by_trial = np.zeros(len(bounds), dtype=np.uint8)
for trial_idx, (start, stop) in enumerate(bounds):
    trial_start_t = timestamps[start]
    trial_end_t = timestamps[stop]
    reward_outcome_by_trial[trial_idx] = np.uint8(np.any((reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
```

iii. Step 19 ("verifying the exact framewise encodings for environment, reward, reward-zone ...") and step 51 ("reward outcome from sparse reward timestamps") show the agent identified that `Reward` is event-timestamped rather than framewise, and handled it in the time domain rather than the frame domain.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *k* > 0 the input is `reward_outcome_by_trial[k-1]` (1 if any reward timestamp fell in the previous trial's `[t_start, t_end)` window, else 0); for the first trial of a session it is set to 0. The value is broadcast constant across the trial as input row 3. Note the reward window used is the on-track window `[trial_start, teleport)` — a reward delivered during the teleport period would not be counted.

ii.
```python
prev_reward = float(reward_outcome_by_trial[trial_idx - 1]) if trial_idx > 0 else 0.0
...
np.full(T, prev_reward, dtype=np.float32),
```

iii. Matches the instruction "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)"; the first-trial convention (0) is not separately justified. The resulting reward-outcome base rate (84.3% rewarded) is identical to the reference's, consistent with the paper's ~15% omission rate.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the `position` behavior time series, and a per-trial reward-zone identity inferred from the `reward_zone` behavior channel. Zone identity is inferred by taking the median position of the frames where `reward_zone > 0` within a trial and snapping it to the nearest of three hardcoded zone centers (A = 80–130, B = 200–250, C = 320–370 cm, centers 105/225/345). Because the zone-entry event is missed on ~10–20% of trials, the per-trial raw labels are then regularized into two blocks: the modal label of trials 0–29 and the modal label of trials 30–end (hardcoded `SWITCH_SPLIT_TRIAL = 30`). Sessions with a single detected zone are labelled constant; if one block has no detections at all, a nearest-neighbour forward/backward fill is used instead.

ii.
```python
ZONE_STARTS_CM = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_ENDS_CM = np.array([130.0, 250.0, 370.0], dtype=np.float32)
ZONE_CENTERS_CM = (ZONE_STARTS_CM + ZONE_ENDS_CM) / 2.0
SWITCH_SPLIT_TRIAL = 30
...
def _infer_reward_zone_by_trial(position, reward_zone, bounds) -> np.ndarray:
    raw_zone = np.full(len(bounds), -1, dtype=np.int64)
    for trial_idx, (start, stop) in enumerate(bounds):
        rz_pos = position[start:stop][reward_zone[start:stop] > 0]
        if rz_pos.size:
            raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))
    ...
    unique_valid = np.unique(raw_zone[valid])
    if unique_valid.size == 1:
        return np.full(len(bounds), int(unique_valid[0]), dtype=np.int64)

    pre_valid = raw_zone[:SWITCH_SPLIT_TRIAL][raw_zone[:SWITCH_SPLIT_TRIAL] >= 0]
    post_valid = raw_zone[SWITCH_SPLIT_TRIAL:][raw_zone[SWITCH_SPLIT_TRIAL:] >= 0]

    if pre_valid.size and post_valid.size:
        inferred = np.empty(len(bounds), dtype=np.int64)
        inferred[:SWITCH_SPLIT_TRIAL] = _safe_mode(pre_valid)
        inferred[SWITCH_SPLIT_TRIAL:] = _safe_mode(post_valid, fallback=int(inferred[SWITCH_SPLIT_TRIAL - 1]))
        return inferred
```

iii. Step 51: "reward-zone block identity from `reward_zone` event positions with gap-filling for trials where the entry event was missed by interpolation." The agent quantified the missing-detection rate first (steps 43/46/49: 6–17 trials of 80 per session lack a `reward_zone` event, and sessions show 1 or 2 distinct zones), and step 150 states the "paper's 30-trial switch structure [is] used to fill trials where the entry event was missed". The A/B/C boundaries match the paper's and the repo's `behavior.py` zone definitions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to `[0, 450]` cm. Signed distance is then the displacement to the nearest edge of the trial's reward zone: `position − zone_start` when before the zone (negative), `position − zone_end` when past it (positive), and exactly 0 anywhere inside the zone. The continuous distance is immediately discretized (7-c) and only the `uint8` bin index is stored, as output row 0.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
...
def _bin_distance_to_reward_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    start = ZONE_STARTS_CM[zone_idx]
    end = ZONE_ENDS_CM[zone_idx]
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < start
    after = position_cm > end
    dist[before] = position_cm[before] - start
    dist[after] = position_cm[after] - end
```

iii. No separate narration; this is the standard "distance to any location in the reward zone" reading of the instructions (0 while inside the 50 cm zone), and matches the paper's reward-relative distance convention. The resulting class fractions match the reference to within 0.1%.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit boolean masks implementing the instruction's bins: `< −50 → 0`, `[−50, −10) → 1`, `[−10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`. The exact-zero class is tested with an exact float comparison, which is safe here because in-zone distances are assigned a literal `0.0` rather than computed.

ii.
```python
    bins = np.empty(position_cm.shape[0], dtype=np.uint8)
    bins[dist < -50.0] = 0
    bins[(dist >= -50.0) & (dist < -10.0)] = 1
    bins[(dist >= -10.0) & (dist < 0.0)] = 2
    bins[dist == 0.0] = 3
    bins[(dist > 0.0) & (dist <= 10.0)] = 4
    bins[(dist > 10.0) & (dist <= 50.0)] = 5
    bins[dist > 50.0] = 6
    return bins
```
```python
"output_values": [
    ["lt_-50cm", "-50_to_-10cm", "-10_to_0cm", "0cm", "0_to_10cm", "10_to_50cm", "gt_50cm"],
    ...
```

iii. The bin edges are transcribed directly from the Decoder Task specification. Class fractions in the converted file (0.253 / 0.102 / 0.074 / 0.237 / 0.021 / 0.072 / 0.242) essentially reproduce the reference's.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same `[start, stop)` frame slice of `position` used for the neural matrix, after the session-wide truncation to the shared frame count, so it is sample-for-sample aligned with `neural` and needs no resampling or interpolation.

ii.
```python
for trial_idx, (start, stop) in enumerate(bounds):
    pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
    ...
    neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
    T = neural_trial.shape[1]
```

iii. Behavior in this release is already resampled onto the imaging frame clock (the agent confirmed the behavior timestamp spacing equals the per-plane frame period), so shared indexing is sufficient; the only correction needed was the shared-minimum-length truncation (step 85).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (VR corridor position in cm), same array as 7-a.

ii.
```python
position = np.asarray(beh["position"].data[:], dtype=np.float32)
```

iii. Verified during the survey that within-trial positions stay within `[-1, 460]` cm across all sessions (step 42: "position issues count 0").

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial slice is clipped to `[0, 450]` cm and then discretized; no smoothing, offsetting or unwrapping. Stored as `uint8` in output row 1.

ii.
```python
TRACK_LENGTH_CM = 450.0
...
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
...
    _bin_absolute_position(pos),
```

iii. The clip is the agent's handling of the handful of samples that fall marginally outside the 450 cm track (equivalent in effect to the reference's open-ended first/last bins — the reported position class fractions are identical to the reference's to all printed digits).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins spanning the 450 cm track: `< 90 → 0`, `[90, 180) → 1`, `[180, 270) → 2`, `[270, 360) → 3`, `>= 360 → 4`.

ii.
```python
def _bin_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    bins = np.empty(position_cm.shape[0], dtype=np.uint8)
    bins[position_cm < 90.0] = 0
    bins[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    bins[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    bins[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    bins[position_cm >= 360.0] = 4
    return bins
```

iii. Transcribed from the Decoder Task specification ("5 equal-sized bins spanning the 450 cm track"); the paper's track length of 450 cm was confirmed from the Methods.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical frame slice to the neural data, after the shared-minimum truncation; no additional alignment.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
...
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. Same as 7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series (per-frame lick counts, which can exceed 1).

ii.
```python
lick = np.asarray(beh["lick"].data[:], dtype=np.float32)
```

iii. The agent inspected the lick channel's value range during the behavior survey (step 27 prints `max_lick` per trial, e.g. 4.0), so it knew the channel is a count rather than a binary flag.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized per frame with `> 0`, stored as `uint8` in output row 3. No smoothing, no conversion to lick rate, and no removal of the ~0.65% of trials the paper flags as having erroneous lick detection (the reference does not remove them either).

ii.
```python
lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
...
output_trial = np.vstack([
    _bin_distance_to_reward_zone(pos, zone_idx),
    _bin_absolute_position(pos),
    _bin_speed(spd),
    lick_bin,
    np.full(T, zone_idx, dtype=np.uint8),
    np.full(T, reward_outcome, dtype=np.uint8),
]).astype(np.uint8, copy=False)
```

iii. Follows the instruction "Lick, time-varying. 0 = no, 1 = yes". Resulting class balance (77.0% / 23.0%) is identical to the reference's.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame slice as the neural data; the lick channel is already on the imaging frame clock.

ii.
```python
lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
```

iii. Same as 7-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source and inference as 7-a: the per-trial `reward_zone > 0` frames, the corresponding `position` values, the three hardcoded zone ranges, and the block-mode regularization split at trial 30.

ii. See 7-a (`_infer_reward_zone_by_trial`).

iii. See 7-a. The agent sanity-checked the outcome via the converter's own summary — trial counts per zone came out 4186 / 4010 / 4020, i.e. close to the 1/3 balance expected from the paper's counterbalanced design.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The inferred zone index (0 = A, 1 = B, 2 = C) is broadcast constant across all timepoints of the trial into output row 4 as `uint8`, with the string labels stored in `output_values[4]` and the per-session label sequence also recorded in `metadata['session_info'][i]['reward_zone_labels']`.

ii.
```python
zone_idx = int(zone_by_trial[trial_idx])
...
    np.full(T, zone_idx, dtype=np.uint8),
...
"reward_zone_labels": [ZONE_LABELS[idx] for idx in zone_by_trial.tolist()],
```

iii. The instruction specifies a per-trial categorical `0 = A, 1 = B, 2 = C`; making it time-varying-but-constant follows the format guidance to prefer time-varying outputs.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The event timestamps of the `Reward` behavior time series, compared against the trial's start/end timestamps (the same `reward_outcome_by_trial` array used for the previous-trial input).

ii.
```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
...
    trial_start_t = timestamps[start]
    trial_end_t = timestamps[stop]
    reward_outcome_by_trial[trial_idx] = np.uint8(np.any((reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
```

iii. Step 51: "reward outcome from sparse reward timestamps". Working in the time domain avoids having to snap reward events onto frame indices.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per trial: 1 if at least one reward timestamp falls in the half-open interval `[timestamps[trial_start], timestamps[teleport])`, else 0 — broadcast constant across all timepoints of the trial as output row 5. Note `timestamps[stop]` is the teleport frame's timestamp, so the window covers the whole on-track lap up to (but not including) the teleport frame. The array is computed in a dedicated pass before the main trial loop so that trial *k* can read trial *k−1*'s value.

ii.
```python
reward_outcome_by_trial = np.zeros(len(bounds), dtype=np.uint8)
for trial_idx, (start, stop) in enumerate(bounds):
    trial_start_t = timestamps[start]
    trial_end_t = timestamps[stop]
    reward_outcome_by_trial[trial_idx] = np.uint8(np.any((reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
...
reward_outcome = int(reward_outcome_by_trial[trial_idx])
...
    np.full(T, reward_outcome, dtype=np.uint8),
```

iii. Follows the instruction's per-trial binary reward outcome. The resulting 84.28% rewarded / 15.72% omitted split is exactly the reference's, consistent with the paper's ~15% random omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, all handled inline:
- **Ophys/behavior frame-count mismatch**: every continuous stream, including the deconvolved matrix, is truncated to the minimum shared length (this covers the known one-frame offsets in multi-plane sessions). Truncation is silent — no warning is printed.
- **Multi-plane ROI bookkeeping**: the pooled `PlaneSegmentation` table is split by `planeIdx` and checked against each plane's `Deconvolved` column count; a mismatch raises `RuntimeError` rather than silently mis-indexing `iscell`.
- **Unpaired trial events**: a leading `teleport` before the first `trial_start`, and a trailing `trial_start` with no `teleport`, are dropped; leftover excess is truncated; non-positive-length pairs are skipped.
- **Missing reward-zone entry events** (6–17 trials per session): those trials get `raw_zone = -1` and are filled by the block-mode / nearest-neighbour logic in `_infer_reward_zone_by_trial`.
- **Invalid environment samples** (`-1` between trials): filtered out before taking the modal value; empty case falls back to 0.
- **Invalid `trial number` samples** (`-1`): fall back to the loop index.
- **Out-of-range position and negative speed**: clipped to `[0, 450]` cm and `>= 0` cm/s respectively.
- **NaNs in speed** during the interneuron screen: replaced with 0 via `np.nan_to_num`; a zero-variance speed vector short-circuits to "keep all cells".
- **Degenerate sessions**: fewer than two trials, or zero surviving neurons, raise `RuntimeError` (which would abort the entire run rather than skip the session; no session triggered it).

ii.
```python
n_frames = min(timestamps.size, ..., deconv.shape[0])
...
if rr.data.shape[1] != int(np.sum(plane_mask)):
    raise RuntimeError(...)
...
env_vals = env_vals[env_vals >= 0]
env = float(_safe_mode(env_vals.astype(int), fallback=0))
tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
spd = np.maximum(speed[start:stop], 0.0).astype(np.float32, copy=False)
```
```python
y = np.nan_to_num(y, nan=0.0)
...
if y_ss <= 0:
    return np.ones(deconv.shape[1], dtype=bool)
```

iii. Each of these was added in response to a concrete failure or survey finding rather than pre-emptively: step 75/80 (plane/ROI mismatch, after an `IndexError`), step 85 (one-frame mismatch, after an `IndexError`), steps 43/46/49 (quantifying missing reward-zone events before choosing the fill strategy), step 42 (checking position range). The agent's stated policy was to "let the validator tell me where the edge cases are" (step 61).

## 13-a. What are the most time-consuming steps of the code?

i. In order:
1. **Reading the `Deconvolved` matrices out of HDF5** — `np.asarray(rr.data[:, :])` materializes *all* ROI columns (up to ~5,000 ROIs × ~23,000 frames per plane) before the `iscell` mask is applied. This is the dominant, I/O-bound cost, and it is paid for every one of the 152 sessions.
2. **Pickling the 4.6 GB output file** in one `pickle.dump`, plus holding the whole converted dataset in RAM while building it.
3. **The speed-correlation screen**, which centers and reduces the full `(n_frames, n_cells)` float32 matrix per session.
4. The per-trial Python loops are comparatively negligible.

ii.
```python
deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
```
```python
with OUTPUT_PATH.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 73: "The conversion is still in the session-read phase. It has to pull the full deconvolved matrix per NWB session, so this first pass is mostly I/O-bound." Step 113 notes the 4.6 GB pickle size. Conversion of all 152 sessions took roughly 8 minutes of wall clock (steps 91–110).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Four candidates, none of them on the hot path:
- The reward-outcome pass, `for trial_idx, (start, stop) in enumerate(bounds)`, does a full `n_rewards`-length comparison per trial; a single `np.searchsorted(reward_timestamps, timestamps[starts])` vs `...[stops]` would give all trials at once.
- The per-trial zone-inference loop in `_infer_reward_zone_by_trial` (median over a boolean-masked slice per trial) could be done with a segment-id array plus grouped reductions.
- The `track_mask` construction loop in `_speed_corr_filter` could be built with `np.repeat`/index arithmetic instead of a Python loop over bounds.
- The main conversion loop calls `_bin_distance_to_reward_zone`, `_bin_absolute_position` and `_bin_speed` once per trial; position and speed binning could be applied once to the whole session array before slicing (distance binning cannot, since the zone changes per trial — though it could be done per block).
- Inside `_bin_*`, the chained boolean masks each traverse the array; `np.digitize`/`np.searchsorted` would do it in one pass.

ii.
```python
    for start, stop in bounds:
        track_mask[start:stop] = True
```
```python
    for trial_idx, (start, stop) in enumerate(bounds):
        rz_pos = position[start:stop][reward_zone[start:stop] > 0]
        if rz_pos.size:
            raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))
```

iii. No explicit trade-off was recorded; the agent's own diagnosis (step 73) was that the job is I/O-bound, which is correct — vectorizing these ~80-iteration loops would not measurably change runtime.

## 13-c. What processing does the code repeat multiple times?

i. The code is structured as a single pass per NWB file (open once, read once, convert, close), so it avoids re-reading the data. The repeats that remain are small:
- `bounds` is iterated twice — once for `reward_outcome_by_trial`, once for the main conversion — instead of one pass with a one-trial lag.
- The per-trial environment mode is computed twice: once in the conversion loop and again, from scratch, inside the `session_info["environments"]` set comprehension.
- `position[start:stop]` / `reward_zone[start:stop]` slices are taken once in `_infer_reward_zone_by_trial` and again in the conversion loop.
- Dtype conversions are applied repeatedly: HDF5 → `float32` → (`np.vstack` / `np.clip` intermediates) → `float16` for neural, and `uint8` rows are re-cast by the final `.astype(np.uint8, copy=False)` on the stacked array.

ii.
```python
    reward_outcome_by_trial = np.zeros(len(bounds), dtype=np.uint8)
    for trial_idx, (start, stop) in enumerate(bounds):
        ...
    for trial_idx, (start, stop) in enumerate(bounds):
        ...
```
```python
"environments": sorted({int(_safe_mode(environment[start:stop][environment[start:stop] >= 0].astype(int), fallback=0)) for start, stop in bounds}),
```

iii. Not discussed by the agent. The two-pass-over-trials structure is a deliberate consequence of needing trial *k−1*'s reward outcome when building trial *k*.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts:
- **Reading every ROI column** of each `Deconvolved` series into memory as `float32` before masking to `iscell` — typically 2–5× more data than is kept (e.g. 3,779 ROIs read, 1,700-ish kept). Chunked/indexed HDF5 reads would avoid it.
- **The interneuron speed-correlation screen** computes a full correlation vector for every cell in every session and then excludes nothing at all (0/138,678 cells), so its entire cost is wasted as implemented.
- **`plane_idx`** is assembled, filtered and returned, but is then used only for `plane_idx.shape[0]`; the actual plane labels are thrown away (`brain_region_idx` is all zeros).
- **`session_info`** recomputes per-trial environments and builds a per-trial `reward_zone_labels` list that the decoder never reads.
- **`print_summary`** re-walks all sessions and trials after conversion purely for the console report, and contains a no-op `env_counter.update()` call that counts nothing.
- Storing `time_from_trial_start` as an explicit per-frame row is redundant with the fixed time bin, though it is required by the target format.

ii.
```python
deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
```
```python
data["brain_region_idx"].append(np.zeros(plane_idx.shape[0], dtype=np.int64))
```
```python
    for sess_out in data["output"]:
        for trial_out in sess_out:
            env_counter.update()
```

iii. Not discussed by the agent. The `float16` storage cast goes the other way — it is an explicit (and effective) size optimization, halving the pickle to 4.6 GB (step 113), at the cost of ~3 decimal digits of precision in the neural traces.
