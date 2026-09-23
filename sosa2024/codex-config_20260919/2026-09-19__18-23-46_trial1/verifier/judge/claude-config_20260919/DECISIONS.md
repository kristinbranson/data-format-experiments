# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All `*.nwb` files below `/app/data/sub-m*/` are globbed (152 files), sorted by numeric `(mouse, session)` parsed from the filename, and opened **directly with `h5py`** rather than `pynwb`. Each file is one session and is read in a single pass. Fixed HDF5 paths are used for behavior (`processing/behavior/BehavioralTimeSeries`) and imaging (`processing/ophys`). All seven frame-aligned behavior streams (`position`, `speed`, `lick`, `environment`, `reward_zone`, `trial_start`, `teleport`) plus the sparse `Reward` timestamps are read whole, and each deconvolved imaging plane is read densely once. `--sample` restricts to two hard-coded switch sessions (m11 ses-03, m12 ses-03); `--full` (default) processes all 152.

ii.
```python
DATA_ROOT = Path("/app/data")
BEHAVIOR = "processing/behavior/BehavioralTimeSeries"
OPHYS = "processing/ophys"

def natural_key(path: str | Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse subject/session from {path}")
    return int(match.group(1)), int(match.group(2))

def discover_files(sample: bool) -> list[Path]:
    files = [Path(p) for p in glob.glob(str(DATA_ROOT / "sub-m*" / "*.nwb"))]
    files.sort(key=natural_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    if sample:
        wanted = {(11, 3), (12, 3)}
        files = [p for p in files if natural_key(p) in wanted]
    return files

def read_series(nwb: h5py.File, name: str) -> tuple[np.ndarray, np.ndarray]:
    group = nwb[f"{BEHAVIOR}/{name}"]
    return group["data"][()], group["timestamps"][()]

with h5py.File(path, "r") as nwb:
    position, timestamps = read_series(nwb, "position")
    speed, speed_time = read_series(nwb, "speed")
    ...
```

iii. From CONVERSION_NOTES Step 2: "The supplied DANDI 001361 dataset contains `dandiset.yaml` plus 152 NWB 2.8 files in `data/sub-m*/sub-m*_ses-*_behavior+ophys.nwb` (92.45 GB). There are 11 mice. Ten have 14 sessions; m11 has sessions 03–14 (12 sessions)." The agent audited the HDF5 layout first and confirmed every required object path exists in every file. Raw h5py was chosen for speed: "Each dense deconvolved plane is read once, its local cell mask is applied in memory, and curated planes fill one preallocated time×cell matrix." A Step-10 audit independently re-opened all 152 NWBs and matched the aggregate counts (12,216 native trials, 312,110 raw ROIs, 138,678 curated cells, 10,342 rewarded trials).

## 1-b. How are the data split into subjects?

i. The subject is parsed from the file name (`sub-m<N>`). The `subjects` list is the sorted-by-number set of mouse ids over the files actually processed, and `subject_idx` maps each session to its index in that list. 11 subjects result: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19.

ii.
```python
subjects = sorted({f"m{natural_key(p)[0]}" for p in files}, key=lambda x: int(x[1:]))
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[info["subject"]])
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. "Subjects | 11: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19" (Step 2), cross-checked against the paper's "counterbalanced across mice (n = 11 mice)". The agent noted the directory name, the file name and the NWB `subject_id` all agree.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are ordered by `(mouse number, session number)`; the session number is the experiment day parsed from `ses-<NN>`. No cross-session ROI alignment/tracking is attempted; each session's cells are treated as independent.

ii.
```python
files.sort(key=natural_key)   # (mouse, session)
...
subject_number, session_number = natural_key(path)
session_id = f"m{subject_number}_ses-{session_number:02d}"
...
for index, path in enumerate(files):
    converted = convert_session(path, show_processing and index < 2)
    session_neural, session_input, session_output, info, stats = converted
    neural.append(session_neural)
```

iii. Step 2/Step 4: "each target session is one NWB". The count (152 = 14×10 + 12) was checked against the paper's "total of 14 days" and the note that m11's imaging began on day 3, which exactly explains its 12 sessions and the 160-trial difference from the paper's cited 12,376 trials.

## 1-d. How are the data split into trials?

i. A trial is the on-track lap `[trial_start_frame, teleport_frame)` in native zero-based NWB frames. Starts are the nonzero samples of the `trial_start` stream, ends are the nonzero samples of `teleport`; the code asserts equal counts and `stop > start` for every pair. The teleport frame itself and the whole ITI/jitter period are excluded. The reference repo's `start-1:stop-1` slicing was deliberately *not* copied, because that offset belongs to the old pickle index convention, not to the NWB export.

ii.
```python
starts = np.flatnonzero(trial_start > 0).astype(np.int64)
stops = np.flatnonzero(teleport > 0).astype(np.int64)
if starts.size != stops.size or np.any(stops <= starts):
    raise ValueError(f"{session_id}: invalid trial start/teleport pairs")
...
for trial, (s, e) in enumerate(zip(starts, stops)):
    ...
    n_time = int(e - s)
    neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. Step 4 discrepancy table: "Uses `trial_start_inds`/`teleport_inds`; historical session arrays are sliced `start-1:stop-1` because those indices came from one-index-adjusted TwoPUtils objects … Use native NWB zero-based `[start:teleport)` slicing. Applying the pickle-specific `-1` adjustment to NWB would be an off-by-one error." Step 2 adds: "All trial starts and teleports are paired. On-track samples run from the nonzero `trial_start` frame through immediately before the paired `teleport`; trial-number transitions independently agree." 12,216 native trials result (mean 80.37/session), matching the paper's "80.5 ± 7.4 trials".

## 1-e. How are trials filtered based on quality controls?

i. One trial-level QC rule: the paper's lick-sensor artifact criterion. A trial is dropped if **more than 30% of its on-track frames have a cumulative lick count > 2**. This removed exactly **81 of 12,216** trials, reproducing the paper's reported 81/12,376. The removed trial is dropped from `neural`, `input` and `output` together. Additionally, a session must retain ≥2 trials (all do). There is **no** minimum-trial-length filter and no speed-based sample exclusion.

ii.
```python
lick_artifact_fraction = np.array(
    [np.mean(lick[s:e] > 2) for s, e in zip(starts, stops)], dtype=np.float64
)
keep = lick_artifact_fraction <= 0.30
...
for trial, (s, e) in enumerate(zip(starts, stops)):
    if not keep[trial]:
        continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than two valid trials")
```

iii. Methods: "These trials were detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2 … (~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice)." Step 4 resolves a code/paper conflict in favour of the paper: "`glmUtils.get_timeseries_data` implements >35% despite a stale '>50%' comment … >30%=81 trials, >35%=69, >50%=44 … The supplied data reproduce the paper's 81 rejected trials exactly, decisively resolving code-version inconsistency." Step 5 Key Decision 3 explains the whole-trial removal: "Keeping them with an invented lick value would contaminate one required output, while masking only lick is unsupported by the target structure." The paper's slow-sample exclusion (<2 cm/s) was intentionally not applied because speed class 0 is a requested output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are the NWB `processing/ophys/Deconvolved/plane*/data` arrays (time × ROI, float32), subset to curated cells by `ImageSegmentation/PlaneSegmentation/iscell[:,0] > 0.5` and pooled across planes via `planeIdx`. The raw `Fluorescence` and `Neuropil` traces are read by no part of the pipeline — the agent explicitly declared them "not needed because author events are present".

ii.
```python
def load_curated_events(nwb, n_behavior_frames):
    segmentation = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation"]
    iscell = segmentation["iscell"][:, 0] > 0.5
    plane_idx = segmentation["planeIdx"][()].astype(np.int64)
    plane_groups = nwb[f"{OPHYS}/Deconvolved"]
    plane_names = sorted(plane_groups, key=lambda x: int(x.replace("plane", "")))
    curated_total = int(np.sum(iscell))
    pooled = np.empty((n_behavior_frames, curated_total), dtype=np.float32)
    cursor = 0
    for plane_name in plane_names:
        plane = int(plane_name.replace("plane", ""))
        local_mask = iscell[plane_idx == plane]
        dataset = plane_groups[f"{plane_name}/data"]
        dense = dataset[:n_behavior_frames, :]
        selected = np.asarray(dense[:, local_mask], dtype=np.float32)
        pooled[:, cursor : cursor + selected.shape[1]] = selected
        cursor += selected.shape[1]
    return pooled, int(iscell.size), plane_info
```

iii. Step 1 Notes: "Native data are already author-processed `multi_anim_sess` pickles, so dF/F does **not** need to be recomputed. The target neural stream should use the stored `sess.timeseries['events']`, matching the paper decoder and avoiding an incompatible second deconvolution." Step 2 repeats: "`processing/ophys/Deconvolved/plane{0,1}/data`: time x ROI float32 author-computed deconvolved calcium events." Step 5 mapping note: "Raw author-computed events, float32, no extra normalization/deconvolution."

## 2-b. How is the `neural` data processed?

i. Essentially none. The stored `Deconvolved` values are cast to float32, curated columns are pooled across planes into one `(T_behavior, n_cells)` matrix, truncated to the number of behavior frames, transposed per trial to `(n_cells, n_time)` and made contiguous. No dF/F, no neuropil subtraction, no per-trial maximin baseline, no Gaussian smoothing, no OASIS deconvolution, no normalisation or z-scoring is applied. A finiteness check is run over the pooled matrix.

ii.
```python
if cursor != curated_total or not np.isfinite(pooled).all():
    raise ValueError("Curated event matrix failed shape/finite validation")
...
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. Step 5 Variable Mapping: "For each plane select columns whose matching `iscell[:,0]>0.5`, transpose time×cell to cell×time, concatenate planes, and slice native zero-based `[trial_start:teleport)` … Raw author-computed events, float32, no extra normalization/deconvolution." Step 10 check 6: "No raw dF/F recomputation is performed because author-deconvolved events are supplied." The agent did read and summarise the paper's `dff` function ("subtract 0.7 neuropil, compute per-trial maximin baseline (15-frame smoothing; 300-frame min/max filters), calculate dF/F, smooth by 2 frames, and optionally OASIS-deconvolve into `events`") but concluded it was already applied upstream.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the manual Suite2p curation flag is applied: cells with `iscell[:,0] > 0.5` are kept (138,678 of 312,110 ROI-session entries). Both planes of the 28 dual-plane m17/m18 sessions are pooled. The paper's second neuron filter — dropping putative interneurons whose dF/F correlates with running speed at r > 0.5 — is deliberately **not** applied. No place-cell selection is applied either.

ii.
```python
iscell = segmentation["iscell"][:, 0] > 0.5
plane_idx = segmentation["planeIdx"][()].astype(np.int64)
...
local_mask = iscell[plane_idx == plane]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
...
if dataset.shape[1] != local_mask.size:
    raise ValueError(
        f"{plane_name}: {dataset.shape[1]} event ROIs != {local_mask.size} segmentation ROIs"
    )
```

iii. Step 3 Curation Steps: "Apply the supplied manual Suite2p `iscell[:,0]` mask and pool planes, matching the paper. The paper additionally excluded speed-correlated putative interneurons (Pearson r>0.5, 0.42±0.85%) from specific downstream place-cell analyses. That analysis-specific filter is not applied to a general neural-to-behavior decoder: the NWB lacks stored dF/F, the target is not restricted to pyramidal place-cell analyses, and excluding neurons based on correlation with the speed output would directly select against a requested prediction." Step 4 also notes and dismisses a statistic mismatch: iscell yields 155–2,341 cells/session versus the paper's stated 155–2,172, attributed to "updated NWB curation/export".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires nothing beyond slicing: the neural matrix shares the behavior frame axis, so `neural_by_time[s:e]` starts exactly at the `trial_start` frame. `off_start = 0.0` and `off_end = None` (variable lap duration). Neural, input and output slices use the identical `[s:e)` index range, and each trial's shapes are asserted to match.

ii.
```python
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
inputs = np.vstack((np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32), ...))
outputs = np.vstack((discretize_distance(distance), ...))
if neural.shape[1] != n_time or inputs.shape != (4, n_time) or outputs.shape != (6, n_time):
    raise ValueError(f"{session_id} trial {trial}: inconsistent converted shapes")
...
"temporal_alignment_event": "start of trial (entry onto the 450-cm virtual track)",
"off_start": 0.0,
"off_end": None,
```

iii. Step 4: "Both use paired trial-start/teleport intervals. The NWB audit confirms zero-based `[start,teleport)` is correct … Exact neural/input/output trial checks rule out a one-frame shift." Step 2 confirmed neural and behavior streams are already frame-synchronised ("their behavioral and per-plane sample interval is still 64.483627 ms"); ten dual-plane files have one extra trailing neural row, which is outside all trials and is truncated.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging/behavior bin of **64.483627 ms (15.5078125 Hz)** is kept for every session and trial; no rebinning, resampling, padding or spatial binning is performed. Trial lengths therefore vary (min 96, max 3,359 samples). The code hard-codes the expected interval and asserts that every session's behavior timestamps are uniform and match it, which also implicitly handles the dual-plane files whose *scanner* rate metadata says 31.015625 Hz (i.e. 15.5 Hz per plane).

ii.
```python
EXPECTED_DT_S = 1.0 / 15.5078125
...
dt = np.diff(timestamps)
if not np.allclose(dt, EXPECTED_DT_S, rtol=1e-6, atol=1e-9):
    raise ValueError(f"{session_id}: nonuniform or unexpected sample interval")
...
"time_bin_size": EXPECTED_DT_S * 1000.0,
```

iii. Step 4: "Single-plane ~15.5 Hz; dual scan metadata can be ~31 Hz before division by `n_planes` … Trust synchronized behavioral timestamps/per-plane description: common bin size 64.483627 ms." Step 5 Key Decision 1: "Preserve every synchronized imaging frame and variable `[start,teleport)` trial length. This meets common-bin-size requirements while retaining timing, speed, and licks; padding/resampling would add fabricated observations." Step 3 grounds this in the Methods' "0.0645 s imaging frame samples".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attached to the behavior `position` time series. Before use, the timestamps of `speed`, `lick`, `environment`, `reward_zone`, `trial_start` and `teleport` are all checked to be identical to them, so the choice of stream is immaterial.

ii.
```python
position, timestamps = read_series(nwb, "position")
speed, speed_time = read_series(nwb, "speed")
lick, lick_time = read_series(nwb, "lick")
environment, env_time = read_series(nwb, "environment")
reward_zone, rz_time = read_series(nwb, "reward_zone")
trial_start, start_time = read_series(nwb, "trial_start")
teleport, stop_time = read_series(nwb, "teleport")
aligned_times = (speed_time, lick_time, env_time, rz_time, start_time, stop_time)
if any(not np.allclose(timestamps, other) for other in aligned_times):
    raise ValueError(f"{session_id}: behavior timestamps are not aligned")
```

iii. Step 2: "`processing/behavior/BehavioralTimeSeries`: frame-aligned full streams … Behavioral timestamps have a constant 64.483627 ms interval". Step 5 mapping: "`timestamp[start:stop] - timestamp[start]`, seconds … Continuous time-varying float32."

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first timestamp from the timestamps in the trial slice, giving a monotonically increasing vector starting at exactly 0 s; cast to float32. Nothing else.

ii.
```python
inputs = np.vstack(
    (
        np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32),
        np.full(n_time, trial_environments[trial], dtype=np.float32),
        np.full(n_time, trial, dtype=np.float32),
        np.full(n_time, previous_outcome, dtype=np.float32),
    )
)
```

iii. Step 5 mapping table, and Step 10 edge-case check 13: "All converted trials start at time 0, have strictly increasing time, exclude the teleport frame". Resulting range across the dataset is [0, 216.5] s, consistent with variable lap durations plus the paper's early-session-termination policy.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No realignment is needed: the timestamps *are* the neural frame times, and the same `[s:e)` slice is used for both. Alignment is enforced rather than assumed — all behavior streams are asserted mutually timestamp-identical, the sample interval is asserted uniform, the pooled neural matrix is truncated to `len(timestamps)`, and per-trial shapes are asserted equal.

ii.
```python
if any(not np.allclose(timestamps, other) for other in aligned_times):
    raise ValueError(f"{session_id}: behavior timestamps are not aligned")
...
neural_by_time, n_raw_rois, plane_info = load_curated_events(nwb, len(timestamps))
...
if dataset.shape[0] < n_behavior_frames:
    raise ValueError(f"{plane_name}: neural stream is shorter than behavior")
dense = dataset[:n_behavior_frames, :]
```

iii. Step 4: "All behavior timestamps have 64.483627-ms spacing; dual-plane neural series metadata say 31.015625 Hz but arrays align one-to-one with behavior frames." Step 10 check 3 independently reconstructed the time vector from raw NWBs for three trials (including a dual-plane session) and compared with `np.allclose`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The behavior `environment` time series (0 = ENV1, 1 = ENV2), sampled per frame.

ii.
```python
environment, env_time = read_series(nwb, "environment")
...
trial_environments = []
for s, e in zip(starts, stops):
    valid = np.unique(environment[s:e][environment[s:e] >= 0])
    if valid.size != 1 or valid[0] not in (0, 1):
        raise ValueError(f"{session_id}: trial lacks one binary environment label")
    trial_environments.append(int(valid[0]))
```

iii. Step 5 mapping: "`environment` → `input[1]` environment type … `get_trial_types` morph … Binary per-trial context represented across time for matrix consistency." Step 2: "Environment is exactly one valid value (0 or 1) per trial; `-1` occurs only outside synchronized acquisition."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the unique non-negative value of `environment` inside the trial is taken (the code errors if there is not exactly one, and if it is not 0 or 1), then broadcast as a constant across the trial's timepoints as float32. Sentinel `-1` samples are excluded from the uniqueness test.

ii.
```python
valid = np.unique(environment[s:e][environment[s:e] >= 0])
if valid.size != 1 or valid[0] not in (0, 1):
    raise ValueError(f"{session_id}: trial lacks one binary environment label")
trial_environments.append(int(valid[0]))
...
np.full(n_time, trial_environments[trial], dtype=np.float32),
```

iii. "Every trial must have exactly one valid environment value" (code comment). The paper's day-8 design, where the reward switch coincides with entry into the novel environment, is reflected in the converted data: several sessions have per-session environment range [0,1] because the environment changes between trials mid-session, while it is constant within each trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The within-session zero-based index of the trial in the ordered list of `(trial_start, teleport)` pairs — i.e. the enumeration counter, not the stored `trial number` time series. The trial boundaries themselves come from the `trial_start` and `teleport` behavior streams.

ii.
```python
for trial, (s, e) in enumerate(zip(starts, stops)):
    if not keep[trial]:
        continue
    ...
    np.full(n_time, trial, dtype=np.float32),
```

iii. Step 5 mapping: "within-session trial index / `trial number` → `input[2]` … Zero-based trial index, repeated across time … NWB trial number agrees with enumeration." Step 2 independently verified that "trial-number transitions independently agree" with the trial_start/teleport pairing.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the loop index across the trial's timepoints as float32. Crucially, the index is the **native** trial number: lick-artifact rejections leave gaps rather than renumbering, so "trial number" and "previous trial" keep their experimental meaning.

ii.
```python
for trial, (s, e) in enumerate(zip(starts, stops)):
    if not keep[trial]:
        continue
    ...
    previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
    np.full(n_time, trial, dtype=np.float32),
```

iii. Step 10 check 3: "This specifically verifies that rejection does not renumber trials or change the meaning of 'previous trial.'" Values range [0, 99] across sessions, matching the paper's 80–100 target trials per session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` time series' `timestamps` (reward delivery events). Each reward timestamp is mapped to the nearest behavior frame index; a trial is "rewarded" iff at least one mapped index falls in `[start, teleport)`. The previous-trial input reads that per-trial outcome vector one trial back.

ii.
```python
def nearest_timestamp_indices(reference, events):
    right = np.clip(np.searchsorted(reference, events), 0, reference.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(reference[left] - events) < np.abs(reference[right] - events)
    out = np.where(choose_left, left, right).astype(np.int64)
    tolerance = np.median(np.diff(reference)) / 2.0 + 1e-9
    if np.any(np.abs(reference[out] - events) > tolerance):
        raise ValueError("A sparse reward timestamp cannot be aligned to behavior frames")
    return out

reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
```

iii. Step 4: "Rewarded iff reward delivery and reward-zone entry occur in trial … Binary outcome is whether any sparse Reward timestamp falls in `[start,teleport)`, equivalent to reference logic here. Rate 84.659%, consistent with paper." Step 5 Key Decision 5: "Zone entry can occur on omissions; sparse `Reward` is the actual delivery event" (52 zone entries lack a delivery).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `previous_outcome = outcomes[trial - 1]` for `trial > 0`, and `0` for the first trial of a session; broadcast constant across the trial. The index is into the *native* trial list, so the previous trial is the chronologically preceding lap even if that lap was itself rejected for lick artifacts. The current trial's own outcome is never used.

ii.
```python
previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
...
np.full(n_time, previous_outcome, dtype=np.float32),
...
"first_trial_previous_outcome": 0,
```

iii. Step 5 mapping: "0 if preceding trial omitted, 1 if rewarded; first trial=0 because no prior within-session observation; repeated across time … Current-trial outcome is never leaked into this input." Verified in Step 10 check 3 against independently reconstructed raw trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The behavior `position` stream plus the per-trial active reward-zone interval. Zone identity is inferred from the `reward_zone` stream: the positions at which `reward_zone > 0` are classified to the nearest of the fixed intervals A = 80–130, B = 200–250, C = 320–370 cm by their median, and a per-session schedule is built assuming one possible switch at trial index 30 (the paper's documented switch point). Trials 0–29 take the modal observed label, trials 30–end take theirs; unobserved (omission) trials inherit their segment's label. Any observation contradicting its segment's label raises an error — across all 152 sessions, zero contradictions occurred.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def classify_observed_zone(values):
    if values.size == 0:
        return None
    median_position = float(np.median(values))
    centers = {label: np.mean(bounds) for label, bounds in ZONE_BOUNDS.items()}
    return min(centers, key=lambda label: abs(median_position - centers[label]))

def infer_zone_labels(position, reward_zone, starts, stops):
    observed = [
        classify_observed_zone(position[s:e][reward_zone[s:e] > 0])
        for s, e in zip(starts, stops)
    ]
    labels = [None] * len(starts)
    for lo, hi in ((0, min(30, len(starts))), (min(30, len(starts)), len(starts))):
        segment_observed = [observed[i] for i in range(lo, hi) if observed[i] is not None]
        if not segment_observed:
            raise ValueError(f"No observed reward-zone entries in trial segment [{lo}, {hi})")
        label, _ = Counter(segment_observed).most_common(1)[0]
        labels[lo:hi] = [label] * (hi - lo)
    contradictions = sum(obs is not None and obs != labels[i] for i, obs in enumerate(observed))
    if contradictions:
        raise ValueError(f"Found {contradictions} zone observations inconsistent with inferred schedule")
    return [str(label) for label in labels], observed, contradictions
```

iii. Methods: "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" and "Each switch occurred after 30 trials"; the reference code's `get_reward_zones` "derive[s] per-trial A/B/C reward-zone coordinates and labels from scene and switch trial (normally trial index 30)". Step 4: "Scene name/explicit label absent in NWB; zone-entry positions reveal schedules without contradictions, and every pre/post segment has observations." Step 5 Key Decision 4: "Infer schedule from actual zone-entry positions rather than hard-coding mouse sequences. Trial 30 segmentation comes from paper/code and is empirically contradiction-free." The resulting balance (A 4,172 / B 3,974 / C 3,989 trials) matches the counterbalanced design.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the active 50-cm interval: `position - zone_start` when before the zone, `position - zone_end` when after, and exactly `0` everywhere inside (inclusive of both edges). The continuous distance is then discretized (see 7-c).

ii.
```python
def distance_to_interval(position, start, end):
    distance = np.zeros(position.shape, dtype=np.float32)
    before = position < start
    after = position > end
    distance[before] = position[before] - start
    distance[after] = position[after] - end
    return distance
...
z0, z1 = ZONE_BOUNDS[label]
distance = distance_to_interval(trial_position, z0, z1)
```

iii. Step 3: "Signed distance is interpreted relative to **any location in the 50-cm active reward-zone interval**: negative before zone start, exactly zero inside the zone, and positive after zone end. This interpretation follows the requested wording and differs appropriately from paper RR coordinates, which are centered only on zone start and circularized." Metadata records `"distance_definition": "signed distance to the nearest point in the active zone interval; zero everywhere inside the interval"`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes by explicit boolean masks, default class 3 (exactly 0 cm, i.e. inside the zone): 0 = `< -50`; 1 = `[-50, -10]`; 2 = `(-10, 0)`; 3 = `0`; 4 = `(0, 10]`; 5 = `(10, 50]`; 6 = `> 50`. Boundary values −10, +10 and +50 fall in the lower-magnitude class by the stated convention.

ii.
```python
def discretize_distance(distance: np.ndarray) -> np.ndarray:
    out = np.full(distance.shape, 3, dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance <= -10)] = 1
    out[(distance > -10) & (distance < 0)] = 2
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. Step 5 mapping: "discretize to requested 7 bins … Boundary convention: class 1 includes -50 and -10; class 2 is (-10,0); class 4 is (0,10]; class 5 is (10,50]." Step 10 check 13: "Synthetic exact-boundary arrays passed expected categories for distance (-50,-10,0,10,50)". Realised class fractions [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243] — all seven classes populated, with the large class-3 mass reflecting the 50 cm-wide zero plateau.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data share the frame axis and the same `[s:e)` slice, so no alignment operation is required; the per-trial shape assertion guarantees equal length.

ii.
```python
trial_position = np.asarray(position[s:e], dtype=np.float32)
distance = distance_to_interval(trial_position, z0, z1)
...
if neural.shape[1] != n_time or inputs.shape != (4, n_time) or outputs.shape != (6, n_time):
    raise ValueError(f"{session_id} trial {trial}: inconsistent converted shapes")
```

iii. Step 7 processing-plot review: "position/distance/speed threshold crossings coincide with class steps, zone interiors map to distance 0/class 3 … No temporal shift or discretization anomaly was visible." Step 10 check 4 compared every reconstructed output row against the converted data with `np.allclose`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The behavior `position` time series (cm along the 450 cm virtual track), used directly.

ii.
```python
position, timestamps = read_series(nwb, "position")
...
trial_position = np.asarray(position[s:e], dtype=np.float32)
```

iii. Step 5 mapping: "`position` → `output[1]` absolute position … Requested five 90-cm categories over 0–450 cm … paper 450-cm track."

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial, casting to float32, and discretizing. No smoothing, unwrapping or circularization; the raw VR position is used as stored.

ii.
```python
trial_position = np.asarray(position[s:e], dtype=np.float32)
outputs = np.vstack((discretize_distance(distance), discretize_position(trial_position), ...))
```

iii. Step 3/Step 5: the 450 cm track is divided into the requested five equal 90 cm bins; the paper's 45 × 10 cm spatial binning is explicitly not applied because "Dropping slow frames or spatially averaging would erase required labels".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes via boolean masks with the default class 0 absorbing anything below 90 cm (including slightly negative values): 0 = `< 90`; 1 = `[90, 180)`; 2 = `[180, 270)`; 3 = `[270, 360]`; 4 = `> 360`. The first and last classes are left open so the few samples marginally outside 0–450 cm fall into the end bins.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.zeros(position.shape, dtype=np.int64)
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
```

iii. Step 5 mapping: "Time-varying int64. 360 is assigned to bin 3; >360 to bin 4." Step 10 check 13: "Synthetic exact-boundary arrays passed expected categories for … position (90,180,270,360)". Realised fractions [0.212, 0.177, 0.231, 0.226, 0.154] — close to uniform as expected for laps traversed at varying speed.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[s:e)` slice as the neural data; no further alignment. Guaranteed by the mutual timestamp `allclose` check and the per-trial shape assertion.

ii.
```python
trial_position = np.asarray(position[s:e], dtype=np.float32)
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. Step 10 checks 2 and 4 verified neural and position slices for the same raw trials (single-plane m11 ses-03 trial 5, dual-plane m18 ses-03 trial 31, artifact-heavy m4 ses-14 trial 70) with exact `np.allclose(..., rtol=0, atol=0)` comparisons.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The behavior `lick` time series, which stores a cumulative lick count per imaging frame (values can exceed 1 because VR events are preserved by interpolating cumulative counts and differencing during 15.5 Hz downsampling).

ii.
```python
lick, lick_time = read_series(nwb, "lick")
...
(lick[s:e] > 0).astype(np.int64),
```

iii. Step 1 Notes: "Author preprocessing preserves discrete VR events during downsampling by interpolating cumulative counts and differencing. Lick values can consequently exceed one and are binarized as `>0`/clipped to one for binary use." Step 2: "Lick and reward-zone signals are cumulative counts per imaging frame and can exceed 1."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarize: `lick > 0 → 1`, else 0, as int64. No smoothing (unlike the paper's GLM, which smooths licks with a 2-sample Gaussian) and no spatial binning or occupancy normalisation (unlike the paper's lick-rate analyses). The QC step upstream has already removed whole trials whose lick trace is a sensor artifact.

ii.
```python
outputs = np.vstack(
    (
        discretize_distance(distance),
        discretize_position(trial_position),
        discretize_speed(trial_speed),
        (lick[s:e] > 0).astype(np.int64),
        np.full(n_time, ZONE_TO_INT[label], dtype=np.int64),
        np.full(n_time, outcomes[trial], dtype=np.int64),
    )
)
```

iii. Step 5 mapping: "`lick>0` after rejecting artifact trials; cumulative within-frame count becomes binary … Time-varying int64, not smoothed"; matched to `glmUtils.get_timeseries_data`, which "binarize[s] cumulative lick counts". Realised fractions [0.777, 0.223]; Step 12 confirmed "Lick has substantial variation (22.3% positive timepoints; neither class near 99%)."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[s:e)` slice as neural; the `lick` stream's timestamps were asserted identical to the reference timestamps at load. No shift is applied.

ii.
```python
if any(not np.allclose(timestamps, other) for other in aligned_times):
    raise ValueError(f"{session_id}: behavior timestamps are not aligned")
...
(lick[s:e] > 0).astype(np.int64),
```

iii. Step 12 specifically re-examined lick because its accuracy margin was modest: "`critical_review_low_outputs.png` overlays raw mean curated activity, position, binary lick, and outcome on the same native time axis. Visual review found no temporal offset."

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same source as 7-a: the `reward_zone` stream (nonzero when the animal is in the active zone) combined with `position`, resolved into a per-session A/B/C schedule via the modal-observation-per-segment rule with a switch at trial 30.

ii. See 7-a (`classify_observed_zone` / `infer_zone_labels`); the label is then mapped to an integer:
```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
...
np.full(n_time, ZONE_TO_INT[label], dtype=np.int64),
```

iii. See 7-a. Metadata records `"reward_zone_bounds_cm": ZONE_BOUNDS` and per-session `"zone_schedule": [zone_labels[0], zone_labels[min(30, len(zone_labels) - 1)]]` plus `"zone_observation_contradictions": 0`.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The inferred per-trial label is mapped A/B/C → 0/1/2 and broadcast as a constant int64 across the trial's timepoints. Omission trials (no zone entry) inherit their segment's label rather than being dropped or labelled unknown.

ii.
```python
label = zone_labels[trial]
z0, z1 = ZONE_BOUNDS[label]
...
np.full(n_time, ZONE_TO_INT[label], dtype=np.int64),
```

iii. Step 5 mapping: "Infer modal label in trials 0–29 and 30–end; fill all trials; map A/B/C→0/1/2 and repeat in time … Every segment has observations and zero contradictions." Output distribution by timepoint is [0.332, 0.336, 0.333].

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` time series' timestamps (actual delivery events), not the `reward_zone` entry signal and not `autoreward`.

ii.
```python
reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
```

iii. Step 5 Key Decision 5: "Outcome from sparse delivery, not zone entry: Zone entry can occur on omissions; sparse `Reward` is the actual delivery event. This exactly preserves 84.66% rewarded trials." Step 4 notes 10,342 rewarded trials, 52 zone entries without delivery, and 3 trials with two delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each sparse reward timestamp is snapped to the **nearest** behavior frame (left/right comparison, not just `searchsorted`), with a hard tolerance check of half a bin. A trial is 1 if at least one snapped index lies in `[start, teleport)`, else 0; trials with two deliveries still yield 1. The scalar is broadcast across the trial's timepoints as int64.

ii.
```python
right = np.clip(np.searchsorted(reference, events), 0, reference.size - 1)
left = np.maximum(right - 1, 0)
choose_left = np.abs(reference[left] - events) < np.abs(reference[right] - events)
out = np.where(choose_left, left, right).astype(np.int64)
tolerance = np.median(np.diff(reference)) / 2.0 + 1e-9
if np.any(np.abs(reference[out] - events) > tolerance):
    raise ValueError("A sparse reward timestamp cannot be aligned to behavior frames")
...
np.full(n_time, outcomes[trial], dtype=np.int64),
```

iii. Step 4: "Rate 84.659%, consistent with paper" (Methods: "the reward was randomly omitted on ~15% of trials"). Step 12 rejected alternatives explicitly: "Changing outcome to switch only after delivery would improve temporal predictability but violate the specified **per-trial** output. Using reward-zone entry as outcome would falsely convert 52 omissions to rewards."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script is strict — most anomalies raise rather than being silently repaired — but it handles the specific irregularities found in this dataset:
- **Extra trailing neural frames**: ten dual-plane files have one more neural row than behavior frames; each plane is sliced `[:n_behavior_frames]`. A neural stream *shorter* than behavior raises instead of being cropped.
- **Sentinel `-1` environment samples** outside synchronized acquisition are excluded before the per-trial uniqueness test.
- **Lick values > 1** (cumulative counts) are binarized; artifact trials (>30% of frames with count > 2) are removed entirely.
- **Missing reward-zone observations** on omission trials are filled from the segment's modal label rather than dropped.
- **Sparse reward timestamps** off the frame grid are snapped to the nearest frame with a half-bin tolerance assertion; trials with two deliveries remain binary 1.
- **Slightly out-of-range values** (negative smoothed speed, position marginally outside 0–450 cm) are absorbed by the open end classes rather than producing extra categories.
- Hard guards: unequal start/teleport counts, `stop <= start`, non-uniform sample interval, misaligned stream timestamps, ROI-count mismatch, non-finite neural/input values, inconsistent per-trial shapes, and sessions left with <2 trials all raise.
- Notably absent: no minimum-trial-length filter (the shortest retained trial is 96 samples, so none would qualify).

ii.
```python
if dataset.shape[0] < n_behavior_frames:
    raise ValueError(f"{plane_name}: neural stream is shorter than behavior")
dense = dataset[:n_behavior_frames, :]
...
valid = np.unique(environment[s:e][environment[s:e] >= 0])
if valid.size != 1 or valid[0] not in (0, 1):
    raise ValueError(f"{session_id}: trial lacks one binary environment label")
...
if not (np.isfinite(neural).all() and np.isfinite(inputs).all()):
    raise ValueError(f"{session_id} trial {trial}: nonfinite neural/input values")
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than two valid trials")
```

iii. Step 2: "Ten dual-plane files contain one extra final neural row relative to behavior; it is outside all trial intervals and will be safely ignored by slicing to behavioral trial endpoints." Step 10 check 13: "Ten one-row-long neural streams are safely truncated only outside trial endpoints"; "All converted trials … contain finite activity/input, use allowed category indices, and each session retains ≥2 trials."

## 13-a. What are the most time-consuming steps of the code?

i. Per the script's own timing instrumentation, the whole conversion is I/O-bound and fast: 152 sessions in 55.73 s plus 8.41 s to write the 9.0 GiB pickle. The dominant costs are (1) the dense HDF5 read + decompression of each `Deconvolved` plane (0.17–1.14 s/session, scaling with ROI count — m18's ~4,900-ROI dual-plane sessions are the slowest), (2) the `np.isfinite(pooled).all()` full scan and the float32 copies made when masking and transposing each trial, and (3) the final `pickle.dump`. Because no dF/F or OASIS deconvolution is recomputed, the expensive signal-processing stage of the reference pipeline is absent entirely.

ii.
```python
started = time.perf_counter()
...
elapsed = time.perf_counter() - started
print(f"[{session_id}] {len(neural_trials)}/{len(starts)} trials, "
      f"{neural_trials[0].shape[0]}/{n_raw_rois} curated cells, {elapsed:.2f} s", flush=True)
...
write_started = time.perf_counter()
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote {args.outpicklefile} ({args.outpicklefile.stat().st_size / 2**30:.3f} GiB) "
      f"in {time.perf_counter() - write_started:.2f} s", flush=True)
```

iii. Step 6: "Per-trial HDF5 reads or fancy selection of thousands of noncontiguous columns would cause excessive chunk decompression and point-selection overhead." Step 7 estimated "~2.3 min for 152 sessions … total safely below 15 min"; the actual 64 s beat the estimate.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several small Python loops over trials remain, each `O(n_trials)` with an obvious array alternative:
- the `outcomes` list comprehension does a full boolean scan of `reward_indices` per trial (a single `np.searchsorted` of `reward_indices` into `starts`/`stops` would do it in one pass);
- `lick_artifact_fraction` slices `lick` per trial (`np.add.reduceat` over the concatenated trial mask would vectorize it);
- the `trial_environments` loop does a `np.unique` per trial;
- `infer_zone_labels` runs `classify_observed_zone` per trial;
- the main conversion loop calls `discretize_distance/position/speed` per trial, although these could be computed once over the whole session array and then sliced.
The per-plane HDF5 read, the `iscell` masking, and every discretization are already vectorized, and there is no per-neuron loop anywhere (the reference's `is_putative_interneuron` per-cell correlation loop has no counterpart here because that filter is not applied).

ii.
```python
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
lick_artifact_fraction = np.array(
    [np.mean(lick[s:e] > 2) for s, e in zip(starts, stops)], dtype=np.float64
)
for s, e in zip(starts, stops):
    valid = np.unique(environment[s:e][environment[s:e] >= 0])
```

iii. The agent did not enumerate these specific loops; Step 6 only records the general principle ("Behavioral streams are loaded once per session and transforms are vectorized"). With ≤100 trials per session these loops cost milliseconds against a ~0.4 s/session HDF5 read, so the residual gain would be negligible — the per-trial structure is also the natural one given variable trial lengths.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB is opened exactly **once** — there is no separate survey/statistics pass, because the reward-zone schedule is inferred from within the same session being converted. The repeated work that does exist is minor: `--show-processing` recomputes `distance_to_interval` and all three `discretize_*` functions for the plotted trial (already computed in the conversion loop); `info` and `stats` store overlapping per-session counts; and the same per-trial `[s:e]` slice is taken independently for each of the seven behavior streams.

ii.
```python
# in plot_processing -- recomputation of values already produced in the conversion loop
distance = distance_to_interval(pos, z0, z1)
axes[3].step(t, discretize_position(pos) * 90, where="mid", label="position class ×90")
axes[4].step(t, discretize_distance(distance) * 10, where="mid", label="distance class ×10")
...
info = {"session_id": ..., "native_trials": int(starts.size), "retained_trials": len(neural_trials), ...}
stats = {"native_trials": int(starts.size), "retained_trials": len(neural_trials), ...}
```

iii. Step 6: "Each dense deconvolved plane is read once, its local cell mask is applied in memory, and curated planes fill one preallocated time×cell matrix. Trials are then copied once into final contiguous cell×time float32 matrices." The single-pass design is a direct consequence of decision 4 (per-session zone inference from within-session observations), which removes the need for a global pre-pass over the corpus.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts:
- `np.isfinite(pooled).all()` sweeps the entire session neural matrix (up to ~2,800 cells × 28,000 frames) purely as a validation assertion.
- `infer_zone_labels` returns `observed` (the per-trial observed labels) and `contradictions`, of which only `contradictions` is consumed; `observed` is discarded by the caller.
- `zone_labels` and `lick_artifact_fraction` are computed for *all* native trials including those about to be dropped.
- `read_series` returns the timestamps of all seven streams, six of which are used only for the `allclose` alignment check and then discarded.
- A substantial `session_info`/`conversion_summary` metadata block (per-plane ROI counts, per-session zone schedules, reward counts) is built and pickled but is not used by `train_decoder.py`.
- Nothing large is computed and thrown away: because no dF/F is computed, there is no discarded intermediate like the reference's `dff_curr` (which the reference does need for its interneuron filter).

ii.
```python
if cursor != curated_total or not np.isfinite(pooled).all():
    raise ValueError("Curated event matrix failed shape/finite validation")
...
zone_labels, observed_zones, contradictions = infer_zone_labels(
    position, reward_zone, starts, stops
)   # observed_zones is never used again
```

iii. These are deliberate validation/provenance costs, consistent with the instructions' emphasis on sanity checks ("Validate data shapes and types at each step"; "Add other relevant fields, e.g. `session_info`"). The agent documented the validation intent in Step 5's Planned Sanity Checks ("Assert uniform 64.483627-ms bins, matched neural/behavior slices, finite values, monotonic within-trial time, allowed category ranges, and zero zone-schedule contradictions") rather than flagging them as waste.
