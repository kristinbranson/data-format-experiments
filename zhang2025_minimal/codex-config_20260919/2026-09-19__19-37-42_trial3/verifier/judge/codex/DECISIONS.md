# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent did not use `SessionLoader`/`SpikeSortingLoader` as the main loading path. It read the session list from `code/code_zhang2025/data/bwm_release.csv`, optionally restricted that list with `data/DATALIMIT_SUBSET.csv`, used a local `ONE` cache only to resolve each `eid` to a filesystem path, and then opened parquet and `.npy` files directly with `pandas`/`numpy`.

ii.
```python
release = pd.read_csv(BWM_TABLE, index_col=0)
...
one = ONE(mode="local", cache_dir=CACHE_DIR, silent=True)
One.load_cache(one, CACHE_DIR / "Brainwidemap")
...
for eid, rows in release.groupby("eid", sort=False):
    session_path = Path(one.eid2path(str(eid)))
```

```python
table = _required_file(
    session_path.glob("alf/**/_ibl_trials.table.pqt"), "trials table"
)
trials = pd.read_parquet(table)
```

iii. In the trajectory, the agent explicitly decided to work from the local release table and direct file access rather than the higher-level loaders after inspecting how the cache was laid out and how revisions were stored locally (steps 15, 27, 45, 55). The stated motivation was to stay offline/local, control revision selection, and make a resumable converter for a very large output (steps 66, 67).

## 1-b. How are the data split into subjects?

i. Subjects come directly from the `subject` column of `bwm_release.csv`. Each `SessionSpec` stores one subject string, and the final `subjects` list is the sorted unique set of those names. `subject_idx` is then built by indexing each payload's stored subject into that sorted list.

ii.
```python
specs.append(
    SessionSpec(
        eid=str(eid),
        subject=str(rows.iloc[0]["subject"]),
        session_path=session_path,
        probes=tuple(rows["probe_name"].astype(str)),
    )
)
```

```python
subjects = sorted({payload["subject"] for payload in payloads})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_to_idx[payload["subject"]] for payload in payloads],
    dtype=np.int32,
),
```

iii. The trajectory does not show a separate derivation step here; the agent treated subject identity as already curated in the release table and preserved it through conversion.

## 1-c. How are the data split into sessions?

i. Sessions are defined by `eid`. The code groups `bwm_release.csv` by `eid`, creates one `SessionSpec` per group, and processes each session independently.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    session_path = Path(one.eid2path(str(eid)))
    specs.append(
        SessionSpec(
            eid=str(eid),
            subject=str(rows.iloc[0]["subject"]),
            session_path=session_path,
            probes=tuple(rows["probe_name"].astype(str)),
        )
    )
```

iii. In the trajectory, the agent repeatedly described the BWM release as a collection of candidate sessions identified by `eid`, then reported counts in terms of candidate sessions and retained sessions (steps 49, 85, 98).

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of the session trials table. The code loads `_ibl_trials.table.pqt`, keeps one row per trial, then uses a boolean mask to retain a subset of those rows for downstream neural/behavioral extraction.

ii.
```python
table = _required_file(
    session_path.glob("alf/**/_ibl_trials.table.pqt"), "trials table"
)
trials = pd.read_parquet(table)
```

```python
stimulus_times = trials["stimOn_times"].to_numpy(dtype=np.float64)
paper_mask = _paper_trial_mask(trials)
...
keep = paper_mask & wheel_good & motion_good
keep_idx = np.flatnonzero(keep)
```

iii. The trajectory shows the agent treating the trials table as the source of truth for trial boundaries and trial metadata, then auditing how many rows survive the applied mask (steps 15, 49, 85).

## 1-e. How are trials filtered based on quality controls?

i. The agent applies a two-stage trial filter. First, `_paper_trial_mask` requires non-missing task-event columns, non-zero choice, reaction time between 80 ms and 2 s, and `feedback_times - goCue_times <= 10 s`. Second, trials are kept only if wheel and whisker traces can be interpolated across the full alignment window and contain finite values. The code does not pre-filter by `probabilityLeft in {0.2, 0.5, 0.8}`; instead it raises an error later if any retained trial falls outside that set.

ii.
```python
return (
    trials[needed].notna().all(axis=1).to_numpy()
    & (trials["choice"].to_numpy() != 0)
    & (reaction_time.to_numpy() >= 0.08)
    & (reaction_time.to_numpy() <= 2.0)
    & (trial_duration.to_numpy() <= 10.0)
)
```

```python
wheel, wheel_good = _load_wheel_speed(spec.session_path, stimulus_times)
motion, motion_good, camera_view = _load_whisker_energy(
    spec.session_path, stimulus_times
)
keep = paper_mask & wheel_good & motion_good
```

iii. The trajectory shows the agent deliberately using the task/paper exclusion mask from the provided repository and then reproducing the 459-to-444 session reduction through wheel/camera coverage checks rather than a fixed session list (steps 15, 49, 50, 85).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The per-trial `neural` arrays are built from per-probe `spikes.times.npy` and `spikes.clusters.npy`. The code also reads `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, but those are used to assign brain-region labels rather than to construct the spike-count matrix itself.

ii.
```python
return (
    _required_file(base.glob("**/spikes.times.npy"), f"{probe} spike times"),
    _required_file(base.glob("**/spikes.clusters.npy"), f"{probe} spike clusters"),
    _required_file(base.glob("**/clusters.channels.npy"), f"{probe} cluster channels"),
    _required_file(
        base.glob("**/channels.brainLocationIds_ccf_2017.npy"),
        f"{probe} channel atlas ids",
    ),
)
```

```python
times = np.load(times_path, mmap_mode="r")
clusters = np.load(clusters_path, mmap_mode="r")
```

iii. In the trajectory, the agent framed the neural stream as “all Kilosort 2.5 clusters from all probes in a session” and focused its audit on spike arrays plus probe metadata needed to merge probes and label neurons (steps 15, 29, 30, 66).

## 2-b. How is the `neural` data processed?

i. Neural activity is represented as non-overlapping 20 ms spike counts, not firing rates. The code bins spikes separately for each probe within the `[-0.5, 1.5)` stimulus-aligned window, keeps one count per unit per bin, and then concatenates all probe-specific neuron axes into one session-level population matrix.

ii.
```python
spike_bins = np.floor(
    (np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE
).astype(np.int64)
...
binned[trial].flat[:] = np.bincount(
    flat, minlength=len(used) * N_BINS
).astype(np.float32, copy=False)
```

```python
counts, used = _bin_probe(
    times_path, clusters_path, interval_begins, len(cluster_channels)
)
...
neural = np.concatenate(probe_bins, axis=1)
```

iii. The trajectory shows the agent explicitly justifying counts rather than rates by referring to the methods paper’s “temporally binned spike counts” decoder formulation and to matching the methods-paper decoder pipeline (steps 9, 15, 66, 98).

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered by cluster quality. The agent keeps all cluster IDs present in the spike train for each probe, without checking `clusters['label']` and without dropping `void` or other atlas labels. Region labels are assigned afterward from channel atlas IDs.

ii.
```python
used = np.unique(clusters)
...
binned = np.zeros((len(interval_begins), len(used), N_BINS), dtype=np.float32)
```

```python
cluster_channels = np.load(cluster_channels_path).astype(np.int64, copy=False)
channel_atlas_ids = np.load(atlas_ids_path)
...
cluster_atlas_ids = channel_atlas_ids[cluster_channels[used]]
regions = atlas.id2acronym(cluster_atlas_ids, mapping="Beryl")
```

iii. The trajectory explicitly states that the converter would use “all Kilosort 2.5 clusters” and later records metadata saying `"neuron_filter": "all Kilosort clusters, matching the methods-paper decoder pipeline"` (steps 66, 98; metadata in the final code).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window begins at `stimOn_times - 0.5 s`, ends at `stimOn_times + 1.5 s`, and spike bins are measured relative to that window start. This makes stimulus onset the common alignment event, with bin 25 corresponding to time 0.

ii.
```python
interval_begins = stimulus_times[keep] + OFF_START
...
spike_bins = np.floor(
    (np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE
).astype(np.int64)
```

iii. The trajectory repeatedly described the data as stimulus-aligned over a common `[-0.5, 1.5)` window and reported validator results in those terms (steps 15, 66, 98, 266).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms, with 100 bins spanning the 2 s window. No temporal rebinning beyond this one fixed binning/interpolation step is applied.

ii.
```python
OFF_START = -0.5
OFF_END = 1.5
BIN_SIZE = 0.02
N_BINS = 100
```

```python
"metadata": {
    ...
    "time_bin_size": 20.0,
```

iii. The trajectory explicitly tied the implementation to the methods-paper 2 s, 20 ms-bin setup (steps 9, 15, 266).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Conceptually it is tied to `trials["stimOn_times"]`, since that is the alignment event for each trial, but the stored input values are a fixed relative-time vector generated from `OFF_START`, `BIN_SIZE`, and `N_BINS` rather than from per-trial timestamps directly.

ii.
```python
stimulus_times = trials["stimOn_times"].to_numpy(dtype=np.float64)
...
relative_time = (
    OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE
).astype(np.float32)
```

iii. The trajectory consistently described stimulus onset as the temporal alignment anchor and treated this input as the shared time coordinate for the aligned trial window (steps 15, 66, 266).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent uses the left edge of each 20 ms neural bin, producing `[-0.5, -0.48, ..., 1.48]`. It does not use bin centers.

ii.
```python
relative_time = (
    OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE
).astype(np.float32)
...
inputs[:, 0, :] = relative_time
```

iii. The trajectory does not contain a separate written defense of left edges versus centers; the only explicit rationale is in the final metadata, which records `"time_coordinate_convention": "left edge of each 20 ms neural bin"`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction to the neural bin grid: the time vector gives the left edge of each neural bin in the stimulus-aligned window. However, the behavioral interpolation code uses a different convention (sampling at the bin right edges), so this input is aligned to neural counts more directly than to the resampled wheel/whisker traces.

ii.
```python
inputs[:, 0, :] = relative_time
...
spike_bins = np.floor(
    (np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE
).astype(np.int64)
```

iii. The trajectory only indirectly justifies this through the final metadata entry about left-edge bin coordinates; it does not mention the mismatch with the behavioral interpolation grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials["probabilityLeft"]`. A block is inferred as a run of consecutive trials with identical `probabilityLeft`.

ii.
```python
def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based trial position in each uninterrupted probability block."""
```

```python
probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
...
trial_in_block = _trial_number_in_block(probability_left)
```

iii. The trajectory describes block structure in terms of the probability-left sequence and preserves that convention in the final metadata (`"trial_number_in_block_convention"`).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code computes a zero-based run length over the full, unfiltered `probabilityLeft` sequence. If the current trial has the same `probabilityLeft` as the previous trial, the counter increments; otherwise it resets to zero. Filtering is applied only afterward when the retained trials are indexed out.

ii.
```python
out = np.zeros(len(probability_left), dtype=np.float32)
for trial in range(1, len(probability_left)):
    out[trial] = (
        out[trial - 1] + 1
        if probability_left[trial] == probability_left[trial - 1]
        else 0
    )
```

```python
inputs[:, 1, :] = trial_in_block[keep, None]
```

iii. The trajectory does not add more than the code and metadata here; the metadata explicitly states this is the “full unfiltered probabilityLeft sequence.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials["choice"]`.

ii.
```python
raw_choice = trials["choice"].to_numpy()
```

iii. The trajectory inspected the IBL `choice` coding directly and used that convention in the converter (steps 31, 32).

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps rightward choices (`-1`) to `1` and everything else to `0`. Because the trial mask already removes `choice == 0`, retained trials are effectively mapped as left `(+1) -> 0` and right `(-1) -> 1`.

ii.
```python
# In IBL choice coding, -1 is the rightward response and +1 is leftward.
choice = (raw_choice == -1).astype(np.int64)
...
outputs[:, 0, :] = choice[keep, None]
```

iii. The trajectory shows the agent checking the IBL coding before committing to this mapping (steps 31, 32).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials["probabilityLeft"]`.

ii.
```python
probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
```

iii. The trajectory consistently treated `probabilityLeft` as the block-prior variable and used it both for the block counter and for the categorical prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2` using `np.isclose` to handle floating-point equality. It initializes everything to `-1` and raises an error if any retained trial remains unmapped.

ii.
```python
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.full(len(trials), -1, dtype=np.int64)
for value, label in prior_lookup.items():
    prior[np.isclose(probability_left, value)] = label
if np.any(prior[keep] < 0):
    raise ValueError("probabilityLeft contains a value outside {0.2, 0.5, 0.8}")
```

iii. The trajectory frames this as preserving the instructed 3-class mapping while being robust to floating-point representation.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
position_path = _required_file(
    session_path.glob("alf/**/_ibl_wheel.position.npy"), "wheel position"
)
timestamps_path = _required_file(
    session_path.glob("alf/**/_ibl_wheel.timestamps.npy"), "wheel timestamps"
)
```

iii. The trajectory explicitly inspected `SessionLoader.load_wheel` and the wheel helper functions before reproducing their logic directly (steps 33, 34, 46).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1 kHz, low-pass filtered while computing velocity (`corner_frequency=20`, `order=8`), converted to speed with `np.abs`, resampled onto the trial grid, and then discretized into three within-session tertile classes pooled across all retained trials and time bins.

ii.
```python
position_1khz, times_1khz = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(
    position_1khz, fs=1000, corner_frequency=20, order=8
)
return _interpolate_trials(times_1khz, np.abs(velocity), stimulus_times)
```

```python
wheel = wheel[keep]
wheel_labels, wheel_edges = _three_bins(wheel)
...
outputs[:, 2, :] = wheel_labels
```

iii. The trajectory says this was chosen to match the repository/IBL wheel preprocessing while avoiding a global threshold that would be dominated by lab- or rig-specific scale differences (steps 33, 34, 66).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The thresholding is session-specific. The code takes the 1/3 and 2/3 quantiles of the retained wheel-speed values from that session, then applies `np.digitize(..., [q1, q2])` to produce labels `0`, `1`, and `2` corresponding to low, medium, and high.

ii.
```python
def _three_bins(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Discretize a session's samples into low/middle/high tertiles."""
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
    return labels, (float(q1), float(q2))
```

iii. The trajectory explicitly justifies within-session tertiles as a way to preserve low/medium/high behavioral state while avoiding absolute-scale drift across sessions and labs (step 66).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is sliced to the same `[-0.5, 1.5]` trial window around `stimOn_times`, but the actual interpolation points are `begins + 0.02, ..., ends`, i.e. the right edge of each 20 ms bin rather than the neural-bin centers used in the reference solution.

ii.
```python
begins = stimulus_times + OFF_START
ends = stimulus_times + OFF_END
...
target_times = np.linspace(
    begins[trial] + BIN_SIZE, ends[trial], N_BINS, dtype=np.float64
)
values[trial] = interp1d(
    times, vals, kind="linear", fill_value="extrapolate"
)(target_times)
```

iii. The trajectory justified the alignment window and full-coverage checks, but it did not separately justify using right-edge sample times instead of the reference bin centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, or, if the left camera is unavailable/invalid, from the corresponding right-camera files.

ii.
```python
for view in ("left", "right"):
    try:
        energy_path = _required_file(
            session_path.glob(f"alf/**/{view}Camera.ROIMotionEnergy.npy"),
            f"{view} whisker motion energy",
        )
        times_path = _required_file(
            session_path.glob(f"alf/**/_ibl_{view}Camera.times.npy"),
            f"{view} camera timestamps",
        )
```

iii. The trajectory explicitly says the converter would prefer the left whisker-pad trace and fall back to the right camera when needed, following the reference-code logic (steps 33, 47, 66).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used directly. If the timestamps array is longer than the energy array, the code trims timestamps from the front to match; if timestamps are shorter than the energy array, that camera view is rejected. The retained trace is then linearly interpolated onto the trial grid and discretized into within-session tertiles.

ii.
```python
energy = np.load(energy_path)
times = np.load(times_path)
if len(times) < len(energy):
    raise ValueError("camera timestamps are shorter than motion energy")
if len(times) > len(energy):
    times = times[-len(energy) :]
values, good = _interpolate_trials(times, energy, stimulus_times)
```

```python
motion = motion[keep]
motion_labels, motion_edges = _three_bins(motion)
...
outputs[:, 3, :] = motion_labels
```

iii. The trajectory shows the agent inspecting `SessionLoader._check_video_timestamps` and then re-implementing that timestamp/trace reconciliation in its own file loader (steps 47, 66).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly like wheel speed: per session, using the 1/3 and 2/3 quantiles over all retained whisker-motion-energy samples, then `np.digitize` to produce labels `0/1/2`.

ii.
```python
motion = motion[keep]
motion_labels, motion_edges = _three_bins(motion)
...
outputs[:, 3, :] = motion_labels
```

```python
labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
```

iii. The same within-session-tertile justification given in the trajectory for wheel speed also applies here (step 66).

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-centered trial window, but sampled at the right edge of each 20 ms bin (`-0.48, -0.46, ..., 1.5`) rather than at bin centers.

ii.
```python
target_times = np.linspace(
    begins[trial] + BIN_SIZE, ends[trial], N_BINS, dtype=np.float64
)
values[trial] = interp1d(
    times, vals, kind="linear", fill_value="extrapolate"
)(target_times)
```

iii. The trajectory justifies the left/right camera fallback and the requirement for full window coverage, but does not separately justify the right-edge sampling convention.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid data are mainly handled by skipping trials or entire sessions. File selection prefers the latest dated revision. Wheel/camera trial windows must cover the full aligned interval and contain finite interpolated values. Left camera is tried first, then right. Sessions with fewer than two valid trials raise an error and are skipped. Missing required files usually fail the session rather than being repaired, except for the specific case where extra camera timestamps are trimmed to match the motion-energy trace.

ii.
```python
def _required_file(paths, description: str) -> Path:
    path = _latest(paths)
    if path is None:
        raise FileNotFoundError(description)
    return path
```

```python
if len(times) > len(energy):
    times = times[-len(energy) :]
...
keep = paper_mask & wheel_good & motion_good
...
if len(keep_idx) < 2:
    raise ValueError(f"only {len(keep_idx)} aligned valid trials")
```

iii. The trajectory repeatedly described the final 444-session dataset as the result of excluding sessions/trials that lacked usable whisker data or full wheel/camera coverage, rather than trying to impute missing streams (steps 49, 85, 98).

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are the per-session spike loading/binning over all probes and the repeated checkpoint serialization of very large session payloads. Behavioral interpolation is also repeated trial by trial, but the large spike arrays dominate both I/O and compute.

ii.
```python
times = np.load(times_path, mmap_mode="r")
clusters = np.load(clusters_path, mmap_mode="r")
...
for trial, (lo, hi) in enumerate(zip(start_idx, end_idx)):
```

```python
with open(tmp_path, "wb") as stream:
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(tmp_path, checkpoint_path)
```

iii. In the trajectory, the agent emphasized the size of the full output and introduced per-session checkpoints specifically because the converted dataset was extremely large (steps 66, 85, 98).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the per-trial interpolation loop in `_interpolate_trials`, the per-trial spike-binning loop in `_bin_probe`, and the sequential run-length loop in `_trial_number_in_block`.

ii.
```python
for trial, (beg_idx, end_idx) in enumerate(zip(idx_beg, idx_end)):
    ...
    values[trial] = interp1d(
        times, vals, kind="linear", fill_value="extrapolate"
    )(target_times)
```

```python
for trial, (lo, hi) in enumerate(zip(start_idx, end_idx)):
    ...
    binned[trial].flat[:] = np.bincount(
        flat, minlength=len(used) * N_BINS
    ).astype(np.float32, copy=False)
```

iii. The trajectory does not discuss vectorization explicitly, but these are the main scalar loops left in the implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans/globs for “latest” revisions, repeatedly reopens checkpoint payloads during validation and final assembly, and repeats nearly identical interpolation logic for wheel and whisker streams through the shared `_interpolate_trials` path. It also recreates per-session `BrainRegions()` state instead of reusing a single global atlas object.

ii.
```python
def _latest(paths) -> Path | None:
    candidates = [Path(p) for p in paths if Path(p).is_file()]
```

```python
with open(path, "rb") as stream:
    payloads.append(pickle.load(stream))
```

```python
wheel, wheel_good = _load_wheel_speed(spec.session_path, stimulus_times)
motion, motion_good, camera_view = _load_whisker_energy(
    spec.session_path, stimulus_times
)
```

iii. The trajectory’s emphasis on resumable checkpoints explains some of this repeated work: the agent chose restartability and local robustness over a leaner single-pass implementation (steps 66, 85).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main extra work is operational rather than scientific: writing and rereading per-session checkpoints, validating checkpoint payloads, building a `failures.json`, and collecting detailed `session_info` metadata such as tertile edges and elapsed times that the downstream decoder does not need. Those steps support robustness and auditing, but they do not affect the decoder inputs/outputs.

ii.
```python
payload = {
    ...
    "session_info": {
        ...
        "wheel_speed_tertile_edges": list(wheel_edges),
        "whisker_motion_energy_tertile_edges": list(motion_edges),
        "elapsed_seconds": float(time.time() - started),
    },
}
```

```python
def _valid_checkpoint(path: Path, eid: str) -> bool:
    ...
```

```python
failure_path = args.output.with_suffix(".failures.json")
with open(failure_path, "w") as stream:
    json.dump(failures, stream, indent=2, sort_keys=True)
```

iii. The trajectory explicitly says these checkpoints were introduced because the final payload was huge and the agent wanted resumability, not because the downstream decoder required them (step 66).
