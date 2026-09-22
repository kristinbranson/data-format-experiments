# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI first discovers all subjects and sessions by scanning `/app/data` for subject directories whose names start with `jm`, then scanning each subject for daily session directories whose names begin with a date-like numeric prefix. It builds a `SessionInfo` record from `ops.npy`, `F.npy`, and `motion_energy_glob.npy` to capture frame counts and neuron counts, then later loads the full per-session arrays from `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy` during processing.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            suite2p_dir = session_dir / "suite2p" / "plane0"
            move_dir = session_dir / "move_deve"
            ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
            f = np.load(suite2p_dir / "F.npy", mmap_mode="r")
            motion = np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r")
            sessions.append(
                SessionInfo(
                    subject=subject_dir.name,
                    session=session_dir.name,
                    session_dir=session_dir,
                    suite2p_dir=suite2p_dir,
                    move_dir=move_dir,
                    nframes=int(ops["nframes"]),
                    nneurons=int(f.shape[0]),
                    motion_len=int(motion.shape[0]),
                )
            )
```

```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
tstamps = np.load(session.move_dir / "tstamps.npy")
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by documenting that the dataset is organized by subject and daily session, with tracked Suite2p outputs in `suite2p/plane0` and behavior in `move_deve/`. In the trajectory, it also stated that the script would “process sessions one by one” and “discover sessions directly from `/app/data`”.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories under `/app/data` whose names start with `jm`, sorted lexicographically.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

iii. The AI’s notes list the subject folders `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and treat each such folder as one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are split by subdirectories inside each subject folder whose names begin with digits, corresponding to date-coded session folders such as `YYYY-MM-DD_a`. These are also sorted lexicographically.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
```

iii. The AI’s notes say each subject “contains daily session folders named like `YYYY-MM-DD_a`” and that session order would be sorted by subject then session date.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous rather than natively trialized. After temporal binning, each session is split into non-overlapping fixed 60-second windows. With 30 Hz imaging and 10-frame bins, that gives 180 bins per trial. Any remainder shorter than one full trial is dropped.

ii.
```python
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES

def split_time_series_into_trials(arr: np.ndarray, bins_per_trial: int) -> list[np.ndarray]:
    usable = arr.shape[-1] - (arr.shape[-1] % bins_per_trial)
    if usable <= 0:
        raise ValueError(f"Array with shape {arr.shape} does not contain a full trial.")
    arr = arr[..., :usable]
    ntrials = usable // bins_per_trial
    return [arr[:, i * bins_per_trial:(i + 1) * bins_per_trial] for i in range(ntrials)]
```

```python
neural_trials = split_time_series_into_trials(binned_neural, BINS_PER_TRIAL)
input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
output_trials = split_time_series_into_trials(motion_bins[np.newaxis, :], BINS_PER_TRIAL)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly notes that the dataset has “no native trials” and that, for the decoder benchmark, it would “split into fixed 60 s chunks using 30 Hz frames” and later “split into fixed 60-second trials after alignment and denoising”.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-quality filtering. It keeps all full 60-second windows from every session. The only implicit curation is that incomplete tails shorter than one full trial are discarded, and a session with fewer than one full trial would raise an error.

ii.
```python
usable = arr.shape[-1] - (arr.shape[-1] % bins_per_trial)
if usable <= 0:
    raise ValueError(f"Array with shape {arr.shape} does not contain a full trial.")
arr = arr[..., :usable]
```

iii. The AI justified this in its notes by saying the released sessions already satisfy the decoder requirement of having at least two trials per session and that it would “keep all sessions and all tracked neurons provided in the release”.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`. The session frame count from `ops.npy` is used for alignment and time construction, but the neural values themselves come from `F` and `Fneu`.

ii.
```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says the released data provide the ingredients needed to reconstruct the paper-consistent dF/F-like signal from `F`, `Fneu`, and `ops`, rather than using `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction with coefficient `0.7`, then runs Suite2p’s baseline preprocessing with `baseline='maximin'`, `win_baseline=60`, `sig_baseline=10`, `prctile_baseline=8`, `fs=30`, and `batch_size=128`. After that, it averages the traces into non-overlapping 10-frame bins.

ii.
```python
corrected = f - NEUROPIL_COEFF * fneu
corrected = suite2p_preprocess(
    corrected.copy(),
    baseline=BASELINE_MODE,
    win_baseline=WIN_BASELINE_SECONDS,
    sig=SIG_BASELINE_FRAMES,
    fs=FRAME_RATE_HZ,
    prctile_baseline=PRCTILE_BASELINE,
    batch_size=SUITE2P_BATCH_SIZE,
    device=torch.device("cpu"),
).astype(np.float32)
```

```python
binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
```

iii. The AI’s justification appears repeatedly in the trajectory and notes: it states that the paper used “baseline corrected fluorescence traces as our dF/F”, not raw `F.npy` or deconvolved spikes, and that it found the likely reconstruction path from the installed Suite2p code: neuropil subtraction followed by Suite2p baseline preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron-level filtering in `convert_data.py`. It assumes the released session files are already Track2p-tracked, already restricted to cells present across all days, and already satisfy the Suite2p `iscell` threshold used in the reference pipeline.

ii.
```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
...
session_data = {
    "neural_trials": neural_trials,
    "input_trials": input_trials,
    "output_trials": output_trials,
    "brain_region_idx": np.zeros(session.nneurons, dtype=np.int64),
}
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says the released suite2p folders already contain “only tracked all-day cells” and that “no extra cell filtering beyond validity checks is needed”. The trajectory also emphasizes preserving Track2p’s tracked-cell conventions rather than recomputing `iscell` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI effectively aligns neural data to the start of each artificial 60-second trial window. It does not use a natural stimulus or behavioral event. This is reflected in the metadata, which defines the temporal alignment event as the start of the fixed 60-second windows tiled across each session.

ii.
```python
neural_trials = split_time_series_into_trials(binned_neural, BINS_PER_TRIAL)
```

```python
"metadata": {
    "temporal_alignment_event": "trial start of fixed 60-second windows tiled across each session",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
    ...
}
```

iii. The AI justified this in Step 5 of `CONVERSION_NOTES.md` by saying the source recordings are continuous and that, for the requested output format, it would “segment continuous sessions into 60-second trials while preserving continuous-time alignment”.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have a temporal resolution of 10 imaging frames at 30 Hz, i.e. `333.33 ms` per bin. The AI applies non-overlapping temporal averaging to both neural and motion streams before trialization.

ii.
```python
BIN_FRAMES = 10
FRAME_RATE_HZ = 30.0
...
binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES).astype(np.float32)
```

```python
"time_bin_size": float(1000.0 * BIN_FRAMES / FRAME_RATE_HZ),
"binning_description": "Non-overlapping means of 10 consecutive 30 Hz samples",
```

iii. The AI justified this by citing the paper’s decoding analysis, which averaged “10 consecutive timestamps” for both neural and behavior traces. Its notes repeatedly describe 10-frame averaging as the paper-consistent denoising step.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not loaded from a dedicated raw timestamp variable. Instead, it is derived from the imaging frame index and the known imaging frame rate (`30 Hz`) for each session.

ii.
```python
frame_times = np.arange(session.nframes, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
```

```python
"input_names": ["time_from_session_start_sec"],
```

iii. In `CONVERSION_NOTES.md`, the AI maps “Session frame index / imaging clock” to the input field and justifies that as the appropriate source for the required “time elapsed from the beginning of the session”.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a full-session frame-time vector in seconds, then averages it in the same non-overlapping 10-frame bins used for neural and behavior. This yields bin-center times rather than left-edge times. It then splits that binned time series into 60-second trials.

ii.
```python
frame_times = np.arange(session.nframes, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
binned_time = non_overlapping_mean_last_axis(frame_times, BIN_FRAMES).astype(np.float32)
...
input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
```

iii. The AI documented this choice in `CONVERSION_NOTES.md` by describing the input as “bin-center times” from the raw 30 Hz imaging timeline and later reporting the resulting range `[0.15, 1799.82]`, which is consistent with 10-frame mean times.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the time input to the neural data by constructing it on the imaging-frame clock, then applying the same 10-frame binning and the same 60-second trial splitting used for the neural traces.

ii.
```python
binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
frame_times = np.arange(session.nframes, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
binned_time = non_overlapping_mean_last_axis(frame_times, BIN_FRAMES).astype(np.float32)
...
neural_trials = split_time_series_into_trials(binned_neural, BINS_PER_TRIAL)
input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
```

iii. The AI’s notes explicitly say it would use the imaging frames as the master clock and later verify that `time_from_session_start_sec` exactly matches the expected binned timing grid. Step 10 reports that this sanity check passed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The motion-energy output is derived from `move_deve/motion_energy_glob.npy`. For temporal alignment and repair of dropped camera frames, the AI uses `move_deve/tstamps.npy`.

ii.
```python
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
tstamps = np.load(session.move_dir / "tstamps.npy")
```

iii. In its notes, the AI identifies dropped camera frames as the main behavioral alignment issue and says it will use motion timestamps to reconstruct behavior on the imaging timeline.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI does not recompute motion energy from video frames; it uses the provided motion-energy trace. It aligns that trace to the imaging frame grid using timestamps, averages duplicate assignments if multiple camera samples fall on the same imaging frame, linearly interpolates frames with no assigned sample, and then applies the same 10-frame non-overlapping averaging as for the neural data.

ii.
```python
motion_aligned, missing_mask, raw_frame_idx, timestamp_scale = align_motion_to_imaging(
    motion=motion_raw,
    tstamps=tstamps,
    nframes=session.nframes,
    fs=FRAME_RATE_HZ,
)
...
binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES).astype(np.float32)
```

```python
span = float(tstamps[-1] - tstamps[0])
nominal_step = span / float(nframes - 1)
frame_idx = np.rint((tstamps - tstamps[0]) / nominal_step).astype(np.int64)
...
np.add.at(sums, frame_idx, motion)
np.add.at(counts, frame_idx, 1)
...
aligned[missing] = np.interp(missing_idx, valid_idx, aligned[valid])
```

iii. The trajectory shows the AI debugging this alignment logic directly: it first found that naïve use of timestamp scale created false missing frames, then changed the approach to direct one-to-one mapping when lengths match and full-span timestamp anchoring when they do not. The notes justify this as necessary because imaging is the master clock and some behavior sessions have dropped frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes the binned motion signal separately within each session into five equal-percentile categories. It first computes session-specific quantile thresholds and applies `np.digitize`. If ties collapse one or more categories, it falls back to a stable rank-based assignment to force five equal-count classes.

ii.
```python
def compute_motion_quintiles(values: np.ndarray, nclasses: int = 5) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, nclasses + 1)[1:-1]
    thresholds = np.quantile(values, quantiles)
    categories = np.digitize(values, thresholds, right=False).astype(np.int64)

    if np.unique(categories).size < nclasses:
        order = np.argsort(values, kind="mergesort")
        categories = np.empty(values.shape[0], dtype=np.int64)
        boundaries = np.linspace(0, values.shape[0], nclasses + 1, dtype=int)
        for cls in range(nclasses):
            categories[order[boundaries[cls]:boundaries[cls + 1]]] = cls
    return categories, thresholds.astype(np.float32)
```

iii. The AI’s notes say the benchmark requires “five equal-percentile bins selected per session”, and its validation notes emphasize that the resulting class distribution is exactly `20%` per class per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses the imaging frame grid as the master timeline. If motion length already equals imaging length, it keeps the motion trace frame-for-frame. Otherwise, it maps behavior timestamps onto the full imaging-frame span, fills missing imaging frames by interpolation, bins motion with the same 10-frame averaging, and splits motion and neural data with the same trial boundaries.

ii.
```python
if motion.size == nframes:
    return (
        motion.astype(np.float32, copy=False),
        np.zeros(nframes, dtype=bool),
        np.arange(nframes, dtype=np.int64),
        scale_to_seconds,
    )
```

```python
motion_aligned, missing_mask, raw_frame_idx, timestamp_scale = align_motion_to_imaging(
    motion=motion_raw,
    tstamps=tstamps,
    nframes=session.nframes,
    fs=FRAME_RATE_HZ,
)
...
binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES).astype(np.float32)
...
neural_trials = split_time_series_into_trials(binned_neural, BINS_PER_TRIAL)
output_trials = split_time_series_into_trials(motion_bins[np.newaxis, :], BINS_PER_TRIAL)
```

iii. The AI’s notes state that “imaging frames are the master clock” and that missing camera samples should be “inserted by timestamp-derived indexing and linearly interpolated”. Its Step 10 sanity checks report that the recovered missing-frame counts exactly matched the raw length discrepancies.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mainly handles missing behavior samples. Dropped or missing camera frames are detected implicitly through timestamp-to-frame alignment, then filled by linear interpolation on the imaging timeline. Sessions where the motion trace already matches imaging length bypass this repair. Invalid timestamp spans or fully missing behavior after alignment raise errors. Any trailing data shorter than a full 60-second trial are discarded.

ii.
```python
if motion.size == nframes:
    return (
        motion.astype(np.float32, copy=False),
        np.zeros(nframes, dtype=bool),
        np.arange(nframes, dtype=np.int64),
        scale_to_seconds,
    )
...
if span <= 0:
    raise ValueError("Motion timestamps are not strictly increasing.")
...
if missing.all():
    raise ValueError("All motion samples are missing after timestamp alignment.")
...
if missing.any():
    valid_idx = np.flatnonzero(valid)
    missing_idx = np.flatnonzero(missing)
    aligned[missing] = np.interp(missing_idx, valid_idx, aligned[valid])
```

```python
usable = arr.shape[-1] - (arr.shape[-1] % bins_per_trial)
arr = arr[..., :usable]
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly identifies missing camera frames as the main edge case, documents fixes to its alignment method, and reports that sessions with `0, 1, 2, 3, 116, and 148` missing frames all process successfully.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies neural preprocessing as the main cost, especially Suite2p baseline preprocessing over full-session calcium traces. It also measures separate timings for loading, motion alignment, binning, trial splitting, and optional plotting, but the heavy numerical step is the Suite2p preprocessing.

ii.
```python
stage_times: dict[str, float] = {}
...
t0 = time.perf_counter()
corrected = f - NEUROPIL_COEFF * fneu
corrected = suite2p_preprocess(
    corrected.copy(),
    baseline=BASELINE_MODE,
    win_baseline=WIN_BASELINE_SECONDS,
    sig=SIG_BASELINE_FRAMES,
    fs=FRAME_RATE_HZ,
    prctile_baseline=PRCTILE_BASELINE,
    batch_size=SUITE2P_BATCH_SIZE,
    device=torch.device("cpu"),
).astype(np.float32)
stage_times["neural_preprocess_s"] = time.perf_counter() - t0
```

iii. The notes include per-session runtime estimates and explicitly describe the pipeline as “session-wise and vectorized”, with full conversion time dominated by the per-session neural preprocessing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes the main temporal averaging and motion-frame repair, so the remaining obvious Python loops are relatively small: the list-comprehension-based trial splitting, the fallback loop that assigns quantile classes one class at a time when thresholds collapse, and the plotting loops used only for diagnostics.

ii.
```python
return [arr[:, i * bins_per_trial:(i + 1) * bins_per_trial] for i in range(ntrials)]
```

```python
for cls in range(nclasses):
    categories[order[boundaries[cls]:boundaries[cls + 1]]] = cls
```

```python
for neuron in range(neurons_to_show):
    trace = corrected_dff[neuron, :plot_frames]
    axes[0].plot(frame_times[:plot_frames], trace + offset, lw=0.9, label=f"neuron {neuron}")
```

iii. The AI’s notes emphasize that the expensive pieces were already vectorized: “binning uses reshape+mean without Python loops” and missing-frame handling was redesigned away from iterative insertion.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat the heavy numerical preprocessing, but it does perform some repeated bookkeeping and I/O. `discover_sessions()` opens `ops.npy`, `F.npy`, and `motion_energy_glob.npy` to collect metadata, and `process_session()` later reloads the full arrays for actual processing. Optional plotting also revisits the processed arrays for visualization after the final data tensors have already been created.

ii.
```python
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
f = np.load(suite2p_dir / "F.npy", mmap_mode="r")
motion = np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r")
```

```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
tstamps = np.load(session.move_dir / "tstamps.npy")
```

iii. The AI’s notes claim “no redundant file I/O across stages”, but the code itself still does a light discovery pass before the full processing pass. That repeated work is modest compared with the per-session Suite2p preprocessing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes optional diagnostic processing that is not needed by the downstream decoder: plotting helpers, z-scoring for a visualization heatmap, computation and storage of timing diagnostics and timestamp scale, and generation of per-session processing figures when `--show-processing` is enabled. These are useful for debugging but are not consumed by the decoder itself.

ii.
```python
def zscore_rows(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std
```

```python
if show_processing:
    plot_processing_summary(
        session=session,
        corrected_dff=corrected,
        binned_neural=binned_neural,
        motion_raw=np.asarray(motion_raw, dtype=np.float32),
        motion_aligned=motion_aligned,
        missing_mask=missing_mask,
        raw_frame_idx=raw_frame_idx,
        binned_motion=binned_motion,
        motion_bins=motion_bins,
        time_input=binned_time[np.newaxis, :],
        thresholds=thresholds,
    )
```

```python
session_meta = {
    ...
    "timestamp_scale_to_seconds": float(timestamp_scale),
    "motion_missing_frames": int(missing_mask.sum()),
    "motion_missing_fraction": float(missing_mask.mean()),
    "motion_quantile_thresholds": thresholds.tolist(),
    "timing": {k: float(v) for k, v in stage_times.items()},
}
```

iii. The AI explicitly added these as diagnostics. `CONVERSION_NOTES.md` discusses runtime profiling, visual processing summaries, and session-level sanity metadata, all of which support validation rather than downstream decoding.
