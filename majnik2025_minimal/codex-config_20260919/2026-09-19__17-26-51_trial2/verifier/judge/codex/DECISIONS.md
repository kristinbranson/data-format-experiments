# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every complete dated session under `jm*` subject directories, then loads each session's fluorescence, cell flags, Suite2p metadata, motion energy, and camera timestamps. It processes sessions sequentially and accumulates the converted trial lists.

ii.
```python
sessions = sorted(
    p for p in data_root.glob("jm*/20*")
    if (p / "suite2p/plane0/F.npy").is_file()
    and (p / "move_deve/motion_energy_glob.npy").is_file()
)
...
fluorescence = np.load(plane_dir / "F.npy")
iscell = np.load(plane_dir / "iscell.npy")
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
raw_motion = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
```

iii. The trajectory says the agent inspected the paper notes, repository, raw layout, and validator. It concluded that dated folders are sessions and the supplied Suite2p arrays are already Track2p-curated, matched cells. Requiring both primary signal files was intended to find complete sessions deterministically.

## 1-b. How are the data split into subjects?

i. A subject is the parent `jm*` directory of a discovered session. Unique parent names are sorted and mapped to integer indices.

ii.
```python
subjects = sorted({session.parent.name for session in sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
...
subject = session_dir.parent.name
subject_idx.append(subject_to_idx[subject])
```

iii. The agent treated each `jm*` folder as one mouse, consistent with the data layout it inspected. Sorting makes subject indices reproducible.

## 1-c. How are the data split into sessions?

i. Each matching dated directory `jm*/20*` containing the required fluorescence and motion files is one session. The full path list is sorted globally.

ii.
```python
def discover_sessions(data_root: Path) -> list[Path]:
    sessions = sorted(
        p for p in data_root.glob("jm*/20*")
        if (p / "suite2p/plane0/F.npy").is_file()
        and (p / "move_deve/motion_energy_glob.npy").is_file()
    )
```

iii. The agent identified every dated directory as a daily session and used file checks to avoid treating unrelated or incomplete directories as recordings.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping 60-second trials. At 30 Hz these contain 1,800 source frames, or 180 bins after 10-frame averaging. Rather than discard a remainder, the agent requires every source session to contain an exact integer number of trials.

ii.
```python
trial_frames = int(TRIAL_SECONDS * FS_HZ)
trial_bins = trial_frames // AVERAGE_FRAMES
if n_frames % trial_frames:
    raise ValueError(...)
...
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
```

iii. The 60-second segmentation directly follows the decoder instruction. The trajectory reports that all 41 actual sessions produced complete 20- or 30-trial recordings, so the strict divisibility check did not discard data.

## 1-e. How are trials filtered based on quality controls?

i. No individual trials are quality-filtered. Sessions with malformed arrays, unexpected sampling rate, unfiltered ROI flags, or incomplete trial length cause an error instead of selective trial removal.

ii.
```python
if not np.isclose(float(ops["fs"]), FS_HZ):
    raise ValueError(...)
if n_frames % trial_frames:
    raise ValueError(...)
```

iii. The agent found no paper-defined trial QC for this continuous recording and verified that the complete dataset satisfied its invariants.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived only from `suite2p/plane0/F.npy`. `iscell.npy` and `ops.npy` are loaded for validation, not as neural values. `Fneu.npy` is not used.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy")
...
neural_binned = average_consecutive(
    baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
)
```

iii. The agent said it traced the repository's actual Track2p call site and found that the processing routine retained its default neuropil coefficient of zero. It therefore deliberately avoided substituting Suite2p's usual 0.7 neuropil subtraction.

## 2-b. How is the `neural` data processed?

i. `F` is Gaussian-smoothed along time (sigma 10 frames), subjected to a 60-second minimum then maximum filter to estimate a maximin baseline, and the baseline is subtracted. The result is averaged in non-overlapping groups of 10 frames and converted to `float32`; there is no baseline division or deconvolution.

ii.
```python
smoothed = gaussian_filter(
    fluorescence, sigma=(0.0, BASELINE_SIGMA_FRAMES), mode="reflect"
)
window = int(BASELINE_WINDOW_SECONDS * FS_HZ)
baseline = minimum_filter1d(smoothed, size=window, axis=1, mode="reflect")
baseline = maximum_filter1d(baseline, size=window, axis=1, mode="reflect")
return fluorescence - baseline
```

iii. The trajectory states that the agent found the repository's “dF/F0” implementation actually returns fluorescence minus the maximin baseline, with Gaussian sigma 10 and a 60-second window, and chose to reproduce it exactly before applying the paper's 10-frame decoding average.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed during conversion. The agent asserts that every row is already marked as a cell in `iscell.npy`; a session fails if that prefiltered Track2p assumption is false.

ii.
```python
if len(iscell) != fluorescence.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(
        f"{session_dir} is not the expected prefiltered Track2p output"
    )
```

iii. The agent concluded from the source layout and repository that these files already contain only cells tracked through every day for a mouse, in matched row order, so additional ROI filtering or rematching would be inappropriate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event. Trial zero begins at session time zero and later trials are contiguous slices. Metadata describes alignment as the start of each artificial 60-second trial, with offsets 0 to 60 seconds.

ii.
```python
"temporal_alignment_event": "start of each consecutive 60-second trial",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The recording is continuous and the instruction creates artificial fixed-duration trials. The agent therefore used each segment boundary as the only meaningful per-trial alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The source rate is 30 Hz. Neural and motion signals are separately averaged over non-overlapping groups of 10 consecutive samples, producing 3 Hz data and 333.333 ms bins.

ii.
```python
FS_HZ = 30.0
AVERAGE_FRAMES = 10
...
"time_bin_size": 1000.0 * AVERAGE_FRAMES / FS_HZ,
```

iii. The agent cited the paper's decoding analysis, which averages both dF/F and behavior in bins of 10 timestamps, and applied this before categorizing motion.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from the index of each averaged neural bin and the fixed 30 Hz sampling rate, rather than read from a raw timestamp file.

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32)
    * (AVERAGE_FRAMES / FS_HZ)
)
```

iii. The agent validated `ops["fs"] == 30` and used the regular imaging clock, making bin index times equivalent to elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Zero-based bin indices are multiplied by `10/30` seconds. The resulting continuous per-session vector uses left bin edges, remains `float32`, and is sliced into trials without resetting at trial boundaries.

ii.
```python
elapsed_seconds = (
    np.arange(neural_binned.shape[1], dtype=np.float32)
    * (AVERAGE_FRAMES / FS_HZ)
)
session_input.append(elapsed_seconds[None, start:stop])
```

iii. This directly represents the requested time elapsed from session start and preserves session position as decoder context.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is created with exactly one value per binned neural column, then sliced with the same `start:stop` indices as neural data.

ii.
```python
session_neural.append(neural_binned[:, start:stop])
session_input.append(elapsed_seconds[None, start:stop])
```

iii. A shared bin clock and identical slicing guarantee sample-for-sample alignment; the validator confirmed all trial arrays have 180 timepoints.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion output comes from `move_deve/motion_energy_glob.npy`; absolute camera timestamps from `move_deve/tstamps.npy` are used to align it to imaging.

ii.
```python
raw_motion = np.load(motion_dir / "motion_energy_glob.npy")
timestamps = np.load(motion_dir / "tstamps.npy")
aligned_motion, n_missing_camera_frames = align_motion_to_imaging(
    raw_motion, timestamps, n_frames
)
```

iii. The agent inspected array metadata and chose timestamps as the most direct representation of camera sampling and dropped frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion samples are linearly interpolated from the camera clock to a regular imaging-length clock, averaged in 10-frame groups, and categorized with session-specific quintile cut points.

ii.
```python
imaging_clock = timestamps[0] + np.arange(n_imaging_frames) * frame_step
aligned = np.interp(imaging_clock, timestamps, motion)
...
motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)
quintile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
```

iii. The agent intended timestamp interpolation to retain every neural frame while repairing documented camera gaps. It followed the paper by denoising before discretization and the task by selecting categories independently per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four within-session quantiles (20%, 40%, 60%, 80%) are calculated from the complete aligned, averaged session. `searchsorted(..., side="right")` assigns integer labels 0–4, with ties at a boundary entering the upper category.

ii.
```python
quintile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
motion_classes = np.searchsorted(
    quintile_edges, motion_binned, side="right"
).astype(np.int64)
```

iii. This implements the requested five equal-percentile bins per session. The trajectory reports exactly 20% in every category for these data.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The median camera timestamp increment defines a frame step. A regular clock of exactly `n_imaging_frames` is built from the first timestamp, and motion is interpolated onto it. Neural and aligned motion are then averaged by the same factor and sliced with identical trial bounds.

ii.
```python
intervals = np.diff(timestamps)
frame_step = float(np.median(intervals))
imaging_clock = timestamps[0] + np.arange(n_imaging_frames) * frame_step
aligned = np.interp(imaging_clock, timestamps, motion)
...
session_output.append(motion_classes[None, start:stop])
```

iii. The agent reasoned that timestamp interpolation handles missing camera samples more generally than positional insertion and preserves the full neural recording. Successful decoder accuracy was used as a sanity check of usable alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera samples are bridged by timestamp-based linear interpolation. Invalid dimensions, unequal motion/timestamp lengths, nonmonotonic timestamps, unexpected ROI flags or sampling rate, and partial trials raise explicit errors. Interpolation endpoints use NumPy's boundary behavior.

ii.
```python
if motion.ndim != 1 or timestamps.ndim != 1 or len(motion) != len(timestamps):
    raise ValueError(...)
if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
    raise ValueError(...)
aligned = np.interp(imaging_clock, timestamps, motion)
```

iii. The agent sought to retain complete imaging sessions while repairing documented camera drops. It reported recovering sessions with many missing frames and verified all converted dimensions afterward.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session Gaussian/minimum/maximum baseline filtering across every neuron is the main compute cost. Loading hundreds of megabytes of arrays and serializing the roughly 396 MB pickle are also substantial. Timestamp interpolation and quantile labeling are comparatively cheap.

ii.
```python
smoothed = gaussian_filter(...)
baseline = minimum_filter1d(...)
baseline = maximum_filter1d(...)
```

iii. The agent did not provide a formal profile, but deliberately used SciPy vectorized filters and processed one session at a time, indicating recognition that full neural preprocessing and memory/I/O dominate.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop could be replaced by reshape/transposition followed by list conversion, although the required nested-list output limits the benefit. Session iteration cannot be cleanly vectorized because neuron counts and session lengths differ. The agent already replaced per-gap insertion with one vectorized `np.interp` call.

ii.
```python
for trial in range(n_trials):
    start = trial * trial_bins
    stop = start + trial_bins
    session_neural.append(neural_binned[:, start:stop])
```

iii. The trajectory contains no explicit loop-efficiency discussion. The implementation itself uses vectorized filtering, averaging, interpolation, and category assignment; only organizational loops remain.

## 6-c. What processing does the code repeat multiple times?

i. Loading, validation, baseline correction, motion alignment, averaging, quantile calculation, and trial assembly repeat once per session. These are necessary because sessions have distinct arrays, subjects, gap patterns, and percentile thresholds; no same intermediate is redundantly recomputed.

ii.
```python
for session_i, session_dir in enumerate(sessions, start=1):
    ...
    neural_binned = average_consecutive(...)
    ...
    motion_binned = average_consecutive(...)
```

iii. The agent intentionally processed each session independently so quintiles remain session-specific and deleted large intermediates to avoid retaining repeated raw data.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No major signal transformation is computed and discarded. `iscell` and `ops` are used only for validation, missing-frame counts and quintile edges only for metadata, and detailed `session_info` is not consumed by the decoder. These add modest work but support provenance and correctness checks.

ii.
```python
iscell = np.load(...)
ops = np.load(..., allow_pickle=True).item()
...
"inferred_missing_camera_frames": n_missing_camera_frames,
"motion_energy_quintile_edges": quintile_edges.tolist(),
```

iii. The trajectory emphasizes validation and diagnostics, so these ancillary computations were deliberate safeguards rather than accidental waste.
