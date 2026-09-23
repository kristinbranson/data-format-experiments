# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subject directories under `/app/data`, then all session subdirectories under each subject, and requires each session to contain `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy`. During conversion it also opens `spks.npy`, `iscell.npy`, `stat.npy`, `ops.npy`, `tstamps.npy`, and `interframe_int.npy` for validation/alignment. Trials are not loaded directly from disk because the recordings are continuous; they are created later after preprocessing and binning.

ii.
```python
def discover_sessions() -> tuple[list[str], list[SessionRef]]:
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    sessions: list[SessionRef] = []
    for subject in subjects:
        subject_dir = DATA_ROOT / subject
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            required = (
                session_dir / "suite2p" / "plane0" / "F.npy",
                session_dir / "suite2p" / "plane0" / "Fneu.npy",
                session_dir / "move_deve" / "motion_energy_glob.npy",
            )
            if not all(p.exists() for p in required):
                raise FileNotFoundError(f"Incomplete session directory: {session_dir}")
            sessions.append(SessionRef(subject, session_dir.name, session_dir))
```

```python
f = np.load(plane / "F.npy", mmap_mode="r")
fneu = np.load(plane / "Fneu.npy", mmap_mode="r")
spks = np.load(plane / "spks.npy", mmap_mode="r")
iscell = np.load(plane / "iscell.npy", mmap_mode="r")
stat = np.load(plane / "stat.npy", allow_pickle=True)
ops = np.load(plane / "ops.npy", allow_pickle=True).item()
```

iii. In `CONVERSION_NOTES.md`, the AI says the release contains six subject folders, each with daily recording sessions, and that the packaged traces are already longitudinally matched Track2p/Suite2p outputs. The trajectory also states the mapping was fixed to “all 41 valid sessions.”

## 1-b. How are the data split into subjects?

i. Subjects are the top-level directories in `/app/data`, sorted lexically, and their names are used directly as subject IDs.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
```

```python
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The notes document six subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and say subject ordering is deterministic and lexical.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject folder. Sessions are sorted chronologically by directory name and stored as separate session entries in the output lists.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
    sessions.append(SessionRef(subject, session_dir.name, session_dir))
```

```python
for index, ref in enumerate(session_refs):
    converted, info = convert_session(
        ref, show_processing=show_processing and index < 2
    )
    neural.append(converted["neural"])
    inputs.append(converted["input"])
    outputs.append(converted["output"])
```

iii. The notes say sessions are “chronologically named daily recording sessions” and that the final ordering is “subjects lexical; sessions and trials chronological.”

## 1-d. How are the data split into trials?

i. The AI treats each recording as a continuous session, downsamples it to 3 Hz by averaging non-overlapping 10-frame blocks, then slices the binned session into contiguous non-overlapping 60-second trials. Because `TRIAL_BINS = 180`, each trial contains 180 time bins. The AI assumes every session divides exactly into full 60-second trials and raises an error otherwise.

ii.
```python
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * OUTPUT_FS_HZ)
```

```python
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(f"Session does not divide into full 60-s trials: {ref.path}")
```

```python
n_trials = n_binned // TRIAL_BINS
for trial in range(n_trials):
    sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
    neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
    input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
    output_trial = np.ascontiguousarray(labels[sl][None, :], dtype=np.int64)
```

iii. The notes say there is no native trial structure, so 60-second non-overlapping segments are a downstream requirement. They also record that all sessions in the released data divide exactly into 20 or 30 full trials, so no remainder handling was needed in practice.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any per-trial quality filter. It only rejects malformed sessions upstream if source arrays are inconsistent or if a session cannot be divided into full 60-second trials.

ii.
```python
if f.ndim != 2 or f.shape != fneu.shape or f.shape != spks.shape:
    raise ValueError(...)
...
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(f"Session does not divide into full 60-s trials: {ref.path}")
```

iii. The notes explicitly say there are no native trial curation rules and that “the only necessary curation is repairing documented missing behavior frames.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii.
```python
f = np.load(plane_dir / "F.npy", mmap_mode="r")
fneu = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
```

```python
corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu
```

iii. The notes say the paper’s decoder input should be baseline-corrected fluorescence rather than deconvolved spikes, and specifically map `F.npy` plus `Fneu.npy` to the final `neural` field.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction with coefficient 0.7, applies a Suite2p-style maximin baseline estimate by Gaussian smoothing followed by 60-second minimum and maximum filters, subtracts that baseline, then averages non-overlapping 10-frame groups.

ii.
```python
corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu
baseline = gaussian_filter(
    corrected_neuropil, sigma=(0.0, BASELINE_SIGMA_FRAMES)
)
baseline = minimum_filter1d(
    baseline, size=baseline_window, axis=1
)
baseline = maximum_filter1d(
    baseline, size=baseline_window, axis=1
)
activity = corrected_neuropil - baseline
result[start:stop] = activity.reshape(
    stop - start, n_binned, DOWNSAMPLE
).mean(axis=2)
```

iii. In Step 4 and Step 5 of the notes, the AI justifies this as paper-style baseline-corrected fluorescence using the session `ops` defaults (`neucoeff=0.7`, sigma 10 frames, 60-second window), and says it intentionally does not divide by baseline because the supplied reference GUI implementation subtracts the baseline rather than computing literal dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any extra neuron filter beyond requiring that the packaged session already satisfies Track2p/Suite2p curation: every ROI must have `iscell[:,0] == 1` and `iscell[:,1] > 0.5`. All rows are kept once the session passes those checks.

ii.
```python
iscell = np.load(plane / "iscell.npy", mmap_mode="r")
...
if not np.all(iscell[:, 0] == 1) or not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"Packaged neurons do not all pass Track2p curation: {ref.path}")
```

iii. The notes say the released matrices are already restricted to cells tracked across all days and already pass the 0.5 `iscell` threshold, so a second filtering pass would incorrectly discard valid rows.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the start of each 60-second segment as the trial alignment point. Neural data are processed on the full continuous session first and then trial slices are taken as consecutive 180-bin windows.

ii.
```python
"metadata": {
    "temporal_alignment_event": (
        "start of each non-overlapping 60-second segment"
    ),
    "off_start": 0.0,
    "off_end": 60.0,
```

```python
for trial in range(n_trials):
    sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
    neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
```

iii. The notes justify this by saying there is no native event structure and that the requested 60-second segmentation is the relevant downstream unit. They also emphasize filtering the full continuous session before slicing so trial boundaries do not create filtering artifacts.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 333.333... ms bins (3 Hz). The AI applies non-overlapping temporal averaging over 10 native 30 Hz frames before trial splitting.

ii.
```python
NATIVE_FS_HZ = 30
DOWNSAMPLE = 10
OUTPUT_FS_HZ = NATIVE_FS_HZ / DOWNSAMPLE
TIME_BIN_MS = 1000.0 / OUTPUT_FS_HZ
```

```python
result[start:stop] = activity.reshape(
    stop - start, n_binned, DOWNSAMPLE
).mean(axis=2)
...
motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
```

iii. The notes repeatedly cite the methods statement that decoding averaged both fluorescence and behavior in bins of 10 consecutive timestamps, and say the same 3 Hz grid must be used for both streams.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not loaded from an explicit raw time variable. The AI derives it from the bin index plus the known 30 Hz sampling rate and 10-frame bin width.

ii.
```python
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
```

iii. The notes say this input is task-specific rather than a paper-provided variable, and that the known constant sampling rate makes reconstruction from frame indices sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one value per 10-frame bin as the mean time of that bin’s native frames, using a `+4.5` frame offset, and keeps the time continuous across the whole session rather than resetting it at trial boundaries.

ii.
```python
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
```

```python
input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
```

iii. The notes explicitly justify this as “mean elapsed times of each 10-frame block” and say that because the decoder input is “time elapsed from session start,” the values should remain global across trials.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI builds the time vector on the same 10-frame-binned grid as the neural data and slices both with the same trial windows, so each time sample corresponds to the same binned interval as the neural columns.

ii.
```python
n_binned = n_frames // DOWNSAMPLE
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
```

```python
sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
```

iii. The notes say the time input should be computed “at the mean time of the represented native frames” and then split with the same 60-second cuts as neural and behavior.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from `move_deve/motion_energy_glob.npy`, with `tstamps.npy` and `interframe_int.npy` used to detect and repair dropped camera frames before alignment to neural frames.

ii.
```python
raw = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. The notes describe `motion_energy_glob.npy` as the provided scalar motion-energy trace and say timestamp gaps exactly explain the documented camera-frame deficits.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first aligns motion to the neural frame count by reconstructing expected frame positions from median camera intervals and interpolating missing positions, then averages motion in 10-frame bins, then computes per-session quintile labels from the binned values.

ii.
```python
median_interval = float(np.median(positive))
interval_steps = np.maximum(1, np.rint(intervals / median_interval).astype(int))
...
observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
aligned = np.interp(
    np.arange(expected_frames, dtype=np.float64), observed_positions, raw
)
```

```python
motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(quantile_edges, motion_binned, side="right").astype(
    np.int64
)
```

iii. The notes justify this by citing the release guidance that missing camera frames should be located from timestamps and either marked missing or interpolated, and by emphasizing that continuous motion should be averaged before discretization so labels are assigned on the final decoder time grid.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses session-specific quintiles of the 10-frame-averaged motion trace. It thresholds at the 20th, 40th, 60th, and 80th percentiles and assigns class labels 0 through 4 with `searchsorted`.

ii.
```python
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(quantile_edges, motion_binned, side="right").astype(
    np.int64
)
counts = np.bincount(labels, minlength=5)
expected = len(labels) // 5
if not np.array_equal(counts, np.full(5, expected)):
    raise ValueError(f"Motion quintiles are not equal: counts={counts.tolist()}")
```

iii. The notes say the decoder task specifically requires “five equal-percentile bins, selected per session,” so the AI intentionally discretizes after alignment and binning and then verifies exact 20% class balance.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion directly to the neural frame indices. If the motion array is short, it uses timestamp-derived observed frame positions and interpolation to expand it to the neural frame count; after that, both streams are binned with the same 10-frame means and trial-sliced with the same windows.

ii.
```python
aligned_motion, motion_info = align_motion(ref.path / "move_deve", n_frames)
motion_binned, quantile_edges, labels = bin_and_discretize_motion(aligned_motion)
...
if len(motion_binned) != n_binned or len(labels) != n_binned:
    raise AssertionError("Processed behavior length mismatch")
```

```python
sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
output_trial = np.ascontiguousarray(labels[sl][None, :], dtype=np.int64)
```

iii. The notes say imaging and behavior are nominally frame-synchronized at 30 Hz, and that only documented missing camera frames should be repaired. They also note that several full-length sessions have timestamp anomalies but are left unchanged because no camera samples are actually absent.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI treats missing camera frames as the main minor data issue. It detects them from timestamp gaps when the motion trace is shorter than the neural trace, interpolates to the expected frame count, validates array consistency aggressively, and leaves full-length anomalous timestamp streams untouched. It also raises errors for malformed sessions rather than silently continuing.

ii.
```python
if len(raw) != len(timestamps) or len(intervals) != len(raw) - 1:
    raise ValueError(f"Malformed motion arrays in {move_dir}")
if len(raw) > expected_frames:
    raise ValueError(f"Motion has more frames than neural data in {move_dir}")
```

```python
if missing:
    observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
    if observed_positions[-1] != expected_frames - 1:
        raise ValueError(...)
    aligned = np.interp(
        np.arange(expected_frames, dtype=np.float64), observed_positions, raw
    )
else:
    aligned = raw.copy()
```

iii. The notes say the release README permits interpolation for missing camera frames and that leaving the full-length jm046 anomalies untouched is a deliberate choice because the length criterion indicates no samples are actually missing in those sessions.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is the continuous-session neural preprocessing: neuropil subtraction, Gaussian smoothing, min/max baseline filtering, and 10-frame averaging across all neurons. Optional diagnostic plotting is extra overhead but only for up to two sessions.

ii.
```python
for start in range(0, n_neurons, NEURON_CHUNK):
    ...
    baseline = gaussian_filter(...)
    baseline = minimum_filter1d(...)
    baseline = maximum_filter1d(...)
    activity = corrected_neuropil - baseline
    result[start:stop] = activity.reshape(
        stop - start, n_binned, DOWNSAMPLE
    ).mean(axis=2)
```

```python
if show_processing:
    plot_path = make_processing_plot(...)
```

iii. In Step 6 and Step 9, the notes identify fluorescence conversion as the core runtime cost and report per-session neural processing times. The AI explicitly optimized this path with chunking and memory mapping.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main numerical work. The remaining Python loops are mostly control-flow loops over sessions, neuron chunks, and trial packaging. The clearest remaining micro-optimization is the per-trial append loop that slices one trial at a time; unlike the human reference, there is no repeated `np.insert` loop for dropped-frame interpolation because alignment uses a single `np.interp` call.

ii.
```python
for trial in range(n_trials):
    sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
    neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
    input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
    output_trial = np.ascontiguousarray(labels[sl][None, :], dtype=np.int64)
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The notes emphasize that vectorized SciPy filters, reshape/mean downsampling, and vectorized interpolation were already chosen as speedups. They do not single out a major remaining numerical loop as a bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats validation and conversion per session, and baseline filtering per neuron chunk within a session. It also repeatedly computes `subjects.index(ref.subject)` during dataset assembly. The AI intentionally avoids a more problematic repetition: it processes each full session once before slicing trials instead of repeating filtering separately for every trial.

ii.
```python
for index, ref in enumerate(session_refs):
    converted, info = convert_session(
        ref, show_processing=show_processing and index < 2
    )
```

```python
for start in range(0, n_neurons, NEURON_CHUNK):
    ...
```

```python
subject_idx.append(subjects.index(ref.subject))
```

iii. The notes explicitly call out “Process full session once, then slice trials” as a speedup that avoids repeated filtering calls and trial-edge artifacts.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several arrays mainly for validation and bookkeeping rather than for the final decoder tensors: `spks.npy`, `iscell.npy`, `stat.npy`, most of `ops.npy`, `timestamps`, and optional processing-plot traces. It also stores example traces for plotting from the first chunk only. These steps support sanity checks and documentation but are not required for downstream decoding once the converted arrays are built.

ii.
```python
spks = np.load(plane / "spks.npy", mmap_mode="r")
iscell = np.load(plane / "iscell.npy", mmap_mode="r")
stat = np.load(plane / "stat.npy", allow_pickle=True)
ops = np.load(plane / "ops.npy", allow_pickle=True).item()
```

```python
if start == 0:
    example = {
        "raw_f": raw_f[0].copy(),
        "raw_fneu": raw_fneu[0].copy(),
        "neuropil_corrected": corrected_neuropil[0].copy(),
        "baseline": baseline[0].copy(),
        "activity": activity[0].copy(),
        "activity_binned": result[0].copy(),
    }
```

iii. The notes frame these as deliberate sanity-check and audit choices rather than essential conversion steps. Step 13 also says the AI kept extra diagnostic artifacts and organized them into cache/documentation outputs.
