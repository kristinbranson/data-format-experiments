# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory under the data root as a subject and every directory below each subject as a session. For each session it loads `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. `build_dataset` eagerly loads every discovered session unless an explicit subject subset is supplied.

ii.
```python
available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
...
for subject in subjects:
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        out.append((subject, session_dir))
...
loaded_sessions = [load_session(subject, session_dir)
                   for subject, session_dir in session_paths(data_dir, selected_subjects)]
```

iii. The trajectory shows that the agent inspected the data directory, README files, notebooks, and array shapes across subjects before implementing this traversal. It treated the released directory hierarchy as authoritative and added an optional explicit subject selector for testing.

## 1-b. How are the data split into subjects?

i. Each top-level data directory is one mouse. Subject names are sorted; the final `subject_idx` is the position of each session's subject in the sorted unique subject list.

ii.
```python
subjects = sorted({session["subject"] for session in loaded_sessions})
...
subject_idx.append(subjects.index(session["subject"]))
```

iii. The agent inspected all top-level directories and confirmed six mouse IDs and stable neuron counts per mouse across sessions. It therefore used the existing directory boundary as the subject boundary.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory under a subject is one session (daily recording), and each becomes one element of the outer `neural`, `input`, and `output` lists.

ii.
```python
for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
    out.append((subject, session_dir))
...
for session in loaded_sessions:
    neural.append(neural_trials)
```

iii. The trajectory shows inspection of session names, durations, and file layouts. The agent concluded that these subdirectories consistently represent recording days and preserved their sorted order.

## 1-d. How are the data split into trials?

i. The agent creates consecutive, non-overlapping **120-second** artificial trials. At 3 Hz this is 360 bins. It requires a session to divide exactly into these blocks rather than discarding a partial final block.

ii.
```python
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)
...
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)
...
start = trial_idx * TRIAL_BINS
end = start + TRIAL_BINS
```

iii. The trajectory records experiments using `TRIAL_SEC = 120.0`, but contains no explicit rationale for choosing two minutes. This conflicts with the evaluation instruction's 60-second requirement; it appears to have been an unsupported interpretation of continuous recordings.

## 1-e. How are trials filtered based on quality controls?

i. No individual trials are quality-filtered. A whole session is rejected if stream shapes disagree, dropped frames cannot be reconciled, or its binned length is not divisible by 360.

ii.
```python
if neural_binned.shape[1] != motion_binned.shape[0]:
    raise ValueError(...)
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)
```

iii. The agent found fixed session lengths and built strict consistency checks. Neither its trajectory nor code identifies a paper-defined trial-quality measure, which is reasonable for artificial blocks of continuous data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p `F.npy` (ROI fluorescence), `Fneu.npy` (neuropil fluorescence), and preprocessing parameters in `ops.npy`. `iscell.npy` is loaded only for a consistency check.

ii.
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
```

iii. The trajectory shows that the agent compared `F`, `Fneu`, and `spks`, inspected the saved Suite2p settings, and selected baseline-corrected fluorescence because it matched the paper's dF/F-style decoding preprocessing more closely than deconvolved spikes.

## 2-b. How is the `neural` data processed?

i. It casts to float32, subtracts neuropil using the saved `neucoeff`, performs Suite2p-style baseline subtraction (normally Gaussian smoothing followed by min/max filters for `maximin`), and averages non-overlapping groups of 10 frames.

ii.
```python
fc = fluorescence.astype(np.float32) - float(ops["neucoeff"]) * neuropil.astype(np.float32)
flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
flow = minimum_filter1d(flow, win)
flow = maximum_filter1d(flow, win)
return (fc - flow).astype(np.float32)
...
neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The agent inspected `ops.npy` and prototyped this computation against raw fluorescence and spikes. It chose saved Suite2p parameters to reproduce the source pipeline and 10-frame averaging because the methods specify that denoising for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROIs are removed. The code asserts that all Track2p-exported ROIs have `iscell[:,1] > 0.5`; if not, it rejects the session rather than filtering cells.

ii.
```python
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The agent examined `iscell` values throughout the release and found the Track2p-exported arrays already contained accepted cells. It therefore avoided another filtering pass that could break longitudinal cell correspondence.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins are sliced into consecutive 120-second blocks. The metadata calls each block's start the alignment event and assigns offsets 0 to 120 seconds.

ii.
```python
neural_trials.append(neural_binned[:, start:end])
...
"temporal_alignment_event": "start of each consecutive 2-minute block within a session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SECONDS,
```

iii. The agent recognized that the continuous recording has no stimulus event, so block starts are the only natural artificial alignment points. It did not justify choosing two-minute rather than required one-minute blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Raw 30 Hz samples are averaged in non-overlapping groups of 10, producing 3 Hz data and 333.333 ms bins. Short tails below 10 frames are truncated.

ii.
```python
FRAME_RATE_HZ = 30.0
FRAME_BIN_SIZE = 10
BIN_SIZE_MS = FRAME_BIN_SIZE / FRAME_RATE_HZ * 1000.0
return x[:, : nbins * bin_size].reshape(...).mean(axis=2)
```

iii. The trajectory and summary explicitly connect this decision to the methods' instruction to average neural and behavior traces over 10 consecutive timestamps for decoding.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthesized from the binned sample index and the assumed 30 Hz frame rate, not loaded from raw timestamps. It represents seconds from the start of each session.

ii.
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. The agent treated each recording session as an experiment and used the known regular sampling rate. The trajectory shows it inspected video timestamps, but used them only to diagnose dropped behavior frames rather than define neural time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It adds 0.5 to every bin index and multiplies by 10/30 seconds, yielding bin-center times (0.1667, 0.5, ... seconds), then casts to float32.

ii.
```python
(np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
```

iii. The conversion summary says “elapsed time from session start at bin centers.” Although no detailed reasoning is recorded, bin centers are a defensible timestamp for values formed by averaging each 10-frame interval.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One time value is generated for every binned neural column, and identical `[start:end]` slices are applied to both.

ii.
```python
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
```

iii. The agent used common indices after binning, ensuring exact shape and index alignment rather than independently resampling time.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses `move_deve/motion_energy_glob.npy`; `interframe_int.npy` supplies timing gaps used to infer missing video frames.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. The trajectory shows searches through source code and notebooks for motion variables and direct inspection of motion, timestamp, and interframe arrays. The global motion-energy trace is the paper-provided behavioral feature.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing frames are linearly interpolated, the trace is averaged over 10-frame groups, each session is z-scored, then all session z-scores are pooled to compute global quintile boundaries and digitized into integers 0–4.

ii.
```python
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
motion_z = ((motion_binned - motion_binned.mean()) /
            (motion_binned.std() + 1e-8)).astype(np.float32)
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES)
```

iii. The original prompt in the trajectory said “normalized and discretized,” which explains the per-session z-score. The agent also tested decoder accuracy under candidate processing. However, its global pooling does not follow the evaluated instruction/reference choice of five equal-percentile bins selected separately per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds are the 20th, 40th, 60th, and 80th percentiles of the pooled collection of all per-session z-scored bins. `np.digitize(..., right=False)` creates labels 0–4.

ii.
```python
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8])
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES)
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
```

iii. The trajectory contains no explicit justification for global rather than session-local thresholds. The printed summary identifies the choice, suggesting it was deliberate, likely to keep shared class meanings after normalization, but it violates “selected per session.”

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If lengths differ, missing counts are inferred by rounding each interframe interval relative to its median. The code validates the total, preallocates an imaging-length array, linearly interpolates every missing position, bins both streams by the same factor, verifies equal length, and slices both with common trial boundaries.

ii.
```python
frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
...
aligned[dst : dst + gap_missing] = np.linspace(
    motion_energy[src], motion_energy[next_src], int(gap_missing) + 2
)[1:-1]
...
output_trials.append(motion_classes[start:end][None, :])
```

iii. The trajectory includes detailed examination of gap ratios and confirms inferred missing counts against imaging-minus-motion lengths. This led to a robust interpolation onto the imaging-frame grid before joint binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Shape inconsistencies cause explicit errors. Missing motion frames are inferred and interpolated only when the inferred total exactly matches the known length deficit. Tail samples below one 10-frame bin are discarded. Non-divisible trial tails cause an error, not truncation.

ii.
```python
if missing_total != expected_missing:
    raise ValueError("Could not reconcile missing motion frames...")
...
nbins = x.shape[0] // bin_size
return x[: nbins * bin_size]...
```

iii. The trajectory shows the agent quantified missing frames across every session and inspected mild and large gaps. It preferred fail-fast validation where the data could not be reconciled, while interpolation preserves aligned samples for recognized dropped camera frames.

## 6-a. What are the most time-consuming steps of the code?

i. The main costs are loading every full-session array and applying Gaussian, minimum, and maximum filters to every neuron's full trace. Full-dataset validation and summary traversal add smaller costs.

ii.
```python
flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
flow = minimum_filter1d(flow, win)
flow = maximum_filter1d(flow, win)
```

iii. The trajectory's long-running conversion/prototype commands center on baseline preprocessing, supporting this assessment. It also eagerly retains all processed sessions, trading memory for straightforward global threshold calculation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial slicing must still create list elements, but the missing-gap loop could be replaced by timestamp-based vectorized interpolation. `subjects.index(...)` inside the session loop could use a precomputed dictionary, and metadata dictionaries could be assembled with comprehensions. The agent already improves on repeated `np.insert` by preallocating the aligned array.

ii.
```python
for gap_missing in missing_after:
    ...
for trial_idx in range(n_trials):
    ...
subject_idx.append(subjects.index(session["subject"]))
```

iii. No explicit efficiency justification appears in the trajectory, but its prototype evolved into a preallocated interpolation implementation, indicating awareness that repeated array insertion would be expensive.

## 6-c. What processing does the code repeat multiple times?

i. Each session independently repeats file loading, baseline correction, binning, z-scoring, validation, and metadata extraction. It also repeatedly performs linear subject lookup and later traverses sessions again for assembly, aggregate metadata, validation, and printing.

ii.
```python
loaded_sessions = [load_session(...) for ...]
...
for session in loaded_sessions:
    ... subjects.index(session["subject"])
```

iii. Most repetition is necessary per-session processing. The two-pass design is driven by the decision to compute global motion thresholds after all sessions have been normalized.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It retains continuous `motion_binned` although only `motion_z` is used for classes; records `missing_motion_frame_indices` but drops that field when building final metadata; computes extensive summary statistics and session metadata unused by decoder training; and traverses the full result for validation and printing.

ii.
```python
"motion_binned": motion_binned,
"motion_z": motion_z,
...
return aligned, {"missing_motion_frame_indices": inserted_indices}
...
valid, errors, warnings = verify_data_format(data)
print_data_summary(data)
```

iii. The trajectory emphasizes sanity checks against the paper and format validation, so this extra work was intentional diagnostic/provenance work rather than required decoder processing. It is useful during conversion but discarded or ignored downstream.
