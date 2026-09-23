# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter discovers every directory under `/app/data` as a subject, sorts subjects and their session directories, verifies each session has `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy`, and then converts every discovered session (41 in full mode). Neural arrays are memory-mapped; motion and timing arrays are loaded normally. It also loads `spks`, `iscell`, `stat`, and `ops` to validate source invariants.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
for subject in subjects:
    subject_dir = DATA_ROOT / subject
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        required = (
            session_dir / "suite2p" / "plane0" / "F.npy",
            session_dir / "suite2p" / "plane0" / "Fneu.npy",
            session_dir / "move_deve" / "motion_energy_glob.npy",
        )
        if not all(p.exists() for p in required):
            raise FileNotFoundError(...)
        sessions.append(SessionRef(subject, session_dir.name, session_dir))
```

iii. The agent justified deterministic lexical/chronological traversal as reproducible and retained all 41 valid released sessions, including valid 30-minute recordings that conflict with the paper's stated duration, rather than silently cropping source data. Memory mapping was chosen to reduce memory use.

## 1-b. How are the data split into subjects?

i. Each directory immediately below the data root is treated as a mouse. Subject identifiers are sorted lexically, and each session's `subject_idx` is the position of its folder name in that list.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
...
subject_idx.append(subjects.index(ref.subject))
```

iii. The notes identify the six source folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) as mice and cite the release directory structure. Lexical ordering makes the mapping deterministic.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory within a subject directory is one continuous recording session. A `SessionRef` retains subject, session directory name, and path. Each converted session becomes one outer-list element.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
    sessions.append(SessionRef(subject, session_dir.name, session_dir))
...
for index, ref in enumerate(session_refs):
    converted, info = convert_session(ref, ...)
    neural.append(converted["neural"])
```

iii. The agent found 41 daily recordings and used chronological directory names/order, consistent with the release organization and the reference's session interpretation.

## 1-d. How are the data split into trials?

i. Continuous sessions are split chronologically into non-overlapping 60-second trials after 10-frame averaging. At 3 Hz this is exactly 180 bins. The source validator requires every session to divide exactly into complete trials, so the code does not implement remainder truncation.

ii.
```python
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * OUTPUT_FS_HZ)
...
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(...)
...
for trial in range(n_trials):
    sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
```

iii. There is no natural trial structure, so the requested 60-second segmentation is applied. The agent verified that all released sessions contain exactly 20 or 30 complete trials and that no samples are lost.

## 1-e. How are trials filtered based on quality controls?

i. No individual trial is filtered. Instead, sessions must satisfy source-shape, cell-curation, sampling-rate, finiteness, divisibility, alignment, and output-balance checks; failure aborts conversion rather than dropping a trial.

ii.
```python
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(...)
...
if neural_trial.shape != (n_neurons, TRIAL_BINS):
    raise AssertionError("Neural trial shape mismatch")
```

iii. The notes report that all source sessions and resulting trials passed the checks, so no task- or paper-specified trial exclusion was warranted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from Suite2p `F.npy` and `Fneu.npy` in each session's `suite2p/plane0` directory. `spks.npy`, `iscell.npy`, `stat.npy`, and `ops.npy` are validation inputs, not the stored neural signal.

ii.
```python
f = np.load(plane_dir / "F.npy", mmap_mode="r")
fneu = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
...
corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu
```

iii. The agent chose the paper's fluorescence representation rather than raw fluorescence or Suite2p spikes because the paper describes fluorescence preprocessing for decoding.

## 2-b. How is the `neural` data processed?

i. The code computes `F - 0.7*Fneu`, Gaussian-smooths along time with sigma 10 frames, estimates a 60-second maximin baseline using minimum then maximum filters, subtracts that baseline, and averages non-overlapping groups of 10 frames. Filtering occurs over each full continuous session in 64-neuron chunks before trial slicing.

ii.
```python
corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu
baseline = gaussian_filter(
    corrected_neuropil, sigma=(0.0, BASELINE_SIGMA_FRAMES)
)
baseline = minimum_filter1d(baseline, size=baseline_window, axis=1)
baseline = maximum_filter1d(baseline, size=baseline_window, axis=1)
activity = corrected_neuropil - baseline
result[start:stop] = activity.reshape(
    stop - start, n_binned, DOWNSAMPLE
).mean(axis=2)
```

iii. This was justified as an explicit implementation of Suite2p's maximin preprocessing and the paper's 10-timestamp denoising. Full-session filtering avoids artificial trial-edge effects; chunking limits peak memory without changing results.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural rows are removed during conversion. The packaged data are asserted already to contain only curated longitudinal cells: all `iscell[:,0] == 1`, all cell probabilities exceed 0.5, and row counts agree across `F`, `Fneu`, `spks`, `iscell`, and `stat`. Processed values must also all be finite.

ii.
```python
if iscell.shape != (n_neurons, 2) or len(stat) != n_neurons:
    raise ValueError(...)
if not np.all(iscell[:, 0] == 1) or not np.all(iscell[:, 1] > 0.5):
    raise ValueError(...)
...
if not np.all(np.isfinite(result)):
    raise ValueError(...)
```

iii. The agent concluded that the release was already restricted to cells passing the paper's `>0.5` criterion and tracked across all days; filtering again would discard valid data or duplicate upstream curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The binned continuous neural stream is sliced at exact consecutive 60-second boundaries. Metadata describes the alignment event as the start of each artificial segment, with offsets 0 to 60 seconds.

ii.
```python
sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
...
"temporal_alignment_event": "start of each non-overlapping 60-second segment",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The recordings have no stimulus-alignment event. The agent therefore used the requested fixed segment boundary as the alignment event while preserving chronological order.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 30 Hz samples are averaged in non-overlapping groups of 10, yielding 3 Hz data and 333.333 ms bins. The same rebinning is applied to neural activity and motion before trial creation and motion categorization.

ii.
```python
NATIVE_FS_HZ = 30
DOWNSAMPLE = 10
OUTPUT_FS_HZ = NATIVE_FS_HZ / DOWNSAMPLE
TIME_BIN_MS = 1000.0 / OUTPUT_FS_HZ
```

iii. The paper says decoding analyses denoised fluorescence and behavior by averaging 10 consecutive timestamps. Paired averaging preserves the common temporal grid.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated analytically from the binned frame index, the native 30 Hz sampling rate, and the center offset of a 10-frame bin; it is not loaded from a raw timestamp variable.

ii.
```python
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
```

iii. The agent reasoned that imaging is uniformly sampled at 30 Hz and chose the mean acquisition time of the ten native samples represented by each converted bin.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For binned index `i`, time is `(10*i + 4.5)/30` seconds, producing bin-center times beginning at 0.15 seconds. Values are float32 and remain continuous from session start across trial boundaries.

ii.
```python
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
```

iii. The `4.5` offset is the average native-frame index within indices 0 through 9. The notes explicitly prefer this center time over resetting time at each trial.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One elapsed-time value is generated for every binned neural column, and the identical trial slice is applied to both streams.

ii.
```python
sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
```

iii. Shared indices and shape assertions guarantee one input timestamp per neural time bin, with no reset, gap, or overlap between trials.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output originates from `move_deve/motion_energy_glob.npy`. `tstamps.npy` and `interframe_int.npy` are used to locate missing camera samples and align motion to neural frames.

ii.
```python
raw = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

iii. The motion file is the release's precomputed global behavioral motion energy; timing files are needed because some video streams have documented dropped frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing camera positions are reconstructed from rounded interframe-interval/median-interval ratios and linearly interpolated onto neural frame indices. The aligned trace is averaged over the same non-overlapping 10-frame bins as neural data, then converted to session-specific quintile labels.

ii.
```python
median_interval = float(np.median(positive))
interval_steps = np.maximum(1, np.rint(intervals / median_interval).astype(int))
observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
aligned = np.interp(np.arange(expected_frames), observed_positions, raw)
...
motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
```

iii. The release permits interpolation of documented gaps. Repair before paired averaging prevents shifted neural/behavior alignment and avoids missing values in the decoder.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After alignment and 10-frame averaging, the 20th, 40th, 60th, and 80th percentiles are computed independently for each complete session. `searchsorted(..., side="right")` maps values to integer classes 0–4, and exact equal class counts are asserted.

ii.
```python
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(
    quantile_edges, motion_binned, side="right"
).astype(np.int64)
counts = np.bincount(labels, minlength=5)
```

iii. This directly implements the task's five equal-percentile bins “selected per session.” Thresholding after averaging categorizes the actual motion values paired with decoder timepoints.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera gaps are repaired to make motion length equal neural-frame length; both signals are then averaged over identical groups of 10 frames and sliced with the same 180-bin trial slices. Full-length motion streams are left unchanged even if timestamps contain clock anomalies.

ii.
```python
if missing:
    observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
    ...
    aligned = np.interp(
        np.arange(expected_frames, dtype=np.float64), observed_positions, raw
    )
else:
    aligned = raw.copy()
...
output_trial = np.ascontiguousarray(labels[sl][None, :], dtype=np.int64)
```

iii. The agent cites synchronous 30 Hz acquisition and the release's length-mismatch definition of dropped frames. It independently checked intact and repaired sessions against the saved output.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion frames are linearly interpolated only when motion is shorter than neural data and timestamp gaps exactly explain the deficit. Malformed arrays, unexplained gaps, extra motion frames, non-finite values, unexpected sampling metadata, or incomplete sessions cause explicit errors. One finite Suite2p bad frame is retained; all valid released data are retained.

ii.
```python
if observed_positions[-1] != expected_frames - 1:
    raise ValueError("Timestamp gaps do not explain frame deficit ...")
aligned = np.interp(...)
...
if len(aligned) != expected_frames or not np.all(np.isfinite(aligned)):
    raise ValueError(...)
```

iii. The agent preferred auditable repair for the 276 documented missing camera frames and fail-fast behavior for unexplained corruption. It documented paper/export discrepancies rather than altering valid source records.

## 6-a. What are the most time-consuming steps of the code?

i. The Gaussian/minimum/maximum filtering for maximin baseline correction across every neuron and full session is the main computational step. Source loading is the principal I/O work; serialization is comparatively small. Optional diagnostic plotting adds overhead when requested.

ii.
```python
t0 = time.perf_counter()
neural_binned, trace_example = baseline_correct_and_bin(...)
neural_seconds = time.perf_counter() - t0
```

iii. The notes identify full-session fluorescence filtering as the core cost and report roughly 0.37 seconds for a 221-neuron session, with the complete conversion taking 48 seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The session and trial loops are structural and remain. Neural operations are vectorized within 64-neuron chunks, motion-gap reconstruction uses a single `np.interp`, and downsampling uses reshape/mean. The trial slicing loop could theoretically be replaced by reshaping into a trial axis, and session processing could be parallelized, but the agent deliberately avoided concurrent I/O and memory multiplication.

ii.
```python
for start in range(0, n_neurons, NEURON_CHUNK):
    ...
    result[start:stop] = activity.reshape(
        stop - start, n_binned, DOWNSAMPLE
    ).mean(axis=2)
...
for trial in range(n_trials):
    sl = slice(...)
```

iii. The agent explicitly optimized the expensive numerical work and noted that parallel disk reads would be counterproductive for the 16-GB, memory-sensitive workload.

## 6-c. What processing does the code repeat multiple times?

i. The same validation, fluorescence processing, motion alignment, binning, and discretization pipeline is invoked once per session. It does not repeat filtering per trial; trials only slice already-processed continuous arrays. Some source arrays are opened once for validation and `F`/`Fneu` are opened again for processing.

ii.
```python
for index, ref in enumerate(session_refs):
    converted, info = convert_session(ref, ...)
...
neural_binned, trace_example = baseline_correct_and_bin(...)
...
for trial in range(n_trials):
    sl = slice(...)
```

iii. The notes stress that processing a full session once avoids 20–30 repeated filter calls and scientifically incorrect boundary artifacts. Repeated per-session work is required because thresholds and recordings are session-specific.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `spks`, `stat`, much of `ops`, timestamps, and interpolation masks are loaded or constructed for validation/metadata but are not decoder fields. The function also copies a first-neuron trace through every processing stage even when no processing plot is requested. With `--show-processing`, z-scored rasters and other diagnostic plot transforms are created and then discarded after saving plots.

ii.
```python
spks = np.load(plane / "spks.npy", mmap_mode="r")
stat = np.load(plane / "stat.npy", allow_pickle=True)
ops = np.load(plane / "ops.npy", allow_pickle=True).item()
...
if start == 0:
    example = {"raw_f": raw_f[0].copy(), ...}
```

iii. These operations were justified as source-integrity checks, audit metadata, and optional visual diagnostics rather than decoder features. Their cost is small relative to baseline filtering, but the trace-example copies are avoidable when plotting is disabled.
