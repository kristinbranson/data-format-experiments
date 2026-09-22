# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers sorted subject/session directories under `/app/data`, retains directories containing all required Suite2p and behavior files, and processes every discovered session in full mode. It memory-maps `F.npy` and `Fneu.npy` and loads `ops.npy`, motion energy, timestamps, and interframe intervals.

ii.
```python
for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        ...
        if all(path.exists() for path in required):
            found.append((subject_dir.name, session_dir))

fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
```

iii. The notes say this found all 6 mice and 41 dated recordings exactly once. Required-file checks avoid treating unrelated directories as sessions, while sorted traversal gives deterministic order.

## 1-b. How are the data split into subjects?

i. The parent directory name is the subject ID. Unique selected IDs are sorted, and each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({subject for subject, _ in selected})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[subject])
```

iii. The agent found the six `jm*` mouse folders and treated each as one mouse, consistent with the data hierarchy.

## 1-c. How are the data split into sessions?

i. Each dated subdirectory beneath a mouse is one session, ordered by sorted subject and directory name.

ii.
```python
for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        ...
        found.append((subject_dir.name, session_dir))
```

iii. The notes identify each dated directory as a continuous daily recording and report 41 sessions (6–7 per mouse).

## 1-d. How are the data split into trials?

i. Each continuous session is divided into contiguous, non-overlapping 60-second trials after 10-frame averaging. At 30 Hz this gives 1,800 raw frames or 180 converted bins per trial. The code requires exact divisibility rather than dropping a tail.

ii.
```python
frames_per_trial = int(TRIAL_SECONDS * ops["fs"])
bins_per_trial = frames_per_trial // RAW_FRAMES_PER_BIN
if n_frames % frames_per_trial:
    raise ValueError(f"Session does not divide into complete 60-s trials: {session_dir}")
...
return [array[:, start : start + bins_per_trial].copy()
        for start in range(0, array.shape[1], bins_per_trial)]
```

iii. There are no natural trials; the task requires 60-second trials. The notes establish that every supplied recording divides evenly, so no data are lost and 1,090 trials result.

## 1-e. How are trials filtered based on quality controls?

i. No trial is rejected. Shape, finiteness, synchronized stream length, at least two trials per session, complete class support, and balanced session-level quintiles are validated; violations stop conversion.

ii.
```python
if not (len(neural_trials) == len(input_trials) == len(output_trials) >= 2):
    raise AssertionError(f"Trial count mismatch in {session_dir}")
...
if not (np.isfinite(neural).all() and np.isfinite(decoder_input).all()
        and np.isfinite(output).all()):
    raise AssertionError("Converted arrays contain NaN/Inf")
```

iii. The notes state that the experiment is continuous spontaneous behavior with no native rejected-trial rule; all source sessions are finite and valid.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from Suite2p `F.npy` and `Fneu.npy`; processing parameters come from `ops.npy`.

ii.
```python
fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
ops = np.load(ops_path, allow_pickle=True).item()
```

iii. The agent identified these as the reference code's fluorescence and neuropil sources and used the session's stored acquisition/preprocessing settings.

## 2-b. How is the `neural` data processed?

i. It computes neuropil-corrected fluorescence `F - neucoeff*Fneu`, applies the Suite2p maximin baseline (Gaussian smoothing followed by minimum and maximum filters), subtracts that baseline, then averages non-overlapping groups of 10 frames. Processing is chunked across neurons.

ii.
```python
corrected = raw_f - neucoeff * raw_fneu
flow = gaussian_filter(corrected, sigma=(0.0, sigma))
flow = minimum_filter1d(flow, size=baseline_window, axis=1)
flow = maximum_filter1d(flow, size=baseline_window, axis=1)
processed = corrected - flow
binned[start:stop] = processed.reshape(
    stop - start, n_bins, RAW_FRAMES_PER_BIN
).mean(axis=2)
```

iii. The notes explain that this reproduces `DataManagement.F_processing` and the paper's 10-timestamp denoising. It deliberately does not divide by the baseline because the executable reference code does not do so.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra row filtering is performed. All supplied neural rows are retained; shapes, frame counts, and finite processed values are checked.

ii.
```python
if fluorescence.shape != neuropil.shape:
    raise ValueError(f"F/Fneu shape mismatch in {session_dir}")
...
if not np.isfinite(binned).all():
    raise ValueError(f"Non-finite processed neural values in {session_dir}")
```

iii. The agent inspected `iscell.npy` and concluded the distributed arrays were already filtered at probability `>0.5` and Track2p-curated to complete longitudinal matches; applying that filter again would change nothing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological trial event. Neural data are split on consecutive 60-second block boundaries, and metadata defines alignment to each block start with offsets 0–60 seconds.

ii.
```python
neural_trials = split_trials(neural_binned, bins_per_trial)
...
"temporal_alignment_event": "start of each non-overlapping 60-second block",
"off_start": 0.0,
"off_end": 60.0,
```

iii. The notes explain that recordings are continuous spontaneous behavior, so the artificial trial boundary required by the task is the only applicable event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converter averages 10 consecutive 30-Hz frames, producing 333.333 ms bins (3 Hz) for both neural and motion streams.

ii.
```python
RAW_FRAMES_PER_BIN = 10
...
"time_bin_size": 1000.0 * RAW_FRAMES_PER_BIN / 30.0,
```

iii. This follows the paper's decoding method of averaging both fluorescence and behavior over 10 consecutive timestamps and preserves their common time base.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is constructed from the converted-bin index, the fixed 10 raw frames per bin, and the session sampling rate from `ops['fs']`; no stored behavioral timestamp values are used for elapsed seconds.

ii.
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
```

iii. The agent judged imaging `ops['fs']` authoritative because the camera was microscope-triggered and behavioral timestamp units were uncertain.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It computes the mean-frame time (center) of every 10-frame bin, casts to float32, and keeps time continuous across artificial trials. Thus the first value is 0.15 s rather than 0 s.

ii.
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
```

iii. The notes justify bin centers as the timestamps of averages over raw-frame times and explicitly avoid resetting time at each 60-second block.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One center time is generated for each neural 10-frame average, and the same `split_trials` boundaries are applied to both arrays.

ii.
```python
neural_trials = split_trials(neural_binned, bins_per_trial)
input_trials = split_trials(input_binned, bins_per_trial)
```

iii. The notes report an independent all-session concatenation check confirming every input bin corresponds to the same neural bin with no boundary duplication or loss.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is based on precomputed `move_deve/motion_energy_glob.npy`. `tstamps.npy` and `interframe_int.npy` are used to validate and repair missing camera samples.

ii.
```python
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
```

iii. The supplied global motion energy already represents summed interframe pixel-change energy, so the agent correctly avoided recomputing video motion.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. When motion is shorter than neural data, timestamp gaps are converted to integer frame steps and all missing positions are linearly interpolated. The repaired trace is averaged over the same 10-frame bins as neural data, then converted to categorical quintiles.

ii.
```python
frame_steps = np.maximum(np.rint(intervals / median_interval).astype(np.int64), 1)
observed_idx = np.concatenate(
    (np.array([0], dtype=np.int64), np.cumsum(frame_steps, dtype=np.int64))
)
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
...
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
```

iii. The notes say sparse interpolation preserves every observed value, restores frame-for-frame synchronization, and matches all 276 source deficits; averaging follows the paper.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds (20th, 40th, 60th, and 80th percentiles) are computed separately on each session's binned motion trace. Right-sided search assigns integer classes 0–4.

ii.
```python
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
motion_classes = np.searchsorted(
    quantile_edges, motion_binned, side="right"
).astype(np.int64)
```

iii. This directly implements the requested five equal-percentile bins selected per session. The agent checks distinct edges and approximately/exactly 20% class fractions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera gaps are reconstructed at timestamp-inferred neural-frame positions; observed samples remain unchanged. Motion and neural data are then averaged in identical 10-frame groups and split at identical trial boundaries.

ii.
```python
if inferred != deficit or observed_idx[-1] != n_frames - 1:
    raise ValueError(...)
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
...
output_trials = split_trials(output_binned, bins_per_trial)
```

iii. The agent relied on microscope-triggered 30-Hz acquisition and used timestamps only where a length deficit proved frames were absent. Independent checks showed no observed value changed and all streams reconstructed exactly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are localized from timestamp-gap multiples and linearly interpolated. Shape mismatches, unexpected counts, non-finite values, non-distinct quintiles, or incomplete trials raise errors instead of being silently accepted. A single finite Suite2p `badframes` entry is retained and averaged.

ii.
```python
deficit = n_frames - raw_motion.size
inferred = int(np.sum(frame_steps - 1))
if inferred != deficit:
    raise ValueError(...)
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
```

iii. The notes justify interpolation because deficits are extremely sparse and precisely identified, and explain that no reference rule calls for deleting the one finite bad imaging frame.

## 6-a. What are the most time-consuming steps of the code?

i. Neural baseline correction—Gaussian smoothing and long-window min/max filtering over every neuron and frame—is the main compute cost. Loading large fluorescence arrays is the main I/O cost.

ii.
```python
for start in range(0, n_neurons, NEURAL_CHUNK_SIZE):
    ...
    flow = gaussian_filter(corrected, sigma=(0.0, sigma))
    flow = minimum_filter1d(flow, size=baseline_window, axis=1)
    flow = maximum_filter1d(flow, size=baseline_window, axis=1)
```

iii. The notes separately time neural processing and report the full conversion completed in about 44 seconds. Chunking and memory mapping bound memory and reduce I/O duplication.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The substantive numerical operations are already vectorized: neural work is performed in 64-neuron blocks, missing-frame interpolation is one `np.interp`, binning uses reshape/mean, and class assignment uses `searchsorted`. Remaining loops traverse sessions/chunks or create the required list of trial arrays; the trial-copy loop could be expressed as a reshape before list conversion but would not eliminate the required per-trial objects.

ii.
```python
for start in range(0, n_neurons, NEURAL_CHUNK_SIZE):
    ...
return [array[:, start : start + bins_per_trial].copy()
        for start in range(0, array.shape[1], bins_per_trial)]
```

iii. The agent explicitly describes chunk-vectorized filters and vectorized gap reconstruction as efficiency choices. Unlike the reference implementation, it avoids repeated `np.insert` reallocations for dropped frames.

## 6-c. What processing does the code repeat multiple times?

i. The same necessary per-session pipeline is repeated for each recording, and `split_trials` is called separately for neural, input, and output streams. No raw source stream is redundantly loaded or processed multiple times within a session.

ii.
```python
for index, (subject, session_dir) in enumerate(selected):
    neural, decoder_input, output, info, diagnostics = convert_session(...)
...
neural_trials = split_trials(neural_binned, bins_per_trial)
input_trials = split_trials(input_binned, bins_per_trial)
output_trials = split_trials(output_binned, bins_per_trial)
```

iii. The notes state that source files are read once per session; repetition is required by independent sessions and by the target's three parallel nested lists.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Core full conversion does not perform material discarded processing. In optional `--show-processing` mode, small diagnostic raw/intermediate slices are copied and plotted, then not stored in the dataset. Metadata also records timing and audit statistics that the decoder does not consume, but these support provenance and validation.

ii.
```python
if collect_diagnostics and start == 0:
    diagnostics = {"raw_f": ..., "baseline": ..., "neural_binned": ...}
...
make_plot = args.show_processing and index < 2
```

iii. The agent made diagnostics opt-in and limited them to two sessions. It intentionally excludes large spatial/ROI objects because they are not decoder variables.
