# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by scanning every subject directory under `/app/data`, then every subdirectory under each subject, and keeping only those session folders that contain the required Suite2p and behavior files. It then loads Suite2p parameters from `ops.npy`, neural fluorescence from `F.npy` and `Fneu.npy`, and behavior from `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`.

ii. 
```python
def discover_sessions() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            plane = session_dir / "suite2p" / "plane0"
            behavior = session_dir / "move_deve"
            required = (
                plane / "F.npy",
                plane / "Fneu.npy",
                plane / "iscell.npy",
                plane / "ops.npy",
                behavior / "motion_energy_glob.npy",
                behavior / "tstamps.npy",
                behavior / "interframe_int.npy",
            )
            if all(path.exists() for path in required):
                found.append((subject_dir.name, session_dir))
```

```python
ops = np.load(ops_path, allow_pickle=True).item()
fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
```

iii. The justification in `CONVERSION_NOTES.md` is that only dated session folders with complete Suite2p outputs and motion files should be converted, because `ops.npy` is needed to reproduce Suite2p preprocessing parameters and `tstamps.npy`/`interframe_int.npy` are needed to repair missing camera frames faithfully.

## 1-b. How are the data split into subjects?

i. Subjects are the unique top-level directory names encountered during session discovery, sorted alphabetically and converted into a lookup table for `subject_idx`.

ii. 
```python
subjects = sorted({subject for subject, _ in selected})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes state that `/app/data` contains six subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), and the converter treats those folder names as the subject IDs without further remapping.

## 1-c. How are the data split into sessions?

i. Each qualifying dated subdirectory under a subject is treated as one session. Sessions are processed in sorted subject/date order, and one converted session is produced per dated recording folder.

ii. 
```python
for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        ...
        if all(path.exists() for path in required):
            found.append((subject_dir.name, session_dir))
```

```python
for index, (subject, session_dir) in enumerate(selected):
    session_id = f"{subject}/{session_dir.name}"
    neural, decoder_input, output, info, diagnostics = convert_session(
        subject, session_dir, make_plot
    )
```

iii. In the notes, the AI says the data hierarchy is already session-based: each dated folder contains one recording session with one `suite2p/plane0` folder and one `move_deve` folder, so that folder boundary is the natural session split.

## 1-d. How are the data split into trials?

i. The AI assumes there are no native trials. It treats each session as one continuous recording, requires that it divide exactly into 60-second chunks after 10-frame binning, and splits all three streams into contiguous, non-overlapping 60-second trials.

ii. 
```python
frames_per_trial_float = TRIAL_SECONDS * ops["fs"]
frames_per_trial = int(frames_per_trial_float)
bins_per_trial = frames_per_trial // RAW_FRAMES_PER_BIN
...
if n_frames % frames_per_trial:
    raise ValueError(f"Session does not divide into complete 60-s trials: {session_dir}")
```

```python
def split_trials(array: np.ndarray, bins_per_trial: int) -> list[np.ndarray]:
    if array.ndim != 2 or array.shape[1] % bins_per_trial:
        raise ValueError(f"Cannot split array of shape {array.shape}")
    return [
        array[:, start : start + bins_per_trial].copy()
        for start in range(0, array.shape[1], bins_per_trial)
    ]
```

iii. The notes explicitly say the recordings are continuous spontaneous behavior with no trial events, and that the decoder task overrides the paper by requiring non-overlapping 60-second blocks. The AI also notes that all supplied recordings divide evenly, so no truncation is needed on this dataset.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-quality filtering. It only enforces structural constraints: sessions must yield at least two trials, all three streams must have matching trial counts, and each trial must have the expected shapes.

ii. 
```python
if not (
    len(neural_trials) == len(input_trials) == len(output_trials) >= 2
):
    raise AssertionError(f"Trial count mismatch in {session_dir}")
for neural, decoder_input, output in zip(
    neural_trials, input_trials, output_trials
):
    if not (
        neural.shape == (n_neurons, bins_per_trial)
        and decoder_input.shape == (1, bins_per_trial)
        and output.shape == (1, bins_per_trial)
    ):
        raise AssertionError(f"Trial shape mismatch in {session_dir}")
```

iii. The notes say there are no native behavioral trials or rejected-trial rules in the reference paper, so the converter keeps every complete artificial 60-second block.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` signal is derived directly from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`, with preprocessing parameters read from `ops.npy`.

ii. 
```python
fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
ops = np.load(ops_path, allow_pickle=True).item()
```

iii. The notes explain that the paper used baseline-corrected fluorescence rather than deconvolved spikes, and that the authoritative preprocessing parameters are in `ops.npy`.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil using the Suite2p `neucoeff` from `ops.npy`, then applies a manual reimplementation of Suite2p/Track2p-style `maximin` baseline correction, subtracts that baseline, and averages the result into non-overlapping 10-frame bins.

ii. 
```python
corrected = raw_f - neucoeff * raw_fneu

if baseline_mode == "maximin":
    flow = gaussian_filter(corrected, sigma=(0.0, sigma))
    flow = minimum_filter1d(flow, size=baseline_window, axis=1)
    flow = maximum_filter1d(flow, size=baseline_window, axis=1)
...
processed = corrected - flow
```

```python
binned[start:stop] = processed.reshape(
    stop - start, n_bins, RAW_FRAMES_PER_BIN
).mean(axis=2)
```

iii. The notes justify this by saying the distributed data are publication-linked Track2p exports, the paper describes default Suite2p baseline correction, and the Track2p GUI code subtracts the maximin baseline rather than dividing by it. The AI therefore chose `Fc - Flow` with parameters pulled from `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any extra neuron filtering at conversion time. It requires that `iscell.npy` exist, but it assumes the provided arrays are already Track2p-curated complete tracks with `iscell > 0.5`, so every row in `F.npy` is retained.

ii. 
```python
required = (
    plane / "F.npy",
    plane / "Fneu.npy",
    plane / "iscell.npy",
    plane / "ops.npy",
    ...
)
```

```python
"neuron_curation": (
    "distributed Track2p complete tracks; source iscell probability >0.5"
),
```

iii. In `CONVERSION_NOTES.md`, the AI says it inspected the shipped `iscell.npy` files and found that all rows already have `iscell[:,0] == 1` and probabilities above 0.5, so reapplying the filter would change nothing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code aligns each trial to the start of its own artificial 60-second block. In metadata it describes the alignment event as the start of each non-overlapping 60-second block, with `off_start = 0.0` and `off_end = 60.0`. However, the accompanying input time remains absolute time from session start rather than resetting within each trial.

ii. 
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each non-overlapping 60-second block",
    "off_start": 0.0,
    "off_end": 60.0,
    "input_time_reference": "absolute elapsed time from session start at bin centers",
    ...
}
```

```python
neural_trials = split_trials(neural_binned, bins_per_trial)
input_trials = split_trials(input_binned, bins_per_trial)
output_trials = split_trials(output_binned, bins_per_trial)
```

iii. The notes say there is no native event to align to, so the AI used the artificial block boundary as the trial alignment event while keeping the requested decoder input as absolute session time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping temporal bins. At 30 Hz this is a 333.33 ms bin size. Both neural and motion streams are rebinned once using a simple mean.

ii. 
```python
RAW_FRAMES_PER_BIN = 10
...
binned[start:stop] = processed.reshape(
    stop - start, n_bins, RAW_FRAMES_PER_BIN
).mean(axis=2)
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
```

```python
"time_bin_size": 1000.0 * RAW_FRAMES_PER_BIN / 30.0,
```

iii. The notes cite the paper’s statement that decoding analyses average both fluorescence and behavior over 10 consecutive timestamps, so the AI preserved that denoising step before trialization and discretization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not loaded from a dedicated raw timestamp array. It is derived from the imaging frame index after binning and the Suite2p sampling rate `ops["fs"]`.

ii. 
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
```

iii. The notes say the authoritative session timebase is the imaging rate in `ops['fs']`; the camera timestamps are used only to localize missing motion frames, not to define elapsed experiment time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one scalar time per 10-frame bin, using the center of each averaged bin rather than its left edge, and stores the result as a `(1, n_timepoints)` float32 array.

ii. 
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
```

iii. The notes justify bin centers as the cleanest way to timestamp signals that are themselves 10-frame averages. They also note explicit sanity checks on the first and last centers.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it is generated after the neural stream has been converted to `n_bins`, then split into trials using the exact same `split_trials` function and `bins_per_trial` as the neural and output streams.

ii. 
```python
n_neurons, n_bins = neural_binned.shape
...
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
...
neural_trials = split_trials(neural_binned, bins_per_trial)
input_trials = split_trials(input_binned, bins_per_trial)
output_trials = split_trials(output_binned, bins_per_trial)
```

iii. The notes explicitly say the input should remain “absolute elapsed time from session start” but must have one value per binned neural sample. The AI therefore built the time vector on the same binned grid as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion signal is derived from `motion_energy_glob.npy`, with `tstamps.npy` and `interframe_int.npy` used to infer and repair dropped camera frames before alignment to neural data.

ii. 
```python
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
```

iii. The notes explain that the supplied motion energy is already the paper’s precomputed whole-frame motion signal, while the camera timestamp files are needed only to restore missing video frames so the motion and neural streams have the same frame count.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI validates the timestamp arrays, reconstructs missing frame positions when the motion stream is shorter than the neural stream, linearly interpolates only the missing samples onto the full neural frame grid, averages the repaired trace into 10-frame bins, and then converts it to classes.

ii. 
```python
if not np.allclose(intervals, np.diff(timestamps), rtol=1e-10, atol=1e-12):
    raise ValueError(f"interframe_int != diff(tstamps) in {session_dir}")
...
median_interval = float(np.median(intervals))
frame_steps = np.maximum(np.rint(intervals / median_interval).astype(np.int64), 1)
observed_idx = np.concatenate(
    (np.array([0], dtype=np.int64), np.cumsum(frame_steps, dtype=np.int64))
)
...
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
```

iii. In the notes, the AI argues that timestamp-derived reconstruction is more faithful than a fixed interval threshold because it uses the actual observed gap structure and requires the inferred missing count to match the observed length deficit exactly.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After 10-frame averaging, the AI computes within-session quintile thresholds at the 20th, 40th, 60th, and 80th percentiles, then assigns classes 0-4 with `np.searchsorted(..., side="right")`. It also checks that thresholds are distinct and class fractions are approximately 20% each.

ii. 
```python
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
if np.unique(quantile_edges).size != 4:
    raise ValueError(f"Non-distinct motion quintile thresholds in {session_dir}")
motion_classes = np.searchsorted(
    quantile_edges, motion_binned, side="right"
).astype(np.int64)
fractions = np.bincount(motion_classes, minlength=5) / motion_classes.size
```

iii. The notes say the decoder task overrides the paper’s continuous target and explicitly requires five equal-percentile bins chosen separately for each session, so the AI enforced session-local quintiles and audited the resulting class balance.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is enforced at the frame level first: the motion trace is repaired to exactly `n_frames`, then it is averaged into the same number of 10-frame bins as the neural data, and finally it is split into trials using the same trial boundaries.

ii. 
```python
repaired_motion, missing_mask, raw_motion, observed_idx = repair_motion(
    session_dir, n_frames
)
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
...
output_binned = motion_classes[np.newaxis, :]
...
output_trials = split_trials(output_binned, bins_per_trial)
```

iii. The notes repeatedly emphasize that the microscope triggered the camera at 30 Hz, so once missing camera frames are restored, framewise indexing is the correct neural/behavior alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles minor issues conservatively. It validates array shapes, checks that `interframe_int.npy` matches `np.diff(tstamps.npy)`, repairs behavior only when it is shorter than the neural data, requires the inferred number of missing frames to equal the length deficit, verifies that interpolation leaves observed samples unchanged, and raises errors rather than silently truncating malformed sessions.

ii. 
```python
if intervals.shape != (raw_motion.size - 1,):
    raise ValueError(f"Invalid interframe interval shape in {session_dir}")
if not np.allclose(intervals, np.diff(timestamps), rtol=1e-10, atol=1e-12):
    raise ValueError(f"interframe_int != diff(tstamps) in {session_dir}")
if raw_motion.size > n_frames:
    raise ValueError(f"More behavior frames than neural frames in {session_dir}")
```

```python
deficit = n_frames - raw_motion.size
inferred = int(np.sum(frame_steps - 1))
if inferred != deficit or observed_idx[-1] != n_frames - 1:
    raise ValueError(
        f"Timestamp gaps infer {inferred} missing frames, expected {deficit}, "
        f"in {session_dir}"
    )
...
if not np.allclose(repaired[observed_idx], raw_float):
    raise AssertionError(f"Motion repair altered observed values in {session_dir}")
```

iii. The notes say the dataset README explicitly allows missing motion values or interpolation, but the AI chose to repair only clearly documented dropped frames and otherwise fail loudly so misalignment is not silently introduced.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is neural preprocessing in `baseline_correct_and_bin`, especially the Gaussian/min/max baseline operations over full fluorescence traces. The code times that section per session and stores the elapsed seconds in metadata.

ii. 
```python
t_neural = time.perf_counter()
neural_binned, neural_diag = baseline_correct_and_bin(
    session_dir, ops, collect_diagnostics
)
neural_seconds = time.perf_counter() - t_neural
```

```python
"neural_processing_seconds": neural_seconds,
```

iii. The notes identify full-session fluorescence processing as the main cost and explain that chunking and memory mapping were added specifically because this stage dominates runtime and memory use.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main places where the human reference had obvious scalar loops. Missing-frame repair is done with one `np.interp` call instead of repeated `np.insert`, and 10-frame averaging is done with `reshape(...).mean(...)`. Remaining explicit loops are mostly over sessions, neuron chunks, and trial extraction.

ii. 
```python
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
```

```python
binned[start:stop] = processed.reshape(
    stop - start, n_bins, RAW_FRAMES_PER_BIN
).mean(axis=2)
```

iii. The notes describe these as deliberate speedups: the AI says it replaced repeated frame insertion with vectorized interpolation and processed neurons in chunks to bound peak memory while keeping the inner operations vectorized.

## 6-c. What processing does the code repeat multiple times?

i. There is not much scientifically redundant processing in the core conversion path. The main repeated work is structural rather than computational: `split_trials` is applied separately to neural, input, and output arrays, and `validate_converted` makes an additional pass over the finished data for integrity checks.

ii. 
```python
neural_trials = split_trials(neural_binned, bins_per_trial)
input_trials = split_trials(input_binned, bins_per_trial)
output_trials = split_trials(output_binned, bins_per_trial)
```

```python
validate_converted(data)
```

iii. The notes do not identify any major repeated scientific transform. Their focus is instead on single-pass session processing followed by optional validation and plotting.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion includes optional diagnostic work that is not consumed by the downstream decoder: diagnostic copies for plotting, optional `processing_*.png` figures, timing metadata, and integrity checks such as class-balance validation. The core neural/input/output transforms are all retained.

ii. 
```python
if collect_diagnostics and start == 0:
    diagnostics = {
        "raw_f": raw_f[:nplot_neurons, :nplot_frames].copy(),
        "raw_fneu": raw_fneu[:nplot_neurons, :nplot_frames].copy(),
        ...
        "motion_classes": motion_classes,
        "quantile_edges": quantile_edges,
        "elapsed_time": input_binned[0],
        "fs": ops["fs"],
    }
```

```python
if make_plot:
    plot_path = plot_processing(session_id, diagnostics)
    print(f"  saved {plot_path}")
...
validate_converted(data)
```

iii. The notes explicitly mention optional plotting, cache artifacts, and extensive sanity checks as audit machinery rather than decoder inputs. Those steps add overhead, but they are deliberate verification rather than discarded scientific preprocessing.
