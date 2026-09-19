# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` under the data root, then discovers sessions as subdirectories within each subject folder that have a date-like name (4-digit prefix, length >= 10). For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. All sessions are processed sequentially and assembled into a single dataset dictionary.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionRef]:
    sessions: list[SessionRef] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
        ):
            sessions.append(
                SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
            )
    return sessions

# Within process_session:
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The AI chose to load `ops.npy` to read suite2p parameters (fs, neucoeff, baseline settings) dynamically rather than hardcoding them. It also uses `tstamps.npy` instead of `interframe_int.npy` for motion energy alignment. The session discovery filter (date-like names) is more restrictive than simply using all subdirectories, but functionally equivalent for this dataset.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data root, sorted alphabetically. A sorted set of unique subject names is built from the discovered sessions.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...

# In build_dataset:
subjects = sorted({session.subject for session in session_refs})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a date-formatted subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording's suite2p output and behavioral data.

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

iii. The AI filters subdirectories by requiring a 4-digit year prefix and minimum length of 10 characters, matching the `YYYY-MM-DD_a` naming convention. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. The AI defines trials as consecutive 120-second (2-minute) non-overlapping segments of the continuous recording. After 10-frame binning at 30 Hz, each trial has 360 time bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_SECONDS = 120.0
...
bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))  # 120 * 30 / 10 = 360
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
...
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
```

iii. The AI chose 2-minute trials because the reference paper uses "consecutive 2-minute blocks" for its decoding analysis. However, the task instructions explicitly state "Split sessions into 60-second trials," which the AI did not follow.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks are included. Sessions too short to form even one trial raise an error.

ii.
```python
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")
```

iii. There is no natural trial structure with quality concerns in this continuous recording dataset. The only filtering is discarding remainder bins at the end.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (suite2p parameters including neucoeff, baseline settings, and fs), all from `plane0`.

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces. The AI additionally reads `ops.npy` to extract processing parameters.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - neucoeff * Fneu` where neucoeff comes from `ops.npy`, which is 0.7), followed by suite2p's `dcnv.preprocess` which performs baseline estimation and correction using the `maximin` method. Parameters are read from `ops.npy`. Then, non-overlapping averaging in 10-frame bins is applied.

ii.
```python
def compute_fluorescence_signal(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
    processed = suite2p_preprocess(
        Fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        batch_size=min(512, max(32, Fc.shape[0])),
        device=torch.device("cpu"),
    )
    return processed.astype(np.float32, copy=False)
```

iii. The AI reads all preprocessing parameters from `ops.npy` rather than hardcoding them, which is more robust to potential per-session variation. The pipeline matches the paper's description of neuropil subtraction plus baseline correction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond what was already done in the Track2p pipeline. All neurons in the exported `F.npy` are included.

ii. N/A (no filtering code)

iii. The AI documented in CONVERSION_NOTES.md that the provided data is already post-Track2p export with `iscell > 0.5` filtering and all-day tracking already applied, so no additional filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 2-minute segments of the continuous recording, no event-based alignment is needed.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute recording block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous behavior, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting 30 Hz to 3 Hz (333.33 ms time bin). This matches the paper's description.

ii.
```python
BIN_FRAMES = 10
...
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
...
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
...
"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
```

iii. The Methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index, the bin size (10 frames), and the frame rate (read from `ops['fs']`), giving seconds from session start.

ii.
```python
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

iii. Since the frame rate is constant at 30 Hz and bins are 10 frames, computing time from bin indices gives seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `bin_index * (BIN_FRAMES / fs)`, giving the left edge of each bin in seconds from session start. It runs continuously across the full session length and is then split into trial-length segments.

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
...
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. The time value preserves the absolute position within the session, so later trials have higher time values.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned because it is computed from the same bin indices used for neural data. Both share the same length and temporal grid.

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
```

iii. No explicit alignment is needed since the time vector is derived from the neural data's own time axis.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used to align the motion energy to the imaging frame grid and detect/interpolate dropped frames.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video. The timestamps file provides the time of each video frame, enabling alignment to the imaging grid.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Five processing steps: (1) motion energy samples are mapped onto the imaging frame grid using timestamps and missing frames are linearly interpolated, (2) the trace is averaged into 10-frame bins, (3) the binned signal is min-max normalized within session, (4) the normalized signal is discretized into 5 quintile-based bins (per session), (5) the result is one-hot encoded into 5 binary channels.

ii.
```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)

def normalize_motion(x):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)

def quintile_one_hot(x):
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. The AI added a min-max normalization step before quintiling. Since normalization is a monotonic transform, it does not change quintile boundaries. The one-hot encoding was chosen because the AI interpreted the decoder as requiring binary outputs for multi-class variables.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses `np.quantile` at [0.2, 0.4, 0.6, 0.8] to compute 4 edges, then `np.searchsorted(edges, x, side="right")` to assign each value to one of 5 classes (0-4). This is done per session on the min-max normalized signal. The result is then one-hot encoded into 5 binary channels.

ii.
```python
def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. The quintile approach produces 5 equal-percentile bins as required by the instructions. The AI uses `np.quantile` (equivalent to `np.percentile` with values /100) and `np.searchsorted` (functionally equivalent to `np.digitize` for sorted edges). The one-hot representation differs from the reference's integer class representation.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses `tstamps.npy` to compute the expected frame index for each motion energy sample, maps them onto the imaging frame grid, averages duplicates, and linearly interpolates any missing frames. This produces a motion trace of exactly `n_imaging_frames` length, aligned frame-for-frame with the neural data.

ii.
```python
def reconstruct_motion_trace(motion_energy, tstamps, n_imaging_frames):
    frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)

    full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
    counts = np.zeros(n_imaging_frames, dtype=np.int64)
    sums = np.zeros(n_imaging_frames, dtype=np.float64)
    np.add.at(sums, frame_idx, motion_energy.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
    valid = counts > 0
    full[valid] = (sums[valid] / counts[valid]).astype(np.float32)

    valid_idx = np.flatnonzero(valid)
    missing = np.flatnonzero(~valid)
    if len(missing):
        full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
    return full, info
```

iii. This approach uses timestamps to reconstruct the motion trace on the imaging grid, which is different from the reference's approach of using `interframe_int.npy` to detect dropped frames. Both approaches achieve the same goal of aligning motion energy to neural data frame-for-frame. The AI's approach is more general (works for any timestamp pattern, not just dropped frames) but uses a different data file (`tstamps.npy` vs `interframe_int.npy`).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected via the timestamp-based reconstruction and linearly interpolated. The AI validates that no NaNs remain after interpolation. Remainder time bins at the end of a session that don't fill a complete 2-minute trial are discarded.

ii.
```python
# In validate_converted_session:
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")

# In split_trials:
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
```

iii. The validation ensures any remaining NaN values would be caught. Discarding remainder frames is a minor data loss.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `dcnv.preprocess` baseline correction. The AI runs this on CPU. Loading `.npy` files is also I/O-bound but relatively fast.

ii. N/A

iii. From CONVERSION_NOTES.md: mean processing time per session was 2.26s, giving an estimated full-dataset time of ~1.54 minutes. The baseline correction involves sliding window operations over the full session length for every neuron.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is already well-vectorized. The motion frame reconstruction uses `np.add.at` and `np.interp` rather than Python loops. The binning uses reshape/mean vectorization. The trial splitting loop is a simple range-based slicing loop that cannot be easily vectorized further.

ii. N/A

iii. The AI explicitly documented vectorization as a speedup in CONVERSION_NOTES.md.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session is processed once in a single pass.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI adds a min-max normalization step to the motion energy before quintiling. Since quintile computation is invariant to monotonic transforms, this normalization is unnecessary. The AI also computes and stores one-hot encoding which adds 5 output dimensions instead of 1.

ii.
```python
def normalize_motion(x: np.ndarray) -> np.ndarray:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)
```

iii. The normalization was likely added for visualization/interpretability purposes but does not affect the final quintile assignments.
