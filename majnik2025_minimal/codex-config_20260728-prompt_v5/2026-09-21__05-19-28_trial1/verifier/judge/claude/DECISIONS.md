# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as all directories in the data root (sorted). Sessions within each subject are identified as subdirectories whose name starts with 4 digits (sorted). For each session, the AI loads suite2p outputs (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy` from `suite2p/plane0/`) and motion energy data (`motion_energy_glob.npy`, `interframe_int.npy` from `move_deve/`).

ii.
```python
def sorted_subjects(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir())

def sorted_sessions(subject_dir: Path) -> list[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )

# In suite2p_dff:
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
f = np.load(plane_dir / "F.npy").astype(np.float32)
fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)

# In reconstruct_motion_energy:
motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy").astype(np.float32)
interframe = np.load(session_dir / "move_deve" / "interframe_int.npy")
```

iii. The agent systematically explored the data directory structure and identified suite2p output files and motion energy files in each session. It chose to load `ops.npy` to read session-specific preprocessing parameters rather than hardcoding them, and loaded `iscell.npy` to verify that all ROIs passed the 0.5 cell classification threshold (as described in the paper's methods). The session name filter (`name[:4].isdigit()`) ensures only actual recording session directories are included.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all directories in the data root, sorted alphabetically. Subject names are the directory names (e.g., `jm031`, `jm032`, etc.).

ii.
```python
def sorted_subjects(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir())

subjects = [subject_dir.name for subject_dir in sorted_subjects(data_root)]
```

iii. The agent identified that each directory in the data root represents one mouse. Unlike the reference which filters by `jm*` prefix, the AI includes all directories. In practice, all directories in the data root are `jm*` directories, so the result is the same.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder whose name starts with 4 digits, sorted alphabetically. Each subdirectory contains one daily recording session.

ii.
```python
def sorted_sessions(subject_dir: Path) -> list[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )
```

iii. The agent observed that session directories are named with date-like prefixes (e.g., `2024...`). The `name[:4].isdigit()` filter ensures only actual session directories are included, excluding any non-session files or folders. This is slightly more restrictive than the reference (which takes all subdirectories) but yields the same result for this dataset.

## 1-d. How are the data split into trials?

i. There is no natural trial structure in this dataset. Trials are defined as consecutive non-overlapping 60-second windows. At 30 Hz with 10-frame bins, each trial is 180 time bins. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
trial_bins = int(TRIAL_SECONDS * FRAME_RATE_HZ / DENOISE_BIN_FRAMES)  # 180
n_complete_trials = neural_binned.shape[1] // trial_bins

usable = n_complete_trials * trial_bins
neural_binned = neural_binned[:, :usable]

for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
    neural_trials.append(neural_binned[:, start:end])
```

iii. The task instructions specify "Split sessions into 60-second trials." Since the recording is continuous spontaneous behavior with no stimulus events, the agent defined trials as fixed-length contiguous segments.

## 1-e. How are trials filtered based on quality controls?

i. The AI checks that each session has at least 2 complete trials, raising an error if not. No other trial-level quality filtering is applied.

ii.
```python
if n_complete_trials < 2:
    raise ValueError(f"{session_dir}: fewer than two complete 60-second trials")
```

iii. The instruction states "There needs to be at least two trials within each session in order to evaluate the decoder performance." The agent included this validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (preprocessing parameters), all from `suite2p/plane0/`. The agent also loads `iscell.npy` for validation.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
f = np.load(plane_dir / "F.npy").astype(np.float32)
fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)
```

iii. The agent followed the paper's description of using Suite2p baseline-corrected fluorescence traces. It loaded `ops.npy` to use session-specific parameters rather than relying on hardcoded defaults.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied using the coefficient from `ops.npy` (default 0.7): `Fc = F - neucoeff * Fneu`. Then Suite2p's `dcnv.preprocess` is called with parameters read from `ops.npy` (baseline='maximin', win_baseline=60.0, sig_baseline=10.0, fs=30, prctile_baseline=8). The result is then averaged in non-overlapping 10-frame bins.

ii.
```python
fc = f - float(ops.get("neucoeff", 0.7)) * fneu

dff = preprocess(
    fc.copy(),
    baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=float(ops.get("fs", FRAME_RATE_HZ)),
    prctile_baseline=float(ops.get("prctile_baseline", 8)),
    batch_size=100,
    device=torch.device("cpu"),
)

neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
```

iii. The agent investigated the paper's statement about using "default Suite2p parameters" for baseline-corrected fluorescence. It found that `ops.npy` stores the actual parameters used during Suite2p processing and chose to read them per-session rather than hardcoding, making the code more robust. The agent also found and resolved the discrepancy between Track2p's GUI code (which hardcodes `neucoeff=0.0`) and Suite2p defaults (`neucoeff=0.7`), choosing to follow `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI loads `iscell.npy` and verifies that all ROIs have probability > 0.5. If any ROI fails this check, a ValueError is raised. In practice, all ROIs in the Track2p output already pass, so no neurons are actually filtered out.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found tracked ROIs below the 0.5 iscell threshold")
```

iii. The paper states that `iscell > 0.5` is used to identify cells. The agent verified that all Track2p-tracked ROIs already pass this threshold, so the check is a validation guard rather than an active filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The metadata reports `off_start=0.0` and `off_end=60.0`.

ii.
```python
"temporal_alignment_event": "start of each consecutive 60-second trial",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. There is no stimulus event in this spontaneous behavior paradigm. Trials are consecutive windows starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The original data is at 30 Hz (one frame per ~33.3 ms). Non-overlapping 10-frame averaging is applied, resulting in a time bin size of 333.33 ms (3 Hz effective rate). Both neural and motion energy are binned together before discretization.

ii.
```python
DENOISE_BIN_FRAMES = 10

neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES).astype(np.float32)

"time_bin_size": 1000.0 * DENOISE_BIN_FRAMES / FRAME_RATE_HZ,  # 333.33 ms
```

iii. The paper methods state: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent adopted this exactly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from frame indices and the known frame rate (30 Hz). Frame times are computed as `np.arange(raw_frames) / 30.0`, then averaged in 10-frame bins.

ii.
```python
frame_times_s = np.arange(raw_frames, dtype=np.float32) / FRAME_RATE_HZ
time_binned = mean_bin_1d(frame_times_s, DENOISE_BIN_FRAMES).astype(np.float32)
```

iii. Since the imaging rate is constant at 30 Hz and no explicit timestamps are stored with the neural data, computing time from frame indices is equivalent and straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame times in seconds are computed for each frame (`frame_index / 30.0`), then averaged in 10-frame bins via `mean_bin_1d`. This yields the temporal center of each bin in seconds from session start. Values start at 0.15s (center of first bin) and increase in steps of 0.333s.

ii.
```python
frame_times_s = np.arange(raw_frames, dtype=np.float32) / FRAME_RATE_HZ
time_binned = mean_bin_1d(frame_times_s, DENOISE_BIN_FRAMES).astype(np.float32)
```

iii. The agent computed per-frame times then applied the same binning as the neural data, yielding bin-center times.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is derived from the same frame indices as the neural data and undergoes the same binning procedure, so it is inherently aligned. Each time bin corresponds exactly to the same neural data bin.

ii.
```python
input_trials.append(time_binned[np.newaxis, start:end].astype(np.float32, copy=False))
```

iii. Since time is computed from the frame indices that index the neural data, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (the pre-computed global motion energy signal) and `interframe_int.npy` (inter-frame intervals for detecting dropped camera frames), both in the `move_deve` subdirectory of each session.

ii.
```python
motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy").astype(np.float32)
interframe = np.load(session_dir / "move_deve" / "interframe_int.npy")
```

iii. The motion energy file contains the behavioral video motion energy, and the interframe interval file is needed to reconstruct the correct frame-to-frame alignment when camera frames were dropped.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) Dropped camera frames are detected using interframe intervals and linearly interpolated to match the neural frame count. (2) The trace is averaged in 10-frame non-overlapping bins. (3) The binned signal is discretized into 5 equal-percentile bins computed within each session.

ii.
```python
# Step 1: Reconstruct dropped frames
median_dt = float(np.median(interframe))
frame_steps = np.rint(interframe / median_dt).astype(np.int64)
frame_steps = np.maximum(frame_steps, 1)
known_positions = np.concatenate([[0], np.cumsum(frame_steps)])
full_motion = np.full(target_len, np.nan, dtype=np.float32)
full_motion[known_positions] = motion
valid = np.flatnonzero(~np.isnan(full_motion))
return np.interp(np.arange(target_len), valid, full_motion[valid]).astype(np.float32)

# Step 2: Bin
motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES).astype(np.float32)

# Step 3: Discretize
edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The agent detected frame mismatches between neural and motion energy data, analyzed interframe intervals to understand the drop pattern, and implemented a reconstruction approach using median interval ratios and linear interpolation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session. Bin edges are computed using `np.quantile` at percentiles [0.2, 0.4, 0.6, 0.8]. `np.digitize` maps values to bins 0-4. A fallback is provided for edge cases where quantile edges are not unique, using rank-based splitting.

ii.
```python
def discretize_equal_percentile(values: np.ndarray, nbins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
    if np.unique(edges).shape[0] == edges.shape[0]:
        return np.digitize(values, edges, right=False).astype(np.int64)

    # Fallback for tied edges
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(values.shape[0], dtype=np.int64)
    for bin_idx, idx in enumerate(np.array_split(order, nbins)):
        bins[idx] = bin_idx
    return bins
```

iii. The task specifies "five equal-percentile bins, selected per session." The agent implemented per-session percentile-based binning with a robust fallback.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy and neural data are acquired synchronously at 30 Hz but occasionally camera frames are dropped. The AI uses interframe intervals to compute the true frame positions of each motion energy sample, places them at the correct indices in a target-length array, then linearly interpolates the gaps. After reconstruction, both streams have the same number of frames and are indexed identically. Both are then binned with the same 10-frame averaging.

ii.
```python
full_motion = np.full(target_len, np.nan, dtype=np.float32)
full_motion[known_positions] = motion
valid = np.flatnonzero(~np.isnan(full_motion))
return np.interp(np.arange(target_len), valid, full_motion[valid]).astype(np.float32)
```

iii. The agent verified that after reconstruction, the motion energy array length exactly matches the neural frame count, ensuring frame-by-frame alignment before binning and trial splitting.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected via interframe intervals and linearly interpolated. The agent validates that reconstructed motion energy matches the neural frame count. Sessions with fewer than 2 complete trials would raise an error. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
# Frame reconstruction
if motion.shape[0] == target_len:
    return motion
# ... reconstruction via interframe intervals and interpolation

# Trial completeness check
if n_complete_trials < 2:
    raise ValueError(...)

# Remainder trimming
usable = n_complete_trials * trial_bins
neural_binned = neural_binned[:, :usable]
```

iii. The agent systematically checked all sessions for frame count mismatches and implemented robust reconstruction. Error-raising (rather than silent data loss) is used for unexpected conditions.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p `dcnv.preprocess` baseline correction, which runs on CPU (the agent hardcodes `torch.device("cpu")`). Loading `.npy` files is I/O bound but relatively fast.

ii.
```python
dff = preprocess(
    fc.copy(),
    ...
    device=torch.device("cpu"),
)
```

iii. The baseline correction involves sliding window operations over the full session length for every neuron.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `discretize_equal_percentile` fallback path uses a Python loop over bins with `np.array_split`, though this is only used when quantile edges are tied (unlikely in practice). The trial-splitting loop is standard and not easily vectorizable due to the list-of-arrays output format.

ii.
```python
for bin_idx, idx in enumerate(np.array_split(order, nbins)):
    bins[idx] = bin_idx
```

iii. The fallback is rarely triggered, so the performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The code iterates over subjects/sessions twice conceptually (once in `sorted_subjects` and once in the main loop), but this is just directory listing, not data processing. No substantive computation is repeated.

ii.
```python
for subject_dir in sorted_subjects(data_root):
    for session_dir in sorted_sessions(subject_dir):
        ...
```

iii. The code is well-structured with a single pass through the data.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `session_to_trials` function returns `motion_bins` (the full-session discretized motion energy) as a fourth return value, but in `build_dataset` it is discarded (assigned to `_`). The code also stores extra metadata fields (session_ids, trial_counts, source_data_root) that aren't required by the target format.

ii.
```python
neural_trials, input_trials, output_trials, _ = session_to_trials(session_dir)
```

iii. The extra return value has minimal overhead. The extra metadata fields are informational and harmless.
