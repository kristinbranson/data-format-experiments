# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for sorted `jm*` subject directories, scans their sorted session directories, and retains sessions containing both `F.npy` and `motion_energy_glob.npy`. Discovery memory-maps those two files to obtain dimensions. Full conversion then loads each selected session's `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`, and `ops.npy` one session at a time. Full mode sorts every discovered session by subject and session; sample mode deliberately selects two representative sessions.

ii.
```python
def sorted_subjects(data_root: Path) -> list[str]:
    return sorted(path.name for path in data_root.iterdir()
                  if path.is_dir() and path.name.startswith("jm"))

for subject in subjects:
    for session_dir in sorted(path for path in (data_root / subject).iterdir()
                              if path.is_dir()):
        f_path = session_dir / "suite2p" / "plane0" / "F.npy"
        motion_path = session_dir / "move_deve" / "motion_energy_glob.npy"
        if not f_path.exists() or not motion_path.exists():
            continue
```
```python
raw_f = np.load(session_info.session_dir / "suite2p" / "plane0" / "F.npy")
raw_fneu = np.load(session_info.session_dir / "suite2p" / "plane0" / "Fneu.npy")
raw_motion = np.load(session_info.session_dir / "move_deve" / "motion_energy_glob.npy")
tstamps = np.load(session_info.session_dir / "move_deve" / "tstamps.npy")
interframe_int = np.load(session_info.session_dir / "move_deve" / "interframe_int.npy")
```

iii. The notes say this deterministic traversal covers all six subjects and 41 daily sessions while processing one session at a time to bound memory. Memory mapping is used only for cataloguing. The full-run checks report 1,090 trials and counts matching the raw release.

## 1-b. How are the data split into subjects?

i. A subject is a sorted directory whose name starts with `jm`. For the chosen sessions, unique subject names are sorted and mapped to integer indices; each session receives the corresponding `subject_idx`.

ii.
```python
selected_subjects = sorted({info.subject for info in session_infos})
subject_to_idx = {subject: idx for idx, subject in enumerate(selected_subjects)}
data["subject_idx"].append(subject_to_idx[session_info.subject])
```

iii. The notes identify each `jm*` folder as one mouse and state that folder naming and per-mouse constant neuron counts agree with the release's longitudinal structure.

## 1-c. How are the data split into sessions?

i. Every valid daily subdirectory of a subject is one session, ordered by `(subject, session)`. One outer list element is produced for each session.

ii.
```python
session_infos = sorted(all_session_infos,
                       key=lambda info: (info.subject, info.session))
for idx, session_info in enumerate(session_infos, start=1):
    payload, stats = convert_one_session(session_info, ...)
    data["neural"].append(payload["neural_trials"])
```

iii. The AI states that the date-named subdirectories are daily recordings and reports all 41 sessions (7, 7, 7, 7, 6, and 7 per mouse).

## 1-d. How are the data split into trials?

i. Continuous sessions are split into contiguous, non-overlapping 60-second windows after 10-frame binning. At 30 Hz this is 180 binned samples per trial. Only complete trials are retained, and sessions with fewer than two are rejected.

ii.
```python
TRIAL_BINS = TRIAL_FRAMES // BIN_FRAMES
usable = (n_bins // TRIAL_BINS) * TRIAL_BINS
if usable < TRIAL_BINS * 2:
    raise ValueError("Session does not contain at least two 60-second trials after binning.")
for trial_idx in range(n_trials):
    sl = slice(trial_idx * TRIAL_BINS, (trial_idx + 1) * TRIAL_BINS)
    neural_trials.append(neural_binned[:, sl])
```

iii. The task explicitly requests 60-second trials and the source recordings have no natural trial structure. The notes say all released sessions divide into 20 or 30 complete windows and concatenation checks reconstruct the pre-trial arrays.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. All complete 60-second windows are kept; incomplete tails would be dropped, and a session is rejected if fewer than two complete trials remain.

ii.
```python
usable = (n_bins // TRIAL_BINS) * TRIAL_BINS
neural_binned = neural_binned[:, :usable]
```

iii. The notes say every session yields at least 20 trials and no session exclusion is justified. The complete-window rule satisfies the decoder's fixed-size and minimum-two-trials requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from matched Suite2p `plane0/F.npy` and `plane0/Fneu.npy`, with preprocessing parameters read from `plane0/ops.npy`.

ii.
```python
f = np.load(f_path).astype(np.float32, copy=False)
fneu = np.load(fneu_path).astype(np.float32, copy=False)
ops = load_ops(session_dir)
```

iii. The AI chose the paper's baseline-corrected fluorescence rather than `spks.npy`, noting that the release already contains Track2p-matched cells and provides the raw fluorescence, neuropil trace, and Suite2p settings required to reproduce it.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil using the session's saved `ops['neucoeff']`, applies Suite2p `dcnv.preprocess` with the session's saved baseline parameters on CPU, casts to float32, then averages non-overlapping groups of 10 frames.

ii.
```python
dff = f.copy()
dff -= np.float32(ops["neucoeff"]) * fneu
dff = dcnv.preprocess(
    dff, baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]), fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 100)), device=torch.device("cpu"),
)
neural_binned = bin_array(dff, BIN_FRAMES)
```

iii. The paper says decoding used Suite2p baseline-corrected fluorescence and averaged 10 consecutive timestamps. Reading saved parameters was intended to reproduce the actual per-session Suite2p setup rather than assume defaults; spot checks against independent recomputation passed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies no new neuron filter. It uses all rows of the released `F.npy`, treating them as already filtered at Suite2p probability greater than 0.5 and already Track2p-matched across all days for each mouse.

ii.
```python
raw_f = np.load(session_info.session_dir / "suite2p" / "plane0" / "F.npy")
# No iscell mask or further row selection is applied.
```

iii. The notes report that every released `iscell[:, 1]` exceeds 0.5 and neuron counts remain constant within a mouse, supporting the release documentation that these are matched exports. Re-filtering or rerunning Track2p would therefore be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial zero is anchored at session start and subsequent trials are consecutive 60-second windows. Neural, input, and output use the identical slice for each window. Metadata describes alignment to the start of each window with offsets 0 to 60 seconds.

ii.
```python
sl = slice(trial_idx * TRIAL_BINS, (trial_idx + 1) * TRIAL_BINS)
neural_trials.append(neural_binned[:, sl])
input_trials.append(time_binned[np.newaxis, sl])
output_trials.append(motion_bins[np.newaxis, sl])
```
```python
"temporal_alignment_event": "start of each contiguous 60-second session window",
"off_start": 0.0,
"off_end": TRIAL_SECONDS,
```

iii. There is no experimental event or native trial onset. The AI therefore treats each artificial window start as the alignment event while preserving absolute session time in the input.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10 frames at 30 Hz, or 333.333 ms. Neural activity, elapsed time, and reconstructed motion energy are all averaged in non-overlapping 10-frame bins before trialization.

ii.
```python
BIN_FRAMES = 10
BIN_SIZE_SEC = BIN_FRAMES / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SEC * 1000.0
return trimmed.reshape(trimmed.shape[0], -1, bin_frames).mean(axis=2)
```

iii. This directly follows the paper's decoding denoising step of averaging 10 consecutive timestamps and keeps every stream on one common grid.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from the imaging frame index and fixed 30 Hz frame rate, not loaded from a raw time variable.

ii.
```python
time_values = (np.arange(session_info.n_frames, dtype=np.float32)
               / np.float32(FRAME_RATE_HZ))
```

iii. The AI notes that the decoder specification requires session-elapsed time and that the imaging frame grid is the canonical complete clock.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Frame indices are converted to seconds and then averaged in the same non-overlapping 10-frame groups. Thus each value is the mean/center time of its bin (0.15, 0.4833, ...), remains absolute from session start, and does not reset at trial boundaries.

ii.
```python
time_values = np.arange(session_info.n_frames, dtype=np.float32) / 30.0
time_binned = bin_array(time_values, BIN_FRAMES)
```

iii. The notes explicitly choose absolute session time because the task asks for elapsed time from session start. Applying the common averaging operation makes its timestamp correspond to the samples represented in each binned neural/behavior value.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is created on the imaging frame grid, binned with the identical 10-frame grouping, trimmed to the same complete-trial extent, and sliced with the same trial slice as neural data.

ii.
```python
neural_binned = bin_array(dff, BIN_FRAMES)
time_binned = bin_array(time_values, BIN_FRAMES)
input_trials.append(time_binned[np.newaxis, sl])
```

iii. The AI chose imaging frames as the canonical time axis and added reconstruction assertions after trial splitting to ensure the trial inputs concatenate back to the aligned session series.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output originates from `move_deve/motion_energy_glob.npy`. `tstamps.npy` is used to infer where observed behavior samples belong on the imaging grid; `interframe_int.npy` is loaded and summarized but does not drive reconstruction.

ii.
```python
raw_motion = np.load(... / "motion_energy_glob.npy")
tstamps = np.load(... / "tstamps.npy")
interframe_int = np.load(... / "interframe_int.npy")
full_motion, frame_idx, missing_mask = reconstruct_motion_to_imaging_grid(
    raw_motion, tstamps, session_info.n_frames)
```

iii. The notes follow the data README's instruction that missing camera frames can be located using timestamps/inter-frame intervals, and use timestamps because they map observed samples directly to the canonical imaging axis.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Timestamp-derived indices place observed motion samples onto an imaging-length array; missing positions are linearly interpolated. The dense trace is then averaged over 10-frame bins and discretized using session-specific quintile thresholds. A rank fallback is used if tied values would otherwise produce fewer than five categories.

ii.
```python
full_motion = np.full(n_frames, np.nan, dtype=np.float32)
full_motion[frame_idx] = motion.astype(np.float32, copy=False)
full_motion = np.interp(np.arange(n_frames), valid_idx, full_motion[valid_idx])
motion_binned = bin_array(full_motion, BIN_FRAMES)
motion_bins, motion_edges, method = discretize_motion_quintiles(motion_binned)
```

iii. The AI intended to preserve every measured motion value, fill only genuinely missing camera frames, reproduce the paper's 10-frame denoising, and satisfy the requested five equal-percentile categories independently per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four within-session quantiles (20%, 40%, 60%, 80%) are calculated from the full binned motion trace. `searchsorted(..., side='right')` assigns integer classes 0–4. If ties collapse the result below five observed classes, stable rank assignment forces five approximately equal-count categories.

ii.
```python
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
discrete = np.searchsorted(edges, motion_binned, side="right").astype(np.int64)
if np.unique(discrete).size < 5:
    order = np.argsort(motion_binned, kind="stable")
    discrete[order] = np.minimum(4, (5 * np.arange(motion_binned.size)) // motion_binned.size)
```

iii. The requested output is five equal-percentile bins selected per session. The fallback was added so ties cannot eliminate a required category; full-run notes report exactly 20% per class in every session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Behavior timestamps are converted to imaging-frame indices using an inferred uniform imaging step. Observations are placed at those indices, gaps are interpolated, and motion is binned and sliced identically to neural activity.

ii.
```python
dt = float((tstamps[-1] - tstamps[0]) / (n_frames - 1))
frame_idx = np.rint((tstamps - tstamps[0]) / dt).astype(np.int64)
full_motion[frame_idx] = motion
...
output_trials.append(motion_bins[np.newaxis, sl])
```

iii. Because the behavior camera was microscope-triggered at 30 Hz but occasionally dropped frames, the AI makes imaging frames canonical. It verifies inferred missing counts and exact preservation of motion at every observed timestamp.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior frames are inferred from timestamp gaps and linearly interpolated on the imaging grid. The code validates timestamp dimensionality, monotonicity, uniqueness, bounds, inferred-versus-expected missing counts, observed sample preservation, finite values, at least two trials, and trial reconstruction. Incomplete bin/trial tails are trimmed; absent required discovery files cause a session directory to be skipped.

ii.
```python
if np.unique(frame_idx).size != frame_idx.size:
    raise ValueError("Behavior timestamps map multiple samples to the same imaging frame.")
...
if inferred_missing != expected_missing:
    raise ValueError("Missing-frame mismatch ...")
if not np.isfinite(neural_binned).all():
    raise ValueError("Non-finite neural values ...")
```

iii. The notes emphasize fail-fast validation rather than silent misalignment. Representative missing-frame and no-drop sessions were tested, and observed values plus inferred gap counts matched the raw data.

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p baseline preprocessing is the main compute cost; array loading is the other notable cost. Optional high-resolution diagnostic plotting adds work. The script records load, neural, motion, postprocessing, and total timings per session.

ii.
```python
t1 = time.perf_counter()
dff, ops = compute_suite2p_dff(session_info.session_dir)
neural_sec = time.perf_counter() - t1
```

iii. The notes identify baseline correction over every neuron and frame as dominant, although the final full run took only about 20.6 seconds. One-session-at-a-time processing controls memory.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most numerical work is vectorized. The session loop is intentionally sequential and could be parallelized, while the short trial loop could be replaced by reshape/split operations. The diagnostic summary/metadata loops are negligible.

ii.
```python
for idx, session_info in enumerate(session_infos, start=1):
    payload, stats = convert_one_session(session_info, ...)
...
for trial_idx in range(n_trials):
    sl = slice(trial_idx * TRIAL_BINS, (trial_idx + 1) * TRIAL_BINS)
```

iii. The AI explicitly notes that conversion is single-process by session. It deliberately vectorized binning and timestamp reconstruction; with 20–30 trials per session and a short total runtime, further vectorization was not considered necessary.

## 6-c. What processing does the code repeat multiple times?

i. `F.npy` and `Fneu.npy` are loaded once in `convert_one_session` for possible plotting, then loaded again inside `compute_suite2p_dff`. `ops.npy` is loaded in neural preprocessing and, when plotting, again inside `save_processing_plot`. Discovery also memory-maps `F.npy` and motion once for shapes. Trial reconstruction concatenates data that were just sliced, solely for validation.

ii.
```python
raw_f = np.load(... / "F.npy")
raw_fneu = np.load(... / "Fneu.npy")
dff, ops = compute_suite2p_dff(session_info.session_dir)  # reloads F/Fneu
```
```python
neuropil_sub = raw_f[...] - load_ops(session_info.session_dir)["neucoeff"] * raw_fneu[...]
```

iii. The notes acknowledge the repeated `ops.npy` load in plotting but call it negligible. They do not explicitly call out the duplicate fluorescence loads; the reconstruction concatenations are purposeful sanity checks.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `interframe_int.npy` is loaded but only used in runtime statistics, not alignment. Optional plots calculate preview traces/histograms and are not part of the pickle. Motion edges, timing statistics, and several validation reconstructions support diagnostics; the local `motion_edges` field in `session_payload` is not consumed by `build_dataset` (the edges are instead copied from `session_stats`). Raw arrays are retained longer than needed when plots are disabled.

ii.
```python
interframe_int = np.load(... / "interframe_int.npy")
...
"interframe_median": float(np.median(interframe_int)),
```
```python
session_payload = { ..., "motion_edges": motion_edges.astype(np.float32) }
# build_dataset does not read payload["motion_edges"]
```

iii. The AI justifies most extra work as validation, provenance, timing, or optional visualization. Its notes only explicitly flag the extra `ops.npy` plotting load; the other discarded intermediates are minor at this dataset scale.
