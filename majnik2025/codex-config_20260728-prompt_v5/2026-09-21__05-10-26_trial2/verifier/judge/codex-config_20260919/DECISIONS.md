# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every sorted `jm*` subject directory and every sorted date-like session directory below it. For each session it loads `ops.npy`, `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`; full mode processes all 41 discovered sessions.

ii.
```python
for subject_dir in sorted(DATA_ROOT.iterdir()):
    if not subject_dir.is_dir() or not subject_dir.name.startswith("jm"):
        continue
    for session_dir in sorted(subject_dir.iterdir()):
        if not session_dir.is_dir() or not session_dir.name[:4].isdigit():
            continue
        sessions.append(SessionPath(subject=subject_dir.name,
                                    session=session_dir.name, path=session_dir))

f = np.load(session_path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
fneu = np.load(session_path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
motion = np.load(session_path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
```

iii. The agent documented that the release is organized as six mouse directories containing daily recordings and that the released Suite2p arrays are already Track2p-matched. It therefore preserved every available recording rather than rerunning tracking or truncating 30-minute sessions to the paper's stated 20 minutes.

## 1-b. How are the data split into subjects?

i. A subject is a directory whose name begins with `jm`. Unique subject names are sorted, and each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({session.subject for session, _ in processed_sessions})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.array(
    [subject_lookup[s.subject] for s, _ in processed_sessions], dtype=np.int64
),
```

iii. The notes identify each `jm*` directory as one mouse and report six mice, consistent with the paper and directory convention.

## 1-c. How are the data split into sessions?

i. Each date-like child directory of a subject is one session. Sessions are sorted first by subject and then directory name/date; one output session is produced per such directory.

ii.
```python
for session_dir in sorted(subject_dir.iterdir()):
    if not session_dir.is_dir() or not session_dir.name[:4].isdigit():
        continue
    sessions.append(SessionPath(...))

for idx, session in enumerate(sessions, start=1):
    payload, summary = convert_session(session, ...)
```

iii. The agent found 41 daily recordings (six or seven per mouse) and treated the actual released session directories as authoritative.

## 1-d. How are the data split into trials?

i. Each continuous session is split after 10-frame binning into consecutive, non-overlapping 60-second windows. At 30 Hz this is 180 binned time points per trial. The implementation requires the binned session length to divide exactly by 180.

ii.
```python
bins_per_trial = int(round(trial_seconds * fs / bin_frames))
reshaped = arr.reshape(n_rows, n_trials, trial_bins)
return [reshaped[:, i, :] for i in range(n_trials)]
```

iii. The notes explain that the source is continuous and has no native trials, so the 60-second windows are imposed specifically by the decoder task. All released session lengths tile exactly into these windows.

## 1-e. How are trials filtered based on quality controls?

i. No complete 60-second trial is filtered. Shape/divisibility and neural/input/output trial-count checks cause an error rather than silently dropping malformed trials.

ii.
```python
if n_time % trial_bins != 0:
    raise ValueError(...)
if not (len(neural_trials) == len(input_trials) == len(output_trials)):
    raise RuntimeError(...)
```

iii. The agent found no native trial-quality rules and chose to keep all windows having valid neural data, repairing missing behavioral frames before trial construction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is reconstructed from Suite2p `F.npy` and `Fneu.npy`; preprocessing parameters and frame count/rate come from `ops.npy`.

ii.
```python
raw_F, raw_Fneu = load_neural_arrays(session.path)
ops = load_ops(session.path)
corrected_F, neural_info = preprocess_neural(raw_F, raw_Fneu, ops)
```

iii. The agent chose baseline-corrected fluorescence rather than raw fluorescence or `spks.npy` because the paper says downstream decoding used baseline-corrected fluorescence traces as dF/F.

## 2-b. How is the `neural` data processed?

i. Neuropil is subtracted using the session's Suite2p coefficient. Suite2p `dcnv.preprocess` then performs baseline correction with the parameters stored in `ops.npy`. The result is cast to float32 and averaged in non-overlapping groups of 10 frames.

ii.
```python
fc = F - np.float32(ops["neucoeff"]) * Fneu
corrected = dcnv.preprocess(
    fc.copy(), baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]), fs=fs,
    prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
    batch_size=int(ops.get("batch_size", 2000)), device=torch.device("cpu"),
)
neural_binned = mean_bin_2d(corrected_F, bin_frames).astype(np.float32)
```

iii. The notes say this recreates the paper's Suite2p-style, baseline-corrected fluorescence representation and its stated decoding denoising by averaging 10 consecutive timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed. Every row in the released `F`/`Fneu` matrices is retained.

ii.
```python
raw_F, raw_Fneu = load_neural_arrays(session.path)
# No iscell mask or match-matrix indexing is applied.
```

iii. The agent checked that released `iscell` entries already pass the paper's 0.5 threshold and that neuron counts are constant across days within each mouse. It concluded that the release is already cell-filtered and Track2p-matched, so filtering again would be redundant and risky.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Neural data are kept in continuous session order and sliced into trial windows beginning at 0, 60, 120 seconds, etc. Metadata describes alignment to the start of each derived trial, with offsets 0 to 60 seconds.

ii.
```python
"temporal_alignment_event": "start of each derived 60-second trial window",
"off_start": 0.0,
"off_end": float(DEFAULT_TRIAL_SECONDS),
```

iii. The notes explain that the recordings are spontaneous and continuous; trial alignment is therefore only the downstream 60-second window boundary, while the input time remains absolute session time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten native 30 Hz frames are mean-binned without overlap, yielding 3 Hz samples and a 333.333 ms time bin for neural, motion, and time streams.

ii.
```python
DEFAULT_BIN_FRAMES = 10
DEFAULT_FS = 30.0
return arr[:, :usable].reshape(n_rows, usable // bin_frames, bin_frames).mean(axis=2)
"time_bin_size": 1000.0 * DEFAULT_BIN_FRAMES / DEFAULT_FS,
```

iii. This directly follows the methods statement that neural and behavioral traces were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from imaging frame indices and the session frame rate in `ops.npy`; it is not read from the behavioral timestamps.

ii.
```python
def full_frame_times_seconds(n_frames, fs):
    return np.arange(n_frames, dtype=np.float64) / fs

time_binned = build_time_input(n_frames, fs, bin_frames)
```

iii. The agent reasoned that constant-rate imaging frame indices give session elapsed time and avoid making neural time depend on dropped camera frames.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Native frame times are formed in seconds and averaged over the same 10-frame groups, so each value is the mean/center time of its bin. Absolute time continues across trial boundaries rather than resetting.

ii.
```python
frame_times = full_frame_times_seconds(n_frames, fs)
binned = mean_bin_1d(frame_times.astype(np.float32), bin_frames)
return binned.astype(np.float32)
```

iii. The notes explicitly select absolute time from session start and report a first-bin value near 0.15 s, viewing bin-center time as the natural coordinate of a mean-binned observation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural activity originate on the same imaging-frame grid, receive identical 10-frame grouping, and are split with the same 180-bin trial boundaries.

ii.
```python
neural_binned = mean_bin_2d(corrected_F, bin_frames)
time_binned = build_time_input(n_frames, fs, bin_frames)
neural_trials = split_trials_2d(neural_binned, bins_per_trial)
input_trials = split_trials_1d(time_binned, bins_per_trial)
```

iii. The shared binning and slicing were chosen to guarantee one input timestamp per neural column while preserving session-absolute elapsed time.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion comes from `move_deve/motion_energy_glob.npy`. `tstamps.npy` and `interframe_int.npy` provide behavioral timing and missing-frame information; imaging frame count and rate come from `ops.npy`.

ii.
```python
motion = np.load(session_path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(session_path / "move_deve" / "tstamps.npy", allow_pickle=True)
interframe = np.load(session_path / "move_deve" / "interframe_int.npy", allow_pickle=True)
```

iii. The agent identified the supplied global motion-energy trace as the paper's already-computed sum of squared video-frame differences and used the supplied timing arrays to repair dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Timestamp units are inferred from median interframe spacing. Motion is linearly interpolated onto the uniform imaging-frame grid, mean-binned over 10 frames, and then converted to five session-specific equal-frequency rank bins.

ii.
```python
scale = infer_timestamp_scale_seconds(interframe, fs)
motion_times = tstamps * scale
imaging_times = np.arange(n_frames, dtype=np.float64) / fs
aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
motion_binned = mean_bin_1d(aligned_motion, bin_frames)
motion_labels, motion_quantiles = equal_frequency_bins(motion_binned.astype(np.float64), 5)
```

iii. The agent chose timestamp interpolation because nine sessions have missing camera frames, then followed the paper's 10-frame denoising. Rank assignment was chosen to guarantee nearly exact 20% class counts even when values repeat.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Within each session, values are stable-sorted, assigned integer ranks, and mapped by `floor(rank * 5 / n)` to labels 0–4. Quantile values are computed for reporting but are not used as thresholds, so tied motion values can receive different categories.

ii.
```python
order = np.argsort(values, kind="mergesort")
ranks = np.empty(n, dtype=np.int64)
ranks[order] = np.arange(n, dtype=np.int64)
bins = (ranks * n_bins) // n
quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
```

iii. The notes justify rank bins as a way to force balanced classes in the presence of repeated low-motion values. This prioritizes exact class balance over assigning identical values to a common threshold-defined category.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Behavioral timestamps are scaled to seconds and motion is interpolated at every imaging time (`0, 1/fs, ...`). Neural and aligned motion are then binned in corresponding 10-frame groups and split at identical trial boundaries.

ii.
```python
motion_times = tstamps * scale
imaging_times = full_frame_times_seconds(n_frames, fs)
aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
neural_binned = mean_bin_2d(corrected_F, bin_frames)
motion_binned = mean_bin_1d(aligned_motion, bin_frames)
```

iii. The agent found behavioral timestamps to be in kiloseconds and used them to locate all dropped-frame gaps. It viewed full timestamp interpolation as more robust than inserting samples solely from array-length differences.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavioral frames are repaired by timestamp-based linear interpolation onto the complete imaging grid. Empty timestamps, non-divisible trial lengths, and stream trial-count mismatches raise errors. No partial trial exists in the released full data.

ii.
```python
if motion_times.size == 0:
    raise ValueError("No motion timestamps available.")
aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
if n_time % trial_bins != 0:
    raise ValueError(...)
```

iii. The data README permits interpolating missing camera frames. The agent chose to preserve all neural samples and explicitly validated alignment, finite values, dimensions, and session/trial counts.

## 6-a. What are the most time-consuming steps of the code?

i. Full-session Suite2p baseline correction is the dominant compute step; loading large arrays is the principal I/O cost. Optional plotting adds work only with `--show-processing`.

ii.
```python
corrected = dcnv.preprocess(fc.copy(), ..., device=torch.device("cpu"))
```

iii. The agent identified the sliding baseline operation over every neuron and frame as the bottleneck and recorded per-session timings and ETAs during the full run.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. High-volume temporal binning and motion interpolation are already vectorized. Trial lists and the session loop remain Python loops because the target format is nested and neuron/session shapes vary. The small loop over detected timestamp gaps only builds diagnostic gap sizes and could be vectorized, though its cost is negligible.

ii.
```python
aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
return arr.reshape(n_rows, usable // bin_frames, bin_frames).mean(axis=2)
for idx in gaps:
    missing = int(round((motion_times[idx + 1] - motion_times[idx]) * fs)) - 1
    gap_sizes.append(max(missing, 0))
```

iii. The notes emphasize that reshape/mean replaced frame loops and `np.interp` replaced repeated insertion. They regard remaining loops as structurally necessary or too small to matter.

## 6-c. What processing does the code repeat multiple times?

i. Loading, baseline correction, timestamp alignment, binning, and discretization repeat once per session. Plot setup repeats for at most two sessions when requested; no full-session transform is accidentally recomputed in the normal conversion path.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    payload, summary = convert_session(session, ...)
```

iii. The agent chose session-wise streaming to bound memory. Repetition across sessions is required because parameters, neurons, lengths, and percentile categories are session-specific.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In a normal run, diagnostic summaries, quantiles, gap statistics, timing, and metadata are computed but not used by decoder training. With `--show-processing`, figures and several plotting-only arrays are also produced. These are validation aids rather than inputs to downstream analysis.

ii.
```python
motion_labels, motion_quantiles = equal_frequency_bins(...)
duration = time.perf_counter() - start
if make_plot:
    build_processing_plot(...)
```

iii. The agent intentionally retained these diagnostics for sanity checks, reproducibility, and visual review; plotting is opt-in and capped at two sessions, so it does not burden the default full conversion substantially.
