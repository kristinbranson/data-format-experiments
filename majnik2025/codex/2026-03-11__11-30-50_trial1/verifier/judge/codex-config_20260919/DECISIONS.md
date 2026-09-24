# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every `jm*` subject directory and every date-like session subdirectory, then processes each session. It loads Suite2p `ops.npy`, `F.npy`, and `Fneu.npy`, plus behavioral `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`. Full mode uses all 41 discovered sessions.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
        sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
...
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. The AI states that the release contains 6 mice and 41 continuous daily sessions and that all sessions pass the core criteria. It uses imaging frames as the master clock and retains the full released recordings, including 30-minute recordings that exceed the paper's stated 20 minutes.

## 1-b. How are the data split into subjects?

i. Subject identity is the sorted `jm*` parent-directory name. Unique names populate `subjects`; each session receives the corresponding integer in `subject_idx`.

ii.
```python
subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.array([subject_to_idx[session["info"].subject]
                         for session in processed_sessions], dtype=np.int64),
```

iii. The notes justify this with the dataset convention: each `jm*` folder is one mouse. The resulting 6 subjects match the paper and release.

## 1-c. How are the data split into sessions?

i. Each sorted date-like subdirectory under a subject is one session. One output session is produced for each discovered daily recording.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir()
                          if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name,
                                session=session_dir.name, path=session_dir))
```

iii. The AI describes the raw data as continuous daily recordings and reports 41 output sessions, matching the release.

## 1-d. How are the data split into trials?

i. The AI creates consecutive, non-overlapping 120-second pseudo-trials after 10-frame binning. At 3 binned samples/s, each trial has 360 samples. Only complete blocks are retained.

ii.
```python
TRIAL_DURATION_SEC = 120.0
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)
n_trials = n_total_bins // trial_bins
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
```

iii. The AI chose two-minute blocks because the paper used consecutive two-minute blocks for decoder cross-validation, calling this a natural paper-consistent trial structure. This overlooks the benchmark's explicit instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral-quality trial filtering. Incomplete terminal blocks are implicitly dropped by floor division, and the code rejects any session yielding fewer than two complete pseudo-trials.

ii.
```python
n_trials = n_total_bins // trial_bins
...
if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
```

iii. The AI notes that the source is continuous spontaneous behavior with no native trial curation. All sessions yield at least 10 of its two-minute blocks, so none is removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from Suite2p `F.npy` and `Fneu.npy`; preprocessing parameters and expected frame count come from `ops.npy`.

ii.
```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. The AI selected paper-consistent baseline-corrected fluorescence rather than `spks.npy` or raw fluorescence, because the methods describe Suite2p-style baseline-corrected fluorescence for downstream analyses.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil using the session's saved `ops["neucoeff"]`, then calls Suite2p `dcnv.preprocess` with the remaining saved baseline parameters. The result is cast to float32 and averaged in non-overlapping 10-frame bins.

ii.
```python
corrected = np.array(F, dtype=np.float32, copy=True)
corrected -= np.float32(ops["neucoeff"]) * np.asarray(Fneu, dtype=np.float32)
processed = dcnv.preprocess(
    corrected, baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]), fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 2000)), device=torch.device("cpu"),
)
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```

iii. The notes say this follows the paper's Suite2p/default-parameter processing. Using saved per-session `ops` values is a defensible refinement over hard-coded defaults.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed; every row in the released `F`/`Fneu` arrays is kept.

ii.
```python
"brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
"neural_binned": neural_binned,
```

iii. The AI concluded that the release already contains Track2p all-day matched, Suite2p-filtered cells. Identical within-subject neuron counts and row identities supported retaining all exported rows.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Each trial is aligned to the start of its consecutive artificial block, with slices shared across neural, time, and output data.

ii.
```python
neural_trial = neural_binned[:, start:stop]
...
"temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SEC,
```

iii. The AI explains that the recordings are continuous and have no stimulus event, so block onset is the only applicable alignment event. The two-minute duration itself conflicts with the requested one-minute trialization.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw 30 Hz data are averaged over non-overlapping groups of 10 frames, yielding 3 Hz samples and a 333.333 ms time bin. A short tail below 10 frames would be discarded.

ii.
```python
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ
trimmed = x[..., : n_bins * bin_frames]
return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32)
```

iii. The paper states that neural and behavioral traces were denoised for decoding by averaging 10 consecutive timestamps. The AI applies the same operation to all aligned streams before trialization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from imaging-frame indices and the sampling rate in `ops.npy`; it does not use behavioral timestamps.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. The AI treats imaging frames as the authoritative clock because imaging is fixed at 30 Hz and behavior is microscope-triggered.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It assigns each raw frame its center time `(frame + 0.5)/fs`, then averages each group of 10 center times. Thus the first binned input is approximately 0.167 s and subsequent values are spaced by one third of a second.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The AI intended to encode elapsed session time at the temporal center of each averaged bin. Its notes report the observed range as about 0.2 to 1799.8 seconds.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is binned with identical 10-frame boundaries and sliced with identical trial indices, so every input sample denotes the center of the neural averaging window at that position.

ii.
```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
...
neural_trial = neural_binned[:, start:stop]
input_trial = time_binned[np.newaxis, start:stop]
```

iii. The notes say all streams use the same master frame clock, bin boundaries, and segmentation, and a raw-versus-converted time check passed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is based on `move_deve/motion_energy_glob.npy`. `interframe_int.npy` is used to reconstruct missing camera frames; `tstamps.npy` is loaded and retained only for diagnostics.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. The AI identifies the global motion-energy trace as the paper's behavioral signal and uses camera timing gaps to correct its occasional length mismatch with imaging.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing frames are located by rounding each interframe interval relative to the median interval, observed samples are placed on the imaging grid, and gaps are linearly interpolated. Motion is then averaged in 10-frame bins, globally min-max normalized across all sessions, and passed to categorical discretization.

ii.
```python
steps = np.rint(interframe_int / median_interval).astype(np.int64)
observed_idx[1:] = np.cumsum(steps)
full[observed_idx] = motion
full = interpolate_nans(full)
...
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
```

iii. The AI says interpolation restores synchronous frame alignment, 10-frame averaging follows the paper, and global min-max normalization was added to satisfy its reading of the benchmark wording. It acknowledges normalization does not affect percentile ranks.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five class boundaries are computed from the 0, 20, 40, 60, 80, and 100 percentiles pooled over every binned sample in every processed session. `np.digitize` produces classes 0–4. Thresholds are global, not session-specific.

ii.
```python
motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions])
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
...
output_trial = np.digitize(motion_norm_binned[start:stop],
                           motion_edges[1:-1], right=False).astype(np.int64)
```

iii. The AI deliberately sought globally balanced classes and documented exact 20% global class frequencies. This conflicts with “five equal-percentile bins, selected per session” and produces strongly unequal within-session distributions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Imaging frames are the master grid. When behavior is short, the code reconstructs observed indices from camera intervals, interpolates missing positions to exactly `ops["nframes"]`, and then uses the same 10-frame bins and trial slices as neural data.

ii.
```python
if observed_idx[-1] != target_len - 1:
    raise ValueError(...)
full = np.full(target_len, np.nan, dtype=np.float32)
full[observed_idx] = motion
full = interpolate_nans(full)
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1])
```

iii. The AI cites microscope-triggered 30 Hz acquisition and the data README's missing-camera-frame guidance. It reports successful spot checks, including sessions with 116 and 148 missing frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short behavioral traces are expanded using timing gaps and linearly interpolated; impossible timing reconstruction, all-NaN input, invalid motion range, no sessions, or fewer than two trials causes an explicit error. Incomplete bin/trial tails are dropped. Thirty-minute sessions are retained rather than cropped.

ii.
```python
if observed_idx[-1] != target_len - 1:
    raise ValueError("Timing reconstruction failed ...")
...
if not valid.any():
    raise ValueError("Cannot interpolate an array containing only NaNs.")
...
n_trials = n_total_bins // trial_bins
```

iii. The AI prioritizes exact post-reconstruction alignment and fail-fast validation over silently truncating modalities. It documents the duration discrepancy and justifies preserving all released data.

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p baseline correction over every neuron and frame is the dominant compute step; loading/copying large fluorescence arrays and serializing the roughly 414 MB pickle are also substantial. Optional plotting and decoder training are outside the core conversion cost.

ii.
```python
processed = dcnv.preprocess(..., device=torch.device("cpu"))
...
with args.outpicklefile.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly identify Suite2p fluorescence preprocessing as the main cost. Full per-session timings grow with neuron and frame count; session-wise handling keeps raw-memory use bounded.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Discovery and per-session processing must largely iterate over heterogeneous files, but the trial-slicing loop could be replaced with reshape/view operations when all trials have equal length. Summary concatenation and list construction could likewise be streamlined. Missing-frame reconstruction and temporal binning are already vectorized.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trials.append(neural_binned[:, start:stop])
...
steps = np.rint(interframe_int / median_interval).astype(np.int64)
observed_idx[1:] = np.cumsum(steps)
```

iii. The AI highlights reshape-based averaging and vectorized timing-step accumulation as implemented speedups. It does not claim that the small trial loop is a major bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. Motion normalization and digitization are performed once for dataset construction and repeated for optional plots. Every session also creates preview arrays and loads `tstamps.npy`, even when plots are not requested. The full motion trace is concatenated once for global statistics and each session is normalized again during assembly.

ii.
```python
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
...
motion_norm = (session["motion_binned"] - motion_min) / (motion_max - motion_min)
...
motion_norm_binned = (motion_binned - motion_min) / (motion_max - motion_min)
motion_classes = np.digitize(motion_norm_binned, motion_edges[1:-1])
```

iii. The AI's notes discuss bounded session-wise processing but do not call out these repetitions. They are minor compared with neural preprocessing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It always builds neural and motion preview dictionaries and loads timestamps although these are used only if `--show-processing` is enabled. Global min-max normalization is mathematically unnecessary before percentile thresholding because it preserves ranks. Diagnostic summaries concatenate all output trials after construction.

ii.
```python
timestamps = np.load(move_dir / "tstamps.npy")
neural_preview = {"F": ..., "Fneu": ..., "corrected": ..., "processed": ...}
motion_preview = {"raw": ..., "timestamps": ..., "interframe_int": ...}
...
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, ...)
```

iii. The AI says normalization was included to satisfy its interpretation of the task even though percentile labels are unchanged. Preview generation supported visual validation, but in ordinary full conversion it is computed and then discarded.
