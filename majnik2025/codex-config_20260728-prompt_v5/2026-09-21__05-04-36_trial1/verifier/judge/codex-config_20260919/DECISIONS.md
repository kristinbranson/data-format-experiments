# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `/app/data` for sorted `jm*` subject directories and sorted date-like session directories. Discovery reads `ops.npy`, memory-mapped `F.npy`, and memory-mapped motion energy to record dimensions. Processing then loads each selected session's full `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. Full mode processes all 41 discovered sessions; sample mode deliberately selects two representative sessions.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
        ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
        f = np.load(suite2p_dir / "F.npy", mmap_mode="r")
        motion = np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r")
```
```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
tstamps = np.load(session.move_dir / "tstamps.npy")
```

iii. The agent states that the release is organized as subject/day folders containing already tracked Suite2p outputs and synchronized behavior. It chose sorted discovery for deterministic ordering and reports that full conversion preserved all 6 mice, 41 sessions, and 1090 generated trials.

## 1-b. How are the data split into subjects?

i. A subject is a top-level directory whose name starts with `jm`. Subjects enter the output in sorted discovery order, and a dictionary maps each subject name to the integer stored in `subject_idx` for every session.

ii.
```python
if session.subject not in subject_to_idx:
    subject_to_idx[session.subject] = len(dataset["subjects"])
    dataset["subjects"].append(session.subject)
subject_idx.append(subject_to_idx[session.subject])
```

iii. The notes identify the six `jm*` folders as mice and say the folder naming convention and data organization define subject identity.

## 1-c. How are the data split into sessions?

i. Each date-like subdirectory under a subject is one session. Sessions are sorted first by subject path and then by session path; each becomes one element of the outer `neural`, `input`, and `output` lists.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, ...))
```
```python
dataset["neural"].append(session_data["neural_trials"])
dataset["input"].append(session_data["input_trials"])
dataset["output"].append(session_data["output_trials"])
```

iii. The agent treats each daily recording folder as a session and preserves the released 20- or 30-minute duration rather than forcing the paper's concise 20-minute description.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into non-overlapping 60-second windows after 10-frame averaging. At 30 Hz this gives 180 binned samples per trial. A trailing partial window is dropped.

ii.
```python
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES
usable = arr.shape[-1] - (arr.shape[-1] % bins_per_trial)
arr = arr[..., :usable]
ntrials = usable // bins_per_trial
return [arr[:, i * bins_per_trial:(i + 1) * bins_per_trial] for i in range(ntrials)]
```

iii. The source recordings have no native trials, while the task explicitly requires 60-second trials. The agent says fixed tiling preserves continuous alignment and equal trial lengths.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filtering. Every complete 60-second window is retained; only an incomplete tail is discarded. The helper raises an error if a session cannot supply even one complete trial.

ii.
```python
if usable <= 0:
    raise ValueError(f"Array with shape {arr.shape} does not contain a full trial.")
arr = arr[..., :usable]
```

iii. The notes report that every session has at least 20 valid full trials, so no additional trial curation was needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from Suite2p `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence) in each session's `suite2p/plane0` directory.

ii.
```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
fneu = np.load(session.suite2p_dir / "Fneu.npy").astype(np.float32)
```

iii. The paper says downstream analyses used baseline-corrected fluorescence. The agent therefore reconstructs that signal from the released raw and neuropil traces rather than using `spks.npy`.

## 2-b. How is the `neural` data processed?

i. It subtracts 0.7 times neuropil fluorescence, applies Suite2p `maximin` baseline preprocessing with a 60-second window and sigma 10 at 30 Hz, casts to float32, and averages non-overlapping groups of 10 frames.

ii.
```python
corrected = f - NEUROPIL_COEFF * fneu
corrected = suite2p_preprocess(
    corrected.copy(), baseline=BASELINE_MODE,
    win_baseline=WIN_BASELINE_SECONDS, sig_baseline=SIG_BASELINE_FRAMES,
    fs=FRAME_RATE_HZ, prctile_baseline=PRCTILE_BASELINE,
    batch_size=SUITE2P_BATCH_SIZE, device=torch.device("cpu"),
).astype(np.float32)
binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES).astype(np.float32)
```

iii. The agent cites the paper's baseline-corrected fluorescence/dF/F choice and its statement that decoding traces were denoised by averaging 10 consecutive timestamps. The 0.7 coefficient and baseline parameters follow Suite2p defaults/reference processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional filtering is performed in the converter; all rows in the released `F.npy` are retained.

ii.
```python
"brain_region_idx": np.zeros(session.nneurons, dtype=np.int64)
```

iii. The notes explain that the released files already contain Suite2p-classified cells tracked across all days; inspected `iscell` values already passed the >0.5 criterion. Reapplying Track2p/`iscell` filtering would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event. Each trial is aligned to the start of its fixed 60-second window, with neural, input, and output sliced using identical boundaries. Metadata records trial start with offsets 0 to 60 seconds.

ii.
```python
neural_trials = split_time_series_into_trials(binned_neural, BINS_PER_TRIAL)
input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
output_trials = split_time_series_into_trials(motion_bins[np.newaxis, :], BINS_PER_TRIAL)
```
```python
"temporal_alignment_event": "trial start of fixed 60-second windows tiled across each session",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The agent notes that recordings are continuous and event-free; trial-start alignment is consequently the meaningful alignment event for the artificial windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 30 Hz samples are averaged in non-overlapping groups of 10, yielding 3 Hz data and a 333.333 ms bin size. Neural activity, motion energy, and frame time are all rebinned this way before trialization.

ii.
```python
usable = arr.shape[-1] - (arr.shape[-1] % factor)
trimmed = arr[..., :usable]
return trimmed.reshape(new_shape).mean(axis=-1)
```
```python
"time_bin_size": float(1000.0 * BIN_FRAMES / FRAME_RATE_HZ)
```

iii. This directly follows the paper's decoding denoising step of averaging 10 consecutive timestamps and keeps all streams on one timebase.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the imaging frame index and the fixed 30 Hz frame rate, rather than read from a raw timestamp variable.

ii.
```python
frame_times = np.arange(session.nframes, dtype=np.float32) / np.float32(FRAME_RATE_HZ)
```

iii. The agent says the imaging clock is the master timebase and the task explicitly asks for elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Per-frame elapsed seconds are averaged in the same non-overlapping 10-frame bins as the signals. Thus each saved time is the mean/center time of its bin (first value 0.15 s), and time continues across trials rather than resetting.

ii.
```python
binned_time = non_overlapping_mean_last_axis(frame_times, BIN_FRAMES).astype(np.float32)
input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
```

iii. The notes explicitly choose bin-center times and absolute elapsed session time because that is the natural timestamp of an average and the requested variable is time from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural activity start on the same imaging frame grid, undergo the same 10-frame grouping, and are split with the same 180-bin trial boundaries.

ii.
```python
binned_neural = non_overlapping_mean_last_axis(corrected, BIN_FRAMES)
binned_time = non_overlapping_mean_last_axis(frame_times, BIN_FRAMES)
neural_trials = split_time_series_into_trials(binned_neural, BINS_PER_TRIAL)
input_trials = split_time_series_into_trials(binned_time[np.newaxis, :], BINS_PER_TRIAL)
```

iii. The agent validated the full time vector against an independently recomputed 30 Hz frame grid and reported an exact tolerance-based match.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output derives from `move_deve/motion_energy_glob.npy`; `tstamps.npy` is used when behavior samples are missing so motion can be mapped to the imaging grid.

ii.
```python
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
tstamps = np.load(session.move_dir / "tstamps.npy")
```

iii. The notes identify global motion energy as the paper's precomputed behavior trace and timestamps as the synchronization information needed for occasional dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If lengths already match, motion is used directly. Otherwise timestamps are converted to imaging-frame indices, duplicate samples are averaged, absent imaging frames are linearly interpolated, the aligned trace is averaged in 10-frame bins, and the result is categorized per session.

ii.
```python
frame_idx = np.rint((tstamps - tstamps[0]) / nominal_step).astype(np.int64)
np.add.at(sums, frame_idx, motion)
np.add.at(counts, frame_idx, 1)
aligned[valid] = sums[valid] / counts[valid]
aligned[missing] = np.interp(missing_idx, valid_idx, aligned[valid])
```
```python
binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES).astype(np.float32)
```

iii. The agent chose imaging frames as the master clock. It says sparse missing behavior must be repaired to provide complete categorical targets and that the data README permits interpolation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20/40/60/80% thresholds are computed separately for each session's complete binned motion trace. `np.digitize` produces classes 0–4. If ties result in fewer than five classes, stable rank ordering assigns exactly equal-count fifths.

ii.
```python
quantiles = np.linspace(0.0, 1.0, nclasses + 1)[1:-1]
thresholds = np.quantile(values, quantiles)
categories = np.digitize(values, thresholds, right=False).astype(np.int64)
if np.unique(categories).size < nclasses:
    order = np.argsort(values, kind="mergesort")
    boundaries = np.linspace(0, values.shape[0], nclasses + 1, dtype=int)
    for cls in range(nclasses):
        categories[order[boundaries[cls]:boundaries[cls + 1]]] = cls
```

iii. Per-session quintiles are explicitly required and normalize session-to-session motion scale. The tie fallback is intended to guarantee five equal-percentile classes.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion is first reconstructed on the `nframes`-long imaging grid, then neural and motion traces are independently averaged over the same consecutive 10-frame groups and cut at identical trial boundaries.

ii.
```python
motion_aligned, missing_mask, raw_frame_idx, timestamp_scale = align_motion_to_imaging(
    motion=motion_raw, tstamps=tstamps, nframes=session.nframes, fs=FRAME_RATE_HZ,
)
binned_motion = non_overlapping_mean_last_axis(motion_aligned, BIN_FRAMES)
output_trials = split_time_series_into_trials(motion_bins[np.newaxis, :], BINS_PER_TRIAL)
```

iii. Hardware synchronization makes framewise alignment appropriate. The agent reports raw-data checks on both missing-frame and complete sessions and above-chance held-out decoding in every session.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped behavior samples are detected through timestamp-to-imaging-grid mapping and linearly interpolated from valid neighbors; equal-length streams bypass reconstruction. Duplicate mapped samples are averaged. Partial final bins and partial 60-second trials are trimmed, while invalid timestamp spans, wholly missing motion, or sessions without a complete trial raise errors.

ii.
```python
if motion.size == nframes:
    return motion.astype(np.float32, copy=False), np.zeros(nframes, dtype=bool), ...
if span <= 0:
    raise ValueError("Motion timestamps are not strictly increasing.")
if missing.any():
    aligned[missing] = np.interp(missing_idx, valid_idx, aligned[valid])
```

iii. The agent documents that missing camera frames are sparse, interpolation is allowed by the dataset README, and complete targets are needed by the decoder. It validated recovered missing counts of 0, 1, 2, 3, 116, and 148 frames.

## 6-a. What are the most time-consuming steps of the code?

i. Whole-session Suite2p neural baseline preprocessing is the principal compute-heavy transformation; loading large fluorescence arrays is the main I/O cost. Optional five-panel plotting can also add appreciable time. Per-stage timers record loading, neural preprocessing, alignment, binning, trial splitting, and plotting.

ii.
```python
stage_times["load_s"] = time.perf_counter() - t0
corrected = suite2p_preprocess(..., device=torch.device("cpu"))
stage_times["neural_preprocess_s"] = time.perf_counter() - t0
```

iii. The notes report roughly 0.34 s for 20-minute and 1.02 s for 30-minute sessions, estimating about 32 seconds for full conversion. They identify session-wide Suite2p operations and file loading as the substantive costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Core alignment and binning are already vectorized (`np.add.at`, reshape/mean, `np.interp`). Remaining Python loops include session iteration, creation of trial views, and the rare five-class rank fallback; trial construction could be expressed as a reshape plus list conversion, although session processing itself cannot simply be vectorized because dimensions differ.

ii.
```python
return [arr[:, i * bins_per_trial:(i + 1) * bins_per_trial] for i in range(ntrials)]
```
```python
for cls in range(nclasses):
    categories[order[boundaries[cls]:boundaries[cls + 1]]] = cls
```

iii. The agent emphasizes that reshape-based temporal averaging and timestamp alignment are vectorized and that session-wise processing bounds memory. It did not claim these small remaining loops were bottlenecks.

## 6-c. What processing does the code repeat multiple times?

i. Session discovery reads `ops.npy`, memory-mapped `F.npy`, and memory-mapped motion energy for shapes, after which processing loads `F.npy` and motion energy again. Neural, input, and output also call the same trial-splitting helper separately. These repetitions are small relative to full processing but contradict the notes' claim of no redundant loads.

ii.
```python
f = np.load(suite2p_dir / "F.npy", mmap_mode="r")
motion = np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r")
```
```python
f = np.load(session.suite2p_dir / "F.npy").astype(np.float32)
motion_raw = np.load(session.move_dir / "motion_energy_glob.npy")
```

iii. The notes say processing is session-wise with “no redundant loads,” apparently treating the discovery-time memory maps as lightweight metadata inspection. The code nevertheless opens the same files in both phases.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Discovery loads `ops.npy` largely to obtain `nframes` and opens arrays for shapes; most `ops` content is unused. The script computes timestamp-scale metadata even though the scale does not affect its relative-span frame mapping. With `--show-processing`, it z-scores a neuron subset and creates plots that are diagnostic only and are not stored in decoder data. It also computes extensive timing/session metadata that the decoder does not consume.

ii.
```python
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
scale_to_seconds = detect_timestamp_scale_to_seconds(tstamps, fs)
```
```python
im = axes[3].imshow(zscore_rows(binned_neural[:nneurons_show]), ...)
fig.savefig(f"/app/processing_{session.session_id}.png", dpi=150)
```

iii. The plotting is explicitly optional and justified as a visual sanity check. Timing and alignment diagnostics support validation and provenance, even though downstream training discards them; the notes describe these checks as evidence against misalignment.
