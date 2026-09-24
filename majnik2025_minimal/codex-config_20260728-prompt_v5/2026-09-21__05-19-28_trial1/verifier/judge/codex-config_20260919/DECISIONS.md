# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent walks every directory directly under the configured data root as a subject, then every date-like child directory (first four characters numeric) as a session. For every session it loads Suite2p `ops.npy`, `iscell.npy`, `F.npy`, and `Fneu.npy`, plus `move_deve/motion_energy_glob.npy` and, when lengths differ, `interframe_int.npy`. It processes one session at a time and appends all complete 60-second trials to the dataset.

ii.
```python
def sorted_subjects(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir())

def sorted_sessions(subject_dir: Path) -> list[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )

for subject_dir in sorted_subjects(data_root):
    for session_dir in sorted_sessions(subject_dir):
        neural_trials, input_trials, output_trials, _ = session_to_trials(session_dir)
```

iii. The trajectory says the agent intended to load “all 6 subjects and all longitudinal sessions in date order.” It inspected the directory layout and concluded that the supplied arrays already contain the Track2p-aligned neurons. Sorting was used for deterministic chronological organization.

## 1-b. How are the data split into subjects?

i. Each immediate subdirectory of `/app/data` is treated as a subject. Subject names are sorted, and a lookup maps each name to the integer stored once per session in `subject_idx`.

ii.
```python
subjects = [subject_dir.name for subject_dir in sorted_subjects(data_root)]
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[subject_dir.name])
```

iii. The agent reported finding six subjects and treated the top-level data directories as the mouse boundary. It did not state a reason for omitting the reference's `jm*` name guard, presumably because inspection showed only subject directories at that level.

## 1-c. How are the data split into sessions?

i. A session is a sorted subject child directory whose name begins with four digits. Each such daily recording becomes one top-level session in `neural`, `input`, and `output`.

ii.
```python
def sorted_sessions(subject_dir: Path) -> list[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )
```

iii. The trajectory explicitly describes loading longitudinal sessions “in date order.” The numeric-prefix check reflects the observed date-based session naming and avoids unrelated child directories.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second trials after 10-frame averaging. At 30 Hz this is 180 binned samples per trial. Only complete trials are retained, and the session must have at least two.

ii.
```python
trial_bins = int(TRIAL_SECONDS * FRAME_RATE_HZ / DENOISE_BIN_FRAMES)
n_complete_trials = neural_binned.shape[1] // trial_bins
if n_complete_trials < 2:
    raise ValueError(f"{session_dir}: fewer than two complete 60-second trials")
...
for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
```

iii. The instructions explicitly require splitting sessions into 60-second trials and at least two trials per session. The agent described these as “consecutive 60-second trials” and verified that the resulting sessions had the expected 20- or 30-trial splits.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level behavioral or signal-quality rejection. Incomplete trailing windows are discarded, and an entire session raises an error if fewer than two complete trials exist.

ii.
```python
n_complete_trials = neural_binned.shape[1] // trial_bins
if n_complete_trials < 2:
    raise ValueError(f"{session_dir}: fewer than two complete 60-second trials")
usable = n_complete_trials * trial_bins
neural_binned = neural_binned[:, :usable]
```

iii. No natural trial QC was identified. The two-trial check directly enforces the decoder-format requirement; truncation is necessary to make every artificial trial exactly 60 seconds.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p plane-0 raw ROI fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`; preprocessing parameters come from `ops.npy`, and `iscell.npy` is used as a validation check.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy")
f = np.load(plane_dir / "F.npy").astype(np.float32)
fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)
```

iii. The agent determined from the loading notebook that `F.npy` is raw fluorescence and that dF/F-like baseline-corrected activity had to be reconstructed according to the paper. It used the already Track2p-aligned ROI arrays rather than rebuilding cell identities from CSV files.

## 2-b. How is the `neural` data processed?

i. The code subtracts neuropil using `ops['neucoeff']` (default 0.7), applies Suite2p `preprocess` with session `ops` settings and maximin defaults, casts to float32, then averages non-overlapping groups of 10 frames.

ii.
```python
fc = f - float(ops.get("neucoeff", 0.7)) * fneu
dff = preprocess(
    fc.copy(), baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=float(ops.get("fs", FRAME_RATE_HZ)),
    prctile_baseline=float(ops.get("prctile_baseline", 8)),
    batch_size=100, device=torch.device("cpu"),
)
neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
```

iii. The agent found that Track2p's GUI helper hardcoded zero neuropil subtraction but judged that inconsistent with the paper's “default Suite2p parameters.” It therefore chose official Suite2p preprocessing: neuropil subtraction plus maximin baseline correction, followed by the paper's stated 10-timestamp averaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not select a subset of ROI rows. Instead, it requires every supplied tracked ROI to have `iscell[:, 1] > 0.5`; if any fails, conversion aborts. Thus all rows are retained when the dataset satisfies that invariant.

ii.
```python
iscell = np.load(plane_dir / "iscell.npy")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found tracked ROIs below the 0.5 iscell threshold")
```

iii. The trajectory says the methods require Suite2p `iscell > 0.5`. Inspection evidently showed that the Track2p-aligned arrays already consist of qualifying ROIs, so the agent encoded the criterion as an integrity assertion rather than row filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external event. Each neural trial is a slice beginning at the start of its consecutive artificial 60-second window. Metadata names that window start as the alignment event and assigns offsets 0 to 60 seconds.

ii.
```python
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
...
"temporal_alignment_event": "start of each consecutive 60-second trial",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The agent treated continuous spontaneous recording as lacking a stimulus event. Its plan was to average first and then trialize, keeping all streams on identical slice boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The original 30 Hz data are averaged in non-overlapping 10-frame bins, producing 3 Hz samples and a nominal bin size of 333.333 ms. Short tails that cannot form a full 10-frame bin are removed.

ii.
```python
DENOISE_BIN_FRAMES = 10
return x.reshape(x.shape[0], -1, bin_size).mean(axis=2)
...
"time_bin_size": 1000.0 * DENOISE_BIN_FRAMES / FRAME_RATE_HZ,
```

iii. The agent quoted the methods' instruction to denoise both dF/F and behavior by averaging 10 consecutive timestamps, and deliberately applies this before 60-second slicing and categorical motion binning.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthesized from the neural raw-frame index and the assumed 30 Hz frame rate, not loaded from a timestamp file.

ii.
```python
frame_times_s = np.arange(raw_frames, dtype=np.float32) / FRAME_RATE_HZ
```

iii. The agent investigated whether time should be “raw seconds, bin centers, or another aligned representation” and chose elapsed seconds derived from frame indices, consistent with a fixed-rate synchronous recording.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Raw frame times are divided by 30 to obtain seconds and then averaged in the same non-overlapping 10-frame bins as neural activity. Consequently the first value is the mean time of frames 0–9 (0.15 s), rather than the left edge at 0 s. The values remain continuous across successive trials within a session.

ii.
```python
frame_times_s = np.arange(raw_frames, dtype=np.float32) / FRAME_RATE_HZ
time_binned = mean_bin_1d(frame_times_s, DENOISE_BIN_FRAMES).astype(np.float32)
```

iii. The agent chose a binned elapsed-time representation so each input sample represents the same set of raw frames as its binned neural and motion samples. Its metadata explicitly describes “elapsed seconds from session start at each 10-frame averaged bin.”

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time, neural data, and motion are independently averaged over identical consecutive 10-frame groups, truncated to the same complete-trial length, and sliced with the same `start:end` trial indices.

ii.
```python
neural_binned = neural_binned[:, :usable]
time_binned = time_binned[:usable]
...
neural_trials.append(neural_binned[:, start:end])
input_trials.append(time_binned[np.newaxis, start:end])
```

iii. The agent selected bin-averaged timestamps to represent bin centers and maintain sample-wise correspondence with averaged neural data. It ran the supplied verifier to confirm dimensions, though that check does not establish agreement with the reference timestamp convention.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from `move_deve/motion_energy_glob.npy`. When frames are missing, `move_deve/interframe_int.npy` determines the positions of observed samples on the neural frame grid.

ii.
```python
motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy").astype(np.float32)
...
interframe = np.load(session_dir / "move_deve" / "interframe_int.npy")
```

iii. The agent found genuine video-frame drops mentioned by the data README. It also established that the decoder rejects NaNs, so it used interframe timing to reconstruct a complete motion trace.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If necessary, motion samples are mapped to full-rate frame positions using rounded interframe intervals relative to their median, missing positions are linearly interpolated, the full trace is averaged over 10-frame windows, and those averages are discretized into five per-session percentile classes.

ii.
```python
frame_steps = np.rint(interframe / median_dt).astype(np.int64)
known_positions = np.concatenate([[0], np.cumsum(frame_steps)])
full_motion[known_positions] = motion
motion = np.interp(np.arange(target_len), valid, full_motion[valid]).astype(np.float32)
motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES).astype(np.float32)
motion_bins = discretize_equal_percentile(motion_binned, MOTION_NBINS)
```

iii. The agent wanted to restore synchronization without leaving NaNs or applying undocumented smoothing. It therefore reconstructs only missing camera frames, then applies the same 10-frame denoising mandated for behavioral traces.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four within-session quantiles at 20%, 40%, 60%, and 80% define five integer categories 0–4. With distinct edges, `np.digitize(..., right=False)` is used. If quantile edges coincide, stable rank sorting and `array_split` force five approximately equal-frequency classes.

ii.
```python
edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
if np.unique(edges).shape[0] == edges.shape[0]:
    return np.digitize(values, edges, right=False).astype(np.int64)
order = np.argsort(values, kind="mergesort")
for bin_idx, idx in enumerate(np.array_split(order, nbins)):
    bins[idx] = bin_idx
```

iii. The instruction requires five equal-percentile bins selected per session. The trajectory says the agent checked session-wise quantiles and class counts before implementation. The fallback was added to guarantee equal-frequency classes when repeated values make percentile thresholds non-unique.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If lengths already match, motion is assumed frame-aligned. Otherwise, interframe intervals reconstruct the integer frame positions, strict checks require the reconstruction to span exactly the neural length, and linear interpolation fills gaps. Neural and motion are then averaged over the same 10-frame windows and sliced with identical trial indices.

ii.
```python
if int(known_positions[-1]) != target_len - 1:
    raise ValueError(...)
...
neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES)
...
output_trials.append(motion_bins[np.newaxis, start:end])
```

iii. The agent explicitly monitored for reconstructed behavior/neural length mismatches during the full conversion. It preferred using relative interframe timing over guessing drop locations and verified all 41 sessions converted without alignment errors.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are linearly interpolated after reconstructing their positions. Unexpected interframe lengths, nonpositive median intervals, mapping failures, span mismatches, low `iscell` probabilities, empty binning inputs, and too-short sessions cause explicit errors. Incomplete 10-frame bins and incomplete final 60-second trials are truncated.

ii.
```python
if interframe.shape[0] != motion.shape[0] - 1:
    raise ValueError(...)
if median_dt <= 0:
    raise ValueError(...)
...
return np.interp(...).astype(np.float32)
...
usable = n_complete_trials * trial_bins
neural_binned = neural_binned[:, :usable]
```

iii. The agent found actual camera drops and noted that exported NaNs would be rejected by the decoder. Strict assertions were intended to prevent silent misalignment, while truncation preserves uniform bins and trial sizes.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant computation is Suite2p maximin baseline preprocessing across every neuron and frame, performed session by session on CPU. Loading and converting the large fluorescence arrays and serializing the full dataset are secondary I/O/memory costs.

ii.
```python
dff = preprocess(
    fc.copy(), ...,
    batch_size=100,
    device=torch.device("cpu"),
)
```

iii. The trajectory spent most end-to-end conversion time waiting for session preprocessing, especially longer 30-minute sessions. The agent chose the official Suite2p routine for fidelity and did not claim performance optimization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-assembly loop could be replaced by reshape/split operations, and the five-bin tie fallback loop could assign the rank partitions vectorially. Directory/session loops are appropriate because sessions have differing neuron counts and are independently loaded.

ii.
```python
for trial_idx in range(n_complete_trials):
    start = trial_idx * trial_bins
    end = start + trial_bins
    neural_trials.append(neural_binned[:, start:end])
...
for bin_idx, idx in enumerate(np.array_split(order, nbins)):
    bins[idx] = bin_idx
```

iii. The trajectory does not discuss vectorization. The agent already vectorized expensive frame binning and missing-frame interpolation, leaving only relatively small organizational loops.

## 6-c. What processing does the code repeat multiple times?

i. `sorted_subjects(data_root)` scans and sorts the same root twice: once to build subject metadata and again to convert sessions. Every session also separately applies analogous 10-frame reshape-and-mean operations to neural, motion, and time arrays.

ii.
```python
subjects = [subject_dir.name for subject_dir in sorted_subjects(data_root)]
...
for subject_dir in sorted_subjects(data_root):
```

iii. The trajectory provides no explicit justification. The repeated root scan is negligible; separate binning helpers keep dimensional behavior clear for 1-D versus 2-D arrays.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `session_to_trials` returns the full truncated `motion_bins`, but `build_dataset` discards it as `_`. It also builds `session_ids` and `trial_counts` metadata that the decoder does not require. `fc.copy()` introduces an extra full fluorescence copy, and motion percentile labels are computed on the binned session tail before that tail is removed for complete trials.

ii.
```python
return neural_trials, input_trials, output_trials, motion_bins
...
neural_trials, input_trials, output_trials, _ = session_to_trials(session_dir)
...
session_ids.append(...)
trial_counts.append(...)
```

iii. The trajectory does not discuss these costs. The metadata aids provenance and verification, while the discarded return and pre-truncation classification appear to be implementation conveniences rather than downstream requirements.
