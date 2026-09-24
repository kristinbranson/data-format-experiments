# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory under `/app/data` as a subject and every directory below each subject as a session, both in sorted order. For each session it loads `ops.npy`, `F.npy`, and `Fneu.npy` from `suite2p/plane0`, plus `motion_energy_glob.npy` and, only when repair is needed, `tstamps.npy` from `move_deve`. It processes all discovered sessions and ultimately loads their trial slices into the output lists.

ii.
```python
def list_subjects(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())

def list_sessions(subject_dir: Path) -> list[Path]:
    return sorted(path for path in subject_dir.iterdir() if path.is_dir())

for subject_dir in list_subjects(data_root):
    for session_dir in list_sessions(subject_dir):
        neural_trace, fs = preprocess_neural(session_dir)
        motion_trace, n_motion_repaired = repair_motion_energy(session_dir, n_frames)
```

iii. The trajectory says the agent inspected the raw hierarchy and found per-session Suite2p and motion-energy streams, then chose chronological/sorted subject-session traversal. It intended to use the supplied tracked-cell outputs directly and reported that all 6 subjects and 41 sessions were converted.

## 1-b. How are the data split into subjects?

i. Each immediate directory beneath the data root is treated as one mouse. Alphabetically sorted directory names form `subjects`; a name-to-index mapping supplies each session's `subject_idx`.

ii.
```python
subjects = [subject_dir.name for subject_dir in list_subjects(data_root)]
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[subject_dir.name])
```

iii. The agent inferred from the directory structure that its six top-level subject directories are the mice. It used sorting to make their order stable.

## 1-c. How are the data split into sessions?

i. Every immediate subdirectory of a subject is one session, sorted by its directory name (and therefore date in this dataset). Each becomes one element of the session-level `neural`, `input`, and `output` lists.

ii.
```python
def list_sessions(subject_dir: Path) -> list[Path]:
    return sorted(path for path in subject_dir.iterdir() if path.is_dir())

for session_dir in list_sessions(subject_dir):
    ...
    neural_sessions.append(neural_trials)
```

iii. The trajectory describes sessions as daily recordings and explicitly plans to load sessions in chronological order. The final response confirms ordering by subject and then date.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second trials after 10-frame temporal averaging. The number of bins per trial is calculated from the session sampling rate; any final incomplete trial is removed.

ii.
```python
bins_per_trial = int(round(TRIAL_DURATION_SEC / (BIN_SIZE_FRAMES / fs)))
n_trials = total_bins // bins_per_trial
usable = n_trials * bins_per_trial
neural_binned = neural_binned[:, :usable]
...
for trial_idx in range(n_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
```

iii. The decoder instructions explicitly require 60-second trials. Because the recordings are continuous rather than naturally trialized, the agent chose fixed contiguous chunks and validated that all trials contained 180 bins.

## 1-e. How are trials filtered based on quality controls?

i. No complete 60-second trial is quality-filtered. Only the trailing bins that cannot form a complete trial are discarded; no session-level minimum-trial or signal-quality exclusion is added.

ii.
```python
n_trials = total_bins // bins_per_trial
usable = n_trials * bins_per_trial
neural_binned = neural_binned[:, :usable]
time_binned_sec = time_binned_sec[:usable]
motion_classes = motion_classes[:usable]
```

iii. The trajectory did not identify a trial-quality criterion in the source material. It therefore retained all full-length artificial trials and relied on exact length/format validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence), with preprocessing parameters and sampling rate read from `ops.npy`.

ii.
```python
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The agent inspected the repository/notebooks and concluded that `F.npy` was raw fluorescence rather than already corrected, so both fluorescence arrays and the saved Suite2p settings were required.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil using the saved `neucoeff` (default 0.7), then calls Suite2p `dcnv.preprocess` for maximin baseline correction using parameters from `ops.npy`. The resulting trace is cast to float32 and averaged over non-overlapping groups of 10 frames.

ii.
```python
neuropil_corrected = F - float(ops.get("neucoeff", 0.7)) * Fneu
baseline_corrected = dcnv.preprocess(
    neuropil_corrected.copy(),
    baseline=ops.get("baseline", "maximin"),
    win_baseline=float(ops.get("win_baseline", 60.0)),
    sig_baseline=float(ops.get("sig_baseline", 10.0)),
    fs=fs,
    prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
    batch_size=64,
    device=torch.device("cpu"),
)
neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
```

iii. The trajectory says the paper and notebooks called for Suite2p-style baseline-corrected fluorescence and 10-timestamp denoising. The agent used the installed Suite2p implementation and saved session parameters to avoid an approximate reimplementation or unsupported hard-coding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit `iscell.npy` or other neuron-quality mask. Every row present in the supplied `F.npy`/`Fneu.npy` tracked-cell arrays is retained.

ii.
```python
F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
...
brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. Early in the trajectory the agent mentioned the source's `iscell > 0.5` convention, but its eventual plan was to use the provided tracked-cell plane outputs directly. It evidently treated those arrays as already curated and applied no additional mask.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Sessions are treated as continuous recordings aligned to session start, and trial 0 starts at session start; later trials are consecutive 60-second slices. Metadata describes the alignment as `session start` with trial-relative offsets 0 to 60 seconds.

ii.
```python
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
...
"temporal_alignment_event": "session start",
"off_start": 0.0,
"off_end": float(TRIAL_DURATION_SEC),
```

iii. The agent found no stimulus/event-defined trials, so it used the natural origin of the continuous recording and maintained common slice indices across streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Non-overlapping blocks of 10 raw frames are averaged. With the saved 30 Hz rate this produces 333.333 ms bins (3 Hz). The agent verifies that all sessions share a sampling rate and derives metadata from that rate.

ii.
```python
BIN_SIZE_FRAMES = 10
return array.reshape(n_rows, -1, bin_size).mean(axis=2, dtype=np.float32)
...
time_bin_size_ms = 1000.0 * BIN_SIZE_FRAMES / float(common_fs)
```

iii. The trajectory cites the paper's decoding procedure: neural and behavior traces are slightly denoised by averaging 10 consecutive timestamps. Applying the same bin boundaries to both preserves alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from raw frame indices and the Suite2p sampling rate in `ops.npy`; it is not read from a standalone experimental-time variable.

ii.
```python
fs = float(ops["fs"])
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
```

iii. The agent observed a consistent 30 Hz acquisition rate and used the authoritative session metadata so elapsed seconds can be reconstructed exactly from frame number.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Raw zero-based frame indices are divided by `fs`, then groups of 10 timestamps are averaged. Thus each converted time value is the center/mean time of its corresponding 10-frame bin (e.g. the first is 0.15 s at 30 Hz), and time continues across trials within a session.

ii.
```python
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
```

iii. The trajectory does not separately justify averaging timestamps, but the implementation applies the same generic 10-frame mean-binning step used for neural and behavioral traces. This selects bin centers rather than the reference's bin-left times.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time uses the same raw frame count, 10-frame grouping, truncation, and trial `[start:end]` indices as neural data, so every time value denotes the mean timestamp of the neural samples averaged into that bin.

ii.
```python
neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
...
neural_trials.append(neural_binned[:, start:end])
input_trials.append(time_binned_sec[start:end][None, :])
```

iii. The agent's stated aim was to bin streams together and slice them identically to preserve temporal alignment. No independent clock correction is needed for a time vector constructed from neural frame indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is based on the precomputed global behavioral-video motion signal `move_deve/motion_energy_glob.npy`. For shortened streams, `move_deve/tstamps.npy` supplies camera timestamp gaps used to locate missing frames.

ii.
```python
motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
...
tstamps = np.load(move_dir / "tstamps.npy")
diffs = np.diff(tstamps)
```

iii. The agent inspected timestamp behavior and found that gaps exactly explained shortened motion streams. It therefore selected timestamps as the basis for dropped-frame reconstruction, while avoiding repair when motion already matched imaging length.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If motion is shorter than neural data, dropped positions are reconstructed and linearly interpolated. The repaired continuous motion trace is float32-averaged in non-overlapping 10-frame bins. Session-specific 20th, 40th, 60th, and 80th percentiles are then computed and used to assign integer classes 0–4.

ii.
```python
motion_trace, n_motion_repaired = repair_motion_energy(session_dir, n_frames)
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
motion_classes = session_motion_bins(motion_binned)
```

iii. The trajectory says missing camera frames must be repaired before common binning, and cites the paper's 10-frame denoising. It discretizes only after averaging because averaging already-discrete class labels would be invalid.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four thresholds are calculated independently for every session at quantiles 0.2, 0.4, 0.6, and 0.8 of the binned motion trace. `np.digitize(..., right=False)` maps values into five zero-based categories; values equal to a boundary go into the higher category.

ii.
```python
def session_motion_bins(motion_binned: np.ndarray) -> np.ndarray:
    thresholds = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(motion_binned, thresholds, right=False).astype(np.int64)
```

iii. The task demands five equal-percentile bins selected per session. The agent explicitly planned session-wise equal-frequency binning after temporal averaging and validated near-balanced classes.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If lengths differ, the agent estimates nominal camera interval from the median timestamp difference, calculates the number of missing samples in every timestamp gap, inserts NaNs at those positions, and linearly interpolates them. It requires the reconstructed motion length to exactly equal neural frame count. Neural and motion are then binned and trial-sliced with matching boundaries.

ii.
```python
missing_counts = np.maximum(np.round(diffs / frame_dt).astype(int) - 1, 0)
expected_len = int(len(motion) + missing_counts.sum())
if expected_len != n_frames:
    raise ValueError(...)
repaired = np.full(n_frames, np.nan, dtype=np.float32)
positions = np.arange(len(motion), dtype=np.int64)
positions[1:] += np.cumsum(missing_counts)
repaired[positions] = motion
repaired[missing_mask] = np.interp(...)
```

iii. The agent empirically checked the files and found timestamp gaps accounted for missing frames. It deliberately repairs only shortened streams because already equal-length streams can contain unusual timestamp jumps that should not change alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion streams are repaired using timestamp-derived missing positions and linear interpolation. Mismatched timestamp length or a reconstructed length unequal to neural length raises `ValueError` rather than silently misaligning data. Partial 10-frame bins and partial 60-second trials at session ends are truncated. Cross-session sampling-rate disagreement also raises an error.

ii.
```python
if len(tstamps) != len(motion):
    raise ValueError(...)
if expected_len != n_frames:
    raise ValueError(...)
...
usable = (len(array) // bin_size) * bin_size
...
elif not np.isclose(common_fs, fs):
    raise ValueError(...)
```

iii. The trajectory says timestamp gaps exactly accounted for missing camera frames in shortened streams and that repair should not be applied otherwise. Strict validation was chosen to expose unexplained inconsistencies; nine sessions were reported as repaired.

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p baseline preprocessing over all neurons and full session lengths is the dominant computation. Loading full `.npy` arrays and serializing the large converted pickle are secondary costs.

ii.
```python
baseline_corrected = dcnv.preprocess(
    neuropil_corrected.copy(),
    ...
    device=torch.device("cpu"),
)
```

iii. The trajectory explicitly calls per-session Suite2p baseline correction the heaviest step. The agent benchmarked a representative session before running the full conversion and then waited for the full preprocessing pass.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-construction loop could be replaced by reshaping/transposing each full-session array into a trial axis, although creating the required list of arrays would still require some iteration. Subject/session traversal is inherently file-oriented. Missing-frame reconstruction and interpolation are already vectorized.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
    input_trials.append(time_binned_sec[start:end][None, :].astype(np.float32, copy=False))
    output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. The trajectory contains no explicit discussion of vectorizing the trial loop. Its timestamp repair is intentionally vectorized through cumulative sums, masks, and `np.interp`, avoiding the reference's repeated `np.insert` allocations.

## 6-c. What processing does the code repeat multiple times?

i. It calls `list_subjects(data_root)` twice, performs similar trim/reshape/mean logic separately in `mean_bin_1d` and `mean_bin_2d`, and repeats per-session loading, baseline correction, binning, and trial construction as necessary for each session. The three trial streams are also sliced independently in the trial loop.

ii.
```python
subjects = [subject_dir.name for subject_dir in list_subjects(data_root)]
...
for subject_dir in list_subjects(data_root):
...
neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
```

iii. The agent did not identify repeated processing as a concern in the trajectory. Most repetition is per-session work required because each session has distinct arrays and motion thresholds; the duplicated directory scan and helper logic are minor.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It creates detailed `session_info` metadata and counts repaired motion samples, which the decoder does not use for training. It also constructs and mean-bins a full-session time vector solely to produce one simple input row. The `neuropil_corrected.copy()` may be an avoidable extra allocation, though it protects the input to Suite2p. Final incomplete bin/trial tails are computed or loaded but ultimately discarded.

ii.
```python
time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
...
session_info.append({
    "n_motion_frames_repaired": int(n_motion_repaired),
    "n_time_bins": int(neural_binned.shape[1]),
    ...
})
```

iii. The trajectory says the agent performed a final metadata inspection specifically to report which sessions were repaired, so this extra bookkeeping supported verification and provenance even though it is not consumed by downstream decoding. No other knowingly discarded processing was discussed.
