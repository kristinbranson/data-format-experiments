# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all subject directories under `/app/data` whose names start with `jm`, then enumerates session subdirectories whose names begin with four digits. For each session it loads imaging metadata from `ops.npy`, neural traces from `F.npy` and `Fneu.npy`, and behavior from `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`. Trials are not loaded directly from disk because the source data are continuous; trials are created later by splitting processed session-long arrays.

ii. 
```python
def list_sessions() -> list[SessionPath]:
    sessions: list[SessionPath] = []
    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("jm"):
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name[:4].isdigit():
                continue
            sessions.append(
                SessionPath(
                    subject=subject_dir.name,
                    session=session_dir.name,
                    path=session_dir,
                )
            )
    return sessions

def load_neural_arrays(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    f = np.load(session_path / "suite2p" / "plane0" / "F.npy", allow_pickle=True).astype(np.float32)
    fneu = np.load(session_path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True).astype(np.float32)
    return f, fneu

def load_behavior_arrays(session_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion = np.load(session_path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session_path / "move_deve" / "tstamps.npy", allow_pickle=True)
    interframe = np.load(session_path / "move_deve" / "interframe_int.npy", allow_pickle=True)
    return motion.astype(np.float64), tstamps.astype(np.float64), interframe.astype(np.float64)
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI says the released dataset is already organized as matched Suite2p session folders plus `move_deve` behavior files, so the conversion should treat each session folder as one recording and load the imaging and behavior arrays directly from that structure.

## 1-b. How are the data split into subjects?

i. Subjects are identified by top-level directory names starting with `jm`, sorted lexicographically, and later stored in `subjects` with a per-session integer `subject_idx`.

ii.
```python
for subject_dir in sorted(DATA_ROOT.iterdir()):
    if not subject_dir.is_dir() or not subject_dir.name.startswith("jm"):
        continue
```

```python
subjects = sorted({session.subject for session, _ in processed_sessions})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.array([subject_lookup[s.subject] for s, _ in processed_sessions], dtype=np.int64),
```

iii. The AI’s notes say the data README and folder layout consistently use `jm*` folders for mice, so subject identity should come directly from those folder names.

## 1-c. How are the data split into sessions?

i. Each session is one dated subdirectory inside a subject directory. The AI sorts them and only keeps directory names starting with four digits, which matches the `YYYY-MM-DD_a` naming scheme in the release.

ii.
```python
for session_dir in sorted(subject_dir.iterdir()):
    if not session_dir.is_dir() or not session_dir.name[:4].isdigit():
        continue
    sessions.append(
        SessionPath(
            subject=subject_dir.name,
            session=session_dir.name,
            path=session_dir,
        )
    )
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the AI documents that subject folders contain daily session folders named like `YYYY-MM-DD_a`, and in Step 5 it states that session order should be sorted by subject then date for reproducibility.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions with no native trial structure. After neural, motion, and time streams are processed and binned, it splits each session into non-overlapping 60-second windows. Unlike the human reference, it does not explicitly discard a partial final trial; instead, it assumes the binned session length is exactly divisible by the 60-second trial length and raises an error otherwise.

ii.
```python
bins_per_trial = int(round(trial_seconds * fs / bin_frames))
neural_trials = split_trials_2d(neural_binned, bins_per_trial)
input_trials = split_trials_1d(time_binned, bins_per_trial)
output_trials = split_trials_1d(motion_labels.astype(np.int64), bins_per_trial)
```

```python
def split_trials_2d(arr: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    n_rows, n_time = arr.shape
    if n_time % trial_bins != 0:
        raise ValueError(f"Time dimension {n_time} is not divisible by trial_bins {trial_bins}.")
    n_trials = n_time // trial_bins
    reshaped = arr.reshape(n_rows, n_trials, trial_bins)
    return [reshaped[:, i, :].astype(arr.dtype, copy=False) for i in range(n_trials)]
```

iii. In Step 2 and Step 5 of `CONVERSION_NOTES.md`, the AI states there is no explicit trial structure in the source recordings, so 60-second non-overlapping windows are an imposed downstream format required by the benchmark.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any trial-level quality filtering. Every 60-second window is kept if the processed session arrays can be evenly partitioned into trials.

ii.
```python
neural_trials = split_trials_2d(neural_binned, bins_per_trial)
input_trials = split_trials_1d(time_binned, bins_per_trial)
output_trials = split_trials_1d(motion_labels.astype(np.int64), bins_per_trial)

if not (len(neural_trials) == len(input_trials) == len(output_trials)):
    raise RuntimeError(f"Trial count mismatch for {session.session_id}.")
```

iii. The AI’s Step 5 notes say it planned to “keep all 60-second windows that have valid neural data” and expected no partial trailing windows in the released sessions after 10-frame binning.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data comes from Suite2p fluorescence traces in `F.npy` and `Fneu.npy`. The AI also reads `ops.npy` to obtain the preprocessing parameters used for neuropil subtraction and baseline correction.

ii.
```python
def load_neural_arrays(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    f = np.load(session_path / "suite2p" / "plane0" / "F.npy", allow_pickle=True).astype(np.float32)
    fneu = np.load(session_path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True).astype(np.float32)
    return f, fneu

def load_ops(session_path: Path) -> dict:
    return np.load(session_path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()
```

iii. In Step 5, the AI explicitly maps `F.npy + Fneu.npy + ops.npy` to `neural`, arguing that the paper’s analyses used baseline-corrected fluorescence rather than raw `F` or `spks`.

## 2-b. How is the `neural` data processed?

i. The AI reconstructs a Suite2p-style baseline-corrected fluorescence trace by subtracting neuropil using the session’s `ops["neucoeff"]`, then calling `suite2p.extraction.dcnv.preprocess` with session-specific preprocessing parameters read from `ops.npy`. It then averages the corrected traces in non-overlapping 10-frame bins.

ii.
```python
def preprocess_neural(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> tuple[np.ndarray, dict]:
    fs = float(ops["fs"])
    fc = F - np.float32(ops["neucoeff"]) * Fneu
    corrected = dcnv.preprocess(
        fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=fs,
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 2000)),
        device=torch.device("cpu"),
    )
    return corrected.astype(np.float32), info
```

```python
neural_binned = mean_bin_2d(corrected_F, bin_frames).astype(np.float32)
```

iii. The justification appears in Step 4 and Step 5 of `CONVERSION_NOTES.md`: the AI concluded the released data are already Track2p-matched, but the paper’s downstream decoding used baseline-corrected fluorescence-like traces, so it should reconstruct that representation from `F`, `Fneu`, and Suite2p parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra neuron filtering is applied during conversion. The AI assumes the released arrays already contain the across-day tracked cells and already satisfy the paper’s `iscell > 0.5` curation.

ii.
```python
def load_neural_arrays(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    f = np.load(session_path / "suite2p" / "plane0" / "F.npy", allow_pickle=True).astype(np.float32)
    fneu = np.load(session_path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True).astype(np.float32)
    return f, fneu
```

```python
"brain_region_idx": get_brain_region_idx(raw_F.shape[0]),
```

iii. Step 2 and Step 4 of `CONVERSION_NOTES.md` say the provided `suite2p` folders are already Track2p matched-cell outputs with constant neuron counts per subject, and that the released data already embody the paper’s `iscell` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align neural data to a biological or task event. It simply cuts the continuous session into consecutive 60-second windows. However, its documentation is internally inconsistent: the notes describe absolute session-time processing, while the saved metadata declares the temporal alignment event to be the “start of each derived 60-second trial window.”

ii.
```python
neural_trials = split_trials_2d(neural_binned, bins_per_trial)
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each derived 60-second trial window",
    "off_start": 0.0,
    "off_end": float(DEFAULT_TRIAL_SECONDS),
    ...
}
```

iii. In Step 5 the AI justified trialing as an imposed downstream format layered on top of a continuous recording. In Step 9 and Step 10 it also emphasized preserving absolute time-from-session-start, which conflicts with the later metadata wording that suggests trial-local alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all time-varying streams into non-overlapping 10-frame averages at 30 Hz, giving a time bin size of `10 / 30 s = 333.33 ms`.

ii.
```python
DEFAULT_BIN_FRAMES = 10
DEFAULT_FS = 30.0
...
neural_binned = mean_bin_2d(corrected_F, bin_frames).astype(np.float32)
motion_binned = mean_bin_1d(aligned_motion.astype(np.float32), bin_frames).astype(np.float32)
time_binned = build_time_input(n_frames, fs, bin_frames)
```

```python
"time_bin_size": 1000.0 * DEFAULT_BIN_FRAMES / DEFAULT_FS,
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` cite the paper’s statement that both neural and behavior traces were averaged in bins of 10 consecutive timestamps for decoding, and the AI follows that rule throughout.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a dedicated timestamp file. It is synthesized from the imaging frame index and frame rate, using `ops["nframes"]` and `ops["fs"]`.

ii.
```python
def full_frame_times_seconds(n_frames: int, fs: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) / fs

def build_time_input(n_frames: int, fs: float, bin_frames: int) -> np.ndarray:
    frame_times = full_frame_times_seconds(n_frames, fs)
    binned = mean_bin_1d(frame_times.astype(np.float32), bin_frames)
    return binned.astype(np.float32)
```

iii. In Step 5, the AI says this variable should be “absolute time-from-session-start in seconds,” and that with a constant 30 Hz imaging clock there is no need for a separate raw time series.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI builds a per-frame session-time vector, then averages it in the same non-overlapping 10-frame bins as the neural and motion data. This means the saved time values are bin centers/means (for example the first value is about `0.15 s`), not left-bin edges.

ii.
```python
def build_time_input(n_frames: int, fs: float, bin_frames: int) -> np.ndarray:
    frame_times = full_frame_times_seconds(n_frames, fs)
    binned = mean_bin_1d(frame_times.astype(np.float32), bin_frames)
    return binned.astype(np.float32)
```

iii. The AI’s justification in Step 5 and Step 10 is that the decoder input should remain synchronized with the binned neural data; averaging the frame-time vector with the same binning procedure keeps the time series length and temporal placement consistent with the other streams.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it uses the same imaging-frame count, the same 10-frame temporal averaging, and the same 60-second trial splitting as the neural data.

ii.
```python
time_binned = build_time_input(n_frames, fs, bin_frames)
...
input_trials = split_trials_1d(time_binned, bins_per_trial)
neural_trials = split_trials_2d(neural_binned, bins_per_trial)
```

iii. In Step 5, the AI explicitly planned to “average in the same 10-frame bins” and then “preserve absolute time across trials within a session,” so the time input would stay aligned with the neural bins.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`. The AI also uses `tstamps.npy` and `interframe_int.npy` to infer timestamp units, reconstruct the behavior timeline, and repair dropped camera frames.

ii.
```python
def load_behavior_arrays(session_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion = np.load(session_path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session_path / "move_deve" / "tstamps.npy", allow_pickle=True)
    interframe = np.load(session_path / "move_deve" / "interframe_int.npy", allow_pickle=True)
    return motion.astype(np.float64), tstamps.astype(np.float64), interframe.astype(np.float64)
```

iii. Step 2 and Step 5 of `CONVERSION_NOTES.md` say the release sometimes has missing video frames and provides `tstamps.npy` / `interframe_int.npy` specifically to detect and repair those gaps.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI rescales timestamps into seconds if needed, interpolates the raw motion signal onto the full imaging frame clock with `np.interp`, averages the aligned motion in non-overlapping 10-frame bins, then discretizes the binned values into five categories.

ii.
```python
def align_motion_to_imaging(
    motion: np.ndarray,
    tstamps: np.ndarray,
    interframe: np.ndarray,
    n_frames: int,
    fs: float,
) -> tuple[np.ndarray, dict]:
    scale = infer_timestamp_scale_seconds(interframe, fs)
    motion_times = tstamps * scale
    imaging_times = full_frame_times_seconds(n_frames, fs)
    aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
    ...
    return aligned.astype(np.float32), info
```

```python
motion_binned = mean_bin_1d(aligned_motion.astype(np.float32), bin_frames).astype(np.float32)
motion_labels, motion_quantiles = equal_frequency_bins(motion_binned.astype(np.float64), 5)
```

iii. The AI’s notes justify this in Step 4 and Step 5: the paper says behavior and imaging were synchronized at 30 Hz, the release exposes dropped video frames, and the benchmark requires a categorical, time-varying motion target after the paper’s 10-frame denoising.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI does **not** use percentile thresholds in the same way as the human reference. Instead, it sorts all binned motion values within a session, assigns ranks, and converts ranks into exactly equal-frequency quintile labels. It also computes true quantiles with `np.quantile`, but only stores them for summaries and plots; they are not used to assign labels.

ii.
```python
def equal_frequency_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    bins = (ranks * n_bins) // n
    bins = np.minimum(bins, n_bins - 1).astype(np.int64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    return bins, quantiles.astype(np.float64)
```

iii. The explicit justification appears in Step 5: the AI wanted exact class balance even in the presence of ties or long immobility periods, and considered rank-based equal-frequency assignment safer for the categorical decoder than percentile-edge thresholding.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data by constructing an imaging-frame time axis from `n_frames` and `fs`, converting behavior timestamps into seconds, and interpolating the motion signal onto every imaging frame before any temporal binning or trial splitting.

ii.
```python
motion_times = tstamps * scale
imaging_times = full_frame_times_seconds(n_frames, fs)
aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
```

```python
motion_binned = mean_bin_1d(aligned_motion.astype(np.float32), bin_frames).astype(np.float32)
output_trials = split_trials_1d(motion_labels.astype(np.int64), bins_per_trial)
```

iii. Step 4 and Step 5 of `CONVERSION_NOTES.md` say the release contains occasional missing camera frames, so the motion signal should be interpolated onto the imaging frame clock to preserve alignment with every neural sample.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling path is for missing behavior frames. The AI infers the timestamp unit, interpolates the motion signal onto the full imaging frame clock, records how many motion frames were missing, and errors out if motion timestamps are absent or if arrays are too short for the requested binning/trialing. It does not implement a separate “discard the remainder” step for partial final trials.

ii.
```python
if motion_times.size == 0:
    raise ValueError("No motion timestamps available.")

aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
...
"missing_motion_frames": int(n_frames - len(motion)),
```

```python
def mean_bin_1d(arr: np.ndarray, bin_frames: int) -> np.ndarray:
    usable = (arr.shape[0] // bin_frames) * bin_frames
    if usable == 0:
        raise ValueError("Array is too short for requested bin size.")
    return arr[:usable].reshape(usable // bin_frames, bin_frames).mean(axis=1)
```

iii. The notes repeatedly justify interpolation as the right way to handle the known dropped-camera-frame issue in the release. No separate justification was recorded for the stricter divisibility checks beyond the claim that the released sessions tile cleanly after 10-frame binning.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies full-session neural baseline correction with `dcnv.preprocess` as the dominant bottleneck. Session-level file loading and optional plotting are secondary.

ii.
```python
corrected = dcnv.preprocess(
    fc.copy(),
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=fs,
    prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
    batch_size=int(ops.get("batch_size", 2000)),
    device=torch.device("cpu"),
)
```

iii. In Step 6 and Step 7 of `CONVERSION_NOTES.md`, the AI explicitly says full-session baseline correction is the main expected bottleneck and uses timing prints to monitor runtime per session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the expensive binning and motion-alignment steps, so the remaining obvious Python loops are relatively minor: the per-session outer loop, the small loop used only to summarize motion gaps, the list-building trial split, and plotting loops over a few example neurons. In particular, it avoided the repeated `np.insert` loop used in the human reference by aligning with a single `np.interp` call.

ii.
```python
for idx in gaps:
    missing = int(round((motion_times[idx + 1] - motion_times[idx]) * fs)) - 1
    gap_sizes.append(max(missing, 0))
```

```python
return [reshaped[:, i, :].astype(arr.dtype, copy=False) for i in range(n_trials)]
```

iii. The justification is indirect: Step 6 says the AI wanted vectorized loops and lists `reshape/mean` binning and `np.interp` as deliberate speedups, implying that the remaining loops were considered small enough not to matter.

## 6-c. What processing does the code repeat multiple times?

i. Within the conversion script, the repeated work is mostly routine per-session processing: it rebuilds imaging-frame time vectors for every session, bins neural/motion/time streams separately, and performs separate neural/input/output trial splits. It also computes motion quantiles for every session even though those quantiles are only used for metadata and optional plots.

ii.
```python
imaging_times = full_frame_times_seconds(n_frames, fs)
...
neural_binned = mean_bin_2d(corrected_F, bin_frames).astype(np.float32)
motion_binned = mean_bin_1d(aligned_motion.astype(np.float32), bin_frames).astype(np.float32)
time_binned = build_time_input(n_frames, fs, bin_frames)
...
neural_trials = split_trials_2d(neural_binned, bins_per_trial)
input_trials = split_trials_1d(time_binned, bins_per_trial)
output_trials = split_trials_1d(motion_labels.astype(np.int64), bins_per_trial)
```

iii. No explicit justification was recorded beyond general simplicity and per-session processing. The notes mainly emphasize bounded memory and straightforward vectorized operations rather than caching or deduplicating these repeated computations.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does extra diagnostic work that the downstream decoder does not need: optional processing plots, detailed session summaries, gap statistics, and stored motion quantiles. More importantly, the computed `motion_quantiles` are not used to assign labels; they are only logged and stored after labels have already been produced by rank-based equal-frequency binning.

ii.
```python
motion_labels, motion_quantiles = equal_frequency_bins(motion_binned.astype(np.float64), 5)
...
"motion_quantiles": motion_quantiles.tolist(),
"motion_gap_sizes_frames": motion_info["gap_sizes_frames"].tolist(),
```

```python
if make_plot:
    build_processing_plot(
        Path(f"/app/processing_{session.session_id}.png"),
        ...
    )
```

iii. The notes justify this as validation support: the plots and summaries were added so the user could inspect alignment and discretization, and the metadata was meant to support later sanity checks. There is no justification for computing quantiles that are then ignored for label assignment other than plotting/reporting.
