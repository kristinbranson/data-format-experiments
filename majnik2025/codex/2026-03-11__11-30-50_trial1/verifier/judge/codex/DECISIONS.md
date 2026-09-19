# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all data by scanning `data/` for subject directories whose names start with `jm`, then scanning each subject for date-like session directories. For each session it loads Suite2p calcium files from `suite2p/plane0/` and behavior files from `move_deve/`. Trials are not loaded from disk because the source data are continuous; they are created later in code by segmenting each processed session.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

```python
neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. In `CONVERSION_NOTES.md`, the AI says the data release is organized as `jm*` subject folders containing daily session folders with `suite2p/plane0` and `move_deve` subdirectories. In trajectory step 82, it justified loading `ops.npy`, fluorescence traces, and timing files so it could reconstruct missing behavior frames and reuse saved Suite2p processing parameters.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the top-level directories under `data/` whose names start with `jm`. They are sorted alphabetically, and the final `subjects` list is rebuilt from the processed sessions.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...

subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the AI notes that the subject folders present are `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and treats each folder as one mouse.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject directory whose name begins with four digits, i.e. the date-like folders such as `2023-10-18_a`. Session order is deterministic because the directories are sorted.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

iii. In `CONVERSION_NOTES.md`, the AI describes each daily recording folder as one session and says all 41 such session folders are kept.

## 1-d. How are the data split into trials?

i. The AI does not use the instruction-specified 60-second trials. Instead, it bins each continuous session first and then splits the binned data into consecutive non-overlapping 2-minute pseudo-trials, producing 10 trials for 20-minute sessions and 15 trials for 30-minute sessions.

ii.
```python
TRIAL_DURATION_SEC = 120.0
...
trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
```

```python
"trial_duration_sec": TRIAL_DURATION_SEC,
"temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
```

iii. The AI explicitly justifies this in Step 5 of `CONVERSION_NOTES.md` and trajectory step 78: because the paper’s decoder evaluation used consecutive 2-minute blocks, it decided to turn those blocks into the target format’s trial structure rather than use arbitrary windows.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The AI simply creates every 2-minute block available in a session. The only structural check is that a session must yield at least two trials after segmentation.

ii.
```python
for trial_idx in range(n_trials):
    ...
if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
```

iii. In `CONVERSION_NOTES.md`, the AI states that it keeps all 41 sessions and all generated pseudo-trials because no session fails its core inclusion criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from `F.npy` and `Fneu.npy` in each session’s `suite2p/plane0/` directory. The AI also reads `ops.npy` to get Suite2p preprocessing parameters such as `neucoeff`, `baseline`, `fs`, and related settings.

ii.
```python
ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. Step 5 of `CONVERSION_NOTES.md` says the neural signal should come from Suite2p-style baseline-corrected fluorescence reconstructed from `F`, `Fneu`, and `ops`, rather than from `spks.npy` or the GUI helper.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil using the `neucoeff` stored in `ops.npy`, then runs `suite2p.extraction.dcnv.preprocess` with the per-session Suite2p parameters from `ops.npy`. After that, it averages the resulting trace in non-overlapping bins of 10 frames.

ii.
```python
corrected = np.array(F, dtype=np.float32, copy=True)
corrected -= np.float32(ops["neucoeff"]) * np.asarray(Fneu, dtype=np.float32)
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 2000)),
    device=torch.device("cpu"),
).astype(np.float32, copy=False)
```

```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```

iii. The AI’s notes say the paper’s downstream analyses used Suite2p baseline-corrected fluorescence with default Suite2p parameters, so it chose to reconstruct that processing directly from `F`, `Fneu`, and the saved `ops.npy` values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no additional neuron-level quality filter inside `convert_data.py`. It uses every neuron row present in the exported session arrays, on the assumption that the released arrays already contain Track2p-matched, cell-filtered neurons.

ii.
```python
F = np.load(s2p_dir / "F.npy", mmap_mode="r")
Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
...
return {
    ...
    "brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
    "neural_binned": neural_binned,
```

iii. In Steps 1, 4, and 5 of `CONVERSION_NOTES.md`, the AI argues that the bundled Suite2p arrays are already the Track2p all-days matched export, so applying `iscell` again would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of a 2-minute pseudo-trial block, not to the session start. The neural matrix for a trial is simply the slice of the binned session trace between that block’s `start` and `stop` indices.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
```

```python
"temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SEC,
```

iii. The AI justifies this with the same 2-minute-block argument used for trialization: because the paper evaluates decoders on 2-minute temporal blocks, it treats the start of each block as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have a temporal resolution of 10 raw frames per sample, i.e. `333.333... ms` at 30 Hz. The AI applies non-overlapping temporal averaging with `BIN_FRAMES = 10` to neural, motion, and time.

ii.
```python
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ
...
def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32).astype(np.float32, copy=False)
```

iii. In `CONVERSION_NOTES.md`, the AI cites the paper’s decoding description that neural and behavior traces were averaged over 10 consecutive timestamps before decoding, and treats this as a hard constraint.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a dedicated raw timestamp variable. It is synthesized from the imaging frame index `np.arange(n_frames)` and the imaging frame rate `ops["fs"]`.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says elapsed session time is a task-specific benchmark input rather than a native paper variable, so it derives it from the session clock implied by frame count and frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI constructs a per-frame time axis in seconds using frame centers (`index + 0.5`), then averages those values in the same 10-frame bins used for neural and behavior data. It therefore stores the mean time of each bin rather than the left edge.

ii.
```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
...
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
```

iii. The notes describe this as using bin-center time for each sample after 10-frame averaging so the time input stays synchronized with the binned neural and motion traces.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the time input to the neural data by binning the frame-wise time axis with the same 10-frame averaging and then slicing the same `start:stop` trial boundaries used for the neural matrix.

ii.
```python
neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
```

iii. The justification in the notes is that elapsed time should be a time-varying decoder input defined on the same samples as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes primarily from `move_deve/motion_energy_glob.npy`. The AI also uses `interframe_int.npy` to reconstruct dropped camera frames; `tstamps.npy` is loaded, but only stored in a preview dictionary for diagnostics rather than used in the conversion itself.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. In Steps 2, 4, and 5 of `CONVERSION_NOTES.md`, the AI says motion energy is the behavior variable to decode and missing behavior frames must be reconstructed from the timing-gap files before alignment.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI reconstructs a full-length motion trace by inferring missing frame positions from `interframe_int.npy`, inserting `NaN` placeholders, and linearly interpolating them. It then averages the reconstructed motion signal in 10-frame bins. Before discretization, it globally min-max normalizes the concatenated binned motion values across all sessions.

ii.
```python
median_interval = float(np.median(interframe_int))
steps = np.rint(interframe_int / median_interval).astype(np.int64)
steps[steps < 1] = 1
...
full = np.full(target_len, np.nan, dtype=np.float32)
full[observed_idx] = motion
missing_idx = np.flatnonzero(np.isnan(full))
full = interpolate_nans(full)
```

```python
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
...
motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions]).astype(np.float32)
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies the reconstruction as a more exact way to use the camera timing gaps, and says global normalization is part of its benchmark-driven output construction.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI does not threshold motion energy within each session. Instead, it computes global quintile edges from the normalized binned motion values pooled across all sessions, then uses those same edges to discretize every session into five categories named `Q1` to `Q5`.

ii.
```python
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the benchmark requires categorical outputs, so the AI chose "global equal-percentile quintiles" over normalized motion energy and expected near-uniform class weights as a sanity check (also reiterated in trajectory step 163).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses the imaging stream as the master clock. It reconstructs missing behavior frames to the neural session length, bins both modalities with the same 10-frame averaging, and then slices matching trial windows from the two binned arrays.

ii.
```python
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
```

```python
neural_trial = neural_binned[:, start:stop].astype(np.float32, copy=False)
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. In Steps 3, 4, 5, and 10 of `CONVERSION_NOTES.md`, the AI argues that imaging frames are the authoritative session clock and that behavior frames should be reinserted from timing gaps before any joint binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing behavior frames by reconstructing their positions from the inter-frame intervals and linearly interpolating over the gaps. It raises an error if timing reconstruction is inconsistent or if interpolation would operate on an all-NaN array. It does not add special handling for missing neural values.

ii.
```python
if observed_idx[-1] != target_len - 1:
    raise ValueError(
        f"Timing reconstruction failed: last observed index {observed_idx[-1]} does not match target {target_len - 1}"
    )
...
missing_idx = np.flatnonzero(np.isnan(full))
full = interpolate_nans(full)
```

```python
if not valid.any():
    raise ValueError("Cannot interpolate an array containing only NaNs.")
```

iii. In Steps 4, 5, 9, and 10 of `CONVERSION_NOTES.md`, the AI describes the missing-camera-frame issue as the main data defect and says the repair should preserve framewise alignment with imaging.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies per-session Suite2p fluorescence preprocessing as the dominant cost, especially the `dcnv.preprocess` baseline-correction pass. Everything else is comparatively light-weight array manipulation and I/O.

ii.
```python
processed = dcnv.preprocess(
    corrected,
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=int(ops.get("batch_size", 2000)),
    device=torch.device("cpu"),
).astype(np.float32, copy=False)
```

iii. In Step 6 of `CONVERSION_NOTES.md`, the AI explicitly says the main cost is Suite2p-style fluorescence preprocessing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the missing-frame reconstruction that was the main obvious target. The remaining Python loops are the session loop and the per-trial loop in `segment_trials`, which could be replaced by a reshape-based trial view if further optimization were needed.

ii.
```python
for session in sessions:
    processed_sessions.append(process_session(session))
```

```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_trial)
```

iii. Step 6 of `CONVERSION_NOTES.md` says the AI intentionally used vectorized timing-step accumulation and reshape-based binning to remove unnecessary Python-loop overhead.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats some diagnostic-only processing. It computes normalized motion values and digitized motion classes once in `build_dataset` for the actual output and then again in `plot_processing` for figures. It also repeatedly loads `ops.npy` inside sample-session selection when checking frame counts.

ii.
```python
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
```

```python
motion_norm_binned = (motion_binned - motion_min) / (motion_max - motion_min)
motion_classes = np.digitize(motion_norm_binned, motion_edges[1:-1], right=False)
```

iii. The AI does not call this out strongly in the notes, but its Step 7 discussion of fixing a plotting-only normalization bug makes clear that figure generation reuses and recomputes part of the motion-processing path separately from the saved dataset.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code always builds preview payloads for plotting and debugging, even on full runs where no plots are requested. That includes loading `tstamps.npy`, slicing preview traces, and storing `neural_preview` and `motion_preview` dictionaries that are never written into the saved dataset. The plotting imports are also always loaded.

ii.
```python
timestamps = np.load(move_dir / "tstamps.npy")
...
preview_len = min(3000, n_frames)
motion_preview = {
    "raw": np.asarray(motion_raw[:preview_len], dtype=np.float32),
    "reconstructed": motion_full[:preview_len].copy(),
    "missing_idx": missing_idx[missing_idx < preview_len].copy(),
    "timestamps": np.asarray(timestamps[: min(preview_len, timestamps.shape[0])], dtype=np.float64),
    "interframe_int": np.asarray(interframe_int[: min(preview_len - 1, interframe_int.shape[0])], dtype=np.float64),
}
```

```python
return {
    ...
    "neural_preview": neural_preview,
    "motion_preview": motion_preview,
}
```

iii. The AI’s notes justify these extras as sanity-check support and optional processing figures, but they are not consumed by the final decoder dataset and are discarded after conversion.
