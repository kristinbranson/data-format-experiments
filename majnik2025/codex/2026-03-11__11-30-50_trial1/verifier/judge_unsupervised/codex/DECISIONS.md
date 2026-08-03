# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `data/` for all subject folders whose names start with `jm`, then scans each subject for session folders whose names begin with digits. For each discovered session it loads neural files from `suite2p/plane0/` and behavior files from `move_deve/`. In full mode it processes every discovered session; in sample mode it chooses two representative sessions.

ii. ```python
def discover_sessions(data_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
            sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
    if not sessions:
        raise RuntimeError("No sessions found under data/.")
    return sessions

def process_session(session: SessionInfo) -> dict:
    s2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"
    neural_processed, ops, neural_preview = compute_baseline_corrected_fluorescence(s2p_dir)
    motion_raw = np.load(move_dir / "motion_energy_glob.npy")
    timestamps = np.load(move_dir / "tstamps.npy")
    interframe_int = np.load(move_dir / "interframe_int.npy")
```

iii. In `CONVERSION_NOTES.md` Step 2 the agent documented the dataset layout as `jm0xx/YYYY-MM-DD_a/suite2p/plane0` plus `move_deve/`, and in Step 5 it explicitly decided to “keep all 41 sessions.” The trajectory shows the same reasoning: it treated the released dataset structure as authoritative and processed the full release rather than the toy paths in `DefaultTrackOps`.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level folder. The subject ID is the folder name such as `jm031`. The final dataset stores the sorted unique subject IDs in `subjects` and maps each session to its subject with `subject_idx`.

ii. ```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...

subjects = sorted({session["info"].subject for session in processed_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

"subjects": subjects,
"subject_idx": np.array(
    [subject_to_idx[session["info"].subject] for session in processed_sessions],
    dtype=np.int64,
),
```

iii. The notes say the subject folders present are `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`, matching `data/README.md`. The agent justified this as the natural subject split used by the release and by the paper’s six-mouse dataset summary.

## 1-c. How are the data split into sessions?

i. Sessions are split by per-subject subdirectories named like `YYYY-MM-DD_a`. Each such folder becomes one session in the output lists.

ii. ```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()):
    sessions.append(SessionInfo(subject=subject_dir.name, session=session_dir.name, path=session_dir))
```

iii. In Step 2 of `CONVERSION_NOTES.md` the agent recorded that each subject contains daily session folders named like `YYYY-MM-DD_a`, and Step 5 says to keep all 41 sessions. The trajectory also shows it read `data/load_data.ipynb`, which iterates over subject subdirectories in chronological order the same way.

## 1-d. How are the data split into trials?

i. The source data are continuous recordings with no native trial table. The agent therefore invents “pseudo-trials” by first averaging in non-overlapping 10-frame bins, then cutting each binned session into consecutive non-overlapping 2-minute blocks. Each 2-minute block becomes one trial. A 20-minute session yields 10 trials and a 30-minute session yields 15.

ii. ```python
RAW_FS_HZ = 30.0
BIN_FRAMES = 10
TRIAL_DURATION_SEC = 120.0

trial_bins = int(TRIAL_DURATION_SEC * RAW_FS_HZ / BIN_FRAMES)

def segment_trials(..., trial_bins: int):
    n_total_bins = neural_binned.shape[1]
    n_trials = n_total_bins // trial_bins
    ...
    for trial_idx in range(n_trials):
        start = trial_idx * trial_bins
        stop = start + trial_bins
        neural_trial = neural_binned[:, start:stop]
        input_trial = time_binned[np.newaxis, start:stop]
        output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
```

iii. The justification is explicit in Step 5 of `CONVERSION_NOTES.md`: there is “no native trial structure,” so the agent chose “consecutive 2-minute pseudo-trials” because the paper’s decoder uses consecutive 2-minute blocks for cross-validation and the target decoder format requires at least two trials per session.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality rejection. The agent keeps every pseudo-trial produced by the fixed segmentation, except that it implicitly drops any incomplete tail at the end of a session because both binning and trial segmentation use floor division. It also refuses to keep a session if fewer than two pseudo-trials remain.

ii. ```python
def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    ...

def segment_trials(...):
    n_total_bins = neural_binned.shape[1]
    n_trials = n_total_bins // trial_bins
    ...

if len(neural_trials) < 2:
    raise ValueError(f"Session {session['info'].session_id} has fewer than 2 trials after segmentation.")
```

iii. In Step 3 the agent wrote that the recordings are not trial-based and that the paper does not describe trial curation. Step 5 then says “keep all 41 sessions” and Step 6 says no session fails the core criteria. For behavior-frame problems the agent chose repair by interpolation rather than dropping trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy`, `Fneu.npy`, and `ops.npy`. The agent does not use `spks.npy` for the final `neural` field.

ii. ```python
def compute_baseline_corrected_fluorescence(s2p_dir: Path) -> tuple[np.ndarray, dict, dict]:
    ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(s2p_dir / "F.npy", mmap_mode="r")
    Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
```

iii. The notebook excerpt the agent read says `F.npy` is the raw fluorescence trace and advises users to compute dF/F “the way as described in the paper (or alternatively use `spks.npy`).” In Step 5 the agent explicitly chose the paper-consistent fluorescence path rather than `spks`.

## 2-b. How is the `neural` data processed?

i. The agent computes neuropil-corrected fluorescence as `F - neucoeff * Fneu` using the per-session Suite2p `ops` values, then applies `suite2p.extraction.dcnv.preprocess(...)` with the saved Suite2p baseline settings to obtain baseline-corrected fluorescence. After that it averages the trace in non-overlapping 10-frame bins.

ii. ```python
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

neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
```

iii. The paper excerpt recovered in trajectory step 44 says the authors “used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters).” In Step 4 and Step 5 of `CONVERSION_NOTES.md` the agent used that sentence to reject the Track2p GUI helper and instead chose Suite2p-style baseline correction from `F`, `Fneu`, and `ops`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not perform a fresh neuron-level quality filter inside `convert_data.py`. It assumes the released `suite2p/plane0` arrays already contain the Track2p all-days matched subset and already satisfy the Suite2p `iscell > 0.5` filter, so it keeps all rows in the provided arrays.

ii. ```python
return {
    "info": session,
    "ops": ops,
    "brain_region_idx": np.zeros(neural_binned.shape[0], dtype=np.int64),
    "neural_binned": neural_binned,
    ...
}
```

iii. The notes justify this in two places. Step 1 records that Track2p downstream analyses use `t2p_match_mat_allday`, meaning only cells tracked across all days are retained. Step 2 records that the released session files already have identical row identities within a subject. Step 3 also records the paper’s `iscell > 0.5` rule, and the data snapshot indeed shows all provided `iscell` probabilities exceed 0.5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no real experimental alignment event in the source data. The agent uses the continuous imaging frame sequence as the master clock, bins it, and then defines trial boundaries purely by cutting the session into consecutive 2-minute blocks. In the metadata it declares the alignment event to be the “start of each consecutive 2-minute block.”

ii. ```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
...
"temporal_alignment_event": "Start of each consecutive 2-minute block cut from a continuous recording session",
"off_start": 0.0,
"off_end": TRIAL_DURATION_SEC,
```

iii. Step 5 of `CONVERSION_NOTES.md` says there is “no native trial structure,” so any trialization must be a downstream segmentation. The agent chose block starts as the alignment event because that let it express the continuous recordings in the requested trial-based decoder format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so each bin is `10 / 30 = 0.333...` s or `333.33 ms`. The code rebins neural, motion, and time streams with non-overlapping mean pooling.

ii. ```python
RAW_FS_HZ = 30.0
BIN_FRAMES = 10
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / RAW_FS_HZ

def bin_array_mean(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_bins = x.shape[-1] // bin_frames
    trimmed = x[..., : n_bins * bin_frames]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_frames)
    return trimmed.reshape(new_shape).mean(axis=-1, dtype=np.float32)
```

iii. The paper text extracted into `cache/paper_text.txt` says that for decoding they “slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps.” The agent cites that directly in Step 3 and Step 5 of the notes.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The `input` time variable is derived from the imaging frame index and the imaging sample rate `ops["fs"]`. The code does not use behavior timestamps as the authoritative experiment clock.

ii. ```python
n_frames = int(ops["nframes"])
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
```

iii. Step 5 in `CONVERSION_NOTES.md` states that imaging frames are the master timeline because `ops.npy` supplies the authoritative frame count and 30 Hz rate, while behavior can have missing frames. That is the reason the agent derived elapsed time from imaging rather than camera timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent computes the center time of each raw imaging frame in seconds, then averages those frame-center times in the same non-overlapping 10-frame bins used for the neural and behavior signals. It preserves absolute time within the original session rather than resetting each pseudo-trial to zero.

ii. ```python
time_full_sec = (np.arange(n_frames, dtype=np.float32) + 0.5) / np.float32(ops["fs"])
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
input_trial = time_binned[np.newaxis, start:stop].astype(np.float32, copy=False)
```

iii. In Step 5 the agent explicitly decided to “use bin-center time for each sample” and to keep the “absolute position within the original recording,” because the decoder task asked for “time elapsed from the beginning of the experiment.”

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is aligned by construction to the neural data: the code uses the same imaging-frame clock, the same 10-frame bins, and the same 2-minute pseudo-trial boundaries as for `neural`.

ii. ```python
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
time_binned = bin_array_mean(time_full_sec[np.newaxis, :], BIN_FRAMES)[0]
...
neural_trial = neural_binned[:, start:stop]
input_trial = time_binned[np.newaxis, start:stop]
```

iii. The agent’s Step 5 “Master clock” decision says imaging frames should define all modalities before trialization so that neural, input, and output share exactly the same sample boundaries.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from `move_deve/motion_energy_glob.npy`, with `interframe_int.npy` and `tstamps.npy` used only to detect and repair missing camera frames before alignment.

ii. ```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy")
timestamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
```

iii. `data/README.md` says `motion_energy_glob.npy` contains the processed behavioral data and that `tstamps.npy` / `interframe_int.npy` indicate missing frames. The notes and trajectory repeat that interpretation verbatim.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent does not recompute motion energy from raw videos. It treats `motion_energy_glob.npy` as the released motion-energy signal, reconstructs missing frames by inferring skipped indices from `interframe_int.npy`, linearly interpolates inserted gaps, averages the repaired trace in non-overlapping 10-frame bins, and later globally min-max normalizes it.

ii. ```python
def reconstruct_motion_trace(motion, interframe_int, target_len):
    median_interval = float(np.median(interframe_int))
    steps = np.rint(interframe_int / median_interval).astype(np.int64)
    ...
    full = np.full(target_len, np.nan, dtype=np.float32)
    full[observed_idx] = motion
    missing_idx = np.flatnonzero(np.isnan(full))
    full = interpolate_nans(full)
    return full, missing_idx

motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
```

iii. The paper text says motion was defined from squared pixelwise differences of consecutive frames, but the released dataset already stores that scalar as `motion_energy_glob.npy`. The agent’s notes say the README explicitly permits either treating missing frames as missing or interpolating them; it chose interpolation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent normalizes all binned motion-energy samples globally with one min and one max over the whole dataset, then computes global quintile edges at `[0, .2, .4, .6, .8, 1]` and discretizes each time bin with `np.digitize(...)` into classes `Q1`–`Q5`.

ii. ```python
motion_all = np.concatenate([session["motion_binned"] for session in processed_sessions]).astype(np.float32)
motion_min = float(np.min(motion_all))
motion_max = float(np.max(motion_all))
motion_norm_all = (motion_all - motion_min) / (motion_max - motion_min)
motion_edges = np.quantile(motion_norm_all, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).astype(np.float32)
motion_edges = np.maximum.accumulate(motion_edges)
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False).astype(np.int64)
```

iii. Step 5 states that this choice was made because the decoder specification requires a categorical output, specifically “motion energy, normalized and discretized into five equal-percentile bins.” The global rather than per-session normalization was a deliberate attempt to define one common label scale over the full dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Imaging frames are the reference timeline. If the behavior array is shorter than the imaging stream, the code reconstructs the missing frame positions from `interframe_int.npy`, inserts missing samples, interpolates them, then bins the repaired behavior trace with the same 10-frame windows and cuts it with the same pseudo-trial boundaries as `neural`.

ii. ```python
n_frames = int(ops["nframes"])
motion_full, missing_idx = reconstruct_motion_trace(motion_raw, interframe_int, n_frames)
neural_binned = bin_array_mean(neural_processed, BIN_FRAMES)
motion_binned = bin_array_mean(motion_full[np.newaxis, :], BIN_FRAMES)[0]
...
output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
```

iii. The notes call this the “Master clock” decision: the paper says video was triggered by the microscope, while the README says missing camera frames must be recovered from timing arrays. That led the agent to align behavior onto the imaging-frame timeline before any later processing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data issue the agent handles is missing camera frames. It identifies missing frame positions from abnormal inter-frame intervals, inserts `NaN`s at the missing indices, and linearly interpolates across them. It errors out if reconstruction fails or if an entire trace were `NaN`. It otherwise keeps all sessions. Minor leftovers at the ends of traces are silently dropped by floor-division binning and segmentation.

ii. ```python
def interpolate_nans(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    ...
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid]).astype(np.float32)
    return out

if observed_idx[-1] != target_len - 1:
    raise ValueError(...)

if not np.isfinite(motion_min) or not np.isfinite(motion_max) or motion_max <= motion_min:
    raise ValueError("Motion energy range is invalid after preprocessing.")
```

iii. The agent’s justification is in Step 3 and Step 4 of `CONVERSION_NOTES.md`: the data README explicitly says missing camera frames may be treated as missing or interpolated over. The agent chose interpolation because it preserves framewise alignment to the imaging stream without discarding sessions.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is the neural preprocessing in `compute_baseline_corrected_fluorescence`, especially loading large `F` and `Fneu` arrays and running `dcnv.preprocess(...)` over all neurons and frames for every session. Session-by-session processing over all 41 sessions is the other major cost.

ii. ```python
def compute_baseline_corrected_fluorescence(s2p_dir: Path) -> tuple[np.ndarray, dict, dict]:
    ops = np.load(s2p_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(s2p_dir / "F.npy", mmap_mode="r")
    Fneu = np.load(s2p_dir / "Fneu.npy", mmap_mode="r")
    ...
    processed = dcnv.preprocess(...)

for session in sessions:
    processed_sessions.append(process_session(session))
```

iii. Step 6 of `CONVERSION_NOTES.md` says “the main cost is Suite2p-style fluorescence preprocessing.” The trajectory also shows per-session timings printed during the full conversion run, consistent with that assessment.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest remaining Python loop is `segment_trials`, which loops over trial indices and appends each slice separately; this could have been reshaped into `(n_trials, n_neurons, trial_bins)` and then transposed once. The sample-session chooser also reloads `ops.npy` and motion lengths repeatedly inside small helper loops.

ii. ```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_bins
    stop = start + trial_bins
    neural_trial = neural_binned[:, start:stop]
    input_trial = time_binned[np.newaxis, start:stop]
    output_trial = np.digitize(motion_norm_binned[start:stop], motion_edges[1:-1], right=False)
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)

for session in all_sessions:
    if session == first:
        continue
    if session_nframes(session) != first_nframes and has_missing_behavior_frames(session):
        second = session
        break
```

iii. The agent itself noted in Step 6 that some bottlenecks remained but that binning and motion reconstruction had already been vectorized. From the final code, `segment_trials` is the main obvious loop that still need not be in Python.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats file reads in `select_sample_sessions` because `session_nframes(...)` and `has_missing_behavior_frames(...)` reload metadata for multiple candidate sessions. During plotting it also recomputes normalized motion and discretized classes that were already computed conceptually in dataset construction.

ii. ```python
def session_nframes(session: SessionInfo) -> int:
    return int(np.load(session.path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()["nframes"])

def has_missing_behavior_frames(session: SessionInfo) -> bool:
    move_dir = session.path / "move_deve"
    motion_len = int(np.load(move_dir / "motion_energy_glob.npy", mmap_mode="r").shape[0])
    nframes = session_nframes(session)
    return motion_len != nframes

motion_norm_binned = (motion_binned - motion_min) / (motion_max - motion_min)
motion_classes = np.digitize(motion_norm_binned, motion_edges[1:-1], right=False)
```

iii. This follows the code directly. The notes do not emphasize these repeats, but they do say the script was optimized mainly for correctness and vectorized binning, not for eliminating every redundant metadata read or plotting recomputation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script builds preview dictionaries for neural and motion traces, loads `timestamps` only for those previews, stores `ops` in the per-session dict even though the final dataset does not keep it, and includes optional plotting paths that are irrelevant to downstream decoder training. Those computations are only for diagnostics and documentation.

ii. ```python
preview = {
    "sample_neuron": sample_neuron,
    "F": np.asarray(F[sample_neuron, :preview_len], dtype=np.float32),
    "Fneu": np.asarray(Fneu[sample_neuron, :preview_len], dtype=np.float32),
    "corrected": corrected[sample_neuron, :preview_len].copy(),
    "processed": processed[sample_neuron, :preview_len].copy(),
}

motion_preview = {
    "raw": np.asarray(motion_raw[:preview_len], dtype=np.float32),
    "reconstructed": motion_full[:preview_len].copy(),
    "missing_idx": missing_idx[missing_idx < preview_len].copy(),
    "timestamps": np.asarray(timestamps[: min(preview_len, timestamps.shape[0])], dtype=np.float64),
    "interframe_int": np.asarray(interframe_int[: min(preview_len - 1, interframe_int.shape[0])], dtype=np.float64),
}

if args.show_processing:
    for session in processed_sessions[:2]:
        plot_processing(session, motion_min, motion_max, motion_edges, plot_path)
```

iii. The agent explicitly framed these as “processing plots,” “preview,” and “sanity checks” in the notes. They support the writeup and debugging, but they are discarded before the final decoder consumes the dataset.
