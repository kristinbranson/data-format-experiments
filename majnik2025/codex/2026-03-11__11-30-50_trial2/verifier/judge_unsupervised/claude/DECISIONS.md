# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all sessions by iterating over sorted subdirectories of `data/` that start with "jm" (subjects), then within each subject, iterating over sorted subdirectories matching a date pattern (sessions). For each session, it loads Suite2p neural files (`F.npy`, `Fneu.npy`, `ops.npy`) from `suite2p/plane0/` and behavioral files (`motion_energy_glob.npy`, `tstamps.npy`) from `move_deve/`. There is no native trial structure; the continuous recordings are trialized later. All 41 sessions across 6 subjects are loaded.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionRef]:
    sessions: list[SessionRef] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
        ):
            sessions.append(
                SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
            )
    return sessions
```

```python
# In process_session():
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md that the data directory contains 6 subject folders (jm031-jm046) each with 6-7 session folders. Each session has `suite2p/plane0/` and `move_deve/` subdirectories. The data README states the provided suite2p traces already contain only cells tracked across all days per mouse, so the agent loads them directly without re-running Track2p matching.

## 1-b. How are the data split into subjects?

i. Subjects are identified by their folder names (e.g., `jm031`, `jm032`, ..., `jm046`). A sorted unique list of subject names is created, and each session is mapped to its subject via `subject_to_idx`. The `subjects` list and `subject_idx` array are stored in the final dataset.

ii.
```python
def build_dataset(converted_sessions: list[dict], session_refs: list[SessionRef]) -> dict:
    subjects = sorted({session.subject for session in session_refs})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    # ...
    "subjects": subjects,
    "subject_idx": np.array([subject_to_idx[session.subject] for session in session_refs], dtype=np.int64),
```

iii. The AI noted that the data directory has 6 subject folders and that subject identity is encoded in the folder hierarchy. This is consistent with the data README documentation.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder that matches a date pattern (YYYY-MM-DD format) is treated as a separate session. Sessions are processed sequentially and stored as separate entries in the neural/input/output lists. Total: 41 sessions (7 per subject except jm040 which has 6).

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

iii. The AI documented in Step 2 of CONVERSION_NOTES.md that sessions are organized as date-stamped folders within each subject directory. The paper states a minimum of 6 consecutive daily sessions per subject, which is consistent with the 41 total sessions found.

## 1-d. How are the data split into trials?

i. Since the source data has no native trial structure (continuous spontaneous-behavior recordings), the AI creates artificial trials by splitting each session into consecutive non-overlapping 2-minute blocks. This matches the paper's decoding methodology which used "consecutive 2-minute blocks." Each trial has exactly 360 time bins (120 seconds * 30 Hz / 10-frame bins). Any trailing frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_SECONDS = 120.0

def split_trials(neural_binned, time_binned_s, output_one_hot, fs, bin_frames):
    bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
    usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
    # ...
    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
        input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
        output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The AI justified this in CONVERSION_NOTES.md Step 5: "Define each trial as one consecutive 2-minute block from a continuous session, matching the paper's decoding split unit exactly." 20-minute sessions produce 10 trials, 30-minute sessions produce 15 trials, totaling 545 trials.

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered or excluded based on quality controls. All complete 2-minute blocks are included. The only data discarded is the trailing partial block at the end of a session if it doesn't fill a complete 2-minute trial. The AI validates each converted session to ensure at least 2 trials exist and that no NaN values are present.

ii.
```python
def validate_converted_session(converted: dict) -> None:
    neural = converted["neural"]
    input_ = converted["input"]
    output = converted["output"]
    n_trials = len(neural)
    if n_trials < 2:
        raise ValueError("Each converted session must contain at least 2 trials.")
    # ...
    if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
        raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The AI noted in CONVERSION_NOTES.md that there are no native trial curation rules in the source data or paper for this analysis, as it involves spontaneous behavior rather than stimulus-locked trials. The validation checks confirm data integrity but do not filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from three Suite2p files: `F.npy` (raw fluorescence traces), `Fneu.npy` (neuropil fluorescence traces), and `ops.npy` (Suite2p processing parameters including `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `fs`, and `prctile_baseline`).

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

iii. The AI documented in CONVERSION_NOTES.md that "the paper analyzes dF/F while the package mainly exports raw F/spks" and decided to reconstruct a paper-consistent fluorescence signal from F, Fneu, and ops rather than using raw F directly.

## 2-b. How is the `neural` data processed?

i. Neural processing has two steps: (1) Neuropil subtraction: `Fc = F - neucoeff * Fneu` using the `neucoeff` value from Suite2p ops. (2) Baseline correction via `suite2p.extraction.dcnv.preprocess()` which applies Suite2p's standard baseline subtraction using parameters from ops (baseline method, window, sigma, percentile). The result is then averaged in non-overlapping 10-frame bins and split into 2-minute trials.

ii.
```python
def compute_fluorescence_signal(F, Fneu, ops):
    Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
    processed = suite2p_preprocess(
        Fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        batch_size=min(512, max(32, Fc.shape[0])),
        device=torch.device("cpu"),
    )
    return processed.astype(np.float32, copy=False)

# Then binned:
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The AI justified this in CONVERSION_NOTES.md: "Paper states downstream analyses used baseline-corrected fluorescence traces as dF/F" and referenced `code/track2p/gui/data_management.py:F_processing` which implements a similar approach using neuropil subtraction plus baseline removal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied. The AI determined that the provided suite2p exports already contain only cells that (1) passed Suite2p's cell classifier with probability > 0.5 (`iscell_thr = 0.5`) and (2) were tracked across all days within each subject by Track2p. The AI verified that neuron counts are constant across sessions within each subject, confirming the data is already curated.

ii.
```python
# No filtering code - all neurons from F.npy are used directly
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
```

iii. The AI documented in CONVERSION_NOTES.md Step 4: "Treat provided rows as already curated tracked cells; still confirm all rows pass iscell > 0.5" and "The relevant curation rules are already baked into the exported rows: Suite2p cell filtering plus all-day Track2p matching."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The temporal alignment event is the start of each consecutive 2-minute recording block. Since the recordings are continuous, neural data is simply split into consecutive 2-minute blocks with no additional temporal shifting. Each trial starts at time 0 of that block (relative to the session start, the input variable carries the absolute session time).

ii.
```python
# In metadata:
"temporal_alignment_event": "start of each consecutive 2-minute recording block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),

# In split_trials:
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop])
```

iii. The AI justified this alignment as matching the paper's decoding methodology which used consecutive 2-minute blocks as the unit of analysis. There is no stimulus onset or other event to align to, as the experiment involves spontaneous behavior.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw data is acquired at 30 Hz. The AI applies non-overlapping 10-frame temporal binning, resulting in a final time bin size of 333.33 ms. This matches the paper's description of "averaging in bins of 10 consecutive timestamps" for decoding.

ii.
```python
BIN_FRAMES = 10

def average_nonoverlapping(x, bin_frames):
    n_frames = x.shape[-1]
    usable = (n_frames // bin_frames) * bin_frames
    x = x[..., :usable]
    new_shape = x.shape[:-1] + (usable // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)

# In metadata:
"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),  # = 333.33 ms
```

iii. The AI documented in CONVERSION_NOTES.md: "Paper decoding uses 10 consecutive timestamps and 2-minute blocks. convert_data.py uses non-overlapping 10-frame binning followed by consecutive 2-minute trials, matching the paper."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from a variable in the raw data files. Instead, it is synthetically constructed from the session's sampling rate (`ops['fs']` = 30 Hz) and the bin frame count (10). It represents the elapsed time in seconds from the start of the recording session.

ii.
```python
def make_time_input(n_bins, fs, bin_frames):
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)

time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
```

iii. The AI stated in CONVERSION_NOTES.md: "Interpret 'time elapsed from the beginning of the experiment' as time from the beginning of the recording session, carried through each 2-minute trial as an absolute-within-session time vector."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A simple linear sequence is generated: `time = bin_index * (bin_frames / fs)`, where `bin_frames=10` and `fs=30.0`. This creates a time vector in seconds, starting at 0 for the first bin. The time values are absolute within the session (not reset at each trial boundary), so the first trial has values [0, 0.333, ..., 119.67] while later trials continue from where the previous trial ended.

ii.
```python
def make_time_input(n_bins, fs, bin_frames):
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)

# Then in split_trials, the session-level time is sliced into trials:
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. The AI noted this provides the decoder with temporal context within the session, which could be useful for capturing developmental or within-session dynamics.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is perfectly aligned with the neural data by construction. Both are computed on the same 10-frame binned grid: the time vector has exactly as many bins as the neural data, and both are sliced into trials at the same boundaries. Each time bin corresponds to the same temporal window as the corresponding neural bin.

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
# Both neural_binned and time_binned_s have the same number of bins
# Both are sliced identically in split_trials
```

iii. The alignment is inherent in the construction; since the time vector is derived from the same frame grid as the neural data, there can be no misalignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from two files in the `move_deve/` directory: `motion_energy_glob.npy` (the raw global motion energy signal, a 1D array of pixel-wise frame-difference values) and `tstamps.npy` (timestamps used to align motion energy samples to the imaging frame grid).

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The AI documented that the paper describes "global motion energy computed from consecutive video frames by taking pixel-wise differences, squaring, and summing across pixels," and the data README describes these files.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The motion energy processing pipeline has multiple steps:
1. **Frame alignment**: Motion energy samples are mapped onto the imaging frame grid using timestamps. Frame indices are computed by dividing timestamp differences by frame duration and rounding.
2. **Aggregation**: When multiple motion samples map to the same frame, they are averaged.
3. **Interpolation**: Missing frames (no motion sample mapped) are filled via linear interpolation.
4. **Binning**: 10-frame non-overlapping averaging, matching neural data binning.
5. **Normalization**: Min-max normalization within each session to [0, 1].
6. **Discretization**: Quantile-based binning into 5 equal-percentile categories (quintiles).
7. **One-hot encoding**: Converted to 5 binary channels for the decoder.

ii.
```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

iii. The AI justified this chain as following the paper's methodology: motion energy is synchronized to imaging at 30 Hz, then averaged in 10-frame bins, and finally discretized into quintiles as required by the task instructions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories using session-wise quintiles (20th, 40th, 60th, 80th percentiles as boundaries). Each time point is assigned to one of 5 bins using `np.searchsorted` on the quantile edges. The result is one-hot encoded into 5 binary channels, where each channel indicates membership in that quintile.

ii.
```python
def quintile_one_hot(x):
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. The AI documented: "Normalize motion energy within session after 10-frame averaging, then discretize into five equal-percentile bins within that session and export them one-hot because the task and provided decoder require categorical outputs."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first aligned to the imaging frame grid (same number of frames as the neural F.npy), then both neural and motion data undergo the same 10-frame binning, and finally both are split into trials at the same boundaries. This ensures frame-by-frame alignment throughout the pipeline.

ii.
```python
# Align to imaging frame grid
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
# Same binning as neural
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
# Same trial splitting
output_trials.append(output_one_hot[:, start:stop])
```

iii. The AI justified this by noting that "imaging and video were synchronized at 30 Hz" per the paper, and the timestamp-based reconstruction explicitly handles the 9 sessions with missing camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The primary data quality issue is missing camera frames in 9 of 41 sessions (mismatches of 1-148 frames between motion and imaging). The AI handles this by:
1. Using timestamps to map each motion sample to its corresponding imaging frame.
2. Detecting frames with no motion data (gaps in camera recording).
3. Linearly interpolating the missing motion values from surrounding valid frames.
4. The NaN-free validation check ensures no missing data propagates to the final output.

ii.
```python
def reconstruct_motion_trace(motion_energy, tstamps, n_imaging_frames):
    # ...
    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)
    full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
    # ...aggregate valid frames...
    missing = np.flatnonzero(~valid)
    if len(missing):
        full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

iii. The AI documented: "Data README says missing camera frames should be treated as missing or interpolated." The approach handles edge cases including sessions with only 1 valid frame (fill all with that value) and ensures no NaN values remain.

## 6-a. What are the most time-consuming steps of the code?

i. Based on the AI's timing analysis, the most time-consuming step is the Suite2p baseline correction (`suite2p_preprocess`), which operates on the full neuron-by-frame fluorescence matrix for each session. The overall conversion took ~2.26 seconds per session on average, with the full 41-session conversion completing in approximately 1.5 minutes.

ii.
```python
processed = suite2p_preprocess(
    Fc.copy(),
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    # ... many parameters
)
```

iii. The AI noted in CONVERSION_NOTES.md: "Need to benchmark Suite2p-style fluorescence preprocessing on sample data before deciding whether further optimization is required." After benchmarking at 2.26s/session, they determined no optimization was needed.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop in the code is the session-level loop in `main()` which iterates over all 41 sessions sequentially. Within sessions, the trial splitting loop iterates over trial boundaries. Most other operations are already vectorized (binning via reshape, motion reconstruction via np.add.at, interpolation via np.interp).

ii.
```python
# Session loop (could potentially be parallelized but not vectorized)
for idx, session in enumerate(sessions, start=1):
    converted, summary = process_session(session, show_processing=do_plot)

# Trial splitting loop (relatively cheap, iterates over 10-15 trials)
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop])
```

iii. The AI documented that vectorized operations were used where possible: "Vectorized motion-frame reconstruction using np.add.at and np.interp" and "Vectorized non-overlapping bin averaging via reshape/mean."

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each session is processed exactly once. The Suite2p baseline correction, motion reconstruction, binning, normalization, and discretization each happen once per session. The trial splitting loop does lightweight array slicing.

ii.
```python
# Each step in process_session() runs once per session:
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES)
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

iii. The AI did not document any repeated processing, which is consistent with the code structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores per-session summary statistics (neuron counts, frame counts, timing, quintile edges) in the metadata, which are informational but not used by the decoder. The motion trace diagnostic info (number of missing frames, frame dt) is used only for logging and optional plots. The validation step (`validate_converted_session`) performs checks that are discarded after passing.

ii.
```python
summary = {
    "subject": session.subject,
    "session_id": session.session_id,
    "n_neurons": int(F.shape[0]),
    "n_frames_raw": int(F.shape[1]),
    # ... many other fields
    "motion_quintile_edges": motion_edges.tolist(),
    "processing_seconds": float(elapsed),
}
```

iii. The AI included these as diagnostic/documentation features. They do not affect the decoder's operation but add to the pickle file size and processing time marginally. The optional `--show-processing` plots are the most compute-heavy unnecessary feature but are only generated when explicitly requested.
