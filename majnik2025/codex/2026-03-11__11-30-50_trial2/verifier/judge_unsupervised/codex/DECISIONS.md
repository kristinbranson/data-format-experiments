# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers all subject folders under `data/` whose names start with `jm`, then all dated session folders inside each subject. For each session it loads neural arrays from `suite2p/plane0` and behavior arrays from `move_deve`, processes the whole continuous session, and only then splits it into trial blocks.

ii. <Code snippets>

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
suite2p_dir = session.path / "suite2p" / "plane0"
move_dir = session.path / "move_deve"

F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The notes say the provided dataset is already organized as subject folders containing daily session folders, with suite2p traces and motion-energy files in fixed subdirectories. The trajectory shows the agent decided the source recordings were continuous rather than trialized, so it first loads complete sessions and derives trials later.

## 1-b. How are the data split into subjects?

i. Subjects are defined by top-level folder names such as `jm031` and `jm046`. The final dataset keeps a sorted unique subject list and assigns each session a `subject_idx` based on its folder.

ii. <Code snippets>

```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

```python
subjects = sorted({session.subject for session in session_refs})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[session.subject] for session in session_refs], dtype=np.int64),
```

iii. In `CONVERSION_NOTES.md`, the agent documented six subject folders and treated the folder name as the mouse identifier. That matches the data README, which says each subject folder corresponds to one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are defined by dated subfolders inside each mouse folder. The session key is the directory name like `2023-10-18_a`, and sessions are processed in chronological sorted order.

ii. <Code snippets>

```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

```python
"session_ids": [f"{session.subject}/{session.session_id}" for session in session_refs],
```

iii. The notes explicitly say each subject folder contains daily session folders named `YYYY-MM-DD_a`, and the trajectory shows the agent used chronological session ordering to preserve the longitudinal sequence.

## 1-d. How are the data split into trials?

i. The source data have no native trials. The agent defines each trial as a consecutive 2-minute block from the continuous session after 10-frame binning. Each trial therefore contains 360 time bins at 30 Hz / 10-frame averaging.

ii. <Code snippets>

```python
BIN_FRAMES = 10
TRIAL_SECONDS = 120.0
```

```python
def split_trials(...):
    bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
    usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
    ...
    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
        input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
        output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes say the paper’s decoding analysis used consecutive 2-minute blocks and that the source recordings are continuous spontaneous behavior sessions. The trajectory also says the agent chose 2-minute blocks specifically to avoid inventing an arbitrary trial structure.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral or motion-based trial rejection. Trials are filtered only by structural constraints: the session must be long enough to contain at least one 2-minute block, trailing bins that do not fill a full block are discarded, and converted sessions must end up with at least two trials for decoder evaluation.

ii. <Code snippets>

```python
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")

neural_binned = neural_binned[:, :usable_bins]
time_binned_s = time_binned_s[:usable_bins]
output_one_hot = output_one_hot[:, :usable_bins]
```

```python
if n_trials < 2:
    raise ValueError("Each converted session must contain at least 2 trials.")
```

iii. The notes say there are no native trial-curation rules in the source paper beyond using consecutive 2-minute blocks. The agent therefore treated trial QC mainly as format validation rather than scientific rejection criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence traces `F.npy` and `Fneu.npy`, using Suite2p options in `ops.npy`. The agent does not use `spks.npy` in the final export.

ii. <Code snippets>

```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

```python
Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
processed = suite2p_preprocess(
    Fc.copy(),
    baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]),
    fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    ...
)
```

iii. The notebook in `/app/data/load_data.ipynb` says raw `F` can be loaded directly, but for proper analysis users should compute dF/F as described in the paper or use `spks.npy`. The notes and trajectory show the agent chose the fluorescence route because it believed the paper’s decoder used dF/F-like traces.

## 2-b. How is the `neural` data processed?

i. The agent computes neuropil-subtracted fluorescence `Fc = F - neucoeff * Fneu`, applies `suite2p.extraction.dcnv.preprocess` using baseline parameters from `ops.npy`, converts to `float32`, and then averages the result in non-overlapping 10-frame bins.

ii. <Code snippets>

```python
def compute_fluorescence_signal(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    """Approximate the paper's Suite2p-based dF/F signal."""
    Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
    processed = suite2p_preprocess(
        Fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        ...
    )
    return processed.astype(np.float32, copy=False)
```

```python
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The notes describe this as reconstructing a paper-consistent fluorescence representation from the exported Suite2p files, and explicitly say the agent did not want to use raw `F` directly because it interpreted the paper as using baseline-corrected fluorescence.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script performs no additional neuron-level filtering itself. It assumes the supplied suite2p exports already contain only tracked cells curated upstream by Track2p/Suite2p, and it keeps every row in `F.npy`. The only enforced checks are shape consistency and absence of NaNs after conversion.

ii. <Code snippets>

```python
converted = {
    "neural": neural_trials,
    "input": input_trials,
    "output": output_trials,
    "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
    "summary": summary,
}
```

```python
for trial_idx in range(n_trials):
    if neural[trial_idx].shape[0] != n_neurons:
        raise ValueError("Neuron count changed across trials within a session.")
    ...
    if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
        raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The notes say the provided data already contain only cells present across all days and that constant neuron counts across days support this. The agent therefore decided not to rerun `iscell` filtering or cross-day matching.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data to the start of each derived 2-minute recording block, not to a stimulus or behavior event. The metadata names the alignment event as the start of each consecutive 2-minute block, and trials begin at offset 0 relative to that block start.

ii. <Code snippets>

```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
```

```python
"temporal_alignment_event": "start of each consecutive 2-minute recording block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The notes say there is no native event structure in the source data, so the agent used the paper’s 2-minute decoder block as the alignment anchor. The trajectory repeats that the recordings are continuous and needed a derived trialization scheme.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping temporal bins at 30 Hz, giving a bin size of 333.33 ms. Yes, both neural and behavioral streams are rebinned.

ii. <Code snippets>

```python
BIN_FRAMES = 10
```

```python
def average_nonoverlapping(x: np.ndarray, bin_frames: int) -> np.ndarray:
    ...
    return x.reshape(new_shape).mean(axis=-1)
```

```python
"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
"raw_frame_rate_hz": 30.0,
"bin_size_frames": BIN_FRAMES,
```

iii. The notes explicitly cite the paper’s “averaging in bins of 10 consecutive timestamps” and 30 Hz acquisition as the reason for using 333.3 ms bins.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not loaded from a raw timestamp file. It is derived from the bin index after neural rebinning together with `ops["fs"]` and the chosen `BIN_FRAMES=10`.

ii. <Code snippets>

```python
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

```python
fs = float(ops["fs"])
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
```

iii. In the notes, the agent mapped the decoder input to “time from session start in seconds” and treated that as the intended meaning of “time from start of experiment.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent creates a monotonic time vector in seconds using `np.arange(n_bins) * (bin_frames / fs)`, then slices that vector into the same 2-minute blocks used for neural and output data.

ii. <Code snippets>

```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
...
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. The notes say this was intended to preserve absolute-within-session elapsed time after the 10-frame averaging step. The trajectory shows the agent chose session elapsed time rather than a separate event-coded input.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time vector is generated at the same binned sampling rate as the neural data and is cut with the same start/stop indices during trialization, so each timepoint matches the corresponding neural bin within a trial.

ii. <Code snippets>

```python
neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

```python
if neural[trial_idx].shape[1] != input_[trial_idx].shape[1]:
    raise ValueError("Input time dimension does not match neural time dimension.")
```

iii. The notes include a planned sanity check that the converted input time values equal the expected elapsed-time vector sampled every 10 imaging frames, showing that alignment was deliberate rather than incidental.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/tstamps.npy` used to place those samples on the imaging-frame grid.

ii. <Code snippets>

```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
```

iii. The notes and the data README both describe `motion_energy_glob.npy` as processed behavioral motion energy and `tstamps.npy` as the source for locating missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent maps motion-energy samples onto the imaging-frame grid, averages duplicate assignments, linearly interpolates missing frames, bins the aligned trace in non-overlapping 10-frame windows, and then min-max normalizes the binned trace within each session.

ii. <Code snippets>

```python
frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)
...
np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
np.add.at(counts, frame_idx, 1)
...
if len(missing):
    full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
motion_binned_norm = normalize_motion(motion_binned)
```

iii. The notes say missing camera frames had to be handled explicitly and that the paper’s decoding used 10-frame averaging. The data README also explicitly allows interpolation over missing motion frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After session-wise normalization, the agent computes the 20th, 40th, 60th, and 80th percentiles of motion energy, assigns each time bin to one of five quantile classes, and exports the result as five one-hot binary rows rather than a single 5-level categorical row.

ii. <Code snippets>

```python
def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

```python
"output_names": [f"motion_energy_q{i}" for i in range(N_OUTPUT_BINS)],
"output_values": [[f"not_q{i}", f"q{i}"] for i in range(N_OUTPUT_BINS)],
```

iii. The notes say the required user-facing target was five equal-percentile bins, but the trajectory also states the agent made the one-hot choice because the provided decoder handled multi-class outputs more safely that way.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first reconstructed on the full imaging-frame grid using `tstamps.npy`, then downsampled with the same 10-frame averaging as neural data, and finally split into trials with the exact same bin boundaries as the neural traces.

ii. <Code snippets>

```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
...
neural_trials, input_trials, output_trials, trial_info = split_trials(
    neural_binned=neural_binned,
    time_binned_s=time_binned_s,
    output_one_hot=output_one_hot,
    fs=fs,
    bin_frames=BIN_FRAMES,
)
```

```python
if neural[trial_idx].shape[1] != output[trial_idx].shape[1]:
    raise ValueError("Output time dimension does not match neural time dimension.")
```

iii. The notes describe behavior alignment as one of the critical choices and say the goal was to align video-derived motion to imaging frames before any binning or trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior frames are detected implicitly through unmatched imaging-frame indices and filled by linear interpolation. Duplicate behavior samples mapping to the same imaging frame are averaged. Partial trailing blocks are dropped during trialization. Sessions that are too short or contain NaNs after conversion raise errors.

ii. <Code snippets>

```python
full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
...
valid = counts > 0
full[valid] = (sums[valid] / counts[valid]).astype(np.float32)
...
missing = np.flatnonzero(~valid)
if len(missing):
    full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
...
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The data README explicitly says missing camera frames can be treated as missing or interpolated over. The notes say the agent chose interpolation and made that a documented discrepancy-resolution step.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work is session-by-session fluorescence preprocessing with `suite2p_preprocess`, loading large `F`/`Fneu` arrays, and optional plotting. The notes focus their runtime concern on benchmarking the Suite2p-style preprocessing step.

ii. <Code snippets>

```python
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
```

```python
for idx, session in enumerate(sessions, start=1):
    ...
    converted, summary = process_session(session, show_processing=do_plot)
```

iii. `CONVERSION_NOTES.md` says the agent needed to benchmark fluorescence preprocessing before deciding if more optimization was required, which indicates this was viewed as the dominant runtime cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loop is the per-trial slicing loop in `split_trials`. The per-session top-level loop is also serial. Within a session, trial extraction could have been reshaped into `(n_trials, ...)` blocks instead of appending one slice at a time.

ii. <Code snippets>

```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

```python
for idx, session in enumerate(sessions, start=1):
    ...
    converted, summary = process_session(session, show_processing=do_plot)
```

iii. The notes mention vectorized motion reconstruction and reshape-based binning as explicit speedups, which implies the agent knew some loops mattered but left the trial-splitting/session loop structure unchanged.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts arrays to `float32`/`int64`, computes per-session summaries for every session, and reruns related session processing for optional diagnostics. It also validates every converted session separately even though some checks overlap with the final decoder verification.

ii. <Code snippets>

```python
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
...
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

```python
validate_converted_session(converted)
...
"session_info": [session["summary"] for session in converted_sessions],
```

iii. The notes emphasize correctness and sanity checks after each step, so some repeated casting and validation appears to reflect a defensive implementation style rather than a minimal runtime design.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script optionally generates large diagnostic plots, computes `motion_classes` and `motion_edges` partly for visualization and summaries, stores verbose per-session metadata, and calculates processing summaries that are not required by the decoder itself. Those outputs are useful for auditing but not used during downstream model fitting.

ii. <Code snippets>

```python
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

```python
if show_processing:
    plot_processing_summary(
        ...
        motion_classes=motion_classes,
        ...
        motion_edges=motion_edges,
        motion_info=motion_info,
    )
```

```python
summary = {
    "subject": session.subject,
    "session_id": session.session_id,
    ...
    "motion_quintile_edges": motion_edges.tolist(),
    "processing_seconds": float(elapsed),
}
...
"session_info": [session["summary"] for session in converted_sessions],
```

iii. The notes repeatedly frame these as sanity-check artifacts and documentation aids. The trajectory’s final message also points to verification outputs and cached investigation artifacts, confirming that the agent intentionally kept extra audit information beyond what the decoder strictly needs.
