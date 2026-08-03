# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script first discovers sessions by scanning `data/` for subject folders whose names start with `jm` and session folders whose names begin with a date. It then processes each session independently by loading Suite2p neural files (`F.npy`, `Fneu.npy`, `ops.npy`) and behavioral files (`motion_energy_glob.npy`, `tstamps.npy`). It does not load `spks.npy`, `iscell.npy`, or `interframe_int.npy` during conversion, and it does not load any native trials because the source data are continuous recordings rather than trial-structured data.

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
suite2p_dir = session.path / "suite2p" / "plane0"
move_dir = session.path / "move_deve"

F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the agent justified this by stating that the provided dataset is already a post-Track2p export: Suite2p files contain the tracked cells and `move_deve/` contains the processed motion-energy signal, so conversion only needs those files.

## 1-b. How are the data split into subjects?

i. Subjects are defined by top-level folder names under `data/` such as `jm031` and `jm046`. The final dataset stores the sorted unique subject IDs and maps each session to a subject index.

ii.
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

iii. The notes say the data README documents six subject folders and that cross-referencing with the paper is done through these subject IDs, so the agent treated each `jm*` folder as one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are defined as dated subdirectories within each subject directory. They are sorted lexicographically and each becomes one session entry in the output.

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

iii. The agent’s notes say each subject folder contains one subfolder per recording day, named `YYYY-MM-DD_a`, and that the order of sessions in the converted dataset should follow this day-wise directory order.

## 1-d. How are the data split into trials?

i. The raw data are continuous recordings with no native trials. The agent therefore defines trials artificially as consecutive non-overlapping 2-minute blocks after 10-frame binning. Each session contributes as many full 2-minute blocks as fit into the usable binned recording.

ii.
```python
TRIAL_SECONDS = 120.0
...
bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
...
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes explicitly justify this with the paper’s decoding setup: the source data have no native trial structure, and the paper’s decoder operates on consecutive 2-minute temporal blocks.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter beyond structural checks. The script drops any partial trailing block that does not make a full 2-minute trial, errors out if a session is too short to yield one full trial, and validates that each converted session has at least two trials with matched time dimensions and no NaNs.

ii.
```python
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")
```

```python
if n_trials < 2:
    raise ValueError("Each converted session must contain at least 2 trials.")
...
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The notes say there are no native trial-curation rules in the source data or paper, so the agent treated trial curation as “keep full 2-minute blocks only” plus decoder-format validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` signal is derived from `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and baseline-related entries in `suite2p/plane0/ops.npy`. The script ignores `spks.npy` for the exported neural signal.

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
...
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
```

iii. In the notes, the agent said the paper described downstream analyses using baseline-corrected fluorescence rather than the saved `spks.npy`, so it chose to reconstruct a fluorescence-based neural signal from `F`, `Fneu`, and `ops`.

## 2-b. How is the `neural` data processed?

i. The script computes a fluorescence-derived signal by subtracting neuropil contamination using `ops["neucoeff"]`, then calling `suite2p.extraction.dcnv.preprocess` with baseline parameters read from `ops.npy`. After that, it averages the result in non-overlapping 10-frame bins.

ii.
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
        batch_size=min(512, max(32, Fc.shape[0])),
        device=torch.device("cpu"),
    )
    return processed.astype(np.float32, copy=False)
```

```python
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The notes justify this as “reconstructing a paper-consistent fluorescence representation” from Suite2p outputs. The trajectory also shows the agent consulted the Track2p GUI `F_processing` path and interpreted it as the closest available reference for a dF/F-like signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply new neuron-level quality filtering during conversion. It assumes the provided rows are already curated tracked cells and keeps all rows in `F.npy` for each session. It also assigns all neurons to one brain region.

ii.
```python
converted = {
    "neural": neural_trials,
    "input": input_trials,
    "output": output_trials,
    "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
    "summary": summary,
}
```

iii. The notes say the Track2p export already applied `iscell > 0.5` and retained only all-day matched neurons, and the data README says the provided matrices contain only successfully tracked neurons across all days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are not aligned to a stimulus or behavior onset because the recordings are continuous. Instead, after 10-frame binning, the neural trace is segmented into consecutive 2-minute blocks; the start of each block functions as the trial alignment point.

ii.
```python
"temporal_alignment_event": "start of each consecutive 2-minute recording block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

```python
neural_trials, input_trials, output_trials, trial_info = split_trials(
    neural_binned=neural_binned,
    time_binned_s=time_binned_s,
    output_one_hot=output_one_hot,
    fs=fs,
    bin_frames=BIN_FRAMES,
)
```

iii. The notes justify this by saying there is no native trial/event structure and that the paper’s decoding analysis uses consecutive 2-minute blocks rather than event-locked trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping temporal bins. With `fs = 30 Hz`, the bin width is `10 / 30 = 0.333... s`, i.e. `333.33 ms`.

ii.
```python
BIN_FRAMES = 10
...
def average_nonoverlapping(x: np.ndarray, bin_frames: int) -> np.ndarray:
    ...
    return x.reshape(new_shape).mean(axis=-1)
```

```python
"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
```

iii. The notes say this matches the paper’s decoding preprocessing, which averaged neural and behavioral traces in bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not taken from an explicit time file. It is derived from the binned sample index together with the imaging frame rate `ops["fs"]`.

ii.
```python
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

```python
fs = float(ops["fs"])
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
```

iii. The notes say the agent interpreted “time elapsed from the beginning of the experiment” as elapsed recording-session time and constructed it from the imaging frame count and frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script creates a monotonic time vector starting at zero seconds, sampled once per 10-frame bin, then slices that vector into the same 2-minute trial blocks as the neural data.

ii.
```python
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

```python
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. The notes justify this as the simplest task-required decoder input that remains synchronized with the 10-frame-binned neural trace.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is aligned by construction: the number of time bins is matched to `neural_binned.shape[1]`, and both neural and input arrays are cut with the same `start:stop` trial boundaries.

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
neural_trials, input_trials, output_trials, trial_info = split_trials(
    neural_binned=neural_binned,
    time_binned_s=time_binned_s,
    output_one_hot=output_one_hot,
    fs=fs,
    bin_frames=BIN_FRAMES,
)
```

```python
if neural[trial_idx].shape[1] != input_[trial_idx].shape[1]:
    raise ValueError("Input time dimension does not match neural time dimension.")
```

iii. The agent’s notes explicitly say the time input should be “carried through each 2-minute trial as an absolute-within-session time vector” and that a raw-data `np.allclose` sanity check confirmed exact alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
...
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
```

iii. The notes cite the data README and methods text: `motion_energy_glob.npy` is the processed behavioral motion-energy signal, while `tstamps.npy` identifies dropped camera frames and supports alignment to imaging.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script projects motion samples onto the imaging-frame grid using timestamps, averages multiple samples that map to the same frame, linearly interpolates missing frames, averages the full trace in non-overlapping 10-frame bins, and then min-max normalizes the binned trace within each session.

ii.
```python
frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)
...
np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
np.add.at(counts, frame_idx, 1)
...
full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
motion_binned_norm = normalize_motion(motion_binned)
```

iii. The notes justify this with two sources: the paper/methods define motion energy as the behavioral variable, and the data README explicitly says missing camera frames can be treated as missing or interpolated over.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After per-session min-max normalization, the script computes the 20th, 40th, 60th, and 80th percentiles, assigns each time bin to one of five bins with `np.searchsorted`, and exports the result as five one-hot binary channels.

ii.
```python
def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. The notes say this was driven directly by the task, which required motion energy to be normalized and discretized into five equal-percentile bins, with one-hot export chosen for compatibility with the provided decoder.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first reconstructed on the same imaging-frame grid used by the neural data, then binned with the same 10-frame averaging and split into trials using the same 2-minute block boundaries.

ii.
```python
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
...
neural_trials, input_trials, output_trials, trial_info = split_trials(
    neural_binned=neural_binned,
    time_binned_s=time_binned_s,
    output_one_hot=output_one_hot,
    fs=fs,
    bin_frames=BIN_FRAMES,
)
```

iii. The notes say timestamp-based reconstruction plus shared binning/trialization was the critical alignment choice, and the agent later documented an `np.allclose` check against recomputed raw motion traces.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion frames are interpolated after identifying absent imaging-frame indices via `tstamps.npy`. Constant motion traces normalize to zeros rather than dividing by zero. Partial trailing data that cannot make a full bin or full 2-minute trial are silently truncated. Sessions too short to form even one trial raise an error, and converted outputs are checked for NaNs.

ii.
```python
if len(missing):
    full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
if xmax <= xmin:
    return np.zeros_like(x, dtype=np.float32)
```

```python
usable = (n_frames // bin_frames) * bin_frames
...
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
```

iii. The notes say the interpolation choice came from the data README’s recommendation for dropped camera frames, while the truncation/error behavior came from the need to guarantee a decoder-valid rectangular session-by-trial structure.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work is per-session fluorescence preprocessing and file loading. The notes specifically single out Suite2p-style fluorescence preprocessing as the step worth benchmarking; optional plotting is also heavier than the purely vectorized motion and binning stages.

ii.
```python
neural_processed = compute_fluorescence_signal(F, Fneu, ops)
```

```python
for idx, session in enumerate(sessions, start=1):
    converted, summary = process_session(session, show_processing=do_plot)
```

iii. In Step 6 and Step 7 of the notes, the agent wrote that fluorescence preprocessing needed benchmarking and concluded that the existing vectorization made the full run fast enough without further optimization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining obvious Python loops are trial assembly in `split_trials`, per-trial validation in `validate_converted_session`, and the top-level per-session conversion loop. The agent already vectorized frame-to-bin averaging and motion reconstruction, so the residual loops are mostly list-building and validation.

ii.
```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

```python
for trial_idx in range(n_trials):
    ...
```

iii. The notes explicitly say the agent added vectorized `np.add.at`, `np.interp`, and reshape-based binning, implying that these smaller remaining loops were left in place because runtime was already acceptable.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly converts arrays to `float32`, repeatedly slices trial blocks for neural/input/output, and recomputes diagnostics such as motion class counts and quintile edges per session. Outside the main converter, the agent also repeated raw-data recomputation in separate validation checks documented in the notes.

ii.
```python
Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
...
motion_energy = motion_energy.astype(np.float32, copy=False)
tstamps = tstamps.astype(np.float64, copy=False)
```

```python
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
...
"motion_quintile_edges": motion_edges.tolist(),
```

iii. The notes describe repeated raw-data `np.allclose` recomputation as a deliberate validation step, not part of the main converter’s output path.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several artifacts that the downstream decoder does not need: `motion_classes` and `motion_edges` are used only for plots/metadata summaries, and the plotting path computes preview figures that are not part of the final dataset. There is also a computed `vmax` value in the plotting code that is never used.

ii.
```python
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
...
"motion_quintile_edges": motion_edges.tolist(),
```

```python
vmax = np.percentile(np.abs(trial_neural), 99)
vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else vmax
ax[2, 0].imshow(trial_neural[: min(80, trial_neural.shape[0])], aspect="auto", cmap="viridis")
```

iii. The notes justify the extra summaries and plots as sanity-check tooling, but they are not consumed by `train_decoder.py` and are therefore auxiliary rather than necessary for downstream analysis.
