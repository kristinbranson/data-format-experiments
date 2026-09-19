# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by scanning `data/` for directories whose names start with `jm`. It discovers sessions by scanning each subject directory for date-like subdirectories, then processes each session by loading `F.npy`, `Fneu.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. Trials are not loaded from disk because the source data are continuous; they are created later by splitting processed session-level arrays.

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
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says the provided `data/` folder is already the post-Track2p export and that each subject/session should therefore be loaded directly from the suite2p and `move_deve` files instead of rerunning tracking.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories whose names start with `jm`, sorted lexicographically.

ii.
```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
    ...
    SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
```

```python
subjects = sorted({session.subject for session in session_refs})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI justifies this in `CONVERSION_NOTES.md` by noting that the dataset organization uses one `jm*` directory per mouse.

## 1-c. How are the data split into sessions?

i. Sessions are split by subdirectories within each subject directory whose names look like dated recordings. Each such directory is treated as one session.

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(
        SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
    )
```

iii. The AI’s notes say each dated folder is one daily recording session and that all 41 such sessions should be included.

## 1-d. How are the data split into trials?

i. The AI does not use 60-second trials from the task instructions. It instead splits each continuous session into consecutive non-overlapping 2-minute blocks after temporal binning. Any trailing bins that do not fill a full 2-minute block are discarded.

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
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says it chose 2-minute blocks because the paper’s decoder used consecutive 2-minute blocks and it wanted to “match the paper’s decoding split unit exactly.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-quality filtering. It only requires that a session be long enough to produce at least one 2-minute block, and later validates that each converted session has at least two trials. Extra tail bins that do not fill a complete trial are dropped.

ii.
```python
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")
```

```python
if n_trials < 2:
    raise ValueError("Each converted session must contain at least 2 trials.")
```

iii. The AI’s notes do not claim any trial-quality curation rule beyond structural validity; it treats the recordings as continuous and trializes them mechanically.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from suite2p fluorescence outputs `F.npy` and `Fneu.npy`, using `ops.npy` only to obtain preprocessing parameters such as `neucoeff`, baseline settings, and frame rate.

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

```python
Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
```

iii. In the notes, the AI says the provided exports already contain the tracked neurons and that the paper’s downstream analyses used a Suite2p-style baseline-corrected fluorescence representation reconstructed from `F`, `Fneu`, and `ops`.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction, then runs `suite2p.extraction.dcnv.preprocess` with parameters taken from `ops.npy`, and finally averages the result into non-overlapping 10-frame bins.

ii.
```python
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
```

```python
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
```

iii. The AI justifies this in `CONVERSION_NOTES.md` as a paper-consistent reconstruction of “Suite2p-style neuropil-subtracted, baseline-corrected fluorescence.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply additional neuron filtering during conversion. It assumes the provided suite2p exports are already curated tracked-cell exports and keeps all rows.

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
...
converted = {
    "neural": neural_trials,
    ...
    "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
}
```

iii. The notes explicitly say the rows are already post-Track2p, already satisfy the `iscell` curation logic, and therefore should not be re-filtered.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to the start of each 2-minute recording block rather than to overall session start. In metadata it treats each block start as the temporal alignment event.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute recording block",
    "off_start": 0.0,
    "off_end": float(TRIAL_SECONDS),
    ...
}
```

iii. The AI’s notes justify this by tying trial alignment to its 2-minute block definition from the paper’s decoder analysis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses non-overlapping 10-frame averaging for both neural and motion streams. With 30 Hz raw sampling, this produces 333.33 ms bins.

ii.
```python
BIN_FRAMES = 10
```

```python
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(
    np.float32, copy=False
)
```

```python
"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
```

iii. The AI repeatedly cites the paper’s “10 consecutive timestamps” decoding preprocessing as justification.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not load a raw timestamp series for this input. It computes elapsed time from the binned frame index using the session frame rate from `ops["fs"]` and the bin width `BIN_FRAMES`.

ii.
```python
fs = float(ops["fs"])
...
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

iii. The AI’s notes say it interprets “time elapsed from the beginning of the experiment” as time from the beginning of the recording session.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a uniformly spaced vector in seconds for the full binned session, then slices that vector into the same trial blocks as the neural data.

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
...
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. In `CONVERSION_NOTES.md`, the AI says it carries an “absolute-within-session time vector” through each trial.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the time input by constructing it after neural binning and then slicing it with the same `start:stop` indices used for neural and output trial arrays.

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

iii. The AI’s sanity-check notes say it recomputed the elapsed-time vector directly from raw session length and confirmed an exact `np.allclose` match with the converted input.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion output from `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`. It does not use `interframe_int.npy`.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
...
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
```

iii. The notes say the AI chose timestamp-based reconstruction because the data README recommended treating missing camera frames as missing or interpolated and because it wanted to map behavior onto the imaging frame grid explicitly.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI maps the raw motion-energy samples onto the imaging-frame grid using `tstamps`, averages duplicate frame assignments, linearly interpolates missing frames, averages the aligned trace into 10-frame bins, normalizes the binned trace within session to `[0, 1]`, then discretizes it.

ii.
```python
frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
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
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

iii. The notes justify this as a way to respect the synchronized 30 Hz acquisition while explicitly repairing missing camera frames and producing session-wise normalized quintiles for the decoder.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes within-session quintile thresholds at the 20th, 40th, 60th, and 80th percentiles of the normalized binned motion trace. It assigns each time bin to one of five classes and then converts those class labels into a 5-row one-hot representation.

ii.
```python
def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

iii. In the notes, the AI says it used session-wise quintiles because the task required 5 equal-percentile bins and exported one-hot channels because it believed the provided decoder expected binary outputs for multiclass targets.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion to neural data by reconstructing a full-length motion trace on the imaging-frame grid, then applying the same 10-frame averaging and the same trial slicing as for neural activity.

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

iii. The AI’s notes explicitly justify this as behavior reconstruction “on the imaging grid from timestamps” followed by interpolation of only the missing frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing motion frames by reconstructing a full-length motion trace and linearly interpolating frame positions with no assigned sample. It also trims leftover bins that do not fill a complete trial and raises errors for degenerate cases such as malformed motion arrays or sessions too short to form a trial.

ii.
```python
if len(motion_energy) != len(tstamps):
    raise ValueError("motion_energy and tstamps must have the same length.")
...
missing = np.flatnonzero(~valid)
if len(missing):
    full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
if usable_bins < bins_per_trial:
    raise ValueError("Session is too short to form even one 2-minute trial.")
```

iii. The notes say the AI chose interpolation because the data documentation said dropped camera frames should be treated as missing or interpolated.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p fluorescence preprocessing in `compute_fluorescence_signal`, which runs per session over every neuron and frame. Optional plotting is also extra work when `--show-processing` is enabled.

ii.
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
if show_processing:
    plot_processing_summary(...)
```

iii. The AI’s notes explicitly benchmark conversion time and describe the fluorescence preprocessing path as the main operation worth timing and optimizing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes the heavy inner work for motion reconstruction and temporal binning. The main remaining Python loop is the trial-assembly loop in `split_trials`, which could be replaced with reshaping or indexed views, although the payoff is probably modest compared with the already-vectorized per-frame processing.

ii.
```python
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
    input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
    output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes claim the important speedups were already added: vectorized `np.add.at` motion reconstruction and reshape-based bin averaging. There is no further explicit vectorization discussion beyond that.

## 6-c. What processing does the code repeat multiple times?

i. Within the script itself, there is little redundant recomputation. Each session is processed once through the main conversion path. The only notable repeated work is the repeated shape/NaN/output-validity checking across trials during validation and the reuse of already-computed motion statistics for plotting and summaries.

ii.
```python
for trial_idx in range(n_trials):
    if neural[trial_idx].shape[0] != n_neurons:
        raise ValueError("Neuron count changed across trials within a session.")
    ...
    if not np.all(output[trial_idx].sum(axis=0) == 1):
        raise ValueError("Each output timepoint must belong to exactly one motion quintile.")
```

iii. The AI does not explicitly justify redundant processing here; the notes mostly argue that the expensive parts were already vectorized and the runtime was acceptable.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does some work that is not needed by the final decoder inputs/outputs: it normalizes motion before computing within-session quantiles even though a monotonic rescaling does not change quantile assignments; it computes and stores `motion_classes` and `motion_edges` mainly for summaries/plots; and in `--show-processing` mode it spends time generating diagnostic figures that are not used downstream.

ii.
```python
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

```python
if show_processing:
    plot_processing_summary(
        ...
        motion_binned_norm=motion_binned_norm,
        motion_classes=motion_classes,
        ...
        motion_edges=motion_edges,
        ...
    )
```

iii. The AI’s notes justify this extra work as documentation and decoder-oriented preprocessing, but they do not argue that normalization is necessary for within-session quantile binning itself.
