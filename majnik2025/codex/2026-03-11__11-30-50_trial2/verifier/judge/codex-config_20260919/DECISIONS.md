# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted `jm*` subject directories and sorted date-like session directories. For every session it loads Suite2p `F.npy`, `Fneu.npy`, and `ops.npy`, plus behavioral `motion_energy_glob.npy` and `tstamps.npy`. Full mode processes all 41 discovered sessions; sample mode deliberately selects only the last two.

ii.
```python
def discover_sessions(data_root: Path) -> list[SessionRef]:
    sessions: list[SessionRef] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
        ):
            sessions.append(SessionRef(subject=subject_dir.name,
                                       session_id=session_dir.name,
                                       path=session_dir))
    return sessions

F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
```

iii. The notes say the top level contains six mice and 41 daily sessions, that these files are the already tracked post-Track2p data, and that all 41 sessions meet the paper's minimum-six-days criterion. Sorting provides deterministic order; restricting session names avoids unrelated subject-level directories.

## 1-b. How are the data split into subjects?

i. A subject is a top-level directory whose name begins with `jm`. Unique names are sorted, and each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({session.subject for session in session_refs})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.array(
    [subject_to_idx[session.subject] for session in session_refs], dtype=np.int64
),
```

iii. The agent documented that the six `jm*` directories are the six mice and verified the converted per-subject session counts against the source data.

## 1-c. How are the data split into sessions?

i. Each date-like subdirectory under a subject is one daily recording session. Sessions are ordered first by sorted subject and then by sorted directory name; each becomes one outer entry in `neural`, `input`, and `output`.

ii.
```python
for session_dir in sorted(
    p for p in subject_dir.iterdir()
    if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
):
    sessions.append(SessionRef(subject=subject_dir.name,
                               session_id=session_dir.name,
                               path=session_dir))

"neural": [session["neural"] for session in converted_sessions],
```

iii. The notes identify each `YYYY-MM-DD_a` folder as one day and report exact agreement between the 41 raw and converted sessions.

## 1-d. How are the data split into trials?

i. The agent treats the continuous recording as consecutive, non-overlapping **120-second** blocks after 10-frame binning. Each trial has 360 bins, and any tail shorter than a full block is discarded. This follows the paper's decoding block size but conflicts with the user's explicit instruction to use 60-second trials.

ii.
```python
TRIAL_SECONDS = 120.0
bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
for start in range(0, usable_bins, bins_per_trial):
    stop = start + bins_per_trial
    neural_trials.append(neural_binned[:, start:stop])
    input_trials.append(time_binned_s[np.newaxis, start:stop])
    output_trials.append(output_one_hot[:, start:stop])
```

iii. The notes explicitly say the agent chose two-minute blocks to match the paper's decoding split unit, yielding 10 or 15 trials for 20- or 30-minute sessions. It prioritized that paper detail over the decoder task's explicit 60-second requirement.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial filtering. A session must yield at least two trials, shapes and trial counts must agree, arrays must contain no NaNs, and output columns must be valid one-hot labels. Incomplete final blocks are dropped.

ii.
```python
if n_trials < 2:
    raise ValueError("Each converted session must contain at least 2 trials.")
if not (len(input_) == len(output) == n_trials):
    raise ValueError("Neural/input/output trial counts do not match.")
if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
    raise ValueError("Converted arrays must not contain NaN values.")
```

iii. The notes report that all sessions passed these checks and that 20- and 30-minute sessions produced complete blocks. No bad-trial criterion exists because the source is continuous spontaneous behavior rather than native trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `plane0/F.npy` and `plane0/Fneu.npy`; preprocessing parameters and sampling rate come from `plane0/ops.npy`.

ii.
```python
F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
```

iii. The agent says these are the paper's fluorescence and neuropil signals and that `ops.npy` preserves the session's Suite2p preprocessing settings.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil using the session's `ops['neucoeff']`, applies Suite2p `dcnv.preprocess` baseline correction with the remaining parameters from `ops`, converts to float32, then averages non-overlapping groups of 10 frames.

ii.
```python
Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
processed = suite2p_preprocess(
    Fc.copy(), baseline=ops["baseline"],
    win_baseline=float(ops["win_baseline"]),
    sig_baseline=float(ops["sig_baseline"]), fs=float(ops["fs"]),
    prctile_baseline=float(ops["prctile_baseline"]),
    batch_size=min(512, max(32, Fc.shape[0])), device=torch.device("cpu"),
)
neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES)
```

iii. The notes characterize this as paper-consistent baseline-corrected fluorescence and cite the Methods' 10-timestamp averaging. A direct recomputation sanity check reportedly matched converted neural values exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not apply a new `iscell` mask or neuron filter. It retains every row of `F`, assigning all rows to barrel cortex.

ii.
```python
"brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
```

iii. The agent found that the supplied arrays already contain only Suite2p-filtered cells successfully tracked across all days of a mouse: neuron counts are constant across days. It therefore judged reapplying `iscell` inappropriate and verified subject-level counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event. Neural bins are sliced at the start of each consecutive 120-second recording block. Metadata calls that block start the alignment event and gives offsets 0 to 120 seconds.

ii.
```python
neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))

"temporal_alignment_event": "start of each consecutive 2-minute recording block",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The notes explain that continuous spontaneous recordings have no native event, so artificial fixed blocks are the alignment unit. The two-minute event definition derives from the agent's trial-length choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged into non-overlapping bins, yielding 3 Hz or 333.33 ms per bin. Short tails below 10 frames are dropped.

ii.
```python
BIN_FRAMES = 10
usable = (n_frames // bin_frames) * bin_frames
x = x[..., :usable]
return x.reshape(x.shape[:-1] + (usable // bin_frames, bin_frames)).mean(axis=-1)

"time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
```

iii. The agent cites the paper's instruction to denoise fluorescence and behavior by averaging 10 consecutive timestamps and applies identical binning before trialization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from the binned sample index, `ops['fs']`, and the constant 10-frame bin width, rather than loaded from a timestamp file. It represents elapsed time from the start of each recording session.

ii.
```python
fs = float(ops["fs"])
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)
```

iii. The notes interpret “beginning of experiment” as beginning of the session and note that synchronous constant-rate acquisition makes the bin index an appropriate clock.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A zero-based index for every binned sample is multiplied by `10/fs` seconds and cast to float32. The absolute-within-session vector is then divided into trials without resetting at trial boundaries.

ii.
```python
time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. The notes say the vector should preserve absolute session time across all blocks; sanity checks found exact agreement with the expected 1/3-second sequence.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One time value is generated per neural 10-frame bin, and input and neural arrays are sliced using exactly the same `start:stop` indices. Validation requires matching time dimensions.

ii.
```python
neural_trials.append(neural_binned[:, start:stop])
input_trials.append(time_binned_s[np.newaxis, start:stop])
if neural[trial_idx].shape[1] != input_[trial_idx].shape[1]:
    raise ValueError("Input time dimension does not match neural time dimension.")
```

iii. The agent reports an exact raw-to-converted time-vector sanity check and no validation errors.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses the precomputed global video motion signal `motion_energy_glob.npy` and video timestamps `tstamps.npy`; the imaging-frame count from `F.npy` defines the target grid.

ii.
```python
motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
```

iii. The notes describe synchronized 30-Hz video and imaging, with occasional missing video frames. They choose timestamps because the data README recommends treating missing frames as missing or interpolating them.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion samples are mapped to a full imaging-frame grid from their timestamps; duplicate mappings are averaged and missing positions linearly interpolated. The trace is averaged over 10-frame bins, min-max normalized within session, divided into session quintiles, and one-hot encoded into five rows.

ii.
```python
frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
np.add.at(counts, frame_idx, 1)
full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)

motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0]
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

iii. The agent justifies interpolation as restoring framewise synchronization, binning as paper-matched denoising, normalization as within-session scaling, and one-hot encoding as compatibility with its interpretation of the decoder's multi-output interface.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four within-session 20th/40th/60th/80th-percentile edges are calculated after binning and normalization. `searchsorted(..., side='right')` gives classes 0–4, then each class is expanded to a five-row one-hot matrix. Thus the saved dataset describes five binary output variables rather than one five-valued categorical variable.

ii.
```python
edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
classes = np.searchsorted(edges, x, side="right").astype(np.int64)
one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T

"output_names": [f"motion_energy_q{i}" for i in range(N_OUTPUT_BINS)],
"output_values": [[f"not_q{i}", f"q{i}"] for i in range(N_OUTPUT_BINS)],
```

iii. The agent correctly sought equal-percentile bins selected per session, but believed the provided decoder required one-hot binary outputs for multiclass targets. Its notes emphasize exactly 20% positives in each row and valid one-hot columns.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Video timestamps are converted to imaging-grid indices spanning the imaging session. Missing grid samples are linearly interpolated. Neural and aligned motion are then independently averaged over the same non-overlapping groups of 10 frames and sliced with the same trial indices.

ii.
```python
frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)

output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. The notes say this uses the acquisition timestamps directly and report exact reproduction of a session with 116 missing motion frames, no remaining NaNs, and matching neural/output lengths.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are reconstructed by timestamp mapping and linear interpolation; duplicate timestamp mappings are averaged. Degenerate inputs raise errors, a single valid behavior sample is extended, and constant motion normalizes to zeros. Incomplete temporal tails are discarded. Per-session validation rejects too few trials, mismatched dimensions, NaNs, changing neuron counts, or invalid one-hot columns.

ii.
```python
if len(motion_energy) != len(tstamps):
    raise ValueError("motion_energy and tstamps must have the same length.")
if len(valid_idx) == 1:
    full[:] = full[valid_idx[0]]
else:
    full[missing] = np.interp(missing, valid_idx, full[valid_idx])
if xmax <= xmin:
    return np.zeros_like(x, dtype=np.float32)
usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
```

iii. The agent explicitly investigated all sessions with 1–148 missing video frames, reported no NaNs after conversion, and used assertions/validation rather than silently accepting shape errors.

## 6-a. What are the most time-consuming steps of the code?

i. The main cost is Suite2p fluorescence baseline preprocessing over every neuron and raw frame; file loading and optional eight-panel diagnostic plotting add secondary I/O/rendering costs.

ii.
```python
processed = suite2p_preprocess(..., device=torch.device("cpu"))

if show_processing:
    plot_processing_summary(...)
```

iii. The notes benchmarked two sessions at 2.26 seconds each and estimated about 1.54 minutes for 41 sessions. They identify fluorescence preprocessing as the substantive per-session computation and judged no further optimization necessary.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The important numerical operations are already vectorized: timestamp accumulation uses `np.add.at`, interpolation uses `np.interp`, and binning uses reshape/mean. The trial slicing loop and outer session loop remain, but they mostly create the required nested list structure and independently process sessions.

ii.
```python
np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
np.add.at(counts, frame_idx, 1)
full[missing] = np.interp(missing, valid_idx, full[valid_idx])
return x.reshape(new_shape).mean(axis=-1)

for start in range(0, usable_bins, bins_per_trial):
    neural_trials.append(neural_binned[:, start:stop])
```

iii. The notes specifically claim speedups from vectorized motion reconstruction and reshape-based averaging. They do not identify a remaining high-value loop to vectorize.

## 6-c. What processing does the code repeat multiple times?

i. Every session independently repeats file loading, fluorescence preprocessing, motion-grid reconstruction, binning, normalization, thresholding, time construction, and validation. Within plotting, some arrays are repeated or rescanned for percentiles, counts, and histograms, but only when requested. No session's core transform is inadvertently run twice during normal conversion.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    converted, summary = process_session(session, show_processing=do_plot)
    validate_converted_session(converted)

vmax = np.percentile(np.abs(trial_neural), 99)
counts = np.bincount(motion_classes, minlength=N_OUTPUT_BINS)
```

iii. The agent treats per-session repetition as necessary because preprocessing parameters, lengths, missing frames, normalization, and quintile edges are session-specific. Its notes do not flag redundant core computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes detailed timing and summary dictionaries, motion class arrays, quintile edges, missing-frame statistics, and—under `--show-processing`—large diagnostic figures. The summary data are stored in metadata but are not model features; `motion_classes` and several plot intermediates are used only for diagnostics. Min-max normalization is also mathematically unnecessary before percentile binning because it is monotonic (except its explicit constant-signal handling).

ii.
```python
motion_binned_norm = normalize_motion(motion_binned)
output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)

summary = {"motion_quintile_edges": motion_edges.tolist(),
           "processing_seconds": float(elapsed), ...}
if show_processing:
    plot_processing_summary(...)
```

iii. The notes justify summaries and plots as sanity checks and preserve summaries in metadata. They justify normalization as within-session scaling, but do not note that it cannot change percentile memberships. These diagnostics helped validate conversion but are discarded by decoder training.
