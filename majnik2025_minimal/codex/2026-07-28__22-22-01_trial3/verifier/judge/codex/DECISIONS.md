# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all subject directories under the data root, then all session subdirectories within each subject. For each session it loads Suite2p calcium files (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`) and behavioral files (`motion_energy_glob.npy`, `interframe_int.npy`). Trials are not loaded directly from disk; the code loads each session as a continuous recording first and splits it into trials later.

ii. 
```python
def session_paths(data_dir: Path, selected_subjects: list[str] | None) -> list[tuple[str, Path]]:
    available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    ...
    for subject in subjects:
        subject_dir = data_dir / subject
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            out.append((subject, session_dir))

def load_session(subject: str, session_dir: Path) -> dict:
    plane_dir = session_dir / "suite2p" / "plane0"
    move_dir = session_dir / "move_deve"

    fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
    motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
```

iii. In the trajectory, the AI said the dataset was "arranged per mouse and day with Suite2p outputs plus motion-energy traces" and later summarized the structure as one matched-cell session per recording day. It explicitly chose to preprocess full-session traces first and split only after loading and alignment.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined as top-level directories in the data root. The code sorts all directories alphabetically; it does not explicitly require the `jm*` prefix.

ii. 
```python
available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
if selected_subjects is None:
    subjects = available_subjects
else:
    ...
    subjects = sorted(selected_subjects)
```

iii. In the trajectory, the AI inspected the root layout and reported subjects like `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`, concluding the dataset was organized "per mouse and day." There was no separate justification for omitting an explicit `jm*` filter beyond relying on the directory structure actually present.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory within a subject directory, sorted alphabetically. The code treats each such directory as a separate recording session.

ii. 
```python
for subject in subjects:
    subject_dir = data_dir / subject
    for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        out.append((subject, session_dir))
```

iii. The AI justified this in the trajectory by describing the data as "per mouse and day" and later saying "each recording day is already one matched-cell session."

## 1-d. How are the data split into trials?

i. The AI defines trials artificially by cutting each binned session into consecutive non-overlapping 2-minute blocks. Trial count is determined after 10-frame temporal binning, with `TRIAL_BINS = 120 s / (10/30 s) = 360` bins per trial.

ii. 
```python
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)

def split_session_into_trials(
    neural_binned: np.ndarray,
    motion_classes: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ...
    n_trials = neural_binned.shape[1] // TRIAL_BINS

    for trial_idx in range(n_trials):
        start = trial_idx * TRIAL_BINS
        end = start + TRIAL_BINS
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
        output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. In the trajectory, the AI said it had "settled the main structural choices" and would "cut the recording into the same consecutive 2-minute blocks described in the paper." It repeated this in the script’s own summary output: "Trials: consecutive 2-minute blocks within each session."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Instead, the code requires the full binned session length to be exactly divisible by the 2-minute trial size and raises an error if it is not, rather than dropping an incomplete final trial.

ii. 
```python
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(
        f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
        f"{TRIAL_BINS}-bin trials"
    )
```

iii. The trajectory does not give a separate trial-QC rationale beyond the AI’s choice of fixed 2-minute blocks and its preference for validating format aggressively. Its summaries emphasized getting the validator to pass on the first pass rather than describing any trial rejection policy.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw Suite2p fluorescence (`F.npy`) and neuropil fluorescence (`Fneu.npy`). The code also reads `ops.npy` to obtain preprocessing parameters and `iscell.npy` to validate that tracked cells satisfy the expected confidence threshold.

ii. 
```python
fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
neural = suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops)
```

iii. In the trajectory, the AI explicitly said it had confirmed that the bundled data "only exposes raw `F.npy`" and that it needed to determine the exact `dF/F` or baseline-corrected trace definition before converting the data.

## 2-b. How is the `neural` data processed?

i. The AI computes neuropil-subtracted fluorescence `fc = F - neucoeff * Fneu`, then applies a hand-written Suite2p-style baseline subtraction routine using parameters loaded from `ops.npy`. For the dataset’s `maximin` baseline mode, it smooths with a Gaussian filter, applies a 1D minimum filter, then a 1D maximum filter, and subtracts that baseline. Afterward it averages neural data into non-overlapping 10-frame bins.

ii. 
```python
def suite2p_baseline_corrected_fluorescence(
    fluorescence: np.ndarray, neuropil: np.ndarray, ops: dict
) -> np.ndarray:
    fc = fluorescence.astype(np.float32) - float(ops["neucoeff"]) * neuropil.astype(np.float32)
    baseline = ops.get("baseline", "maximin")

    if baseline == "maximin":
        win = int(round(float(ops["win_baseline"]) * float(ops["fs"])))
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = minimum_filter1d(flow, win)
        flow = maximum_filter1d(flow, win)
    ...
    return (fc - flow).astype(np.float32)

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The AI justified this repeatedly in the trajectory: it said `ops.npy` carried Suite2p defaults such as `neucoeff=0.7`, `baseline=maximin`, and a 60 s baseline window, and it summarized the chosen neural signal as "Suite2p baseline-corrected fluorescence from Track2p-exported F/Fneu."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not remove neurons at conversion time. Instead, it checks that every ROI in `iscell.npy` has confidence above 0.5 and errors out otherwise. In effect, it assumes the exported Track2p session matrices already contain the tracked cell set.

ii. 
```python
iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
...
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
```

iii. The trajectory shows the AI initially thought it should "use Suite2p `iscell > 0.5`," then inspected the data and concluded the session matrices were "already Track2p-matched cell sets." It checked `iscell` values and recorded that all first-column flags were 1 and all probabilities exceeded 0.5, leading it to keep all exported neurons and merely validate the assumption.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to the start of a consecutive 2-minute block within the session, not to session start as a single continuous recording. The metadata explicitly names the alignment event as the start of each 2-minute block and sets offsets to `[0, 120]` seconds.

ii. 
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive 2-minute block within a session",
    "off_start": 0.0,
    "off_end": TRIAL_DURATION_SECONDS,
    ...
}
```

iii. In the trajectory, the AI wrote that it would "cut the recording into the same consecutive 2-minute blocks described in the paper" and later echoed this in the conversion summary. That is the explicit justification the AI gave for the alignment choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins at 30 Hz, giving a time bin size of `10 / 30 = 0.333...` s or `333.33` ms. The AI applies this rebinning to both neural and motion traces before trial splitting and motion discretization.

ii. 
```python
FRAME_RATE_HZ = 30.0
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0

def mean_bin_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    nbins = x.shape[1] // bin_size
    return x[:, : nbins * bin_size].reshape(x.shape[0], nbins, bin_size).mean(axis=2)

neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The AI cited the paper-level rule in the trajectory as "both neural plus behavior averaged in bins of 10 imaging frames before decoding," and its printed summary described this as "Denoising: non-overlapping averages over 10 consecutive frames for neural and motion traces."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw timestamp variable. It is synthesized from the binned sample index after temporal binning, using the known frame rate and 10-frame bin size.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
```

iii. In the trajectory and conversion summary, the AI described the decoder input as "elapsed time from session start at bin centers (seconds)." That indicates it intentionally derived time from bin indices rather than from `tstamps.npy` or another recorded timestamp stream.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a continuous per-session time vector after binning and uses bin centers rather than left bin edges by adding `0.5` before multiplying by the bin duration. No normalization or discretization is applied.

ii. 
```python
elapsed_time_seconds = (
    (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
)
...
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. The trajectory justification is brief but explicit: in its summaries the AI repeatedly described the input as "elapsed time from session start at bin centers (seconds)." It did not cite a raw-data timestamp source for this variable.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is generated on the same binned frame grid as the neural data and then sliced using the same `[start:end]` trial boundaries. This gives one scalar time per neural time bin.

ii. 
```python
n_trials = neural_binned.shape[1] // TRIAL_BINS
...
neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
```

iii. The AI justified alignment indirectly through its general pipeline description: preprocess full sessions, align motion to the imaging-frame grid, bin both streams, then split all arrays together into fixed blocks. The printed conversion summary likewise presents elapsed time as defined on the same trial grid as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived primarily from `move_deve/motion_energy_glob.npy`. The AI also uses `interframe_int.npy` to infer where camera frames were dropped so that motion energy can be aligned to the imaging frame count.

ii. 
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)
...
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy,
    n_imaging_frames=n_frames,
    interframe_int=interframe_int,
)
```

iii. In the trajectory, the AI explicitly said it was checking "how missing behavior frames are handled" and later summarized the choice as repairing "camera dropouts on the frame grid" before binning.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first aligns motion energy to the imaging frame count by inferring the number of missing camera frames from `interframe_int.npy` and filling those gaps with linear interpolation between neighboring observed motion-energy samples. It then averages the aligned trace into non-overlapping 10-frame bins and applies a per-session z-score normalization.

ii. 
```python
def infer_missing_frames(interframe_int: np.ndarray) -> np.ndarray:
    median_interval = float(np.median(interframe_int))
    frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
    return np.clip(frame_jumps - 1, 0, None)

...
aligned[dst : dst + gap_missing] = np.linspace(
    motion_energy[src],
    motion_energy[next_src],
    int(gap_missing) + 2,
    dtype=np.float32,
)[1:-1]

motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
motion_z = (
    (motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)
).astype(np.float32)
```

iii. The trajectory justification is explicit. The AI said some behavior traces were short because camera frames dropped, so it would "map those missing frames so alignment stays frame-accurate before binning." It later summarized the output processing as "session-z-scored motion energy discretized with global quintile edges."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI pools all per-session z-scored motion energy values across sessions, computes global quintile edges at 20%, 40%, 60%, and 80%, and then digitizes each session against those pooled global thresholds. It names the five categories as percentile ranges.

ii. 
```python
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
...
pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions], axis=0)
motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)
...
motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
...
"output_values": [[
    "lowest_20pct",
    "20_40pct",
    "40_60pct",
    "60_80pct",
    "highest_20pct",
]],
```

iii. The trajectory and script summary are explicit here: the AI described the decoder output as "session-z-scored motion energy discretized with global quintile edges," and its validation output printed the resulting global edges.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by reconstructing a frame-aligned motion trace with the same number of samples as the imaging data, using `interframe_int.npy` to infer missing video frames and linearly interpolate them. After this alignment, both motion and neural streams are binned with the same 10-frame averaging and then sliced with identical trial boundaries.

ii. 
```python
motion_aligned, motion_info = align_motion_to_imaging_frames(
    motion_energy=motion_energy,
    n_imaging_frames=n_frames,
    interframe_int=interframe_int,
)

if neural_binned.shape[1] != motion_binned.shape[0]:
    raise ValueError(f"{session_dir}: binned neural and motion lengths do not match")

neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))
```

iii. The AI justified this strongly in the trajectory: it said some behavior traces were short because camera frames dropped and that it was mapping the missing frames "so alignment stays frame-accurate." The printed summary restated this as "Behavior alignment: imaging-frame grid with interpolation over inferred dropped camera frames."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles short motion-energy traces by inferring missing camera frames from `interframe_int.npy` and interpolating those missing values. It validates several assumptions with hard errors: calcium and neuropil shapes must match, all exported cells must satisfy `iscell > 0.5`, binned neural and motion lengths must match, and session lengths must be exactly divisible into 2-minute trials. It records missing-frame counts in metadata rather than silently ignoring them.

ii. 
```python
if fluorescence.shape != neuropil.shape:
    raise ValueError(f"{session_dir}: F and Fneu shapes do not match")
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")
...
if missing_total != expected_missing:
    raise ValueError(
        f"Could not reconcile missing motion frames: inferred {missing_total}, "
        f"expected {expected_missing}"
    )
...
if neural_binned.shape[1] != motion_binned.shape[0]:
    raise ValueError(f"{session_dir}: binned neural and motion lengths do not match")
if motion_binned.shape[0] % TRIAL_BINS != 0:
    raise ValueError(...)
```

iii. The trajectory justification emphasized verification and repair rather than permissive handling: the AI said it was building sanity checks against the paper, wanted alignment to stay "frame-accurate," and used the validator to catch mistakes after conversion.

## 6-a. What are the most time-consuming steps of the code?

i. The heaviest final-script work is per-session preprocessing of full continuous recordings before any splitting: loading large `.npy` arrays, running the Suite2p-style baseline correction over all neurons and frames, and reconstructing motion traces on the imaging frame grid. Trial splitting and metadata assembly are comparatively lightweight.

ii. 
```python
loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
...
neural = suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops)
motion_aligned, motion_info = align_motion_to_imaging_frames(...)
neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
```

iii. The AI said in the trajectory that the "full conversion is running now; it’s heavier because I’m preprocessing every session before any splits are written." Earlier it also focused most of its prototyping effort on neural preprocessing and motion alignment rather than on later formatting steps.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized inner loop is the frame-gap reconstruction in `align_motion_to_imaging_frames`, which iterates across every interframe interval and fills missing samples gap by gap. The trial-splitting loop also repeatedly slices and appends trial arrays, though that is structurally simpler and less likely to dominate runtime.

ii. 
```python
for gap_missing in missing_after:
    next_src = src + 1
    if gap_missing:
        inserted_indices.extend(range(dst, dst + int(gap_missing)))
        aligned[dst : dst + gap_missing] = np.linspace(
            motion_energy[src],
            motion_energy[next_src],
            int(gap_missing) + 2,
            dtype=np.float32,
        )[1:-1]
        dst += int(gap_missing)
    aligned[dst] = motion_energy[next_src]
    dst += 1
    src = next_src

for trial_idx in range(n_trials):
    start = trial_idx * TRIAL_BINS
    end = start + TRIAL_BINS
    ...
```

iii. The trajectory does not contain a separate optimization discussion, but the AI’s own exploration centered on correctness rather than vectorization. The code structure makes the interpolation loop the clearest candidate for vectorization.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats subject-index lookup with `subjects.index(session["subject"])` for every session, casts trial slices with `astype(..., copy=False)` on each append, and performs a full per-session preprocessing pass before later iterating over all sessions again to discretize and assemble trials. The two-pass session handling is intentional but still repeated work over the session list.

ii. 
```python
subjects = sorted({session["subject"] for session in loaded_sessions})
...
for session in loaded_sessions:
    motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
    neural_trials, input_trials, output_trials = split_session_into_trials(...)
    ...
    subject_idx.append(subjects.index(session["subject"]))
```

iii. There is no explicit trajectory justification for this repetition. The AI’s stated priority was to preprocess sessions first, validate the result, and only then assemble the standardized dataset, which explains the two-pass structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed quantities are not used by the downstream decoder inputs/outputs: `inserted_indices` are collected only for metadata, `motion_binned` and `motion_z` are intermediate traces that are discarded after discretization, and extensive summary metadata (`tracked_neurons_mean`, per-session missing-frame counts, session info) are saved for reporting rather than for model training.

ii. 
```python
inserted_indices: list[int] = []
...
return aligned, {
    "missing_motion_frames": missing_total,
    "missing_motion_frame_indices": inserted_indices,
}
...
"motion_processing": {
    ...
    "global_quintile_edges_zscore": motion_bin_edges.tolist(),
},
...
"session_info": session_info,
```

iii. The AI explicitly wanted "sanity checks against reference materials" and added rich reporting to the script summary. The extra intermediate bookkeeping and metadata support those checks, but they are not needed for downstream decoding once the categorical output and neural/input trial matrices are produced.
