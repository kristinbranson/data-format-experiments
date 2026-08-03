# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over subject directories (folders starting with "jm") under `/app/data`, then iterates over session directories (folders whose name starts with 4 digits) within each subject. For each session, it loads Suite2p neural data (`F.npy`, `Fneu.npy`, `ops.npy`) from `suite2p/plane0/` and behavioral data (`motion_energy_glob.npy`, `interframe_int.npy`) from `move_deve/`. All subjects and sessions are collected in `collect_session_dirs()` and processed sequentially in `build_dataset()`.

ii.
```python
def sorted_subject_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("jm")]
    )

def sorted_session_dirs(subject_dir: Path) -> list[Path]:
    return sorted(
        [path for path in subject_dir.iterdir() if path.is_dir() and path.name[:4].isdigit()]
    )

def collect_session_dirs(root: Path, mode: str) -> list[Path]:
    session_dirs = []
    for subject_dir in sorted_subject_dirs(root):
        sessions = sorted_session_dirs(subject_dir)
        if mode == "sample":
            if sessions:
                session_dirs.append(sessions[0])
        else:
            session_dirs.extend(sessions)
    return session_dirs
```

iii. The AI noted in CONVERSION_NOTES.md that it inventoried 6 subjects and 41 sessions in the full dataset. It verified this matched the paper's description of 6 mice imaged daily for at least 6 consecutive days. The data loading notebook (`load_data.ipynb`) provided by the dataset authors uses the same directory traversal pattern.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the parent directory name of each session directory (e.g., `jm031`, `jm032`). A sorted list of unique subject names is built, and each session is mapped to its subject via `subject_to_index`. The `subject_idx` array indexes into the `subjects` list for each session.

ii.
```python
subjects = sorted({session_dir.parent.name for session_dir in session_dirs})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}
# ...
"subject_idx": np.array(
    [subject_to_index[record["subject"]] for record in session_records], dtype=np.int64
),
```

iii. The AI confirmed 6 subjects (jm031-jm046), matching the paper's description. Each subject folder contains 6-7 sessions, consistent with the paper's "at least 6 consecutive days" statement.

## 1-c. How are the data split into sessions?

i. Each recording day directory within a subject folder becomes one session. Sessions are sorted alphabetically (which is chronological since names are date-formatted). All sessions from all subjects are flattened into a single list, with `subject_idx` tracking which subject each session belongs to.

ii.
```python
for session_index, session_dir in enumerate(session_dirs):
    subject = session_dir.parent.name
    session_id = session_dir.name
    session_key = f"{subject}/{session_id}"
    # ... load and process session data ...
    session_records.append({
        "subject": subject,
        "session_id": session_id,
        # ...
    })
```

iii. The CONVERSION_NOTES state "Session unit: one recording day per session," matching the data README's description that "each session folder corresponds to one recording day."

## 1-d. How are the data split into trials?

i. Each session's continuous recording is split into consecutive non-overlapping 2-minute (3600-frame) blocks. Each block becomes one trial. Trailing frames that don't fill a complete 2-minute block are discarded. Sessions must have at least 2 trials.

ii.
```python
TRIAL_SECONDS = 120.0
TRIAL_FRAMES = int(TRIAL_SECONDS * FRAME_RATE_HZ)  # 3600

def session_to_trials(neural_full, motion_full, session_duration_frames):
    usable_frames = (session_duration_frames // TRIAL_FRAMES) * TRIAL_FRAMES
    if usable_frames < TRIAL_FRAMES:
        raise ValueError(...)
    n_trials = usable_frames // TRIAL_FRAMES
    if n_trials < 2:
        raise ValueError(...)
    # reshape and split into trials
```

iii. The CONVERSION_NOTES explain: "Each session is cut into consecutive 2-minute blocks because the paper's decoder uses consecutive 2-minute splits." The paper's methods state: "splits were done on consecutive 2 minute blocks of the recording."

## 1-e. How are trials filtered based on quality controls?

i. No individual trial quality filtering is applied. The only filter is that a session must yield at least 2 full 2-minute blocks. Trailing frames that don't complete a full block are dropped. No neuron-level or trial-level quality exclusion is performed beyond what Track2p already did (the provided data only contains successfully tracked neurons with iscell=1).

ii.
```python
if n_trials < 2:
    raise ValueError(f"Session only has {n_trials} trials after blocking; at least 2 are required")
```

iii. The CONVERSION_NOTES state that the bundled release only contains already-tracked cells, so the untracked ROIs are not available. The iscell arrays in the data all have values of 1.0, confirming no further cell filtering is needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from three files per session: `F.npy` (raw fluorescence traces), `Fneu.npy` (neuropil fluorescence traces), and `ops.npy` (Suite2p processing parameters including `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `fs`, `prctile_baseline`).

ii.
```python
def load_suite2p_dff(session_dir: Path) -> tuple[np.ndarray, dict]:
    plane_dir = session_dir / "suite2p" / "plane0"
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
    Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)
```

iii. The AI noted in CONVERSION_NOTES that it uses "Suite2p neuropil-subtracted fluorescence converted to dF/F using Suite2p default baseline estimation parameters stored in ops.npy."

## 2-b. How is the `neural` data processed?

i. Processing pipeline: (1) Neuropil subtraction: `Fc = F - neucoeff * Fneu`. (2) Baseline estimation via `suite2p.extraction.dcnv.preprocess()` with ops parameters. (3) Baseline recovery: `F0 = Fc - preprocess(Fc)`, clamped to minimum 1e-3. (4) dF/F computation: `preprocess(Fc) / F0`. (5) Temporal binning: average over non-overlapping 10-frame bins.

ii.
```python
Fcorr = F
Fcorr -= float(ops["neucoeff"]) * Fneu

dff_num = Fcorr.copy()
dff_num = dcnv.preprocess(
    F=dff_num,
    baseline=ops["baseline"],
    win_baseline=ops["win_baseline"],
    sig_baseline=ops["sig_baseline"],
    fs=ops["fs"],
    prctile_baseline=ops["prctile_baseline"],
    batch_size=ops.get("batch_size", 100),
    device=torch.device("cpu"),
)

baseline = Fcorr
baseline -= dff_num
np.maximum(baseline, 1e-3, out=baseline)
dff_num /= baseline
```

iii. The CONVERSION_NOTES explain this matches "the paper's statement that decoding used slightly denoised dF/F" and uses "Suite2p's own baseline-preprocessing code via suite2p.extraction.dcnv.preprocess(...) with parameters loaded from ops.npy."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied in the conversion code. The provided data already contains only tracked neurons (all iscell values are 1.0). The AI relies on Track2p's prior filtering.

ii. There is no iscell filtering code in `convert_data.py`. All neurons from `F.npy` are used directly.

iii. The CONVERSION_NOTES state: "The bundled release only contains already-tracked cells." The AI verified that iscell arrays contain only 1.0 values across all sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each consecutive 2-minute block. Each trial begins at the block boundary. The temporal alignment event is "start of each consecutive 2-minute block from a continuous recording" with `off_start=0.0` and `off_end=120.0`.

ii.
```python
neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
# ...
neural_trials = [
    neural_binned[:, trial_index, :].astype(np.float32, copy=True) for trial_index in range(n_trials)
]
```

iii. The CONVERSION_NOTES state: "Trials are aligned to block start for metadata.off_start = 0 and metadata.off_end = 120."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw data is at 30 Hz. The AI applies temporal rebinning by averaging 10 consecutive frames, resulting in a time bin size of 333.33 ms (10 frames / 30 Hz * 1000 ms). Each 2-minute trial has 360 time bins (3600 frames / 10).

ii.
```python
BIN_FRAMES = 10
BINS_PER_TRIAL = TRIAL_FRAMES // BIN_FRAMES  # 360
TIME_BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FRAME_RATE_HZ  # 333.33...

neural_binned = neural_full[:, :usable_frames].reshape(
    neural_full.shape[0], n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. The paper's methods state: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI correctly replicates this 10-frame averaging.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is not directly loaded from any raw data file. It is computed from the frame indices, the frame rate (30 Hz), and the bin size (10 frames). The time represents seconds from the start of the session.

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The time is synthesized from the known frame rate and binning parameters. The AI names it `time_from_session_start_s` in the output.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each time bin, the center time is computed as `(bin_index * 10 + 5) / 30` seconds from the start of the session. For each trial, the trial's offset within the session is added: `trial_index * 120` seconds. This gives absolute time from session start.

ii.
```python
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The CONVERSION_NOTES state: "The decoder input is not trial-relative time; it is absolute time-from-session-start in seconds, carried through as a 1-by-time array."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed to match the same temporal grid as the neural data. Each time bin center corresponds to the center of the same 10-frame window used for neural and motion binning. The time array has the same number of time points (360 per trial) as the neural data.

ii.
```python
# Same BINS_PER_TRIAL = 360 used for both neural and time
base_time = (np.arange(BINS_PER_TRIAL, dtype=np.float32) * BIN_FRAMES + (BIN_FRAMES / 2.0)) / FRAME_RATE_HZ
```

iii. Alignment is guaranteed by construction since both neural and time arrays use the same bin count and bin size.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in each session's `move_deve/` directory. When camera frames are missing (motion energy length doesn't match neural frame count), `interframe_int.npy` is also used to identify and repair missing frames.

ii.
```python
def repair_motion_energy(session_dir: Path, n_frames: int) -> tuple[np.ndarray, int]:
    move_dir = session_dir / "move_deve"
    motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)
    if len(motion) == n_frames:
        return motion, 0
    interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
```

iii. The CONVERSION_NOTES describe: "Load motion_energy_glob.npy. If camera frames are missing, identify them from interframe_int.npy."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing pipeline: (1) Load raw motion energy. (2) If frames are missing, detect gaps from `interframe_int.npy` by finding intervals that are integer multiples of the median interval. (3) Insert NaN at missing positions and linearly interpolate. (4) Average in 10-frame bins (same as neural). (5) Compute global min/max and 20/40/60/80 percentiles across all binned motion values from all sessions. (6) Normalize with global min-max. (7) Discretize into 5 quintile bins using `np.digitize`.

ii.
```python
# Missing frame repair
jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
# ... insert NaN and interpolate ...

# Binning
motion_binned = motion_full[:usable_frames].reshape(n_trials, BINS_PER_TRIAL, BIN_FRAMES).mean(axis=2)

# Global normalization and discretization
all_motion = np.concatenate([trial for record in session_records for trial in record["motion_trials"]])
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80])
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale)
# ...
motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
```

iii. The CONVERSION_NOTES detail this pipeline and note that quintiles are computed globally across all sessions, not per-session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After global min-max normalization, motion energy values are discretized into 5 equal-percentile (quintile) bins using `np.digitize` with thresholds at the 20th, 40th, 60th, and 80th percentiles. The bins are labeled: `lowest_20pct`, `20_40pct`, `40_60pct`, `60_80pct`, `highest_20pct`.

ii.
```python
motion_quantiles_raw = np.percentile(all_motion, [20, 40, 60, 80]).astype(np.float32)
motion_quantiles_norm = ((motion_quantiles_raw - motion_min) / motion_scale).astype(np.float32)
# ...
motion_norm = (motion_trial - motion_min) / motion_scale
motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False).astype(np.int64)
output_trials.append(motion_bins[np.newaxis, :])
```

iii. The instructions specify "five equal-percentile bins." The output confirms exactly 20% in each bin globally. The CONVERSION_NOTES confirm "discretized into 5 equal-percentile bins."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned with neural data by using the same temporal binning scheme: both are averaged over the same 10-frame bins and split into the same 2-minute trial blocks. The motion energy is first repaired to match the neural frame count, then binned identically.

ii.
```python
# Motion binned with same parameters as neural
motion_binned = motion_full[:usable_frames].reshape(
    n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=2)
```

iii. The data README notes that the microscope acquisition triggers camera frame acquisition, "allowing for simple synchronisation across the two modalities." The AI repairs any missing camera frames to match the neural frame count before binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where `motion_energy_glob.npy` has fewer samples than neural frames) are detected by comparing lengths. The missing frame positions are identified from `interframe_int.npy` by finding intervals that are approximately integer multiples of the median inter-frame interval. NaN values are inserted at missing positions, then linearly interpolated. A total of 276 missing frames were repaired across 9 sessions.

ii.
```python
def repair_motion_energy(session_dir: Path, n_frames: int) -> tuple[np.ndarray, int]:
    # ...
    interframe = np.load(move_dir / "interframe_int.npy").astype(np.float64, copy=False) * 1000.0
    nominal = float(np.median(interframe))
    jumps = np.maximum(0, np.rint(interframe / nominal).astype(np.int64) - 1)
    missing_count = int(jumps.sum())
    expected_missing = n_frames - len(motion)
    if missing_count != expected_missing:
        raise ValueError(...)
    # Insert NaN at gaps and interpolate
    repaired[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], repaired[~nan_mask])
```

iii. The data README instructs: "The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over." The AI chose to interpolate, which is one of the two acceptable approaches.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the `load_suite2p_dff()` function, specifically the `dcnv.preprocess()` call which performs Suite2p's baseline estimation on the full fluorescence trace for each session. This involves filtering operations over the entire neural matrix (n_neurons x n_frames). Loading large `.npy` files and the missing-frame repair loop are also relatively slow but less so than the baseline computation.

ii.
```python
dff_num = dcnv.preprocess(
    F=dff_num,
    baseline=ops["baseline"],
    win_baseline=ops["win_baseline"],
    sig_baseline=ops["sig_baseline"],
    fs=ops["fs"],
    prctile_baseline=ops["prctile_baseline"],
    batch_size=ops.get("batch_size", 100),
    device=torch.device("cpu"),
)
```

iii. The trajectory shows the agent tested this processing on small subsets before running on full data, suggesting awareness of the computational cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `repair_motion_energy` function uses a Python for-loop to iterate over `jumps` and insert missing values one gap at a time. This could be vectorized using cumulative sums and fancy indexing. The trial-splitting in `session_to_trials` uses list comprehensions over trial indices, but these are already efficient via numpy slicing.

ii.
```python
# Loop in repair_motion_energy that could be vectorized:
for gap in jumps:
    repaired[dst] = motion[src]
    src += 1
    dst += 1
    if gap:
        repaired[dst:dst + gap] = np.nan
        dst += gap
```

iii. No explicit justification was given for the loop approach. The loop is straightforward but not performance-critical since it runs once per session and processes a 1-D array.

## 6-c. What processing does the code repeat multiple times?

i. The code iterates over motion trials twice: once during `session_to_trials()` to bin them, and again in the final output-building loop to normalize and discretize them. The motion values are concatenated across all sessions to compute global statistics, then each trial is processed again individually. This is a two-pass design (collect stats, then apply), which is intentional rather than redundant.

ii.
```python
# First pass: collect all motion
all_motion = np.concatenate(
    [trial for record in session_records for trial in record["motion_trials"]]
)
# Second pass: normalize and discretize each trial
for record in session_records:
    for motion_trial in record["motion_trials"]:
        motion_norm = (motion_trial - motion_min) / motion_scale
        motion_bins = np.digitize(motion_norm, motion_quantiles_norm, right=False)
```

iii. The two-pass approach is necessary because global percentiles must be computed before discretization. No truly redundant computation is present.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes absolute time-from-session-start as the decoder input, which accumulates across trials (e.g., trial 2 starts at 120s, trial 3 at 240s). Since the decoder processes trials independently, this absolute time could be seen as containing information beyond what's needed. However, it is the requested input format. The code also stores extensive metadata (missing frame counts, per-session statistics) that is not used by the decoder but serves documentation purposes. The `conversion_summary` dictionary is printed but not saved to the pickle file.

ii.
```python
time_trials = [
    (base_time + trial_index * TRIAL_SECONDS)[np.newaxis, :].astype(np.float32, copy=False)
    for trial_index in range(n_trials)
]
```

iii. The CONVERSION_NOTES acknowledge: "The decoder input is not trial-relative time; it is absolute time-from-session-start in seconds."
