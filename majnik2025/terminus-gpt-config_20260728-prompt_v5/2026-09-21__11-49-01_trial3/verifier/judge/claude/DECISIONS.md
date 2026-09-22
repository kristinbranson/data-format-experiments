# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subject directories under `/app/data`, then iterates over session subdirectories within each subject. For each session, it loads suite2p outputs (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) from `suite2p/plane0/` and behavioral data (`motion_energy_glob.npy`, `tstamps.npy`) from `move_deve/`. It validates that both `suite2p/plane0` and `move_deve` directories exist before including a session.

ii.
```python
def list_sessions(data_root: Path):
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            s2p = sess_dir / 'suite2p' / 'plane0'
            mov = sess_dir / 'move_deve'
            if s2p.exists() and mov.exists():
                sessions.append((subj_dir.name, sess_dir.name, sess_dir))
    return sessions

def load_session(sess_dir: Path):
    s2p = sess_dir / 'suite2p' / 'plane0'
    mov = sess_dir / 'move_deve'
    ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
    F = np.load(s2p / 'F.npy').astype(np.float32)
    Fneu = np.load(s2p / 'Fneu.npy').astype(np.float32)
    iscell = np.load(s2p / 'iscell.npy')
    motion = np.load(mov / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
    return ops, F, Fneu, iscell, motion, tstamps
```

iii. The AI documented that data are organized by subject directory and session subdirectory, each containing suite2p outputs and motion energy files. The existence check for `suite2p/plane0` and `move_deve` ensures only complete sessions are included.

## 1-b. How are the data split into subjects?

i. Subjects are the top-level sorted directories under the data root. Each directory name (e.g., `jm031`) corresponds to one mouse.

ii.
```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI identified subject folders from the data directory structure. All directories are included as subjects.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder represents one recording session. Sessions are sorted alphabetically and each contains one day's recording.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        ...
        sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. Each session subdirectory contains suite2p output and behavioral data for one daily recording session.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are artificial 60-second non-overlapping segments of the continuous recording. After binning (10 frames per bin at 30 Hz), each trial is 180 time bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
dt = float(bin_size_frames / fs)
trial_len = int(round(trial_seconds / dt))  # 180 bins
n_trials = neural_b.shape[1] // trial_len
...
usable = n_trials * trial_len
neural_b = neural_b[:, :usable]
...
for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The instructions specify splitting sessions into 60-second trials. Since the recording is continuous with no stimulus-driven trial structure, fixed-length segmentation is appropriate.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are included. Only the incomplete trailing segment (if any) is discarded.

ii.
```python
n_trials = neural_b.shape[1] // trial_len
if n_trials < 2:
    raise ValueError(f'Not enough 60 s trials in session {sess_dir.name}: {n_trials}')
```

iii. The AI checks that at least 2 trials exist per session (as required by instructions for decoder evaluation) but applies no further quality filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `iscell.npy` (cell classification) from suite2p `plane0` output.

ii.
```python
F = np.load(s2p / 'F.npy').astype(np.float32)
Fneu = np.load(s2p / 'Fneu.npy').astype(np.float32)
iscell = np.load(s2p / 'iscell.npy')
```

iii. These are the standard suite2p output files for raw fluorescence, neuropil fluorescence, and cell classification.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) filter neurons by `iscell` classification, (2) neuropil subtraction (`F - neucoeff * Fneu`), and (3) a simple percentile-based dF/F computation using the 20th percentile as baseline: `(Fcorr - baseline) / baseline`.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
F = F[cell_mask]
Fneu = Fneu[cell_mask]

Fcorr = F - neucoeff * Fneu
neural = robust_dff(Fcorr)

def robust_dff(Fcorr: np.ndarray):
    baseline = np.percentile(Fcorr, 20, axis=1, keepdims=True)
    baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
    return (Fcorr - baseline) / baseline
```

iii. The AI's CONVERSION_NOTES.md states that methods text says decoding used dF/F traces, so it chose to compute a dF/F-like representation. The neuropil coefficient is read from `ops.npy` when available.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using suite2p's `iscell` classification. Only neurons where `iscell[:, 0]` is True are included.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
F = F[cell_mask]
Fneu = Fneu[cell_mask]
```

iii. The AI noted in CONVERSION_NOTES.md that reference code uses Suite2p `iscell` labels/thresholds for cell filtering, and the Track2p code applies iscell filtering when exporting matched outputs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments, no event-based alignment is needed. The neural data is truncated to the overlap between neural and behavioral data lengths before segmentation.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
neural = neural[:, :n_overlap]
...
# metadata
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 60.0,
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments from the start of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing from 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
neural_b = moving_average_bin(neural, bin_size_frames).astype(np.float32)
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)

def moving_average_bin(x: np.ndarray, bin_size: int):
    n = x.shape[-1] // bin_size
    trimmed = x[..., : n * bin_size]
    new_shape = x.shape[:-1] + (n, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The methods text states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI follows this exactly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from imaging frame indices and the sampling rate (`ops['fs']`), not from any stored timestamp variable. It is named `session_time_seconds`.

ii.
```python
raw_time = np.arange(n_overlap, dtype=np.float32) / fs
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
```

iii. The AI initially tried using `tstamps.npy` directly but found the units were not seconds, so it switched to deriving time from frame indices and sampling rate. This is documented in CONVERSION_NOTES.md.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A time array is constructed from frame indices divided by sampling rate, then binned using the same 10-frame moving average as neural and behavioral data.

ii.
```python
raw_time = np.arange(n_overlap, dtype=np.float32) / fs
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
```

iii. Since the frame rate is constant at 30 Hz, computing time from frame indices is equivalent to using actual timestamps.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time array is computed from the same frame indices as the neural data and binned with the same function, ensuring perfect alignment. Both use `n_overlap` as the number of valid frames.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
raw_time = np.arange(n_overlap, dtype=np.float32) / fs
time_b = moving_average_bin(raw_time[None, :], bin_size_frames)[0].astype(np.float32)
```

iii. Alignment is guaranteed by construction since all arrays use the same length and binning.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory.

ii.
```python
motion = np.load(mov / 'motion_energy_glob.npy').astype(np.float32)
```

iii. This is the pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Two processing steps: (1) the motion energy trace is averaged into 10-frame bins, (2) the binned signal is discretized into 5 equal-percentile bins per session using `np.quantile` and `np.digitize`.

ii.
```python
motion_b = moving_average_bin(motion[None, :], bin_size_frames)[0].astype(np.float32)
motion_bins, edges = compute_quantile_bins(motion_b, n_bins=5)

def compute_quantile_bins(values: np.ndarray, n_bins: int = 5):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins.astype(np.int64), edges
```

iii. The instructions specify discretizing motion energy into five equal-percentile bins selected per session. The AI applies this after binning, which is correct since averaging categorical labels would be meaningless.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes 5 quantile edges per session using `np.quantile` with `np.linspace(0, 1, 6)`, then uses `np.digitize` with `edges[1:-1]` to assign each time bin to one of 5 categories (0-4). A small epsilon is added to handle duplicate edges.

ii.
```python
def compute_quantile_bins(values: np.ndarray, n_bins: int = 5):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins.astype(np.int64), edges
```

iii. Per-session quantile binning ensures equal-percentile bins within each session, as required by the instructions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI truncates all arrays (neural, motion, timestamps) to the minimum overlapping length (`n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])`), then applies the same binning to both streams.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
neural = neural[:, :n_overlap]
motion = motion[:n_overlap]
tstamps = tstamps[:n_overlap]
```

iii. The AI noted that motion energy arrays are typically ~2 samples shorter than neural frame count. Rather than interpolating dropped frames, it truncates to the overlap region.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles length mismatches between neural and behavioral data by truncating all streams to their minimum overlapping length. Remainder frames at the end of a session that don't fill a complete trial are discarded. Sessions with fewer than 2 trials raise an error.

ii.
```python
n_overlap = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
neural = neural[:, :n_overlap]
motion = motion[:n_overlap]
...
if n_trials < 2:
    raise ValueError(f'Not enough 60 s trials in session {sess_dir.name}: {n_trials}')
```

iii. The truncation approach is simpler than interpolation but discards a small number of valid neural data frames at the end of the recording.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation (percentile calculation over the full session for all neurons) and the binning operations. Loading `.npy` files is also I/O bound. Total processing time for all 41 sessions was ~20 seconds.

ii. N/A

iii. The code is relatively efficient, completing full conversion in about 20 seconds. No GPU-accelerated processing is used for neural data (unlike the reference which uses suite2p's `dcnv.preprocess`).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-slicing loop could potentially be vectorized using array reshaping, but since it just performs slicing operations, the overhead is minimal.

ii.
```python
for i in range(n_trials):
    sl = slice(i * trial_len, (i + 1) * trial_len)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The loops are all simple slicing operations with negligible overhead. No significant vectorization opportunities exist.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any major processing. Each session is processed once in a single pass.

ii. N/A

iii. The code processes each session independently in a single loop, with no redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` and `ops.npy` for each session but only uses `ops['fs']` (sampling rate) and `ops.get('neucoeff', 0.7)`. The timestamps are used only for determining `n_overlap`. The code also computes and stores `session_info` metadata that is not used by the decoder.

ii.
```python
tstamps = np.load(mov / 'tstamps.npy').astype(np.float64)
ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
```

iii. Loading extra files adds minor I/O overhead but is negligible relative to the total processing time.
