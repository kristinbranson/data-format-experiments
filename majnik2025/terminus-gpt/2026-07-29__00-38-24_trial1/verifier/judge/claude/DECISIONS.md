# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by iterating over sorted subject directories and their sorted session subdirectories under the `data/` root. For each session, it loads suite2p outputs (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `spks.npy`) from `suite2p/plane0/` and behavioral files (`motion_energy_glob.npy`, `tstamps.npy`) from `move_deve/`. It checks that all required files exist before including a session.

ii.
```python
def discover_sessions(root=ROOT):
    sessions = []
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            suite = sess_dir / 'suite2p' / 'plane0'
            move = sess_dir / 'move_deve'
            req = [suite/'F.npy', suite/'Fneu.npy', suite/'iscell.npy', suite/'ops.npy', suite/'spks.npy', move/'motion_energy_glob.npy', move/'tstamps.npy']
            if all(p.exists() for p in req):
                sessions.append((subj_dir.name, sess_dir.name, sess_dir))
    return sessions

def load_session(sess_dir):
    suite = sess_dir / 'suite2p' / 'plane0'
    move = sess_dir / 'move_deve'
    F = np.load(suite / 'F.npy').astype(np.float32)
    Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
    iscell = np.load(suite / 'iscell.npy')
    ops = np.load(suite / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    spks = np.load(suite / 'spks.npy').astype(np.float32)
    return F, Fneu, spks, iscell, ops, motion, tstamps
```

iii. The AI identified the data directory structure through exploration in Steps 1-2 of its workflow. It loads all available suite2p and behavioral files, checking for their existence before including a session. The `F.npy` and `Fneu.npy` are loaded but ultimately not used for the neural signal (see 2-a).

## 1-b. How are the data split into subjects?

i. Subjects are the sorted top-level directories under the data root. A unique subject list is built from the discovered sessions, and each session is assigned a subject index.

ii.
```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
subject_idx.append(subject_to_idx[subj])
```

iii. Each top-level directory (e.g., `jm031`, `jm032`) represents one mouse. The AI sorts them alphabetically and assigns integer indices.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder represents one session (one daily recording). Sessions are sorted alphabetically within each subject.

ii.
```python
for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        # ...
        sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. The directory structure directly maps to sessions. Each session directory contains one recording's worth of suite2p and behavioral data.

## 1-d. How are the data split into trials?

i. There are no native trials. The AI segments each continuous session recording into consecutive non-overlapping 120-second (2-minute) windows. This produces 360 time bins per trial (120s * 3 Hz). Sessions with fewer than 2 windows are skipped.

ii.
```python
WINDOW_SECONDS = 120.0

def segment_session(neural_b, motion_b, time_b, fs_binned, window_seconds=WINDOW_SECONDS):
    bins_per_window = max(1, int(round(window_seconds * fs_binned)))
    n_windows = neural_b.shape[1] // bins_per_window
    neural_trials, input_trials, motion_trials = [], [], []
    for i in range(n_windows):
        sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
        neural_trials.append(neural_b[:, sl].astype(np.float32))
        input_trials.append(make_time_input(time_b[sl]))
        motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
    return neural_trials, input_trials, motion_trials, bins_per_window
```

iii. The AI noted in CONVERSION_NOTES.md Step 3 that the reference paper used "consecutive 2 minute blocks of the recording" for its decoding analysis. The AI chose 120-second windows to match this paper convention, despite the instructions specifying 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. Sessions that produce fewer than 2 windows are skipped. No other trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. The minimum of 2 trials per session is required by the decoder format specification to allow train/test splitting.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spks.npy` (suite2p deconvolved spikes), filtered by `iscell.npy`. The AI initially tried using `F.npy` and `Fneu.npy` to compute dF/F but switched to `spks.npy` after finding below-chance decoding accuracy with dF/F.

ii.
```python
F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 10: "Initial dF/F-like fluorescence signal yielded below-chance sample decoding; switched to Suite2p `spks.npy`, which improved sample and full decoding substantially."

## 2-b. How is the `neural` data processed?

i. The deconvolved spikes from `spks.npy` are filtered by iscell, trimmed to the common length with motion energy and timestamps, then averaged into non-overlapping bins of 10 frames. No additional preprocessing (neuropil subtraction, baseline correction) is applied since the spks signal is already deconvolved.

ii.
```python
neural = spks[keep].astype(np.float32)
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
```

iii. The AI chose to use the pre-computed deconvolved spikes rather than re-deriving a fluorescence-based signal. The 10-frame binning matches the reference paper's stated processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by suite2p's `iscell` classification: only ROIs with probability > 0.5 are kept (the default threshold).

ii.
```python
CELL_THRESHOLD = 0.5
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 3: "We considered all ROIs above the default threshold of 0.5 as true cells" — the AI matched this directly from the paper's methods section.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since the recordings are continuous and trials are consecutive windows, each trial starts where the previous one ended. No event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
'off_start': 0.0,
'off_end': WINDOW_SECONDS,
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy data are averaged into non-overlapping bins of 10 consecutive frames, converting from 30 Hz to 3 Hz (333.33 ms per bin). This rebinning is applied before discretization of motion energy.

ii.
```python
BIN_FRAMES = 10

def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)

def bin_array_1d(x, bin_frames):
    n_bins = x.shape[0] // bin_frames
    x = x[:n_bins * bin_frames]
    return x.reshape(n_bins, bin_frames).mean(axis=1)
```

iii. From CONVERSION_NOTES.md Step 3: the methods state "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the bin index and the binned frame rate, giving seconds from the start of the session. It is not derived from any raw timestamp variable. The AI computes it as `np.arange(n_bins) / fs_binned` where `fs_binned = 30 / 10 = 3 Hz`.

ii.
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 10: "Initial time input construction from raw timestamps produced incorrect scale; replaced with frame-rate-derived elapsed seconds from session start." The AI reads `fs` from `ops.npy` (which defaults to 30 Hz) and divides by the bin factor.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index / fs_binned` where `fs_binned = ops['fs'] / BIN_FRAMES`. This gives elapsed seconds from session start. The time input is reshaped to `(1, n_timepoints)` for each trial.

ii.
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
# ...
def make_time_input(t_binned):
    return t_binned[None, :].astype(np.float32)
```

iii. The computation is straightforward: elapsed seconds = bin index / bins-per-second. The time increases across trials within a session (it's not reset per trial).

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time array is computed from the same bin indices as the neural data, so it is inherently aligned. After segmentation into trials, the same slice is used for both neural and time data.

ii.
```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(make_time_input(time_b[sl]))
```

iii. Since time is derived from bin indices rather than from a separate data stream, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
```

iii. This is the pre-computed global motion energy signal from the behavioral video, consistent with the paper's description.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) trim to common length with neural data, (2) average into 10-frame bins, (3) discretize into 5 equal-percentile bins. The percentile binning is done **globally** across all sessions (all data pooled together) rather than per-session.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
motion_b = bin_array_1d(motion, BIN_FRAMES)
# ... later:
def percentile_bin_outputs(all_motion_trials, n_classes=5):
    vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
    edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    out = []
    for sess in all_motion_trials:
        sess_out = []
        for m in sess:
            b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
            sess_out.append(b[None, :])
        out.append(sess_out)
    return out, edges
```

iii. The AI chose global percentile binning to ensure exactly balanced classes across the full dataset, as confirmed by the verification output showing exactly 20% per bin globally. The instructions say "five equal-percentile bins, selected per session" but the AI's code pools all sessions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using percentile edges computed **globally** across all sessions. The bin edges are `[0, 20, 40, 60, 80, 100]` percentiles of all motion energy values pooled together. The lowest edge is set to `-inf` and the highest to `+inf`.

ii.
```python
vals = np.concatenate([m.ravel() for sess in all_motion_trials for m in sess])
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
edges[0] = -np.inf
edges[-1] = np.inf
# ...
b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. Global percentile binning produces exactly balanced class distributions overall (each bin has 20% of all data), but per-session distributions can be very unbalanced (some sessions have 0% in certain bins).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy and neural data are aligned by trimming both to the minimum common length. The AI detects that both streams are acquired synchronously at 30 Hz. When there is a length mismatch (e.g., due to dropped video frames), both arrays are simply truncated to the shorter length.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
```

iii. The AI noted in CONVERSION_NOTES.md Step 4 that "most sessions have exact frame count matches; some have small mismatches." Rather than interpolating dropped frames (as the reference does), the AI trims to the shorter stream.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Length mismatches between neural data, motion energy, and timestamps are handled by trimming all streams to the minimum common length. Remainder bins at the end of a session that don't fill a complete trial window are discarded. Sessions with fewer than 2 trial windows are skipped entirely.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
# ...
if len(neural_trials) < 2:
    continue
```

iii. The AI's approach is simpler than the reference's dropped-frame interpolation but loses a small amount of data from sessions with mismatches.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading the `.npy` files (I/O bound) and array operations for binning. The code processes all 41 sessions in about 16 seconds. There is no heavy preprocessing like the reference's `dcnv.preprocess` since the AI uses pre-computed `spks.npy`.

ii. N/A

iii. From conversion_full_out.txt, total conversion time was 15.90 seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over windows, slicing arrays for each trial. This could potentially be vectorized using reshape operations, though the current approach is already fast.

ii.
```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The loop is simple and fast, so vectorization would provide minimal benefit.

## 6-c. What processing does the code repeat multiple times?

i. The code loads `F.npy` and `Fneu.npy` but never uses them (only `spks.npy` is used for the neural signal). This is redundant I/O.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
# These are loaded but never used in the main processing path
```

iii. These files are vestiges of the AI's earlier approach using dF/F, which was abandoned in favor of `spks.npy`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `F.npy`, `Fneu.npy`, and `tstamps.npy` but does not use them in the final processing pipeline (only `spks.npy` is used for neural data, and time is computed from frame rate rather than timestamps). The `compute_df_f` function is defined but never called.

ii.
```python
def compute_df_f(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    Fc = F - neucoeff * Fneu
    prctile = float(ops.get('prctile_baseline', 8.0))
    baseline = np.percentile(Fc, prctile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fc - baseline) / baseline
    return dff.astype(np.float32)
```

iii. The `compute_df_f` function and the loading of `F.npy`/`Fneu.npy` are remnants of the AI's iterative development process. The timestamps are loaded but only used for trimming (and could be replaced by using the neural frame count directly).
