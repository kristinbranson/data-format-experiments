# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the `data/` directory for subject subdirectories, then iterates over session subdirectories within each subject. For each session, it loads suite2p outputs (`F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `ops.npy`) from `suite2p/plane0/` and behavioral files (`motion_energy_glob.npy`, `tstamps.npy`) from `move_deve/`. It validates that all required files exist before including a session.

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

iii. The AI identified from the data directory structure that subjects are organized as top-level folders and sessions as their subdirectories. It checks for required files before including a session, which is a defensive approach. Although it loads F.npy and Fneu.npy, it ultimately uses spks.npy for neural data (see 2-a).

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted top-level directories within the data root. Each directory name becomes a subject ID. Unlike the reference which filters for `jm*` prefix, the AI includes all directories.

ii.
```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. All top-level data directories are assumed to be subjects. In practice, all directories do start with `jm`, so the result is the same.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject folder. Each subdirectory containing the required suite2p and motion energy files is treated as one session.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    # ... check required files exist ...
    sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. Each subdirectory within a subject folder represents one daily recording session, consistent with the longitudinal imaging design described in the paper.

## 1-d. How are the data split into trials?

i. There is no native trial structure. The AI creates pseudo-trials by segmenting continuous recordings into consecutive 120-second (2-minute) non-overlapping windows. This corresponds to 360 binned timepoints at the AI's binned rate of 3 Hz (30 Hz / 10 frames). Sessions producing fewer than 2 trials are excluded.

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

iii. The AI chose 2-minute windows based on the paper's statement: "splits were done on consecutive 2 minute blocks of the recording." However, this refers to cross-validation splits in the paper's decoding analysis, not necessarily to trial definitions. The reference solution uses 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. Sessions producing fewer than 2 trials are excluded. No other trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. The minimum of 2 trials per session is needed for decoder training/validation. No trial-quality filtering is described in the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `spks.npy` (suite2p deconvolved spikes), filtered by `iscell.npy`. It initially planned to use F.npy and Fneu.npy to compute dF/F (as described in the paper), but switched to spks.npy after the dF/F-based decoder performed below chance level.

ii.
```python
F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 10: "Initial dF/F-like fluorescence signal yielded below-chance sample decoding; switched to Suite2p `spks.npy`, which improved sample and full decoding substantially." The trajectory shows the AI reasoned that since spks.npy is a native suite2p output and gave better decoding performance, it was a valid alternative despite the paper saying "We used baseline corrected fluorescence traces as our dF/F... for all subsequent analyses."

## 2-b. How is the `neural` data processed?

i. The AI's processing consists of: (1) filtering neurons using `iscell[:, 0] > 0.5`, (2) using pre-computed deconvolved spikes from `spks.npy` directly (no neuropil subtraction or baseline correction), (3) trimming to common length with behavior data, and (4) temporal binning by averaging 10 consecutive frames.

ii.
```python
CELL_THRESHOLD = 0.5
BIN_FRAMES = 10

keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)

def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)
```

iii. The AI followed the paper's mention of "averaging in bins of 10 consecutive timestamps" for decoding analyses and the cell inclusion threshold of 0.5. However, it deviated from the paper's description of using baseline-corrected fluorescence (dF/F) by using deconvolved spikes instead.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using suite2p's `iscell` classification with a threshold of 0.5. Only ROIs with `iscell[:, 0] > 0.5` are retained.

ii.
```python
CELL_THRESHOLD = 0.5
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The paper states: "We considered all ROIs above the default threshold of 0.5 as true cells." The AI directly implemented this criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since the recordings are continuous and trials are consecutive segments, there is no event-based alignment. The alignment event is described as "session start."

ii.
```python
'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
'off_start': 0.0,
'off_end': WINDOW_SECONDS,
```

iii. There is no stimulus-driven trial structure, so alignment to session start is the natural choice for continuous recordings segmented into windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames. The native imaging rate is 30 Hz, so the binned rate is 3 Hz (time bin size = ~333.33 ms).

ii.
```python
BIN_FRAMES = 10

neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)

'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values])),
```

iii. The paper states: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI followed this exactly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is synthetically computed from the binned frame index and the binned sampling rate.

ii.
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. Since the imaging rate is constant (30 Hz), time can be computed from frame indices. The AI computes elapsed seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `frame_index / binned_sampling_rate`, giving seconds from session start. The time array is then segmented into windows along with the neural and output data, so each trial's time input reflects its position within the session.

ii.
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
# Then segmented into windows:
input_trials.append(make_time_input(time_b[sl]))

def make_time_input(t_binned):
    return t_binned[None, :].astype(np.float32)
```

iii. The decoder task asks for "time elapsed from the beginning of the experiment," so the AI uses absolute session time rather than resetting per trial.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same frame indices as the neural data, so they are inherently aligned. Both are binned with the same bin size and segmented into the same windows.

ii.
```python
# All computed from same base array length and segmented with same slice:
sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
neural_trials.append(neural_b[:, sl].astype(np.float32))
input_trials.append(make_time_input(time_b[sl]))
```

iii. Since time is derived from frame indices matching the neural array dimensions, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
```

iii. This file contains pre-computed global motion energy from behavioral video, as described in the paper's methods section.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) trim motion energy to common length with neural data, (2) temporally bin by averaging 10 consecutive frames, (3) discretize into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
motion_b = bin_array_1d(motion, BIN_FRAMES)

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

iii. The AI follows the paper's instruction to bin by 10 frames for decoding and uses equal-percentile discretization as required by the task. No standard deviation normalization is applied (the reference normalizes by std per session before pooling). No dropped frame interpolation is performed (the reference detects dropped frames via interframe intervals and interpolates them).

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using global equal-percentile edges. The edges are computed across all sessions and subjects, with the first and last edges set to -inf and +inf respectively.

ii.
```python
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
edges[0] = -np.inf
edges[-1] = np.inf
b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. Global percentile-based binning ensures approximately balanced class distributions, matching the task requirement of "five equal-percentile bins."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI trims neural, motion energy, and timestamp arrays to their shared minimum length before any further processing. This contrasts with the reference approach of detecting dropped video frames and interpolating them.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
```

iii. The AI noted that "most sessions have exact frame count matches; some have small mismatches." Trimming to the shortest array is a simpler approach than dropped-frame interpolation, but may cause temporal misalignment when frames are dropped in the middle of a recording rather than at the end.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Length mismatches between neural and behavior arrays are handled by trimming to the minimum shared length. Remainder frames that don't fill a complete trial window are discarded. Sessions with fewer than 2 trials are excluded.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]

# Remainder handled by integer division in segment_session:
n_windows = neural_b.shape[1] // bins_per_window

# Minimum trial count:
if len(neural_trials) < 2:
    continue
```

iii. The trimming approach is simple and robust but may lose data or cause misalignment when the mismatch is due to dropped frames in the middle of a recording. The reference solution handles this more precisely by detecting and interpolating dropped frames.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and processing the numpy files for each session. Since the AI uses `spks.npy` directly without GPU-based baseline correction, there is no expensive preprocessing step like the reference's `dcnv.preprocess`. The full conversion runs relatively quickly.

ii. N/A (no explicit timing bottleneck in the AI's code beyond I/O)

iii. The AI's pipeline is simpler than the reference's because it skips the suite2p baseline correction step, which is the reference's primary bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop in `segment_session` iterates over windows to slice arrays, but this is straightforward slicing and would be difficult to vectorize meaningfully. The `percentile_bin_outputs` function loops over sessions and trials, which could potentially be done with a single concatenation and reshape.

ii.
```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The loops are simple and the number of iterations is small, so vectorization would provide minimal benefit.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once, and discretization is done in a single global pass after all sessions are loaded.

ii. N/A

iii. The code follows a clean two-pass structure: first pass loads and preprocesses all sessions, second pass discretizes and assembles the output.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `F.npy`, `Fneu.npy`, and `ops.npy` but never uses them for the final neural signal (it uses `spks.npy` instead). The `tstamps.npy` array is loaded and binned but only used for trimming; the actual time input is computed from frame indices. The `compute_df_f` function exists in the code but is never called (dead code from the initial dF/F approach).

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)         # loaded but not used for neural
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)    # loaded but not used for neural
# compute_df_f function defined but never called
```

iii. These are remnants of the AI's initial dF/F approach. Loading unused files wastes I/O time and memory.
