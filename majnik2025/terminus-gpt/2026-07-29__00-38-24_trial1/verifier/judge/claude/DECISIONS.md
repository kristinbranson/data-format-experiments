# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by iterating over sorted subdirectories of each subject folder under `data/`. For each session, it loads suite2p outputs (`F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `ops.npy`) from `suite2p/plane0/` and behavioral data (`motion_energy_glob.npy`, `tstamps.npy`) from `move_deve/`. Sessions are only included if all required files exist.

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

iii. The AI documented that data are organized as `data/<subject>/<session>/...` with suite2p and behavioral outputs per session. It loads all files needed for its processing pipeline, including `spks.npy` and `tstamps.npy` which the reference solution does not use.

## 1-b. How are the data split into subjects?

i. Subjects correspond to sorted top-level subdirectories under the `data/` directory. Unique subject names are extracted from discovered sessions and mapped to integer indices.

ii.
```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. Each top-level directory represents one mouse. The AI derives the subject list from discovered sessions rather than directly scanning for `jm*`-prefixed directories, but the result is equivalent.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject folder. Sessions are sorted alphabetically and processed in order. Each session's data is loaded independently.

ii.
```python
for subj, sess_name, sess_dir in sessions:
    F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
    # ... process session ...
```

iii. Sessions are sorted to ensure deterministic ordering. Each subdirectory contains one daily recording.

## 1-d. How are the data split into trials?

i. Trials are artificial segments: the continuous recording (after binning) is split into consecutive non-overlapping 120-second windows. At a binned frame rate of 3 Hz, each trial is 360 time bins. Any remainder that doesn't fill a complete window is discarded. Sessions producing fewer than 2 trials are excluded.

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

iii. The AI chose 120-second (2-minute) windows to match the reference paper's use of "consecutive 2 minute blocks" for decoding analysis. This is documented in Step 3 and Step 4 of CONVERSION_NOTES.md.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials after segmentation are excluded. No other trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. The minimum-2-trials requirement is imposed by the decoder format specification. No additional quality filtering is described in the reference paper for trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spks.npy` (suite2p deconvolved spikes), filtered by `iscell.npy`. Although `F.npy` and `Fneu.npy` are loaded, they are not used in the final conversion.

ii.
```python
F, Fneu, spks, iscell, ops, motion, tstamps = load_session(sess_dir)
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The AI initially planned to use dF/F (baseline-corrected fluorescence) as described in the methods text, but switched to `spks.npy` after finding that dF/F yielded below-chance decoding accuracy (documented in CONVERSION_NOTES.md Step 10: "Initial dF/F-like fluorescence signal yielded below-chance sample decoding; switched to Suite2p spks.npy").

## 2-b. How is the `neural` data processed?

i. After loading `spks.npy` and filtering by `iscell > 0.5`, the neural data is trimmed to a common length with behavioral data, then temporally binned by averaging in 10-frame bins.

ii.
```python
neural = spks[keep].astype(np.float32)
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)

def bin_array_2d(x, bin_frames):
    n_bins = x.shape[1] // bin_frames
    x = x[:, :n_bins * bin_frames]
    return x.reshape(x.shape[0], n_bins, bin_frames).mean(axis=2)
```

iii. The 10-frame binning matches the reference paper's description: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI documented this decision in Steps 3-5 of CONVERSION_NOTES.md.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using suite2p's cell classification: only ROIs with `iscell` probability > 0.5 are retained.

ii.
```python
CELL_THRESHOLD = 0.5
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The AI documented this decision based on the reference paper: "We considered all ROIs above the default threshold of 0.5 as true cells" (CONVERSION_NOTES.md Step 3).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of a continuous recording, no event-based alignment is needed. The temporal alignment event is documented as "session start."

ii.
```python
'temporal_alignment_event': 'session start (continuous recording segmented into consecutive windows)',
'off_start': 0.0,
'off_end': WINDOW_SECONDS,
```

iii. There is no stimulus-driven trial structure, so alignment to session start is the natural choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz data is rebinned by averaging 10 consecutive frames, yielding a 3 Hz effective sampling rate (time bin size ~333.33 ms).

ii.
```python
BIN_FRAMES = 10
neural_b = bin_array_2d(neural, BIN_FRAMES)
# ...
'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values]))
# = 1000 * (1 / 3.0) = 333.33 ms
```

iii. This matches the reference paper's description of denoising by averaging 10 consecutive timestamps for both neural and behavioral traces.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from a raw data variable. It is computed synthetically from the binned frame index divided by the binned sampling rate.

ii.
```python
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. Since the frame rate is constant and known, computing time from indices is straightforward. The AI noted that timestamps from `tstamps.npy` initially caused issues with incorrect scaling.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index / fs_binned` where `fs_binned = 30 / 10 = 3 Hz`. This gives elapsed seconds from session start at the binned temporal resolution.

ii.
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. The AI switched from using raw timestamps to computed time after encountering scale issues (CONVERSION_NOTES.md Step 10).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is inherently aligned because it is derived from the same array indices as the binned neural data. For each trial, the time slice corresponds exactly to the neural data slice.

ii.
```python
input_trials.append(make_time_input(time_b[sl]))
# where sl is the same slice used for neural_trials
```

iii. N/A (alignment is trivial by construction).

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
```

iii. This is the pre-computed global motion energy signal from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) trim to common length with neural data, (2) temporal binning by averaging 10 consecutive frames, (3) discretization into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
motion_b = bin_array_1d(motion, BIN_FRAMES)
# ... later globally:
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

iii. No normalization by standard deviation is applied before discretization. The percentile-based binning uses `np.quantile` with edge values extended to `±inf`.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using global percentile edges. The edges are computed from the full dataset using `np.quantile` at evenly-spaced percentiles (0%, 20%, 40%, 60%, 80%, 100%), with the first and last edges set to `-inf` and `+inf`. Values are then assigned bins using `np.digitize`.

ii.
```python
edges = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
edges[0] = -np.inf
edges[-1] = np.inf
b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
```

iii. This produces approximately equal-frequency bins across the entire dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Neural and motion energy arrays are trimmed to their common minimum length before any binning. After binning, both have the same number of time bins, ensuring frame-for-frame alignment.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
```

iii. Rather than interpolating dropped frames (as the reference solution does), the AI trims all streams to the shortest common length. This discards a small number of frames at the end of longer streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Length mismatches between neural and behavioral streams are handled by trimming to the minimum shared length. Sessions with fewer than 2 trials after segmentation are skipped. Remainder frames that don't fill a complete trial/window are discarded.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
# ...
if len(neural_trials) < 2:
    continue
```

iii. The AI documented in CONVERSION_NOTES.md Step 10 that "Some sessions have neural/behavior length mismatches; conversion trims to shared minimum length before binning."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the session data files from disk (multiple `.npy` files per session). No GPU-based preprocessing (like suite2p's `dcnv.preprocess`) is used, so computation is relatively fast. The binning and segmentation are vectorized numpy operations.

ii. N/A

iii. The AI's approach avoids the expensive `dcnv.preprocess` baseline correction used in the reference solution. Loading 7 `.npy` files per session is the main bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over windows to slice arrays, but this is inherent to creating a list of variable-length arrays and cannot be easily vectorized. The percentile binning loop iterates over sessions and trials but is simple.

ii.
```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. These loops are lightweight (slicing operations) and do not represent significant performance bottlenecks.

## 6-c. What processing does the code repeat multiple times?

i. The code loads `F.npy` and `Fneu.npy` but never uses them in the final pipeline (it uses `spks.npy` instead). This is redundant I/O.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32)
# These are never used after load_session returns
```

iii. The `compute_df_f` function exists in the code but is never called. These files are loaded unnecessarily, wasting I/O time and memory.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Loading `F.npy`, `Fneu.npy`, and computing the `_t_b_raw` (binned raw timestamps) are unnecessary since they are not used in the final output.

ii.
```python
_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)  # computed but never used
```

iii. These are artifacts of the iterative development process where the AI initially tried dF/F and raw timestamps but switched to `spks.npy` and computed time.
