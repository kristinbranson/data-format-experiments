# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all sessions by iterating over subject directories under `data/`, then session directories under each subject. For each session, it checks that all required files exist (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, `spks.npy`, `motion_energy_glob.npy`, `tstamps.npy`). Valid sessions are loaded one at a time in a loop, with each session's suite2p data and behavioral data loaded via `np.load`.

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

iii. The AI documented in CONVERSION_NOTES.md that data are organized as `data/<subject>/<session>/...`, with neural files in `suite2p/plane0/` and behavioral files in `move_deve/`. The discovery function only includes sessions where all required files are present.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the top-level directory names under `data/` (e.g., `jm031`, `jm032`, ..., `jm046`). A sorted unique list of subjects is created, and each session is associated with its subject via `subject_to_idx`.

ii.
```python
subjects = sorted({s[0] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
subject_idx.append(subject_to_idx[subj])
```

iii. The AI documented 6 subjects in the dataset, matching the data folder structure. Subject identity is derived from the directory name.

## 1-c. How are the data split into sessions?

i. Each subdirectory under a subject directory represents one session (one recording day). Sessions are sorted alphabetically (chronologically by date). All 41 sessions with complete data files are included.

ii.
```python
for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
        # ... check required files exist
        sessions.append((subj_dir.name, sess_dir.name, sess_dir))
```

iii. The AI noted 41 valid sessions across 6 subjects (7 sessions each except jm040 with 6), consistent with the data directory structure.

## 1-d. How are the data split into trials?

i. Since there are no native trials (continuous recordings), the AI segments each session into consecutive non-overlapping windows of 120 seconds (2 minutes). This is done after temporal binning by computing `bins_per_window = round(window_seconds * fs_binned)` and slicing the binned arrays into consecutive segments.

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

iii. The AI justified this by referencing the methods text: "splits were done on consecutive 2 minute blocks of the recording." The remainder of the recording that doesn't fill a complete 2-minute window is discarded.

## 1-e. How are trials filtered based on quality controls?

i. Sessions that produce fewer than 2 trials (windows) are skipped. Otherwise, no per-trial quality filtering is applied. This is consistent with the data being continuous recordings without native trial structure requiring quality-based filtering.

ii.
```python
if len(neural_trials) < 2:
    continue
```

iii. The AI did not document explicit trial filtering rationale beyond requiring a minimum of 2 trials per session for decoder evaluation, which is a requirement from the instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spks.npy` (Suite2p deconvolved spikes), filtered by `iscell.npy` cell classification. The AI initially planned to use dF/F (from `F.npy` and `Fneu.npy`) based on the methods text but switched to `spks.npy` after dF/F yielded below-chance decoder accuracy.

ii.
```python
spks = np.load(suite / 'spks.npy').astype(np.float32)
# ...
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 10: "Initial dF/F-like fluorescence signal yielded below-chance sample decoding; switched to Suite2p `spks.npy`, which improved sample and full decoding substantially." The load_data.ipynb in the data directory also suggests spks.npy as an alternative: "for more proper analysis compute dF/F the way as described in the paper (or alternatively use spks.npy)".

## 2-b. How is the `neural` data processed?

i. The deconvolved spike data (`spks.npy`) is filtered to keep only cells with `iscell > 0.5`, trimmed to match the length of the motion energy and timestamps arrays, then temporally binned by averaging every 10 consecutive frames.

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

iii. The 10-frame binning matches the methods text: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." Cell threshold of 0.5 matches: "We considered all ROIs above the default threshold of 0.5 as true cells."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells with `iscell` probability > 0.5 are included. No additional quality filtering (e.g., based on activity levels, signal-to-noise, etc.) is applied.

ii.
```python
CELL_THRESHOLD = 0.5
keep = iscell[:, 0] > CELL_THRESHOLD if iscell.ndim == 2 else iscell > CELL_THRESHOLD
neural = spks[keep].astype(np.float32)
```

iii. The AI documented this matches the paper: "We considered all ROIs above the default threshold of 0.5 as true cells." Furthermore, the data only contains cells tracked across all days by Track2p, so they are already curated.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no specific event alignment. The continuous recording is segmented into consecutive 2-minute windows starting from the session beginning. Each trial's neural data starts at a fixed offset from the session start (trial_index * window_duration).

ii.
```python
def segment_session(neural_b, motion_b, time_b, fs_binned, window_seconds=WINDOW_SECONDS):
    bins_per_window = max(1, int(round(window_seconds * fs_binned)))
    n_windows = neural_b.shape[1] // bins_per_window
    for i in range(n_windows):
        sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
        neural_trials.append(neural_b[:, sl].astype(np.float32))
```

iii. The AI set `temporal_alignment_event` to "session start (continuous recording segmented into consecutive windows)" in the metadata. This matches the reference approach of using consecutive 2-minute blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw data is acquired at ~30 Hz (one frame per imaging frame). The AI applies temporal rebinning by averaging 10 consecutive frames, resulting in ~3 Hz temporal resolution (~333.3 ms bins).

ii.
```python
BIN_FRAMES = 10
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
# ...
'time_bin_size': float(1000.0 * np.median([1.0 / fs for fs in fs_binned_values]))
```

iii. The AI documented this matches the methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not directly derived from any raw data variable. It is synthetically constructed using the frame rate from the `ops.npy` file and the number of binned timepoints.

ii.
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)
```

iii. The AI initially used the raw `tstamps.npy` but found it produced incorrect scale and replaced it with frame-rate-derived elapsed time from session start (documented in CONVERSION_NOTES.md Step 10).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as bin_index / fs_binned, where fs_binned = fs / BIN_FRAMES. This creates a linearly increasing time vector in seconds from the start of the session.

ii.
```python
fs_binned = float(ops.get('fs', 30.0)) / BIN_FRAMES
t_b = (np.arange(neural_b.shape[1], dtype=np.float32) / fs_binned).astype(np.float32)

def make_time_input(t_binned):
    return t_binned[None, :].astype(np.float32)
```

iii. The AI documented switching from raw timestamps to synthesized time due to scaling issues. The time represents elapsed seconds from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because it is constructed from the same bin indices. For each trial (window), the time slice corresponds exactly to the same time bins as the neural data slice.

ii.
```python
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    input_trials.append(make_time_input(time_b[sl]))
```

iii. By constructing time from the bin index, the AI ensures perfect alignment between time input and neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, which contains global motion energy computed from videography as described in the paper.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
```

iii. The AI documented this matches the methods text: "This yielded a scalar value quantifying the motion of the mouse at each time point."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The raw motion energy is trimmed to match the neural data length, then temporally binned by averaging 10 consecutive frames, matching the neural data binning.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
motion_b = bin_array_1d(motion, BIN_FRAMES)

def bin_array_1d(x, bin_frames):
    n_bins = x.shape[0] // bin_frames
    x = x[:n_bins * bin_frames]
    return x.reshape(n_bins, bin_frames).mean(axis=1)
```

iii. The 10-frame averaging matches the methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins computed globally across all sessions and all time points. Percentile edges are computed using `np.quantile` with `np.linspace(0, 1, 6)`, then `np.digitize` assigns each value to a bin (0-4).

ii.
```python
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

iii. The instructions specify "normalized and discretized into five equal-percentile bins." The AI's implementation computes global percentile edges ensuring each bin contains approximately 20% of data globally (verified: exact 0.200 per bin in output).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy and neural data are trimmed to the same length before binning, and then binned identically with the same 10-frame bins. They are sliced into the same trial windows using the same slice indices.

ii.
```python
neural, motion, tstamps = trim_to_common_length(neural, motion, tstamps)
neural_b = bin_array_2d(neural, BIN_FRAMES)
motion_b = bin_array_1d(motion, BIN_FRAMES)
# ...
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))
    motion_trials.append(motion_b[sl].astype(np.float32)[None, :])
```

iii. The AI documented that imaging and videography are synchronized (microscope triggers camera), so frame-level alignment is inherent. Trimming to the common length handles minor length mismatches.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles length mismatches between neural, motion energy, and timestamps arrays by trimming all to the minimum common length. Missing frames (as described in the data README) are not explicitly interpolated or treated as missing values; instead, the code simply truncates to the shortest array.

ii.
```python
def trim_to_common_length(neural, motion, tstamps):
    n = min(neural.shape[1], motion.shape[0], tstamps.shape[0])
    return neural[:, :n], motion[:n], tstamps[:n]
```

iii. The CONVERSION_NOTES.md documents: "Some sessions have neural/behavior length mismatches; conversion trims to shared minimum length before binning." The data README notes that missing frames can be identified from `tstamps.npy` or `interframe_int.npy` and suggests interpolation, but the AI chose truncation instead.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the raw data files (especially `F.npy`, `Fneu.npy`, and `spks.npy` which are large matrices). From the timing output, each session takes 0.1-0.7 seconds, with larger sessions (more neurons) taking longer. The total conversion for 41 sessions took ~16 seconds.

ii.
```python
# Timing output shows:
# processed jm031/2023-10-18_a in 0.14s: neurons=221, trials=10
# processed jm038/2023-04-30_a in 0.53s: neurons=685, trials=15
# processed jm039/2024-04-30_a in 0.64s: neurons=746, trials=15
```

iii. The AI did not document specific bottleneck analysis beyond total timing. The conversion is fast enough (~16 seconds total) that optimization was not needed.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `segment_session` function uses a Python loop to slice each trial window. This could be vectorized using `np.reshape` to split the entire array at once. The `percentile_bin_outputs` function also uses nested Python loops over sessions and trials.

ii.
```python
# segment_session loop:
for i in range(n_windows):
    sl = slice(i * bins_per_window, (i + 1) * bins_per_window)
    neural_trials.append(neural_b[:, sl].astype(np.float32))

# percentile_bin_outputs loop:
for sess in all_motion_trials:
    sess_out = []
    for m in sess:
        b = np.digitize(m.ravel(), edges[1:-1], right=False).astype(np.int64)
        sess_out.append(b[None, :])
```

iii. The AI did not document vectorization opportunities. However, since windows are equal-length, `segment_session` could reshape the array directly. Since total processing time is only ~16 seconds, the impact would be minimal.

## 6-c. What processing does the code repeat multiple times?

i. The code loads `F.npy` and `Fneu.npy` even though they are not used in the final conversion (since the AI switched to `spks.npy`). These files are loaded but never referenced after loading.

ii.
```python
def load_session(sess_dir):
    F = np.load(suite / 'F.npy').astype(np.float32)       # Loaded but unused
    Fneu = np.load(suite / 'Fneu.npy').astype(np.float32) # Loaded but unused
    spks = np.load(suite / 'spks.npy').astype(np.float32)
    # ...
    return F, Fneu, spks, iscell, ops, motion, tstamps
```

iii. This is a leftover from when the AI was using dF/F. The `compute_df_f` function also remains in the code but is never called.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `F.npy` and `Fneu.npy` files but doesn't use them (only `spks.npy` is used for neural data). The `compute_df_f` function exists in the code but is never called. The raw timestamps are binned (`_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)`) but the result is assigned to a throwaway variable and not used.

ii.
```python
F = np.load(suite / 'F.npy').astype(np.float32)       # Unused
Fneu = np.load(suite / 'Fneu.npy').astype(np.float32) # Unused

def compute_df_f(F, Fneu, ops):   # Never called
    # ...

_t_b_raw = bin_array_1d(tstamps, BIN_FRAMES)  # Assigned to throwaway variable
```

iii. These are artifacts from the AI's earlier approach using dF/F. The AI did not clean up these unused components after switching to `spks.npy`.
