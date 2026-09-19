# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over all subdirectories in the `data/` directory (subjects), then iterating over all subdirectories within each subject (sessions). For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, `ops.npy` for parameters, and `motion_energy_glob.npy` from `move_deve/` for behavioral data. All sessions are collected into lists for batch processing.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
...
for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir)
                      if os.path.isdir(os.path.join(mouse_dir, d))])
    for sess in sessions:
        sess_dir = os.path.join(mouse_dir, sess)
        all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

```python
def load_session_data(session_dir):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The AI follows the standard directory convention: subject folders contain session subfolders, each with suite2p output and motion energy files. The AI also loads `ops.npy` to read session-specific parameters (fs, neucoeff, win_baseline, sig_baseline), whereas the reference hard-codes these values.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all subdirectories in the data directory, sorted alphabetically. Unlike the reference which filters by `jm*` prefix, the AI includes all directories.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
```

iii. The AI assumes all directories in the data folder are subjects. In practice this works because the data directory only contains `jm*` subject folders (other items are files, not directories). However, it does not explicitly filter by the `jm*` prefix.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a sorted subdirectory within a subject's folder. All subdirectories are included.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. Each subdirectory within a subject folder is treated as one recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as fixed-length non-overlapping segments of the continuous recording. The AI uses **120-second** (2-minute) trials, resulting in 360 time bins per trial (120s * 30Hz / 10 = 360 bins). The instructions specify 60-second trials.

ii.
```python
trial_duration_sec = 120
...
trial_bins = int(trial_duration_sec * effective_fs)  # 120 * 3 = 360
n_trials = n_bins // trial_bins
```

iii. The AI chose 120-second trials based on the paper's mention of "5 fold splits on consecutive 2 minute blocks" for cross-validation. However, the instructions explicitly state "Split sessions into 60-second trials." The AI's CONVERSION_NOTES.md (Step 5) documents: "Trials: 2-minute blocks (matching paper CV)."

## 1-e. How are trials filtered based on quality controls?

i. No trial quality filtering is applied. All trials that fit within the session duration are included. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
n_trials = n_bins // trial_bins
# No filtering code present
```

iii. There is no stimulus-driven trial structure in this dataset, so no quality-based trial filtering is applicable. The reference also does not filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), from `suite2p/plane0/`. The AI also loads `ops.npy` to get preprocessing parameters.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. These are the standard suite2p outputs. The reference also uses F.npy and Fneu.npy from the same path.

## 2-b. How is the `neural` data processed?

i. The AI applies: (1) neuropil subtraction (Fc = F - 0.7 * Fneu), (2) Gaussian smoothing (sigma=10 frames), (3) minimum filter (window = 60s * 30Hz = 1800 frames), (4) maximum filter (same window), (5) baseline subtraction (Fc - Flow). This is a manual reimplementation of suite2p's `maximin` baseline method using scipy filters, rather than calling `dcnv.preprocess()` directly.

ii.
```python
def compute_dff(F, Fneu, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, fs=30.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)  # 1800 frames
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff
```

iii. The AI states this matches Suite2p's `baseline_maximin` function. The processing steps (gaussian smooth -> min filter -> max filter -> subtract) match the suite2p algorithm conceptually, but the reference calls suite2p's own `dcnv.preprocess()` function directly, which may have subtle differences in edge handling and batching.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filtering is applied to neurons. All neurons present in the F.npy file are included.

ii. N/A (no filtering code)

iii. The AI notes that all ROIs in iscell.npy have iscell[:,0]==1, so no filtering is needed. The reference also does not filter neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed. `off_start` is set to 0.0 (reference uses None).

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments starting from session start. The reference also aligns to session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms time bins). This matches the paper and the reference.

ii.
```python
bin_size = 10
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

iii. The paper states "averaging in bins of 10 consecutive timestamps" for decoding analyses. Both the AI and reference implement this identically.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and bin duration, giving seconds from session start.

ii.
```python
effective_fs = fs / bin_size  # 3.0 Hz
time_vec = np.arange(n_bins) / effective_fs
...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. Since the frame rate is constant at 30 Hz, computing time from bin indices is equivalent to recording timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index / effective_fs` where `effective_fs = 30/10 = 3 Hz`. This gives time in seconds from session start for each bin. No additional processing.

ii.
```python
time_vec = np.arange(n_bins) / effective_fs
```

iii. The computation is equivalent to the reference's `(s + np.arange(trial_frames)) * BIN_FRAMES / FS`, yielding the same values.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is indexed using the same `start:end` slice as the neural data, so they are inherently aligned.

ii.
```python
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. Both time and neural data are sliced from the same session-length arrays using identical indices, ensuring perfect alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Unlike the reference, the AI does NOT load `interframe_int.npy` for dropped-frame detection.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) the motion energy is resampled via linear interpolation to match the neural frame count (using `np.interp`), (2) the resampled signal is averaged into 10-frame bins, (3) the binned signal is discretized into 5 equal-percentile bins computed **globally** across all sessions.

ii.
```python
def interpolate_missing_frames(me, n_target):
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp

def discretize_output(all_me_trials, n_bins=5):
    all_values = []
    for session_trials in all_me_trials:
        for trial in session_trials:
            all_values.append(trial.flatten())
    all_values = np.concatenate(all_values)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ...
```

iii. The AI uses uniform resampling for frame count mismatch rather than detecting specific dropped frames. The discretization is done globally rather than per-session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes motion energy into 5 equal-percentile bins using **global** percentiles computed across all sessions combined. The reference computes percentile edges **within each session separately**.

ii.
```python
def discretize_output(all_me_trials, n_bins=5):
    all_values = []
    for session_trials in all_me_trials:
        for trial in session_trials:
            all_values.append(trial.flatten())
    all_values = np.concatenate(all_values)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)
```

iii. The instructions state "Motion energy, discretized into five equal-percentile bins, selected per session." The AI chose global discretization, resulting in highly skewed per-session distributions (e.g., some sessions have 0% in certain bins). This is evident in the verification output where sessions 34-40 show 0% for bins 0-2.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy with neural data by resampling the motion energy signal to match the neural frame count using `np.interp` (uniform linear interpolation). The reference instead detects specific dropped video frames via `interframe_int.npy` and inserts interpolated values at those positions.

ii.
```python
def interpolate_missing_frames(me, n_target):
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp
```

iii. The AI treats the frame count mismatch as a uniform resampling problem, while the reference treats it as specific dropped frames that need to be filled in at specific positions. The uniform resampling approach slightly shifts the temporal alignment of all samples rather than inserting values only where frames were dropped.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames via uniform linear interpolation (`np.interp`) to match the neural frame count. Remainder frames at the end of sessions that don't fill a complete trial are discarded. The reference instead uses `interframe_int.npy` to detect exactly which frames were dropped and inserts interpolated values only at those positions.

ii.
```python
me_interp = np.interp(x_target, x_orig, me.astype(float))
```

iii. The AI's approach is simpler but less precise than the reference's dropped-frame detection method.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the manual baseline correction (`compute_dff`), which applies Gaussian smoothing, minimum filter, and maximum filter across the full session for every neuron using scipy. The reference uses suite2p's `dcnv.preprocess` which can leverage GPU acceleration.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. Processing took about 66.6s total for 41 sessions (about 1.6s per session), which is reasonable.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials and appends to lists. This could be vectorized with array reshaping. However, the loop is not a significant bottleneck.

ii.
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    ...
```

iii. The loop is simple slicing and not computationally expensive.

## 6-c. What processing does the code repeat multiple times?

i. When `--show-processing` is enabled, the code reloads the session data and recomputes dF/F for plotting, duplicating the processing already done during conversion.

ii.
```python
if args.show_processing:
    for si in range(n_plot_sessions):
        sess_data = load_session_data(sess_dir)
        dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)
```

iii. This is only triggered when the visualization flag is set, so it doesn't affect normal conversion performance.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ops.npy` for each session to read parameters (fs, neucoeff, win_baseline, sig_baseline), but these values are the same for all sessions and match the defaults. This is unnecessary I/O but not computationally expensive.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', 30.0)
neucoeff = ops.get('neucoeff', 0.7)
```

iii. Loading ops is a minor overhead. It could be considered defensive programming rather than unnecessary processing.
