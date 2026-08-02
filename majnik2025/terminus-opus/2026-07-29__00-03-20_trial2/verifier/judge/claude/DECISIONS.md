# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over mouse directories in `data/`, then over session subdirectories within each mouse. For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` from `move_deve/`. It does NOT load `interframe_int.npy` (used in the reference for dropped frame detection). It also does NOT filter directories by the `jm*` prefix.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
# ...
def load_session_data(session_dir):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The AI follows a standard directory traversal. It loads ops.npy to extract processing parameters (fs, neucoeff, win_baseline, sig_baseline) rather than hardcoding them. It does not load interframe_int.npy, instead using generic interpolation for frame count mismatches.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified as all subdirectories in the `data/` directory, sorted alphabetically. Unlike the reference, no `jm*` prefix filter is applied.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
```

iii. In practice, since the data directory only contains `jm*` directories (plus files like README.md and load_data.ipynb which are filtered by `os.path.isdir`), the result is the same as filtering by `jm*`. The AI noted 6 mice in its output.

## 1-c. How are the data split into sessions?

i. Sessions are identified as sorted subdirectories within each mouse directory. Each subdirectory corresponds to one daily recording session.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. Same approach as the reference. Sorting ensures deterministic ordering.

## 1-d. How are the data split into trials?

i. Trials are defined as 120-second (2-minute) non-overlapping segments of the continuous recording, at the binned 3 Hz rate (360 time bins per trial). This differs from the reference which uses 60-second trials at 30 Hz (1800 frames per trial).

ii.
```python
trial_duration_sec = 120
# ...
trial_bins = int(trial_duration_sec * effective_fs)  # 120 * 3 = 360
n_trials = n_bins // trial_bins
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. The AI chose 2-minute trials based on the paper's description of "5 fold splits on consecutive 2 minute blocks" for cross-validation. This produces 10 trials for 20-min sessions and 15 trials for 30-min sessions, consistent with the conversion output.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete trials are included; only remainder frames that don't fill a complete trial are discarded.

ii. N/A (no filtering code)

iii. The AI did not apply any trial quality filtering, consistent with the reference. There is no natural trial structure or quality metric for this continuous recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) in `suite2p/plane0/`. Processing parameters are read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. Same source variables as the reference (F.npy and Fneu.npy). The AI additionally loads ops.npy to dynamically get processing parameters.

## 2-b. How is the `neural` data processed?

i. The AI reimplements Suite2p's baseline correction manually using scipy: (1) neuropil subtraction (F - 0.7*Fneu), (2) Gaussian smoothing (sigma=10 frames), (3) minimum filter (window=1800 frames), (4) maximum filter (window=1800 frames), (5) baseline subtraction (Fc - Flow). After baseline correction, data is temporally binned by averaging groups of 10 frames (30 Hz to 3 Hz).

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

# Then binning:
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

iii. The AI documented matching Suite2p's baseline_maximin exactly. It went through multiple iterations to correct the filter order and window size. The 10-frame binning follows the paper's description: "averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons from F.npy are included.

ii. N/A (no filtering code)

iii. The AI noted in CONVERSION_NOTES.md that "All ROIs in iscell.npy are classified as cells (all have iscell[:,0]==1 and iscell[:,1]>0.5)," consistent with the paper's 0.5 threshold. Since all ROIs pass, no filtering changes the result.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Since trials are contiguous segments of continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity. The AI set off_start to 0.0 (reference uses None).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 10 frames are averaged into one bin, reducing the effective rate from 30 Hz to 3 Hz (333.33 ms bins). The reference does NOT apply rebinning and keeps the native 30 Hz rate (33.33 ms bins).

ii.
```python
bin_size = 10
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
# ...
'time_bin_size': 1000.0 * bin_size / 30.0,  # 333.33 ms
```

iii. The AI justifies this by citing the paper: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." This is documented in CONVERSION_NOTES.md Step 3.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from bin indices divided by the effective frame rate (3 Hz), giving time in seconds from the start of the session. It is not derived from any raw data variable.

ii.
```python
effective_fs = fs / bin_size  # 30 / 10 = 3 Hz
time_vec = np.arange(n_bins) / effective_fs
```

iii. Since the frame rate is constant and there are no timestamp files, computing time from indices is equivalent to actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the index divided by the effective frame rate (3 Hz after binning). For each trial, the time vector preserves the absolute time from session start (not relative to trial start).

ii.
```python
time_vec = np.arange(n_bins) / effective_fs
# ...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. The time values correctly represent elapsed time from session start in seconds. The maximum time for a 20-minute session is ~1199.7s and for 30-minute sessions ~1799.7s.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction - both use the same indexing into the binned time series. Each time bin corresponds to the same temporal position as the corresponding neural data bin.

ii.
```python
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. Since both neural and time data are sliced using the same indices after the same binning, they are inherently aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory. Unlike the reference, the AI does NOT use `interframe_int.npy` for dropped frame handling.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) linear interpolation to match neural frame count using np.interp, (2) temporal binning by averaging 10 consecutive frames, (3) discretization into 5 equal-percentile bins computed across all sessions. Notably, the AI does NOT normalize by standard deviation (which the reference does).

ii.
```python
def interpolate_missing_frames(me, n_target):
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp

# Binning:
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()

# Discretization:
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
```

iii. The AI chose generic linear interpolation rather than precise dropped-frame insertion. Since percentile-based binning is invariant to monotonic transformations, the omission of std normalization does not affect the final discrete bin assignments.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins using np.digitize. Bin edges are computed globally across all sessions. The AI explicitly sets the first edge to -inf and last to +inf, then clips to [0, n_bins-1].

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. The global percentile approach ensures balanced bin counts (each ~20%). The output distribution confirms exactly 20% per bin.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses `np.interp` to linearly interpolate the motion energy signal to match the neural frame count, then bins it by the same factor (10 frames). This differs from the reference which identifies specific dropped frames using interframe intervals and inserts interpolated values at those positions.

ii.
```python
me = interpolate_missing_frames(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
# Both neural and ME are then sliced with the same indices:
me_trial = me_binned[start:end].reshape(1, -1).astype(np.float32)
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. The camera is triggered by microscope acquisition at 30 Hz, so the signals should be synchronous. The AI's generic interpolation handles frame count mismatches but doesn't precisely identify which frames were dropped.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are handled by linearly interpolating the entire motion energy signal to match the neural frame count (using np.interp). Remainder frames that don't fill a complete trial are discarded. The AI does not use interframe_int.npy for precise dropped frame detection.

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

iii. This approach works but is less precise than the reference's method of detecting specific dropped frames via interframe intervals. The generic interpolation uniformly stretches the signal rather than inserting values at the exact missing positions.

## 6-a. What are the most time-consuming steps of the code?

i. The AI notes in CONVERSION_NOTES.md that the baseline correction (gaussian smooth + min/max filter) is the most time-consuming step. The full conversion took 66.6s for 41 sessions (~1.6s per session). The scipy-based implementation runs on CPU, unlike the reference which uses suite2p's GPU-accelerated dcnv.preprocess.

ii. N/A

iii. The baseline correction involves three sliding-window operations (gaussian, min, max) over the full session length for every neuron. The AI did not use GPU acceleration for this step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials to slice arrays, but this is efficient as it uses numpy slicing. No significant vectorization opportunities are present since the AI doesn't have the per-frame dropped-frame insertion loop that the reference has.

ii. N/A

iii. The code is already reasonably vectorized. The main operations (baseline correction, binning, discretization) all use vectorized numpy/scipy functions.

## 6-c. What processing does the code repeat multiple times?

i. The `--show-processing` mode reloads session data and recomputes dF/F for plotting sessions, repeating work already done during the main conversion. This recomputation is only done for visualization and doesn't affect the output.

ii.
```python
if args.show_processing:
    for si in range(n_plot_sessions):
        sess_data = load_session_data(sess_dir)
        dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)
```

iii. This redundancy only occurs when `--show-processing` is used and is limited to 2 sessions.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `ops.npy` for each session to extract processing parameters, but the parameters are the same defaults across all sessions (neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, fs=30.0). Loading ops.npy adds unnecessary I/O.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', 30.0)
neucoeff = ops.get('neucoeff', 0.7)
```

iii. While loading ops.npy is a reasonable practice (checking actual parameters), it adds small overhead. The parameters could have been hardcoded as in the reference.
