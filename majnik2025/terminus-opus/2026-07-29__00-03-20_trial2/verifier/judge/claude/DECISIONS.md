# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating through all subdirectories in the `data/` directory (subjects), then iterating through each subject's subdirectories (sessions). For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` from `move_deve/`. Unlike the reference, it does not load `interframe_int.npy` for dropped frame detection.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])

def load_session_data(session_dir):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The AI documents loading F.npy, Fneu.npy, ops.npy, and motion_energy_glob.npy. It notes the data structure and file organization in CONVERSION_NOTES.md Steps 1-2. It reads processing parameters from ops.npy rather than hardcoding them.

## 1-b. How are the data split into subjects?

i. Subjects are identified as all subdirectories in the `data/` directory, sorted alphabetically. Unlike the reference, the AI does not filter by the `jm*` prefix.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
```

iii. The AI identifies subjects from CONVERSION_NOTES.md Step 2: "jm031/ (Mouse A) - 7 sessions, 221 neurons" through "jm046/ (Mouse F)". It finds 6 subjects, matching the paper. No explicit filtering by `jm*` prefix is applied, but all directories in the data folder happen to be subject directories.

## 1-c. How are the data split into sessions?

i. Sessions are identified as sorted subdirectories within each subject's folder. Each subdirectory contains one day's recording.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. The AI documents 41 total sessions across 6 subjects (7,7,7,7,6,7 sessions respectively) in CONVERSION_NOTES.md Step 9.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. The AI creates artificial trials of 120 seconds (2-minute blocks), based on the paper's description of cross-validation using "consecutive 2 minute blocks." Data is binned by 10 frames first (3 Hz), so each trial has `120 * 3 = 360` time bins. Remainder frames are discarded.

ii.
```python
trial_duration_sec = 120
trial_bins = int(trial_duration_sec * effective_fs)  # 120 * 3 = 360
n_trials = n_bins // trial_bins

for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. From CONVERSION_NOTES.md Step 5: "Trials: 2-minute blocks (matching paper CV)". The AI interpreted the paper's "5 fold splits on consecutive 2 minute blocks" as defining the trial duration. The reference solution uses 60-second trials instead.

## 1-e. How are trials filtered based on quality controls?

i. No trial quality filtering is applied. All complete trials are kept; only remainder frames that don't fill a full trial are discarded.

ii. N/A (no filtering code)

iii. No justification provided; there is no quality-based trial filtering described in the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. Processing parameters are read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. From CONVERSION_NOTES.md Step 1: "load_traces() ... Load F.npy raw fluorescence" and Step 5: "F.npy, Fneu.npy -> neural, Suite2p baseline correction, bin by 10."

## 2-b. How is the `neural` data processed?

i. The AI reimplements suite2p's baseline correction manually using scipy instead of calling `dcnv.preprocess` directly. The steps are: (1) neuropil subtraction with coefficient 0.7, (2) Gaussian smoothing (sigma=10 frames), (3) minimum filter (window=1800 frames), (4) maximum filter (window=1800 frames), (5) baseline subtraction. After baseline correction, data is temporally binned by averaging 10 consecutive frames (30 Hz -> 3 Hz).

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

dff_binned = bin_data(dff, bin_size=bin_size, axis=1)  # bin_size=10
```

iii. From CONVERSION_NOTES.md Step 1: "Suite2p baseline correction (maximin): gaussian smooth -> min filter -> max filter -> subtract, Window = win_baseline * fs = 60 * 30 = 1800 frames." Step 10 documents fixing the filter order and window size to match suite2p.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron quality filtering is applied. All neurons in F.npy are included.

ii. N/A (no filtering code)

iii. From CONVERSION_NOTES.md Step 1: "All ROIs in iscell.npy are classified as cells (all have iscell[:,0]==1 and iscell[:,1]>0.5)." The AI verified that all ROIs pass the quality threshold, so no filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the recording session. Since trials are contiguous segments of continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. From CONVERSION_NOTES.md: there is no stimulus event. The recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames, reducing the effective rate from 30 Hz to 3 Hz. The time bin size is 333.33 ms.

ii.
```python
bin_size = 10
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)

'time_bin_size': 1000.0 * bin_size / 30.0,  # 333.33 ms
```

iii. From CONVERSION_NOTES.md Step 3: "Binning: 10 frames" and from the paper: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from bin indices and the effective frame rate (3 Hz after binning). It is not derived from any raw data variable.

ii.
```python
effective_fs = fs / bin_size  # 3.0 Hz
time_vec = np.arange(n_bins) / effective_fs
```

iii. No explicit justification; time is straightforwardly computed from frame/bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as bin_index / effective_frame_rate, giving seconds from session start. Each trial's time vector represents the absolute time within the session.

ii.
```python
effective_fs = fs / bin_size  # 30/10 = 3.0
time_vec = np.arange(n_bins) / effective_fs
# ...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. No additional processing beyond index-to-time conversion.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction: the same bin indices are used for neural data and time. Both share the same temporal grid after binning.

ii.
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end]
    time_trial = time_vec[start:end].reshape(1, -1)
```

iii. Alignment is implicit through shared indexing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Unlike the reference, the AI does not load `interframe_int.npy` for dropped frame detection.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. From CONVERSION_NOTES.md Step 5: "motion_energy_glob.npy -> output[0], Interpolate, bin, discretize 5 bins."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) if ME length doesn't match neural length, linearly interpolate using `np.interp` to match, (2) bin by 10 frames (same as neural data), (3) discretize into 5 equal-percentile bins using `np.digitize`. The AI does NOT normalize ME by standard deviation before discretization (unlike the reference).

ii.
```python
def interpolate_missing_frames(me, n_target):
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp

me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()

# Discretization:
bin_edges = np.percentile(all_values, percentiles)
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
```

iii. From CONVERSION_NOTES.md Step 3: "Missing camera frames: interpolate" and Step 5: "ME discretization: 5 equal-percentile bins globally."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins. Bin edges are computed from all ME values across all sessions. `np.digitize` maps values to bins 0-4. The first and last edges are set to -inf/+inf to capture all values, and clipping ensures values stay in [0, 4].

ii.
```python
def discretize_output(all_me_trials, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    binned = np.digitize(trial.flatten(), bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. The output distribution shows approximately 20% per bin, confirming balanced discretization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. ME is interpolated to match neural frame count, then binned identically (10-frame bins). The same bin indices are used to slice both neural and ME data into trials.

ii.
```python
me = interpolate_missing_frames(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
# Same trial slicing as neural:
me_trial = me_binned[start:end]
```

iii. Alignment is achieved by ensuring ME and neural have the same number of frames before binning, then using identical indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames (ME shorter than neural) are handled by linear interpolation (`np.interp`) to stretch the ME signal to match the neural frame count. Remainder frames at session end that don't fill a complete trial are discarded. The AI does NOT use `interframe_int.npy` to detect specific dropped frames (unlike the reference).

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

iii. From CONVERSION_NOTES.md Step 3: "Missing camera frames: interpolate." The AI chose generic linear interpolation rather than using the interframe interval data to identify which specific frames were dropped.

## 6-a. What are the most time-consuming steps of the code?

i. The manual baseline correction (gaussian smoothing + min/max filtering over 1800-frame windows) is the most time-consuming step, running on CPU via scipy. Unlike the reference which uses suite2p's GPU-accelerated `dcnv.preprocess`, the AI's implementation runs entirely on CPU.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. From CONVERSION_NOTES.md Step 9: total processing time was tracked per session. The baseline correction with large window filters is inherently expensive.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The discretization loop iterates over sessions and trials individually, but this is minor. The main computation (baseline correction and binning) is already vectorized over neurons.

ii.
```python
for session_trials in all_me_trials:
    for trial in session_trials:
        binned = np.digitize(trial.flatten(), bin_edges[1:-1])
```

iii. No significant vectorization opportunities missed.

## 6-c. What processing does the code repeat multiple times?

i. In `--show-processing` mode, the code reloads raw data and recomputes dF/F for visualization after already having processed it during conversion. This duplicates the most expensive computation.

ii.
```python
# In show-processing block:
sess_data = load_session_data(sess_dir)  # reload
dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)  # recompute
```

iii. No explicit justification for the duplication. The visualization code is separate from the conversion pipeline.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins the data by 10 frames (30 Hz -> 3 Hz). While this matches the paper's decoding analysis, the reference solution keeps the data at 30 Hz. The binning is an additional processing step that the reference does not perform. Also, in `--show-processing` mode, extensive visualization data is computed and plotted.

ii.
```python
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

iii. The AI justified binning based on the paper's methods: "averaging in bins of 10 consecutive timestamps." The reference solution leaves binning to the downstream decoder.
