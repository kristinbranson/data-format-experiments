# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory, iterates over each subject's session subdirectories, and loads per-session data files: `F.npy`, `Fneu.npy`, `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. All subjects and sessions are discovered via sorted directory listing.

ii.
```python
def get_subjects_and_sessions(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions

def load_session_data(data_dir, subject, session):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
    return F, Fneu, ops, me, tstamps
```

iii. The AI loads all subjects and sessions discovered in the data directory. It additionally loads `ops.npy` (to read Suite2p parameters) and `tstamps.npy` (for motion energy alignment), which the reference solution does not load. The CONVERSION_NOTES document that the directory structure follows a standard convention with subject folders containing session subfolders.

## 1-b. How are the data split into subjects?

i. Subjects are identified as directories starting with `jm` in the data directory, sorted alphabetically. All 6 subjects are included.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Sessions are identified as subdirectories within each subject's folder, sorted alphabetically. Each subdirectory contains one recording session's data.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as 60-second non-overlapping segments of the continuous recording. After binning into 10-frame bins at 30 Hz (3 Hz effective rate), each trial has 180 timepoints. Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
TIMEPOINTS_PER_TRIAL = int(TRIAL_DURATION_SEC * BINNED_RATE)  # 180

def split_into_trials(data_2d, timepoints_per_trial):
    n_timepoints = data_2d.shape[-1]
    n_trials = n_timepoints // timepoints_per_trial
    trials = []
    for t in range(n_trials):
        start = t * timepoints_per_trial
        end = start + timepoints_per_trial
        if data_2d.ndim == 2:
            trials.append(data_2d[:, start:end].copy())
        else:
            trials.append(data_2d[start:end].copy())
    return trials
```

iii. Per the task specification, sessions are split into 60-second trials. The CONVERSION_NOTES confirm 20 trials for 20-minute sessions and 30 trials for 30-minute sessions, with 180 timepoints each.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped. No other trial-level quality filtering is applied.

ii.
```python
if len(neural_trials) < 2:
    print(f"  WARNING: Session {subj}/{sess} has only {len(neural_trials)} trials, skipping")
    continue
```

iii. The minimum 2-trial check ensures decoder evaluation is possible. No trial-level quality criteria are applied since the continuous recording has no discrete trial structure.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The AI also loads `ops.npy` to read Suite2p processing parameters.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - 0.7 * Fneu`), followed by Suite2p's `preprocess` function which performs baseline estimation and correction using the `maximin` method. Parameters are read from the `ops.npy` file with fallback defaults matching Suite2p defaults.

ii.
```python
def compute_dff(F, Fneu, ops):
    Fc = F.copy() - NEUCOEFF * Fneu
    dff = preprocess(
        F=Fc.copy(),
        baseline=ops.get('baseline', 'maximin'),
        win_baseline=ops.get('win_baseline', 60.0),
        sig_baseline=ops.get('sig_baseline', 10.0),
        fs=ops.get('fs', 30.0),
        prctile_baseline=ops.get('prctile_baseline', 8.0),
        device=device
    )
    return dff
```

iii. The CONVERSION_NOTES state: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)". Parameters are read from ops.npy rather than hardcoded.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the suite2p `F.npy` output are included. The CONVERSION_NOTES document that all cells in the data already have `iscell = 1.0` because Track2p outputs only tracked cells.

ii. N/A (no filtering code)

iii. The CONVERSION_NOTES explain: "All cells in the data are tracked cells (iscell all = 1.0)" and "Track2p filters cells using iscell_thr (probability threshold from Suite2p classifier)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),
```

iii. There is no stimulus event to align to. The recording is continuous.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both the neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bin). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10  # frames per bin
FRAME_RATE = 30  # Hz

def bin_data(data, bin_size, axis=-1):
    n = data.shape[axis]
    n_bins = n // bin_size
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, n_bins * bin_size)
    data_trunc = data[tuple(slices)]
    new_shape = list(data_trunc.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)
    data_reshaped = data_trunc.reshape(new_shape)
    data_binned = data_reshaped.mean(axis=axis + 1)
    return data_binned
```

iii. The CONVERSION_NOTES quote the paper: "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the trial index and the binned rate, giving seconds from the start of the session.

ii.
```python
for t in range(n_trials):
    trial_start_sec = t * TRIAL_DURATION_SEC
    time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
    input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))
```

iii. Since the frame rate is constant at 30 Hz and bins are of fixed size (10 frames), time can be computed purely from indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `trial_start_seconds + bin_index / binned_rate`. For trial 0, this gives `[0.0, 0.333, 0.667, ...]`. For trial 1, this gives `[60.0, 60.333, ...]`. The result is converted to float32.

ii.
```python
trial_start_sec = t * TRIAL_DURATION_SEC
time_in_trial = trial_start_sec + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE
input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))
```

iii. This produces the same values as computing from bin indices, since `bin_index / BINNED_RATE = bin_index * BIN_SIZE / FRAME_RATE`.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is directly computed from the same bin indices used for neural data, so alignment is inherent. Each time value corresponds to the left edge of the corresponding neural data time bin.

ii. (Same code as 3-b)

iii. Since both the neural data and time use the same indexing scheme, they are inherently aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Camera timestamps from `tstamps.npy` are used for alignment when frames are missing.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) When camera frames are missing, the motion energy is interpolated to match neural frame count using `np.interp` with camera timestamps. (2) The trace is averaged into 10-frame bins. (3) The binned signal is discretized into 5 equal-percentile bins computed per session using `np.digitize`.

ii.
```python
# Step 1: Alignment/interpolation
def align_motion_energy(me, tstamps, n_neural_frames):
    if n_cam_frames == n_neural_frames:
        return me.astype(np.float64)
    neural_times = np.arange(n_neural_frames) / FRAME_RATE
    cam_times = tstamps * 1000.0
    me_aligned = np.interp(neural_times, cam_times, me.astype(np.float64))
    return me_aligned

# Step 2: Binning
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE, axis=1).flatten()

# Step 3: Discretization
def discretize_motion_energy(me_binned_trials, n_bins=5):
    all_me = np.concatenate(me_binned_trials)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me, percentiles)
    for me_trial in me_binned_trials:
        bin_indices = np.digitize(me_trial, bin_edges[1:-1], right=False)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
```

iii. The CONVERSION_NOTES explain that interpolation handles dropped camera frames, binning matches the paper's specification, and per-session percentile bins follow the task specification.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session. Percentile boundaries are computed from all trials within a session using `np.linspace(0, 100, 6)` to get quintile edges, then `np.digitize` maps values to bins 0-4. Values are clipped to the valid range.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_me, percentiles)
bin_indices = np.digitize(me_trial, bin_edges[1:-1], right=False)
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
```

iii. The task specification requires "discretized into five equal-percentile bins, selected per session." The AI computes bin edges from all trials of a session combined, then applies to each trial.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When the camera frame count differs from the neural frame count (due to dropped frames), the AI uses `np.interp` with camera timestamps (`tstamps.npy`) to interpolate the motion energy to match the neural frame count. When frame counts match, no interpolation is needed.

ii.
```python
def align_motion_energy(me, tstamps, n_neural_frames):
    n_cam_frames = len(me)
    if n_cam_frames == n_neural_frames:
        return me.astype(np.float64)
    neural_times = np.arange(n_neural_frames) / FRAME_RATE
    cam_times = tstamps * 1000.0  # kiloseconds to seconds
    me_aligned = np.interp(neural_times, cam_times, me.astype(np.float64))
    return me_aligned
```

iii. The AI uses continuous interpolation via `np.interp` with timestamp information, whereas the reference solution uses `interframe_int.npy` to detect dropped frames at specific positions and inserts interpolated values there. Both achieve the same goal of aligning motion energy to neural data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of data issues are handled: (1) Missing camera frames are detected by comparing camera frame count to neural frame count, and interpolated using timestamps. (2) Remainder bins at the end of a session that don't fill a complete 60-second trial are discarded. (3) Sessions with fewer than 2 trials are skipped.

ii.
```python
# Missing camera frames
me_aligned = align_motion_energy(me, tstamps, n_neural_frames)

# Incomplete last trial (in split_into_trials)
n_trials = n_timepoints // timepoints_per_trial  # integer division drops remainder

# Minimum trial check
if len(neural_trials) < 2:
    print(f"  WARNING: Session {subj}/{sess} has only {len(neural_trials)} trials, skipping")
    continue
```

iii. The CONVERSION_NOTES document that 8 sessions have missing camera frames (1-148 frames missing). The interpolation approach ensures both streams have matching lengths.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p `preprocess` (baseline correction) function, which takes 0.07-0.47s per session depending on neuron count and GPU availability. Data loading is the second most expensive step (0.05-0.33s).

ii.
```python
dff = preprocess(
    F=Fc.copy(),
    baseline=ops.get('baseline', 'maximin'),
    ...
    device=device
)
```

iii. The conversion output shows per-session timing breakdowns. The baseline correction involves sliding window operations across all neurons for the full session length.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `split_into_trials` function uses a Python loop to split data into trials, creating individual array copies. This could potentially be done with a single reshape operation (though the copies are intentional for memory independence).

ii.
```python
def split_into_trials(data_2d, timepoints_per_trial):
    for t in range(n_trials):
        start = t * timepoints_per_trial
        end = start + timepoints_per_trial
        if data_2d.ndim == 2:
            trials.append(data_2d[:, start:end].copy())
        else:
            trials.append(data_2d[start:end].copy())
    return trials
```

iii. The loop is simple and fast relative to the baseline correction. The performance impact is negligible.

## 6-c. What processing does the code repeat multiple times?

i. The `F.copy()` is called once, and then `Fc.copy()` is passed to preprocess, making two copies of the fluorescence data. This is not a significant repetition but is unnecessary duplication.

ii.
```python
Fc = F.copy() - NEUCOEFF * Fneu
dff = preprocess(F=Fc.copy(), ...)
```

iii. The double copy is a minor inefficiency. The expression `F.copy() - NEUCOEFF * Fneu` already creates a new array, so the initial `.copy()` is redundant.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ops.npy` and `tstamps.npy` for every session. The `ops.npy` values end up being the same as the hardcoded defaults, making the load unnecessary. The `tstamps.npy` is only needed for sessions with missing camera frames (~8 out of 41 sessions) but is loaded for all sessions.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. Loading `ops.npy` for every session adds a small overhead. In principle, reading parameters from ops is more robust than hardcoding, but in practice the values are the same across all sessions.
