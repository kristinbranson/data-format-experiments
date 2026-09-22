# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory. For each subject, sessions are subdirectories sorted alphabetically. Neural data is loaded from suite2p output files (`F.npy`, `Fneu.npy` in `suite2p/plane0/`), and motion energy from `motion_energy_glob.npy` in the `move_deve/` subdirectory. All data is loaded per-session in nested loops over subjects and sessions.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
...
for si, subject in enumerate(subjects):
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, d))])
    for sess in sessions:
        ...
        F = np.load(os.path.join(s2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
        me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The AI examined the data directory structure and identified the convention of `jm*` subject folders with session subfolders. This follows the standard organization from the paper.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data directory, sorted alphabetically.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically.

ii.
```python
sessions = sorted([d for d in os.listdir(subject_dir)
                  if os.path.isdir(os.path.join(subject_dir, d))])
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. Sessions are split into 60-second non-overlapping trials. After binning (10 frames per bin at 30 Hz), each trial has 180 time bins. Any remainder bins that don't fill a complete trial are discarded. Sessions with fewer than 2 trials are skipped.

ii.
```python
bins_per_trial = int(TRIAL_DUR * FS / BIN_SIZE)  # 180
...
n_trials = n_bins_total // bins_per_trial
if n_trials < 2:
    print(f"Skipping {subject}/{sess}: only {n_trials} trials")
    continue
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
```

iii. The instructions specify "Split sessions into 60-second trials." The AI chose non-overlapping segments from the continuous recording, discarding incomplete final segments.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials are skipped (n_trials < 2 check). No other trial-level filtering is applied.

ii.
```python
if n_trials < 2:
    print(f"Skipping {subject}/{sess}: only {n_trials} trials")
    continue
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI implemented this check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from the `suite2p/plane0/` subdirectory of each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces, as described in the paper.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`Fc = F - 0.7 * Fneu`), then computes a maximin baseline using Gaussian smoothing (sigma=10 frames), minimum filter (window=1800 frames = 60s at 30Hz), and maximum filter (same window). The baseline is subtracted from the neuropil-corrected trace (`dff = Fc - Flow`). This is a manual reimplementation of suite2p's `dcnv.preprocess` function rather than calling it directly.

ii.
```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff
```

iii. The AI initially tried dF/F with division by Flow, which caused numerical instability (huge values when Flow was near zero). After investigating suite2p's source and the track2p code, the AI determined that suite2p's `preprocess` just does baseline subtraction (`Fc - Flow`) without division, and updated accordingly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI notes that iscell is all 1s since data is from Track2p (only neurons tracked across all days).

ii. N/A (no filtering code)

iii. The AI's docstring states: "No additional neuron filtering: Data is already from Track2p (only neurons tracked across all days for each mouse). iscell is all 1s."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Start of imaging session',
'off_start': 0.0,
'off_end': float(TRIAL_DUR),
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10       # frames to average
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
...
'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33
```

iii. The paper's Methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index and the bin duration, giving seconds from the start of the session.

ii.
```python
time_bin_dur = BIN_SIZE / FS  # seconds per bin
time_vec = np.arange(n_bins_total) * time_bin_dur
```

iii. Since the frame rate is constant at 30 Hz and time bin size is known, computing time from bin indices is equivalent to recording timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the bin index multiplied by the bin duration (10/30 = 1/3 seconds per bin). This gives the time at the left edge of each bin.

ii.
```python
time_bin_dur = BIN_SIZE / FS  # seconds per bin = 0.3333...
time_vec = np.arange(n_bins_total) * time_bin_dur
```

iii. No additional processing beyond simple arithmetic.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural data share the same bin indices, so they are inherently aligned. The time vector is sliced to the same trial boundaries as neural and output data.

ii.
```python
session_input.append(time_vec[start:end].reshape(1, -1))
```

iii. Both are indexed by the same bin positions, so alignment is automatic.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) interpolation to match neural frame count when frames are missing, (2) averaging into 10-frame bins, (3) discretization into 5 equal-percentile bins per session using `np.digitize`.

ii.
```python
me = interpolate_me(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
...
percentiles = np.linspace(0, 100, N_BINS_OUTPUT + 1)
bin_edges = np.percentile(me_binned, percentiles)
me_discrete = np.digitize(me_binned, bin_edges[1:-1])
```

iii. Interpolation aligns ME to neural frames, binning matches the paper's decoding method, and 5-quintile discretization follows the instructions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins computed per session. `np.percentile` computes edges at 0%, 20%, 40%, 60%, 80%, 100%, then `np.digitize` assigns each value to bins 0-4.

ii.
```python
percentiles = np.linspace(0, 100, N_BINS_OUTPUT + 1)
bin_edges = np.percentile(me_binned, percentiles)
me_discrete = np.digitize(me_binned, bin_edges[1:-1])
```

iii. The instructions specify "Motion energy, discretized into five equal-percentile bins, selected per session."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses `np.interp` to linearly interpolate the motion energy trace to match the neural frame count when they differ (due to dropped camera frames). After interpolation, both streams have the same length and are binned/sliced identically.

ii.
```python
def interpolate_me(me, target_len):
    if len(me) == target_len:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_orig, me)
```

iii. The AI's trajectory shows it recognized that ME can be shorter than neural data due to dropped camera frames. Rather than using `interframe_int.npy` to identify specific dropped frames, the AI used generic linear interpolation to stretch ME to the target length.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (ME shorter than neural data) are handled by linear interpolation of the motion energy to match neural frame count. Sessions with fewer than 2 trials are skipped. Remainder frames at end of session that don't fill a complete trial are discarded.

ii.
```python
me = interpolate_me(me, n_frames)
...
if n_trials < 2:
    print(f"Skipping {subject}/{sess}: only {n_trials} trials")
    continue
```

iii. The AI chose a general interpolation approach rather than specifically detecting dropped frames via interframe intervals.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction (Gaussian filtering, minimum/maximum filtering over all neurons for all sessions), which runs on CPU using scipy. Unlike the reference solution which uses suite2p's GPU-accelerated `dcnv.preprocess`, the AI's manual implementation is CPU-only.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The filtering operations over full-length session data for all neurons are computationally intensive, especially without GPU acceleration.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials one at a time, creating slices. This is already efficient since it just creates views. The main inefficiency is the per-session processing loop which is inherent to the data structure.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end])
```

iii. The loop creates array slices (views), which is already efficient.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each session is processed once.

ii. N/A

iii. The code has a single-pass architecture: load, process, bin, discretize, split into trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes the full session length including remainder frames that are discarded when they don't fill a complete trial. This is minor since at most 59 seconds of data per session is wasted.

ii.
```python
n_trials = n_bins_total // bins_per_trial
# remainder bins are implicitly discarded
```

iii. Processing the entire session before splitting into trials means the baseline correction and binning are applied to frames that may not end up in any trial.
