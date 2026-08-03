# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all 6 mouse directories in `data/`, then iterates over all session subdirectories within each mouse. For each session, it loads `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `ops.npy` (Suite2p metadata), and `motion_energy_glob.npy` (behavioral data) from the standard Suite2p and `move_deve` subdirectory structure. No `iscell.npy` filtering is applied because the agent verified all ROIs are already classified as cells.

ii.
```python
def load_session_data(session_dir):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    ...
```

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

iii. The AI verified this structure by reading `data/README.md` and the `load_data.ipynb` notebook, which describe the directory hierarchy. The AI confirmed that all ROIs in `iscell.npy` are already classified as cells (all have iscell[:,0]==1), so no cell filtering was needed at load time. The data provided by Track2p already contains only tracked neurons.

## 1-b. How are the data split into subjects (mice)?

i. Each top-level directory under `data/` corresponds to one mouse subject (jm031 through jm046). The AI sorts these alphabetically and uses the directory names as subject identifiers. Subject indices are assigned based on sorted order.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
...
'subjects': unique_mice,  # ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
'subject_idx': subject_idx_arr,
```

iii. The AI documented that there are 6 mice matching the paper's "full dataset of 6 mice." The mapping follows the paper's alphabetical naming convention (jm031 = Mouse A, jm032 = Mouse B, etc.) as described in the data README.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a mouse folder is one session (one recording day). Sessions are sorted chronologically. Each session becomes one entry in the output `neural`, `input`, and `output` lists.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
for sess in sessions:
    sess_dir = os.path.join(mouse_dir, sess)
    all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

iii. The AI found 41 total sessions (7+7+7+7+6+7 across mice), matching expectations from the paper (minimum 6 consecutive days per mouse).

## 1-d. How are the data split into trials?

i. The AI splits each session's continuous recording into fixed-length 2-minute (120-second) "trials." After binning at 3 Hz (bin_size=10 from 30 Hz), each trial has 360 timepoints. Sessions with 36,000 frames (20 min) yield 10 trials; sessions with 54,000 frames (30 min) yield 15 trials. Any remainder frames at the end are discarded.

ii.
```python
trial_bins = int(trial_duration_sec * effective_fs)  # 120 * 3 = 360 bins
n_trials = n_bins // trial_bins

for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    ...
```

iii. The AI chose 2-minute blocks based on the paper's description of "5 fold splits on consecutive 2 minute blocks" for cross-validation. This creates pseudo-trials from the continuous recording for the decoder framework.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All generated trials are included in the output dataset. The AI does not check for or exclude trials with missing data, artifacts, or other quality issues.

ii. There is no filtering code. All trials produced by the splitting loop are appended directly:
```python
neural_trials.append(neural_trial)
input_trials.append(time_trial)
me_trials.append(me_trial)
```

iii. The AI verified that all ROIs are classified as cells and that motion energy frames match or can be interpolated to match neural frames. Since the data is from continuous spontaneous activity recordings (no task structure with invalid periods), the AI determined no trial filtering was necessary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence traces) from each session's `suite2p/plane0/` directory. Parameters from `ops.npy` (frame rate, neuropil coefficient, baseline parameters) are also used.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. The AI noted that `spks.npy` (deconvolved spikes) was also available but chose `F.npy` and `Fneu.npy` because the paper states they used "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." This requires computing dF/F from raw fluorescence rather than using pre-computed deconvolved spikes.

## 2-b. How is the `neural` data processed?

i. The processing pipeline is:
1. Neuropil correction: Fc = F - 0.7 * Fneu
2. Gaussian smoothing of Fc (sigma=10 frames)
3. Minimum filter (window = 1800 frames = 60 seconds * 30 Hz)
4. Maximum filter (window = 1800 frames)
5. Baseline subtraction: dff = Fc - Flow (NOT division)
6. Temporal binning: average 10 consecutive frames (30 Hz -> 3 Hz)

ii.
```python
def compute_dff(F, Fneu, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, fs=30.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)  # 60 * 30 = 1800 frames
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff

def bin_data(data, bin_size=10, axis=-1):
    ...
    return data_trimmed.reshape(new_shape).mean(axis=axis + 1)
```

iii. The AI went through multiple iterations to get this right. Initially implemented max-then-min filter order and used division (Fc-F0)/F0. After comparing with Suite2p's `baseline_maximin` function in `dcnv.py` and Track2p's `F_processing` function, corrected to gaussian->min->max order with subtraction only. Verified the scipy implementation matches Suite2p's torch implementation with mean absolute difference < 0.06.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All neurons present in the tracked data are included. The AI verified that all ROIs in `iscell.npy` already have `iscell[:,0]==1` and probability > 0.5.

ii. No filtering code exists. All neurons from `F.npy` are processed:
```python
n_neurons, n_frames = F.shape
# All neurons are used directly
```

iii. The AI justified this by noting that the Track2p pipeline already outputs only successfully tracked neurons across all days, meaning the data is pre-curated. The paper mentions "default threshold of 0.5 as true cells" but the AI found all neurons already pass this threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Trials are consecutive 2-minute segments from the beginning of each session. The temporal alignment event is "start of recording session" with `off_start=0.0`.

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

And in the trial splitting:
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end]
```

iii. Since there is no task structure (spontaneous activity recordings), there is no natural event to align to. The AI chose the session start, which is the only meaningful reference point. Each trial's time input starts from 0 and goes to the session-relative time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The original data is at 30 Hz. Temporal rebinning is applied by averaging 10 consecutive frames, resulting in 3 Hz (333.33 ms bins). This matches the paper's description.

ii.
```python
bin_size = 10
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
...
'time_bin_size': 1000.0 * bin_size / 30.0,  # 333.33 ms
```

iii. The AI followed the paper's specification: "averaging in bins of 10 consecutive timestamps" at 30 Hz imaging rate, yielding 3 Hz effective rate.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not derived from any raw data variable. It is computed synthetically from the frame indices and the effective sampling rate after binning.

ii.
```python
effective_fs = fs / bin_size  # 30 / 10 = 3 Hz
time_vec = np.arange(n_bins) / effective_fs
...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. The AI computed time as frame_index / effective_frame_rate, giving time in seconds from the start of the recording session.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `frame_index / effective_sampling_rate` where effective_sampling_rate = 30/10 = 3 Hz. For each trial, the time values represent absolute time from the session start (not trial-relative time). Trial 0 starts at 0s, trial 1 starts at 120s, etc.

ii.
```python
time_vec = np.arange(n_bins) / effective_fs
...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. The AI chose absolute time from session start rather than trial-relative time. This means the input captures both within-trial timing and the trial's position within the session.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because it is computed from the same frame indices used for the neural data after binning. Each time bin corresponds to the same bin of neural data.

ii.
```python
# Both use the same indexing
neural_trial = dff_binned[:, start:end]
time_trial = time_vec[start:end].reshape(1, -1)
```

iii. Since the time vector and neural data share the same temporal grid (both derived from the same binned frame indices), alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in each session's `move_deve/` directory.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The AI identified this as the processed behavioral data (motion energy extracted from videography of spontaneous behaviour), as described in the data README.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps:
1. Interpolate motion energy to match neural frame count if lengths differ (handles missing camera frames)
2. Temporally bin by averaging 10 consecutive frames (same as neural data)
3. Discretize into 5 equal-percentile bins computed globally across all sessions

ii.
```python
me = interpolate_missing_frames(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
def discretize_output(all_me_trials, n_bins=5):
    all_values = np.concatenate([...])
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ...
    binned = np.digitize(trial.flatten(), bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. The AI chose global percentile binning (across all sessions) rather than per-session normalization. The AI considered per-session normalization but decided against it, noting that equal-percentile bins effectively perform rank normalization. This results in skewed per-session distributions (e.g., jm046 sessions have mostly high bins, jm031 mostly low bins) because younger mice are less active.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins using global percentile thresholds. The bins are computed so that each bin contains exactly 20% of all data points across all sessions. `np.digitize` assigns each value to a bin, with boundary values set to -inf and +inf.

ii.
```python
def discretize_output(all_me_trials, n_bins=5):
    all_values = np.concatenate([...])
    percentiles = np.linspace(0, 100, n_bins + 1)  # [0, 20, 40, 60, 80, 100]
    bin_edges = np.percentile(all_values, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ...
    binned = np.digitize(trial.flatten(), bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins." The AI implemented this with global percentiles, resulting in exactly 20% per bin globally. Per-session distributions are naturally skewed due to developmental differences in mouse activity levels.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first interpolated to match the neural frame count (handling missing camera frames), then binned with the same bin size (10 frames) as the neural data, and finally split into trials using the same trial boundaries. This ensures temporal alignment.

ii.
```python
me = interpolate_missing_frames(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
# Same trial boundaries as neural data
me_trial = me_binned[start:end]
```

iii. The AI noted that some sessions have mismatched motion energy and neural frame counts due to missing camera frames, and used linear interpolation to align them before binning and trial splitting.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where `motion_energy_glob.npy` length doesn't match neural frame count) are handled by linear interpolation of the motion energy signal to match the neural frame count. No other missing data handling is implemented.

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

iii. The data README mentions that "In some recordings there might be some missing frames from the camera" and suggests interpolating over them. The AI followed this recommendation. During initial development, the AI also encountered issues with neurons having negative corrected fluorescence (F < 0.7*Fneu for all timepoints), but after switching from division to subtraction for baseline correction, this was no longer problematic.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the `compute_dff` function, specifically the minimum and maximum filters with a 1800-frame window applied to all neurons. Each session takes 0.5-2.6 seconds depending on neuron count, with total processing time of ~67 seconds for all 41 sessions.

ii. From the output log:
```
Processing session 22/41: jm039/2024-04-30_a... 746 neurons, 15 trials, 2.6s
...
Total processing time: 66.6s
```

iii. The AI tracked per-session timing and estimated full conversion time. The total time (66.6s) was well under the 15-minute threshold mentioned in the instructions, so no optimization was deemed necessary.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially, creating separate arrays for each trial. This could potentially be done with a single reshape operation. The discretization loop over sessions and trials could also be vectorized.

ii.
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    ...
```

```python
for session_trials in all_me_trials:
    session_output = []
    for trial in session_trials:
        binned = np.digitize(trial.flatten(), bin_edges[1:-1])
        ...
```

iii. The AI noted the need for efficiency but the total runtime was fast enough (~67 seconds) that optimization was not prioritized. The main numerical operations (dF/F computation, binning) are already vectorized across neurons.

## 6-c. What processing does the code repeat multiple times?

i. In `--show-processing` mode, the code reloads session data and recomputes dF/F for the plotting sessions, duplicating work already done during the main processing loop. The `load_session_data` function is called twice for sessions being plotted.

ii.
```python
if args.show_processing:
    for si in range(n_plot_sessions):
        mouse, mouse_idx, sess_dir, sess_name = all_sessions[si]
        sess_data = load_session_data(sess_dir)  # Reloads data
        ...
        dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)  # Recomputes dF/F
```

iii. The AI did not explicitly address this redundancy. In normal (non-plotting) mode, each session's data is loaded and processed exactly once, so this only affects the `--show-processing` flag.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes time as absolute seconds from session start for the input. However, the input values range from 0 to 1799.7 seconds, and the time values reset differently per trial (trial 0 starts at 0, trial 1 at 120, etc.). This absolute time encoding means the decoder must learn both within-trial dynamics and trial identity from a single monotonically increasing value, which may not be the most useful representation. Additionally, remainder frames at the end of sessions (after the last complete trial) are computed through the dF/F pipeline but then discarded during trial splitting.

ii.
```python
time_vec = np.arange(n_bins) / effective_fs  # Absolute time
...
# Remainder frames discarded:
n_trials = n_bins // trial_bins  # Integer division drops remainder
```

iii. The AI did not explicitly discuss the efficiency of computing dF/F for frames that are later discarded. The amount of wasted computation is small (at most 119 seconds of data per session, or ~360 frames out of 36000-54000).
