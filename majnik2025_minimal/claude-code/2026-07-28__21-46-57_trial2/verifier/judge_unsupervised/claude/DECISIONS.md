# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all subject directories (folders starting with 'jm') in the data directory, then iterates over all session subdirectories within each subject. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/` for behavioral data. All sessions are processed in a first pass to collect data and motion energy values, then a second pass builds the output structure.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess_name in sessions:
        sess_dir = os.path.join(subj_dir, sess_name)
        # load_and_process_session loads F.npy, Fneu.npy, motion_energy_glob.npy, tstamps.npy
```

iii. The AI's CONVERSION_NOTES.md states: "For each subject there is a folder corresponding to the subject id" and lists 6 subjects with 6-7 sessions each, 41 sessions total. The approach follows the data README's directory structure.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified as directories starting with 'jm' in the data directory. Each subject directory contains session subdirectories. The subject list is sorted alphabetically, and a `subject_idx` array maps each session to its subject.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
# ...
subj_i = subjects.index(subj)
subject_idx.append(subj_i)
```

iii. The AI identified 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046) matching the paper's description of 6 mice. From CONVERSION_NOTES: "6 subjects: jm031, jm032, jm038, jm039, jm040, jm046."

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder corresponds to one recording session (one day). Sessions are sorted alphabetically (which corresponds to chronological order since folder names are dates in YYYY-MM-DD format). Each session becomes one entry in the output lists.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
for sess_name in sessions:
    sess_dir = os.path.join(subj_dir, sess_name)
    dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
```

iii. From CONVERSION_NOTES: "6-7 daily sessions per subject (41 sessions total)." The data README confirms each session folder corresponds to one recording day.

## 1-d. How are the data split into trials?

i. Each session's continuous recording is split into consecutive 2-minute (120-second) blocks. With 30 Hz imaging and 10-frame bins, each trial has 360 bins. The remainder at the end of a session is discarded. 20-minute sessions yield 10 trials, 30-minute sessions yield 15 trials.

ii.
```python
TRIAL_DURATION_S = 120  # 2 minutes per trial
def split_into_trials(dff_binned, me_binned, time_bins, trial_duration_s=TRIAL_DURATION_S):
    bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end])
```

iii. From CONVERSION_NOTES: "Following the paper's cross-validation approach: 'splits were done on consecutive 2 minute blocks of the recording.' Trial duration: 2 minutes = 120 seconds. Bins per trial: 120s * 30Hz / 10 frames = 360 bins."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks are included. Only the remainder frames at the end of a session (less than one full 2-minute block) are discarded.

ii.
```python
n_trials = n_bins // bins_per_trial  # integer division discards remainder
```

iii. There is no explicit discussion of trial filtering in the CONVERSION_NOTES or trajectory. The AI included all complete trials without quality checks. The paper does not describe trial-level filtering for the decoding analysis; the 2-minute blocks are used directly as cross-validation folds.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence traces) in the Suite2p output directory for each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. From CONVERSION_NOTES: "Following the paper's methods ('We used baseline corrected fluorescence traces as our dF/F using the default Suite2p parameters')." The paper explicitly specifies dF/F rather than deconvolved spikes (spks.npy).

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F using Suite2p's default "maximin" baseline method:
1. Neuropil correction: `Fc = F - 0.7 * Fneu`
2. Gaussian smoothing with sigma = 300 frames (10s * 30Hz)
3. Running minimum with window = 1800 frames (60s * 30Hz)
4. Running maximum with window = 1800 frames
5. dF/F = (Fc - baseline) / baseline
Then the dF/F is temporally binned by averaging in 10-frame bins.

ii.
```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)  # 1800 frames
    sig = int(sig_baseline * fs)  # 300 frames
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    Flow = np.maximum(Flow, 1e-6)
    dff = (Fc - Flow) / Flow
    return dff.astype(np.float32)

dff_binned = bin_data(dff, BIN_SIZE)  # 10-frame bins
```

iii. From CONVERSION_NOTES: "Parameters confirmed from ops.npy: fs=30, neucoeff=0.7, baseline='maximin', win_baseline=60.0, sig_baseline=10.0." The AI verified these parameters match Suite2p defaults in the ops.npy file.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. The AI uses all neurons present in `F.npy` without checking `iscell.npy`. However, since the data comes from Track2p which already outputs only tracked neurons, all entries in `iscell.npy` have iscell=1, making this a non-issue.

ii.
```python
# No iscell filtering in the code
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
dff = compute_dff(F, Fneu)
```

iii. The AI's trajectory reasoning notes that "All ROIs are tracked neurons across days, recorded from barrel cortex layer 2/3." The data README confirms: "the data only includes traces for the cells present across all days." No explicit mention of iscell filtering is made in CONVERSION_NOTES.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the recording onset. Each trial's neural data corresponds to a consecutive 2-minute block of the session, with time measured from the start of the recording. There is no event-based alignment; the temporal alignment event is simply `recording_onset`.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
# ...
'temporal_alignment_event': 'recording_onset',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. From CONVERSION_NOTES: The experiment involves spontaneous behavior with no explicit stimulus or trial events. The alignment is to recording onset, which is the natural choice for continuous recordings without explicit trial structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned from the original 30 Hz (33.33 ms per frame) to 10-frame bins, giving a temporal resolution of 333.33 ms per bin. Both neural and behavioral data are binned identically.

ii.
```python
BIN_SIZE = 10  # frames per bin
FS = 30  # imaging rate in Hz
# time_bin_size = BIN_SIZE / FS * 1000 = 333.33 ms

def bin_data(data, bin_size):
    n = data.shape[-1]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    if data.ndim == 1:
        return data[:n_use].reshape(n_bins, bin_size).mean(axis=1)
    else:
        return data[:, :n_use].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
```

iii. From CONVERSION_NOTES: "Following the paper's decoding methods: 'we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps.' Bin size: 10 frames = 333.33 ms."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not derived from any raw data variable. It is computed synthetically from the bin indices and the known imaging rate (30 Hz) and bin size (10 frames).

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. From CONVERSION_NOTES: "Time in seconds from recording onset. Computed as bin center times: `(bin_index * 10 + 5) / 30` seconds."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each 10-frame bin, converting from frame indices to seconds. For each bin index `i`, time = `(i * 10 + 5) / 30` seconds. This gives times starting at 0.167s (first bin center) and incrementing by 0.333s per bin.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
# For a 20-min session: times from 0.167s to ~1199.83s
# For a 30-min session: times from 0.167s to ~1799.83s
```

iii. The AI chose to represent time as seconds from recording onset, using bin centers rather than bin edges. This is a straightforward computation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time values are computed from the same bin indices as the neural data, so they are inherently aligned. Each time bin center corresponds exactly to the same temporal bin as the neural data. When split into trials, the time values preserve absolute session time (e.g., trial 2 starts at 120s, not 0s).

ii.
```python
# Time bins computed from same indexing as neural bins
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
# Split identically to neural data
time_trials.append(time_bins[start:end])
# Reshaped to (1, n_timepoints) for output
sess_input.append(t_trial.reshape(1, -1))
```

iii. The alignment is guaranteed by construction since time and neural data share the same binning scheme and trial splitting.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the pre-computed `motion_energy_glob.npy` file in each session's `move_deve/` directory. The raw motion energy was computed from video frames (pixel-wise differences of consecutive frames, squared and summed), but this computation was already done before the conversion.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. From CONVERSION_NOTES: "From the paper: 'we quantified these by looking at the pixel-wise difference of consecutive frames... squared all individual pixel-wise values and summed across pixels.' The motion energy data is provided pre-computed in `move_deve/motion_energy_glob.npy`."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves three steps:
1. Interpolation to match 2p frame count when camera frames are dropped
2. Temporal binning in 10-frame bins (averaging)
3. Discretization into 5 equal-percentile bins using global thresholds computed across all sessions

ii.
```python
# 1. Interpolation for dropped frames
me = interpolate_motion_energy(me, tstamps, n_frames)
# 2. Temporal binning
me_binned = bin_data(me, BIN_SIZE)
# 3. Global percentile discretization
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
me_disc = apply_discretization(me_trial, thresholds)
```

iii. From CONVERSION_NOTES: "Missing frames: Some camera recordings have dropped frames (up to 116 in one session). These are handled by: 1. Detecting gaps using interframe intervals from tstamps.npy, 2. Mapping camera frames to 2p frame indices, 3. Linear interpolation to fill missing frames."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins. Thresholds are computed globally across all binned motion energy values from all sessions. Values are assigned to bins 0-4 using `np.digitize` with the interior thresholds (20th, 40th, 60th, 80th percentiles).

ii.
```python
def discretize_motion_energy(all_me_values, n_bins=N_BINS_OUTPUT):
    percentiles = np.linspace(0, 100, n_bins + 1)  # [0, 20, 40, 60, 80, 100]
    thresholds = np.percentile(all_me_values, percentiles)
    return thresholds

def apply_discretization(me_values, thresholds):
    n_bins = len(thresholds) - 1
    binned = np.digitize(me_values, thresholds[1:-1])  # values 0 to n_bins-1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. From CONVERSION_NOTES: "Computed quintile thresholds (0%, 20%, 40%, 60%, 80%, 100%) across all binned motion energy values from all sessions globally. Assigned values 0-4 (bin_0 through bin_4). Global discretization ensures consistent bin definitions across sessions."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first interpolated to match the number of 2p imaging frames, then binned using the same 10-frame bins as the neural data, and finally split into the same 2-minute trial blocks. This ensures temporal alignment.

ii.
```python
# Interpolate ME to match neural frame count
me = interpolate_motion_energy(me, tstamps, n_frames)
# Bin with same bin size
me_binned = bin_data(me, BIN_SIZE)
# Split into same trials
neural_trials, me_trials, time_trials = split_into_trials(dff_binned, me_binned, time_bins)
```

iii. The AI addressed the frame count mismatch between video and 2p imaging through interpolation, then applied identical temporal binning and trial splitting to both streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data quality issue is dropped camera frames (motion energy has fewer samples than neural data). The AI handles this by detecting gaps using inter-frame intervals from `tstamps.npy`, mapping camera frames to 2p frame indices based on the ratio of actual to expected inter-frame intervals, and linearly interpolating to fill missing frames. No other missing data handling (e.g., NaN removal, outlier rejection) is implemented.

ii.
```python
def interpolate_motion_energy(me, tstamps, n_frames):
    if len(me) == n_frames:
        return me
    ifi = np.diff(tstamps)
    median_ifi = np.median(ifi)
    frame_indices = np.zeros(len(me), dtype=int)
    frame_indices[0] = 0
    cum_idx = 0
    for i in range(len(ifi)):
        n_skipped = int(np.round(ifi[i] / median_ifi))
        cum_idx += n_skipped
        frame_indices[i + 1] = cum_idx
    all_indices = np.arange(n_frames)
    me_interp = np.interp(all_indices, frame_indices, me)
    return me_interp
```

iii. From CONVERSION_NOTES: "Missing frames: Some camera recordings have dropped frames (up to 116 in one session)." The data README also notes: "In some recordings there might be some missing frames from the camera... they can be interpolated over."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Computing dF/F for each session, which involves Gaussian smoothing, running minimum, and running maximum over large arrays (n_neurons x n_frames, up to 746 x 54000).
2. Loading large .npy files (F.npy, Fneu.npy) from disk for each of 41 sessions.

ii.
```python
# dF/F computation involves 3 passes over the full neuron x time matrix
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. No explicit timing information is provided in the trajectory, but these operations dominate because they process the largest arrays in the pipeline.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interpolate_motion_energy` function contains a Python loop over all inter-frame intervals to compute cumulative frame indices. This could be vectorized using `np.cumsum` on the rounded interval ratios.

ii.
```python
# Current loop-based implementation
frame_indices = np.zeros(len(me), dtype=int)
frame_indices[0] = 0
cum_idx = 0
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx

# Could be vectorized as:
# n_skipped = np.round(ifi / median_ifi).astype(int)
# frame_indices = np.concatenate([[0], np.cumsum(n_skipped)])
```

iii. No justification for the loop-based approach is given. The loop iterates over ~36000-54000 elements per session, making it a candidate for vectorization, though the practical impact is modest compared to the dF/F computation.

## 6-c. What processing does the code repeat multiple times?

i. The code performs two passes over all sessions: first to load, process, and collect motion energy values (for computing global percentile thresholds), then a second pass to build the output data structure and apply discretization. However, the neural and behavioral data are stored in memory from the first pass, so no file I/O or heavy computation is duplicated.

ii.
```python
# First pass: load and process
for subj in subjects:
    for sess_name in sessions:
        dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
        all_sessions_data.append({...})

# Second pass: build output structure (lightweight, just reorganizing data)
for sess_data in all_sessions_data:
    neural.append(sess_data['neural_trials'])
    # ...apply discretization...
```

iii. This two-pass approach is necessary because global percentile thresholds cannot be computed until all motion energy values are collected. The design is efficient in that it caches intermediate results.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes dF/F from raw fluorescence (F.npy) and neuropil (Fneu.npy), even though Suite2p already provides deconvolved spike estimates in `spks.npy`. However, the paper explicitly specifies using dF/F, so this is the correct choice. The `time_bins` (session-level absolute time) for each trial are computed and stored in `time_trials` during processing but then only used as the decoder input; the full session-level context is partially discarded since the input preserves absolute time within the session anyway.

The code also computes the full dF/F for the entire session length, but frames at the end that don't fit into a complete 2-minute trial are discarded (truncated by `bin_data` and `split_into_trials`).

ii.
```python
# Remainder frames discarded in binning
n_use = n_bins * bin_size  # truncates remainder frames

# Remainder bins discarded in trial splitting
n_trials = n_bins // bins_per_trial  # truncates remainder bins
```

iii. The truncation of remainder data is minimal (at most 9 frames from binning and up to 359 bins from trial splitting, though in practice sessions appear to divide evenly into 2-minute blocks).
