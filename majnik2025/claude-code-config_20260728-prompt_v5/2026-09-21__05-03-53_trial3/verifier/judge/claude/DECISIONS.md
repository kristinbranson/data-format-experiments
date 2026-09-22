# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by scanning the data directory for subject folders, then scanning each subject folder for session subdirectories. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/` for behavioral data. All subjects (directories in the data folder) and all sessions (subdirectories within each subject) are included.

ii.
```python
def get_subjects_and_sessions(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
    subject_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        subject_sessions[subj] = sessions
    return subjects, subject_sessions

# Per session:
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The AI documents in CONVERSION_NOTES.md that the directory structure follows a standard convention: subject folders contain session subfolders, each with suite2p output and motion energy files. All directories are included as subjects, and all subdirectories within each subject as sessions.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all directories in the data directory, sorted alphabetically. The AI does not filter by a naming prefix (unlike the reference which filters for `jm*` prefix), but since all directories in the data folder are subject directories, this produces the same result.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))])
```

iii. The AI relies on the data directory containing only subject folders, which is true for this dataset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The AI notes that each subdirectory contains suite2p output and motion energy files for one recording session, and sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. Trials are defined as 60-second non-overlapping segments of the continuous recording. After binning by 10 frames (30 Hz -> 3 Hz), each trial has 180 time bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 60
TRIAL_BINS = int(TRIAL_DURATION_S / (BIN_SIZE / FRAME_RATE))  # 180 bins per trial

def split_into_trials(data, trial_length, axis=-1):
    n = data.shape[axis]
    n_trials = n // trial_length
    trials = []
    for i in range(n_trials):
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(i * trial_length, (i + 1) * trial_length)
        trials.append(data[tuple(slices)])
    return trials
```

iii. The AI justifies this by noting that the decoder task instructions specify 60-second trials. Since the recording is continuous spontaneous activity with no stimulus-driven trial structure, fixed-length segmentation is the appropriate approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second segments are included; only the incomplete remainder at the end of a session is discarded.

ii. N/A (no filtering code)

iii. The AI notes that there are no explicit trial curation rules mentioned for spontaneous behavior, and no quality criteria are applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from the `suite2p/plane0/` directory of each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces. The paper states "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)".

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - 0.7 * Fneu`), followed by a manually reimplemented version of suite2p's `maximin` baseline correction. The reimplementation uses PyTorch to: (1) apply Gaussian smoothing with sigma=10, (2) apply min-filter then max-filter with window = 60s * 30Hz = 1800 frames, (3) subtract the resulting baseline from Fc.

ii.
```python
def compute_dfof(F, Fneu, fs=FRAME_RATE):
    Fc = F - NEUCOEFF * Fneu
    Fc = Fc.astype(np.float32)

    win = int(WIN_BASELINE * fs)  # 1800 frames
    if win % 2 == 0:
        win += 1

    # Gaussian smoothing
    gwid = int(np.round(SIG_BASELINE * 3))
    gaussian = torch.exp(-torch.arange(-gwid, gwid + 1, ...)**2 / (2 * SIG_BASELINE**2))
    gaussian /= gaussian.sum()

    # Min filter then max filter (maximin)
    data = pad(data, (gwid, gwid), 'replicate')
    data = conv1d(data.unsqueeze(1), gaussian.unsqueeze(0).unsqueeze(0), padding=0)
    data = pad(data, (win // 2, win // 2), 'replicate')
    data = -max_pool1d(-data, kernel_size=win, stride=1, padding=0)
    data = pad(data, (win // 2, win // 2), 'replicate')
    data = max_pool1d(data, kernel_size=win, stride=1, padding=0)

    Flow[nstart:nend] = data.squeeze(1).cpu().numpy()
    dff = Fc - Flow
    return dff
```

iii. The AI notes that the paper says "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)". The AI reimplemented the suite2p baseline correction manually rather than calling `suite2p.extraction.dcnv.preprocess` directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron quality filtering is applied. All neurons present in the `F.npy` output are included. The AI notes that the provided data is already filtered by Track2p (only tracked neurons) and by iscell > 0.5.

ii. N/A (no filtering code)

iii. The AI verifies in CONVERSION_NOTES.md Step 4 that all cells in the provided data have iscell > 0.5, meaning the data is pre-filtered and no additional filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The metadata records `temporal_alignment_event` as `'Session start (spontaneous activity, no task events)'` with `off_start` and `off_end` as `None`.

ii.
```python
'temporal_alignment_event': 'Session start (spontaneous activity, no task events)',
'off_start': None,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural data and motion energy are averaged into non-overlapping bins of 10 consecutive frames, converting from 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10    # frames per bin
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # 333.33 ms

def bin_data(data, bin_size=BIN_SIZE, axis=-1):
    n = data.shape[axis]
    n_bins = n // bin_size
    slices[axis] = slice(0, n_bins * bin_size)
    data_trunc = data[tuple(slices)]
    new_shape = data_trunc.shape[:-1] + (n_bins, bin_size)
    return data_trunc.reshape(new_shape).mean(axis=-1)

dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_interp.reshape(1, -1), BIN_SIZE, axis=1).flatten()
```

iii. The paper's Methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the bin index and bin duration. It is not derived from any raw data variable; it is synthetically constructed from the known frame rate and bin size.

ii.
```python
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)  # seconds
```

iii. The frame rate is constant at 30 Hz, so computing time from bin indices is equivalent to using actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (10 / 30)` seconds, giving the time of the left edge of each bin from session start. The time array is then split into per-trial segments matching the trial boundaries. Within each trial, the time values reflect absolute session time, not trial-relative time.

ii.
```python
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)
time_trials = split_into_trials(time_s.reshape(1, -1), TRIAL_BINS, axis=1)
```

iii. The decoder task specifies "Time elapsed from the beginning of the session in seconds", so absolute session time is the correct representation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time array is constructed from the same bin indices as the neural data, so alignment is inherent. Both the neural data and the time input share the same number of bins per trial (180), and both are split using the same trial boundaries.

ii.
```python
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)
# Split into trials using same trial_length as neural and output
time_trials = split_into_trials(time_s.reshape(1, -1), TRIAL_BINS, axis=1)
neural_trials = split_into_trials(dff_binned, TRIAL_BINS, axis=1)
```

iii. Since time is computed from the bin index, it is inherently aligned with the binned neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Additionally, `tstamps.npy` from the same directory is loaded for use in interpolation to handle missing camera frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) interpolation of missing camera frames using `np.interp` with linearly-spaced camera indices mapped to neural frame indices, (2) binning into 10-frame averages (matching neural data), (3) discretization into 5 equal-percentile bins computed per session.

ii.
```python
# Interpolation
camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
me_interp = np.interp(neural_indices, camera_indices, me_float)

# Binning
me_binned = bin_data(me_interp.reshape(1, -1), BIN_SIZE, axis=1).flatten()

# Discretization
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(me_binned, percentiles)
binned = np.digitize(me_binned, edges[1:-1], right=False)
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The AI notes in CONVERSION_NOTES.md that binning by 10 frames is per the paper's Methods, and that discretization into 5 quintile bins is per the decoder task instructions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) computed within each session. The bin edges are determined by the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of the binned motion energy for that session. `np.digitize` maps values into bins 0-4.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_ME_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, edges[1:-1], right=False)
    binned = np.clip(binned, 0, n_bins - 1)
    return binned, edges
```

iii. The decoder task specifies "Motion energy, discretized into five equal-percentile bins, selected per session." The AI correctly implements per-session percentile-based discretization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy with neural data by interpolating the motion energy array to match the neural frame count using `np.interp`. The camera indices are assumed to be evenly spread across the neural frame range. After interpolation, both streams are binned by the same factor (10 frames) and split into trials using the same boundaries.

ii.
```python
def interpolate_motion_energy(me, tstamps, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    neural_indices = np.arange(n_neural_frames)
    camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
    me_interp = np.interp(neural_indices, camera_indices, me_float)
    return me_interp
```

iii. The AI notes that the camera is triggered by the microscope at 30 Hz, but some frames are missed. The interpolation approach stretches the available ME frames evenly across the neural frame range.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (motion energy shorter than neural frames) are handled by linear interpolation using `np.interp` to stretch the available ME values to match the neural frame count. Remainder frames at the end of sessions that don't fill a complete 60-second trial are discarded.

ii.
```python
# ME interpolation
camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
me_interp = np.interp(neural_indices, camera_indices, me_float)

# Trial remainder
n_trials = n_timebins // TRIAL_BINS  # incomplete trials silently dropped
```

iii. The AI documents these handling strategies in CONVERSION_NOTES.md under Steps 4 and 10.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F baseline correction (maximin filter), which involves sliding window operations over the full session length for every neuron. This is run on CPU. Loading .npy files is also I/O-bound but relatively fast. The conversion log shows dF/F takes ~0.2-1.1s per session, with the full conversion completing in 32.4s for 41 sessions.

ii.
```python
# Per session timing output:
print(f"  {subj}/{sess_name}: ... load={t_load:.1f}s dfof={t_dfof:.1f}s bin={t_bin:.2f}s")
```

iii. The AI includes timing instrumentation in the code and reports timing per session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `split_into_trials` function uses a Python loop over trials to slice the data. This could be replaced with a single reshape operation (as the reference does). The batch processing loop for dF/F baseline computation processes 100 neurons at a time, which is already reasonably efficient.

ii.
```python
def split_into_trials(data, trial_length, axis=-1):
    n = data.shape[axis]
    n_trials = n // trial_length
    trials = []
    for i in range(n_trials):
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(i * trial_length, (i + 1) * trial_length)
        trials.append(data[tuple(slices)])
    return trials
```

iii. The trial-splitting loop produces a list of arrays which is the required output format, so while it could be done with reshape, the loop is straightforward and not a bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once in a single pass. The discretization is done per-session after all sessions are processed.

ii. N/A

iii. The AI processes each session independently in a single loop.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `tstamps.npy` for each session to pass to the interpolation function, but the interpolation function does not actually use the timestamps — it only uses the length of the ME array and linearly spaces camera indices. So `tstamps.npy` is loaded but effectively unused.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
# Passed to interpolate_motion_energy but not actually used for index computation
me_interp = interpolate_motion_energy(me, tstamps, n_frames)
```

iii. The `tstamps` parameter is accepted by `interpolate_motion_energy` but the interpolation is done using evenly-spaced indices rather than actual timestamps.
