# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by scanning the data directory for subject folders (directories whose names match known patterns), then scanning each subject folder for session subdirectories. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` from `move_deve/` for behavioral data. All subjects and sessions are discovered automatically through directory listing.

ii.
```python
def get_subjects_and_sessions():
    """Get all subjects and their session directories."""
    subjects = sorted([d for d in os.listdir(DATA_ROOT)
                      if os.path.isdir(os.path.join(DATA_ROOT, d))])
    result = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_ROOT, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        for sess in sessions:
            result.append((subj, sess))
    return result

# In process_session():
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The AI notes in CONVERSION_NOTES.md that each subject has a directory containing session subdirectories, each with suite2p output and motion energy files. The directory structure follows the Track2p pipeline convention.

## 1-b. How are the data split into subjects?

i. Subjects correspond to top-level directories in the data root, sorted alphabetically. Unlike the reference, the AI does not filter by `jm` prefix -- it includes all directories.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d))])
```

iii. The AI identifies each directory in the data root as a distinct subject (mouse). The CONVERSION_NOTES.md confirms 6 subjects: jm031, jm032, jm038, jm039, jm040, jm046.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject's folder represents one session (recording day). Sessions are sorted alphabetically (which corresponds to chronological order given the date-based naming).

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as 60-second non-overlapping segments of the continuous recording. After binning (10 frames at 30 Hz = 3 Hz), each trial has 180 timepoints. Remainder frames at the end that don't fill a complete trial are discarded.

ii.
```python
binned_rate = FRAME_RATE / BIN_SIZE  # 3 Hz
timepoints_per_trial = int(TRIAL_DURATION * binned_rate)  # 180
n_binned_frames = dff_binned.shape[1]
n_trials = n_binned_frames // timepoints_per_trial

for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. Per the decoder task instructions, sessions are split into 60-second trials. Since the recording is continuous with no stimulus-driven trial structure, fixed-length segmentation is used.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second segments are included.

ii. N/A (no filtering code exists)

iii. The AI notes in CONVERSION_NOTES.md that there are no trial curation rules described in the paper -- sessions are continuous recordings of spontaneous behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI performs three steps: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) maximin baseline estimation (Gaussian smoothing with sigma=10 frames, then running minimum, then running maximum, using a 60s window), and (3) **dF/F normalization by dividing by the baseline** (`dff = (Fc - F0) / F0`). This differs from the reference which uses suite2p's `dcnv.preprocess` directly, which only performs baseline subtraction (`Fc - F0`) without dividing by F0.

ii.
```python
def compute_baseline_maximin(Fc, win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE, fs=FRAME_RATE):
    win = int(win_baseline * fs)
    if win % 2 == 0:
        win += 1
    n_neurons, n_frames = Fc.shape
    Flow = np.zeros_like(Fc, dtype=np.float32)
    for i in range(n_neurons):
        trace = Fc[i].astype(np.float32)
        smoothed = gaussian_filter1d(trace, sig_baseline)
        smoothed = minimum_filter1d(smoothed, win)
        smoothed = maximum_filter1d(smoothed, win)
        Flow[i] = smoothed
    return Flow

def compute_dff(F, Fneu):
    Fc = F.astype(np.float32) - NEUCOEFF * Fneu.astype(np.float32)
    F0 = compute_baseline_maximin(Fc)
    F0_safe = np.maximum(F0, 1e-6)
    dff = (Fc - F0) / F0_safe
    return dff
```

iii. The AI states in CONVERSION_NOTES.md: "dF/F computation using Suite2p maximin baseline (verified against suite2p source code)". It describes using Suite2p default parameters. However, the AI reimplements the baseline computation manually rather than calling suite2p's `dcnv.preprocess`, and adds a dF/F normalization step (division by F0) that the suite2p function does not perform.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the F.npy file are included.

ii. N/A (no filtering code)

iii. The AI notes that the data already contains only tracked neurons that passed the iscell > 0.5 threshold during Track2p processing, so no additional filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments, no event-based alignment is needed. The alignment event is described as "Session start (beginning of recording)".

ii.
```python
'temporal_alignment_event': 'Session start (beginning of recording)',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION),
```

iii. There is no stimulus event to align to. The recording is continuous.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy data are averaged into non-overlapping bins of 10 consecutive frames, reducing the 30 Hz signal to 3 Hz (333.33 ms time bins).

ii.
```python
BIN_SIZE = 10    # frames per bin
def bin_data(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    trimmed = data[..., :n_bins * bin_size]
    if data.ndim == 1:
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
    else:
        return trimmed.reshape(*data.shape[:-1], n_bins, bin_size).mean(axis=-1)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and the binned sampling rate (3 Hz), giving seconds from the start of the session.

ii.
```python
time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
input_trials.append(time_seconds.reshape(1, -1))
```

iii. Since the frame rate is constant at 30 Hz and there are no stored timestamps, computing time from bin indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as bin_index / binned_rate, where binned_rate = 30/10 = 3 Hz. This gives continuous time in seconds from the start of the session, running across trial boundaries.

ii.
```python
binned_rate = FRAME_RATE / BIN_SIZE  # 3 Hz
time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
```

iii. No special processing -- straightforward arithmetic from indices.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input uses the same bin indices as the neural data, so they are inherently aligned. Both share the same timepoints per trial (180).

ii.
```python
for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
    input_trials.append(time_seconds.reshape(1, -1))
```

iii. Using the same index range ensures alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) interpolation to match neural frame count if frames are missing (using global linear interpolation via `np.interp`), (2) averaging into 10-frame bins, (3) discretization into 5 equal-percentile bins per session.

ii.
```python
def interpolate_motion_energy(me, n_target_frames):
    if len(me) == n_target_frames:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target_frames)
    me_interp = np.interp(x_target, x_orig, me)
    return me_interp

me = interpolate_motion_energy(me, n_frames)
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
# Then discretization per session
```

iii. The AI notes that interpolation is needed to handle dropped camera frames where the motion energy array is shorter than the neural data.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins per session. Bin edges are computed from the concatenated binned motion energy of all trials within a session. The outer edges are set to -inf and +inf to capture all values.

ii.
```python
def discretize_motion_energy(me_binned_trials, n_bins=N_BINS):
    all_me = np.concatenate(me_binned_trials)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    discretized = []
    for me_trial in me_binned_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to n_bins-1
        discretized.append(binned)
    return discretized, bin_edges
```

iii. The instructions specify "five equal-percentile bins, selected per session." The AI computes percentile boundaries from all trial data within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. When the motion energy array is shorter than the neural data (due to dropped camera frames), the AI uses global linear interpolation (`np.interp`) to stretch the entire signal to match the neural frame count. This differs from the reference which detects specific dropped frames via interframe intervals and inserts interpolated values at those specific positions.

ii.
```python
def interpolate_motion_energy(me, n_target_frames):
    if len(me) == n_target_frames:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target_frames)
    me_interp = np.interp(x_target, x_orig, me)
    return me_interp

me = interpolate_motion_energy(me, n_frames)
```

iii. The AI notes that interpolation handles the frame count mismatch. The global interpolation approach is simpler but less precise than detecting specific dropped frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where motion energy is shorter than neural data) are handled by global linear interpolation of the motion energy signal. Remainder frames at the end of a session that don't fill a complete 60-second trial are discarded.

ii.
```python
me = interpolate_motion_energy(me, n_frames)
# ...
n_trials = n_binned_frames // timepoints_per_trial
# remainder is implicitly discarded
```

iii. The AI addresses missing frames through interpolation and handles incomplete trials by discarding remainders.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation (`compute_dff`), specifically the baseline estimation which involves Gaussian smoothing, running minimum, and running maximum operations over the full session length for every neuron, implemented as a Python loop over neurons. The conversion logs show dF/F takes 0.3-1.5s per session.

ii.
```python
def compute_baseline_maximin(Fc, ...):
    for i in range(n_neurons):
        trace = Fc[i].astype(np.float32)
        smoothed = gaussian_filter1d(trace, sig_baseline)
        smoothed = minimum_filter1d(smoothed, win)
        smoothed = maximum_filter1d(smoothed, win)
        Flow[i] = smoothed
    return Flow
```

iii. The AI logs timing information showing dF/F computation dominates processing time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The baseline estimation loops over neurons one at a time (`for i in range(n_neurons)`). The `gaussian_filter1d`, `minimum_filter1d`, and `maximum_filter1d` scipy functions can operate on 2D arrays along a specified axis, so this loop could be eliminated by passing the entire `(n_neurons, n_frames)` array at once.

ii.
```python
for i in range(n_neurons):
    trace = Fc[i].astype(np.float32)
    smoothed = gaussian_filter1d(trace, sig_baseline)
    smoothed = minimum_filter1d(smoothed, win)
    smoothed = maximum_filter1d(smoothed, win)
    Flow[i] = smoothed
```

iii. The AI did not explicitly address this in CONVERSION_NOTES.md, though total processing time was ~47s for all 41 sessions, which was considered acceptable.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once in a single pass.

ii. N/A

iii. The AI processes each session independently in a single loop.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins the full session of motion energy data, but any remainder bins that don't fill a complete trial are discarded when splitting into trials. This is a minor inefficiency.

ii.
```python
me_binned = bin_data(me, BIN_SIZE)  # bins entire session
# ...
n_trials = n_binned_frames // timepoints_per_trial  # discards remainder
```

iii. The amount of discarded data is at most 59 seconds per session, which is negligible.
