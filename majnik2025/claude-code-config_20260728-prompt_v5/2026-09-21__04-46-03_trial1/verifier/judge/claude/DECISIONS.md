# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory. Sessions are subdirectories within each subject folder (filtered to start with `'2'`). For each session, neural data is loaded from suite2p output files (`F.npy`, `Fneu.npy`, `ops.npy`) and motion energy from `motion_energy_glob.npy` and `tstamps.npy`. All files are loaded via `np.load`.

ii.
```python
SUBJECTS = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

def get_sessions(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d)) and d[0] == '2'])
    return sessions

F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
```

iii. The AI documented in CONVERSION_NOTES.md that the directory structure follows standard conventions: subject folders contain session subfolders, each with suite2p output and behavioral data. The session filter `d[0] == '2'` selects directories starting with '2' (matching the date-based naming convention like `2023-10-18_a`).

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data directory, sorted alphabetically. This produces 6 subjects.

ii.
```python
SUBJECTS = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset. The AI noted 6 subjects: jm031, jm032, jm038, jm039, jm040, jm046.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject's folder, filtered to start with `'2'` and sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
def get_sessions(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d)) and d[0] == '2'])
    return sessions
```

iii. The AI noted 6-7 sessions per subject (41 total). Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. There is no natural trial structure in this dataset. Trials are defined as 60-second non-overlapping segments of the continuous recording. After binning (10 frames at 30 Hz), each trial is 180 bins. Incomplete trailing trials are discarded.

ii.
```python
BINS_PER_TRIAL = int(TRIAL_DURATION_S / BIN_DURATION_S)  # 180 bins

def split_into_trials(data, bins_per_trial):
    if data.ndim == 2:
        n_neurons, n_bins = data.shape
        n_trials = n_bins // bins_per_trial
        trials = []
        for t in range(n_trials):
            start = t * bins_per_trial
            end = start + bins_per_trial
            trials.append(data[:, start:end])
        return trials
```

iii. Per the decoder task specification, trials are defined as 60-second non-overlapping segments. Since the recording is continuous spontaneous behavior with no stimulus-driven trial structure, fixed-length segmentation is used.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second trials are included.

ii. N/A (no filtering code)

iii. The AI noted that there is no specific trial curation mentioned since this is spontaneous behavior with no task trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (suite2p parameters), from `plane0`.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
```

iii. These are the standard suite2p output files. The AI loads ops.npy to extract baseline correction parameters.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`Fc = F - 0.7 * Fneu`), then computes dF/F using a reimplemented version of suite2p's maximin baseline method: Gaussian smoothing -> running min -> running max to estimate baseline, then `dF/F = (Fc - baseline) / max(baseline, 10.0)`. This divides by the baseline, which differs from the reference's use of `dcnv.preprocess` which only subtracts the baseline.

ii.
```python
def compute_dff(F, Fneu, ops):
    Fc = F - NEUCOEFF * Fneu
    sig_baseline = ops.get('sig_baseline', 10.0)
    win_baseline = ops.get('win_baseline', 60.0)
    fs = ops.get('fs', FRAME_RATE)
    win = int(win_baseline * fs)
    smoothed = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    flow_min = minimum_filter1d(smoothed, size=win, axis=1)
    baseline = maximum_filter1d(flow_min, size=win, axis=1)
    baseline_safe = np.maximum(baseline, 10.0)
    dff = (Fc - baseline) / baseline_safe
    return dff
```

iii. The AI documented that the paper says "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and attempted to replicate the Suite2p baseline correction. However, the AI implemented actual dF/F (dividing by baseline) rather than just baseline subtraction as suite2p's `dcnv.preprocess` does.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the suite2p output are included.

ii. N/A (no filtering code)

iii. The AI noted that the data already contains only tracked neurons (Track2p output) and all iscell values are 1. No further filtering was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Session start (beginning of recording)',
'off_start': 0.0,
'off_end': TRIAL_DURATION_S,
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms time bin). Binning is applied to both streams before motion energy is discretized.

ii.
```python
BIN_SIZE = 10
BIN_DURATION_S = BIN_SIZE / FRAME_RATE  # ~0.333 seconds

def bin_data(data, bin_size):
    if data.ndim == 2:
        n_neurons, n_frames = data.shape
        n_bins = n_frames // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
    else:
        n_frames = len(data)
        n_bins = n_frames // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
```

iii. The AI correctly cited the paper: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index and the bin duration, giving seconds from the start of the session.

ii.
```python
time_input = np.arange(n_bins) * BIN_DURATION_S
```

iii. Since the frame rate is constant at 30 Hz and the bin size is fixed, computing time from bin indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * bin_duration_in_seconds`. No further processing is applied.

ii.
```python
time_input = np.arange(n_bins) * BIN_DURATION_S
```

iii. Simple arithmetic derivation from bin indices.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data since both are indexed by the same bin indices after binning.

ii.
```python
time_input = np.arange(n_bins) * BIN_DURATION_S
# Then split into trials along with neural and output data
input_trials = split_into_trials(time_input, BINS_PER_TRIAL)
```

iii. Since time is computed from the same bin indices as the neural data, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used for frame alignment/interpolation.

ii.
```python
me = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) motion energy is aligned/interpolated to match neural frame count using `np.interp` with camera timestamps, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 equal-percentile bins computed per session.

ii.
```python
# Alignment
me_aligned = align_motion_energy(me, ts, n_frames)

# Binning
me_binned = bin_data(me_aligned, BIN_SIZE)

# Discretization
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, bin_edges[1:-1], right=False)
    labels = np.clip(labels, 0, n_bins - 1)
    return labels, bin_edges
```

iii. The AI documented these steps in CONVERSION_NOTES.md, citing the paper's description of binning and the need to handle missing camera frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session using `np.digitize`. Bin edges are computed as 0th, 20th, 40th, 60th, 80th, and 100th percentiles of the binned motion energy within each session.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, bin_edges[1:-1], right=False)
    labels = np.clip(labels, 0, n_bins - 1)
    return labels, bin_edges
```

iii. The decoder task specifies "Motion energy, discretized into five equal-percentile bins, selected per session." The AI implements this correctly.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses `np.interp` with camera timestamps (`tstamps.npy`) to interpolate the motion energy signal to match neural frame times. Camera timestamps are in kiloseconds and are converted by multiplying by 1000. Neural frame times are computed as `np.arange(n_neural_frames) / FRAME_RATE`.

ii.
```python
def align_motion_energy(me, me_timestamps, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.copy()
    cam_times_s = me_timestamps * 1000.0
    neural_times_s = np.arange(n_neural_frames) / FRAME_RATE
    me_aligned = np.interp(neural_times_s, cam_times_s, me)
    return me_aligned
```

iii. The AI noted that the camera is triggered by the microscope, so frames should be 1:1. When the motion energy has fewer frames (dropped camera frames), interpolation fills in the gaps.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (motion energy shorter than neural data) are handled by interpolating the motion energy signal using timestamps. Incomplete trailing bins from the 10-frame binning are truncated. Incomplete trailing trials are discarded.

ii.
```python
me_aligned = align_motion_energy(me, ts, n_frames)
# In bin_data:
trimmed = data[:n_bins * bin_size]
# In split_into_trials:
n_trials = n_bins // bins_per_trial  # drops incomplete last trial
```

iii. The AI documented the missing frame counts per session in CONVERSION_NOTES.md.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation, specifically the scipy filter operations (gaussian_filter1d, minimum_filter1d, maximum_filter1d) applied over the full session length for every neuron. Processing times range from ~0.4s to ~2.3s per session.

ii. N/A

iii. The AI reported per-session processing times in the conversion output.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `split_into_trials` function uses a loop over trials that could be replaced with array reshaping. However, since it produces a list of arrays, the loop is functionally necessary for the output format.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    trials.append(data[:, start:end])
```

iii. The loop is simple and has minimal performance impact since the number of trials per session is small (20-30).

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once.

ii. N/A

iii. The code processes each session sequentially without redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `ops.npy` for suite2p parameters but only uses default values from it (the `ops.get()` calls use fallback defaults that match the constants). The plotting functionality (`_plot_processing`) is optional and only triggered by `--show-processing`.

ii.
```python
ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
sig_baseline = ops.get('sig_baseline', 10.0)
win_baseline = ops.get('win_baseline', 60.0)
fs = ops.get('fs', FRAME_RATE)
```

iii. Loading ops.npy is a minor overhead. The processing is otherwise efficient.
