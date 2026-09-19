# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by listing directories in the data folder (sorted alphabetically). For each subject, it lists session subdirectories (also sorted). For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. All data files are loaded via `np.load`.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

# For each subject:
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])

# Per session:
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI identified the standard suite2p directory structure from exploring the data and reference code. All directories are included as subjects/sessions, consistent with the dataset being pre-filtered by track2p.

## 1-b. How are the data split into subjects?

i. Subjects correspond to top-level directories in the data folder, sorted alphabetically. Unlike the reference which filters for `jm*` prefix, the AI includes all directories.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

iii. The AI notes in CONVERSION_NOTES.md that there are 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046). Since these are the only directories, the lack of a `jm*` filter doesn't change the result.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject's folder is treated as one session, sorted alphabetically. Each session directory contains one recording day's data.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The AI identified from exploring the data that session directories contain date-stamped folder names (e.g., `2023-10-18_a`), one per recording day.

## 1-d. How are the data split into trials?

i. The AI splits continuous recordings into **120-second (2-minute)** non-overlapping trials. This yields 360 bins per trial (120s * 30Hz / 10 frames). 20-minute sessions produce 10 trials, 30-minute sessions produce 15 trials. Remainder bins are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360

n_trials = n_bins // trial_frames
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
```

iii. The AI chose 2-minute trials based on the paper's statement "splits were done on consecutive 2 minute blocks." However, the task instructions explicitly say "Split sessions into 60-second trials."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials (those with enough bins to fill the trial duration) are included.

ii. N/A (no filtering code)

iii. The AI notes there is no explicit trial curation in the reference paper or code, which is consistent with the reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files, identified from both the reference code and data exploration.

## 2-b. How is the `neural` data processed?

i. The AI applies: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) maximin baseline estimation using scipy's `minimum_filter1d`, `maximum_filter1d`, and `gaussian_filter1d`, (3) baseline subtraction (`dff = Fc - F0`). This is a **reimplementation** of suite2p's `dcnv.preprocess` rather than calling it directly.

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0,
                prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu
    win_frames = int(win_baseline * fs)
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
    dff = Fc - F0
    return dff

def _maximin_baseline(Fc, win_frames, sig_frames):
    Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
    Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
    Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
    return Flow
```

iii. The AI aimed to match suite2p's default parameters. However, the reimplementation may differ from the actual suite2p `dcnv.preprocess` in subtle ways (e.g., suite2p's implementation may use different boundary handling, batching, or GPU acceleration). The reference solution directly calls `dcnv.preprocess`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the `F.npy` file are included.

ii. N/A (no filtering code)

iii. The AI notes that all `iscell` values are 1.0 in the provided data, indicating the data was pre-filtered by track2p. This is consistent with the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 2-minute block. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Start of 2-minute recording block',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,  # 120
```

iii. There is no stimulus event to align to. The recording is continuous with artificial trial segmentation. The AI's metadata uses off_start=0.0 and off_end=120 (vs reference which uses None).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before discretization.

ii.
```python
BIN_SIZE = 10  # number of frames per bin
def bin_data(data, bin_size):
    n_bins = n // bin_size
    return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)

dff_binned = bin_data(dff, bin_size)  # (n_neurons, n_bins)
me_binned = bin_data(me_interp, bin_size)  # (n_bins,)
```

iii. The paper states "averaging in bins of 10 consecutive timestamps." The AI correctly applies this to both streams before discretization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and bin duration.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. Since the frame rate is constant at 30 Hz, computing time from bin indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (BIN_SIZE / FS)`, giving seconds from the start of **each trial** (not the session). The range is [0, 119.7] seconds for each trial.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
# n_timepoints = 360 bins, BIN_SIZE/FS = 10/30 = 0.333s
# range: [0.0, 0.333, 0.667, ..., 119.667]
```

iii. The AI computes time relative to trial start rather than session start. The instructions say "Time elapsed from the beginning of the session in seconds," which suggests session-relative time.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input has the same number of time bins as the neural data (360 per trial), computed from the same bin indices. Both are indexed identically.

ii.
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. Alignment is trivially correct since both are derived from the same binned frame indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect dropped frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped video frames are detected using interframe intervals and interpolated via `np.interp`, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 equal-percentile bins computed **globally across all sessions**.

ii.
```python
# Interpolation
median_ifi = np.median(interframe_int)
ratios = interframe_int / median_ifi
missed_counts = np.round(ratios).astype(int) - 1
me_interp = np.interp(neural_positions, me_positions, motion_energy)

# Global discretization
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
bin_edges = np.percentile(all_values, percentiles)
binned = np.digitize(me, bin_edges[1:-1])
```

iii. The AI chose global discretization to maintain consistent bin meanings across sessions. However, the instructions specify "five equal-percentile bins, selected per session."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The binned motion energy is discretized into 5 categories using equal-percentile (quintile) bins. Bin edges are computed **globally** across all sessions using `np.percentile` with percentiles [0, 20, 40, 60, 80, 100], then `np.digitize` assigns each time point to a bin (0-4).

ii.
```python
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The AI's global discretization produces exactly 20% in each bin overall, but individual sessions can have very skewed distributions (e.g., session 0 has 0% in bin 0, sessions 35-41 have 0% in bins 0-2). The reference uses per-session discretization which ensures ~20% per bin within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy and neural data are acquired synchronously at 30 Hz. Dropped video frames are detected using interframe intervals (ratio to median IFI) and interpolated using `np.interp` to match the neural frame count. After interpolation, both are binned identically and sliced into the same trial segments.

ii.
```python
median_ifi = np.median(interframe_int)
ratios = interframe_int / median_ifi
missed_counts = np.round(ratios).astype(int) - 1
me_positions = np.zeros(n_me, dtype=int)
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
```

iii. The median IFI ratio approach is more robust than a fixed threshold. The reference uses `dt * 1000 > 0.04` which appears to be a threshold on interframe intervals (though the units are unusual).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (1) Dropped video frames are detected and interpolated to match neural frame count. (2) If motion energy has more frames than neural data, it is truncated. (3) Remainder bins at the end of a session that don't fill a complete trial are discarded. (4) Duplicate bin edges during discretization are made strictly increasing by adding 1e-10.

ii.
```python
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)

# Handle duplicate edges
for i in range(1, len(bin_edges)):
    if bin_edges[i] <= bin_edges[i-1]:
        bin_edges[i] = bin_edges[i-1] + 1e-10
```

iii. The AI handles several edge cases. The duplicate bin edge handling is a defensive measure for the global discretization approach.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation (maximin baseline using scipy filters). The AI's reimplementation runs on CPU only, taking about 1.5s per session (~60s total for 41 sessions).

ii. N/A

iii. The AI documented timing information in the conversion output: each session takes 0.5-2.3s depending on neuron count. Total conversion time is 60.5s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The missing frame interpolation loop (`for i in range(1, n_me): me_positions[i] = ...`) iterates frame by frame to build position mapping. This could be vectorized with `np.cumsum`. The trial-splitting loop could also be vectorized using array reshaping.

ii.
```python
me_positions = np.zeros(n_me, dtype=int)
me_positions[0] = 0
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
```

iii. The number of frames per session is ~36000-54000, so this loop has a non-trivial iteration count, though the actual runtime impact is small compared to the baseline computation.

## 6-c. What processing does the code repeat multiple times?

i. The AI's code does not appear to repeat any significant processing. Each session is processed once in a single pass.

ii. N/A

iii. The code is reasonably efficient with a single-pass design.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI's `_plot_processing` function (for `--show-processing` mode) loads and processes data for visualization that is not used in the final output. The duplicate bin edge adjustment (`bin_edges[i] = bin_edges[i-1] + 1e-10`) is unnecessary defensive code.

ii. N/A

iii. The plotting is only triggered with the `--show-processing` flag, so it doesn't affect normal conversion.
