# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by listing directories in the data directory (sorted), then lists session subdirectories within each subject. For each session, it loads `F.npy`, `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))

# Per session:
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI notes in CONVERSION_NOTES.md that the directory structure follows standard suite2p output organization. All subject directories and their session subdirectories are included.

## 1-b. How are the data split into subjects?

i. Subjects are identified as all directories within the data directory, sorted alphabetically. Unlike the reference which filters for `jm*` prefix, the AI uses all directories.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

iii. The AI identifies 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046) from the data directory.

## 1-c. How are the data split into sessions?

i. Sessions are all subdirectories within each subject folder, sorted alphabetically. Each subdirectory contains one recording session's data.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The AI documents 41 total sessions across 6 subjects (7,7,7,7,6,7 sessions per subject).

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive 2-minute (120-second) blocks of the binned recording. After binning by 10 frames, each trial is 360 bins long. Remainder bins that don't fill a complete trial are discarded. This differs from the reference which uses 60-second trials at native 30 Hz (1800 frames per trial).

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360

def split_into_trials(neural_binned, me_binned, trial_frames):
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // trial_frames
    for t in range(n_trials):
        start = t * trial_frames
        end = start + trial_frames
        neural_trials.append(neural_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. The AI cites the paper: "splits were done on consecutive 2 minute blocks" as justification for 120-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials are included.

ii. N/A

iii. The AI notes no explicit trial curation is described in the reference materials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from suite2p's `plane0` output.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p outputs for calcium imaging data.

## 2-b. How is the `neural` data processed?

i. The AI reimplements suite2p's preprocessing pipeline manually: (1) neuropil subtraction with coefficient 0.7, (2) maximin baseline estimation using scipy's `minimum_filter1d`, `maximum_filter1d`, and `gaussian_filter1d`, (3) baseline subtraction (Fc - F0). The result is then binned by averaging 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0,
                prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu
    win_frames = int(win_baseline * fs)  # window in frames
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
    dff = Fc - F0
    return dff

def _maximin_baseline(Fc, win_frames, sig_frames):
    Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
    Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
    Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
    return Flow

# Then binning:
dff_binned = bin_data(dff, bin_size)  # average 10 consecutive frames
```

iii. The AI states this matches Suite2p's `dcnv.preprocess()` using the maximin baseline method. It notes fixing a bug where `sig_baseline` was initially multiplied by `fs` (giving 300 instead of 10).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI notes that all `iscell` values are 1.0 in the provided data, indicating neurons are pre-filtered by the track2p pipeline.

ii. N/A

iii. The AI documents that "all iscell values are 1.0 in the provided data (pre-filtered by track2p)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 2-minute recording block. No event-based alignment is used since there is no stimulus-driven trial structure.

ii.
```python
'temporal_alignment_event': 'Start of 2-minute recording block',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,  # 120
```

iii. The continuous recording is split into consecutive blocks with no specific alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 10 consecutive frames are averaged into one bin, reducing the frame rate from 30 Hz to 3 Hz. The time bin size is 333.33 ms. This differs from the reference which keeps native 30 Hz resolution (33.33 ms bins).

ii.
```python
BIN_SIZE = 10  # number of frames per bin (paper: "bins of 10 consecutive timestamps")

def bin_data(data, bin_size):
    n = data.shape[-1]
    n_bins = n // bin_size
    trimmed = data[..., :n_bins * bin_size]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)

'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33
```

iii. The AI cites the paper: "averaging in bins of 10 consecutive timestamps" as justification for this binning.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from a raw data variable. It is computed from the bin index and bin duration for each trial.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. Time is computed synthetically from bin indices, representing elapsed time from the start of each trial.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (BIN_SIZE / FS)`, giving time in seconds from the start of each trial (not from the start of the experiment). The range is [0, 119.67] seconds per trial. This differs from the reference which computes time from the start of the experiment (absolute frame index / FS).

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
# Range: [0, 359] * (10/30) = [0.0, 119.67] seconds
```

iii. The AI describes this as "Time elapsed from start of trial in seconds."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is constructed to have the same number of timepoints as the binned neural data for each trial, so alignment is inherent.

ii.
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. Each time input array matches the shape of the corresponding neural trial.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` and `interframe_int.npy` in the `move_deve` subdirectory.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) detect and interpolate missing video frames using interframe intervals, (2) bin by averaging 10 consecutive frames, (3) discretize into 5 equal-percentile (quintile) bins computed globally across all sessions. Unlike the reference, no normalization by standard deviation is applied before discretization.

ii.
```python
# Missing frame interpolation:
me_interp = interpolate_missing_frames(me, interframe, n_frames)

# Binning:
me_binned = bin_data(me_interp, bin_size)

# Discretization:
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
binned = np.digitize(me, bin_edges[1:-1])
```

iii. The AI follows global quintile discretization for balanced class counts and handles missing frames via interpolation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using global percentile edges. `np.digitize` with inner edges produces values in range [0, 4]. Duplicate bin edges are handled by adding small epsilon values.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
for i in range(1, len(bin_edges)):
    if bin_edges[i] <= bin_edges[i-1]:
        bin_edges[i] = bin_edges[i-1] + 1e-10
binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. Quintile bins ensure roughly 20% of data in each bin.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Missing video frames are detected using interframe intervals (median IFI ratio method) and interpolated using `np.interp`. After interpolation, both neural and ME data are binned by the same factor (10 frames), ensuring frame-for-frame alignment. They are then split into trials using the same indices.

ii.
```python
def interpolate_missing_frames(motion_energy, interframe_int, n_neural_frames):
    median_ifi = np.median(interframe_int)
    ratios = interframe_int / median_ifi
    missed_counts = np.round(ratios).astype(int) - 1
    me_positions = np.zeros(n_me, dtype=int)
    for i in range(1, n_me):
        me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
    neural_positions = np.arange(n_neural_frames)
    me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
    return me_interp
```

iii. The interpolation ensures ME length matches neural frame count before both are binned and split identically.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are detected via interframe interval analysis and interpolated. If ME has more frames than neural data, it is truncated. End-of-recording bins that don't fill a complete trial are discarded.

ii.
```python
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)

# For duplicate bin edges:
if bin_edges[i] <= bin_edges[i-1]:
    bin_edges[i] = bin_edges[i-1] + 1e-10
```

iii. The AI handles edge cases for both shorter and longer ME arrays relative to neural data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation using the reimplemented maximin baseline (scipy filters over full session data for all neurons). The AI notes that fixing `sig_baseline` from 300 to 10 reduced runtime from 1420s to 60s.

ii. N/A

iii. CONVERSION_NOTES.md: "Fix also dramatically improved runtime: 1420s -> 60s"

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The missing frame interpolation uses a Python loop to build `me_positions`:
```python
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
```
This is a cumulative sum that could be vectorized with `np.cumsum`.

ii. See above.

iii. The number of ME frames per session is relatively small (~36000-54000), so the performance impact is modest.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing is observed. Each session's data is loaded and processed once. Discretization is done as a single global pass after all sessions are processed.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code applies temporal binning (averaging 10 consecutive frames), which reduces the temporal resolution of the data from 30 Hz to 3 Hz. While this follows the paper's methodology for their specific ridge regression decoder, it discards temporal information that might be useful for the downstream neural network decoder. The reference solution does not apply this binning.

ii.
```python
BIN_SIZE = 10
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
```

iii. The AI justifies this by citing the paper's methods: "averaging in bins of 10 consecutive timestamps."
