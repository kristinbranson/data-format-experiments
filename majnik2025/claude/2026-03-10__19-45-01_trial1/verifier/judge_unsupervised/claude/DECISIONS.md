# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by listing subdirectories of the `data/` directory, then for each subject lists session subdirectories. For each session, it loads four numpy files from the suite2p and move_deve subdirectories: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `motion_energy_glob.npy` (motion energy), and `interframe_int.npy` (inter-frame intervals). Sessions are processed sequentially in a single loop. Data is not loaded all at once; each session is loaded, processed, and stored before moving to the next.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
# ...
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
# ...
def load_session(subject_dir, session_name):
    session_dir = os.path.join(subject_dir, session_name)
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
    return F, Fneu, me, interframe
```

iii. The AI noted from the data README and reference code (load_data.ipynb) that data is organized as subject folders containing session subfolders, each with suite2p outputs and behavioral data. The AI followed the same directory traversal pattern as the reference notebook.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the top-level subdirectories within the `data/` directory (e.g., jm031, jm032, ..., jm046). A unique ordered list of subjects is maintained, and each session is assigned a subject index that maps into this list.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
# ...
subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))  # unique, ordered
# ...
subject_idx.append(subject_list.index(subj))
```

iii. The AI identified 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046) matching the paper's "full dataset of 6 mice." The data README confirms each top-level folder corresponds to a subject ID.

## 1-c. How are the data split into sessions?

i. Sessions correspond to the subdirectories within each subject folder (e.g., `jm031/2023-10-18_a`). Each subdirectory represents one recording day. All sessions for all subjects are collected into a flat list and processed sequentially. Each session yields one entry in the output data structure.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. The AI found 41 total sessions (7 per subject except jm040 with 6), consistent with the paper's statement of recordings on consecutive days (P7-P14).

## 1-d. How are the data split into trials?

i. Each session's continuous recording is split into fixed-length 2-minute blocks. At 30 Hz imaging with 10-frame binning, each trial is 360 binned time points. The number of complete trials depends on session length: 20-minute sessions yield 10 trials, 30-minute sessions yield 15 trials. Leftover frames that don't fill a complete 2-minute block are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360

def split_into_trials(neural_binned, me_binned, trial_frames):
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // trial_frames
    neural_trials = []
    me_trials = []
    for t in range(n_trials):
        start = t * trial_frames
        end = start + trial_frames
        neural_trials.append(neural_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
    return neural_trials, me_trials
```

iii. The AI cited the paper: "splits were done on consecutive 2 minute blocks." The trial splitting approach directly follows this description. The AI noted that some sessions are 30 minutes (not 20 as stated in the paper) and chose to use all available data.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial-level quality filtering is applied. All complete 2-minute blocks are included. Incomplete blocks at the end of a recording are silently discarded by the integer division in `n_trials = n_bins // trial_frames`.

ii.
```python
n_trials = n_bins // trial_frames  # incomplete trials discarded
```

iii. The AI stated in CONVERSION_NOTES.md: "No explicit curation; missing video frames interpolated." The AI noted that the data is pre-filtered by track2p and no further trial filtering criteria were found in the reference paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from two suite2p output files per session: `F.npy` (raw fluorescence traces, shape n_neurons x n_frames) and `Fneu.npy` (neuropil fluorescence traces, same shape).

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The AI documented that the paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and that computing dF/F requires both F and Fneu for neuropil correction.

## 2-b. How is the `neural` data processed?

i. Neural processing has three stages:
1. Neuropil correction: `Fc = F - 0.7 * Fneu` (coefficient from suite2p ops)
2. Baseline estimation using the maximin method: rolling minimum, then rolling maximum over a window of `win_baseline * fs = 1800` frames, then Gaussian smoothing with sigma=10 frames
3. Baseline subtraction: `dF/F = Fc - F0` (subtraction, not division)
4. Temporal binning: averaging 10 consecutive frames

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

dff_binned = bin_data(dff, bin_size)  # averaging 10 consecutive frames
```

iii. The AI justified this by referencing Suite2p's `dcnv.preprocess()` function and the paper's statement about using "default Suite2p parameters." The AI initially had bugs (using `sig_baseline * fs = 300` instead of `10`, and division instead of subtraction) which were caught and fixed during development.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI determined that the provided data has already been filtered by track2p — only neurons that are successfully tracked across all sessions are included. The AI verified that all `iscell` values in the provided data are 1.0.

ii.
```python
# No filtering code — all neurons from F.npy are used directly
dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)
```

iii. The AI stated: "All iscell values are 1.0 in the provided data (pre-filtered by track2p)" and "Neurons: iscell > 0.5 + track2p matching (already applied in data)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 2-minute recording block. Trials are sequential non-overlapping chunks of the continuous recording, starting from time 0. There is no alignment to a specific behavioral event; the "alignment event" is simply the start of each 2-minute block.

ii.
```python
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
```

iii. The AI set `temporal_alignment_event` to "Start of 2-minute recording block" and `off_start` to 0.0, `off_end` to 120s (the trial duration). This is consistent with the paper's description of splitting into "consecutive 2 minute blocks."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 333.33 ms (10 frames at 30 Hz). Rebinning is applied by averaging 10 consecutive raw frames, matching the paper's description.

ii.
```python
BIN_SIZE = 10  # number of frames per bin
FS = 30.0  # imaging frame rate (Hz)

def bin_data(data, bin_size):
    if data.ndim == 1:
        n = len(data)
        n_bins = n // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        n = data.shape[-1]
        n_bins = n // bin_size
        trimmed = data[..., :n_bins * bin_size]
        new_shape = trimmed.shape[:-1] + (n_bins, bin_size)
        return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The AI cited the paper: "averaging in bins of 10 consecutive timestamps" and confirmed the effective rate is 3 Hz. The metadata records `time_bin_size: 333.33 ms`.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data variable. It is synthetically computed from the bin index within each trial. The time input represents elapsed time from the start of each 2-minute trial (not from the start of the full experiment/session).

ii.
```python
for session_trials in neural_all:
    session_inputs = []
    for trial in session_trials:
        n_timepoints = trial.shape[1]
        time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
        session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
    input_all.append(session_inputs)
```

iii. The AI described this as "Elapsed time in seconds from start of 2-min trial" in its mapping plan. The range is [0, 119.7] seconds for every trial.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The processing is a simple linear computation: `np.arange(n_timepoints) * (BIN_SIZE / FS)`, which creates a ramp from 0 to ~119.67 seconds in steps of 0.333 seconds. This is identical for every trial across all sessions.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
# BIN_SIZE / FS = 10 / 30 = 0.3333 seconds per bin
```

iii. The AI chose this representation as "Time elapsed in seconds from start of 2-min trial."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is constructed to have exactly the same number of timepoints as the neural data for each trial (n_timepoints = trial.shape[1] = 360). Since both are indexed identically, they are inherently aligned.

ii.
```python
n_timepoints = trial.shape[1]  # matches neural trial timepoints
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. The AI ensured alignment by deriving the number of time points directly from the neural trial shape.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Additionally, `interframe_int.npy` is used to handle missing video frames during interpolation.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI noted that motion energy was "Already computed in motion_energy_glob.npy" and referenced the data README's description of the move_deve directory.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy processing has three steps:
1. Missing frame interpolation: uses interframe intervals to detect dropped frames and interpolates motion energy values at those positions
2. Temporal binning: averaging 10 consecutive frames (same as neural data)
3. Quintile discretization: computing global percentile-based bin edges across all sessions, then discretizing each value into 5 bins

ii.
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
me_binned = bin_data(me_interp, bin_size)
# Later:
output_all, bin_edges, bin_labels = discretize_motion_energy(me_all_raw, N_OUTPUT_BINS)
```

iii. The AI cited the data README regarding missing frames and the instructions requiring "five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins. Global percentile edges are computed across all sessions' binned motion energy values using `np.percentile` at [0, 20, 40, 60, 80, 100] percentiles. Values are then binned using `np.digitize` with the inner edges, producing integer labels 0-4.

ii.
```python
def discretize_motion_energy(all_me_trials, n_bins=N_OUTPUT_BINS):
    all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)
    # Handle duplicate edges
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i-1]:
            bin_edges[i] = bin_edges[i-1] + 1e-10
    discretized = []
    for session_trials in all_me_trials:
        session_disc = []
        for me in session_trials:
            binned = np.digitize(me, bin_edges[1:-1])
            binned = np.clip(binned, 0, n_bins - 1)
            session_disc.append(binned.reshape(1, -1).astype(np.int64))
        discretized.append(session_disc)
    bin_labels = [f'{percentiles[i]:.0f}-{percentiles[i+1]:.0f}%ile' for i in range(n_bins)]
    return discretized, bin_edges, bin_labels
```

iii. The AI's instructions specified "five equal-percentile bins" and the output distribution confirms exactly 20% per bin globally.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned with neural data through shared frame-level indexing. After interpolating missing video frames to match the neural frame count, both signals are binned with the same bin size (10 frames) and split into the same 2-minute trial blocks. This ensures temporal alignment at every step.

ii.
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)  # match neural frame count
me_binned = bin_data(me_interp, bin_size)  # same binning as neural
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)  # same split
```

iii. The AI stated: "Neural and video are synchronized at 30 Hz; missing frames interpolated."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The primary data quality issue is missing video frames (up to 148 per session). The AI handles these by detecting gaps using interframe intervals: if the interval between consecutive frames is roughly N times the median interval, N-1 frames are considered missing. The motion energy is then interpolated (using `np.interp`) to fill all neural frame positions. If motion energy has more frames than neural data, it is truncated.

ii.
```python
def interpolate_missing_frames(motion_energy, interframe_int, n_neural_frames):
    n_me = len(motion_energy)
    if n_me == n_neural_frames:
        return motion_energy.astype(np.float64)
    n_missing = n_neural_frames - n_me
    if n_missing < 0:
        return motion_energy[:n_neural_frames].astype(np.float64)
    median_ifi = np.median(interframe_int)
    ratios = interframe_int / median_ifi
    missed_counts = np.round(ratios).astype(int) - 1
    me_positions = np.zeros(n_me, dtype=int)
    me_positions[0] = 0
    for i in range(1, n_me):
        me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
    neural_positions = np.arange(n_neural_frames)
    me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
    return me_interp
```

iii. The AI referenced the data README: "The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over."

## 6-a. What are the most time-consuming steps of the code?

i. The dF/F baseline computation (maximin method with rolling min/max/Gaussian filters over large windows) is the dominant cost. The full conversion of 41 sessions takes ~60 seconds, with per-session times of 0.5-2.3s depending on neuron count. The discretization step is negligible by comparison.

ii.
```python
# Most expensive operations:
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)  # win_frames=1800
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
```

iii. The AI documented timing per session and noted the initial buggy version (sigma=300 instead of 10) took 1420s vs 60s after the fix, highlighting the baseline computation as the bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The `interpolate_missing_frames` function uses a Python for-loop to compute `me_positions` cumulatively. This could use `np.cumsum`.
2. The `split_into_trials` function loops over trials to slice arrays. This could be done with `np.array_split` or reshaping.
3. The `discretize_motion_energy` function has nested loops over sessions and trials. The digitization could be vectorized by concatenating all values, digitizing once, then splitting back.

ii.
```python
# Loop in interpolate_missing_frames:
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]

# Loop in split_into_trials:
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))

# Nested loops in discretize_motion_energy:
for session_trials in all_me_trials:
    for me in session_trials:
        binned = np.digitize(me, bin_edges[1:-1])
```

iii. The AI noted timing but did not explicitly discuss vectorization opportunities. The total runtime of 60s was considered acceptable.

## 6-c. What processing does the code repeat multiple times?

i. No major processing is repeated. Each session is loaded and processed exactly once. The motion energy discretization is done once globally after all sessions are processed. The time input computation loops over all trials but each computation is trivial. No data is re-loaded or re-computed.

ii. N/A - no repeated processing identified.

iii. The AI designed the pipeline as a single sequential pass through all sessions, followed by global discretization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Two forms of potentially unnecessary processing:
1. The `--show-processing` plotting code generates detailed visualizations that are not used in the final data product (though only activated by flag).
2. The `bin_edges` and `bin_labels` computed during discretization are used for metadata but the raw binned motion energy (`me_all_raw`) is discarded after discretization.
3. The full dF/F baseline computation processes the entire continuous recording, including frames at the very end that may be discarded when splitting into complete 2-minute trials (partial blocks).

ii.
```python
# Partial frames at end of sessions discarded:
n_bins = n // bin_size  # truncates
n_trials = n_bins // trial_frames  # discards partial trials
```

iii. The AI did not explicitly discuss unnecessary processing. The overall design is relatively lean.
