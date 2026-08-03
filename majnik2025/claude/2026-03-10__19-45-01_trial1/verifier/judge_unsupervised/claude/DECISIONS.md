# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subject directories within the `data/` folder, then iterates over each session subdirectory. For each session, it loads `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. Data is loaded per-session, processed, and accumulated into lists.

ii.
```python
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

iii. The AI identified from the data README and `load_data.ipynb` that Suite2p-format data and movement data are stored in these standard locations. The AI noted that the data is already track2p output with pre-filtered neurons.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by top-level directories under the data folder. They are sorted alphabetically to produce the subject list: `['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`. A `subject_idx` array maps each session to its subject.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
# ...
subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))
# ...
subject_idx.append(subject_list.index(subj))
```

iii. The AI followed the data organization described in the data README: "For each subject there is a folder corresponding to the subject id."

## 1-c. How are the data split into sessions?

i. Sessions are identified as subdirectories within each subject folder, sorted alphabetically (chronologically). Each session directory corresponds to one recording day. All 41 sessions are included (7 per subject, except jm040 with 6).

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. The AI identified from the data README that session folders correspond to recording dates in YYYY-MM-DD format.

## 1-d. How are the data split into trials?

i. Continuous recordings are split into fixed-length 2-minute (120s) blocks. At the binned resolution (10 frames/bin at 30 Hz = 3 Hz), each trial is 360 time bins. Sessions with 36000 frames (20 min) yield 10 trials; sessions with 54000 frames (30 min) yield 15 trials. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360

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

iii. The AI cited the paper: "splits were done on consecutive 2 minute blocks." This produces 545 total trials (14 sessions x 10 + 27 sessions x 15).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks are included. Partial blocks at the end of sessions are discarded by the integer division in `n_trials = n_bins // trial_frames`.

ii.
```python
n_trials = n_bins // trial_frames  # discards partial block at end
```

iii. The AI noted: "No explicit curation; missing video frames interpolated." The CONVERSION_NOTES state that neuron filtering was already applied by track2p (all iscell = 1.0 in the provided data), so no additional neuron or trial filtering was needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence traces), both loaded from the Suite2p output directory for each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The AI identified these as the standard Suite2p output files, consistent with the `load_data.ipynb` notebook and the paper's description of using Suite2p for calcium imaging data processing. The AI noted that `spks.npy` (deconvolved spikes) also exists but chose `F` and `Fneu` to compute dF/F following the paper's description.

## 2-b. How is the `neural` data processed?

i. Processing pipeline: (1) Neuropil correction: `Fc = F - 0.7 * Fneu`; (2) Maximin baseline estimation using rolling minimum, rolling maximum, and Gaussian smoothing; (3) Baseline subtraction: `dff = Fc - F0`; (4) Temporal binning by averaging 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu
    win_frames = int(win_baseline * fs)  # 1800 frames
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
    dff = Fc - F0
    return dff

def _maximin_baseline(Fc, win_frames, sig_frames):
    Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
    Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
    Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
    return Flow
```

iii. The AI documented this decision carefully. It initially had bugs (dividing by baseline, wrong sigma), which it caught and fixed. The paper states: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." Suite2p's `dcnv.preprocess()` uses subtraction (Fc - F0), not division. Parameters from `ops.npy`: neucoeff=0.7, baseline='maximin', win_baseline=60.0, sig_baseline=10.0.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural quality filtering is applied. The AI determined that the data was already pre-filtered by track2p, with all `iscell` values set to 1.0. All neurons present in the data files are included.

ii. No explicit filtering code exists in `convert_data.py`. The AI loads all neurons from `F.npy` directly.

iii. From CONVERSION_NOTES: "The provided data is ALREADY track2p output in suite2p format — neurons are pre-filtered" and "All iscell values are 1.0 in the provided data (pre-filtered by track2p)." The data README confirms: "the data only includes traces for the cells present across all days."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each 2-minute recording block. The continuous recording is split into consecutive non-overlapping 2-minute blocks, so each trial starts at time 0 (the beginning of that block). The `temporal_alignment_event` is set to "Start of 2-minute recording block."

ii.
```python
'temporal_alignment_event': 'Start of 2-minute recording block',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,  # 120
```

iii. The AI chose block-start alignment because the paper describes "splits were done on consecutive 2 minute blocks" — there is no stimulus or behavioral event to align to in this spontaneous activity paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 333.33 ms (10 frames at 30 Hz = 3 Hz effective rate). The raw 30 Hz data is rebinned by averaging 10 consecutive frames, matching the paper's description.

ii.
```python
BIN_SIZE = 10  # number of frames per bin
FS = 30.0  # imaging frame rate (Hz)

def bin_data(data, bin_size):
    n = data.shape[-1]
    n_bins = n // bin_size
    trimmed = data[..., :n_bins * bin_size]
    new_shape = trimmed.shape[:-1] + (n_bins, bin_size)
    return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The paper states: "averaging in bins of 10 consecutive timestamps." The AI correctly implements this as averaging, and the metadata reports `time_bin_size: 333.33` ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data variable. It is synthetically constructed from the time bin indices within each trial, using the known bin size (10 frames / 30 Hz = 0.333s per bin).

ii.
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
    session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The AI computed time elapsed from the start of each trial (not the full experiment). No raw data variable is needed since the time axis is implicit in the binned frame indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The input is computed as `np.arange(360) * (10 / 30)`, giving values from 0 to 119.67 seconds in steps of 0.333s. This represents time elapsed within each 2-minute trial, not time from the start of the full experiment.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
# Result: [0.0, 0.333, 0.667, ..., 119.667]
```

iii. The AI named this input `time_elapsed_s`. It resets to 0 at the start of each trial rather than accumulating across the full session. The instructions specified "Time elapsed from the beginning of the experiment," which could be interpreted as cumulative time from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time vector is constructed to have the same number of time bins as the neural data for each trial (360 bins), so alignment is trivial — each time bin corresponds to the same time bin in the neural data.

ii.
```python
n_timepoints = trial.shape[1]  # same as neural timepoints
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. Since both are derived from the same binning structure, no explicit alignment step is needed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy`, loaded from the `move_deve/` subdirectory of each session. Additionally, `interframe_int.npy` is used to handle missing video frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The data README describes this as "processed behavioural data (motion energy extracted from videography of spontaneous behaviour)."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing pipeline: (1) Interpolate missing video frames to match neural frame count using interframe intervals; (2) Bin by averaging 10 consecutive frames (same as neural data); (3) Discretize into 5 equal-percentile (quintile) bins across all sessions globally.

ii.
```python
# 1. Interpolate missing frames
me_interp = interpolate_missing_frames(me, interframe, n_frames)

# 2. Bin
me_binned = bin_data(me_interp, bin_size)

# 3. Discretize globally
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The AI handled missing frames by detecting gaps via interframe intervals and interpolating. Quintile discretization was chosen based on the instruction: "Motion energy, normalized and discretized into five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 quintile bins (0-20th, 20-40th, 40-60th, 60-80th, 80-100th percentile). Percentile boundaries are computed globally across all sessions and trials. Values are assigned bin indices 0-4 using `np.digitize`.

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
    # Discretize
    binned = np.digitize(me, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
```

iii. The instructions specified "five equal-percentile bins." The AI's global quintile approach ensures roughly 20% of data per bin across all sessions. The output distribution confirms this: each bin has exactly 20% of the data.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned with neural data by matching the frame count. If there are missing video frames, they are interpolated using interframe intervals to match the neural frame count. Then both are binned identically (10-frame bins) and split into the same trial boundaries.

ii.
```python
# In process_session:
me_interp = interpolate_missing_frames(me, interframe, n_frames)  # match to n_frames of F
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)

# In split_into_trials: same trial boundaries for both
neural_trials.append(neural_binned[:, start:end])
me_trials.append(me_binned[start:end])
```

iii. The AI ensured frame-level alignment before binning by interpolating missing video frames. The same bin boundaries and trial splits are applied to both neural and behavioral data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames (where ME frame count < neural frame count) are handled by interpolating motion energy values using interframe intervals to detect gap positions. Up to 148 frames can be missing in some sessions. If ME has more frames than neural (shouldn't happen), it is truncated.

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

iii. The data README notes: "In some recordings there might be some missing frames from the camera... they can be interpolated over." The AI implemented interpolation using interframe intervals to identify gap positions, which is one of the suggested approaches.

## 6-a. What are the most time-consuming steps of the code?

i. The dF/F computation (maximin baseline estimation) is the most time-consuming step, involving rolling minimum, rolling maximum, and Gaussian smoothing over large arrays. The total conversion time is ~60 seconds for all 41 sessions.

ii. From conversion output:
```
Processing session 22/41: jm039/2024-04-30_a... 15 trials, 746 neurons, 2.3s
```

iii. The AI noted that the initial buggy implementation with sig_baseline*fs=300 took 1420s, while the fixed version with sig_baseline=10 took only 60s. The large Gaussian sigma was the performance bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interpolate_missing_frames` function contains a Python loop to build `me_positions`:
```python
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
```
This could be vectorized with `np.cumsum`. The trial splitting loop could also be vectorized using array reshaping.

ii.
```python
# Current loop-based approach:
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]

# Could be: me_positions = np.cumsum(1 + np.concatenate([[0], missed_counts[:-1]]))
```

iii. The AI did not explicitly identify this vectorization opportunity, though the impact is minimal since most sessions have no or very few missing frames.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. The code processes each session once in a single pass, and discretization is done once globally after all sessions are processed.

ii. The main loop processes each session exactly once:
```python
for i, (subj, sess) in enumerate(all_sessions):
    F, Fneu, me, interframe = load_session(subj_dir, sess)
    dff_binned, me_binned = process_session(F, Fneu, me, interframe)
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
```

iii. The AI designed the pipeline as a single-pass operation, avoiding redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation applies neuropil correction and baseline estimation to the full continuous recording, including any frames that fall in the remainder after trial splitting (frames beyond the last complete 2-minute block). For 20-min sessions with 36000 frames: 36000 frames / 10 = 3600 bins, 3600 / 360 = 10 trials exactly (no waste). For 30-min sessions with 54000 frames: 54000 / 10 = 5400 bins, 5400 / 360 = 15 trials exactly (no waste). So in practice, no frames are wasted.

ii.
```python
n_bins = n // bin_size  # integer division discards remainder frames
n_trials = n_bins // trial_frames  # integer division discards remainder bins
```

iii. While the code structure allows for discarding remainder frames, the actual data dimensions (36000 and 54000 frames) divide evenly by the bin size (10) and trial length (3600 frames), so no data is discarded in practice.
