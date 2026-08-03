# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all subject directories (sorted alphabetically, filtered by `jm` prefix) in the `data/` directory. For each subject, it iterates over all session subdirectories (sorted chronologically). For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. Data is loaded using `np.load()`.

ii.
```python
def get_subjects_and_sessions(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    sessions = {}
    for subj in subjects:
        subj_path = os.path.join(data_dir, subj)
        sess_list = sorted([s for s in os.listdir(subj_path)
                           if os.path.isdir(os.path.join(subj_path, s))])
        sessions[subj] = sess_list
    return subjects, sessions
```

```python
# Inside process_session:
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The AI noted from the data README that the data is organized as `data/{subject_id}/{date}_a/{suite2p,move_deve}/` and followed the same loading pattern as the reference `load_data.ipynb` notebook, which loads F.npy from `suite2p/plane0/`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the top-level subdirectories in the `data/` directory that start with `jm`. Each directory corresponds to one mouse. A list of unique used subjects is maintained, and each session is assigned a subject index.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

```python
if subj not in used_subjects:
    used_subjects.append(subj)
subject_idx_list.append(used_subjects.index(subj))
```

iii. The AI identified 6 subjects (jm031=A, jm032=B, jm038=C, jm039=D, jm040=E, jm046=F) from the data README and paper, which maps them alphabetically to mouse A-F.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject directory is treated as one session (e.g., `jm031/2023-10-18_a`). Sessions are sorted chronologically. Each session corresponds to one recording day. The total is 41 sessions (7+7+7+7+6+7).

ii.
```python
for subj in subjects:
    subj_path = os.path.join(data_dir, subj)
    sess_list = sorted([s for s in os.listdir(subj_path)
                       if os.path.isdir(os.path.join(subj_path, s))])
    sessions[subj] = sess_list
```

iii. The AI noted that the paper describes daily recordings from P8 to P14, with one session per day per mouse. The data organization matches this (one folder per recording day per subject).

## 1-d. How are the data split into trials?

i. Each session's continuous recording is split into fixed-length 2-minute blocks (trials). After binning by 10 frames, each trial contains 360 binned timepoints. Sessions with 36000 raw frames (20 min) yield 10 trials; sessions with 54000 raw frames (30 min) yield 15 trials.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)  # 3600 raw frames per trial
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE  # 360 binned frames per trial

def split_into_trials(data, trial_length):
    if data.ndim == 2:
        n_neurons, n_timepoints = data.shape
        n_trials = n_timepoints // trial_length
        trials = []
        for i in range(n_trials):
            start = i * trial_length
            end = start + trial_length
            trials.append(data[:, start:end].astype(np.float32))
        return trials
```

iii. The AI justified this by noting the paper uses "splits were done on consecutive 2 minute blocks" for cross-validation, making 2-minute blocks the natural trial unit. The experiment involves spontaneous behavior with no explicit trial structure.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All trials derived from the continuous recordings are included.

ii. No filtering code is present. All trials from `split_into_trials` are directly appended to the output.

iii. The AI noted that the paper does not mention explicit trial curation rules. Since the experiment involves continuous spontaneous behavior (no task structure with correct/incorrect trials), there is no basis for trial-level quality control beyond handling missing camera frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from two raw variables: `F.npy` (raw fluorescence traces, shape n_neurons x n_timepoints) and `Fneu.npy` (neuropil fluorescence, same shape).

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The AI followed the paper's description: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The reference notebook `load_data.ipynb` also loads F.npy and notes that for proper analysis one should compute dF/F as described in the paper.

## 2-b. How is the `neural` data processed?

i. Processing steps:
1. Neuropil correction: `Fc = F - 0.7 * Fneu`
2. Baseline subtraction using Suite2p's `preprocess` function with the `maximin` method (Gaussian smoothing with sig=10 frames, then min filter with window=1801 frames, then max filter with same window, then subtract baseline). This is NOT dF/F by division; it is baseline subtraction only.
3. Temporal binning: average every 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, neucoeff=NEUCOEFF, win_baseline=WIN_BASELINE,
                sig_baseline=SIG_BASELINE, fs=FS):
    Fc = F - neucoeff * Fneu
    device = torch.device('cpu')
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
    return dff
```

```python
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The AI investigated the Suite2p source code and discovered that `preprocess` performs baseline subtraction (F - Flow), not division (F-Flow)/Flow. The paper says "baseline corrected fluorescence traces as our dF/F" and uses Suite2p default parameters (neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, fs=30, baseline='maximin').

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. All neurons in the data are used, as they are already pre-filtered by the Track2p pipeline (all `iscell` values are 1).

ii. No filtering code is present. The AI verified:
```python
# From CONVERSION_NOTES: "All iscell values are 1 (all cells already filtered)"
```

iii. The AI noted that the Track2p output only includes neurons successfully tracked across all days for a given mouse. The Suite2p iscell threshold of 0.5 was already applied during Track2p processing, so no additional filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the session start. There is no specific event alignment since the experiment involves continuous spontaneous behavior. Trials are simply consecutive 2-minute blocks from session start. The temporal alignment event is documented as `'session_start'`.

ii.
```python
'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': None,
```

iii. The AI noted that there is no explicit trial onset event in this spontaneous behavior paradigm. Each trial begins at a fixed offset from session start (trial_index * 2 minutes). The off_start is 0.0 (trials start at the beginning of the recording).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 333.33 ms (10 frames at 30 Hz). Temporal rebinning is applied by averaging every 10 consecutive frames.

ii.
```python
FS = 30.0  # imaging rate in Hz
BIN_SIZE = 10  # number of frames to average
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # 333.33 ms

def bin_data(data, bin_size=BIN_SIZE):
    if data.ndim == 2:
        n_neurons, n_frames = data.shape
        n_bins = n_frames // bin_size
        data_trimmed = data[:, :n_bins * bin_size]
        return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
```

iii. The AI referenced the paper: "averaging using a bin size of 10 frames" and "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data variable. It is computed synthetically from the frame indices and the known imaging rate (30 Hz) and bin size (10 frames).

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The AI noted that the decoder input is "time elapsed from beginning of experiment" as specified in the instructions. Since imaging is at a constant 30 Hz, the time of each binned timepoint can be computed directly from its index.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as the center of each time bin, in seconds from session start. For binned timepoint index `i`, the time is `(i * 10 + 5) / 30` seconds. This gives values starting at 0.167s and ending at 1199.8s (for 20-min sessions) or 1799.8s (for 30-min sessions).

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The AI chose to use the center of each time bin (adding BIN_SIZE/2 before dividing by sampling rate) to represent the time more accurately. This is a standard approach for binned data.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because it is computed from the same binned time indices. Each binned timepoint `i` in the neural data corresponds to time `(i * 10 + 5) / 30` seconds. The time input is split into trials using the same splitting logic as the neural data, preserving absolute time (not resetting per trial).

ii.
```python
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
```

iii. The AI noted that time input is "time elapsed from beginning of experiment" per the instructions, so absolute time from session start is used (not reset per trial).

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` (motion energy computed from videography of spontaneous behavior) and `tstamps.npy` (camera timestamps in kiloseconds, used for alignment when camera frames are missing).

ii.
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The AI identified motion energy as the behavioral variable to decode, consistent with the paper's description of motion energy extracted from videography of spontaneous behavior.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps:
1. Align motion energy to neural frames (handle missing camera frames via IFI-based alignment and interpolation)
2. Bin by averaging 10 consecutive frames
3. Discretize into 5 equal-percentile bins per session using `np.percentile` at 20th, 40th, 60th, 80th percentiles, then `np.digitize`

ii.
```python
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
me_binned = bin_data(me_aligned, BIN_SIZE)
me_discrete = discretize_motion_energy(me_binned, N_BINS)
```

```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The AI followed the instruction to discretize into "five equal-percentile bins" and chose per-session normalization to account for varying motion levels across recording days and mice.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories using equal-percentile (quintile) bins computed per session. Bin edges are at the 20th, 40th, 60th, and 80th percentiles of the session's binned motion energy. `np.digitize` assigns each value to bin 0-4. This produces exactly 20% of data in each bin per session.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The AI chose per-session discretization based on the instruction to use "five equal-percentile bins." The verification confirmed exactly 20% per bin per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural frames using inter-frame interval (IFI) analysis. When camera frames are missing (fewer ME frames than neural frames), the IFI from timestamps is used to detect dropped frames (gaps > 1.5x median interval). Each ME value is mapped to its correct neural frame index, and missing values are linearly interpolated. After alignment, ME has the same number of frames as the neural data, and both are binned and split into trials using the same indices.

ii.
```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    ifi = np.diff(tstamps)
    median_ifi = np.median(ifi)
    neural_idx = np.zeros(len(me), dtype=int)
    neural_idx[0] = 0
    for i in range(1, len(me)):
        n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
        neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
    neural_idx = np.clip(neural_idx, 0, n_neural_frames - 1)
    me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_full[neural_idx] = me.astype(np.float64)
    valid = ~np.isnan(me_full)
    if not valid.all():
        indices = np.arange(n_neural_frames)
        me_full = np.interp(indices, indices[valid], me_full[valid])
    return me_full
```

iii. The AI initially used a timestamp-based mapping but discovered it was incorrect (timestamps accumulated differently than neural frame times for dropped frames). The AI then switched to IFI-based alignment, which correctly identifies dropped frames by detecting gaps in inter-frame intervals. This improved decoder accuracy from 0.2866 to 0.2954.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where ME has fewer frames than neural data) are handled by:
1. Using IFI analysis to detect which neural frames lack camera data
2. Mapping existing ME values to their correct neural frame indices
3. Linearly interpolating missing values using `np.interp`

Sessions where ME frame count matches neural frame count require no special handling. The number of missing frames ranges from 0 to 148 across sessions.

ii.
```python
# In align_motion_energy:
me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
valid = ~np.isnan(me_full)
if not valid.all():
    indices = np.arange(n_neural_frames)
    me_full = np.interp(indices, indices[valid], me_full[valid])
```

iii. The AI followed the data README which states: "The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over." The AI chose interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The dF/F computation using Suite2p's `preprocess` function is by far the most time-consuming step, taking 0.34-1.56 seconds per session (depending on neuron count and session length). The total conversion time is ~45 seconds for all 41 sessions, with dF/F accounting for the vast majority. Binning and other operations take <0.1s per session.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
print(f"    dF/F computation: {time.time()-t1:.2f}s")
```

iii. The AI documented timing information and noted that dF/F computation dominates processing time. The total conversion time of ~45 seconds was considered acceptable (well under the 15-minute threshold mentioned in the instructions).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `align_motion_energy` function contains a Python for-loop over all camera frames to build the neural index mapping. This could be vectorized using numpy operations on the IFI array. The `split_into_trials` function also uses a Python for-loop that could be replaced with array reshaping.

ii.
```python
# In align_motion_energy - could be vectorized:
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
```

```python
# In split_into_trials - could be vectorized:
for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))
```

iii. The AI did not explicitly discuss vectorization opportunities. The loops were kept because they are fast enough (the total conversion time is only ~45 seconds).

## 6-c. What processing does the code repeat multiple times?

i. The dF/F computation is done independently per session, but since each session has independent neural data, this repetition is necessary. The `bin_data` function is called separately for neural data and motion energy, but since these are different data streams, this is also necessary. No truly redundant processing is present - each call operates on different data.

ii. No redundant processing code to show.

iii. The AI did not identify any redundant processing in the conversion notes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes the time input as center-of-bin absolute time (`(i * 10 + 5) / 30`), which includes the BIN_SIZE/2 offset. This offset is unnecessary for a monotonically increasing time series used as decoder input - the decoder would learn the same mapping regardless of the constant offset. Additionally, the code loads `tstamps.npy` even for sessions where ME frame count matches neural frame count (though it returns early in that case).

ii.
```python
# BIN_SIZE/2 offset is cosmetic, not functionally necessary:
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The AI did not explicitly discuss unnecessary processing. The overall code is relatively lean with minimal wasted computation.
