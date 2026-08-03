# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subject directories starting with `jm` in the data directory, then finds all session subdirectories within each subject. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. All 6 subjects and 41 sessions are processed.

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

# In process_session:
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The AI followed the standard directory structure convention where subject folders contain session subfolders. The CONVERSION_NOTES.md documents the data organization thoroughly.

## 1-b. How are the data split into subjects?

i. Subjects are directories matching `jm*` in the data directory, sorted alphabetically. Six subjects found: jm031, jm032, jm038, jm039, jm040, jm046.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, sorted alphabetically. Each subdirectory contains one daily recording. This produces 41 total sessions (7+7+7+7+6+7).

ii.
```python
sess_list = sorted([s for s in os.listdir(subj_path)
                   if os.path.isdir(os.path.join(subj_path, s))])
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as 2-minute (120-second) non-overlapping segments of the continuous recording. With 10-frame binning at 30 Hz, each trial is 360 binned timepoints (3600 raw frames). The AI chose 2-minute blocks because the reference paper uses "consecutive 2-minute blocks" for cross-validation splits. 20-minute sessions yield 10 trials; 30-minute sessions yield 15 trials.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)  # 3600 raw frames per trial
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE  # 360 binned frames per trial

# In split_into_trials:
n_trials = n_timepoints // trial_length
for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))
```

iii. The AI justified this by citing the paper's CV structure which uses 2-minute blocks. The CONVERSION_NOTES state: "Split each session into 2-minute blocks (as used for CV in paper)."

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All trials from all sessions are included. Remainder frames that don't fill a complete trial are discarded.

ii. N/A (no filtering code)

iii. No explicit trial curation rules were mentioned in the paper for this dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) neuropil subtraction `Fc = F - 0.7 * Fneu`, (2) suite2p `preprocess` with maximin baseline subtraction (win_baseline=60, sig_baseline=10, fs=30), (3) temporal binning by averaging 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, neucoeff=NEUCOEFF, win_baseline=WIN_BASELINE,
                sig_baseline=SIG_BASELINE, fs=FS):
    Fc = F - neucoeff * Fneu
    device = torch.device('cpu')
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
    return dff

# Then binned:
dff_binned = bin_data(dff, BIN_SIZE)

def bin_data(data, bin_size=BIN_SIZE):
    if data.ndim == 2:
        n_neurons, n_frames = data.shape
        n_bins = n_frames // bin_size
        data_trimmed = data[:, :n_bins * bin_size]
        return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
```

iii. The AI's CONVERSION_NOTES document the neuropil coefficient of 0.7 and maximin baseline method as matching the paper. The binning of 10 frames is justified by citing the paper's methods: "averaging using a bin size of 10 frames."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the suite2p `F.npy` output are included. The AI notes that all `iscell` values are already 1, meaning Track2p pre-filtered the cells.

ii. N/A (no filtering code)

iii. CONVERSION_NOTES state: "All iscell values are 1 (all cells already filtered)" and "Suite2p iscell > 0.5 (already applied in Track2p output)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since the recording is continuous with no stimulus events, trials are simply contiguous segments from the beginning of each session. No event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames. The native 30 Hz data (33.33 ms per frame) is binned to 3 Hz (333.33 ms per bin). This is based on the paper's methods describing "averaging using a bin size of 10 frames."

ii.
```python
BIN_SIZE = 10  # number of frames to average
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # 333.33 ms

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. CONVERSION_NOTES: "Binning for decoding: 10 frames - 'averaging using a bin size of 10 frames'" from the paper's methods.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the binned frame index, the bin size, and the frame rate. Each binned timepoint represents the center of its temporal bin.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. Since the frame rate is constant at 30 Hz, computing time from frame indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as (bin_index * BIN_SIZE + BIN_SIZE/2) / FS, giving seconds from session start at the center of each temporal bin. For the first bin this is 0.167s, for the last bin of a 20-min session it's ~1199.8s.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
# For each trial:
trial_time = time_input[start:end].astype(np.float32)
input_trials_list.append(trial_time.reshape(1, -1))
```

iii. The AI uses bin centers rather than bin edges for the time representation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same binned frame indices used for the neural data, so they are inherently aligned. Both share the same temporal grid after binning.

ii.
```python
# Same binning applied to neural and time:
dff_binned = bin_data(dff, BIN_SIZE)
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
# n_binned = dff_binned.shape[1]
```

iii. The indexing ensures the time input and neural data cover the same temporal range.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Camera timestamps from `tstamps.npy` are used to detect and handle dropped frames.

ii.
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from behavioral video. Timestamps are needed for dropped frame detection.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) dropped frames detected via inter-frame interval analysis of tstamps and interpolated; (2) temporal binning by averaging 10 frames; (3) discretized into 5 equal-percentile bins per session; (4) no normalization by standard deviation is applied before discretization.

ii.
```python
# Dropped frame alignment:
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)

# Binning:
me_binned = bin_data(me_aligned, BIN_SIZE)

# Per-session discretization:
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The AI chose per-session discretization to account for different motion levels across days/mice. CONVERSION_NOTES state: "Bins computed per session to account for different motion levels across days/mice."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins computed per session. The percentile boundaries are at 20th, 40th, 60th, 80th percentiles of each session's binned motion energy. `np.digitize` assigns values to bins 0-4.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. Per-session binning ensures exactly 20% of timepoints fall in each bin within each session, as confirmed by the output logs showing perfectly equal bin counts.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses camera timestamps (tstamps.npy) to detect dropped frames. Inter-frame intervals are computed from the timestamps, and gaps exceeding 1.5x the median interval indicate dropped frames. A mapping from camera frame index to neural frame index is built, then NaN-interpolation fills the missing entries to produce a motion energy array matching the neural frame count.

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

iii. The AI noted timestamps are in kiloseconds and uses ratio-based detection of dropped frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of data issues are handled: (1) dropped camera frames are detected via inter-frame interval analysis and interpolated to match neural frame count; (2) remainder frames at the end of sessions that don't fill a complete trial are discarded (no remainder occurs because sessions are exact multiples of trial length after binning).

ii.
```python
# Dropped frame handling in align_motion_energy (see 4-d above)

# Trial splitting discards remainder:
n_trials = n_timepoints // trial_length
```

iii. The dropped frame interpolation ensures motion energy aligns with neural data. The trial splitting naturally handles partial trials.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` (maximin baseline correction), taking 0.3-1.6s per session depending on neuron count. The AI runs this on CPU (not GPU), which is slower than the reference approach. Total conversion takes ~45s.

ii.
```python
device = torch.device('cpu')
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
```

iii. The CONVERSION_NOTES confirm dF/F computation is the bottleneck at 0.4-1.6s per session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame alignment loop iterates frame-by-frame to build the neural index mapping. The trial splitting also uses a loop rather than array reshaping for the 1D case.

ii.
```python
# Frame-by-frame loop in align_motion_energy:
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped

# Trial splitting loop:
for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[start:end].astype(np.float32))
```

iii. The dropped frame loop is O(n_frames) but has a sequential dependency (cumulative sum), though it could be vectorized with cumsum. The trial splitting loop could use reshape for the 2D case (which it does) but not the 1D case.

## 6-c. What processing does the code repeat multiple times?

i. The `process_session` function is called independently for each session, which means there is no shared state between sessions. The discretization is done per-session, so no repeated global computation exists. However, the bin_data function is called separately for neural and behavioral data, each doing similar reshape-and-mean operations.

ii. N/A

iii. Each session is processed independently, so there is no unnecessary repetition.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code copies the Fc array before preprocessing (`Fc.copy()`), though this may be necessary if `preprocess` modifies in-place. No significant unnecessary processing is performed. The show-processing plotting code only runs when explicitly requested.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
```

iii. The code is relatively lean with no major unnecessary processing steps.
