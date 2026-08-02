# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the `data/` directory. Sessions are subdirectories within each subject folder. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. All subjects and sessions are iterated over to build a flat list of sessions.

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

iii. The AI documented in CONVERSION_NOTES.md that the directory structure follows `data/{subject_id}/{date}_a/{suite2p,move_deve}/`. It systematically explored all subjects and sessions and noted the dataset contains 6 subjects and 41 sessions total.

## 1-b. How are the data split into subjects?

i. Subjects are identified as directories starting with `jm` in the data directory, sorted alphabetically. The AI maintains a `used_subjects` list to track unique subjects and maps each session to its subject via index.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

# In main loop:
used_subjects = []
for sess_i, (subj, sess_name) in enumerate(session_list):
    if subj not in used_subjects:
        used_subjects.append(subj)
    subject_idx_list.append(used_subjects.index(subj))
```

iii. Each `jm*` directory represents one mouse. The AI noted 6 subjects: jm031 (A), jm032 (B), jm038 (C), jm039 (D), jm040 (E), jm046 (F).

## 1-c. How are the data split into sessions?

i. Sessions correspond to subdirectories within each subject's folder, sorted alphabetically. Each subdirectory contains one daily recording session.

ii.
```python
sess_list = sorted([s for s in os.listdir(subj_path)
                   if os.path.isdir(os.path.join(subj_path, s))])
sessions[subj] = sess_list
```

iii. The AI identified 41 total sessions (7,7,7,7,6,7 per subject) from exploring the data directory.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. The AI defines trials as 120-second (2-minute) non-overlapping segments of the continuous recording. After binning by 10 frames, each trial contains 360 binned timepoints. Remainder frames that don't fill a complete trial are discarded.

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

iii. The AI chose 2-minute trials based on the paper's description that "splits were done on consecutive 2 minute blocks" for cross-validation in the decoding analysis.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials are included.

ii. N/A (no filtering code)

iii. The AI noted in CONVERSION_NOTES.md that "No explicit trial curation mentioned" in the reference materials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. Standard suite2p output files for calcium imaging data.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction (`Fc = F - 0.7 * Fneu`) followed by suite2p's `dcnv.preprocess` with maximin baseline correction. Then the data is temporally binned by averaging every 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, neucoeff=NEUCOEFF, win_baseline=WIN_BASELINE,
                sig_baseline=SIG_BASELINE, fs=FS):
    Fc = F - neucoeff * Fneu
    device = torch.device('cpu')
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
    return dff

# Then binning:
def bin_data(data, bin_size=BIN_SIZE):
    if data.ndim == 2:
        n_neurons, n_frames = data.shape
        n_bins = n_frames // bin_size
        data_trimmed = data[:, :n_bins * bin_size]
        return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
```

iii. The AI documented that the paper describes "averaging using a bin size of 10 frames" for both dF/F and behavior for their decoding analysis. The neuropil coefficient of 0.7 and maximin baseline are suite2p defaults matching the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in F.npy are included.

ii. N/A (no filtering code)

iii. The AI noted that all iscell values are 1 (Track2p pre-filtered to only include tracked cells), so no additional filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging every 10 consecutive frames. The native 30 Hz data is rebinned to 3 Hz (333.33 ms bins).

ii.
```python
BIN_SIZE = 10  # number of frames to average
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # 333.33 ms

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The AI cited the paper's methods: "averaging using a bin size of 10 frames" for the decoding analysis. This was applied to both neural and behavioral data.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from frame indices and the known frame rate/bin size.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. Since the frame rate is constant at 30 Hz, time can be computed from indices without raw timestamp data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each bin in seconds from session start: `(bin_index * 10 + 5) / 30`.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS

# Then split per trial with absolute session time:
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
```

iii. The AI uses bin-center times and preserves absolute session time across trials (not resetting per trial).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by construction since both neural and time arrays share the same binned indexing.

ii.
```python
# Both use same indices:
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
# input trials use same start:end indices
```

iii. No separate alignment needed - time is derived from the same frame indices as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (pre-computed global motion energy). `tstamps.npy` (camera timestamps) is used to align motion energy to neural frames when camera frames are dropped.

ii.
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The AI identified that timestamps are in kiloseconds and used them to detect dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) Align motion energy to neural frames by detecting dropped camera frames via timestamp intervals and interpolating missing values. (2) Bin by averaging every 10 frames. (3) Discretize into 5 equal-percentile bins per session.

ii.
```python
def align_motion_energy(me, n_neural_frames, tstamps):
    ifi = np.diff(tstamps)  # in kiloseconds
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

me_binned = bin_data(me_aligned, BIN_SIZE)
me_discrete = discretize_motion_energy(me_binned, N_BINS)
```

iii. The AI reasoned that camera frames drop occasionally, so timestamps are used to build a mapping from camera frame indices to neural frame indices, then interpolation fills gaps. Binning by 10 frames matches the paper's described method. Per-session percentile discretization ensures balanced classes per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins computed per session. Bin edges are at the 20th, 40th, 60th, and 80th percentiles of each session's binned motion energy.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The AI confirmed each session has approximately 20% of timepoints in each bin, as expected from equal-percentile binning.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural frames using camera timestamps (`tstamps.npy`). The inter-frame intervals are computed from timestamps, and gaps >1.5x the median interval indicate dropped frames. A mapping from camera frame indices to neural frame indices is built, and missing positions are filled via linear interpolation. After alignment and binning, the output uses the same indices as the neural data.

ii.
```python
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
# Then both are binned and split using same indices:
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The AI noted that some sessions have 1-148 fewer camera frames than neural frames, requiring alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected via timestamp analysis and interpolated. Remainder frames at the end of sessions that don't fill a complete trial are discarded. The AI verified no NaN/Inf values exist in the converted data.

ii.
```python
# Dropped frame handling in align_motion_energy (see 4-d)
# Remainder discarding in split_into_trials:
n_trials = n_timepoints // trial_length  # integer division discards remainder
```

iii. The AI documented that missing camera frames are handled via interpolation, and edge cases were verified in Step 10 sanity checks.

## 6-a. What are the most time-consuming steps of the code?

i. The suite2p `dcnv.preprocess` (maximin baseline correction) is the most time-consuming step, taking 0.4-0.6s per session. The AI runs this on CPU rather than GPU. Total conversion time is ~45s for all 41 sessions.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
print(f"    dF/F computation: {time.time()-t1:.2f}s")
```

iii. The AI included timing information throughout the code and estimated full conversion time before running it.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame alignment loop in `align_motion_energy` iterates frame-by-frame to build a neural index mapping. This could be vectorized using cumulative sums.

ii.
```python
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
```

iii. The number of camera frames per session is ~36000-54000, so this loop is non-trivial but runs quickly in practice.

## 6-c. What processing does the code repeat multiple times?

i. The `split_into_trials` function is called separately for neural, input, and output data for each session, with essentially the same splitting logic duplicated for 1D and 2D arrays.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
# Then separate loops for input and output:
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end]
    ...
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end]
```

iii. The trial splitting uses the same start/end indices three times (once for neural, input, and output). These could be combined into a single loop.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No obviously unnecessary processing was identified. All processing steps contribute to the final output. However, the binning step (averaging 10 frames) discards temporal resolution that the downstream decoder could potentially use.

ii. N/A

iii. The AI followed the paper's processing pipeline faithfully. The 10-frame binning is described in the paper's methods for their decoding analysis.
