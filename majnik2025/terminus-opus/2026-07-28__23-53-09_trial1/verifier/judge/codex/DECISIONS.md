# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates subject directories in `data/` whose names start with `jm`, enumerates all subdirectories inside each subject as sessions, then processes each session by loading calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy` plus behavioral motion energy from `move_deve/motion_energy_glob.npy` and camera timestamps from `move_deve/tstamps.npy`. Trials are not loaded from disk; they are created later by splitting the preprocessed continuous session data.

ii.
```python
def get_subjects_and_sessions(data_dir):
    """Get all subjects and their sessions."""
    subjects = sorted([d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    sessions = {}
    for subj in subjects:
        subj_path = os.path.join(data_dir, subj)
        sess_list = sorted([s for s in os.listdir(subj_path) 
                           if os.path.isdir(os.path.join(subj_path, s))])
        sessions[subj] = sess_list
    return subjects, sessions
...
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
...
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI documented the dataset as `data/{subject_id}/{date}_a/{suite2p,move_deve}/` and explicitly listed `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy` as the key loaded files. It justified this as matching the observed session directory structure and said later in Step 6 that each session should be processed from those continuous recordings before trialization.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted directory names in `data/` that begin with `jm`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. The AI’s notes say the dataset contains 6 mice and list them as `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, so it treated the `jm*` directory naming convention as the subject split.

## 1-c. How are the data split into sessions?

i. Sessions are the sorted subdirectories within each subject folder. The code flattens them into a `session_list` of `(subject, session_name)` pairs and processes each one as a separate session.

ii.
```python
for subj in subjects:
    subj_path = os.path.join(data_dir, subj)
    sess_list = sorted([s for s in os.listdir(subj_path) 
                       if os.path.isdir(os.path.join(subj_path, s))])
    sessions[subj] = sess_list
...
session_list = []  # (subject, session_name)
for subj in subjects:
    for sess in sessions[subj]:
        session_list.append((subj, sess))
```

iii. In the notes, the AI described the organization as daily recordings under each mouse and recorded session counts per subject. That is the stated reason for treating each dated subdirectory as one session.

## 1-d. How are the data split into trials?

i. The AI assumes there is no natural trial structure and creates artificial trials by first averaging every 10 frames, then splitting each session into consecutive 2-minute blocks. At 30 Hz and a bin size of 10, each trial contains 360 binned timepoints; 20-minute sessions become 10 trials and 30-minute sessions become 15 trials.

ii.
```python
BIN_SIZE = 10  # number of frames to average
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)  # 3600 raw frames per trial
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE  # 360 binned frames per trial
...
dff_binned = bin_data(dff, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
```

iii. The AI justified this in `CONVERSION_NOTES.md` Step 3 and Step 5 by citing the paper’s decoder description: “averaging using a bin size of 10 frames” and “splits were done on consecutive 2 minute blocks.” It explicitly chose “2-minute blocks matching paper CV structure.”

## 1-e. How are trials filtered based on quality controls?

i. The AI applies no explicit trial-quality exclusion rules. Trials are whatever full 2-minute blocks fit after binning; any leftover partial block at the end of a session is dropped implicitly by floor division.

ii.
```python
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

iii. The notes say “No explicit trial curation mentioned” and frame dropping is handled at the motion-energy alignment stage instead of by rejecting trials. No separate trial QC logic is described in the notes or trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The AI’s Step 2 notes list these as the raw fluorescence and neuropil fluorescence traces, and Step 5 maps `F.npy, Fneu.npy -> neural`.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil (`F - 0.7 * Fneu`), applies Suite2p’s `preprocess(..., baseline='maximin')` baseline correction on CPU, then averages the resulting continuous neural signal into 10-frame bins before storing it as trial data.

ii.
```python
def compute_dff(F, Fneu, neucoeff=NEUCOEFF, win_baseline=WIN_BASELINE, 
                sig_baseline=SIG_BASELINE, fs=FS):
    """Compute dF/F using Suite2p default method.
    
    Fc = F - neucoeff * Fneu, then maximin baseline subtraction.
    """
    Fc = F - neucoeff * Fneu
    device = torch.device('cpu')
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
    return dff
...
dff = compute_dff(F, Fneu)
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The notes justify the preprocessing as “Suite2p dF/F computation: `Fc = F - 0.7*Fneu`, then baseline subtraction ... using maximin filter,” and Step 6 says the created pipeline should “compute dF/F” and then “bin both neural and behavioral data by averaging 10 consecutive frames.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed in the code. All ROIs in `F.npy` are used.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_neural_frames = F.shape
...
brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The AI’s notes say `iscell.npy` contains all 1s and that Track2p output already contains only tracked cells across days, so it decided no additional `iscell` or quality-threshold filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `session_start`. After preprocessing and binning, the code cuts each session into consecutive blocks starting from the beginning of the session; there is no other event alignment.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
...
'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': 0.0,
    'off_end': None,
```

iii. The notes say the decoder input should be “Time in seconds from session start” and the trial structure should be consecutive 2-minute blocks. That is the stated rationale for session-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral data by averaging every 10 original frames, so the converted data have a temporal resolution of `10 / 30 = 0.333...` seconds, i.e. `333.33 ms`.

ii.
```python
BIN_SIZE = 10  # number of frames to average
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # 333.33 ms
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. In Step 3 and Step 5 of the notes, the AI cites the paper’s decoder description as requiring 10-frame averaging and explicitly records “Time bin size: 10 frames at 30Hz = 333.33 ms.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a stored time variable. The AI synthesizes it from array indices together with `FS` and `BIN_SIZE`.

ii.
```python
n_binned = dff_binned.shape[1]
...
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The notes’ variable-mapping table says `Time index -> input[0]`, and Step 6 says the script should “Compute time input (seconds from session start).”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one timestamp per 10-frame bin using the center of each bin, then converts the values to `float32` and reshapes them to `(1, n_timepoints)` per trial.

ii.
```python
# Create time input (time elapsed from session start in seconds)
# Each binned timepoint represents the center of the bin
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
...
trial_time = time_input[start:end].astype(np.float32)
input_trials_list.append(trial_time.reshape(1, -1))  # (1, n_timepoints)
```

iii. The notes say the input should be “Time in seconds from session start, binned by 10,” and the trajectory explicitly comments that the task requires “absolute time from session start,” not a reset within each trial.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time to neural data by generating timestamps on the same binned grid as the neural data and slicing each trial with the same `start:end` indices used for the trial structure.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
...
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
```

iii. In Step 10, the AI says it verified the time values against the converted data exactly, and the code comments emphasize that the values represent absolute session time on the same binned timeline as neural activity.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy from `move_deve/motion_energy_glob.npy` and uses `move_deve/tstamps.npy` to identify missing camera frames. It does not use `interframe_int.npy` in the final code.

ii.
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
...
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
```

iii. The notes first catalogued both `tstamps.npy` and `interframe_int.npy`, then Step 5 chose to “Use tstamps to identify which frames are missing and interpolate.” Later, Step 10 says the alignment function was changed from an earlier timestamp mapping to an IFI-based method computed from `tstamps`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code aligns the motion-energy vector to neural frame count by inferring dropped frames from inter-frame intervals computed from `tstamps`, fills missing positions by linear interpolation, averages 10 frames at a time, and then discretizes the binned signal. Despite the notes claiming per-session normalization before discretization, the code does not actually normalize the motion-energy values.

ii.
```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
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
...
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
me_binned = bin_data(me_aligned, BIN_SIZE)
me_discrete = discretize_motion_energy(me_binned, N_BINS)
```

iii. Step 5 of the notes says the intended pipeline was “Align to neural, bin by 10, normalize, discretize to 5 bins,” and the trajectory says the alignment was revised to an “IFI-based alignment” because the previous timestamp mapping was wrong. The same notes also claim “ME normalization: Per-session normalization before discretization,” which is not present in the code.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds motion energy by computing the 20th, 40th, 60th, and 80th percentiles within each session’s binned motion-energy trace and using `np.digitize` to assign labels `0` through `4`.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    """Discretize motion energy into equal-percentile bins.
    
    Computes percentile bin edges per session, assigns each timepoint to a bin.
    Returns integer bin labels (0 to n_bins-1).
    """
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The Step 5 notes explicitly justify this as “Bins computed per session to account for different motion levels across days/mice.”

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns output to neural data by mapping the camera motion-energy samples onto neural-frame indices, interpolating missing camera frames, then applying the same 10-frame binning and the same per-trial slicing used for neural data.

ii.
```python
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
...
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
    output_trials_list.append(trial_me.reshape(1, -1))
```

iii. The notes say motion energy should be interpolated “to match F length” before binning, and Step 10 says the final alignment uses “tstamps interpolation” after fixing an earlier timestamp-mapping bug with an IFI-based method.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mainly handles shorter-than-expected motion-energy arrays. It infers which camera frames are missing from timestamp gaps, interpolates missing values onto a full neural-length vector, and clips mapped indices to valid neural-frame bounds. Partial bins and partial trials at session ends are simply dropped by integer division. There are no explicit asserts that post-alignment lengths exactly match or that missing-data cases beyond dropped camera frames are absent.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
...
neural_idx = np.clip(neural_idx, 0, n_neural_frames - 1)
me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
...
if not valid.all():
    indices = np.arange(n_neural_frames)
    me_full = np.interp(indices, indices[valid], me_full[valid])
...
n_bins = n_frames // bin_size
data_trimmed = data[:n_bins * bin_size]
...
n_trials = n_timepoints // trial_length
```

iii. The Step 5 notes say “When ME has fewer frames than F, interpolate ME to match F length,” and Step 10 says the agent fixed the alignment logic after detecting a bug in its earlier timestamp-based mapping. No broader missing-data policy is documented.

## 6-a. What are the most time-consuming steps of the code?

i. The AI treats calcium preprocessing via `compute_dff`/Suite2p `preprocess` as the main bottleneck; binning is described as cheap by comparison.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
print(f"    dF/F computation: {time.time()-t1:.2f}s")
...
t2 = time.time()
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
print(f"    Binning: {n_neural_frames} -> {n_binned} timepoints ({time.time()-t2:.2f}s)")
```

iii. In Step 7 of the notes, the AI reports `dF/F` taking `0.4-0.6s` per session and binning `0.02s`, which is its explicit justification for where the runtime goes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code has several obvious Python loops that could be vectorized: the frame-by-frame loop in `align_motion_energy`, the per-trial loops in `split_into_trials`, and the separate loops that rebuild input and output trials after the neural trials have already established the trial boundaries.

ii.
```python
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
...
for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))
...
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
...
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
    output_trials_list.append(trial_me.reshape(1, -1))
```

iii. The AI’s notes emphasize efficiency and runtime tracking, but these loops were kept for implementation simplicity rather than fully vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats trial-boundary calculations multiple times. After it splits neural data into trials, it separately recomputes the same `start:end` windows to create time-input trials and motion-energy-output trials. It also implements separate near-duplicate trial-splitting branches for 1D and 2D arrays.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
...
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
...
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
    output_trials_list.append(trial_me.reshape(1, -1))
```

iii. There is no explicit note justifying this repetition beyond straightforward implementation. The trajectory focuses more on correctness and debugging of alignment than on eliminating repeated slicing logic.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is the optional `--show-processing` path, which generates multi-panel diagnostic plots for up to two sessions. Those figures are only for human inspection and are not used downstream. Outside that path, there is little obviously wasted mandatory computation besides preparing plotting-only intermediates and printing extensive summaries.

ii.
```python
if show_processing and sess_idx < 2:
    plot_processing(subj, sess_name, dff, me_aligned, dff_binned, me_binned, 
                   me_discrete, time_input, neural_trials, output_trials_list,
                   sess_idx)
...
fig, axes = plt.subplots(5, 1, figsize=(16, 20))
...
plt.savefig(fname, dpi=100)
```

iii. The instructions explicitly asked for plots that “visually convince the user” the processing is correct, and the AI’s Step 7 notes say “Plots generated for both sessions. No anomalies observed.” That is the stated reason for doing this otherwise downstream-irrelevant work.
