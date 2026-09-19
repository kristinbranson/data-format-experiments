# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all subjects by scanning `data/` for directories whose names start with `jm`, then loads all session subdirectories under each subject. For each session it loads calcium data from `suite2p/plane0/F.npy` and `Fneu.npy`, and behavioral data from `move_deve/motion_energy_glob.npy` plus `move_deve/tstamps.npy`. Trials are not loaded directly; they are created later by splitting the processed session-long arrays.

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

F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset is organized as `data/{subject_id}/{date}_a/{suite2p,move_deve}/` and identifies these files as the relevant raw sources. The trajectory also shows the agent concluded that subject folders are `jm*` directories and session folders are their subdirectories.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory name: every directory under `data/` with prefix `jm` is treated as one mouse, sorted alphabetically.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. The notes explicitly state there are 6 mice and list the `jm031` to `jm046` subject IDs. The trajectory shows the agent relied on the `jm*` naming convention as the subject split.

## 1-c. How are the data split into sessions?

i. Sessions are split by subject subdirectory: every directory immediately under a subject folder is treated as one session, sorted alphabetically.

ii.
```python
for subj in subjects:
    subj_path = os.path.join(data_dir, subj)
    sess_list = sorted([s for s in os.listdir(subj_path) 
                       if os.path.isdir(os.path.join(subj_path, s))])
    sessions[subj] = sess_list
```

iii. `CONVERSION_NOTES.md` describes the organization as one folder per date/session under each mouse. The agent also counted sessions per mouse from this directory structure during exploration.

## 1-d. How are the data split into trials?

i. The agent does not use any natural trial markers. It splits each continuous session into fixed, non-overlapping 2-minute blocks after 10-frame temporal binning. This yields 360 binned timepoints per trial, and any leftover partial block at the end is dropped implicitly.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE

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

neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)
```

iii. The notes justify this by citing the paper’s decoder cross-validation setup: “Split each session into 2-minute blocks (as used for CV in paper).” The trajectory explicitly shows the agent deciding to use 2-minute blocks because the recordings are continuous and the paper mentions “2 minute blocks.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filtering. The only effective filtering is that incomplete trailing data that does not fill a full fixed-length block is dropped by integer floor division when splitting into trials.

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

iii. In the notes the agent says “No explicit trial curation mentioned.” No stronger quality-control rule for trials appears in the code or trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from Suite2p fluorescence traces `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The notes identify these as the raw fluorescence and neuropil traces and state that neural output should be based on Suite2p’s default correction pipeline.

## 2-b. How is the `neural` data processed?

i. The agent applies neuropil subtraction `F - 0.7 * Fneu`, then Suite2p `preprocess` with `maximin` baseline subtraction, then averages non-overlapping 10-frame bins. The resulting session-long binned trace is later split into trials.

ii.
```python
Fc = F - neucoeff * Fneu
device = torch.device('cpu')
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)

dff_binned = bin_data(dff, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` says the paper uses Suite2p default processing: `Fc = F - 0.7*Fneu`, then `maximin` baseline subtraction, and says decoding uses 10-frame averaging for denoising. The trajectory repeats this as the intended neural pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural filtering is applied in the conversion script. The agent keeps all rows in `F.npy` for each session and assumes earlier processing already restricted these to valid tracked cells.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_neural_frames = F.shape
```

iii. The notes justify this by saying all `iscell` values are 1 and the Track2p output already contains only neurons tracked across days, so no extra `iscell` or session-specific filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to session start, not to any stimulus or behavioral event. Trials are contiguous chunks cut from the session-long neural time series.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

iii. The notes and trajectory both frame the recordings as continuous spontaneous behavior with no natural event structure, so the agent chose session start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame temporal bins at 30 Hz, so each bin is 333.33 ms. The agent rebins both neural activity and motion energy by averaging non-overlapping groups of 10 frames.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # 333.33 ms

def bin_data(data, bin_size=BIN_SIZE):
    ...
    return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The notes cite the methods text saying decoding analysis “slightly denoised” dF/F and behavior by averaging 10 consecutive timestamps, and the agent uses that as the reason for 10-frame binning.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw timestamp file. It is generated from the binned sample index using the imaging rate and bin size, so it is derived from the position within the processed session rather than a stored raw variable.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The trajectory explicitly says the input should be “time elapsed from beginning of experiment” and that, because the session is sampled regularly at 30 Hz, time can be computed directly from frame/bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The agent computes one time value per binned sample as the center of each 10-frame bin in seconds, then slices that session-long vector into per-trial arrays and reshapes each to `(1, n_timepoints)`.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
```

iii. In the notes the agent says the decoder input should be “Time in seconds from session start, binned by 10.” The inline code comment says it intentionally uses absolute session time rather than resetting time within each trial.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction to the same 10-frame bins as the binned neural data, then both are cut into trials using the same start and end indices. The agent uses absolute session time within each trial rather than trial-relative time.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE)
...
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
```

iii. The code comment says: “Actually, the task says ‘time elapsed from beginning of experiment’ / So we use absolute time from session start.” The trajectory uses the same reasoning.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy output is derived primarily from `move_deve/motion_energy_glob.npy`. The agent also uses `move_deve/tstamps.npy` to infer dropped camera frames and align the motion trace to the neural frame count.

ii.
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The notes identify `motion_energy_glob.npy` as the motion signal and discuss missing camera frames. The trajectory shows the agent initially investigating both `tstamps.npy` and `interframe_int.npy`, then settling on timestamp-gap-based alignment in the final code.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent aligns motion energy to the neural frame count by inferring dropped frames from timestamp gaps, fills missing neural-frame positions by linear interpolation, averages into 10-frame bins, and then discretizes the binned signal into five within-session percentile bins.

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
    me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_full[neural_idx] = me.astype(np.float64)
    valid = ~np.isnan(me_full)
    if not valid.all():
        indices = np.arange(n_neural_frames)
        me_full = np.interp(indices, indices[valid], me_full[valid])
    return me_full

me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
me_binned = bin_data(me_aligned, BIN_SIZE)
me_discrete = discretize_motion_energy(me_binned, N_BINS)
```

iii. The notes say missing frames should be interpolated to match the neural frame count and that motion energy should be binned by 10 and discretized per session. The trajectory shows the agent revising its alignment logic after discovering problems with an earlier timestamp mapping.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent computes per-session percentile thresholds at 20, 40, 60, and 80 percent, then applies `np.digitize` to map each binned motion-energy value into one of five categories labeled `0` to `4`.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The notes justify this as “5 equal-percentile bins per session” to account for different motion levels across days and mice. The trajectory repeats that rationale.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If motion energy already has the same number of samples as the neural recording, the agent leaves it unchanged. Otherwise it infers the number of dropped camera frames from gaps in `tstamps`, maps each observed motion-energy sample onto a neural-frame index, creates a full-length array with missing entries as `NaN`, and linearly interpolates those missing positions before binning.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)

ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
neural_idx = np.zeros(len(me), dtype=int)
neural_idx[0] = 0
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped

me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
valid = ~np.isnan(me_full)
if not valid.all():
    indices = np.arange(n_neural_frames)
    me_full = np.interp(indices, indices[valid], me_full[valid])
```

iii. The trajectory shows the agent spending significant effort on dropped-frame alignment and explicitly states it switched to an “IFI-based mapping” after deciding its earlier timestamp logic was wrong. `CONVERSION_NOTES.md` says this change improved decoder validation accuracy.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling logic is for missing camera frames: the agent reconstructs a full-length motion-energy trace by identifying dropped frames from timestamp gaps and interpolating missing values. For trialing, it silently discards trailing partial blocks that do not fit a complete fixed-length trial.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
valid = ~np.isnan(me_full)
if not valid.all():
    indices = np.arange(n_neural_frames)
    me_full = np.interp(indices, indices[valid], me_full[valid])

n_trials = n_timepoints // trial_length
for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
```

iii. The notes repeatedly discuss missing camera frames as the main data irregularity and justify interpolation as necessary to align behavior to neural data. No other explicit missing-data policy is documented.

## 6-a. What are the most time-consuming steps of the code?

i. The agent identifies Suite2p dF/F computation / baseline subtraction as the dominant cost. Binning and bookkeeping are treated as minor by comparison.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
print(f"    dF/F computation: {time.time()-t1:.2f}s")

t2 = time.time()
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
print(f"    Binning: {n_neural_frames} -> {n_binned} timepoints ({time.time()-t2:.2f}s)")
```

iii. `CONVERSION_NOTES.md` includes runtime tables and reports dF/F as roughly `0.4-0.6s` per session versus about `0.02s` for binning. The trajectory also focuses on dF/F computation as the expensive step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the Python loop used to build the motion-energy frame mapping from timestamp gaps, the trial-splitting loops in `split_into_trials`, and the repeated per-trial loops used to construct input and output lists.

ii.
```python
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped

for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
    output_trials_list.append(trial_me.reshape(1, -1))
```

iii. The agent does not explicitly justify these loops in the notes. The trajectory emphasizes correctness and debugging of alignment rather than optimization, so these loops appear to be straightforward implementation choices rather than considered performance decisions.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats trial slicing multiple times over the same session-long indices: once in `split_into_trials` for neural data, again when slicing time input into per-trial arrays, and again when slicing discretized motion energy into per-trial arrays. It also maintains separate loops for building subject/session lists and for processing all sessions.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_FRAMES_BINNED)

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))

for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_me = me_discrete[start:end].astype(np.int64)
    output_trials_list.append(trial_me.reshape(1, -1))
```

iii. No explicit justification for this repetition appears in the notes or trajectory. It looks incidental to the implementation style rather than a documented design choice.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints diagnostic timing and motion-energy bin-count summaries that are not used downstream, and it includes optional plotting code for visual checks. These are useful for debugging but not required for the final converted dataset.

ii.
```python
t1 = time.time()
...
print(f"    dF/F computation: {time.time()-t1:.2f}s")

bin_counts = np.bincount(me_discrete, minlength=N_BINS)
print(f"    ME bin distribution: {bin_counts} (total: {bin_counts.sum()})")

if show_processing and sess_idx < 2:
    plot_processing(...)
```

iii. The notes emphasize sanity checks, visual verification, and runtime measurement throughout, so these extra computations were justified by the agent as validation aids rather than as part of the final data representation.
