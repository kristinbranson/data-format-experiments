# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subjects in a `SUBJECTS` list, enumerates every subdirectory under each subject as a session, and loads per-session neural and behavioral arrays from `suite2p/plane0` and `move_deve`. Trial structure is not loaded from disk; it is created later from the continuous session recordings.

ii. 
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(subject_dir):
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d))])
    return sessions

def load_session_data(subject_dir, session_name):
    session_dir = os.path.join(subject_dir, session_name)
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    
    return F, Fneu, ops, me

all_sessions = []
for subject in SUBJECTS:
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = get_sessions(subject_dir)
    for s in sessions:
        all_sessions.append((subject, s))
```

iii. In `CONVERSION_NOTES.md`, the AI states that the dataset contains six subjects `jm031-jm046`, with session folders underneath each subject, and that the needed raw arrays are `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. The trajectory shows it treating the dataset as a fixed six-subject corpus rather than discovering subjects dynamically.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by a fixed, hard-coded mouse list. The saved `subjects` field preserves the order in which those hard-coded subject IDs are first encountered.

ii. 
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

subjects_seen = []

for sess_idx, (subject, session_name) in enumerate(all_sessions):
    if subject not in subjects_seen:
        subjects_seen.append(subject)
    subj_idx = subjects_seen.index(subject)
    ...
    subject_idx_list.append(subj_idx)

data = {
    ...
    'subjects': subjects_seen,
    'subject_idx': np.array(subject_idx_list, dtype=np.int64),
}
```

iii. The notes list exactly six mice and describe subject folder names as a direct mapping target. The trajectory repeatedly summarizes the dataset as “6 subjects” and plans around those known IDs.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory under a subject directory. Session directories are sorted alphabetically, then processed one by one.

ii. 
```python
def get_sessions(subject_dir):
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d))])
    return sessions

for subject in SUBJECTS:
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = get_sessions(subject_dir)
    for s in sessions:
        all_sessions.append((subject, s))
```

iii. In `CONVERSION_NOTES.md`, the AI records session counts per subject and treats each `{date}_a` folder as one daily recording session. The sorted traversal appears to be for deterministic ordering.

## 1-d. How are the data split into trials?

i. The AI treats each continuous recording session as a sequence of non-overlapping 60-second trials after temporal binning. At 30 Hz with 10-frame bins, that is `180` bins per trial. Any remainder shorter than a full trial is implicitly discarded.

ii. 
```python
TRIAL_DURATION = 60.0
TIME_PER_BIN = BIN_SIZE / FS
BINS_PER_TRIAL = int(TRIAL_DURATION / TIME_PER_BIN)  # 180

def split_into_trials(data, bins_per_trial):
    n_total_bins = data.shape[-1]
    n_trials = n_total_bins // bins_per_trial
    trials = []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        trials.append(data[..., start:end])
    return trials

neural_trials = split_into_trials(dFF_binned, BINS_PER_TRIAL)
```

iii. The notes say there is no native trial structure and that the decoder task requires 60-second trials. The trajectory explicitly computes that 60 seconds at 3 Hz gives 180 timepoints per trial and reports that this structure “works out cleanly.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply explicit trial-level quality filtering. All full-length 60-second segments are retained.

ii. 
```python
def split_into_trials(data, bins_per_trial):
    n_total_bins = data.shape[-1]
    n_trials = n_total_bins // bins_per_trial
    trials = []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        trials.append(data[..., start:end])
    return trials
```

iii. `CONVERSION_NOTES.md` says there is “No explicit trial curation mentioned” and that sessions are simply split into 60-second segments for this decoder task.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from `F.npy` and `Fneu.npy`. The AI also loads `ops.npy` to recover Suite2p preprocessing parameters such as `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, and `fs`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()

neucoeff = ops.get('neucoeff', NEUCOEFF)
baseline = ops.get('baseline', 'maximin')
win_baseline = ops.get('win_baseline', 60.0)
sig_baseline = ops.get('sig_baseline', 10.0)
fs = ops.get('fs', FS)
```

iii. The notes map `F.npy, Fneu.npy -> neural` and explain that Suite2p defaults are confirmed in `ops.npy`, so the file is used as metadata/configuration for preprocessing.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction, runs Suite2p baseline correction, and then temporally averages the resulting traces into non-overlapping 10-frame bins.

ii. 
```python
def compute_dff(F, Fneu, ops):
    neucoeff = ops.get('neucoeff', NEUCOEFF)
    baseline = ops.get('baseline', 'maximin')
    win_baseline = ops.get('win_baseline', 60.0)
    sig_baseline = ops.get('sig_baseline', 10.0)
    fs = ops.get('fs', FS)
    
    Fc = F - neucoeff * Fneu
    Fc = Fc.astype(np.float32)
    device = torch.device('cpu')
    dFF = s2p_preprocess(Fc, baseline, win_baseline, sig_baseline, fs, device=device)
    return dFF

dFF_binned = bin_data(dFF, BIN_SIZE)
```

iii. The notes say the intended pipeline is `Fc = F - 0.7*Fneu -> Suite2p preprocess (maximin) -> bin 10 frames`. The trajectory shows the AI correcting itself after inspecting Suite2p and deciding that the paper’s “dF/F” corresponds to Suite2p-style baseline-subtracted traces rather than division by baseline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed in the conversion script. The AI assumes the provided `F.npy` rows are already Track2p-matched and already pass the `iscell` threshold.

ii. 
```python
def load_session_data(subject_dir, session_name):
    ...
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    return F, Fneu, ops, me
```

iii. In the notes, the AI states that all ROIs have `iscell=1`, that the data have already been processed through Track2p, and that no extra neuron filtering is necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to session start, not to a task event. The recording is treated as continuous, and each trial is a contiguous slice from that running session timeline.

ii. 
```python
neural_trials = split_into_trials(dFF_binned, BINS_PER_TRIAL)

data = {
    ...
    'metadata': {
        ...
        'temporal_alignment_event': 'session_start',
        'off_start': 0.0,
        'off_end': None,
    }
}
```

iii. The notes and trajectory repeatedly describe the recordings as continuous spontaneous activity with no native task trials, so “session_start” is the temporal anchor the AI chose.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame non-overlapping bins at 30 Hz, so the time bin is `10 / 30 = 0.333...` seconds, or `333.33 ms`. This is the only temporal rebinning step.

ii. 
```python
BIN_SIZE = 10
FS = 30.0
TIME_PER_BIN = BIN_SIZE / FS  # seconds per bin

def bin_data(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    binned = truncated.reshape(new_shape).mean(axis=-1)
    return binned

'time_bin_size': TIME_PER_BIN * 1000,
```

iii. The notes quote the paper’s statement that decoding analyses averaged “10 consecutive timestamps” and explicitly record `333.33 ms` as the intended output resolution.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not taken from `tstamps.npy`. It is derived from the binned frame index, using the assumed constant sampling rate of 30 Hz.

ii. 
```python
FS = 30.0
TIME_PER_BIN = BIN_SIZE / FS
...
n_bins = dFF_binned.shape[1]
time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN
```

iii. The notes explicitly say that although `tstamps.npy` exists and appears to be in kiloseconds, “for our conversion, we use frame_index / fs for time in seconds instead.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a session-long binned time axis in seconds using the center of each 10-frame bin, then slices that vector into per-trial `1 x 180` arrays.

ii. 
```python
n_bins = dFF_binned.shape[1]
time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN  # seconds

for t in range(n_trials):
    start_bin = t * BINS_PER_TRIAL
    end_bin = start_bin + BINS_PER_TRIAL
    trial_time = time_axis[start_bin:end_bin]
    input_trials.append(trial_time.reshape(1, -1))
```

iii. The notes justify this as “Time input: Center of each bin (offset by 0.5 × time_per_bin),” and the trajectory later treats the resulting range `[0.2, 1799.8]` seconds as a validation check.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned one-to-one with the binned neural data: the AI builds the time axis from the same session-wide bin count used for neural traces and uses the same trial boundaries when slicing both.

ii. 
```python
dFF_binned = bin_data(dFF, BIN_SIZE)
...
n_bins = dFF_binned.shape[1]
time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN
neural_trials = split_into_trials(dFF_binned, BINS_PER_TRIAL)

for t in range(n_trials):
    start_bin = t * BINS_PER_TRIAL
    end_bin = start_bin + BINS_PER_TRIAL
    trial_time = time_axis[start_bin:end_bin]
    input_trials.append(trial_time.reshape(1, -1))
```

iii. The notes describe the time input as representing the midpoint of each temporal bin, and the trajectory treats the neural and input streams as sharing the same binned session timeline.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion-energy signal is derived from `move_deve/motion_energy_glob.npy`. The script does not use `interframe_int.npy` or `tstamps.npy`.

ii. 
```python
def load_session_data(subject_dir, session_name):
    ...
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    return F, Fneu, ops, me
```

iii. The notes treat `motion_energy_glob.npy` as the already-computed global motion-energy trace and focus on aligning that single array to the neural trace length.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI pads or truncates the raw motion-energy vector to match the neural frame count, averages it into 10-frame bins, and then discretizes the binned trace into five equal-percentile bins within each session.

ii. 
```python
def align_me_to_neural(me, n_neural_frames):
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    elif n_me < n_neural_frames:
        padded = np.zeros(n_neural_frames, dtype=np.float64)
        padded[:n_me] = me.astype(np.float64)
        padded[n_me:] = me[-1]
        return padded
    else:
        return me[:n_neural_frames].astype(np.float64)

me_aligned = align_me_to_neural(me, n_frames)
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE).flatten()
me_discrete, bin_edges = discretize_me(me_binned, N_ME_BINS)
```

iii. In the notes, the AI resolves motion-energy length mismatches by “Pad with last ME value to match F length,” arguing that this handles minor camera-trigger drops. The trajectory shows it considering both truncation and padding before settling on padding the motion-energy array.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes per-session percentile cutoffs at 0, 20, 40, 60, 80, and 100 percent and uses `np.digitize` to convert binned motion energy into categories `0` through `4`.

ii. 
```python
def discretize_me(me_binned, n_bins=N_ME_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
    me_discrete = np.clip(me_discrete, 0, n_bins - 1)
    return me_discrete, bin_edges
```

iii. The notes say “ME discretization: 5 equal-percentile bins per session,” and later validate that the resulting bin distribution is approximately uniform.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes camera and microscope streams are synchronized frame-for-frame. It forces equal lengths by padding shorter motion-energy arrays with their last value or truncating longer arrays, then bins neural and behavioral streams in the same way and slices them with the same trial boundaries.

ii. 
```python
me_aligned = align_me_to_neural(me, n_frames)

dFF_binned = bin_data(dFF, BIN_SIZE)
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE).flatten()

neural_trials = split_into_trials(dFF_binned, BINS_PER_TRIAL)
...
trial_me = me_discrete[start_bin:end_bin]
output_trials.append(trial_me.reshape(1, -1))
```

iii. The notes state that motion energy and neural data are synchronized 1:1 because the camera is microscope-triggered, and that padding is the chosen fix when the motion-energy trace is shorter than the neural trace.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling strategy is tolerant length correction for motion energy: short motion-energy traces are padded with their last value and long traces are truncated. Partial leftover bins or trials are dropped automatically by integer division. There is no explicit interpolation or assertion-based failure on mismatched motion-energy timing.

ii. 
```python
def align_me_to_neural(me, n_neural_frames):
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    elif n_me < n_neural_frames:
        padded = np.zeros(n_neural_frames, dtype=np.float64)
        padded[:n_me] = me.astype(np.float64)
        padded[n_me:] = me[-1]
        return padded
    else:
        return me[:n_neural_frames].astype(np.float64)

def bin_data(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    ...

def split_into_trials(data, bins_per_trial):
    n_total_bins = data.shape[-1]
    n_trials = n_total_bins // bins_per_trial
    ...
```

iii. `CONVERSION_NOTES.md` explicitly records the mismatch resolution as “Pad with last ME value to match F length,” and the trajectory frames this as a pragmatic fix for “minor camera trigger drops.”

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p preprocessing call that computes the baseline-corrected neural traces. The script explicitly times data loading, dF/F preprocessing, and binning, and presents dF/F as the dominant cost.

ii. 
```python
t0 = time.time()
F, Fneu, ops, me = load_session_data(subject_dir, session_name)
t_load = time.time() - t0

t1 = time.time()
dFF = compute_dff(F, Fneu, ops)
t_dff = time.time() - t1

t2 = time.time()
dFF_binned = bin_data(dFF, BIN_SIZE)
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE).flatten()
t_bin = time.time() - t2

print(f"  {subject}/{session_name}: ... load={t_load:.1f}s dff={t_dff:.1f}s "
      f"bin={t_bin:.2f}s total={t_total:.1f}s")
```

iii. The notes report per-session timings and specifically call out dF/F computation as the main computational expense, with binning described as fast vectorized work.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loops are the trial-splitting loop, the loop that builds per-trial `input` and `output` arrays, and some optional plotting loops. The actual binning step is already vectorized with `reshape(...).mean(...)`.

ii. 
```python
def split_into_trials(data, bins_per_trial):
    ...
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        trials.append(data[..., start:end])

for t in range(n_trials):
    start_bin = t * BINS_PER_TRIAL
    end_bin = start_bin + BINS_PER_TRIAL
    trial_time = time_axis[start_bin:end_bin]
    input_trials.append(trial_time.reshape(1, -1))
    trial_me = me_discrete[start_bin:end_bin]
    output_trials.append(trial_me.reshape(1, -1))
```

iii. The notes say binning was intentionally vectorized, but they do not discuss the remaining Python-level loops in detail. The code reflects a preference for simple explicit loops over more aggressive reshaping.

## 6-c. What processing does the code repeat multiple times?

i. The script repeats session-wise indexing and trial slicing work. Neural trials are created in one pass via `split_into_trials`, then time and motion-energy trials are re-sliced in a second loop using the same boundaries. Subject lookup via `subjects_seen.index(subject)` is also repeated for every session.

ii. 
```python
neural_trials = split_into_trials(dFF_binned, BINS_PER_TRIAL)
...
for t in range(n_trials):
    start_bin = t * BINS_PER_TRIAL
    end_bin = start_bin + BINS_PER_TRIAL
    trial_time = time_axis[start_bin:end_bin]
    input_trials.append(trial_time.reshape(1, -1))
    trial_me = me_discrete[start_bin:end_bin]
    output_trials.append(trial_me.reshape(1, -1))

if subject not in subjects_seen:
    subjects_seen.append(subject)
subj_idx = subjects_seen.index(subject)
```

iii. There is no explicit written justification for these repetitions. The surrounding notes emphasize readability and straightforward validation rather than minimizing repeated indexing operations.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does some extra work that is not needed for the saved dataset: it loads `ops.npy` to recover defaults that mostly mirror constants, returns `me_binned` and `bin_edges` from `process_session` even though the caller never uses them, and optionally generates diagnostic plots and z-scored visualizations that are not stored in `converted_data.pkl`.

ii. 
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
...
return neural_trials, input_trials, output_trials, n_neurons, me_binned, bin_edges

neural_trials, input_trials, output_trials, n_neurons, me_binned, bin_edges = \
    process_session(...)

def plot_processing(...):
    ...
    from scipy.stats import zscore
    dFF_z = zscore(dFF_binned, axis=1)
    ...
```

iii. The notes justify this extra work as part of “processing visualizations,” sanity checks, and reference matching, not as part of the minimal data-conversion path.
