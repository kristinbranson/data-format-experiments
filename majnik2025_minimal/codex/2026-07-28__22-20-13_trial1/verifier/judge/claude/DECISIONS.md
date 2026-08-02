# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories starting with `jm` in the data directory. Sessions are all subdirectories within each subject folder. For each session, calcium data is loaded from suite2p outputs (`F.npy`, `Fneu.npy`) in `suite2p/plane0/`, and motion energy from `motion_energy_glob.npy` and `interframe_int.npy` in `move_deve/`. Data is loaded in a two-pass approach: first all sessions are preprocessed, then motion energy is globally discretized, and finally trials are assembled.

ii.
```python
def get_subjects(base_path):
    return sorted(
        d.name for d in os.scandir(base_path)
        if d.is_dir() and d.name.startswith('jm')
    )

def get_sessions(base_path, subject):
    subject_dir = os.path.join(base_path, subject)
    sessions = [d.path for d in os.scandir(subject_dir) if d.is_dir()]
    sessions.sort()
    return sessions

# Loading per session:
F = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(session_path, 'move_deve', 'motion_energy_glob.npy'))
dt = np.load(os.path.join(session_path, 'move_deve', 'interframe_int.npy'))
```

iii. The directory structure follows the standard convention from the Track2p release: subject folders contain session subfolders, each with suite2p output and motion energy files. The two-pass approach allows global percentile computation for motion energy discretization across all sessions before trial assembly.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data directory, sorted alphabetically. Each directory represents one mouse.

ii.
```python
subjects = get_subjects(base_path)
# returns sorted list of directory names matching 'jm*'
```

iii. The naming convention `jm*` is consistent across the dataset (jm031, jm032, jm038, jm039, jm040, jm046).

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. All subdirectories are included as sessions (no filtering by name pattern).

ii.
```python
def get_sessions(base_path, subject):
    subject_dir = os.path.join(base_path, subject)
    sessions = [d.path for d in os.scandir(subject_dir) if d.is_dir()]
    sessions.sort()
    return sessions
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session. Sorting ensures deterministic ordering.

## 1-d. How are the data split into trials?

i. There is no natural trial structure in this dataset. Trials are defined as 60-second non-overlapping segments of the continuous recording (60s x 30 Hz = 1800 frames per trial). Remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DUR = 60  # trial duration in seconds
FS = 30  # Hz
trial_frames = TRIAL_DUR * FS  # 1800
n_trials = n_frames // trial_frames
remainder = n_frames - n_trials * trial_frames

for ti in range(n_trials):
    s = ti * trial_frames
    e = s + trial_frames
    neural_trials.append(Fc[:, s:e])
```

iii. The CONVERSION_NOTES.md states "consecutive 2-minute blocks" matching the paper's decoding split strategy, but the code implements 60-second (1-minute) blocks. This is a discrepancy between the code and its documentation.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All trials from all sessions are included. The only filtering is discarding remainder frames that don't fill a complete trial at the end of each session.

ii.
```python
if remainder > 0:
    print(f'  session {idx}: discarding last {remainder} frames '
          f'({remainder/FS:.1f}s) that do not fill a full trial')
```

iii. No justification is given for lack of trial filtering. The recording is continuous spontaneous activity without stimulus-driven events that would warrant trial rejection.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from two suite2p output files: `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence traces), both from `plane0`.

ii.
```python
F = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces. The `ops.npy` file (which contains processing parameters) is NOT loaded in the code, though the CONVERSION_NOTES.md claims parameters are read from it.

## 2-b. How is the `neural` data processed?

i. Two processing steps are applied: (1) neuropil subtraction with a hardcoded coefficient of 0.7 (`Fc = F - 0.7 * Fneu`), and (2) suite2p's `dcnv.preprocess` function with hardcoded parameters for baseline estimation using the `maximin` method. Notably, no dF/F computation is performed -- the output of `dcnv.preprocess` is used directly as the neural signal.

ii.
```python
NEUCOEFF = 0.7
Fc = F - NEUCOEFF * Fneu
Fc = dcnv.preprocess(
    F=Fc,
    baseline='maximin',
    win_baseline=60.0,
    sig_baseline=10,
    fs=FS,
    prctile_baseline=8.0,
    batch_size=BATCH_SIZE,
    device=DEVICE,
)
```

iii. The CONVERSION_NOTES.md describes computing dF/F by recovering the baseline and dividing (`(Fc - F0) / F0`), but the code does not implement this step. The code only applies the `dcnv.preprocess` baseline correction. Parameters are hardcoded rather than read from `ops.npy` as claimed in the notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All neurons present in the suite2p `F.npy` output are included.

ii. N/A (no filtering code present)

iii. Suite2p's cell detection pipeline already identifies ROIs. The `iscell` classifier output is not used for additional filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 60-second block within the session. Since trials are contiguous segments of the continuous recording starting from frame 0, alignment is implicit via array indexing.

ii.
```python
for ti in range(n_trials):
    s = ti * trial_frames
    e = s + trial_frames
    neural_trials.append(Fc[:, s:e])

# metadata
'temporal_alignment_event': 'session_start',
'off_start': None,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native suite2p frame rate of 30 Hz (time bin size of ~33.33 ms). No temporal rebinning or averaging is applied.

ii.
```python
FS = 30  # Hz
'metadata': {
    'time_bin_size': 1.0 / FS * 1000,  # ~33.33 ms
}
```

iii. The CONVERSION_NOTES.md describes 10-frame averaging (producing 360 timepoints per 2-minute trial), but the code does not implement any binning. Each trial has 1800 frames at native 30 Hz resolution.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed synthetically from frame indices and the known frame rate.

ii.
```python
t = ((s + np.arange(trial_frames)) / FS).astype(np.float32)
inp_trials.append(t[np.newaxis, :])  # (1, trial_frames)
```

iii. Since the frame rate is constant at 30 Hz, computing time from frame indices is equivalent to having explicit timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as the absolute time from session start in seconds. For each trial, the starting frame offset `s = ti * trial_frames` is added to the within-trial frame index, then divided by the frame rate. This gives a continuous, monotonically increasing time across trials.

ii.
```python
t = ((s + np.arange(trial_frames)) / FS).astype(np.float32)
# s = ti * trial_frames, so time continues across trials
```

iii. This provides the decoder with information about the absolute position within the recording session, which could capture slow drifts or non-stationarities.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because both are computed from the same frame indices. Each time point corresponds one-to-one with a neural data time point.

ii.
```python
# Same loop, same indices:
neural_trials.append(Fc[:, s:e])
inp_trials.append(t[np.newaxis, :])
```

iii. No explicit alignment is needed since time is synthesized from the same frame structure as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from two files per session: `motion_energy_glob.npy` (pre-computed global motion energy from behavioral video) and `interframe_int.npy` (inter-frame intervals used to detect dropped video frames).

ii.
```python
me = np.load(os.path.join(session_path, 'move_deve', 'motion_energy_glob.npy'))
dt = np.load(os.path.join(session_path, 'move_deve', 'interframe_int.npy'))
```

iii. The motion energy file contains a pre-computed signal from the behavioral camera. The interframe interval file is needed to identify and repair dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) Dropped video frames are detected using a threshold on interframe intervals (`dt * 1000 > 0.04`) and repaired by inserting interpolated values (average of neighbors). (2) Motion energy is normalized per-session by dividing by its standard deviation. (3) The continuous signal is discretized into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
# Dropped frame repair
if me.shape[0] < expected_len:
    drop_indices = np.where(dt * 1000 > 0.04)[0]
    for offset, idx in enumerate(drop_indices):
        insert_pos = idx + 1 + offset
        interp_val = (me[insert_pos - 1] + me[insert_pos]) / 2.0
        me = np.insert(me, insert_pos, interp_val)

# Per-session std normalization
me = me / me.std()

# Global percentile binning
concatenated = np.concatenate(all_me_flat)
percentiles = np.linspace(0, 100, n_levels + 1)
bin_edges = np.percentile(concatenated, percentiles)
output = np.digitize(me, bin_edges[1:-1])
```

iii. The dropped frame interpolation ensures the motion energy signal matches the neural data length. Per-session std normalization is applied before pooling for global percentile computation. The 5 equal-percentile bins produce approximately balanced class counts.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 quintile bins using global percentiles. The bin edges are computed from all sessions' motion energy values concatenated together, using evenly-spaced percentiles (0, 20, 40, 60, 80, 100). `np.digitize` maps values to bins 0-4.

ii.
```python
concatenated = np.concatenate(all_me_flat)
percentiles = np.linspace(0, 100, n_levels + 1)  # [0, 20, 40, 60, 80, 100]
bin_edges = np.percentile(concatenated, percentiles)
output = np.digitize(me, bin_edges[1:-1])  # bins [0, n_levels-1]
```

iii. Global percentile-based discretization ensures balanced class counts across the full dataset. The approach uses all data to determine bin boundaries, then assigns each time point to one of 5 levels.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. Dropped video frames make the motion energy array shorter than the neural data. These are detected via interframe interval thresholds and filled by inserting interpolated values. After repair, an assertion verifies length equality. Both signals are then sliced using the same frame indices for trial assembly.

ii.
```python
me = preprocess_motion_energy(session_path, expected_len=Fc.shape[1])
assert me.shape[0] == expected_len

# Same indices for both:
neural_trials.append(Fc[:, s:e])
output_trials.append(out[np.newaxis, s:e])
```

iii. The dropped frame repair ensures frame-for-frame alignment between neural and behavioral signals. The assertion provides a safety check against misalignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of data irregularities are handled: (1) Dropped video frames are detected and interpolated to match neural data length. (2) Remainder frames at the end of sessions that don't fill a complete trial are discarded. An assertion verifies motion energy length matches neural data length after interpolation.

ii.
```python
# Dropped frame detection and repair
if me.shape[0] < expected_len:
    drop_indices = np.where(dt * 1000 > 0.04)[0]
    for offset, idx in enumerate(drop_indices):
        insert_pos = idx + 1 + offset
        interp_val = (me[insert_pos - 1] + me[insert_pos]) / 2.0
        me = np.insert(me, insert_pos, interp_val)

assert me.shape[0] == expected_len

# Remainder frame discarding
if remainder > 0:
    print(f'  session {idx}: discarding last {remainder} frames ...')
```

iii. The assertion catches frame count mismatches. Discarding remainder frames causes minor data loss (at most 59 seconds per session with 60s trials).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the `dcnv.preprocess` baseline correction, which runs on GPU (`DEVICE = torch.device('cuda')`). It processes all neurons for each session using sliding window operations over the full recording length. Loading `.npy` files is I/O bound but relatively fast.

ii.
```python
DEVICE = torch.device('cuda')
Fc = dcnv.preprocess(
    F=Fc, baseline='maximin', win_baseline=60.0,
    sig_baseline=10, fs=FS, prctile_baseline=8.0,
    batch_size=BATCH_SIZE, device=DEVICE,
)
```

iii. GPU acceleration mitigates the computational cost, but this step still dominates runtime since it processes every neuron in every session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation loop inserts one frame at a time using `np.insert`, which reallocates the array each iteration. This could be vectorized by pre-allocating the output array and filling in all interpolated values at once.

ii.
```python
for offset, idx in enumerate(drop_indices):
    insert_pos = idx + 1 + offset
    interp_val = (me[insert_pos - 1] + me[insert_pos]) / 2.0
    me = np.insert(me, insert_pos, interp_val)
```

iii. The number of dropped frames is typically small (0-148 per session), so the performance impact is negligible in practice.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. The code uses a clean two-pass structure: first pass preprocesses all sessions, second pass assembles trials. Motion energy discretization is computed once globally.

ii. N/A

iii. The two-pass architecture avoids redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Per-session standard deviation normalization of motion energy is applied before global percentile binning. Since percentile-based discretization is rank-preserving, the per-session normalization changes relative scales between sessions but the final bin assignments depend on the global distribution of the (already-normalized) values. Whether this normalization is beneficial or harmful depends on whether session-level scale differences are meaningful.

ii.
```python
me = me / me.std()  # per-session normalization before global binning
```

iii. The normalization step changes the global distribution used for percentile computation but does not add information for the rank-based binning within each session individually.
