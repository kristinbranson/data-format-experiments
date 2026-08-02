# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes the seven subject IDs in `ANIMALS`, then loads each subject's full `.mat` file with `mat73.loadmat`. It reads the `trace`, `position`, and `envs` arrays from the loaded MATLAB structure and then iterates through sessions and 1-minute trial slices in Python.

ii.
```python
from mat73 import loadmat

ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
    n_sessions = len(dat['trace'])
    for sess_idx in range(n_sessions):
        trace = np.array(dat['trace'][sess_idx])
        position = np.array(dat['position'][sess_idx])
        env_name = dat['envs'][sess_idx][0]
```

iii. `CONVERSION_NOTES.md` says the data were loaded from MATLAB files using `mat73.loadmat`, and claims this matches the reference code's `load_dat` function. The trajectory also states that the AI wanted to follow the reference repository's `load_dat` path rather than use HDF5 reference traversal directly.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the hardcoded `ANIMALS` list. The output `subjects` field is set directly to that list, and `subject_idx` is built from the index of each hardcoded animal during the outer loop.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal_idx, animal in enumerate(ANIMALS):
    ...
    subject_idx_list.append(animal_idx)

data = {
    ...
    'subjects': ANIMALS,
    'subject_idx': np.array(subject_idx_list, dtype=int),
}
```

iii. `CONVERSION_NOTES.md` explicitly lists the seven animal names and presents them as the full subject set. No separate justification beyond matching the published dataset is given.

## 1-c. How are the data split into sessions?

i. The AI treats each element of `dat['trace']` as one recording session. It loops over `range(len(dat['trace']))` and appends one session entry to the converted output for each index.

ii.
```python
n_sessions = len(dat['trace'])

for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])
    position = np.array(dat['position'][sess_idx])
    env_name = dat['envs'][sess_idx][0]
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The notes say each recording session corresponds to one environment per day and that sessions are 40 minutes long. The trajectory summary likewise says "Each recording session (40 min) is one session."

## 1-d. How are the data split into trials?

i. Sessions are split into non-overlapping 1-minute trials at 30 Hz, so each trial is 1800 frames. The number of full trials is `n_timepoints // 1800`, and any trailing remainder is dropped.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S

n_full_trials = n_timepoints // FRAMES_PER_TRIAL

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says sessions are split into 1-minute trials "as specified in the decoder task" and reports the resulting 39-40 trials per session, with leftover frames unused.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Instead, the AI skips whole sessions if they have zero valid neurons or fewer than two complete 1-minute trials. Incomplete trailing frames are discarded implicitly by `n_timepoints // FRAMES_PER_TRIAL`.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()

if n_valid == 0:
    continue

n_full_trials = n_timepoints // FRAMES_PER_TRIAL

if n_full_trials < 2:
    continue
```

iii. The trajectory says the AI was trying to satisfy the decoder requirement that each session contain at least two trials. The notes otherwise say "No Filtering Applied" beyond excluding invalid neurons.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from the raw `trace` variable for each session.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. `CONVERSION_NOTES.md` says the neural data are the paper's binary rising-phase calcium transients, and the trajectory summary describes the neural stream as "binary trace data (rising phase transients) at 30 Hz."

## 2-b. How is the `neural` data processed?

i. The AI assumes the `trace` arrays already contain binarized rising-phase activity, then keeps only valid neurons, replaces any remaining `NaN` values with zero, and casts to `float32`. It does not do additional deconvolution or temporal processing.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The notes justify this by quoting the paper's preprocessing description and stating that the stored data are already binary rising-phase transients. The trajectory says the AI concluded the trace data were already binary and that `float32` would reduce file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are retained if they are not all-`NaN` within a session. All-`NaN` neurons are dropped. If a retained neuron still contains isolated `NaN` values, those values are replaced with zero.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The notes say `NaN` neurons reflect CellReg-tracked cells that were not detected in that session, so those neurons are excluded. They also state that any remaining `NaN` values in valid neurons are set to zero.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the start of the recording session as the temporal alignment event. Trials are contiguous 1-minute windows beginning at frame 0 of the session rather than being aligned to an external behavioral event.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]

...

'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
}
```

iii. The main justification appears in the metadata and trajectory summary rather than in the notes: the trajectory frames the recordings as continuous 40-minute sessions split into 1-minute trials, and the code encodes that choice explicitly as "Start of recording session."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native 30 Hz sampling rate, corresponding to 33.33 ms time bins. No temporal rebinning or downsampling is applied.

ii.
```python
FPS = 30  # frames per second
...
'metadata': {
    ...
    'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
    'recording_fps': FPS,
}
```

iii. `CONVERSION_NOTES.md` explicitly states a 30 Hz sampling rate and 33.33 ms time bin size. The trajectory includes a separate discussion about file size where the AI decides to keep the native 30 Hz bins rather than downsample.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from the raw `envs` session label, not from the raw `blocked` variable.

ii.
```python
env_name = dat['envs'][sess_idx][0]
env_mat = get_env_mat(env_name).flatten().astype(np.float32)
```

iii. The notes say the input is a 3x3 geometry matrix and that this "matches the `get_env_mat` function in the reference code." The trajectory says the AI inspected `get_env_mat` and decided to use environment geometry as the decoder input.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped to a hand-coded 3x3 binary matrix in which `1` means accessible and `0` means blocked/omitted. The matrix is flattened to a length-9 vector and reused as a static trial-level input.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)

env_mat = get_env_mat(env_name).flatten().astype(np.float32)
trial_input = env_mat.copy()
```

iii. `CONVERSION_NOTES.md` says the 3x3 geometry comes from the reference repository's `get_env_mat` function and is intended to represent accessible versus blocked partitions of the arena.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The environment geometry is treated as static within a session. The same 9-element vector is copied into every trial, with no framewise temporal variation.

ii.
```python
for trial_idx in range(n_full_trials):
    ...
    trial_input = env_mat.copy()
    session_input.append(trial_input)
```

iii. The notes explicitly say the geometry input is "Static per trial (same environment for all trials within a session)." No further alignment logic is described.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw `position` variable for each session.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
```

iii. The notes say the position data come from DeepLabCut tracking and are used as the mouse-position decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI discretizes the continuous 2D position into a 3x3 grid. For each axis, it computes a session-specific bin size from that axis's maximum observed position and then uses `floor(position / bin_size)` followed by clipping. The 2D bin pair is then collapsed to a single class index.

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS):
    pos = position.copy()
    max_vals = np.nanmax(pos, axis=1, keepdims=True)
    bin_size = (max_vals + BUFFER) / n_bins
    binned = np.floor(pos / bin_size).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)
    bin_idx = binned[0] * n_bins + binned[1]
    return bin_idx
```

iii. The notes justify this at a high level by saying that position values in `[0, ~75]` are divided into three equal bins per dimension. They do not mention the code's use of session-specific maxima rather than a fixed 75 cm arena extent.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each timepoint is assigned to one of 9 categories by thresholding x and y separately into three bins and combining the two bin indices as `x_bin * 3 + y_bin`.

ii.
```python
max_vals = np.nanmax(pos, axis=1, keepdims=True)
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
```

iii. `CONVERSION_NOTES.md` says the output values 0-8 use "row-major ordering," but the code itself implements `x_bin * 3 + y_bin`. The trajectory only says the AI wanted a 3x3 spatial discretization.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are assumed to be frame-aligned at the original 30 Hz sampling rate. Both are segmented with the same start and end indices for each 1-minute trial.

ii.
```python
n_timepoints = trace.shape[1]
pos_bins = discretize_position(position, N_SPATIAL_BINS)

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say the output is time-varying at 30 Hz "same as neural data." The trajectory also says the decoder output should be time-varying at 30 Hz.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the original 30 Hz frame rate, so each time bin is 33.33 ms. No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes explicitly state 30 Hz and 33.33 ms, and the trajectory includes a discussion where the AI considered downsampling for file-size reasons but ultimately did not do it.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output streams are aligned frame-for-frame because both are sliced from the same session indices at 30 Hz. The input stream is treated as static per trial, so the same 9-element geometry vector is attached to every trial without any time axis.

ii.
```python
trial_neural = trace_valid[:, start:end]
trial_input = env_mat.copy()
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say the geometry is static per trial and the position output is time-varying at the same 30 Hz rate as the neural data. No more detailed temporal-joining procedure is described.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing-session neurons are removed if their full trace is `NaN`. Remaining `NaN` values inside kept neurons are replaced with zero. Sessions with zero valid neurons or fewer than two complete trials are skipped. Trailing frames that do not fill a full minute are dropped. Unknown environment labels fall back to an all-zero geometry matrix.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
if n_valid == 0:
    continue

trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)

if n_full_trials < 2:
    continue

return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

iii. The notes justify the `NaN` handling in terms of CellReg-tracked neurons that are absent in some sessions. The trajectory says the AI wanted to ensure every session had at least two trials for decoder evaluation. No explicit justification is given for mapping unknown environments to zeros.

## 7-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each full `.mat` file with `mat73.loadmat`, iterating over every session and trial in Python, and writing the very large full pickle plus the extra sample pickle. The code also re-traverses all sessions again when making the sample dataset.

ii.
```python
dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
...
for sess_idx in range(n_sessions):
    ...
    for trial_idx in range(n_full_trials):
        ...

with open(save_path, 'wb') as f:
    pickle.dump(data, f)

sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. The trajectory explicitly discusses the large dataset size ("39 GB is very large") and file-size pressure from storing raw 30 Hz traces. The notes also emphasize the large full dataset and separate sample dataset outputs.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over `trial_idx` could have been replaced by a split/reshape operation for neural and output arrays. The nested loop used only to build `pos_labels` is also unnecessary. The sample-building loop over `subject_idx` could be replaced by grouped indexing.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_input = env_mat.copy()
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
    session_neural.append(trial_neural)
    session_input.append(trial_input)
    session_output.append(trial_output)

for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        pos_labels.append(f"bin_({i},{j})")
```

iii. No explicit justification for keeping these loops appears in the notes or trajectory. This is an implementation property inferred from the code.

## 7-c. What processing does the code repeat multiple times?

i. The code repeatedly copies the same per-session geometry vector into every trial, repeats manual start/end slicing logic for every session, and performs a second pass over all sessions to build `sample_data`. It also recomputes subject session counts during sample creation instead of carrying them forward from the main conversion pass.

ii.
```python
trial_input = env_mat.copy()
session_input.append(trial_input)

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    ...

sample_data = create_sample(data, max_sessions_per_animal=2)
...
for i, si in enumerate(data['subject_idx']):
    ...
```

iii. No explicit justification for these repeated passes is given. The trajectory only indicates that the AI chose to generate both full and sample datasets as required deliverables.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion function always builds and writes `sample_data.pkl`, computes file sizes for logging, and stores extra metadata/counters used only for reporting. It also duplicates the static geometry input separately for every trial even though the value is session-constant.

ii.
```python
sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)

print(f"Saved ({os.path.getsize(save_path) / 1e9:.2f} GB)")
print(f"Saved sample to {sample_path} ({os.path.getsize(sample_path) / 1e6:.1f} MB)")

trial_input = env_mat.copy()
```

iii. The notes justify the sample dataset only as an auxiliary testing artifact. No justification is given for duplicating a session-constant input across trials beyond satisfying the target trial structure.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data the same way as in 6: all-`NaN` neurons are removed, residual `NaN`s are zero-filled, sessions with zero valid neurons or fewer than two full trials are skipped, incomplete trailing frames are dropped, and unknown environments are mapped to an all-zero geometry.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)

if n_valid == 0:
    continue

if n_full_trials < 2:
    continue

return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

iii. The notes and trajectory provide the same justifications as in 6: absent neurons are explained by across-session cell tracking, and the minimum-trial rule is motivated by decoder evaluation requirements.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are again whole-file `mat73.loadmat` reads, Python loops over all sessions and trials, and writing both the large full pickle and the extra sample pickle.

ii.
```python
dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
...
for sess_idx in range(n_sessions):
    ...
    for trial_idx in range(n_full_trials):
        ...

with open(save_path, 'wb') as f:
    pickle.dump(data, f)

sample_data = create_sample(data, max_sessions_per_animal=2)
```

iii. The trajectory explicitly comments on the large size of the converted data and the cost of storing raw 30 Hz traces. No different efficiency rationale is given elsewhere.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The same loops identified in 7-b remain the main vectorization candidates: the per-trial slicing/appending loop, the nested output-label loop, and the sample-selection loop.

ii.
```python
for trial_idx in range(n_full_trials):
    ...

for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        pos_labels.append(f"bin_({i},{j})")

for i, si in enumerate(data['subject_idx']):
    ...
```

iii. No explicit justification for these non-vectorized loops appears in the notes or trajectory.

## 9-c. What processing does the code repeat multiple times?

i. The same repeated work from 7-c applies here: repeated copying of static trial inputs, repeated manual slicing logic for each trial, and a second traversal of all sessions to make the sample dataset.

ii.
```python
trial_input = env_mat.copy()

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    ...

sample_data = create_sample(data, max_sessions_per_animal=2)
```

iii. The notes only justify the existence of a sample dataset as a deliverable; they do not justify these repeated implementation passes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The same unnecessary work from 7-d applies here: always producing `sample_data.pkl`, computing output file sizes for reporting, and duplicating a session-constant input into every trial object.

ii.
```python
sample_data = create_sample(data, max_sessions_per_animal=2)
...
print(f"Saved ({os.path.getsize(save_path) / 1e9:.2f} GB)")
print(f"Saved sample to {sample_path} ({os.path.getsize(sample_path) / 1e6:.1f} MB)")

trial_input = env_mat.copy()
```

iii. No further justification is given beyond meeting the required output-file list and the required session/trial data structure.
