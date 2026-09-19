# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script hard-codes the 7 animal IDs, then loads each corresponding MATLAB file from `/app/data` with `mat73.loadmat`. It loads the whole file into a Python dictionary at once, then iterates through session arrays inside `dat['trace']`, `dat['position']`, and `dat['envs']`. Trials are created later by slicing each session into 1-minute chunks.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
    n_sessions = len(dat['trace'])
    ...
    for sess_idx in range(n_sessions):
        trace = np.array(dat['trace'][sess_idx])
        position = np.array(dat['position'][sess_idx])
        env_name = dat['envs'][sess_idx][0]
```

iii. In the trajectory, the agent said it would "load 7 animals' CA1 calcium imaging data from MATLAB files" and explicitly chose `mat73` after inspecting a sample `.mat` file. It justified this by confirming the files contained `trace`, `position`, `envs`, and `blocked`, and by matching the paper counts of 7 animals, 207 sessions, and 5,413 neurons.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded entries in `ANIMALS`. Each animal name corresponds to one `.mat` file and one entry in `subjects`; `subject_idx` stores the index of the animal for every session appended.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for animal_idx, animal in enumerate(ANIMALS):
    ...
    all_neural.append(session_neural)
    ...
    subject_idx_list.append(animal_idx)
...
'subjects': ANIMALS,
'subject_idx': np.array(subject_idx_list, dtype=int),
```

iii. In the trajectory, the agent checked all seven named animals and noted that the totals matched the paper, so it treated each named animal file as one subject.

## 1-c. How are the data split into sessions?

i. Within each subject file, each element of `dat['trace']`, `dat['position']`, and `dat['envs']` is treated as one recording session. The outer output lists contain one item per session.

ii.
```python
n_sessions = len(dat['trace'])
...
for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])
    position = np.array(dat['position'][sess_idx])
    env_name = dat['envs'][sess_idx][0]
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The agent stated in the trajectory that "Each recording session (40 min) is one session" and verified that the per-animal session counts matched the paper.

## 1-d. How are the data split into trials?

i. Each continuous session is split into non-overlapping 1-minute trials at 30 Hz, so each trial is 1,800 frames. The number of usable trials is `n_timepoints // 1800`, so any remainder at the end of a session is dropped.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
...
n_full_trials = n_timepoints // FRAMES_PER_TRIAL
...
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_input = env_mat.copy()
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The trajectory says the agent followed the task instruction that sessions should be split into 1-minute trials, and it repeatedly described this as 30 Hz, 1,800 frames per trial.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality-control filter. The script only requires that a session contain at least 2 full 1-minute trials; otherwise the whole session is skipped.

ii.
```python
n_full_trials = n_timepoints // FRAMES_PER_TRIAL

if n_full_trials < 2:
    print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
    continue
```

iii. In the trajectory, the agent noted the decoder format requirement that each session must have at least two trials and added this check for compatibility, not because of any trial-specific QC described in the source data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-session `trace` arrays in the MATLAB files.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. The agent inspected the raw files and described the traces as binary rising-phase calcium transients at 30 Hz.

## 2-b. How is the `neural` data processed?

i. The script keeps the session's valid neuron rows, copies them, replaces any remaining NaNs with zeros, and casts the result to `float32`. It does not resample or otherwise transform the traces.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
...
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
...
trial_neural = trace_valid[:, start:end]
```

iii. The trajectory says the agent viewed the data as binary 0/1 transients and chose `float32` for decoder compatibility and file-size reduction. It also inferred that NaN rows indicated neurons not present in a session, and treated remaining NaNs as fillable with zero.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by dropping rows that are entirely NaN within a session. Sessions with zero valid neurons are skipped completely.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()

if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue
```

iii. The agent explicitly justified this in the trajectory by saying the neurons were tracked across sessions with CellReg and that all-NaN rows meant a neuron was not detected in that session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not perform event-triggered alignment. It uses contiguous slices of the continuous recording. However, in metadata it labels the alignment event as `"Start of recording session"` and records offsets of 0 to 60 seconds.

ii.
```python
trial_neural = trace_valid[:, start:end]
...
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
    ...
}
```

iii. In the trajectory, the agent never identified a real task event; instead it described the recordings as continuous 40-minute sessions split into artificial 1-minute trials. The `"Start of recording session"` metadata appears to be a convenience label added to satisfy the target format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz frame rate, which the script records as `1000 / 30` ms per bin. No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
...
trial_neural = trace_valid[:, start:end]
```

iii. The agent repeatedly stated in the trajectory that the traces were already at 30 Hz and that each 1-minute trial should therefore contain 1,800 native frames.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` session label, not from `blocked`. For each session it reads the environment name and maps it to a hard-coded 3x3 geometry template.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
...
env_name = dat['envs'][sess_idx][0]
env_mat = get_env_mat(env_name).flatten().astype(np.float32)
```

iii. In the trajectory, the agent examined both `envs` and `blocked`, then decided that the decoder input should be "Environment geometry as 3x3 binary matrix" and referred to `get_env_mat` from the reference code as support for that choice.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The processing is a lookup from environment name to a hard-coded 3x3 binary template, flattened to length 9 and copied into every trial of the session. The code uses `1` for open cells and `0` for blocked cells.

ii.
```python
env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)
...
trial_input = env_mat.copy()
...
session_input.append(trial_input)
```

iii. The trajectory says the agent wanted a static per-trial geometry input and believed the hard-coded matrices were the right representation of the 10 named environments.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the per-session `position` arrays.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
```

iii. The agent inspected the raw files, confirmed the arrays were 2D x/y coordinates in the 75 cm arena, and chose them as the decoder target.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The script discretizes each session's 2D position into a 3x3 grid by computing the per-axis session maximum, dividing each axis into 3 equal-width bins based on that maximum, flooring the normalized coordinates, clipping to `[0, 2]`, and converting the 2D bin to a single class index.

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

iii. In the trajectory, the agent justified this by checking that position ranged from roughly 0 to 75 cm and then deciding to discretize into a 3x3 grid for the decoder task.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each timepoint is assigned to one of 9 categories according to the 3x3 bin computed by `discretize_position`. Category indices are encoded as a single integer per frame, in the order `x_bin * 3 + y_bin`.

ii.
```python
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
...
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The trajectory says the agent wanted the required 9-class output and viewed a single integer label per frame as the natural categorical encoding.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position labels are computed from the same session timebase as the neural traces and then sliced into trials using the same `start:end` frame indices, so the alignment is frame-by-frame within each trial.

ii.
```python
pos_bins = discretize_position(position, N_SPATIAL_BINS)  # (n_timepoints,)
...
trial_neural = trace_valid[:, start:end]
...
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. In the trajectory, the agent treated the position stream and trace stream as already synchronized in the source files and therefore used identical slicing for both.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing all-NaN neurons are removed. Any remaining NaNs in retained neurons are replaced with 0. Session tail frames that do not fill a full 1-minute trial are discarded implicitly. Sessions with zero valid neurons or fewer than two full trials are skipped.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
...
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
...
n_full_trials = n_timepoints // FRAMES_PER_TRIAL
...
if n_valid == 0:
    ...
if n_full_trials < 2:
    ...
```

iii. The trajectory shows the agent explicitly checking that many neurons were all-NaN in a session and interpreting that as "not detected/tracked". It also added the two-trial session filter to satisfy decoder-format requirements.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are whole-file MATLAB loading with `mat73.loadmat`, iterating through all sessions and trials to materialize per-trial arrays, and serializing the very large pickle outputs. The extra sample dataset creation also adds another pass over the data structure and another pickle write.

ii.
```python
dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
...
for sess_idx in range(n_sessions):
    ...
    for trial_idx in range(n_full_trials):
        ...
        session_neural.append(trial_neural)
        session_input.append(trial_input)
        session_output.append(trial_output)
...
with open(save_path, 'wb') as f:
    pickle.dump(data, f)
...
sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. The trajectory repeatedly complains about the output size (first about 39 GB, then about 20 GB), which shows the agent itself identified loading and serialization of full-resolution data as the dominant cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-building loop could have been vectorized by reshaping or by using a helper like the human reference `split_into_trials`, rather than appending one trial at a time in Python. The output label-name construction loop is also unnecessary Python iteration.

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
...
pos_labels = []
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        pos_labels.append(f"bin_({i},{j})")
```

iii. The trajectory does not mention vectorization; these opportunities are visible from the final code structure.

## 6-c. What processing does the code repeat multiple times?

i. The same static environment vector is copied once per trial even though it is session-constant. After building the full dataset, the script iterates through sessions again to create `sample_data`, duplicating part of the work for a byproduct file.

ii.
```python
trial_input = env_mat.copy()
...
session_input.append(trial_input)
...
sample_data = create_sample(data, max_sessions_per_animal=2)
...
for i, si in enumerate(data['subject_idx']):
    if subject_counts.get(si_int, 0) < max_sessions_per_animal:
        neural.append(data['neural'][i])
        inp.append(data['input'][i])
        out.append(data['output'][i])
```

iii. The trajectory makes clear that `sample_data.pkl` was added for testing convenience after the main conversion was already implemented.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script always creates `sample_data.pkl`, even though the required deliverable is `converted_data.pkl`. It also stores extensive summary metadata that is not needed by the decoder, and it prints file-size summaries purely for inspection.

ii.
```python
sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
...
'metadata': {
    ...
    'session_duration_min': 40,
    'n_animals': len(ANIMALS),
    'n_sessions': total_sessions,
    'n_trials': total_trials,
    'n_unique_neurons': 5413,
    'environments': ['square', 'o', 't', 'u', 'rectangle', '+', 'i', 'l', 'bit donut', 'glenn'],
    'neural_data_type': 'Binary rising-phase calcium transients',
}
```

iii. In the trajectory, the agent explicitly created the sample file and extra reporting to support its own verification workflow, not because the target format required them.
