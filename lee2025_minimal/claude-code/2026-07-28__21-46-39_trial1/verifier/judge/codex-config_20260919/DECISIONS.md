# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses `mat73.loadmat` to load seven explicitly named MATLAB v7.3 files from `/app/data`. It loads each whole animal file, then iterates through every entry of `dat['trace']`; the corresponding `position` and `envs` entries are read by the same session index. The full conversion has no sampling limit.

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
    for sess_idx in range(n_sessions):
        trace = np.array(dat['trace'][sess_idx])
        position = np.array(dat['position'][sess_idx])
        env_name = dat['envs'][sess_idx][0]
```

iii. The trajectory says this is the same loading approach as the repository's `load_dat` utility. The agent inspected every animal and confirmed 5,413 neurons and 207 sessions, matching the paper.

## 1-b. How are the data split into subjects?

i. Subject identity is defined by the seven hard-coded animal IDs. One `.mat` file is loaded per subject, and each retained session receives the integer `animal_idx` in `subject_idx`.

ii.
```python
for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
    ...
    subject_idx_list.append(animal_idx)
...
'subjects': ANIMALS,
'subject_idx': np.array(subject_idx_list, dtype=int),
```

iii. The agent found that the seven named files correspond to the paper's seven mice and used the file/animal identity as the subject boundary.

## 1-c. How are the data split into sessions?

i. Every element of an animal's `trace` list is treated as one recording session; `position` and `envs` are indexed in parallel. A session is appended only after neuron- and trial-count checks.

ii.
```python
n_sessions = len(dat['trace'])
for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])
    position = np.array(dat['position'][sess_idx])
    env_name = dat['envs'][sess_idx][0]
    ...
    all_neural.append(session_neural)
```

iii. The trajectory identifies each approximately 40-minute, single-environment recording/day as a session, consistent with the paper and source structure.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second trials: 1,800 frames at 30 Hz. Only complete trials are kept; the trailing partial interval is discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
...
n_full_trials = n_timepoints // FRAMES_PER_TRIAL
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
```

iii. The agent explicitly cited the decoder instruction requiring one-minute trials and retained the native 30 Hz samples.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level behavioral quality filter. Incomplete trailing chunks are dropped, and an entire session is skipped if it has no valid neurons or fewer than two complete trials.

ii.
```python
if n_valid == 0:
    continue
...
n_full_trials = n_timepoints // FRAMES_PER_TRIAL
if n_full_trials < 2:
    continue
```

iii. The agent stated that velocity and place-cell filters were analysis-specific and should not be applied here. The two-trial session check follows the target format's minimum-session requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from each session's raw `trace` array.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. After inspecting values and the methods, the agent identified `trace` as binary rising-phase calcium transients, already sampled at 30 Hz.

## 2-b. How is the `neural` data processed?

i. The array is already neuron-by-time, so it is not transposed or rebinned. Valid neuron rows are selected, the result is copied, any residual NaNs are replaced by zero, and values are cast to `float32` before trial slicing.

ii.
```python
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
...
trial_neural = trace_valid[:, start:end]
```

iii. The agent concluded from the paper and its inspection that the traces were already binarized/deconvolved, so further neural processing was unwarranted. It chose `float32` for decoder compatibility and storage efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. For each session, neurons whose complete trace is NaN are removed. Any NaNs remaining in otherwise retained neurons are filled with zero. No activity-rate, velocity, or place-cell filter is used.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()
...
trace_valid = np.nan_to_num(trace[valid_neurons].copy(), nan=0.0).astype(np.float32)
```

iii. The trajectory records the inference that all-NaN rows represent CellReg-tracked neurons absent from that session. The agent intentionally retained every actually recorded neuron to maximize decoder information.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Trial zero starts at recording-session frame zero, and each later trial starts at the next 1,800-frame boundary. Metadata calls the alignment event the start of the recording session.

ii.
```python
start = trial_idx * FRAMES_PER_TRIAL
end = start + FRAMES_PER_TRIAL
trial_neural = trace_valid[:, start:end]
...
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. The agent treated the recordings as continuous free exploration with no stimulus event, making artificial one-minute boundaries the applicable alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz resolution is preserved, giving 33.333 ms per frame. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The agent verified that the source traces are at 30 Hz and decided that the already consistent sampling rate required no resampling.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the session's environment name in raw `envs`, not from raw `blocked`. That name indexes a manually specified geometry lookup table.

ii.
```python
env_name = dat['envs'][sess_idx][0]
...
env_mat = get_env_mat(env_name).flatten().astype(np.float32)
```

iii. The agent examined the repository's `get_env_mat` function and chose to reproduce it, describing this as the reference-code representation of the ten geometries.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `get_env_mat` maps each name to a 3×3 binary accessible-space matrix (1 accessible, 0 blocked), flattens it to nine `float32` values, and copies the same static vector into every trial of that session. An unknown name silently maps to nine zeros.

ii.
```python
env_mats = {
    'square': [[1,1,1],[1,1,1],[1,1,1]],
    'o':      [[1,1,1],[1,0,1],[1,1,1]],
    ...
}
return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
trial_input = env_mat.copy()
```

iii. The agent reasoned that a flattened binary 3×3 matrix directly describes the arena, follows repository code, and is static because one geometry is used throughout a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the two-coordinate time series in each session's raw `position` entry.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
pos_bins = discretize_position(position, N_SPATIAL_BINS)
```

iii. The agent inspected the values and identified them as the mouse's tracked x/y location, approximately spanning the 75×75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. A copy is made, each coordinate is divided by one third of that session's observed coordinate-wise maximum (plus a small buffer), and the quotient is floored and clipped to 0–2. The two bins are combined as `x_bin * 3 + y_bin`, then trial outputs are reshaped to `(1, 1800)` and cast to `int64`.

ii.
```python
max_vals = np.nanmax(pos, axis=1, keepdims=True)
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
```

iii. The agent wanted nine coarse spatial classes and described the operation as dividing the observed 0-to-about-75 range into three equal bins per dimension.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The thresholds are session-dependent: for coordinate `d`, they are approximately one-third and two-thirds of `nanmax(position[d])`. Values below, between, and above those boundaries become categories 0, 1, and 2; clipping handles extremes. Paired coordinate categories yield labels 0–8.

ii.
```python
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
```

iii. The agent's stated goal was equal thirds of the arena along each axis, with a buffer and clipping to keep maxima inside the valid class range.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace samples are assumed to be frame-synchronous. The discretized full-session position and neural trace are cut with identical `start:end` indices for each trial.

ii.
```python
trial_neural = trace_valid[:, start:end]
...
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The agent treated both streams as native 30 Hz arrays of the same session and therefore used direct frame-for-frame alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neuron rows are removed; residual NaNs in retained traces are changed to zero. Position maxima use `nanmax`, out-of-range position bins are clipped, incomplete final trials are discarded, empty/too-short sessions are skipped, and unknown environments silently become all-zero geometry.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
...
binned = np.clip(binned, 0, n_bins - 1)
...
if n_valid == 0: continue
if n_full_trials < 2: continue
```

iii. The trajectory specifically explains all-NaN rows as absent tracked cells, zero filling as a safeguard, clipping as boundary protection, and dropping remainders as the consequence of requiring fixed one-minute trials.

## 6-a. What are the most time-consuming steps of the code?

i. Loading seven large MATLAB files with `mat73`, copying/converting all valid traces, serializing the roughly 20 GB full pickle, and then redundantly serializing a sample are the dominant conversion costs. The nested trial construction also creates thousands of Python objects, though trial neural slices are generally views.

ii.
```python
dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
trace_valid = trace[valid_neurons].copy()
...
with open(save_path, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory repeatedly notes that full conversion produced a 20 GB file and spent substantial time rerunning conversion and verification. It considered storage precision because full-data I/O was burdensome.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial slicing is done in a Python loop and could be reshaped into trial-major arrays before converting to the required lists. The two nested loops used only to create nine position labels could be a comprehension. Animal/session iteration cannot usefully be eliminated because files and sessions have heterogeneous neuron counts.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    session_neural.append(trace_valid[:, start:end])
...
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        pos_labels.append(f"bin_({i},{j})")
```

iii. The trajectory provides no explicit vectorization analysis. The agent prioritized constructing the required nested-list representation, for which some per-trial iteration remains necessary.

## 6-c. What processing does the code repeat multiple times?

i. Within each trial it repeatedly copies the same session-static environment vector and performs the same start/end arithmetic. After full conversion, `create_sample` walks the session collections again and serializes a second dataset containing selected session data. During development, the trajectory also shows several complete conversion reruns after dtype changes, though those reruns are not part of one script execution.

ii.
```python
for trial_idx in range(n_full_trials):
    ...
    trial_input = env_mat.copy()
...
sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. No explicit justification was given for repeated per-trial input copies. The sample pass was intended to create a quick-testing dataset for verification and decoder training.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The required full dataset does not need the automatically generated sample pickle, extensive printing, or much of the descriptive metadata. `position.copy()` is also unnecessary because position is never mutated. The code computes and stores no major analysis product that is later discarded from the full output.

ii.
```python
pos = position.copy()
...
sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. The trajectory justifies the sample as useful for quick validation, not as required downstream data. It gives no justification for the position copy; the additional metadata is documentary.
