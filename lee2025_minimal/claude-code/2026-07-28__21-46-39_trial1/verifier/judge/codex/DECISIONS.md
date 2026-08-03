# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes the seven animal IDs, then iterates through those `.mat` files in `/app/data`. Each file is loaded in full with `mat73.loadmat`, and the per-session arrays are then read from the loaded dictionary via `trace`, `position`, and `envs`.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

for animal_idx, animal in enumerate(ANIMALS):
    print(f"\nLoading {animal}...")
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))

    n_sessions = len(dat['trace'])
```

iii. In `CONVERSION_NOTES.md`, the AI says it "Loaded from original MATLAB (.mat) files using `mat73.loadmat`" and that this was the "Same loading approach as reference code (`load_dat` function in `utils.py`)."

## 1-b. How are the data split into subjects?

i. Subjects are the seven hardcoded animal IDs. The `subjects` field is populated directly from `ANIMALS`, and each session gets the integer index of its animal in that list.

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
```

iii. The notes justify this by listing the seven animals explicitly and treating each mouse as one subject.

## 1-c. How are the data split into sessions?

i. Within each animal file, the AI treats each index of `dat['trace']` as one recording session. The same session index is used to fetch the matching `position` and `envs` entries.

ii.
```python
n_sessions = len(dat['trace'])

for sess_idx in range(n_sessions):
    trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
    position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
    env_name = dat['envs'][sess_idx][0]
```

iii. In the notes, the AI states that "Each recording session corresponds to one environment per day" and uses the paper's statement that all sessions were 40 minutes long.

## 1-d. How are the data split into trials?

i. Each session is split into consecutive non-overlapping 1-minute trials at 30 Hz. The AI computes `1800` frames per trial, takes `n_timepoints // 1800` full trials, and slices each trial with explicit start/end indices. Any remainder is dropped.

ii.
```python
FPS = 30  # frames per second
TRIAL_DURATION_S = 60  # 1 minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial

n_full_trials = n_timepoints // FRAMES_PER_TRIAL

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL

    trial_neural = trace_valid[:, start:end]
    trial_input = env_mat.copy()
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say sessions were "split into 1-minute trials as specified in the decoder task" and explicitly mention leftover frames such as "1266 frames unused."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a trial-level quality filter. It only skips whole sessions with no valid neurons or with fewer than two full 1-minute trials.

ii.
```python
if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue

...

if n_full_trials < 2:
    print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
    continue
```

iii. The notes explicitly say "No Filtering Applied," while the code still enforces the decoder requirement that a session must have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the raw `trace` entry for each session.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
```

iii. The notes say the traces are "Binary rising-phase calcium transients (0 or 1)," citing the paper's preprocessing description.

## 2-b. How is the `neural` data processed?

i. The AI assumes `trace` is already arranged as `(n_neurons, n_timepoints)`. It filters to valid neurons, copies that subset, replaces any remaining NaNs with `0.0`, and casts to `float32`. It does not apply additional deconvolution or temporal processing.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
...
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The notes justify this by saying the traces are already binary rising-phase transients, that NaN neurons are excluded, that any remaining NaNs are set to zero, and that `float32` is used for storage efficiency and decoder compatibility.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if they are not all-NaN within that session. After that, any remaining NaNs inside kept neurons are zero-filled.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()

if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue

trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. The notes say NaNs indicate neurons not detected or tracked in a session, so those neurons are excluded, and "any remaining NaN values in valid neurons are set to 0."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. In the actual data arrays, the neural data is not aligned to any experimental event; it is just sliced into contiguous 60-second chunks from the continuous session. However, the metadata labels the alignment event as `"Start of recording session"` with offsets `0.0` to `60.0`.

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
```

iii. The notes frame the recordings as continuous 40-minute sessions split into 1-minute trials, but they do not give a separate event-based alignment justification beyond that.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz sampling rate, so the time bin size is `1000 / 30 ~= 33.33 ms`. No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
TRIAL_DURATION_S = 60  # 1 minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial

'metadata': {
    ...
    'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The notes explicitly say "Sampling rate: 30 Hz" and "Time bin size: 33.33 ms (1000/30)."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the raw `envs` variable, not from `blocked`. The session's environment name is looked up and converted into a geometry matrix.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
env_name = dat['envs'][sess_idx][0]

...

env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)
```

iii. The notes justify this by saying the decoder input is "Environment geometry" and that the 10 named environments match the `get_env_mat` function from the reference code.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI uses a hardcoded lookup table from environment name to a 3x3 binary matrix of accessible versus blocked locations, flattens that matrix to length 9, casts it to `float32`, and copies the same vector into every trial from that session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        'u':         [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+':         [[0,1,0],[1,1,1],[0,1,0]],
        'i':         [[1,1,1],[0,1,0],[1,1,1]],
        'l':         [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn':     [[1,1,0],[1,1,1],[0,1,1]],
    }
    return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)

...

env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)
...
trial_input = env_mat.copy()
```

iii. The notes say this is a 3x3 binary environment geometry, "1 = accessible partition, 0 = blocked partition," and that it "Matches the `get_env_mat` function in the reference code."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw `position` variable for each session.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
```

iii. The notes say the position comes from DeepLabCut tracking and spans approximately a 75 cm by 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI discretizes the 2D position into a 3x3 grid by taking the per-session maximum of each coordinate, dividing each axis into three equal ranges based on that session-specific maximum, flooring to integer bins, clipping to `[0, 2]`, and combining the two axis bins into one class index.

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

...

pos_bins = discretize_position(position, N_SPATIAL_BINS)  # (n_timepoints,)
```

iii. The notes justify the output as 3x3 spatial bins and say the values `[0, ~75]` are divided into three equal bins per dimension, although the code actually uses per-session maxima rather than a fixed 75 cm boundary.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is thresholded by `floor(position / bin_size)` where `bin_size = (session_max + 1e-5) / 3` for that coordinate. The resulting x/y bins are clipped to `0, 1, 2` and combined into one of nine categories by `x_bin * 3 + y_bin`.

ii.
```python
max_vals = np.nanmax(pos, axis=1, keepdims=True)
bin_size = (max_vals + BUFFER) / n_bins
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]
```

iii. The notes say the position is divided into three equal bins per dimension and that the final labels use row-major ordering, but they do not separately justify the code's session-max thresholding rule.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The output and neural streams are assumed to be frame-synchronous within a session. The AI computes `pos_bins` from the full session position array and slices neural and output trials with the same `start:end` frame indices.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)

...

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL

    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say the position output is "Time-varying at 30 Hz (same as neural data)," which is the AI's justification for framewise alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes neurons that are entirely NaN within a session, zero-fills any remaining NaNs inside kept neurons, skips sessions with no valid neurons or with fewer than two full trials, and drops leftover frames that do not fill a 1-minute trial.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()

if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue

trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)

n_full_trials = n_timepoints // FRAMES_PER_TRIAL
if n_full_trials < 2:
    print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
    continue
```

iii. The notes say NaN neurons are excluded because they reflect missing tracking across sessions, that any remaining NaNs are set to zero, and that incomplete trailing frames are left unused.

## 6-a. What are the most time-consuming steps of the code?

i. The AI does not explicitly document this, but the code suggests the slowest steps are loading full `.mat` files with `mat73.loadmat`, iterating through all session/trial slices, and writing the very large full and sample pickle outputs.

ii.
```python
dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))

...

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_input = env_mat.copy()
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)

...

with open(save_path, 'wb') as f:
    pickle.dump(data, f)

...

with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. No explicit performance justification was given in `CONVERSION_NOTES.md` or the captured trajectory.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop is the main vectorization target: the code repeatedly computes `start:end` indices and appends slices one trial at a time instead of reshaping or splitting once. The nested `pos_labels` loop is minor by comparison.

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

iii. No explicit justification for retaining these Python loops appears in the notes or trajectory.

## 6-c. What processing does the code repeat multiple times?

i. The same static environment vector is copied once per trial even though it is constant within a session. The code also replays the session data in a second pass to build `sample_data.pkl` after the full dataset is already assembled.

ii.
```python
env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)

for trial_idx in range(n_full_trials):
    ...
    trial_input = env_mat.copy()
    ...

...

sample_data = create_sample(data, max_sessions_per_animal=2)

...

for i, si in enumerate(data['subject_idx']):
    si_int = int(si)
    if subject_counts.get(si_int, 0) < max_sessions_per_animal:
        neural.append(data['neural'][i])
        inp.append(data['input'][i])
        out.append(data['output'][i])
```

iii. The AI did not explicitly justify these repeated operations in its notes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Relative to the core full-dataset decoder path, the script also builds and saves a separate sample dataset, constructs verbose label/metadata bookkeeping, and copies the same static environment vector into every trial. Those extra steps are not needed to define the full converted session/trial tensors themselves.

ii.
```python
pos_labels = []
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        pos_labels.append(f"bin_({i},{j})")

...

'metadata': {
    'task_description': 'Decode mouse position from CA1 neural activity during free exploration of geometric environments',
    'time_bin_size': 1000.0 / FPS,
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
    'recording_fps': FPS,
    'trial_duration_s': TRIAL_DURATION_S,
    'n_spatial_bins': N_SPATIAL_BINS,
    'arena_size_cm': 75,
    'session_duration_min': 40,
    'n_animals': len(ANIMALS),
    'n_sessions': total_sessions,
    'n_trials': total_trials,
    'n_unique_neurons': 5413,
    'environments': ['square', 'o', 't', 'u', 'rectangle', '+', 'i', 'l', 'bit donut', 'glenn'],
    'neural_data_type': 'Binary rising-phase calcium transients',
}

...

sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. No explicit justification for these extra steps appears in the AI's notes beyond satisfying the broader deliverables.
