# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded the seven animal IDs, loaded one `.mat` file per animal with `mat73.loadmat`, and then iterated through every session in `dat['trace']`. Within each session it pulled `trace`, `position`, and `envs`, then later split those session arrays into trials. It did not load the raw `blocked` field that the reference converter uses.

ii.
```python
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

iii. In `CONVERSION_NOTES.md`, the agent says it used `mat73.loadmat` and considered this the "same loading approach as reference code (`load_dat` function in `utils.py`)". The trajectory shows it inspected `load_dat` in the paper code and then chose direct `.mat` loading.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by file. Each filename/animal ID in the hard-coded `ANIMALS` list is treated as one mouse, and every session produced from that file receives that animal's integer index in `subject_idx`.

ii.
```python
for animal_idx, animal in enumerate(ANIMALS):
    dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    subject_idx_list.append(animal_idx)

...
'subjects': ANIMALS,
'subject_idx': np.array(subject_idx_list, dtype=int),
```

iii. The notes explicitly list the seven animal IDs and describe them as the full subject set from the paper.

## 1-c. How are the data split into sessions?

i. The agent treated each entry of `dat['trace']`, `dat['position']`, and `dat['envs']` as a separate recording session. One output session is created for each `sess_idx`.

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

iii. In the notes, the agent states that "each recording session corresponds to one environment per day (40 min)". The trajectory also shows it inspected session counts and environment order per animal before implementing this.

## 1-d. How are the data split into trials?

i. Each session is split into consecutive non-overlapping 1-minute trials at 30 Hz, so each trial has 1800 frames. The agent drops any leftover tail shorter than a full minute.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial

n_full_trials = n_timepoints // FRAMES_PER_TRIAL

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say sessions are "split into 1-minute trials as specified in the decoder task" and report how many complete 1-minute trials each animal yields.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality-control filter. Trials are kept as long as they fall inside a session that has at least one valid neuron and at least two complete 1-minute trials.

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

iii. The notes defend a minimal filtering strategy: "No velocity filtering or place cell filtering applied" and "All valid neurons included for maximum information available to the decoder."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the raw `trace` variable in each `.mat` file.

ii.
```python
trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
...
trial_neural = trace_valid[:, start:end]
```

iii. The notes identify the trace data as "binary rising-phase calcium transients" and describe it as matching the paper's processed calcium signal.

## 2-b. How is the `neural` data processed?

i. The agent keeps the provided binary trace values, removes neurons that are all `NaN` in a session, fills any remaining `NaN` values with `0`, and casts the data to `float32`. No additional smoothing, deconvolution, thresholding, or temporal rebinning is applied.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
trace_valid = trace[valid_neurons].copy()
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the neural data are already "Binary rising-phase calcium transients (0 or 1)" and that `float32` is used for storage/decoder compatibility. The trajectory shows the agent intentionally moved `nan_to_num` before the cast.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC is excluding neurons whose entire session trace is `NaN`. All other neurons are retained.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
n_valid = valid_neurons.sum()

if n_valid == 0:
    print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
    continue
```

iii. The notes explain this as tracking-related missingness: "NaN values indicate neurons not detected/tracked in that session (CellReg tracking across days)." The same notes explicitly reject extra velocity/place-cell filtering at conversion time.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align to a behavioral event. It simply chops the continuous session recording into consecutive 1-minute windows and uses identical frame boundaries for neural and behavioral slices, so alignment is to trial window start within the recording.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)

...
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. The notes describe long 40-minute recordings being split into 1-minute trials; there is no separate event-alignment rationale in the notes or trajectory beyond this fixed-window segmentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the original 30 Hz sampling rate, so each bin is about 33.33 ms. No temporal rebinning is applied.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The notes explicitly state "Sampling rate: 30 Hz" and "Time bin size: 33.33 ms (1000/30)".

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The agent derives the decoder input from the raw `envs` labels, not from the raw `blocked` field. It converts each environment name into a hard-coded geometry template.

ii.
```python
env_name = dat['envs'][sess_idx][0]
...
env_mat = get_env_mat(env_name).flatten().astype(np.float32)
```

iii. The notes say this "matches the `get_env_mat` function in the reference code" and describe the input as a 3x3 binary matrix of accessible partitions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent maps each environment name to a 3x3 binary accessibility matrix, flattens it to length 9, casts it to `float32`, and copies the same 9-vector into every trial of that session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)

env_mat = get_env_mat(env_name).flatten().astype(np.float32)
trial_input = env_mat.copy()
```

iii. The notes justify this as a decoder input that marks "1 = accessible partition, 0 = blocked partition" and say it is static within a session/trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the raw `position` variable in each session.

ii.
```python
position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
...
pos_bins = discretize_position(position, N_SPATIAL_BINS)
```

iii. The notes describe this as "Mouse position (from DeepLabCut tracking)".

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent discretizes each session's 2D position stream into a 3x3 grid by taking the per-axis session maxima, dividing each axis by its own bin width, flooring to integers, clipping to `[0, 2]`, and then converting each `(x_bin, y_bin)` pair into a single class index.

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

iii. The notes justify this as dividing position values in `[0, ~75]` into "3 equal bins per dimension" and say that this is consistent with the task requirement to decode 3x3 spatial bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each timepoint is assigned one of nine categories by thresholding x and y coordinates into three bins each, then flattening the 2D bin to a scalar index using `x * 3 + y`. The category labels are named `bin_(i,j)` in the same nested-loop order.

ii.
```python
binned = np.floor(pos / bin_size).astype(int)
binned = np.clip(binned, 0, n_bins - 1)
bin_idx = binned[0] * n_bins + binned[1]

pos_labels = []
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        pos_labels.append(f"bin_({i},{j})")
```

iii. The notes state that output values `0-8` use "row-major ordering of the 3x3 grid: `bin_idx = row * 3 + col`."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is discretized at the full session frame rate first, then each trial's output slice is cut with the same `[start:end]` indices used for the neural trace. There is no extra temporal offset.

ii.
```python
pos_bins = discretize_position(position, N_SPATIAL_BINS)

for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes describe the output as "Time-varying at 30 Hz (same as neural data)."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles all-`NaN` neurons by dropping them, replaces any remaining neural `NaN` values with `0`, drops incomplete trailing trial fragments, and falls back to an all-zero geometry matrix for an unrecognized environment label. It does not explicitly handle missing position samples.

ii.
```python
valid_neurons = ~np.isnan(trace).all(axis=1)
...
trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)
...
n_full_trials = n_timepoints // FRAMES_PER_TRIAL
...
return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

iii. The only explicit justification in the notes is for neural missingness: NaNs were interpreted as neurons not tracked in a given session. No separate justification was given for the zero fallback on unknown environments or for leaving position missingness unhandled.

## 6-a. What are the most time-consuming steps of the code?

i. The slowest steps are loading the large `.mat` files into memory, iterating over all animals/sessions/trials to slice out every 1-minute trial, and serializing the full nested dataset to pickle. Creating `sample_data.pkl` adds a second full pass over the already-built dataset.

ii.
```python
for animal_idx, animal in enumerate(ANIMALS):
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

iii. No explicit performance justification appears in the notes or trajectory. This is an inference from the implemented control flow and the size of the dataset.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop could have been vectorized by reshaping session arrays into `(n_trials, 1800)` chunks instead of appending one trial at a time. The nested label-building loop and the sample-creation loop are also simple Python loops that could be replaced with array/list comprehensions or reshapes.

ii.
```python
for trial_idx in range(n_full_trials):
    start = trial_idx * FRAMES_PER_TRIAL
    end = start + FRAMES_PER_TRIAL
    trial_neural = trace_valid[:, start:end]
    ...

for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        pos_labels.append(f"bin_({i},{j})")

for i, si in enumerate(data['subject_idx']):
    ...
```

iii. No explicit justification was given. These loops are simply present in the code as written.

## 6-c. What processing does the code repeat multiple times?

i. The same environment geometry vector is rebuilt for every session from a small fixed lookup table and then copied once per trial. After finishing the full conversion, the code traverses the converted dataset again to build a sample subset. The position stream is discretized for a whole session and then repeatedly re-sliced per trial.

ii.
```python
env_mat = get_env_mat(env_name).flatten().astype(np.float32)
...
trial_input = env_mat.copy()
...
sample_data = create_sample(data, max_sessions_per_animal=2)
...
pos_bins = discretize_position(position, N_SPATIAL_BINS)
for trial_idx in range(n_full_trials):
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. No explicit justification was given beyond convenience and producing both full and sample outputs.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main converter always builds and writes a `sample_data.pkl` dataset even though it is separate from the full converted dataset. It also stores many metadata summary fields that the downstream decoder does not need. The code additionally duplicates the same static session input vector into every trial rather than storing it once per session.

ii.
```python
print(f"\nCreating sample dataset...")
sample_data = create_sample(data, max_sessions_per_animal=2)
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)

'metadata': {
    'recording_fps': FPS,
    'trial_duration_s': TRIAL_DURATION_S,
    'n_spatial_bins': N_SPATIAL_BINS,
    'arena_size_cm': 75,
    'session_duration_min': 40,
    'n_animals': len(ANIMALS),
    'n_sessions': total_sessions,
    'n_trials': total_trials,
    'n_unique_neurons': 5413,
    ...
}
```

iii. No explicit justification was given in the notes or trajectory beyond wanting a sample dataset and rich documentation metadata.
