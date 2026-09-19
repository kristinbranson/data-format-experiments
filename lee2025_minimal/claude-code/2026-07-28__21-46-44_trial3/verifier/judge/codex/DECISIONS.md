# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the `.mat` files. It hard-coded seven animal IDs, then loaded one joblib-serialized file per animal from `/app/data/<animal>`. Within each loaded object, it read the `trace`, `position`, and `envs` arrays and then iterated over days and 1-minute trial windows.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal_idx, animal in enumerate(animals):
    print(f"Loading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]

    n_days = d['trace'].shape[0]
    n_cells_total = d['trace'].shape[1]
    n_timepoints = d['trace'].shape[2]
```

iii. In the trajectory, the AI first inspected the joblib files and saw keys including `trace`, `position`, `envs`, and `blocked`. It then stated its key decision that each animal file would be loaded directly and each day treated as a session. It did not justify switching away from the raw `.mat` files beyond having identified the joblib files as convenient structured data.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded animal ID list. Each loaded joblib file corresponds to one subject, and the subject name is the animal ID string.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
subjects = list(animals)
...
for animal_idx, animal in enumerate(animals):
    ...
    subject_idx_list.append(animal_idx)
```

iii. In the trajectory, the AI inspected all seven serialized animal files, printed their shapes, and then summarized the dataset as “7 animals (mice).” Its justification was that each file corresponded to one animal.

## 1-c. How are the data split into sessions?

i. Sessions are defined as recording days within each animal file. The AI iterates over `day` in the first dimension of `d['trace']`, and each day becomes one output session.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    trace_day = d['trace'][day]  # (n_cells, n_timepoints)
    pos_day = d['position'][day]  # (2, n_timepoints)
    env_name = str(d['envs'][day, 0])
```

iii. The trajectory says “Sessions = days (one recording per day)” and later “Each session = one recording day.” That was the explicit justification.

## 1-d. How are the data split into trials?

i. Within each session/day, trials are defined as consecutive non-overlapping 1-minute chunks at 30 Hz, i.e. `1800` frames each. The number of trials is `n_timepoints // 1800`, so any remainder is discarded.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
...
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL

    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. The trajectory explicitly says “Trials = 1-minute segments within each session” and notes that sessions are about 40 minutes at 30 Hz, giving about 39 to 40 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control filter. Instead, the AI skips entire days if they would produce fewer than 2 trials, and it skips days with fewer than 5 registered neurons before trial construction.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()

if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
if n_trials < 2:
    print(f"  Day {day}: skipping, only {n_trials} possible trials")
    continue
```

iii. The trajectory justification was practical rather than paper-based: it said a minimum of 2 trials was needed by the decoder format, and it treated low neuron counts as grounds to skip a day. It did not cite a paper or reference-code criterion for the `<5` neurons threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from `d['trace'][day]` in the joblib animal file.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. In the trajectory, the AI inspected the serialized files and identified `trace` as the calcium-event array, then summarized the neural source as “binary calcium transient traces.”

## 2-b. How is the `neural` data processed?

i. The AI treats `trace_day` as already session-specific neural activity with shape `(cells, time)`. It removes all-NaN cells, replaces any remaining NaNs with zero, slices the activity into 1-minute trials, and casts each trial to `float32`. It does not rebin or otherwise denoise the traces.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
...
traces = trace_day[registered]  # (n_registered, n_timepoints)
traces = np.nan_to_num(traces, nan=0.0)
...
trial_traces = traces[:, start:end]
neural_trials.append(trial_traces.astype(np.float32))
```

iii. The AI justified this in the trajectory by claiming the traces were already “binary calcium transient events” and that only registered cells should be kept. It presented the rest as formatting for the decoder rather than new signal processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is kept only if its whole-day trace is not all NaN. In addition, the AI skips a whole session/day if fewer than 5 neurons survive that filter.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()

if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue
```

iii. The trajectory explicitly discusses “registered cells (non-NaN traces)” and notes that cells are not always detected every day. It did not provide a reference-based justification for the extra `<5` neurons session filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use an experimental event. It aligns trial windows to the start of the recording session and defines trials as consecutive 1-minute recording segments.

ii.
```python
'metadata': {
    'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric deformation of environment',
    'time_bin_size': 1000.0 / FPS,
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_S,
```

iii. In the trajectory, the AI wrote that all time series were natively aligned at 30 Hz and that “Trials aligned to start of each 1-minute segment from session start.” It treated session start as the alignment event for metadata purposes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native 30 Hz frame rate, i.e. about 33.33 ms per bin. No temporal rebinning is applied.

ii.
```python
FPS = 30  # recording frame rate
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The trajectory explicitly lists “Time bin: 1/30 s ≈ 33.33 ms (native 30 Hz)” and describes the streams as already aligned at that frame rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the `envs` session labels, not from the `blocked` variable. It converts each environment name into a hand-coded 3x3 geometry matrix.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
```

iii. The trajectory shows that the AI inspected both `envs` and `blocked`, then explicitly chose “Input: 3x3 environment geometry” and said it was “Derived from `get_env_mat()` function in reference code.” It also checked that the `blocked` patterns matched the environment names before committing to this choice.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps the environment name to a fixed 3x3 matrix with `1 = open` and `0 = blocked`, then flattens that matrix into a 9-element vector. The same vector is copied into every trial within the session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    if env in env_mats:
        return np.array(env_mats[env], dtype=float)
...
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()
...
input_trials.append(env_input.astype(np.float32))
```

iii. The trajectory justification was that the decoder input should be “Environment geometry to represent which part of the arena is blocked” and that `get_env_mat` from the reference code directly supplies that structure.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from `d['position'][day]`, the 2D position trace for each day.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The trajectory identifies `position` as x-y coordinates and treats it as the raw source for the decoder target.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips x and y to the arena range, divides each axis into three equal-width bins, floors to integer bin IDs, and combines the two axis bins into one categorical label.

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS, arena_size=ARENA_SIZE):
    x = np.clip(position[0], 0, arena_size - 1e-10)
    y = np.clip(position[1], 0, arena_size - 1e-10)

    bin_size = arena_size / n_bins
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)

    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)

    bin_idx = x_bin * n_bins + y_bin
    return bin_idx
```

iii. In the trajectory, the AI justified this as matching the 3x3 partitioning of the 75 cm square arena, so that each position sample becomes one of 9 coarse spatial classes.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded at 25 cm and 50 cm, yielding 3 bins per axis. The final category index is `x_bin * 3 + y_bin`, which the AI describes as row-major ordering with x treated as the row coordinate.

ii.
```python
bin_size = arena_size / n_bins
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
...
bin_idx = x_bin * n_bins + y_bin
...
for r in range(N_SPATIAL_BINS):
    for c in range(N_SPATIAL_BINS):
        position_bin_names.append(f"row{r}_col{c}")
```

iii. The trajectory justification was that a 3x3 discretization matched the environment partitioning and created 9 position classes for the decoder. It did not discuss the axis-order choice separately.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned frame-by-frame with neural data because both come from the same day-level arrays and are sliced with identical trial start and end indices.

ii.
```python
trial_traces = traces[:, start:end]  # (n_registered, FRAMES_PER_TRIAL)
trial_pos = pos_bins[start:end]  # (FRAMES_PER_TRIAL,)

neural_trials.append(trial_traces.astype(np.float32))
output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
```

iii. The trajectory explicitly says that neural and behavioral streams were already aligned at 30 Hz and that the code applies the same 1-minute segmentation to both.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes cells with all-NaN traces, fills any remaining NaNs in the kept neural traces with zero, discards incomplete trailing frames at the end of each session by integer-dividing the frame count by 1800, and skips sessions with fewer than 5 surviving neurons or fewer than 2 trials.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
...
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
if n_trials < 2:
    ...
```

iii. The trajectory justification was mostly practical: keep only registered cells, guard against residual NaNs “for safety,” and ensure the decoder-format minimum of at least 2 trials per session.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the large per-animal `joblib.load(...)` calls, the construction of the full in-memory nested trial lists, and serializing the final 19 GB pickle. The per-trial Python loop also contributes because it touches every session and every 1-minute chunk.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_days):
    ...
    for t in range(n_trials):
        ...
        neural_trials.append(trial_traces.astype(np.float32))
        input_trials.append(env_input.astype(np.float32))
        output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
...
with open(output_path, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The trajectory did not explicitly analyze runtime, but it shows the agent treating the full conversion and full-dataset save as significant steps. This performance assessment is inferred from the implementation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit `for t in range(n_trials)` loop could have been replaced by reshaping/splitting once per session instead of slicing each trial in Python. The loops that build `position_bin_names` and `input_names_list` are minor and could also be compressed, though they are not performance critical.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
    neural_trials.append(trial_traces.astype(np.float32))
    input_trials.append(env_input.astype(np.float32))
    output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
...
for r in range(N_SPATIAL_BINS):
    for c in range(N_SPATIAL_BINS):
        position_bin_names.append(f"row{r}_col{c}")
```

iii. The trajectory did not mention vectorization. This is an assessment from the final code.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts `env_input` to `float32` once per trial even though it is constant across the session, repeatedly casts trial arrays inside the inner loop, and reruns full-dataset statistics/sanity checks after conversion rather than streaming validation during construction.

ii.
```python
input_trials.append(env_input.astype(np.float32))
...
neural_trials.append(trial_traces.astype(np.float32))
output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
...
for s in range(n_sessions):
    n_trials = len(data['neural'][s])
    for t in range(n_trials):
        ...
```

iii. The trajectory does not explicitly justify these repetitions; they appear to come from a straightforward implementation style.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints extensive summary statistics and sanity-check outputs that are not part of the saved dataset or downstream decoding inputs. It also tracks `total_neurons_per_session`, `total_sessions`, and `total_trials` mainly for reporting.

ii.
```python
total_sessions = 0
total_trials = 0
total_neurons_per_session = []
...
print(f"\n=== Conversion Summary ===")
print(f"Animals: {len(animals)}")
print(f"Sessions: {total_sessions}")
print(f"Total trials: {total_trials}")
print(f"Neurons per session: mean={np.mean(total_neurons_per_session):.1f}, "
      f"min={np.min(total_neurons_per_session)}, max={np.max(total_neurons_per_session)}")
...
def sanity_checks(data):
    ...
```

iii. The trajectory shows that the AI intentionally added conversion summaries and sanity checks to validate the output, but these computations are not required by the saved data structure itself.
