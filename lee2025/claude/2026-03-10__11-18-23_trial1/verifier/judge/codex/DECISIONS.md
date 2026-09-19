# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded the seven animal IDs in `ANIMALS`, then loaded one preprocessed joblib file per animal from `/app/data/{animal}` with `joblib.load`. It did not scan `.mat` files or use `h5py`; instead it loaded the per-animal Python dictionary and then processed sessions and trials from that structure.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_animal_data(animal):
    """Load preprocessed data for one animal from joblib file."""
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]
```

iii. In `CONVERSION_NOTES.md`, the agent justified this by saying dataset exploration showed each animal was stored as a joblib file containing `trace`, `position`, `envs`, and `blocked`, and later explicitly claimed `joblib.load(f'/app/data/{animal}')` matched the reference loader.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by animal ID. The script iterates through the fixed `ANIMALS` list, processes one loaded animal at a time, and uses the position of that animal in `ANIMALS` as the subject index for all of its sessions.

ii.
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(
        animal, show_processing=show_processing
    )

    subject_id = ANIMALS.index(animal)

    for s_idx in range(len(neural)):
        all_neural.append(neural[s_idx])
        ...
        subject_idx_list.append(subject_id)
```

iii. The notes repeatedly describe the dataset as "7 mice" stored per animal, and Step 2 documents "Each animal stored as joblib file: `data/{animal_id}`".

## 1-c. How are the data split into sessions?

i. Sessions are the per-day recordings inside each animal file. The script treats the first axis of `d['trace']` as days/sessions and processes each `day` as a separate output session.

ii.
```python
d = load_animal_data(animal)

n_days = d['trace'].shape[0]
...
for day in range(n_days):
    env_name = envs[day]
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. In the notes, the agent states that each joblib file contains arrays with shape `(n_days, ...)`, lists session counts per subject, and refers to each day as a session throughout the sanity checks.

## 1-d. How are the data split into trials?

i. Each continuous session is segmented into consecutive 1-minute windows starting every 1800 frames. Unlike the human reference, the code keeps a final partial segment if it is at least 30 seconds long; shorter remainders are dropped.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
MIN_TRIAL_FRAMES = FPS * 30

trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
...
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start

    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The notes say the intended trial definition was 1-minute segments, but they also explicitly justify dropping "partial trials <900 frames (30s)" and record that the last trial can be shorter when a session length is not divisible by 1800.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral or neural quality-control filter at the trial level. The only trial-level filtering is duration based: segments shorter than 30 seconds are discarded, and sessions with fewer than 2 retained trials are skipped.

ii.
```python
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
n_trials = len(neural_trials)
if n_trials < 2:
    print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
    continue
```

iii. The notes justify the `>=2` trial rule by citing the decoder requirement, and they describe short-partial-trial removal as "correct behavior." They also say "No trial filtering," which does not fully match the implemented code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is taken from the `trace` variable in the loaded per-animal dictionary, specifically `d['trace'][day]` for each session.

ii.
```python
trace = d['trace'][day]
```

iii. Step 2 of the notes identifies `trace` as the neural variable and describes it as binary calcium events with shape `(n_days, n_cells, n_frames)`.

## 2-b. How is the `neural` data processed?

i. The script keeps the session-day trace in its existing `(cells, frames)` orientation, filters to "registered" cells, and casts each trial slice to `float32`. It does not rebin, smooth, deconvolve, or otherwise transform the values.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The notes justify this by stating that the data is "already preprocessed: binary calcium trace (0/1 for significant events)" and that the planned mapping was "Raw binary trace (0/1), only registered cells per session."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control step is removing cells considered unregistered on a given day. The code infers this from whether the first frame is `NaN` for a cell and keeps all remaining cells; it does not apply place-cell or activity-based filtering.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)

if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue

trace_registered = trace[registered_mask]
```

iii. The notes justify this as "Use ALL registered cells on each day (non-NaN trace). No place-cell filtering," and later claim this first-frame mask is the same approach as the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental-event alignment. Neural data is aligned to the start of each artificial 1-minute segment created from the continuous session.

ii.
```python
'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
...
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
```

iii. The notes frame trial segmentation as the key temporal structure and do not identify any stimulus or task event to align to.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz sampling rate, corresponding to 33.33 ms bins. No temporal rebinning or resampling is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes explicitly state "Time bin: 30Hz native sampling (33.33ms)."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the session's environment identity string in `d['envs']`, not from the `blocked` variable. The code uses the environment name to look up a 3x3 geometry template.

ii.
```python
envs = d['envs'].squeeze()
...
env_name = envs[day]
...
env_mat = get_env_mat(env_name).flatten()
```

iii. The notes justify this choice by mapping "Environment geometry 3x3" to `get_env_mat(env)` from the reference code and describing the input as a 3x3 binary geometry representation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is converted to a hard-coded 3x3 binary matrix using `get_env_mat`, then flattened into a length-9 vector and reused as a static per-trial input.

ii.
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    ...

env_mat = get_env_mat(env_name).flatten()
...
input_trial = env_mat.astype(np.float32)
```

iii. The notes say this comes from the reference `get_env_mat` function and justify it as "1=accessible, 0=blocked" with the input being static per trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the per-session `position` array in the loaded animal dictionary.

ii.
```python
position = d['position'][day]  # (2, n_frames)
```

iii. The notes identify `position` as the source of mouse location and describe it as x,y position in cm over time.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code converts the continuous 2D position into 3 bins per axis over a 75 cm arena using floor-based binning, clips indices to the valid range, and combines the per-axis bins into one categorical label per frame.

ii.
```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx
```

iii. The notes justify this as "Bin position into 3x3 grid (25cm/bin), integer 0-8," using the native 75x75 cm arena.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into 9 categories by assigning each frame to one of three x-bins and three y-bins, then encoding the class as `x_bin * 3 + y_bin` (the agent calls this row-major and labels outputs as `x{r}y{c}`).

ii.
```python
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin

output_values = [
    [f"x{r}y{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]
]
```

iii. In Step 5, the notes explicitly justify the category convention as "row-major ordering (bin_idx = x_bin*3 + y_bin)."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position categories are computed on the same per-frame timeline as the neural traces, then sliced with the exact same trial start and end indices as the neural data.

ii.
```python
pos_bins = bin_position_3x3(position)
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes justify this by treating the trace and position streams as native 30 Hz recordings from the same session and reporting spot checks where converted position bins matched the source data exactly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable cells are handled by dropping cells whose first frame is `NaN`, and entire sessions are skipped if no such cells remain. Short trailing trial fragments under 30 seconds are dropped; longer partial trailing trials are kept. The code does not interpolate missing position data or repair corrupted values.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
...
if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue
...
if trial_len < MIN_TRIAL_FRAMES:
    continue
```

iii. The notes justify this as using "ALL registered cells on each day," dropping only short partial trials, and otherwise treating the preprocessed source data as already clean enough for decoding.

## 6-a. What are the most time-consuming steps of the code?

i. The code is organized as if the heavy steps are per-animal data loading and the per-session/trial processing loop, with optional plotting adding more cost in `--show-processing` mode.

ii.
```python
t0 = time.time()
print(f"\nProcessing {animal}...")
d = load_animal_data(animal)
...
for day in range(n_days):
    t_day = time.time()
    ...
    for start in trial_starts:
        ...
    dt = time.time() - t_day
    print(f"  Day {day} ({env_name}): {n_registered} cells, {n_trials} trials, {dt:.1f}s")
...
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)
```

iii. The notes mainly justify this indirectly through timing instrumentation and runtime estimates; they do not name a single bottleneck beyond emphasizing efficiency and measured run time.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still uses explicit Python loops over animals, days, sessions, and trial starts. The most obvious vectorization opportunity is trial splitting: full-length 1800-frame trials could have been reshaped/split in bulk instead of being appended one by one.

ii.
```python
for day in range(n_days):
    ...
    trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
    neural_trials = []
    input_trials = []
    output_trials = []

    for start in trial_starts:
        end = min(start + FRAMES_PER_TRIAL, n_frames)
        ...
        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
```

iii. The only explicit justification in the notes is the claim that the script uses "Efficient vectorized operations (no inner loops for trace/position)," even though trial formation itself remains loop based.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts the same static environment vector to `float32` for every trial, repeatedly appends identical per-trial environment inputs within a session, and later recomputes a full concatenated output distribution just for summary printing.

ii.
```python
env_mat = get_env_mat(env_name).flatten()
...
input_trial = env_mat.astype(np.float32)
...
input_trials.append(input_trial)
...
all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
```

iii. There is no explicit justification for these repetitions beyond the general Step 6 emphasis on producing plots, summaries, and readable processing logic.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional visualization (`plot_processing`), timing/summary reporting, and the end-of-run output-class distribution calculation are not used by downstream decoder training. The stored `session_info` metadata is also ancillary to decoding.

ii.
```python
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)
...
print("\n=== Summary ===")
...
all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
...
'session_info': all_session_info,
```

iii. The notes justify these extras as validation, sanity checking, and documentation support rather than as part of the downstream analysis itself.
