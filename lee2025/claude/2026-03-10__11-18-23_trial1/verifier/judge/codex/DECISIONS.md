# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the raw `.mat` files. It hard-coded the seven animal IDs, then loaded one preprocessed joblib file per animal from `/app/data/{animal}`. From each loaded animal dictionary it pulled session-wise arrays such as `trace`, `position`, and `envs`, and later split each session into trials in Python.

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

iii. In `CONVERSION_NOTES.md`, the AI justified this by saying the reference code uses `load_dat` via `joblib.load`, and by documenting the dataset as “Each animal stored as joblib file.” The trajectory also shows it inspected the joblib structure and then committed to using those files.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. Each joblib file is treated as one mouse, and the index of that animal in `ANIMALS` becomes the session’s `subject_idx`.

ii.
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(
        animal, show_processing=show_processing
    )

    subject_id = ANIMALS.index(animal)

    for s_idx in range(len(neural)):
        all_neural.append(neural[s_idx])
        all_input.append(inp[s_idx])
        all_output.append(out[s_idx])
        all_session_info.append(sess_info[s_idx])
        subject_idx_list.append(subject_id)
```

iii. The justification in the notes is that the data directory contains “7 mice” with those exact names, so the AI treated those filenames as the canonical subject IDs.

## 1-c. How are the data split into sessions?

i. Within each subject/joblib file, the AI treated each recording day as one session. It iterated over the first axis of `d['trace']`, and for each `day` extracted that day’s `trace`, `position`, and `env` entry.

ii.
```python
d = load_animal_data(animal)

n_days = d['trace'].shape[0]
envs = d['envs'].squeeze()

for day in range(n_days):
    env_name = envs[day]
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. `CONVERSION_NOTES.md` describes the data structure as `trace: shape (n_days, n_cells, n_frames)` and repeatedly equates “day” with a recording session.

## 1-d. How are the data split into trials?

i. Trials are made by walking through each session in non-overlapping chunks of `1800` frames (60 s at 30 Hz). Unlike the human reference, the AI keeps the final partial chunk if it is at least `900` frames long; shorter remainders are dropped.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
MIN_TRIAL_FRAMES = FPS * 30

trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start

    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The notes justify 60-second trials by the decoder instructions, and the trajectory shows the AI explicitly choosing to “skip short partial trials.” It did not justify why partial trials of 30 to 59.9 seconds should be kept instead of discarded.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two trial/session-level filters: it skips any final partial trial shorter than 30 seconds, and it drops any session that ends up with fewer than two trials.

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

iii. The notes justify the `>=2` rule by the decoder format requirement that sessions should contain at least two trials. There is no stronger reference-based justification given for the 30-second cutoff.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-day `trace` array inside each animal’s joblib dictionary.

ii.
```python
trace = d['trace'][day]
```

iii. The notes describe `trace` as the neural source variable and characterize it as binary calcium-event data already prepared by the reference pipeline.

## 2-b. How is the `neural` data processed?

i. The AI keeps the trace at native frame resolution, selects “registered” cells, slices it into trials, and casts each trial to `float32`. It does not smooth, deconvolve, rebin, or otherwise transform the values.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The notes justify this by claiming the source `trace` is already “binary calcium events (0/1)” and that no delta-F/F or other neural preprocessing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI defines usable neurons as cells whose first frame is not `NaN`. Cells failing that test are treated as unregistered for the whole session and removed.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)

if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue

trace_registered = trace[registered_mask]
```

iii. The justification in the notes is “Use ALL registered cells on each day (non-NaN trace).” The trajectory also says the agent interpreted NaN-marked cells as unregistered cells tracked across sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use a biological or task event. Instead, it defines the alignment event as the start of each artificial 1-minute segment taken from the continuous recording.

ii.
```python
'metadata': {
    'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric deformation task',
    'time_bin_size': 1000.0 / FPS,
    'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
```

iii. The notes justify this implicitly by treating trials as artificial 1-minute windows in a continuous 40-minute recording. There is no separate event-alignment argument in the notes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the data at the native 30 Hz frame rate, corresponding to `1000/30 ≈ 33.33 ms` bins, with no temporal rebinning.

ii.
```python
FPS = 30  # Recording frame rate (Hz)
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The notes explicitly say “Time bin: 30Hz native sampling (33.33ms)” and “No velocity filtering” / no rebinning for the conversion itself.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from the per-session environment label `envs`, not from the raw `blocked` field. For each session it maps the environment name to a hand-coded 3x3 binary geometry.

ii.
```python
envs = d['envs'].squeeze()
...
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()
```

iii. The notes justify this by citing the reference `get_env_mat` function and by framing the decoder input as “Environment geometry 3x3.”

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is converted into a 3x3 binary matrix via `get_env_mat`, flattened to length 9, cast to `float32`, and copied into every trial of the session as a static vector. In this representation, `1` means accessible and `0` means blocked.

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

iii. `CONVERSION_NOTES.md` explicitly states: “`get_env_mat(env)` -> flatten to 9 values” and “1=accessible, 0=blocked.”

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the per-day `position` array.

ii.
```python
position = d['position'][day]  # (2, n_frames)
```

iii. The notes describe `position` as `x,y position in cm (0-75 range)` and use it as the sole behavioral source for the decoded output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI maps continuous `(x, y)` positions to a 3x3 grid using `floor(position / bin_size)`, clips each axis to `[0, 2]`, and combines them as `x_bin * 3 + y_bin`. The result is a framewise categorical time series.

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

iii. The notes justify this as a 3x3 discretization of the 75 cm arena and explicitly say it uses “row-major ordering (`bin_idx = x_bin*3 + y_bin`).”

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded into three equal-width 25 cm bins over the 75 cm arena, with clipping at the endpoints to keep labels in `0..8`.

ii.
```python
bin_size = (env_size + buffer) / N_POS_BINS
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
```

iii. The notes justify the choice as “3x3 grid, 25 cm/bin,” matching the decoder requirement for 9 spatial bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position-derived outputs are aligned frame-by-frame within each session and are cut with the same `start:end` indices when trials are created.

ii.
```python
neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The justification is implicit in the code structure: both arrays come from the same session, at the same frame rate, and are sliced with the same trial boundaries.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI does not impute or repair missing data. It handles missingness by removing cells whose first sample is `NaN`, skipping sessions with no such registered cells, dropping short final partial trials, and skipping sessions with fewer than two surviving trials.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
...
if n_registered == 0:
    ...
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
if n_trials < 2:
    ...
    continue
```

iii. The notes justify this as using “ALL registered cells” and enforcing decoder-format requirements, but they do not describe any interpolation or finer-grained missing-data handling.

## 6-a. What are the most time-consuming steps of the code?

i. The AI treated data loading and full-dataset serialization/validation as the expensive steps. The core per-session processing is simple array slicing, while the notes emphasize multi-minute end-to-end runtime and very large output files.

ii.
```python
dat = joblib.load(filepath)
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)
...
print(f"Saved {filesize:.1f} MB in {dt:.1f}s")
```

iii. The notes report roughly 160 seconds for the full conversion and a `19,255 MB` output file, which is the AI’s main justification that I/O dominates runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI left the per-trial loop explicit. Because sessions are regular 1800-frame windows except for the tail, the full-length trials could have been reshaped/sliced in bulk instead of appending one trial at a time in Python.

ii.
```python
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

iii. The notes claim the script uses “Efficient vectorized operations,” but they do not explicitly justify keeping this Python-level trial loop.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly re-creates identical per-trial input vectors within a session, repeatedly casts those same vectors to `float32`, and does additional whole-dataset passes for summary statistics and optional plotting.

ii.
```python
env_mat = get_env_mat(env_name).flatten()
...
for start in trial_starts:
    ...
    input_trial = env_mat.astype(np.float32)
    input_trials.append(input_trial)
...
all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
```

iii. There is no explicit justification beyond convenience. The notes focus on correctness and validation rather than eliminating repeated bookkeeping work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bundled optional plotting code and plotting imports into the conversion script, collected `session_info` metadata not needed by the decoder, and computed summary/output-distribution statistics that are only printed. None of this affects the saved neural/input/output tensors used downstream.

ii.
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
...
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)
...
'session_info': all_session_info,
...
all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
```

iii. The trajectory shows these additions were meant for visualization and sanity checking, not for the final decoder input format.
