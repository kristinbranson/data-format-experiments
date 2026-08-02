# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the per-animal joblib files in `data/`, not from the `.mat` files. It uses a hard-coded list of seven animal IDs, loads one joblib file per animal, then iterates over each animal's recording days and later splits each day into trials.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]

for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    envs = np.array(rec['envs']).reshape(-1)
    trace = np.asarray(rec['trace'])
    position = np.asarray(rec['position'])
    blocked = rec['blocked']
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says: "Use native joblib files as source: `load_dat` in reference code directly consumes the joblib files; no need to recompute from MATLAB."

## 1-b. How are the data split into subjects?

i. The AI treats each animal ID in the hard-coded `ANIMALS` list as one subject. `subjects` is initialized from that list, and `subject_idx` stores the corresponding subject index for each session.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
data = {
    ...
    'subjects': animals.copy(),
    'subject_idx': [],
    ...
}

for subj_idx, animal in enumerate(animals):
    ...
    data['subject_idx'].append(subj_idx)
```

iii. The notes say the dataset is organized "per animal" and list the seven animal datasets in `data/`.

## 1-c. How are the data split into sessions?

i. Within each animal file, each recording day / environment entry is treated as one session. The AI iterates over `envs` and uses the same day index into `trace`, `position`, and `blocked`.

ii.
```python
envs = np.array(rec['envs']).reshape(-1)
trace = np.asarray(rec['trace'])
position = np.asarray(rec['position'])
blocked = rec['blocked']
n_days = len(envs)

for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
    blocked_vec = blocked_to_vec(blocked[day])
    trace_day = np.asarray(trace[day], dtype=np.float32)
    pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The notes say: "Sessions correspond to recording days / environments within each animal file."

## 1-d. How are the data split into trials?

i. Each continuous session is split into contiguous, non-overlapping 1-minute trials at 30 Hz. The AI uses `1800` frames per trial, discards trailing partial trials, and uses the same trial boundaries for neural and position data.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS

def split_session_into_trials(trace_day, pos_day, blocked_vec):
    n_frames = min(trace_day.shape[1], pos_day.shape[1])
    n_trials = n_frames // FRAMES_PER_TRIAL
    used = n_trials * FRAMES_PER_TRIAL
    trace_day = trace_day[:, :used]
    pos_day = pos_day[:, :used]
    ...
    for i in range(n_trials):
        s = i * FRAMES_PER_TRIAL
        e = (i + 1) * FRAMES_PER_TRIAL
        neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
        input_trials.append(blocked_vec.copy())
        output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The notes say: "Derive 1-minute trials from continuous 40 min sessions" and "splitting each 40 min session into 1-minute chunks while preserving frame alignment."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter individual trials by behavioral or neural quality. Its only session/trial-count filter is to skip any session with fewer than two full 1-minute trials.

ii.
```python
neural_trials, input_trials, output_trials, n_frames, used, n_trials = split_session_into_trials(
    trace_day, pos_day, blocked_vec
)
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The justification comes from the task format requirement, reflected in the notes: converted data must have at least two trials per session for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the raw `trace` field inside each animal joblib file.

ii.
```python
trace = np.asarray(rec['trace'])
...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. The notes state that the native neural data are rise-extracted calcium event traces and that the conversion should "Use `trace` as neural activity."

## 2-b. How is the `neural` data processed?

i. The AI assumes `trace[day]` is already organized as neurons by time. It casts each session to `float32`, drops invalid neurons, fills any remaining non-finite values with zero, then slices trials and finally stores each trial as `uint8`.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The notes justify using the native trace directly: "Use `trace` as neural activity" and "Reference methods/code treat the binarized rising-phase vector as firing rate for all analyses." The later `uint8` cast was added after sample validation to reduce file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removes any neuron row that is not finite at every timepoint in that session. This is stricter than removing only all-NaN neurons.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. `CONVERSION_NOTES.md` says: "unregistered/non-finite neurons are removed on a per-session basis." The trajectory shows this was added after verification failures from non-finite neural values.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The AI treats each session as continuous and aligns trials to session start by cutting contiguous 1-minute windows.

ii.
```python
'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The notes explicitly say: "Preserve native 30 Hz frame alignment" and "Native data have no trial structure; trials will be derived by splitting each session into 1-minute chunks."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native 30 Hz sampling, corresponding to `33.333... ms` bins. It does not rebin in time during conversion.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes say: "Preserve native 30 Hz frame alignment in converted data" and describe no temporal downsampling in conversion.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from the raw `blocked` field, interpreting it as the blocked geometry / blocked partitions of the environment.

ii.
```python
blocked = rec['blocked']
...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. The notes say: "Use blocked geometry as decoder input" and describe `blocked` as the per-day nested list/array of blocked partition indices.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI flattens the nested `blocked` entry, ignores `NaN`s, interprets `-1` as no blocked partition, and converts the result into a 9-dimensional binary vector.

ii.
```python
def blocked_to_vec(blocked_entry):
    vec = np.zeros(9, dtype=np.float32)
    flat = np.array(blocked_entry, dtype=object).reshape(-1)
    vals = []
    for item in flat:
        arr = np.array(item).reshape(-1)
        for v in arr:
            if np.isnan(v):
                continue
            vals.append(int(v))
    if len(vals) == 1 and vals[0] == -1:
        return vec
    for v in vals:
        if 0 <= v <= 8:
            vec[v] = 1.0
    return vec
```

iii. The notes justify the representation as "Encode blocked partitions as 9-d binary vector over 3x3 arena partitions; static per trial."

## 3-c. How is `input` *Environment geometry* aligned with the neural data?

i. The input is treated as static within a session. The same blocked vector is copied into every derived trial, with no time dimension.

ii.
```python
for i in range(n_trials):
    ...
    input_trials.append(blocked_vec.copy())
```

iii. The notes call blocked geometry "static contextual information" and say it should be represented as a per-trial constant.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives the decoder output from the raw `position` field for each session/day.

ii.
```python
position = np.asarray(rec['position'])
...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The notes say: "Use 3x3 discretized position as decoder output."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips `x` and `y` to the 75 cm arena, divides each axis into 3 equal bins using floor division, and maps the `(x_bin, y_bin)` pair to a single label `0..8`.

ii.
```python
def position_to_bins_3x3(position_xy):
    pos = np.asarray(position_xy, dtype=np.float32)
    x = pos[0]
    y = pos[1]
    eps = 1e-6
    xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
    xbin = np.clip(xbin, 0, N_POS_BINS - 1)
    ybin = np.clip(ybin, 0, N_POS_BINS - 1)
    return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. The notes say: "Use 3x3 discretized position as decoder output: required by task."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. It is thresholded into 9 categorical bins by dividing each spatial axis into thirds and flattening the 3x3 grid into class labels `0` through `8`.

ii.
```python
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
...
return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. The notes justify this as the task-required coarse discretization of position.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI first trims neural and position streams to the same frame count using the shorter of the two, then applies the same trial boundaries to both.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
pos_bins = position_to_bins_3x3(pos_day)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The notes say that position and neural traces are already aligned framewise at 30 Hz and that trial splitting should preserve that alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native 30 Hz frame rate, so the time bin size is `1000/30 = 33.333... ms`. No temporal rebinning is applied in the conversion script.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The notes repeatedly say the conversion should preserve the native 30 Hz frame alignment.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position data are aligned frame-for-frame within each session after trimming to a shared length and cutting identical 1-minute windows. The blocked-geometry input is constant per trial and is copied once per trial.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
...
input_trials.append(blocked_vec.copy())
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The notes justify this with "Preserve native 30 Hz frame alignment" and "blocked geometry" as static trial-level context.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI handles nested/malformed `blocked` entries by flattening them and ignoring `NaN`s. It handles non-finite neural and position values with `np.nan_to_num`, drops non-finite neurons, trims neural/position streams to the shorter length, discards trailing incomplete trials, and would skip sessions with fewer than two full trials.

ii.
```python
flat = np.array(blocked_entry, dtype=object).reshape(-1)
...
if np.isnan(v):
    continue
...
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
...
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
```

iii. The trajectory shows the non-finite handling was added after verification failed on sample data. The notes describe this as fixing "non-finite neural values from unregistered cells."

## 7-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is loading the large joblib animal files and iterating through all animal/day sessions to slice long framewise arrays into many trial arrays. Optional plotting also adds extra cost.

ii.
```python
for subj_idx, animal in enumerate(animals):
    print(f'Loading {animal}...')
    rec = load_animal(data_dir, animal)
    ...
    for day in range(n_days):
        ...
        neural_trials, input_trials, output_trials, n_frames, used, n_trials = split_session_into_trials(
            trace_day, pos_day, blocked_vec
        )
```

iii. The notes and trajectory emphasize runtime and file-size concerns around full conversion and sample/full validation, but they do not give a deeper explicit performance analysis.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several Python loops in place: nested loops in `blocked_to_vec`, per-trial append loops in `split_session_into_trials`, and the outer subject/day loops. The trial slicing could have been reshaped in bulk, and blocked-vector extraction could have been flattened more directly.

ii.
```python
for item in flat:
    arr = np.array(item).reshape(-1)
    for v in arr:
        ...

for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    input_trials.append(blocked_vec.copy())
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. There is no explicit justification in the notes beyond straightforward implementation convenience.

## 7-c. What processing does the code repeat multiple times?

i. The code repeatedly computes environment strings with `flatten_env_name`, copies the same blocked vector once per trial, and recomputes position bins for plotting even after already computing them during trial splitting.

ii.
```python
session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
...
'environment': flatten_env_name(envs[day]),
...
input_trials.append(blocked_vec.copy())
...
pos_bins = position_to_bins_3x3(pos_day[:, :used])
make_processing_plot(...)
```

iii. The notes do not explicitly justify these repeats; they appear to be convenience choices for metadata and diagnostics.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code can generate diagnostic PNG plots, stores extensive `session_info` metadata that the decoder does not use, keeps human-readable environment strings only for metadata, and stores neural/output as `uint8` even though the downstream verifier immediately warns that neural data will be converted to `float32` during training.

ii.
```python
if show_processing and plotted < 2:
    ...
    make_processing_plot(...)

data['metadata']['session_info'] = session_info
data['metadata']['conversion_runtime_sec'] = time.time() - t0
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The plotting is explicitly justified by the `--show-processing` option. The extra metadata and dtype optimization were added for inspection and file-size reasons, not because the decoder requires them.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6. The AI flattens malformed `blocked` entries, ignores `NaN`s there, fills non-finite neural and position values with zero, drops non-finite neurons, trims to a common frame count, discards remainder frames, and would skip sessions with fewer than two trials.

ii.
```python
if np.isnan(v):
    continue
...
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
...
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
```

iii. Same justification as 6: the trajectory shows this handling was added after validation exposed non-finite data.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a. The dominant cost is loading the large per-animal joblib files and then iterating through sessions to split long recordings into many trial arrays.

ii.
```python
for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    ...
    for day in range(n_days):
        neural_trials, input_trials, output_trials, n_frames, used, n_trials = split_session_into_trials(
            trace_day, pos_day, blocked_vec
        )
```

iii. The notes mention conversion runtime and file-size concerns, but not a more formal performance analysis.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b. The AI keeps Python loops for blocked-vector parsing and per-trial list construction that could have been replaced with more vectorized array operations.

ii.
```python
for item in flat:
    ...
    for v in arr:
        ...

for i in range(n_trials):
    ...
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. No explicit justification is given beyond implementation simplicity.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c. The AI repeats environment-name flattening, blocked-vector copying, and position-bin computation for plotting.

ii.
```python
flatten_env_name(envs[day])
...
input_trials.append(blocked_vec.copy())
...
pos_bins = position_to_bins_3x3(pos_day[:, :used])
```

iii. No explicit justification beyond convenience and diagnostics.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d. Optional plots, detailed `session_info`, and some metadata are not used by downstream decoder training; neural `uint8` storage is also undone later by the training script's conversion warnings.

ii.
```python
if show_processing and plotted < 2:
    make_processing_plot(...)

data['metadata']['session_info'] = session_info
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The explicit justifications are debugging/inspection (`--show-processing`) and storage reduction, not decoder necessity.
