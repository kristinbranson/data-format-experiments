# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded the seven animal IDs, loaded one joblib file per animal from `data/`, extracted the per-animal record with keys like `envs`, `trace`, `position`, and `blocked`, and then iterated over all days in that record. Trials were not loaded natively from disk; they were created later by splitting each day/session into 1-minute windows.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]
```

```python
for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    envs = np.array(rec['envs']).reshape(-1)
    trace = np.asarray(rec['trace'])
    position = np.asarray(rec['position'])
    blocked = rec['blocked']
    n_days = len(envs)
```

iii. In `CONVERSION_NOTES.md`, the agent justified using the native joblib files because the reference `load_dat` function “directly consumes the joblib files; no need to recompute from MATLAB.” The trajectory also records that the agent inspected `load_dat` and concluded the correct source was the per-animal joblib files rather than rebuilding from `.mat`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the outer loop over the hard-coded animal IDs. Each animal name becomes one subject string in `subjects`, and every session derived from that animal gets the corresponding integer in `subject_idx`.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
data = {
    ...
    'subjects': animals.copy(),
    'subject_idx': [],
    ...
}
```

```python
for subj_idx, animal in enumerate(animals):
    ...
    data['subject_idx'].append(subj_idx)

data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. The notes say “Subjects from animal IDs” and “Brain region is CA1 for all neurons,” and Step 2 documented that the dataset is organized as one file per animal. The trajectory shows the agent treated the seven animal files as the seven mice.

## 1-c. How are the data split into sessions?

i. Sessions are defined as recording days within each animal file. The agent flattened `envs`, used its length as `n_days`, and iterated over `day` indices; each `day` becomes one session.

ii.
```python
envs = np.array(rec['envs']).reshape(-1)
...
n_days = len(envs)

for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
```

iii. In `CONVERSION_NOTES.md`, the agent documented that “Sessions correspond to recording days / environments within each animal file,” and the trajectory records that it inferred from the reference data and methods that there is “one session recorded per day.”

## 1-d. How are the data split into trials?

i. The agent imposed a trial structure by chopping each continuous session into contiguous 60-second windows at 30 Hz. It computed `FRAMES_PER_TRIAL = 1800`, took only the largest multiple of 1800 frames, and emitted one trial per chunk.

ii.
```python
FPS = 30
TRIAL_SECONDS = 60
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
```

```python
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

iii. The notes explicitly justify this as an adaptation to the task: “Native data have no trial structure; task specification requires at least two trials per session, so each day/session will become ~40 contiguous 1-minute trials.”

## 1-e. How are trials filtered based on quality controls?

i. There is no real per-trial quality control. The only trial-level filtering is implicit: trailing partial minutes are dropped because `n_trials` uses floor division, and any session with fewer than two full 1-minute trials is skipped.

ii.
```python
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
```

```python
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The notes acknowledge that there is “No explicit trial concept in native data” and frame the minute chunks as derived trials rather than reference-native units. I did not find a stronger trial-QC justification in Step 6; the absence of trial filtering appears to be a default consequence of the implementation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken directly from the raw `trace` variable for each animal and day.

ii.
```python
trace = np.asarray(rec['trace'])
...
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. The notes repeatedly justify this by citing the reference methods: the stored `trace` is already the binarized rising-phase calcium event vector and “should be used as neural data; do not compute dF/F.”

## 2-b. How is the `neural` data processed?

i. The agent converts each day’s trace array to `float32`, drops neuron rows with any non-finite values, replaces any remaining non-finite values with zero, slices the session into contiguous 1-minute windows, and finally casts each trial to `uint8`. It does not recompute event extraction, smooth traces, or apply temporal rebinning before saving.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
```

```python
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The main justification in the notes is that the reference paper/code already treat the stored binary rise-event trace as the analysis signal, so the agent believed no dF/F computation or event extraction was needed. The trajectory also records a deliberate choice to “preserve native aligned framewise traces” rather than store a decoder-prebinned representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC in `convert_data.py` is removing day-specific neuron rows with any non-finite values. The code does not apply place-cell selection, the reference decoder’s active-cell threshold, or the reference decoder’s velocity-based frame filter.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes justify skipping place-cell restriction by saying “The paper’s decoding code filters by velocity and cell activity, not place-cell status, and the task asks to decode from neural activity broadly.” That explains why place-cell curation was omitted, but I did not find a justification for also omitting the reference decoder’s active-cell and velocity filters in the saved dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no behavioral event alignment. Neural data stay in native frame order and are aligned to session start, with each trial defined as a contiguous 1-minute chunk from the beginning of the session.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
    'off_start': 0.0,
    'off_end': float(TRIAL_SECONDS),
    ...
}
```

```python
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The notes justify this by pointing out that the source experiment is continuous free exploration with no native trial onset event. The trajectory says the agent chose to “split each 40 min session into contiguous 1-minute trials” while preserving native frame alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved data are at the native 30 Hz frame rate, i.e. `1000 / 30 = 33.33 ms` per time bin. No temporal rebinning is applied before saving.

ii.
```python
FPS = 30
```

```python
'metadata': {
    ...
    'time_bin_size': 1000.0 / FPS,
    ...
}
```

```python
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. `CONVERSION_NOTES.md` says “Preserve native 30 Hz frame alignment in converted data,” and the trajectory explicitly contrasts that with the reference decoder’s later 3-frame temporal binning. The agent chose not to bake the reference decoder’s ~100 ms binning into the stored dataset.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the raw `blocked` field, one entry per day/session.

ii.
```python
blocked = rec['blocked']
...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. The notes say “Use blocked geometry as decoder input” and Step 2 documented that `blocked` stores the blocked partition indices, with `-1` meaning no blocked partition.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code turns each raw `blocked` entry into a length-9 binary vector representing the 3x3 partition grid. Nested arrays are flattened, `NaN` values are ignored, `-1` produces an all-zero vector, and valid indices 0 through 8 are set to 1.

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

```python
input_trials.append(blocked_vec.copy())
```

iii. The justification comes from the code README and the notes: the reference dataset organizes blocked partitions over a 3x3 design `[[0,1,2],[3,4,5],[6,7,8]]`, and the task asks for “Environment geometry to represent which part of the arena is blocked. Static per-trial.”

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The decoder output is derived directly from the raw `position` variable for each animal and day.

ii.
```python
position = np.asarray(rec['position'])
...
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The notes and trajectory both state that `position` is the aligned behavioral stream used by the reference code, acquired simultaneously with calcium imaging at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code converts `position` to `float32`, optionally zero-fills any non-finite values, truncates it to match neural length, converts continuous `x` and `y` coordinates into a 3x3 grid over a 75 cm arena, and stores the resulting class index for every frame.

ii.
```python
ARENA_SIZE_CM = 75.0
N_POS_BINS = 3
```

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

```python
pos_day = np.asarray(position[day], dtype=np.float32)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_bins = position_to_bins_3x3(pos_day)
```

iii. The notes justify the 75 cm arena and 3x3 discretization using the methods text plus the decoder task specification. The trajectory explicitly says this was an “adapted” coarsening of the reference decoder’s finer spatial binning.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is categorized by equally partitioning the continuous `x` and `y` coordinates into three bins each, then flattening the 2D bin index to one label from 0 to 8 via `ybin * 3 + xbin`.

ii.
```python
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) / (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
xbin = np.clip(xbin, 0, N_POS_BINS - 1)
ybin = np.clip(ybin, 0, N_POS_BINS - 1)
return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

```python
'output_names': ['position_bin_3x3'],
'output_values': [[f'bin_{i}' for i in range(9)]],
```

iii. The justification is direct: the task required “Mouse position discretized into 3 x 3 = 9 spatial bins,” and the notes describe this as a coarser version of the paper’s within-session position decoding target.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned framewise to neural data by truncating both streams to the same minimum length within a session, then splitting them with the same trial boundaries. The output keeps one categorical position label per neural frame.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
```

```python
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The notes justify this by citing the methods statement that behavioral and cellular streams were acquired simultaneously at 30 Hz. The trajectory says the agent’s goal was to “preserve native frame alignment when splitting into 1-minute trials.”

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses pragmatic cleanup rules rather than reference-specific imputation. Non-finite neuron rows are dropped entirely for that day, any remaining `NaN`/`inf` values in `trace` or `position` are replaced with zero, position/neural length mismatches are resolved by truncating to the minimum length, and leftover partial trials are discarded.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
```

```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
```

iii. The notes mention that initial validation errors came from “non-finite neural values from unregistered cells; fixed by dropping non-finite neurons per session/day.” I did not find an explicit justification for zero-filling position values; that appears to be a generic defensive cleanup step.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the large per-animal joblib files, materializing per-day/per-trial arrays for the full dataset, and writing the multi-gigabyte pickle. Optional processing plots also add cost when enabled.

ii.
```python
for subj_idx, animal in enumerate(animals):
    print(f'Loading {animal}...')
    rec = load_animal(data_dir, animal)
```

```python
for day in range(n_days):
    ...
    neural_trials, input_trials, output_trials, n_frames, used, n_trials = split_session_into_trials(
        trace_day, pos_day, blocked_vec
    )
```

```python
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. Step 6 of `CONVERSION_NOTES.md` is effectively blank on runtime analysis, so this is inferred from the implementation and the produced `converted_data.pkl` size. The code itself contains no alternative optimized path beyond a small `--sample` mode.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the Python loops in `blocked_to_vec`, the per-trial loop in `split_session_into_trials`, and some repeated per-session bookkeeping like repeated string flattening and metadata assembly. Trial splitting could have been done with reshape/view logic instead of appending one trial at a time.

ii.
```python
for item in flat:
    arr = np.array(item).reshape(-1)
    for v in arr:
        if np.isnan(v):
            continue
        vals.append(int(v))
```

```python
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    input_trials.append(blocked_vec.copy())
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. There is no explicit efficiency justification in the notes. The agent’s Step 6 section does not document any vectorization work, so the retained loops appear to be simplicity-first choices.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats some avoidable work: `flatten_env_name(envs[day])` is called multiple times per session, `position_to_bins_3x3` is recomputed in the optional plotting branch after already being computed inside trial splitting, and the static blocked vector is copied once per trial even though it never changes within a session.

ii.
```python
session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
...
'environment': flatten_env_name(envs[day]),
```

```python
pos_bins = position_to_bins_3x3(pos_day)
...
if show_processing and plotted < 2:
    pos_bins = position_to_bins_3x3(pos_day[:, :used])
```

```python
input_trials.append(blocked_vec.copy())
```

iii. I did not find an explicit rationale for these repeated computations in the notes or trajectory. They appear to be incidental consequences of a straightforward implementation rather than deliberate choices.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional plotting path computes occupancy histograms and figures that are not used by the converted dataset. Even in the main path, the code converts traces to `float32`, runs `nan_to_num`, and then casts saved trials to `uint8`, so some intermediate numeric work is discarded; similarly, the large `session_info` metadata block is useful for logging but not for decoder inputs/outputs.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

```python
if show_processing and plotted < 2:
    pos_bins = position_to_bins_3x3(pos_day[:, :used])
    make_processing_plot(...)
```

```python
data['metadata']['session_info'] = session_info
```

iii. Step 6 does not document a conscious tradeoff here. These look like convenience/debugging choices rather than processing required by the downstream decoder.
