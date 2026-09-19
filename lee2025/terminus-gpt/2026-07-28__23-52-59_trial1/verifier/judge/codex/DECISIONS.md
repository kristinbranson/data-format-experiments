# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not scan `.mat` files. It hard-codes seven animal IDs, loads one joblib file per animal with `joblib.load`, then iterates through each animal's day/session arrays and later splits each day into 1-minute trials.

ii.
```python
ANIMAL_IDS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]

for animal_id in animal_ids:
    animal = load_animal(animal_id, data_dir)
    envs = np.array(animal['envs']).squeeze()
    traces = np.asarray(animal['trace'])
    positions = np.asarray(animal['position'])
    blocked = animal['blocked']
```

iii. In `CONVERSION_NOTES.md` and the trajectory, the AI justified this by saying the reference notebooks used `load_dat(..., format="joblib")`, that the per-animal joblib structure exposed `trace`, `position`, `envs`, and `blocked`, and that these files were the "natural mapping" for conversion.

## 1-b. How are the data split into subjects?

i. Subjects are the seven hard-coded animal IDs. Each loaded joblib file is treated as one mouse, and `subjects` is initialized directly from that list.

ii.
```python
animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_to_idx[animal_id])
```

iii. The AI's notes say the data directory and `behav_dict` contained seven animal IDs and that "one animal = one subject" was the natural subject split.

## 1-c. How are the data split into sessions?

i. Each day within an animal file is treated as one session. The first dimension of `trace` is taken as `n_days`, and each `day_idx` becomes one output session if it yields at least two full 1-minute trials.

ii.
```python
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
        ...
    )
    if len(neural_trials) < 2:
        continue
    data['neural'].append(neural_trials)
    data['input'].append(input_trials)
    data['output'].append(output_trials)
```

iii. The trajectory repeatedly states that "sessions = recording days" because the paper/notes described one recording session per day and the joblib arrays were organized by day.

## 1-d. How are the data split into trials?

i. Each session/day is split into contiguous non-overlapping 60 s trials at 30 Hz, so each trial contains 1800 frames. Trailing leftover frames are dropped.

ii.
```python
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 60.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)
...
n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The AI explicitly described this in the trajectory as a task-driven deviation from the paper's continuous sessions: the decoder instructions required 1-minute trials, and 30 Hz implied 1800 frames per trial.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filter. In practice, the AI drops incomplete trailing chunks by only keeping full 1800-frame trials, and it skips an entire session if that session produces fewer than two full trials.

ii.
```python
n_full_trials = n_frames // FRAMES_PER_TRIAL
...
if len(neural_trials) < 2:
    continue
```

iii. The session-level `len(neural_trials) < 2` rule is not explained in the notes beyond satisfying the decoder-format requirement that each session must contain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-animal `trace` array loaded from the joblib files.

ii.
```python
traces = np.asarray(animal['trace'])
...
def process_day(day_trace, day_pos, env_name, blocked_entry, ...):
    valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
    trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The notes say `trace` is the analysis-ready neural signal, described as "binary rising-phase calcium trace events" / rise-extracted calcium traces used directly by the original analysis.

## 2-b. How is the `neural` data processed?

i. For each day/session, the AI removes neurons that are all NaN across time, replaces any remaining NaNs with 0, casts to `float32`, truncates the neural matrix to match the shorter of neural/position frame counts, and then slices it into 1800-frame trials. It does not perform further temporal binning or deconvolution.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
...
neural_trials.append(trace[:, s:e])
```

iii. The notes and trajectory justify using the provided `trace` directly because the paper/methods described the binary rising-phase vector as the signal used for downstream analyses. The zero-filling of non-all-NaN missing values was not separately justified.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is dropping neurons that are entirely NaN within a day/session.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The trajectory describes this as "filtering neurons with all-NaN on a day," motivated by session-specific inactive/unregistered cells in the across-day animal arrays.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. Trials are aligned to session/day start and then chunked into contiguous 1-minute windows. The metadata explicitly names the alignment event as session/day start.

ii.
```python
'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials',
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The notes/trajectory say the source data are continuous sessions with framewise aligned neural and behavioral streams, so the AI used session start as the only alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native 30 Hz frame rate, i.e. `1000 / 30` ms per sample. No temporal rebinning is applied beyond chopping the stream into 60 s trials.

ii.
```python
FRAME_RATE_HZ = 30.0
...
'time_bin_size': 1000.0 / FRAME_RATE_HZ,
```

iii. The AI's planning notes explicitly infer a 30 Hz common time base from the paper/data and state that the framewise signal should be preserved.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives `input` from two raw variables in the joblib animal files: `envs` and `blocked`. `envs` supplies a named arena shape and `blocked` supplies blocked partitions.

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
...
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The notes say the reference code defined environment-shape helpers and that the dataset documented both `envs` and `blocked`, so the AI concluded the decoder input should be a "3x3 environment geometry/block mask."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the environment name into a hard-coded 3x3 occupancy mask, converts the blocked entry into another 3x3 mask where blocked cells are zeroed, multiplies the two masks elementwise, flattens the result to length 9, and repeats that same static vector for every trial in the session.

ii.
```python
def env_to_mask(env_name):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    ...

def blocked_to_mask(entry):
    ...
    mask = np.ones((3, 3), dtype=np.float32)
    ...
    for v in vals:
        ...
        mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)

mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
...
input_trials.append(mask.copy())
```

iii. The trajectory describes this as the "natural mapping" from named shapes plus blocked partitions to a static per-trial geometry input. The agent appears to have taken this from notebook/source helpers referenced in its notes rather than from the human reference converter.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the per-animal `position` array.

ii.
```python
positions = np.asarray(animal['position'])
...
def process_day(day_trace, day_pos, ...):
    pos = np.asarray(day_pos, dtype=np.float32)
```

iii. The notes state that the paper's position stream comes from DeepLabCut head tracking and is already frame-aligned to the neural signal.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI keeps the 2D `position` trajectory, converts it to `float32`, truncates it to the common neural/position length, and discretizes each trial into a 3x3 grid by computing per-axis min and max over the provided slice and scaling positions into three bins along each axis. Because discretization is called inside the trial loop on `pos[:, s:e]`, the effective scaling is trial-specific. Invalid coordinates are marked with label `-1`.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    x = pos_xy[0]
    y = pos_xy[1]
    valid = np.isfinite(x) & np.isfinite(y)
    ...
    xmin, xmax = np.min(xv), np.max(xv)
    ymin, ymax = np.min(yv), np.max(yv)
    ...
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2)
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2)
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]

output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The notes justify 3x3 categorical position outputs because the decoder instructions required 9 classes. The metadata says this was intended to use the "observed x/y range" within each session/day, but the code actually recomputes the range on each 1-minute trial slice.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholding is dynamic rather than fixed to arena coordinates: along each axis, the AI maps the observed coordinate range in the current slice to bins `0, 1, 2` using `floor(normalized_value * 3)` and clipping. The final class is `y_bin * 3 + x_bin`, giving labels `0..8`; missing positions would become `-1`.

ii.
```python
if xmax == xmin:
    xbins = np.zeros(np.sum(valid), dtype=np.int64)
else:
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
if ymax == ymin:
    ybins = np.zeros(np.sum(valid), dtype=np.int64)
else:
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
labels[~valid] = -1
```

iii. The justification in the notes is only that the task required a 3x3 categorical decoder output. No separate justification is given for using dynamic observed-range thresholds instead of fixed arena boundaries.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes neural and position samples are framewise aligned within a day. It truncates both streams to the shared minimum frame count, then slices them with the same `s:e` trial boundaries.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
...
neural_trials.append(trace[:, s:e])
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The trajectory explicitly says position and trace "share frame dimensions" and should use a common framewise time base, so the AI chose simple truncation plus identical slicing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: all-NaN neurons are dropped; remaining neural NaNs are replaced with zero; neural and position streams are truncated to the shorter length if they disagree; trailing partial trials are dropped; sessions with fewer than two full trials are skipped; and invalid position samples would be assigned label `-1`.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
n_frames = min(trace.shape[1], pos.shape[1])
...
n_full_trials = n_frames // FRAMES_PER_TRIAL
...
if len(neural_trials) < 2:
    continue
...
labels[~valid] = -1
```

iii. The notes explicitly justify the all-NaN neuron removal and dropping trailing partial-minute frames. The zero-filling of partial NaNs, min-length truncation, and `-1` invalid-position labels are present in code but were not clearly justified in the notes.

## 6-a. What are the most time-consuming steps of the code?

i. The code's main costs are loading and deserializing each large joblib animal file, scanning/copying large `trace` arrays day by day, and the optional plotting path when `--show-processing` is enabled. The training logs also show that the output contains very large total timepoint counts per session after concatenation.

ii.
```python
def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]
...
for animal_id in animal_ids:
    animal = load_animal(animal_id, data_dir)
    ...
    for day_idx in range(n_days):
        neural_trials, input_trials, output_trials, valid_neurons = process_day(...)
```

iii. The AI did not explicitly document a runtime analysis beyond storing `conversion_runtime_sec` in metadata, so this is inferred from the code structure and the size of the arrays mentioned in the notes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization opportunity is the per-trial Python loop in `process_day`. Trial extraction could be done with reshaping/splitting once per session, and position discretization could be computed in bulk rather than recalculated on each trial slice. The loop over blocked indices is also small but could be replaced with indexed assignment.

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))

for v in vals:
    if v == -1:
        continue
    r, c = divmod(int(v), 3)
    mask[r, c] = 0.0
```

iii. The notes contain a placeholder "Code inefficiencies identified" but do not spell them out. The vectorization opportunities are evident from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes position bin thresholds for every 1-minute trial, repeatedly copies the same static input mask once per trial, and repeatedly performs NaN scans per day/session. Because discretization happens inside the trial loop, the min/max computation for binning is repeated thousands of times.

ii.
```python
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
...
for ti in range(n_full_trials):
    ...
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. This repeated work was not justified in the notes; it is an implementation consequence of the chosen structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional `--show-processing` path generates diagnostic plots that are not used downstream. The converter also records metadata such as runtime and stores per-trial copies of an identical static geometry vector instead of a more compact representation. More importantly, the code computes dynamic position thresholds separately for every trial even though only the final labels are retained.

ii.
```python
if show_processing and plt is not None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ...
    fig.savefig(f'processing_{session_tag}.png', dpi=150)
...
input_trials.append(mask.copy())
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The notes mention `--show-processing` and runtime reporting as validation aids, not as part of the downstream decoder input/output specification.
