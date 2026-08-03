# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, loads one joblib file per animal, then pulls `envs`, `trace`, `position`, and `blocked` from the loaded per-animal dictionary. It treats each animal file as containing multiple day-sessions and processes them in nested loops.

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

iii. In `CONVERSION_NOTES.md`, the agent says the reference notebook used per-animal loading with `load_dat(..., format="joblib")`, and the trajectory repeatedly cites the canonical per-animal fields `trace`, `position`, `envs`, and `blocked` as the basis for loading.

## 1-b. How are the data split into subjects?

i. Subjects are defined directly by the hard-coded animal ID list. `subjects` is just that list, and `subject_idx` records which hard-coded subject each appended session belongs to.

ii.
```python
animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}

data = {
    ...
    'subjects': subjects,
    'subject_idx': [],
    ...
}

data['subject_idx'].append(subject_to_idx[animal_id])
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. The notes and trajectory both say the dataset contains seven animal IDs and that those seven mice should define the subject list.

## 1-c. How are the data split into sessions?

i. Each day within an animal is treated as one session. The script uses the first dimension of `trace` as the day/session axis and appends one output session per day, unless that day produces fewer than two full 1-minute trials.

ii.
```python
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
        show_processing=show_processing and total_sessions < 2,
        session_tag=f'{animal_id}_day{day_idx+1:02d}'
    )
    if len(neural_trials) < 2:
        continue
    data['neural'].append(neural_trials)
    data['input'].append(input_trials)
    data['output'].append(output_trials)
```

iii. The notes say the paper has one recording session per day and that the reference analyses operate within day/session, so the agent chose day = session before trial-splitting.

## 1-d. How are the data split into trials?

i. Each session/day is split into contiguous non-overlapping 1-minute trials. At 30 Hz, each trial is 1800 frames, and any trailing remainder shorter than 1800 frames is dropped.

ii.
```python
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 60.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)

n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The trajectory says the sessions are about 40 minutes at 30 Hz, so the agent inferred 1800 frames per minute and explicitly documented that trailing partial-minute frames are dropped.

## 1-e. How are trials filtered based on quality controls?

i. There is no substantive per-trial quality-control filter. The only trial-level filtering is implicit dropping of the final incomplete chunk via floor division. There is also a session-level filter that skips sessions with fewer than two full trials.

ii.
```python
n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    ...

if len(neural_trials) < 2:
    continue
```

iii. The agent’s notes say the decoder requires at least two trials per session, and they describe dropping trailing partial-minute frames. No further trial-quality rule was documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` variable for each animal and day.

ii.
```python
traces = np.asarray(animal['trace'])

neural_trials, input_trials, output_trials, valid_neurons = process_day(
    traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
    ...
)
```

iii. The notes state that `trace` is the provided analysis-ready neural signal and corresponds to the rise-extracted calcium-event trace used by the authors.

## 2-b. How is the `neural` data processed?

i. For each day, the agent removes neurons that are all NaN that day, replaces remaining NaNs with 0, casts to `float32`, truncates neural and position streams to the same number of frames, and then slices the day into 1-minute neuron-by-time trial matrices.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
pos = np.asarray(day_pos, dtype=np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]

for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
```

iii. The notes argue that `trace` should be used directly because the reference materials describe it as a binary rising-phase event signal and show no need to recompute dF/F or deconvolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by whether they are all NaN within a given day/session. Those neurons are dropped from that session.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)

data['brain_region_idx'].append(np.zeros(int(np.sum(valid_neurons)), dtype=np.int64))
```

iii. The notes explicitly say the implementation filters neurons that are all-NaN within a day and treat that as removing neurons not recorded in that session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align neural activity to a behavioral event. Instead, it treats recording/session start as time zero and creates contiguous 1-minute windows from the start of each day/session.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials',
    'off_start': 0.0,
    'off_end': TRIAL_SECONDS,
    ...
}

for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
```

iii. The notes and trajectory say the natural unit in the source data is a day/session, not an event-locked trial, so the agent used session start as the alignment anchor required to build 1-minute trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native 30 Hz frame rate, corresponding to 33.33 ms bins. No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 60.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)

'metadata': {
    ...
    'time_bin_size': 1000.0 / FRAME_RATE_HZ,
    'frame_rate_hz': FRAME_RATE_HZ,
    ...
}
```

iii. The notes cite 30 Hz position trajectories from the paper and say the agent chose to keep a common framewise time base.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The agent derives the decoder input from both `envs` and `blocked`. `envs` chooses a named arena-shape template and `blocked` specifies blocked partitions within that template.

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)

mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The notes say the reference code had an environment-shape helper and that `blocked` stores 3x3 partition indices, so the agent decided to combine both sources into one static geometry vector.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The script maps each named environment to a hand-coded 3x3 binary template, normalizes the nested `blocked` entry, converts blocked indices into a 3x3 mask, multiplies the two masks together, flattens the result to length 9, and reuses that same vector for every trial in the session.

ii.
```python
def env_to_mask(env_name):
    ...

def normalize_blocked_entry(entry):
    ...

def blocked_to_mask(entry):
    ...
    mask = np.ones((3, 3), dtype=np.float32)
    if len(vals) == 1 and vals[0] == -1:
        return mask
    for v in vals:
        ...
        mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)

mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
input_trials.append(mask.copy())
```

iii. The notes say the environment should be represented as a static 3x3 geometry/block mask per trial, and the trajectory shows the agent inspecting raw nested `blocked` entries to make this conversion robust.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the raw per-frame `position` variable.

ii.
```python
positions = np.asarray(animal['position'])

neural_trials, input_trials, output_trials, valid_neurons = process_day(
    traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
    ...
)

pos = np.asarray(day_pos, dtype=np.float32)
```

iii. The notes identify `position` as the x-y trajectory from DeepLabCut head tracking and use it as the source for decoded mouse position.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. For each day/session, the agent takes x and y coordinates, finds the observed min and max on that day, divides each axis into three equal-width bins over that observed range, clips to `[0, 2]`, combines the x and y bins into one label `y * 3 + x`, and returns a `(1, time)` categorical time series.

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
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
    ...
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
    ...
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]
```

iii. The notes say the paper’s continuous position output had to be adapted to the benchmark’s required 3x3 categorical output while preserving time variation.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholds are session-specific: each axis is thresholded into thirds of that session’s observed coordinate range. If an axis has no range, everything on that axis goes to bin 0. The final category index is `y_bin * 3 + x_bin`.

ii.
```python
xmin, xmax = np.min(xv), np.max(xv)
ymin, ymax = np.min(yv), np.max(yv)
if xmax == xmin:
    xbins = np.zeros(np.sum(valid), dtype=np.int64)
else:
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
if ymax == ymin:
    ybins = np.zeros(np.sum(valid), dtype=np.int64)
else:
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
```

iii. The metadata note says position was “discretized independently within each session/day into 3x3 bins using observed x/y range,” which is the agent’s direct justification.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are aligned framewise by trimming both to the same frame count, then cutting both streams with the same 1-minute slice boundaries. The output remains time-varying within each trial.

ii.
```python
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
pos = np.asarray(day_pos, dtype=np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]

for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The notes say the agent kept a common framewise time base and reported no obvious temporal misalignment during verification.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent adds several robustness rules: it unwraps singleton nested `blocked` entries, converts NumPy arrays and tuples inside `blocked` to lists, removes neurons that are all NaN for the day, replaces any remaining neural NaNs with 0, trims trace and position to the shorter stream if lengths differ, assigns `-1` to position samples with non-finite coordinates, and drops trailing partial trials.

ii.
```python
def normalize_blocked_entry(entry):
    cur = entry
    while isinstance(cur, list) and len(cur) == 1:
        cur = cur[0]
    if isinstance(cur, np.ndarray):
        cur = cur.tolist()
    if isinstance(cur, tuple):
        cur = list(cur)
    return cur

valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
n_frames = min(trace.shape[1], pos.shape[1])
...
labels[~valid] = -1
...
n_full_trials = n_frames // FRAMES_PER_TRIAL
```

iii. The trajectory shows the agent explicitly worrying about ragged `blocked` structures and slight frame-count variation across animals, and the notes describe these choices as pragmatic robustness steps for decoder-compatible output.

## 6-a. What are the most time-consuming steps of the code?

i. The heaviest work is loading each large animal file from disk, iterating through every day/session and every 1-minute trial to materialize Python lists of arrays, and finally serializing the large converted dataset. Optional plotting also adds overhead when enabled.

ii.
```python
def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]

for animal_id in animal_ids:
    animal = load_animal(animal_id, data_dir)
    ...
    for day_idx in range(n_days):
        neural_trials, input_trials, output_trials, valid_neurons = process_day(...)
        ...

if show_processing and plt is not None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ...

with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory shows long waits during full conversion and notes that the final pickle is very large, which is consistent with I/O and list materialization dominating runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop in `process_day` could be vectorized or replaced with reshaping after truncation to full trials. The `env_to_mask` `if/elif` ladder could be replaced with a lookup table. The nested animal/day loop is unavoidable at a high level but still performs a lot of repeated Python bookkeeping.

ii.
```python
def env_to_mask(env_name):
    if env == 'square':
        ...
    elif env == 'o':
        ...
    ...

for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))

for animal_id in animal_ids:
    ...
    for day_idx in range(n_days):
        ...
```

iii. These are direct consequences of the code structure; the notes leave performance placeholders, but the conversion runtime observed in the trajectory makes these loops the obvious vectorization targets.

## 6-c. What processing does the code repeat multiple times?

i. The code recomputes position discretization separately for every 1-minute slice instead of discretizing the full day once and then slicing it. It also duplicates the same static geometry vector with `mask.copy()` for every trial, and it reruns the full conversion logic separately in sample and full modes.

ii.
```python
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
for ti in range(n_full_trials):
    ...
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))

mode = 'sample' if args.sample and not args.full else 'full'
data = convert_dataset(data_dir='data', mode=mode, show_processing=args.show_processing)
```

iii. The trajectory shows that sample conversion and full conversion are separate runs, and the code itself clearly recalculates the same static/session-level values inside trial loops.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional `show_processing` plotting block generates diagnostic PNGs that are not used by downstream decoding. The code also stores bookkeeping metadata such as runtime and session counts, and repeatedly copies a static mask per trial, none of which affects the decoder’s learned inputs or outputs.

ii.
```python
if show_processing and plt is not None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ...
    fig.savefig(f'processing_{session_tag}.png', dpi=150)
    plt.close(fig)

input_trials.append(mask.copy())

data['metadata']['n_sessions'] = total_sessions
data['metadata']['n_trials'] = total_trials
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The notes explicitly describe the processing plots as sanity checks, and nothing in the decoder format or training code consumes those PNGs or the extra runtime metadata.
