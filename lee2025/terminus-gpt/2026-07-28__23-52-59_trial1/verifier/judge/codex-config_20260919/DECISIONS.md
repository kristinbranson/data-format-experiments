# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs and loads each extensionless joblib file. It indexes the top-level dictionary by animal ID, then extracts `envs`, `trace`, `position`, and `blocked`. Full mode processes all seven animals; sample mode uses the first two.

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

iii. The notes say the authors' demo uses `load_dat(..., format="joblib")` and directly accesses these canonical fields. Thus the agent considered the provided joblib representation an analysis-ready equivalent of the MATLAB files.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is one subject. Subject order follows `ANIMAL_IDS`, and each retained day/session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_to_idx[animal_id])
```

iii. The notes identify seven animal files and state that each per-animal joblib object contains that animal's days, traces, positions, environments, and blocked locations.

## 1-c. How are the data split into sessions?

i. The first axis of each animal's `trace` is treated as recording day/session. Every day is processed independently and appended as one output session, except a day yielding fewer than two full trials is skipped.

ii.
```python
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx], ...)
    if len(neural_trials) < 2:
        continue
    data['neural'].append(neural_trials)
```

iii. The notes call day/session the natural unit because the paper recorded one approximately 40-minute session per day and the original decoder operated within day.

## 1-d. How are the data split into trials?

i. Each session is cut into contiguous, non-overlapping 60-second trials of 1,800 frames at 30 Hz. Only full trials are retained; trailing frames are discarded.

ii.
```python
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)
n_full_trials = n_frames // FRAMES_PER_TRIAL
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
```

iii. This directly follows the task's definition of one-minute trials. The metadata and notes explicitly document dropping the trailing partial minute.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality metric. Incomplete trailing trials are dropped, and an entire session is omitted if fewer than two complete trials remain. Otherwise all complete trials are retained.

ii.
```python
n_full_trials = n_frames // FRAMES_PER_TRIAL
...
if len(neural_trials) < 2:
    continue
```

iii. The two-trial session rule enforces the target-format requirement that decoder evaluation needs at least two trials per session. No additional trial-curation rule was found in the notes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from each animal's `trace` array, indexed by day.

ii.
```python
traces = np.asarray(animal['trace'])
...
traces[day_idx]
```

iii. The notes identify `trace` as the supplied rise-extracted, binary calcium-event signal and say no raw-fluorescence or dF/F recomputation is needed.

## 2-b. How is the `neural` data processed?

i. Neurons that are all NaN for the day are removed, remaining NaNs are replaced with zero, the array is cast to `float32`, truncated to the common neural/position length, and sliced into trials. No smoothing, deconvolution, or temporal rebinning is performed.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
```

iii. The agent concluded that the supplied trace is already the analysis-ready signal used by the paper. It chose zero for isolated missing neural samples and truncation as a defensive alignment measure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained if it has at least one non-NaN sample during that day. There is no place-cell, reliability, event-rate, or other neural-quality threshold.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The notes interpret all-NaN rows as cells not recorded in that session. Although they discuss split-half place-cell analysis, they do not apply it because the supplied event trace is used directly for decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Day start is time zero, and neural data are partitioned into consecutive one-minute windows from that point.

ii.
```python
'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials',
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The notes explain that the recordings are continuous sessions and that day/session is the natural analysis unit; the requested trials are artificial windows rather than event-centered epochs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz resolution is retained, giving 33.333 ms bins. No temporal aggregation or resampling is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
...
'time_bin_size': 1000.0 / FRAME_RATE_HZ,
```

iii. The notes cite 30 Hz framewise position trajectories and matching neural/position frame dimensions, and choose the common native time base.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input combines the raw per-day environment name (`envs`) with the per-day blocked partition indices (`blocked`).

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
...
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The notes state that named environments define arena geometry and `blocked` stores unavailable locations in a 3×3 indexing scheme, so both were considered relevant to “which parts of the arena are blocked.”

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A named environment is mapped to a hard-coded 3×3 accessible-area mask. A second all-ones mask is zeroed at blocked indices. Their elementwise product is flattened to nine float32 features and copied once per trial; 1 means accessible and 0 means absent/blocked.

ii.
```python
def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
...
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
...
input_trials.append(mask.copy())
```

iii. The agent says this static mask captures both the arena shape and explicit blockages, matching the study's changing geometry while satisfying the decoder-input requirement.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from the two-coordinate, framewise `position` array for each day.

ii.
```python
positions = np.asarray(animal['position'])
...
pos = np.asarray(day_pos, dtype=np.float32)
```

iii. The notes identify these as DeepLabCut-derived x-y trajectories recorded at 30 Hz.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is cast to float32, truncated to the common neural/position frame count, sliced into one-minute trials, and then each trial is independently converted to a single row of integer class labels. Finite samples determine that trial's x/y extrema; nonfinite samples receive label -1.

ii.
```python
pos = np.asarray(day_pos, dtype=np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
pos = pos[:, :n_frames]
...
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The notes justify adapting continuous position to the required categorical 3×3 output and claim positions are discretized by the observed session/day range. The actual code instead invokes the range calculation separately for every trial.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Within each trial, finite x and y values are min-max scaled independently, multiplied by three, floored, and clipped to bins 0–2. The class is `y_bin * 3 + x_bin`, yielding 0–8; invalid coordinates become -1. Constant axes are assigned bin 0.

ii.
```python
xmin, xmax = np.min(xv), np.max(xv)
ymin, ymax = np.min(yv), np.max(yv)
xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
labels = y_out * 3 + x_out
labels[~valid] = -1
```

iii. The agent describes this as a coarse categorical representation suited to the benchmark. It does not document why trial-relative extrema are preferable to the known 75×75 cm arena boundaries.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are truncated to their minimum frame count and then sliced with identical start/end frame indices for every trial.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
...
neural_trials.append(trace[:, s:e])
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The notes report that neural and position arrays share a frame dimension and use the same 30 Hz time base; common truncation is intended to preserve framewise correspondence if lengths differ.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons are removed; remaining neural NaNs become zero. Nonfinite positions become output -1. Length mismatches are resolved by truncating both streams to the shorter one. Unknown environment names raise an error. A `-1` blocked sentinel means no explicit blocks; mixed `-1` entries are ignored. Partial final trials are dropped.

ii.
```python
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
...
labels[~valid] = -1
...
raise ValueError(f'Unknown environment name: {env!r}')
```

iii. The notes explicitly justify removing unrecorded cells and dropping partial minutes. Other behaviors are defensive choices visible in code but are not substantively justified in the notes.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant work is loading/deserializing each full per-animal joblib object and copying/slicing large day-level neural arrays. Optional plotting adds substantial work when enabled. The agent did not provide measured per-step timing, only total conversion runtime.

ii.
```python
animal = load_animal(animal_id, data_dir)
...
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The notes leave runtime tables and speed-up fields blank, so no evidence-backed bottleneck analysis was documented.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop could be replaced by truncation followed by reshape/split views for neural data. Position labels could be computed for the whole session and reshaped into trials. `blocked_to_mask` could use vectorized indexed assignment rather than looping over blocked indices. Animal/day loops are structurally appropriate because session neuron counts and metadata vary.

ii.
```python
for ti in range(n_full_trials):
    ...
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
...
for v in vals:
    ...
    mask[r, c] = 0.0
```

iii. The notes contain placeholders for inefficiencies and speedups and therefore give no explicit justification for retaining these loops.

## 6-c. What processing does the code repeat multiple times?

i. Position validity, minima, maxima, scaling, and binning are recomputed for every one-minute trial. The static environment mask is copied for every trial. Neural and output trial lists are repeatedly appended, and optional plot setup repeats for two sessions.

ii.
```python
for ti in range(n_full_trials):
    ...
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. No repeated-processing analysis appears in the notes. In particular, the notes describe session/day range normalization even though the implementation redundantly recalculates ranges per trial.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. With `--show-processing`, it constructs and saves diagnostic figures that are not used in the pickle or decoder. It also stores conversion runtime and summary counts as metadata; these are useful documentation but not decoder features. The environment-mask copies are redundant because the static arrays are not mutated downstream.

ii.
```python
if show_processing and plt is not None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ...
    fig.savefig(f'processing_{session_tag}.png', dpi=150)
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The agent intended plots as sanity checks for only the first two sessions, not as downstream data. The notes state plots were reviewed, so this optional work served validation even though the decoder discards it.
