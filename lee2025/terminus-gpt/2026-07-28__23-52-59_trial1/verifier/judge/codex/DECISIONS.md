# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data from a hard-coded list of seven per-animal `joblib` files in `data/`, not by scanning `.mat` files. Each loaded object is expected to be a dict keyed by the animal ID, and the conversion iterates those animal records.

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
```

iii. The notes and trajectory say the agent treated the reference notebook call `load_dat(..., format="joblib")` as canonical, so it preferred the joblib-packaged per-animal files over the `.mat` files.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is treated as one subject. `subjects` is initialized from `ANIMAL_IDS`, and `subject_idx` is filled from a lookup table while sessions are appended.

ii.
```python
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}

data['subject_idx'].append(subject_to_idx[animal_id])
```

iii. The trajectory repeatedly describes the dataset as “seven animals” and treats the per-animal files as the natural subject split.

## 1-c. How are the data split into sessions?

i. Each day index inside one animal’s arrays is treated as one session. The agent uses the first axis of `trace` as `n_days`, then takes matching slices from `trace`, `position`, `envs`, and `blocked`.

ii.
```python
traces = np.asarray(animal['trace'])
positions = np.asarray(animal['position'])
blocked = animal['blocked']
n_days = traces.shape[0]

for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
    )
```

iii. The notes and trajectory say the paper/code operate “within day/session,” so the agent decided that one recording day should map to one output session.

## 1-d. How are the data split into trials?

i. Each session/day is split into contiguous non-overlapping 60 s chunks at 30 Hz, so each trial has 1800 frames. Trailing frames that do not fill a complete chunk are discarded.

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

iii. The trajectory says the agent inferred 1800-frame trials from the paper’s 30 Hz sampling and the task’s 1-minute-trial requirement.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter. The code only drops incomplete trailing chunks and skips an entire session if fewer than two full trials remain.

ii.
```python
n_full_trials = n_frames // FRAMES_PER_TRIAL
...
if len(neural_trials) < 2:
    continue
```

iii. The only stated justification is decoder-format compliance: the trajectory notes that sessions need at least two trials, and the code enforces that requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the per-animal `trace` array, indexed by day.

ii.
```python
traces = np.asarray(animal['trace'])
...
process_day(traces[day_idx], ...)
```

iii. The notes say `trace` is the “analysis-ready neural signal” and identify it as the binary rising-phase calcium-event trace used by the authors.

## 2-b. How is the `neural` data processed?

i. The code treats `trace` as already-processed neural activity. It removes neurons that are all NaN for that day, replaces any remaining NaNs with 0, casts to `float32`, and crops to the minimum shared frame count with position. There is no temporal rebinning.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
```

iii. The notes explicitly justify using `trace` directly because the reference materials describe it as the already-thresholded rising-phase signal; the extra `nan_to_num`/cropping behavior is not justified separately in the notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The day-level neuron filter is “drop neurons that are all NaN.” Neurons with at least one non-NaN sample are retained, and any remaining NaNs are zero-filled.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. The notes and trajectory describe all-NaN neurons as cells not recorded on that day, so the agent treated all-NaN removal as the needed neuron curation step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no real event alignment. The code treats session/day start as the alignment origin and cuts continuous recordings into consecutive 1-minute windows.

ii.
```python
'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials',
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The trajectory repeatedly describes the recordings as continuous 40-minute sessions with no stimulus onset to align to, so the agent used session/day start as a bookkeeping alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz frame rate, so the time bin is 33.33 ms and no temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
...
'time_bin_size': 1000.0 / FRAME_RATE_HZ,
```

iii. The notes and trajectory cite the paper’s 30 Hz position sampling and state the intent to keep a common framewise time base.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from both the per-day environment label `envs` and the per-day blocked-entry `blocked`.

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
...
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The notes say `envs` gives environment identity and `blocked` gives blocked partition locations, and the trajectory says the decoder input should be a static 3x3 geometry/block mask.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is converted into a 3x3 occupancy mask, the blocked entry is converted into a 3x3 mask with blocked cells set to 0, and the two masks are multiplied and flattened into a length-9 vector.

ii.
```python
def env_to_mask(env_name):
    ...

def blocked_to_mask(entry):
    ...
    mask = np.ones((3, 3), dtype=np.float32)
    ...
    mask[r, c] = 0.0
    return mask

def combined_env_mask(env_name, blocked_entry):
    return env_to_mask(env_name) * blocked_to_mask(blocked_entry)
```

iii. The notes explicitly cite reference-code helpers that map environment names to 3x3 masks and describe `blocked` as 3x3 partition indices, which the agent combined into one geometry input.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry input is static within a day/session. The code computes one 9-element mask for the day and attaches a copy of that same vector to every 1-minute trial cut from that day.

ii.
```python
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
...
input_trials.append(mask.copy())
```

iii. The trajectory frames this variable as “static per-trial environment geometry/block mask,” so the agent kept it constant over time and only indexed it by session/day.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the per-day `position` array.

ii.
```python
positions = np.asarray(animal['position'])
...
process_day(..., positions[day_idx], ...)
```

iii. The notes identify `position` as x-y position per frame/day from DeepLabCut.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code discretizes x and y into a 3x3 grid by rescaling each input slice to that slice’s observed min/max range, then converting the resulting x/y bins into one class label. Because `discretize_position_to_3x3` is called inside the trial loop on `pos[:, s:e]`, the actual implementation computes bins independently for each 1-minute trial, not once for the full session/day.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    ...
    xmin, xmax = np.min(xv), np.max(xv)
    ymin, ymax = np.min(yv), np.max(yv)
    ...
    labels = y_out * 3 + x_out
    labels[~valid] = -1
    return labels[np.newaxis, :]

...
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The notes justify adapting continuous position to the required 3x3 categorical output, and metadata says position is discretized using the “observed x/y range”; however, the code applies that observed-range rule at the trial level, not the session/day level claimed in the metadata.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Valid x/y samples are each mapped to bins 0, 1, or 2 by floor-scaling within the observed range, and the final category is `y_bin * 3 + x_bin`. Invalid samples are assigned `-1`.

ii.
```python
xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
...
labels = y_out * 3 + x_out
labels[~valid] = -1
```

iii. The notes/trajectory justify a 3x3 categorical position output because the decoder task requires 9 classes, but they do not separately justify the invalid `-1` label.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned frame-for-frame with neural data after first truncating both arrays to their shared minimum frame count. The code then slices both arrays using the same 1800-frame trial boundaries.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
...
neural_trials.append(trace[:, s:e])
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The trajectory says the paper records trace and position on a common frame axis, and the code uses matched slicing to preserve that alignment while defensively truncating mismatched lengths.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code keeps the native 30 Hz frame bins, so each time bin is 33.33 ms and no temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 30.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)
...
'time_bin_size': 1000.0 / FRAME_RATE_HZ,
```

iii. The notes repeatedly treat the reference data as already sampled on the correct framewise time base.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and position streams are truncated to the same day-level frame count, then split with identical 1800-frame boundaries. The geometry input is a constant day-level vector copied into each corresponding trial, so all three streams are aligned by shared session/day and trial index.

ii.
```python
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
...
neural_trials.append(trace[:, s:e])
input_trials.append(mask.copy())
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The trajectory says the agent wanted a “common framewise time base” with static per-trial context input.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The code handles minor issues by removing neurons that are all NaN, zero-filling remaining NaNs, truncating trace/position length mismatches to the shorter stream, dropping incomplete trailing trials, skipping sessions with fewer than two full trials, and converting invalid position samples to label `-1`. It also normalizes irregular nested `blocked` entries before decoding them.

ii.
```python
while isinstance(cur, list) and len(cur) == 1:
    cur = cur[0]
...
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
n_frames = min(trace.shape[1], pos.shape[1])
...
labels[~valid] = -1
...
if len(neural_trials) < 2:
    continue
```

iii. Some of this is justified in the notes/trajectory as practical normalization of ragged `blocked` entries and day-level NaN filtering; the length truncation and invalid-position handling are mostly implicit in the code rather than explicitly documented.

## 7-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large per-animal `joblib` objects and then iterating over every day and every 1-minute trial to slice traces and discretize positions. Optional plot generation adds more cost when enabled.

ii.
```python
animal = load_animal(animal_id, data_dir)
...
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(...)
...
for ti in range(n_full_trials):
    ...
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The notes do not quantify runtime hotspots, but the trajectory repeatedly waited on dataset loading and full-dataset processing, which makes the I/O plus per-trial loop structure the main inferred bottleneck.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial loop in `process_day` could have been vectorized or replaced with reshape/split operations, and position discretization could have been computed once on a larger array instead of once per trial.

ii.
```python
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. This is not explicitly discussed in the notes; it is inferred from the code structure.

## 7-c. What processing does the code repeat multiple times?

i. The code recomputes min/max-based position discretization separately for every trial and copies the same static geometry vector into every trial of a session/day.

ii.
```python
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
input_trials.append(mask.copy())
```

iii. The repetition is implicit in the implementation; the notes do not call it out.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. When `--show-processing` is used, the code generates PNG diagnostics that are not consumed downstream. Even without plotting, it duplicates identical static input vectors across trials and records conversion-runtime metadata that decoder training does not use.

ii.
```python
if show_processing and plt is not None:
    ...
    fig.savefig(f'processing_{session_tag}.png', dpi=150)

input_trials.append(mask.copy())
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The notes describe the plots as sanity checks, not as part of the decoder payload.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: all-NaN neurons are dropped, remaining NaNs are zero-filled, mismatched stream lengths are truncated, incomplete trailing trials are discarded, sessions with fewer than two full trials are skipped, invalid position samples become `-1`, and ragged `blocked` entries are normalized.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
n_frames = min(trace.shape[1], pos.shape[1])
...
labels[~valid] = -1
```

iii. The justification is the same as in 6: some of this is explicit in the notes/trajectory, while some is only implicit in the code.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: the main cost is loading large per-animal `joblib` files and iterating over every day and trial to slice traces and discretize positions, with optional plotting adding extra work.

ii.
```python
animal = load_animal(animal_id, data_dir)
...
for day_idx in range(n_days):
    ...
for ti in range(n_full_trials):
    ...
```

iii. Same inferred justification as 7-a.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-trial loop in `process_day` is the main vectorization target, especially trial slicing and repeated calls to `discretize_position_to_3x3`.

ii.
```python
for ti in range(n_full_trials):
    ...
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. Same inferred justification as 7-b.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: it recomputes adaptive min/max position binning for every trial and duplicates the same static geometry vector for every trial.

ii.
```python
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
input_trials.append(mask.copy())
```

iii. Same inferred justification as 7-c.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: optional PNG generation is discarded by downstream analyses, and the code also duplicates static input vectors and stores runtime metadata that the decoder does not use.

ii.
```python
if show_processing and plt is not None:
    ...
    fig.savefig(f'processing_{session_tag}.png', dpi=150)
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. Same justification as 7-d.
