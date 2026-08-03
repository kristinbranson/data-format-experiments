# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes seven animal IDs, then loads one extensionless per-animal file from `data/<animal_id>` with `joblib.load`. Inside each animal object it reads `envs`, `trace`, `position`, and `blocked`, and later iterates days and derived trials. It does not enumerate `.mat` files.

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

iii. The justification in `CONVERSION_NOTES.md` Step 1 is that the reference code/demo uses `load_dat(..., format="joblib")` and exposes per-animal fields `trace`, `position`, `envs`, and `blocked`. Step 6 also explicitly describes the implementation as using "per-animal joblib loading."

## 1-b. How are the data split into subjects?

i. Subjects are the seven hard-coded animal IDs. The output `subjects` list is just that constant list, and `subject_idx` is assigned from a dictionary mapping each ID to its list index.

ii.
```python
animal_ids = ANIMAL_IDS if mode == 'full' else ANIMAL_IDS[:2]
subjects = list(animal_ids)
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_to_idx[animal_id])
```

iii. The notes say the dataset contains seven animal IDs and treat each animal/day bundle as the natural source object. The AI therefore used one animal file per mouse and kept the given IDs as subject names.

## 1-c. How are the data split into sessions?

i. Each day inside an animal file is treated as a separate session. The code reads `traces.shape[0]` as the number of days and iterates `day_idx` from `0` to `n_days - 1`.

ii.
```python
traces = np.asarray(animal['trace'])
positions = np.asarray(animal['position'])
...
n_days = traces.shape[0]
for day_idx in range(n_days):
    neural_trials, input_trials, output_trials, valid_neurons = process_day(
        traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
        show_processing=show_processing and total_sessions < 2,
        session_tag=f'{animal_id}_day{day_idx+1:02d}'
    )
```

iii. `CONVERSION_NOTES.md` Step 1 says the reference analyses are "within-day" and that "day/session is the natural unit before our required 1-minute trial splitting." That is the stated rationale for making each day a session.

## 1-d. How are the data split into trials?

i. Within each session/day, the code computes 60-second non-overlapping trials at 30 Hz, so each trial is 1800 frames. It drops any leftover frames that do not fill a full trial.

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

iii. The notes repeatedly state that the decoder task requires 1-minute trials, and the metadata describes sessions as "split into contiguous 1-minute trials." That is the explicit reason for this segmentation.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only filtering near this stage is that any session/day yielding fewer than two full 1-minute trials is skipped entirely.

ii.
```python
n_full_trials = n_frames // FRAMES_PER_TRIAL
...
if len(neural_trials) < 2:
    continue
```

iii. The justification is implicit in the task requirements: the target format requires at least two trials per session for decoder evaluation. `CONVERSION_NOTES.md` Step 6 also notes that trailing partial-minute frames are dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from each day's `trace` array inside the per-animal object.

ii.
```python
traces = np.asarray(animal['trace'])
...
neural_trials, input_trials, output_trials, valid_neurons = process_day(
    traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
    ...
)
```

iii. The notes say the provided `trace` is the analysis-ready neural signal and that there is "no evidence ... suggesting recomputing calcium preprocessing from raw fluorescence."

## 2-b. How is the `neural` data processed?

i. For each day/session, the code removes neurons that are all NaN, replaces any remaining NaNs with `0.0`, casts to `float32`, truncates the neural stream to the common length shared with position, and then slices contiguous trial windows. It assumes the trace is already in neuron-by-time orientation and does not transpose it.

ii.
```python
def process_day(day_trace, day_pos, env_name, blocked_entry, show_processing=False, session_tag=''):
    valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
    trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
    pos = np.asarray(day_pos, dtype=np.float32)
    n_frames = min(trace.shape[1], pos.shape[1])
    trace = trace[:, :n_frames]
    ...
    neural_trials.append(trace[:, s:e])
```

iii. The stated rationale is that `trace` already contains "binary rising-phase calcium trace events" / "analysis-ready neural signal," so no dF/F or deconvolution is recomputed. The notes do not separately justify the extra `nan_to_num` or truncation steps; those are implicit defensive choices in the code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are all NaN across a day are removed. Any remaining NaN samples inside retained neurons are converted to zero.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says the implementation "filters neurons that are all-NaN within a day." The zero-filling of remaining NaNs is not separately justified in the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no internal event alignment. The AI treats session/day start as the alignment reference and divides the continuous stream into contiguous 1-minute windows.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Session/day start; sessions split into contiguous 1-minute trials',
    'off_start': 0.0,
    'off_end': TRIAL_SECONDS,
    ...
}
```

iii. The notes say the reference analyses are continuous within-session/day and that the task itself requires converting those sessions into 1-minute trials. That is the stated alignment logic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keeps the native 30 Hz frame resolution, i.e. `1000 / 30` ms per time bin. No temporal rebinning is applied beyond slicing into 60-second trials.

ii.
```python
FRAME_RATE_HZ = 30.0
TRIAL_SECONDS = 60.0
FRAMES_PER_TRIAL = int(FRAME_RATE_HZ * TRIAL_SECONDS)
...
'time_bin_size': 1000.0 / FRAME_RATE_HZ,
'frame_rate_hz': FRAME_RATE_HZ,
```

iii. `CONVERSION_NOTES.md` Step 3 cites 30 Hz position trajectories and says the conversion should preserve native framewise bins or a common framewise time base.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from both `envs` and `blocked`. `envs` chooses a named arena-shape mask, and `blocked` chooses additional blocked bins.

ii.
```python
envs = np.array(animal['envs']).squeeze()
blocked = animal['blocked']
...
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
```

iii. The notes explicitly cite both the environment-shape helper in the reference code and the `blocked` 3x3 partition indices, concluding that the decoder input should be built from a 3x3 geometry/block mask.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the named environment into a 3x3 occupancy mask, converts blocked indices into a second 3x3 mask where blocked cells are zeroed, multiplies the two masks together, flattens the result to length 9, and reuses that same static vector for every trial in the session/day.

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
    if len(vals) == 1 and vals[0] == -1:
        return mask
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

iii. `CONVERSION_NOTES.md` Step 4 says the AI resolved the input representation as "3x3 geometry/block mask per trial, static within each 1-minute trial." The rationale was to encode overall arena geometry, not just the blocked site list.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from each day's `position` array.

ii.
```python
positions = np.asarray(animal['position'])
...
neural_trials, input_trials, output_trials, valid_neurons = process_day(
    traces[day_idx], positions[day_idx], envs[day_idx], blocked[day_idx],
    ...
)
```

iii. The notes describe `position` as x-y position per day/frame and state that target output should be a 3x3 categorical version of this trajectory.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI's stated intent is to convert continuous x-y position into a time-varying 3x3 categorical output using the observed x/y range, but the actual code computes the min and max separately for each trial window and normalizes coordinates within that trial before assigning one of 9 labels.

ii.
```python
def discretize_position_to_3x3(pos_xy):
    x = pos_xy[0]
    y = pos_xy[1]
    ...
    xmin, xmax = np.min(xv), np.max(xv)
    ymin, ymax = np.min(yv), np.max(yv)
    ...
    xbins = np.clip(np.floor((xv - xmin) / (xmax - xmin + 1e-12) * 3), 0, 2).astype(np.int64)
    ybins = np.clip(np.floor((yv - ymin) / (ymax - ymin + 1e-12) * 3), 0, 2).astype(np.int64)
    ...
    return labels[np.newaxis, :]

output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. `CONVERSION_NOTES.md` Step 4 says "For target output, discretize position into 3x3 categorical bins over the arena, preserving time variation," and metadata says position is discretized using observed x/y range. No separate justification is given for doing that normalization per trial rather than globally.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The categories are produced by splitting the x and y coordinates into thirds of the observed coordinate range, then mapping `(x_bin, y_bin)` to `y_bin * 3 + x_bin`. Invalid positions get label `-1`. Because `discretize_position_to_3x3` is called inside the trial loop, those thresholds are recomputed trial by trial.

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
...
labels = y_out * 3 + x_out
labels[~valid] = -1
```

iii. The only explicit rationale is adapting continuous position to the required 9-class decoder target while preserving time variation. The notes do not justify the moving per-trial thresholds or the `-1` invalid class.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The code first truncates neural and position streams to a common frame count using `min(trace.shape[1], pos.shape[1])`, then slices the same trial boundaries from both streams. This gives framewise alignment after truncation.

ii.
```python
pos = np.asarray(day_pos, dtype=np.float32)
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
...
neural_trials.append(trace[:, s:e])
output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The justification is implicit: the agent wanted a defensively aligned common frame base before trial splitting. The notes say a common framewise time base should be used and that processing plots should reveal temporal misalignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code removes neurons that are all NaN, fills any remaining neural NaNs with zero, truncates neural and position to the shorter stream if their lengths differ, discards trailing frames that do not fill a full minute, and assigns `-1` to any position sample with non-finite x or y.

ii.
```python
valid_neurons = ~np.all(np.isnan(day_trace), axis=1)
trace = np.nan_to_num(day_trace[valid_neurons], nan=0.0).astype(np.float32)
...
n_frames = min(trace.shape[1], pos.shape[1])
trace = trace[:, :n_frames]
pos = pos[:, :n_frames]
...
n_full_trials = n_frames // FRAMES_PER_TRIAL
...
labels[~valid] = -1
```

iii. The notes explicitly mention dropping trailing partial-minute frames and filtering all-NaN neurons. The remaining safeguards are only implicit in the code; they are not separately justified in the notes.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive parts are I/O-heavy: loading each animal with `joblib.load` and writing the full output with `pickle.dump`. Optional plotting in `--show-processing` also adds overhead.

ii.
```python
def load_animal(animal_id, data_dir):
    return joblib.load(Path(data_dir) / animal_id)[animal_id]
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory shows the full conversion running animal by animal and later notes that `converted_data.pkl` was about 19 GB, so serialization is clearly a major runtime cost. The notes also describe "per-animal joblib loading."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_day` could have been vectorized. In particular, the code slices one trial at a time, discretizes position one trial at a time, and appends one copy of the static input mask per trial instead of reshaping the full day/session in bulk.

ii.
```python
neural_trials = []
input_trials = []
output_trials = []
for ti in range(n_full_trials):
    s = ti * FRAMES_PER_TRIAL
    e = s + FRAMES_PER_TRIAL
    neural_trials.append(trace[:, s:e])
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. The task instructions explicitly asked for vectorization where possible, but the code keeps the trial construction in Python loops. No separate justification for this choice appears in the notes.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes position discretization for each trial, repeatedly copies the same static environment mask for every trial in a session, and repeatedly builds Python lists of trial arrays instead of reshaping in one pass.

ii.
```python
mask = combined_env_mask(env_name, blocked_entry).reshape(-1).astype(np.float32)
...
for ti in range(n_full_trials):
    ...
    input_trials.append(mask.copy())
    output_trials.append(discretize_position_to_3x3(pos[:, s:e]).astype(np.int64))
```

iii. There is no explicit justification in the notes beyond making each trial an explicit element of the target-format lists.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional `--show-processing` branch renders and saves diagnostic plots that are not used by the downstream decoder. The code also stores runtime bookkeeping in metadata, which is descriptive rather than analytically necessary.

ii.
```python
if show_processing and plt is not None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ...
    fig.savefig(f'processing_{session_tag}.png', dpi=150)
    plt.close(fig)
...
data['metadata']['conversion_runtime_sec'] = time.time() - t0
```

iii. The plotting branch is justified by the task instruction requiring visual processing checks, but it is still extra work relative to the final converted dataset consumed by `train_decoder.py`.
