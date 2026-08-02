# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the raw `.mat` files directly. It hard-coded the seven animal IDs, then loaded one extensionless joblib file per animal from `data/`. Inside each loaded object it accessed `trace`, `position`, and `envs`, and later split each day into trials.

ii.
```python
data_dir = 'data'
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
dat = joblib.load(os.path.join(data_dir, animal_name))
d = dat[animal_name]
...
trace = d['trace']
position = d['position']
envs = d['envs'].flatten()
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as matching the reference code’s joblib-loading path, stating: "`joblib.load(f'data/{animal}')` | `load_dat(animal, p, format='joblib')` | YES — same joblib loading."

## 1-b. How are the data split into subjects?

i. Subjects are split by the hard-coded animal list. Each animal name is treated as one subject, and `subject_idx` is assigned from the animal’s index in that list.

ii.
```python
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
subjects = animals  # All 7 animals are subjects
...
for animal in animals_to_process:
    subject_id = animals.index(animal)
...
    subject_idx_list.append(subject_id)
```

iii. The notes repeatedly describe the dataset as "7 animals" and "all 7 animals are subjects," and the trajectory shows the AI deliberately fixed the subject list rather than discovering it from filenames.

## 1-c. How are the data split into sessions?

i. Each day within an animal’s loaded arrays is treated as a separate session. The code iterates over `n_days`, and each `day` contributes one session to the output.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
position = d['position'] # (n_days, 2, n_frames)
envs = d['envs'].flatten()

n_days, n_cells_total, n_frames_total = trace.shape
...
for day in range(n_days):
    trace_day = trace[day]
    pos_day = position[day]
    env_name = str(envs[day])
...
    sessions_neural.append(trials_neural)
```

iii. In the notes, the AI explicitly equated "one session = one decoder session" and described sessions as "each recording day."

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 1-minute trials at 30 Hz, so each trial is 1800 frames. The remainder at the end of the session is discarded because `n_trials` is computed with floor division.

ii.
```python
fps = 30
trial_duration_sec = 60
trial_duration_frames = fps * trial_duration_sec  # 1800 frames
...
n_trials = n_frames_total // trial_duration_frames
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes justify this directly from the task: "Trial duration = 1 minute: Instructions say '1-minute trials within each session'."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filtering. Trials are kept if they fit into a full 1-minute chunk; partial trailing chunks are dropped implicitly.

ii.
```python
n_trials = n_frames_total // trial_duration_frames
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
```

iii. The notes say "Trial curation: None in original (1 session = 1 day). We split into 1-min trials," and later call the discarded tail frames "acceptable."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from the loaded `trace` array for each animal, specifically `trace[day]` for each session/day.

ii.
```python
trace = d['trace']       # (n_days, n_cells, n_frames)
...
trace_day = trace[day]       # (n_cells, n_frames)
```

iii. The notes describe `trace` as the binary calcium-event signal and state that this is the neural source variable: "Use raw binary trace (0/1 events) at native 30 Hz."

## 2-b. How is the `neural` data processed?

i. The AI keeps the binary trace at native frame rate, filters to active cells, replaces any remaining NaNs with zero, slices it into trials, and casts each trial to `float32`. It does not apply place-cell filtering, velocity filtering, smoothing, or temporal rebinning.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
```

iii. The notes justify this with "Neural: binarized calcium trace ... already preprocessed," "All registered cells included," and "No velocity filtering." They also say "NO additional processing needed," with the code adding `np.nan_to_num(..., nan=0.0)` as a safety step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells whose trace is all NaN on a given day are removed. Any remaining NaNs in kept cells are silently replaced with 0.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
...
active_trace = np.nan_to_num(active_trace, nan=0.0)
```

iii. The notes justify the all-NaN filter as removing unregistered cells: "A cell is registered if its trace is not all NaN." The zero-fill step is justified only as "safety."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological or task event alignment. The code aligns each trial to the start of an artificial 1-minute segment within a continuous session, and the metadata explicitly describes that segment start as the alignment event.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
    'off_start': 0.0,
    'off_end': float(trial_duration_sec),
    ...
}
```

iii. The notes say the original data are continuous and that "1-minute trials" were imposed by the task; the alignment label was introduced by the AI to satisfy the metadata field.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keeps the native 30 Hz sampling rate, corresponding to 33.33 ms bins, with no temporal rebinning.

ii.
```python
fps = 30  # recording frame rate
...
'time_bin_size': 1000.0 / fps,  # ~33.33 ms
```

iii. The notes state "Time bin size = 1/30 s ≈ 33.33 ms (raw frame rate)" and "No additional binning of neural data."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `envs` array, not from `blocked`. For each session/day, the environment name string is mapped to a 3×3 geometry matrix.

ii.
```python
envs = d['envs'].flatten()  # (n_days,) string array
...
env_name = str(envs[day])
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The notes justify this by saying "`get_env_mat()` returns 3x3 binary matrix: 1=accessible partition, 0=blocked. This will be our decoder input," and later call it "same function from reference code."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is converted with `get_env_mat()` to a 3×3 binary accessibility map, flattened to length 9, cast to `float32`, and reused as a static per-trial vector.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
trial_input = env_mat.astype(np.float32)
```

iii. The notes justify this as reusing reference code and representing "which partitions are accessible" for each geometry. The AI chose accessibility coding rather than blocked-location one-hot coding.

## 3-c. How is `input` *Environment geometry* aligned with the neural data?

i. The input is static for a session/day, and the same 9-element vector is attached to every 1-minute trial from that session.

ii.
```python
env_mat = get_env_mat(env_name).flatten()
...
for trial_idx in range(n_trials):
    ...
    trial_input = env_mat.astype(np.float32)
    trials_input.append(trial_input)
```

iii. The notes justify this with "Static per trial" and "Environment geometry ... doesn't change within a trial."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the per-day `position` array, using the x and y coordinates for each frame.

ii.
```python
position = d['position'] # (n_days, 2, n_frames)
...
pos_day = position[day]      # (2, n_frames)
```

iii. The notes describe `position` as the DeepLabCut-tracked x-y coordinates in `[0, 75]` cm and identify it as the source of the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The x and y coordinates are clipped to the arena bounds, divided into 25 cm bins along each axis, converted with `floor`, and combined into a single class ID with the formula `x_bin * 3 + y_bin`.

ii.
```python
bin_size = env_size / 3.0
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
x_bin = np.clip(x_bin, 0, 2)
y_bin = np.clip(y_bin, 0, 2)
bin_ids = x_bin * 3 + y_bin
```

iii. The notes justify this as matching the task’s "3 x 3 = 9 spatial bins" requirement and say it "gives 9 spatial bins matching the 3x3 partition structure."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded at 25 cm and 50 cm, producing three bins per axis. The two per-axis bins are then collapsed into one of nine integer categories using `x_bin * 3 + y_bin`.

ii.
```python
bin_size = env_size / 3.0
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
...
bin_ids = x_bin * 3 + y_bin
...
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes explicitly state "Position (x,y) in [0, 75] cm → 3x3 grid of 25 cm bins" and "Combine x_bin and y_bin into single label: `bin_id = x_bin * 3 + y_bin`."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The output bins are computed frame-by-frame from `position`, then sliced with the same `start:end` indices used for neural trials, so output and neural data are aligned within each trial.

ii.
```python
bin_ids = discretize_position_3x3(pos_day)  # (n_frames,)
...
trial_neural = active_trace[:, start:end].astype(np.float32)
...
trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes justify this by saying all streams are synchronous at 30 Hz and the trial-boundary spot checks showed "no overlap, no gaps."

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms per sample (30 Hz), and no rebinning is applied.

ii.
```python
fps = 30  # recording frame rate
...
'time_bin_size': 1000.0 / fps
```

iii. The notes say "Use raw binary trace (0/1 events) at native 30 Hz" and "Time bin size = 1/30 s."

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are aligned by identical frame indices within each day/session. The input is static for the whole session/day and duplicated per trial, so it is aligned at the trial level rather than frame-by-frame.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes justify this with "All streams synchronous at 30 Hz" and "Environment geometry as input ... static per trial."

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The code handles minor issues by removing all-NaN cells, replacing remaining NaNs in kept cells with zero, clipping out-of-range positions into the valid arena range, dropping any partial trailing trial, and returning an all-NaN geometry matrix if an unknown environment name is encountered.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
...
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
...
n_trials = n_frames_total // trial_duration_frames
```

iii. The explicit justifications in the notes are sparse. They say zero-filling "shouldn't happen ... but safety," describe discarded remainder frames as acceptable, and treat unknown-geometry fallback as part of the helper copied from reference code.

## 7-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify per-animal loading as the main cost, with processing the loaded sessions second. The code also spends extra time in optional plotting when `--show-processing` is enabled.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal_name))
...
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s", flush=True)
...
if show_processing:
    plot_processing(...)
```

iii. In Step 7 of the notes, the AI recorded: "Load 1 animal ~13s," "Process 31 sessions ~9s," and concluded "Well under 15-minute threshold, no optimization needed."

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop is the clearest candidate. It repeatedly slices arrays and appends identical static input vectors trial-by-trial instead of using a shared trial-splitting helper or bulk construction. The plotting code also loops through cells and figure panels.

ii.
```python
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    trial_neural = active_trace[:, start:end].astype(np.float32)
    trial_input = env_mat.astype(np.float32)
    trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)
    trials_neural.append(trial_neural)
    trials_input.append(trial_input)
    trials_output.append(trial_output)
...
for i in range(n_show):
    spikes = np.where(trial0_neural[i] > 0)[0]
    ax.scatter(...)
```

iii. The notes did not explicitly discuss these loops as bottlenecks. Instead, they argued optimization was unnecessary because runtime was acceptable.

## 7-c. What processing does the code repeat multiple times?

i. The code repeatedly casts the same static geometry vector to `float32` for every trial, recalculates trial boundaries inside a Python loop, and duplicates the same per-session input across all trials rather than sharing a single object or constructing it in one step.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
for trial_idx in range(n_trials):
    start = trial_idx * trial_duration_frames
    end = start + trial_duration_frames
    ...
    trial_input = env_mat.astype(np.float32)
    trials_input.append(trial_input)
```

iii. The notes again justify this only indirectly by saying the conversion was fast enough and "no optimization needed."

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script can generate processing plots that are not part of the saved dataset, stores `sessions_env` only for logging/plotting, computes some summary statistics solely for console output, and imports unused modules (`sys`, `ListedColormap`). These do not affect the final pickle.

ii.
```python
import sys
from matplotlib.colors import ListedColormap
...
sessions_env = []
...
sessions_env.append(env_name)
...
if show_processing:
    plot_processing(...)
...
print(f"Trials per session: {[len(s) for s in all_neural[:5]]}...")
```

iii. The notes explicitly required visualization in `--show-processing` mode and emphasized validation/inspection, so this extra work was justified as verification rather than core conversion.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: all-NaN cells are removed, residual NaNs are zero-filled, positions are clipped to arena bounds, partial end-of-session fragments are dropped, and unknown environment names would yield an all-NaN geometry matrix.

ii.
```python
active_mask = ~np.all(np.isnan(trace_day), axis=1)
active_trace = trace_day[active_mask]
active_trace = np.nan_to_num(active_trace, nan=0.0)
...
x = np.clip(position[0], 0, env_size - 1e-10)
y = np.clip(position[1], 0, env_size - 1e-10)
...
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
```

iii. The same justifications apply: mostly implicit safety handling, with only the all-NaN-cell removal and dropped tail frames described clearly in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading each animal file dominates the runtime, with per-session processing next and plotting as optional extra overhead.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal_name))
...
if show_processing:
    plot_processing(...)
```

iii. The notes’ timing table is the direct justification.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the per-trial construction loop and the plotting loops are the main vectorization candidates.

ii.
```python
for trial_idx in range(n_trials):
    ...
    trials_neural.append(trial_neural)
    trials_input.append(trial_input)
    trials_output.append(trial_output)
```

iii. The notes did not identify these loops as problems because the measured runtime was considered acceptable.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: repeated casting and duplication of the same environment vector across all trials, plus repeated boundary calculations inside the loop.

ii.
```python
trial_input = env_mat.astype(np.float32)
trials_input.append(trial_input)
```

iii. The notes justify leaving this repetition in place because the run time was still within the AI’s target.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: optional plotting, extra logging/statistics, and some retained intermediate bookkeeping are not used by downstream decoder training.

ii.
```python
if show_processing:
    plot_processing(...)
...
print(f"Neurons per session: {[all_neural[i][0].shape[0] for i in range(min(5, total_sessions))]}...")
```

iii. The notes frame these as validation and documentation features rather than essential conversion work.
