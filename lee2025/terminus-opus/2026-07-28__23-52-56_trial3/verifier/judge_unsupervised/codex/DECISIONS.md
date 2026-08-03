# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven mouse IDs, loops over them, and loads each animal directly from the joblib file in `data/`. Inside each animal file it only reads `trace`, `position`, and `envs`, then converts each recording day into one session and each session into 1-minute trials.

ii. ```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for a_idx, animal in enumerate(animals_to_process):
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
```

```python
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]

trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)
```

iii. In `CONVERSION_NOTES.md`, the agent says the reference `load_dat` function loads joblib files and that the non-`.mat` files in `data/` are the intended preprocessed dataset. The notes also say the target mapping only needs `trace`, `position`, and `envs`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the outer loop over the hard-coded animal list. Each animal contributes multiple sessions, and `subject_idx` stores the current animal-loop index once per session.

ii. ```python
for a_idx, animal in enumerate(animals_to_process):
    ...
    for sess in sessions:
        all_neural.append(sess['neural'])
        all_input.append(sess['input'])
        all_output.append(sess['output'])
        all_subject_idx.append(a_idx)
```

```python
'subjects': [a for a in animals_to_process],
'subject_idx': np.array(all_subject_idx, dtype=int),
```

iii. The notes explicitly say "Each animal contributes multiple sessions" and list the per-animal session counts `(31,31,31,21,31,31,31)`, which the agent later checks against `subject_idx`.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day in the raw arrays as one session. It iterates over `range(n_sessions)` where `n_sessions = trace.shape[0]`.

ii. ```python
n_sessions = trace.shape[0]

for day in range(n_sessions):
    env_name = envs[day, 0]
    tr = trace[day]
    pos = position[day]
```

iii. The notes say "Session definition: Each day/recording is one session" and that this matches the paper's "one session was recorded per day."

## 1-d. How are the data split into trials?

i. Each continuous session is cut into consecutive 1-minute windows at 30 Hz, so each trial is exactly 1800 frames. The number of trials is `n_timepoints // 1800`, so any leftover tail shorter than a minute is dropped.

ii. ```python
FPS = 30
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
...
n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
```

iii. The notes say "Trial definition: Split each 40-min session into 1-minute trials (1800 frames at 30fps)." The trajectory shows the agent chose this because the decoder task explicitly requested 1-minute trials even though the source data are continuous sessions.

## 1-e. How are trials filtered based on quality controls?

i. There is no substantive per-trial quality-control filter. Trials are kept if they are full 1800-frame chunks; there is no velocity-based filtering, no removal of low-quality trials, and no trial rejection based on neural or behavioral quality.

ii. ```python
n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes explicitly justify this by saying "Velocity filtering: NOT applied at data conversion stage" and "Cell filtering: NOT applied at conversion." The only implicit filter is dropping leftover frames that do not make a full minute.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived entirely from `d['trace']`, the per-session calcium transient trace array.

ii. ```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
...
tr = trace[day]
active_neurons = tr[active_mask]
```

iii. The notes identify `trace` as "binary calcium transient data" and say the target `neural` field should use the raw binary trace with NaN filtering.

## 2-b. How is the `neural` data processed?

i. The agent does minimal processing: select the current day's trace matrix, remove neurons that are all-NaN for that day, replace any remaining NaNs with zero, cast to `float32`, and slice into 1-minute trial matrices. It does not recompute the rising-phase binarization, smooth the signal, or temporally average frames.

ii. ```python
tr = trace[day]  # (n_neurons, n_timepoints)
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The notes say "Neural data: Use raw binary trace data (0/1) for active neurons. No additional temporal binning at this stage." The trajectory also summarizes the paper methods as saying the stored trace is already the binarized rising-phase vector treated as firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC filter actually implemented is dropping neurons that are all-NaN on a given day. The agent does not apply the reference decoder's movement-only activity threshold (`cell_threshold=5`) and does not restrict to split-half reliable place cells.

ii. ```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]
...
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. In the notes the agent explicitly records the reference choices "cell_threshold=5" and place-cell reliability p-values, then explicitly decides "Cell filtering: NOT applied at conversion. Include all active (non-NaN) neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align trials to an experimental event from the paper. Instead, it creates arbitrary 60-second contiguous windows from the recording stream. In metadata it labels the alignment event as `"Start of recording session"`, but in practice each trial is just aligned to its own slice start index.

ii. ```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

```python
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
}
```

iii. The notes say there is "No explicit trial structure in original data (continuous 40-min sessions)." The agent therefore chose a synthetic segmentation and documented session start as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native 30 Hz resolution, so each bin is `1000/30 = 33.33 ms`. No temporal rebinning is applied in the conversion script.

ii. ```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

```python
# Neural data: Use raw binary trace data (0/1) for active neurons.
# No additional temporal binning at this stage (decoder handles that).
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The notes explicitly contrast this with the reference decoder: "Temporal binning: 3 frames in fit_decoder" but say "Keep original 30Hz resolution."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment input is derived from `d['envs']`, specifically the per-session environment name string `envs[day, 0]`. The `blocked` field is not used.

ii. ```python
envs = d['envs']         # (n_sessions, 1)
...
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The notes say the reference code uses `get_env_mat(env_name)` and explicitly resolves a discrepancy between `blocked` and the canonical environment matrix by choosing the `get_env_mat` route.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The session's environment name is mapped through `get_env_mat` to a canonical 3x3 occupancy matrix, then flattened to a 9-element vector and repeated once per trial as a static input.

ii. ```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
trial_input.append(env_mat.astype(np.float32))
```

iii. The notes justify this as using the "canonical environment representation" from the reference code and say the decoder input should be a 9-element static vector describing which parts of the 3x3 arena are open.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position variable is derived from `d['position']`, the per-session `(x, y)` trajectory.

ii. ```python
position = d['position'] # (n_sessions, 2, n_timepoints)
...
pos = position[day]  # (2, n_timepoints)
pos_bins = position_to_bin(pos)  # (n_timepoints,)
```

iii. The notes identify `position` as head position in centimeters tracked by DeepLabCut and map it directly to the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Continuous `(x, y)` coordinates in the 75 cm arena are discretized into a 3x3 grid. Each axis is floored into bins of width 25 cm, clipped to `[0, 2]`, and combined into a single categorical label from 0 to 8.

ii. ```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. The notes say this was chosen because the task overrides the paper's 15x15 decoder target and instead requires "Mouse position discretized into 3x3 = 9 spatial bins."

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholds are set implicitly by the 25 cm bin width: `[0,25)`, `[25,50)`, and `[50,75]` on each axis after clipping. The two axis bins are then combined into 9 categorical states.

ii. ```python
BIN_SIZE_CM = ENV_SIZE_CM / N_SPATIAL_BINS  # 25cm per bin
...
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. The notes make this explicit: "Position (0-75cm) / 25 = 0,1,2 for each axis. Combined bin = row*3 + col = 0-8."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are aligned by taking the same session/day and the same frame indices for both streams. No additional lag correction or resampling is applied.

ii. ```python
pos = position[day]  # (2, n_timepoints)
pos_bins = position_to_bin(pos)  # (n_timepoints,)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes summarize the raw data as behavior and calcium acquired at 30 Hz with synchronized frames, so the agent kept the original framewise alignment and used identical `start:end` slices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The script removes neurons whose whole-day trace is NaN, replaces any remaining neural NaNs with zero, clips out-of-range position values into the valid 3x3 bin range, and silently drops leftover session frames that do not fill a complete 1-minute trial. It does not add any dedicated handling for missing position samples.

ii. ```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

```python
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
```

iii. The notes mention NaN neurons as "not tracked on a given day" and later record edge-case checks for boundary clipping and for 39-vs-40 trial session lengths. There is no fuller missing-data policy in the notes.

## 6-a. What are the most time-consuming steps of the code?

i. The heavy steps are loading each large joblib animal file, looping over every session and trial to materialize trial lists, and finally pickling the full 19.98 GB converted dataset. Optional plotting is also nontrivial when enabled.

ii. ```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_sessions):
    ...
    for t in range(n_trials):
        ...
```

```python
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes' runtime table says load time and processing time dominate per animal, and the full-output section reports a 19.98 GB pickle, confirming that loading and saving are major costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_animal` could largely be replaced by reshaping/splitting arrays once per session. The later loop that rebuilds `all_bins` for the output-distribution summary is also avoidable with more direct concatenation. The plotting code also loops over neurons and trials in Python.

ii. ```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

```python
all_bins = []
for sess_outputs in all_output:
    for trial_out in sess_outputs:
        all_bins.append(trial_out[0])
all_bins = np.concatenate(all_bins)
```

iii. The agent did not record an optimization rationale here; this is inferred from the code structure and from the runtime estimates in the notes.

## 6-c. What processing does the code repeat multiple times?

i. It repeatedly casts the same session-level environment vector to `float32` once per trial, repeatedly slices fixed-size windows in Python rather than reshaping once, and later recomputes aggregate bin distributions by concatenating outputs that were already constructed.

ii. ```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
for t in range(n_trials):
    ...
    trial_input.append(env_mat.astype(np.float32))
```

```python
for sess_outputs in all_output:
    for trial_out in sess_outputs:
        all_bins.append(trial_out[0])
all_bins = np.concatenate(all_bins)
```

iii. No explicit justification is given in the notes. The repeated work is visible directly in the script.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes human-facing diagnostics that are not used by downstream decoding: runtime prints, output-bin distribution summaries, optional plotting, and some intermediate bookkeeping such as `n_neurons_total` and the per-session visualization data path. These do not affect `converted_data.pkl`.

ii. ```python
n_neurons_total = trace.shape[1]
...
if show_processing:
    plot_processing(d, animal, sessions)
```

```python
print(f"Summary:")
...
counts = np.bincount(all_bins.astype(int), minlength=9)
print(f"  Output distribution (9 bins): {counts / counts.sum()}")
```

iii. The notes emphasize validation and sanity checks, so these extra computations were added for verification rather than as part of the converted dataset itself.
