# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, then iterates over them and loads each animal's joblib file from `data/<animal>`. Within each loaded animal dict, it treats `dat[animal]` as the source object and then iterates over every day/session. Trials are not loaded from disk directly; they are created in memory later by splitting each session.

ii. ```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    n_days = animal_data['trace'].shape[0]
    ...
    for day in range(n_days):
        neural_trials, input_trials, output_trials, n_valid = process_session(
            animal_data, day, ...
        )
```

iii. In `CONVERSION_NOTES.md`, the agent says each animal has a joblib file in `data/` keyed by animal name and that the dataset contains 7 subjects and 207 sessions. In the trajectory, it also identified `load_dat` in the reference code but chose direct `joblib.load(...)` because the joblib files already had the needed structure.

## 1-b. How are the data split into subjects?

i. Subjects are split by the outer loop over `ANIMALS`. The converted dataset stores a `subjects` list in that same order and appends one `subject_idx` entry per retained session/day.

ii. ```python
subjects = list(animals_to_process)

for a_idx, animal in enumerate(animals_to_process):
    ...
    for day in range(n_days):
        ...
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        subject_idx_list.append(a_idx)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent explicitly states "Session = day" and that "Each animal contributes multiple sessions," which implies subject identity is the animal name and session ownership is carried by `subject_idx`.

## 1-c. How are the data split into sessions?

i. The agent equates one recording day with one session. For each loaded animal, it uses `animal_data['trace'].shape[0]` as the number of sessions and processes each `day` separately.

ii. ```python
n_days = animal_data['trace'].shape[0]

for day in range(n_days):
    neural_trials, input_trials, output_trials, n_valid = process_session(
        animal_data, day, ...
    )
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent's first key decision is "Session = day." The notes tie this to the paper/methods statement that one session was recorded per day.

## 1-d. How are the data split into trials?

i. Within each day/session, the agent first temporally bins the session to 100 ms bins, then splits the full session into consecutive 1-minute trials. Because 60 seconds at 30 Hz is 1800 frames, and the temporal bin size is 3 frames, each trial contains 600 time bins.

ii. ```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE

n_total_bins = binned_trace.shape[1]
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The justification is explicit in `CONVERSION_NOTES.md` Step 5: "Trial = 1-minute segment" because the task instructions required splitting long sessions into 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent does almost no trial-level quality filtering. It drops sessions with zero valid cells by returning empty lists from `process_session`, and it skips any processed day with fewer than 2 resulting trials. It also implicitly drops incomplete trailing data by floor-dividing the session length into full 1-minute trials.

ii. ```python
valid_mask = ~np.isnan(trace[:, 0])
...
if n_valid == 0:
    return [], [], [], 0
...
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    ...
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    continue
```

iii. In `CONVERSION_NOTES.md` Step 3, the agent wrote "No explicit trial filtering in the reference (entire sessions used)." In Step 5 it separately notes the format requirement that each session must have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived only from the raw `trace` array for each day. The code does not use `maps`, `SFPs`, `centroids`, or any calcium fluorescence variable.

ii. ```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
...
valid_trace = trace[valid_mask]
...
binned_trace = temporal_bin_trace(valid_trace)
```

iii. The notes say the reference dataset already provides binarized calcium-event traces and that "No dF/F needed." The trajectory also captured the paper excerpt stating that the final binarized rising-phase vector was treated as the firing rate in further analyses.

## 2-b. How is the `neural` data processed?

i. The agent treats `trace` as already binarized events, removes unregistered cells, applies Gaussian smoothing along time with `sigma=3` frames, then average-pools into non-overlapping 3-frame bins and casts the result to `float32`.

ii. ```python
def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says this was meant to match reference `fit_decoder`, and the trajectory captured the reference code line that bins traces with `AvgPool1d(... stride=3)` after `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter applied in the conversion is removing cells whose trace is `NaN` on that day, which the agent interprets as "not registered." It does not apply place-cell filtering, a movement-only mask, or the reference decoder's activity threshold of more than 5 events during moving periods.

ii. ```python
# Identify valid (registered) cells for this day
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
n_valid = valid_mask.sum()
```

iii. The notes justify this directly: "Cell filtering: Include only registered cells per session (not NaN). No velocity filtering (that's decoder-specific)." The same notes also acknowledge that the reference decoder used `v_thresh=5 cm/s` and a `>5 events during moving periods` cell threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural trials to the start of each recording session, not to a behavioral event. Trial 0 begins at session start, and later trials are consecutive non-overlapping 1-minute windows from that same origin.

ii. ```python
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]

'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
}
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says "Temporal alignment: Trials start at beginning of session recording. Align to session start." That choice came from the task-defined 1-minute trialization rather than a paper-defined trial event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 100 ms time bins. Yes, temporal rebinning is applied: three 30 Hz frames are Gaussian-smoothed and then pooled into one output bin.

ii. ```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms

smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
...
binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
```

iii. The notes repeatedly justify this from the reference decoder: "Temporal binning in decoder: 3 frames at 30Hz = 100ms bins."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The agent derives environment geometry from `animal_data['envs'][day]`, not from the raw `blocked` field. It uses the environment name string as a lookup key into a hard-coded 3x3 geometry table.

ii. ```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
env_input = get_env_input(env_name)
```

iii. The notes say the mapping is based on reference `get_env_mat`, and the trajectory shows the agent inspected both `envs` and `blocked` and concluded that `blocked` indicates the omitted partitions while `get_env_mat` converts the environment string into the binary 3x3 matrix.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent uses a hard-coded dictionary of 3x3 binary masks, one per environment name, and flattens the selected matrix into a length-9 static vector stored once per trial.

ii. ```python
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    ...
}

def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    if mat is None:
        raise ValueError(f"Unknown environment: {env_name}")
    return mat.flatten().astype(np.float32)
...
input_trials.append(env_input)  # (9,) static
```

iii. `CONVERSION_NOTES.md` Step 5 states that `envs[day] -> get_env_mat()` is flattened to 9 values and used as static per-trial input. The trajectory also captured the reference `get_env_mat` docstring saying it returns a binary 3x3 matrix with zeros for omitted partitions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived only from the raw `position` array for each day.

ii. ```python
position = animal_data['position'][day_idx]  # (2, n_frames)
...
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
```

iii. The notes map `position[day, :, :]` directly to the output field and identify it as x-y position in centimeters.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent first averages position within each 3-frame temporal bin, then converts each binned x-y sample into a discrete 3x3 spatial-bin index.

ii. ```python
def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned

binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent says position should be time-varying, temporally binned to match the reference decoder, then discretized into a 3x3 grid because the task required 9 categories instead of the paper's 15x15 map.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The agent divides each x and y coordinate by 25 cm, floors the result, clamps it into `[0, 2]`, and combines the two indices into a single class `x_bin * 3 + y_bin`, giving category labels `0` through `8`.

ii. ```python
SPATIAL_BIN_SIZE = ARENA_SIZE_CM / N_SPATIAL_BINS  # 25 cm per bin

def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. The notes explicitly justify this as the task-specific discretization rule for turning continuous 75 cm x 75 cm position into 3 x 3 = 9 categories.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position outputs are aligned by applying the same 3-frame temporal binning and then slicing both arrays with the same trial boundaries. Each trial uses the same `[start:end]` indices for binned neural activity and binned position labels.

ii. ```python
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
...
start = t * BINS_PER_TRIAL
end = (t + 1) * BINS_PER_TRIAL
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say temporal alignment is to session start and that both neural and behavior use the same 100 ms bins matching the reference decoder.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing neural registrations are handled by dropping cells with `NaN` in the first trace sample for that day. Completely empty sessions return no trials. Unknown environment names raise an exception rather than falling back to a placeholder matrix. Partial end-of-session bins/trials are silently truncated by integer division. The code does not add explicit handling for missing position samples.

ii. ```python
valid_mask = ~np.isnan(trace[:, 0])
...
if n_valid == 0:
    return [], [], [], 0
...
mat = ENV_MATRICES.get(env_name)
if mat is None:
    raise ValueError(f"Unknown environment: {env_name}")
...
n_bins = n_frames // bin_size
trimmed = smoothed[:, :n_bins * bin_size]
...
n_trials = n_total_bins // BINS_PER_TRIAL
```

iii. The notes justify only the NaN-based cell removal and trial-count requirement. There is no stronger justification in the notes for missing-position handling or for raising on unknown environments; the trajectory instead shows the agent copied the environment table manually rather than reusing the reference function's NaN fallback.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the repeated `joblib.load` of very large animal files, per-session Gaussian smoothing and temporal binning across all valid cells and frames, appending thousands of large trial arrays into nested Python lists, and writing the multi-gigabyte pickle. Optional plotting is another avoidable cost when enabled.

ii. ```python
dat = joblib.load(os.path.join(data_dir, animal))
...
binned_trace = temporal_bin_trace(valid_trace)
...
for t in range(n_trials):
    ...
    neural_trials.append(trial_neural)
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes only give runtime estimates, not a detailed hotspot analysis. The runtime table in Step 7 implies the agent understood the heavy steps were full-animal processing and full-dataset serialization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that slices and appends `neural`, `input`, and `output` one trial at a time could have been reorganized into pre-shaped arrays or batched views. The nested loop that builds `output_values_position` is minor but also unnecessary as an explicit Python loop.

ii. ```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
    neural_trials.append(trial_neural)
    input_trials.append(env_input)
    output_trials.append(trial_output)

output_values_position = []
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        output_values_position.append(f"row{i}_col{j}")
```

iii. The agent did not explicitly justify these loops in the notes. They appear to have been written for simplicity and compatibility with the required nested list format.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly extracts `env_name` both in `convert_dataset` and again inside `process_session`. It also repeats the same per-session workflow for every day: load day arrays, find valid cells, smooth/bin neural data, bin/discretize position, and then slice into 1-minute trials.

ii. ```python
env_name = str(animal_data['envs'][day].squeeze())
neural_trials, input_trials, output_trials, n_valid = process_session(...)
```

```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
```

iii. There is no explicit justification beyond the agent's general plan in `CONVERSION_NOTES.md`: process each animal and each day in the same way to keep the converted dataset consistent with the reference pipeline.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional visualization branch computes and saves processing figures that are not used by downstream decoder analyses. The script also computes timing/summary statistics and metadata counters that are useful for validation but not required by training. It inserts the reference code path but does not actually import or call the reference function it mentions.

ii. ```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'georepca1', 'src'))
...
if show_processing and fig_axes is not None and n_trials > 0:
    ...
    ax_neural.imshow(...)
    ax_pos.plot(...)
    ax_output.plot(...)
    ax_env.imshow(...)
...
print(f"Total conversion time: {total_time:.1f}s")
```

iii. The notes justify the plots as sanity-check visualizations during development, not as part of the final decoded representation. They do not claim this processing is needed by downstream analyses.
