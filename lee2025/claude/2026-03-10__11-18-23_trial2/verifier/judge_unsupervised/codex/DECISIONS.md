# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes the 7 animal IDs, loads each animal's joblib file from `data/`, indexes the top-level dict by animal name, then iterates over every day/session. Trial lists are created inside `process_session()` after per-session processing.

ii. ```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    n_days = animal_data['trace'].shape[0]

    for day in range(n_days):
        neural_trials, input_trials, output_trials, n_valid = process_session(
            animal_data, day, ...
        )
```

iii. In `CONVERSION_NOTES.md`, the agent says this matches the reference `load_dat(..., format="joblib")` path and that each day should be treated as one session from each mouse.

## 1-b. How are the data split into subjects?

i. Subjects are split by the outer loop over `ANIMALS`. The output `subjects` list is the ordered animal list, and `subject_idx` stores the animal index once per retained session.

ii. ```python
subjects = list(animals_to_process)

for a_idx, animal in enumerate(animals_to_process):
    ...
    for day in range(n_days):
        ...
        all_neural.append(neural_trials)
        ...
        subject_idx_list.append(a_idx)
```

iii. The agent explicitly justified `subject = mouse` and used the paper's 7 named mice as the canonical subject IDs.

## 1-c. How are the data split into sessions?

i. Sessions are split by recording day. The code treats each `day_idx` slice of `trace`, `position`, and `envs` as one session.

ii. ```python
trace = animal_data['trace'][day_idx]
position = animal_data['position'][day_idx]
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
```

iii. The agent's notes say "Session = day" and cite the paper's description of one recording session per day.

## 1-d. How are the data split into trials?

i. Each session is split into contiguous 1-minute trials after temporal binning. At 30 Hz and 3-frame bins, a trial is 1800 raw frames or 600 binned timepoints. Any trailing partial trial is dropped.

ii. ```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE

n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The agent justified this directly from the task instructions, which required splitting long sessions into 1-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality-control filtering. The only related check is session-level: sessions with fewer than 2 generated trials are skipped.

ii. ```python
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    ...
    continue
```

iii. The agent wrote that the reference had "No explicit trial filtering" and therefore kept all 1-minute segments, aside from the target-format requirement that each session must contain at least 2 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the raw `trace` array for each day/session.

ii. ```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The agent's notes state that `trace` is already a binarized calcium-event representation and that no new dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The script keeps session-valid cells, smooths each cell's trace with a Gaussian filter over time, then average-pools into 3-frame bins and stores the result as float32.

ii. ```python
def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)
```

iii. The justification in the notes is that this matches the reference `fit_decoder` temporal processing: Gaussian smoothing with `sigma=3` followed by 3-frame pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC is dropping cells whose first sample is `NaN`, which the agent interprets as cells not registered on that day. It does not apply the reference decoder's moving-period filter or its per-session activity threshold.

ii. ```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
...
if n_valid == 0:
    return [], [], [], 0
```

iii. The agent justified this by saying "Include only registered cells per session (not NaN). No velocity filtering" and "No place cell filtering." Its notes acknowledge that the reference decoder used velocity and activity thresholds, but the script does not implement them.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to the start of the recording session, then split into fixed 1-minute windows. There is no stimulus or behavioral event alignment.

ii. ```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
}
```

iii. The agent's notes say the task involved continuous free exploration with no better explicit event, so it chose session start as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. Yes: the script rebins from 30 Hz frames into non-overlapping 3-frame bins after Gaussian smoothing.

ii. ```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
```

iii. The justification is that the reference decoder code used `temporal_bin_size=3`, so 3 frames at 30 Hz should be preserved.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The script derives environment geometry from `envs[day]`, not from the raw `blocked` field.

ii. ```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
env_input = get_env_input(env_name)
```

iii. The agent justified this by referencing the paper code's `get_env_mat(env)` helper and treating the environment name as the canonical lookup key for geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The script looks up a hard-coded 3x3 binary geometry matrix for the named environment and flattens it into a 9-element float vector.

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
```

iii. The notes say this was copied from the reference `get_env_mat` logic so that the decoder input reflects which 3x3 partitions are blocked.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. It is static within a trial and repeated once per trial of a session. Alignment is by session/day and the same 1-minute trial segmentation used for neural/output data.

ii. ```python
env_input = get_env_input(env_name)  # (9,)
...
input_trials.append(env_input)  # (9,) static
```

iii. The agent justified this by noting that environment geometry does not change within a recording session, so a per-trial static input is sufficient.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the raw `position` array for each day/session.

ii. ```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The agent's notes map `position[day, :, :]` directly to the decoder output after binning and discretization.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The script first temporally bins x/y position by averaging within each 3-frame window, then discretizes the binned coordinates into a 3x3 spatial grid.

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

iii. The agent justified this as matching the reference decoder's behavior binning before adapting the position output to the task's required 3x3 categories.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each x and y coordinate is floored after division by 25 cm, clipped into `[0, 2]`, and combined into a single class index `x_bin * 3 + y_bin`, producing labels 0 through 8.

ii. ```python
def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. The justification in the notes is that the task explicitly required 3 x 3 = 9 spatial bins for mouse position.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output is aligned by using the same source frame axis, the same 3-frame temporal bins, and the same fixed 1-minute trial windows as the neural data.

ii. ```python
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)

trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The agent justified this as direct temporal alignment from shared session time, without any additional event-based alignment.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are in 100 ms bins, produced by 3-frame temporal rebinning at 30 Hz after neural smoothing and by 3-frame averaging for position.

ii. ```python
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000
```

iii. The agent cites the reference decoder's `temporal_bin_size=3` as the reason for this choice.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output use the same session frames, the same 3-frame bins, and the same 1-minute trial boundaries. Input is static environment geometry copied into each trial of that session.

ii. ```python
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
...
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    neural_trials.append(binned_trace[:, start:end])
    input_trials.append(env_input)
    output_trials.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The agent's justification is that all three streams should inherit the same continuous session timeline, with session start as the anchor.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The script handles missing cells by filtering `NaN`-marked unregistered cells, squeezes odd `envs` container shapes into strings, raises an error on unknown environment names, and silently trims leftover frames/bins that do not make a full bin or full trial.

ii. ```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
if isinstance(env_name, np.ndarray):
    env_name = str(env_name.squeeze())

valid_mask = ~np.isnan(trace[:, 0])
...
trimmed = smoothed[:, :n_bins * bin_size]
...
if mat is None:
    raise ValueError(f"Unknown environment: {env_name}")
```

iii. The agent justified the `NaN` handling from the dataset structure notes, where `NaN` indicated cells not registered on a given day. Beyond that, the script uses minimal defensive handling.

## 7-a. What are the most time-consuming steps of the code?

i. The heavy steps are loading each large joblib animal file, Gaussian smoothing over all valid cells and all frames for every session, and storing the full per-trial arrays for the entire dataset.

ii. ```python
dat = joblib.load(os.path.join(data_dir, animal))
...
binned_trace = temporal_bin_trace(valid_trace)
...
all_neural.append(neural_trials)
```

iii. The agent's runtime notes estimate about 4.3 minutes for the full pass and repeatedly print per-animal/per-day timings, implying file I/O plus full-session smoothing/binning were the dominant costs.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop could be replaced by reshaping sessions into `(n_trials, ...)` blocks, and the small loop that constructs output value names is also manually iterative.

ii. ```python
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    ...

output_values_position = []
for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        output_values_position.append(f"row{i}_col{j}")
```

iii. The agent did not justify these loops as necessary; they are straightforward implementation choices that could be vectorized or precomputed.

## 7-c. What processing does the code repeat multiple times?

i. It repeatedly converts `envs[day]` into strings, recomputes trial slicing in Python for every session, and duplicates static `env_input` objects into every trial list entry. In `convert_dataset()`, it also computes `env_name` once for logging even though `process_session()` recomputes it.

ii. ```python
env_name = str(animal_data['envs'][day].squeeze())
neural_trials, input_trials, output_trials, n_valid = process_session(...)
```

iii. There is no explicit justification for the repeated work; it appears to be convenience-driven rather than required by the reference pipeline.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional plotting branch builds full figure panels that are not part of the dataset, and continuous `binned_pos` is computed even though only discrete `pos_bins` is ultimately saved.

ii. ```python
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
...
if show_processing and fig_axes is not None and n_trials > 0:
    ...
    ax_pos.plot(time_axis, binned_pos[0, t_start:t_end], ...)
```

iii. The agent added this work to satisfy the `--show-processing` requirement and for manual verification, not because downstream training consumed it.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: the script filters `NaN`-marked missing cells, coerces environment labels into strings, errors on unknown environment names, and drops incomplete tail bins/trials by trimming.

ii. ```python
valid_mask = ~np.isnan(trace[:, 0])
...
trimmed = smoothed[:, :n_bins * bin_size]
...
n_trials = n_total_bins // BINS_PER_TRIAL
```

iii. The agent's justification is implicit in its dataset notes: `NaN` marks non-registered cells, while all other edge handling is pragmatic and minimal.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: large file loads, full-session Gaussian smoothing/temporal binning, and building the full in-memory nested output structure dominate runtime.

ii. ```python
dat = joblib.load(os.path.join(data_dir, animal))
...
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
```

iii. The agent's timing logs and runtime estimate support this conclusion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the session-to-trial slicing loop and the small label-generation loops are the clearest candidates.

ii. ```python
for t in range(n_trials):
    ...

for i in range(N_SPATIAL_BINS):
    for j in range(N_SPATIAL_BINS):
        ...
```

iii. No substantive justification was given for keeping these loops unvectorized.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: repeated environment-name coercion, repeated trial slicing, repeated static-input insertion, and redundant `env_name` computation across `convert_dataset()` and `process_session()`.

ii. ```python
env_name = str(animal_data['envs'][day].squeeze())
...
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
```

iii. This repeated work is not tied to any reference requirement.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: optional plotting work and continuous binned positions are computed for inspection even though the saved dataset only uses discrete position classes.

ii. ```python
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
...
if show_processing and fig_axes is not None and n_trials > 0:
    ...
```

iii. The agent's justification was validation and visualization rather than downstream necessity.
