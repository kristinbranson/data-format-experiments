# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a hard-coded list of seven animal files from `/app/data` using `joblib.load`, not the `.mat` files used by the human reference. For each animal, it pulls the nested dictionary `dat[animal]`, then reads `trace`, `position`, and `envs` day by day inside the main conversion loop.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal_idx, animal in enumerate(animals):
    print(f"Loading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
```

iii. In the trajectory, the agent inspected both `.mat` and joblib-style files, then decided the joblib animal files were the most direct structure to work with because they already exposed per-animal arrays such as `trace`, `position`, `envs`, and `blocked` (trajectory steps 23, 27, 37, 45). The notes frame this as using the georepca1 dataset directly rather than reconstructing sessions from HDF5 references.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. The output `subjects` field is simply `list(animals)`, and `subject_idx` is populated from the loop index over that list.

ii.
```python
subjects = list(animals)

for animal_idx, animal in enumerate(animals):
    ...
    subject_idx_list.append(animal_idx)
```

iii. The trajectory and notes both treat the seven named mice as the complete dataset and repeatedly list those subject IDs explicitly. The AI justified this by matching the seven animals reported in the paper and found in the data directory.

## 1-c. How are the data split into sessions?

i. Each day in an animal's arrays is treated as one session. The AI reads `n_days = d['trace'].shape[0]` and iterates `for day in range(n_days)`, producing one output session per day.

ii.
```python
n_days = d['trace'].shape[0]

for day in range(n_days):
    trace_day = d['trace'][day]
    pos_day = d['position'][day]
    env_name = str(d['envs'][day, 0])
```

iii. The agent states several times that "sessions = recording days" because each day has one environment geometry and a full continuous recording (trajectory steps 37 and 45; CONVERSION_NOTES.md "Sessions and Trials").

## 1-d. How are the data split into trials?

i. Within each day/session, the AI creates non-overlapping 1-minute trials at 30 Hz, so each trial is 1800 frames. It computes `n_trials` by integer division and slices `[start:end]` windows for each trial, dropping the remainder at the end of the session.

ii.
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S

n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
```

iii. The AI explicitly justified 1-minute trials from the task instructions and from the observation that each session is about 40 minutes long, which yields about 39 to 40 trials per session (trajectory steps 37, 45; CONVERSION_NOTES.md).

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The AI instead skips whole sessions if they would yield fewer than two 1-minute trials, and it also skips whole sessions with fewer than five registered cells.

ii.
```python
if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue

n_trials = n_timepoints // FRAMES_PER_TRIAL
if n_trials < 2:
    print(f"  Day {day}: skipping, only {n_trials} possible trials")
    continue
```

iii. The notes claim that a minimum of two trials per session is required and "guaranteed" by the session length, while the trajectory shows the agent added a defensive skip anyway. I did not find a separate justification for the `<5` registered-cell session skip beyond it being a safety threshold inserted in code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from the `trace` array in the per-animal joblib dictionary. The AI indexes one day at a time as `trace_day = d['trace'][day]`.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. The trajectory shows the agent inspected the data structure and concluded that `trace` contains the calcium-event time series needed for decoding (trajectory steps 23, 36, 37). The notes describe these as binary calcium transient events.

## 2-b. How is the `neural` data processed?

i. The AI keeps only registered cells, leaves the array in `(neurons, time)` order, fills any remaining NaNs with zero, and casts each per-trial matrix to `float32`. It does not smooth, normalize, or temporally rebin the activity.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)
...
trial_traces = traces[:, start:end]
neural_trials.append(trial_traces.astype(np.float32))
```

iii. The AI justified this by treating the source traces as already preprocessed binary calcium-transient events and by focusing only on registered cells per day (trajectory steps 36, 37, 45; CONVERSION_NOTES.md "Neural Data").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered out if all timepoints are NaN on that day, which the AI interprets as "not registered" for that session. In addition, the code contains a session-level filter that drops any day with fewer than five registered cells.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()

if n_registered < 5:
    print(f"  Day {day}: skipping, only {n_registered} registered cells")
    continue
```

iii. The notes justify the all-NaN filter by saying cell registration varies across days and only registered cells should be included. The notes also claim no further filtering was applied, which does not fully match the code because the code also contains the `<5` registered-cell session skip.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align neural activity to a stimulus or behavioral event. It treats the start of the recording session as the alignment anchor and then cuts the continuous recording into consecutive 1-minute windows.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]

'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_S,
}
```

iii. The notes explicitly say that all series are aligned at the native 30 Hz recording rate and that trials are aligned to the start of each 1-minute segment from session start. The trajectory also shows the agent concluded there was no event-based alignment to reproduce.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data stays at the native 30 Hz frame rate, corresponding to 33.33 ms bins. No temporal rebinning or downsampling is applied.

ii.
```python
FPS = 30  # recording frame rate
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The trajectory and notes both say the data streams are already aligned at 30 Hz and should remain at that native resolution (trajectory step 45; CONVERSION_NOTES.md "Temporal Alignment").

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from `envs`, not from the raw `blocked` field. It reads the environment name for each day and uses that name to look up a hard-coded 3x3 geometry matrix.

ii.
```python
env_name = str(d['envs'][day, 0])
env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()
```

iii. The trajectory shows the agent inspected `blocked`, checked that it matched the named environments, and then decided to use the environment name plus the reference-code geometry lookup because the task asked for "environment geometry" as decoder input (trajectory steps 33, 43, 44, 45).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps each environment name to a hard-coded binary 3x3 matrix where `1` means open and `0` means blocked, then flattens that matrix to a length-9 vector. That vector is static for all trials in the session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    if env in env_mats:
        return np.array(env_mats[env], dtype=float)

env_mat = get_env_mat(env_name)
env_input = env_mat.flatten()  # shape (9,)
```

iii. The AI justified this as directly representing which arena partitions are open versus blocked and stated that `get_env_mat()` was copied from the reference code. The notes also say the geometry matches the physical 3x3 arena partitions.

## 3-c. How is `input` *Environment geometry* aligned with the neural data?

i. The geometry input is not frame-varying. The AI attaches the same length-9 vector to every trial in a session, so alignment is at the session/trial level rather than the frame level.

ii.
```python
for t in range(n_trials):
    ...
    input_trials.append(env_input.astype(np.float32))  # static per trial, shape (9,)
```

iii. The notes repeatedly describe the environment geometry as static within a recording day, so the AI used one constant input vector per trial rather than expanding it across timepoints.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output comes from the `position` array in the joblib animal dictionary, one day at a time via `pos_day = d['position'][day]`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
```

iii. The trajectory shows the agent inspected the position arrays, confirmed a `[0, 75]` cm range, and treated them as the behavioral target for decoding (trajectory step 36; CONVERSION_NOTES.md "Output").

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI clips x and y coordinates to the arena bounds, divides each axis into three equal-width bins, floors the result to integer bins, and combines the two bin indices into a single class label. It then slices the class labels into 1-minute trials and stores each trial as shape `(1, time)`.

ii.
```python
def discretize_position(position, n_bins=N_SPATIAL_BINS, arena_size=ARENA_SIZE):
    x = np.clip(position[0], 0, arena_size - 1e-10)
    y = np.clip(position[1], 0, arena_size - 1e-10)

    bin_size = arena_size / n_bins
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    ...
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx

output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
```

iii. The notes justify the 3x3 discretization by saying each bin is 25 cm by 25 cm and matches the 3x3 arena partitioning. The trajectory likewise describes the decoder output as a categorical 9-bin position signal.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded at 25 cm and 50 cm, producing three bins per axis. The final category index is `x_bin * 3 + y_bin`, which the AI describes as row-major indexing.

ii.
```python
bin_size = arena_size / n_bins
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)

x_bin = np.clip(x_bin, 0, n_bins - 1)
y_bin = np.clip(y_bin, 0, n_bins - 1)

bin_idx = x_bin * n_bins + y_bin
```

iii. The notes say the output is time-varying and encoded as `row * 3 + col`, with 25 cm by 25 cm partitions. The trajectory shows the agent chose this because the task asked for 3x3 categorical position decoding.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes the position and neural traces are already synchronized sample-by-sample within each day. It uses the same `start:end` indices for neural and output slices inside the per-trial loop.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL

    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. The notes justify this by saying the behavioral and neural streams were acquired together at 30 Hz and are therefore natively aligned.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All converted time-varying arrays use the native 30 Hz sampling interval, or 33.33 ms per bin. No temporal rebinning is performed.

ii.
```python
FPS = 30
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
...
'time_bin_size': 1000.0 / FPS,
```

iii. The AI's notes say all time series remain at the acquisition frame rate because they are already synchronized there.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are aligned frame-for-frame by slicing the same time windows from each session. Input is treated as static session context and copied once per trial rather than stored as a time series.

ii.
```python
trial_traces = traces[:, start:end]
trial_pos = pos_bins[start:end]

neural_trials.append(trial_traces.astype(np.float32))
input_trials.append(env_input.astype(np.float32))
output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
```

iii. The notes explicitly say neural and position are natively aligned at 30 Hz, while environment geometry is static within a session and therefore only needs per-trial alignment.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI handles all-NaN cells by dropping them, replaces any residual NaNs inside kept traces with zeros, discards leftover frames that do not fill a full minute, skips sessions with fewer than two trials or fewer than five registered cells, and raises an error for an unknown environment name.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
...
traces = np.nan_to_num(traces, nan=0.0)
...
if n_registered < 5:
    ...
if n_trials < 2:
    ...
...
else:
    raise ValueError(f"Unknown environment: {env}")
```

iii. The notes and trajectory justify the NaN handling in terms of cross-day cell registration, and the trajectory shows the agent treated the extra skips and the unknown-environment exception as defensive checks. I did not find a more specific rationale for converting residual NaNs to zero beyond "safety."

## 7-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each full animal file with `joblib.load`, scanning every day/session, and then looping over every 1-minute trial to slice and append separate NumPy arrays. The later `sanity_checks` pass also revisits every session and trial.

ii.
```python
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    for day in range(n_days):
        ...
        for t in range(n_trials):
            ...
            neural_trials.append(...)
            input_trials.append(...)
            output_trials.append(...)

for s in range(n_sessions):
    for t in range(n_trials):
        neural = data['neural'][s][t]
        output = data['output'][s][t]
        inp = data['input'][s][t]
```

iii. The AI did not explicitly justify these as the expensive parts, but they follow directly from the implementation structure. The notes emphasize summary statistics and validation, which implies the agent accepted the extra pass for checking.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop could have been replaced by a reshape or batched split for neural and output arrays. The loops that build `position_bin_names`, `input_names_list`, and the repeated sanity-check scans are also straightforward vectorization or preallocation candidates.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
    neural_trials.append(trial_traces.astype(np.float32))
    input_trials.append(env_input.astype(np.float32))
    output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))

for r in range(N_SPATIAL_BINS):
    for c in range(N_SPATIAL_BINS):
        position_bin_names.append(f"row{r}_col{c}")
```

iii. I did not find an explicit trajectory justification for keeping these as Python loops. The AI appears to have favored clear procedural code over vectorization.

## 7-c. What processing does the code repeat multiple times?

i. The code repeatedly casts arrays inside loops, repeatedly appends the same static environment vector once per trial, and performs a second full scan of all sessions/trials in `sanity_checks`. It also regenerates input and output label lists from nested loops every run.

ii.
```python
for t in range(n_trials):
    ...
    neural_trials.append(trial_traces.astype(np.float32))
    input_trials.append(env_input.astype(np.float32))
    output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))

for s in range(n_sessions):
    for t in range(n_trials):
        neural = data['neural'][s][t]
        output = data['output'][s][t]
        inp = data['input'][s][t]
```

iii. The AI did not explicitly justify this repetition. The notes suggest the agent prioritized validation and readability over minimizing duplicate work.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `sanity_checks` function computes and prints sample unique neural values, total neuron-session counts, and per-subject session statistics that are not saved into the final dataset. The summary counters and printed diagnostics are also discarded after the run.

ii.
```python
sample_neural = data['neural'][0][0]
unique_vals = np.unique(sample_neural)
print(f"Sample neural unique values: {unique_vals}")

total_neurons = sum(data['neural'][s][0].shape[0] for s in range(n_sessions))
print(f"Total neuron-sessions: {total_neurons}")

for subj_id, subj_name in enumerate(data['subjects']):
    ...
    print(f"  {subj_name}: {len(session_indices)} sessions, neurons/session: "
          f"min={min(n_neurons_list)}, max={max(n_neurons_list)}, mean={np.mean(n_neurons_list):.0f}")
```

iii. The notes present these checks as validation against paper statistics, so they were intentionally added for assurance rather than for the downstream decoder format itself.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is the same as in section 6: all-NaN cells are removed, residual NaNs are zero-filled, partial trailing trials are dropped via integer division, sessions can be skipped for too few trials or too few registered cells, and unknown environments raise an exception.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)

if n_registered < 5:
    ...
if n_trials < 2:
    ...
```

iii. The AI's stated rationale is again that NaNs reflect unregistered cells across days and that the remaining checks are defensive guards for malformed or degenerate sessions.

## 9-a. What are the most time-consuming steps of the code?

i. As in 7-a, the dominant costs are whole-animal file loading, iterating through all days, slicing every trial into separate arrays, and the second pass through the dataset for sanity checking.

ii.
```python
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    ...
    for day in range(n_days):
        ...
        for t in range(n_trials):
            ...

def sanity_checks(data):
    ...
    for s in range(n_sessions):
        for t in range(n_trials):
            ...
```

iii. No separate explicit justification was found beyond the general emphasis on full-dataset validation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. As in 7-b, the clearest targets are the per-trial slicing/appending loop, the repeated label-building loops, and some of the repeated whole-dataset scans in validation.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    ...

for r in range(N_SPATIAL_BINS):
    for c in range(N_SPATIAL_BINS):
        input_names_list.append(f"grid_{r}_{c}")
```

iii. The trajectory does not contain a justification for avoiding vectorization; this appears to be a simplicity choice.

## 9-c. What processing does the code repeat multiple times?

i. As in 7-c, the code repeatedly performs per-trial casts, reuses the same static input vector by appending a fresh converted copy per trial, and does a second dataset-wide validation sweep.

ii.
```python
input_trials.append(env_input.astype(np.float32))
neural_trials.append(trial_traces.astype(np.float32))
output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))

for s in range(n_sessions):
    for t in range(n_trials):
        ...
```

iii. The AI did not document this as a deliberate optimization tradeoff, but its notes make clear that extra validation and explicitness were intentional.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. As in 7-d, the main discarded work is summary printing and validation-only computations in `sanity_checks`, along with run-level counters such as total sessions, total trials, and neuron-per-session statistics that are printed but not stored in a way used by the downstream decoder.

ii.
```python
total_sessions += 1
total_trials += len(neural_trials)
total_neurons_per_session.append(n_registered)
...
print(f"Neurons per session: mean={np.mean(total_neurons_per_session):.1f}, "
      f"min={np.min(total_neurons_per_session)}, max={np.max(total_neurons_per_session)}")
```

iii. The notes show these were added as sanity checks against the paper rather than as part of the final model input format.
