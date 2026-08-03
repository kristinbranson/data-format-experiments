# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the seven animal IDs, iterates over them sequentially, loads each animal file with `joblib.load()`, and only pulls `trace`, `position`, and `envs` from the raw dictionary. Trials are not loaded directly from disk; they are created later inside `process_session()`.

ii. <Code snippets>

```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
```

```python
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))

    trace_all = dat[animal]['trace']
    position_all = dat[animal]['position']
    envs_all = dat[animal]['envs']
```

iii. The notes say the reference `load_dat` function loads the non-`.mat` joblib files, and the Step 10 comparison explicitly states: “Data loading: Using `joblib.load()` same as reference `load_dat()`.”

## 1-b. How are the data split into subjects?

i. Subject identity is defined entirely by the outer `ANIMALS` list. The agent uses `subjects = list(animals)` and records one `animal_idx` per session in `subject_idx`.

ii. <Code snippets>

```python
subjects = list(animals)
```

```python
for animal_idx, animal in enumerate(animals):
    ...
    all_subject_idx.append(animal_idx)
```

iii. In Step 5, the agent wrote that “animal IDs as subject names” and treated the 7 animal files as the 7 subjects reported by the paper/data.

## 1-c. How are the data split into sessions?

i. Each recording day inside an animal file is treated as one session. The code loops over `day in range(n_days)` and appends one session-level entry to `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx` for each day that has at least one registered cell.

ii. <Code snippets>

```python
n_days = trace_all.shape[0]
...
for day in range(n_days):
    env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
```

```python
if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue

all_neural.append(neural_trials)
all_input.append(input_trials)
all_output.append(output_trials)
```

iii. Step 5 says: “Sessions: Each day for each animal = 1 session.” That matches the methods text the agent read: “one session was recorded per day.”

## 1-d. How are the data split into trials?

i. Sessions are chopped into contiguous fixed-length one-minute trials after temporal binning. The code computes the number of full 600-bin segments per session and keeps only those complete windows.

ii. <Code snippets>

```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE
```

```python
n_timebins_total = neural_binned.shape[1]
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL

for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
```

iii. The Step 5 notes justify this as a task-driven choice: “Split each 40-min session into 1-minute trials.” The trajectory also shows the agent believed the instructions required turning long sessions into one-minute trials.

## 1-e. How are trials filtered based on quality controls?

i. There is effectively no explicit trial-level quality-control filter. All complete one-minute windows are kept if the parent day/session has at least one registered neuron. The only drops are: entire sessions with zero registered cells, and incomplete trailing fragments at the end of a session.

ii. <Code snippets>

```python
if n_registered == 0:
    return [], [], [], 0
```

```python
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
...
for t in range(n_trials):
    ...
    neural_trials.append(neural_trial)
```

```python
if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue
```

iii. The notes justify the lack of trial filtering by saying there was “No explicit trial-level curation in the paper” and by arguing that the reference velocity filtering should not be applied here.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived only from the raw/session `trace` array. The code never uses `maps`, `SFPs`, `centroids`, or any precomputed place-cell outputs to build `neural`.

ii. <Code snippets>

```python
trace_all = dat[animal]['trace']
...
neural_trials, input_trials, output_trials, n_registered = process_session(
    trace_all[day], position_all[day], env_name
)
```

```python
def process_session(trace_day, position_day, env_name):
    ...
    registered_mask = ~np.isnan(trace_day[:, 0])
    registered_trace = trace_day[registered_mask]
```

iii. The notes and trajectory both say `trace` is already the binarized rising-phase calcium-event vector described in the methods, and the agent intentionally used that as the neural source variable.

## 2-b. How is the `neural` data processed?

i. The agent keeps the registered rows of `trace`, fills any remaining `NaN` with zero, Gaussian-smooths each cell’s time series with `sigma=3` frames, then temporally average-pools with kernel/stride 3 to produce 100 ms bins.

ii. <Code snippets>

```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
```

```python
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. Step 5 and Step 10 both justify this by pointing to the reference `fit_decoder`, which the trajectory shows smooths traces with `gaussian_filter1d(..., sigma=temporal_bin_size)` and then uses `AvgPool1d(kernel_size=3, stride=3)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC is registration status: cells whose first sample is `NaN` on that day are excluded. The code does not apply the reference decoder’s movement-period event threshold (`>5` events) and does not restrict to place cells.

ii. <Code snippets>

```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
```

```python
if n_registered == 0:
    return [], [], [], 0
```

iii. The Step 5 notes explicitly justify this deviation: “Use ALL registered cells per session (not just place cells)” and “No velocity filtering ... specific to the reference Bayesian decoder.” Step 10 repeats that the reference uses `cell_threshold > 5 events`, but the agent chose not to copy it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align to an experimental event from the paper. It aligns each trial to the start of an arbitrary contiguous one-minute segment within a session, then slices neural and behavioral arrays with the same bin indices.

ii. <Code snippets>

```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL

    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
```

```python
'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),
```

iii. The justification is task-based rather than reference-based. In Step 5 the agent wrote that the experiment “will be split into 1-minute trials,” so it defined trial start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. Temporal rebinning is applied once through 3-frame average pooling at 30 Hz; there is no second rebinning stage.

ii. <Code snippets>

```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS
```

```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
```

iii. The notes repeatedly justify this as matching the reference decoder: “Temporal binning: AvgPool1d with kernel_size=3 (100ms bins)” and “Binning: Gaussian smooth sigma=3, AvgPool1d kernel=3.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the `envs` field only. The code does not use the raw `blocked` field to construct the decoder input.

ii. <Code snippets>

```python
envs_all = dat[animal]['envs']
...
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
```

```python
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()
```

iii. The notes say this was chosen because the reference repository already provides `get_env_mat(env)` for converting environment names into the 3x3 geometry representation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent maps the session’s environment name to a hand-copied 3x3 binary matrix, flattens that matrix to length 9, and repeats the same static 9-vector for every trial in the session.

ii. <Code snippets>

```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
```

```python
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()
...
input_trial = env_flat.astype(np.float32)
```

iii. The Step 5 notes justify this as a direct reuse of the reference `get_env_mat` and as a good match to the task’s requested “environment geometry to represent which part of the arena is blocked.”

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output labels are derived only from the raw `position` array for each day/session.

ii. <Code snippets>

```python
position_all = dat[animal]['position']
...
neural_trials, input_trials, output_trials, n_registered = process_session(
    trace_all[day], position_all[day], env_name
)
```

```python
def process_session(trace_day, position_day, env_name):
    ...
    pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
```

iii. The Step 5 mapping table says `position` is the source variable for the decoder output, with 3x3 discretization layered on top.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code first temporally average-pools the 2D position trace with the same 3-frame pooling used for neural data. It then converts the pooled x/y values into 3 equal-width spatial bins per axis and combines the two axes into one categorical output stream.

ii. <Code snippets>

```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
```

```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. The Step 5 notes justify this as adapting the reference decoder’s temporal binning while changing the spatial output from the paper’s 15x15 position grid to the task’s required 3x3 grid.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized with fixed thresholds at approximately 25 cm and 50 cm on both axes, with a small buffer so the 75 cm edge still falls into the last bin. The final category is `x_bin * 3 + y_bin`, giving labels 0-8.

ii. <Code snippets>

```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
```

```python
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. The agent’s Step 5 rationale is explicit: “3 bins: [0-25), [25-50), [50-75] cm” and “The 3x3 grid matches the environment partition structure.”

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are aligned by sharing the same 3-frame pooling operation and the same `t_start:t_end` trial slices. The code does not apply the reference decoder’s movement-only mask before that alignment.

ii. <Code snippets>

```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()
```

```python
neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. The notes justify this by saying the task asks to decode position “at all times” and therefore no velocity filtering should be applied, even though the trajectory shows the reference `decode_position_within` masks data to movement periods before fitting/testing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases heuristically: it treats `NaN` in the first frame as the registration flag, skips sessions with zero registered cells, replaces any remaining neural `NaN` with zero, clips position bins into the valid 0-2 range, adds a small buffer to avoid dropping the 75 cm edge, returns all-`NaN` geometry for unknown environments, and silently drops any incomplete session tail that cannot fill a one-minute trial.

ii. <Code snippets>

```python
registered_mask = ~np.isnan(trace_day[:, 0])
...
if n_registered == 0:
    return [], [], [], 0
registered_trace = np.nan_to_num(registered_trace, nan=0.0)
```

```python
return env_mats.get(env, np.full((3,3), np.nan)).astype(float)
```

```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
```

```python
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
```

iii. The Step 10 notes explicitly mention three of these choices: discarding incomplete remainder frames, using a buffer for the 75 cm edge, and the assumption that there are no empty sessions after registration filtering. The code comments add “shouldn’t happen but be safe” for remaining neural `NaN`s.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant costs are loading each large animal joblib file from disk, then session-wise smoothing/pooling and final pickle serialization. The notes report loading as much slower than per-session processing.

ii. <Code snippets>

```python
for animal_idx, animal in enumerate(animals):
    t0 = time.time()
    dat = joblib.load(os.path.join(data_dir, animal))
    print(f"  Loaded in {time.time()-t0:.1f}s")
```

```python
smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

```python
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. In Step 7, the agent estimated loading at roughly 20-80 s per animal and processing at roughly 0.7-1.0 s per session; the full-conversion log and notes say loading dominated the runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the per-trial append loop inside `process_session()`. Since trial windows are contiguous and fixed-size, the code could trim once and reshape rather than slice/appending 39-40 times per session. The per-trial duplication of the static environment vector is also repetitive.

ii. <Code snippets>

```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL

    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    input_trial = env_flat.astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)

    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. There is no explicit justification for keeping this loop. The agent’s notes instead claim the script is “efficient,” but the instructions had asked it to vectorize loops where possible.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly recreates the pooling layer per session, repeatedly casts the same flattened environment vector once per trial, and recomputes registration/environment summaries again inside `save_processing_plot()`. These are small but real repeated computations.

ii. <Code snippets>

```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

```python
for t in range(n_trials):
    ...
    input_trial = env_flat.astype(np.float32)
```

```python
def save_processing_plot(...):
    registered_mask = ~np.isnan(trace_day[:, 0])
    ...
    env_mat = get_env_mat(env_name)
```

iii. The notes do not meaningfully justify these repetitions. The agent mostly justified correctness, not efficiency, and only gave rough runtime estimates after the fact.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional plotting path does a fair amount of extra work that is not used by the downstream decoder: it recomputes summaries, concatenates trial data for histograms, and writes eight-panel diagnostics. Outside `--show-processing`, there is relatively little obviously unnecessary computation in the main conversion path.

ii. <Code snippets>

```python
if show_processing and sessions_processed <= 2:
    save_processing_plot(trace_all[day], position_all[day], env_name,
                        neural_trials, input_trials, output_trials,
                        animal, day)
```

```python
all_pos = np.concatenate([ot[0] for ot in output_trials])
counts = np.bincount(all_pos.astype(int), minlength=9)
...
all_neural = np.concatenate([nt.flatten() for nt in neural_trials])
ax.hist(all_neural[all_neural > 0], bins=50, alpha=0.7)
```

iii. This extra work is justified by the instructions rather than the scientific pipeline: Step 6 explicitly required a `--show-processing` mode with visualizations of each processing step.
