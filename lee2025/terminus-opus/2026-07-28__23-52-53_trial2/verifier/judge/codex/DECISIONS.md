# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes the seven animal IDs, loads one joblib file per animal from `data/`, extracts `trace`, `position`, and `envs`, and then iterates over all days in each animal file. Trials are not loaded directly; they are created later inside `process_session`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))

    trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
    position_all = dat[animal]['position']  # (n_days, 2, n_frames)
    envs_all = dat[animal]['envs']          # (n_days, 1)
```

iii. In `CONVERSION_NOTES.md` Step 1-2 and trajectory steps 4-6, the AI states that the reference repository uses joblib-format animal files and that these files directly expose the needed arrays, so it chose them as the primary source.

## 1-b. How are the data split into subjects?

i. Each hardcoded animal ID is treated as one subject. The `subjects` list is just the `ANIMALS` list, and each exported session gets the loop index of its animal in `subject_idx`.

ii.
```python
subjects = list(animals)  # animal IDs as subject names

for animal_idx, animal in enumerate(animals):
    ...
    all_subject_idx.append(animal_idx)
```

iii. The notes say there are 7 animals and treat animal IDs as the subject identifiers.

## 1-c. How are the data split into sessions?

i. Each day in an animal file is treated as one session. The AI reads `trace_all.shape[0]` as the number of days/sessions and appends one session to the output for each `day`.

ii.
```python
n_days = trace_all.shape[0]

for day in range(n_days):
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
    ...
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “Each day for each animal = 1 session. Total 207 sessions.”

## 1-d. How are the data split into trials?

i. Within each session, the AI first rebins the data to 100 ms bins, then splits the rebinned time series into non-overlapping 1-minute trials. Since 60 seconds at 10 Hz is 600 bins, each trial is 600 pooled time bins. Any leftover pooled bins are dropped by floor division.

ii.
```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
TIME_BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600

n_timebins_total = neural_binned.shape[1]
n_trials = n_timebins_total // TIME_BINS_PER_TRIAL

for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. The notes justify this as matching the task’s 1-minute trial requirement after applying the reference decoder’s 3-frame temporal binning pipeline.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality filtering. The AI only skips entire sessions with zero registered cells and implicitly drops incomplete final trial fragments.

ii.
```python
if n_registered == 0:
    return [], [], [], 0

n_trials = n_timebins_total // TIME_BINS_PER_TRIAL

if len(neural_trials) == 0:
    print(f"  Day {day}: skipped (no registered cells)")
    continue
```

iii. In the notes, the AI says there is “No explicit trial-level curation” and that it would not apply the decoder’s movement filter to the exported dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported `neural` data is derived from the per-animal `trace` array, specifically one day/session slice `trace_all[day]`.

ii.
```python
trace_all = dat[animal]['trace']      # (n_days, n_cells, n_frames)
...
neural_trials, input_trials, output_trials, n_registered = process_session(
    trace_all[day], position_all[day], env_name
)
```

iii. The AI’s notes describe `trace` as the binarized rising-phase calcium event trace used throughout the paper code.

## 2-b. How is the `neural` data processed?

i. The AI keeps only registered cells, converts to `float32`, Gaussian-smooths each cell’s time series with sigma 3 frames, and then average-pools with kernel/stride 3 to produce 100 ms bins. It exports these pooled values as the `neural` signal.

ii.
```python
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)

smoothed_trace = gaussian_filter1d(registered_trace.astype(np.float32),
                                    sigma=GAUSS_SIGMA, axis=1)

pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
```

iii. The main justification appears in `CONVERSION_NOTES.md` Step 1 and Step 5 and in trajectory step 23: the AI decided to reuse the reference decoder’s `fit_decoder` preprocessing and also reacted to the format checker warning that raw binary traces lacked variability.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are kept if the first frame is not `NaN`, which the AI uses as a proxy for being registered that day. It skips sessions with zero such cells. Any remaining `NaN` values are replaced with zeros. The AI does not apply place-cell filtering or the decoder’s `>5` events threshold.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
...
if n_registered == 0:
    return [], [], [], 0

registered_trace = np.nan_to_num(registered_trace, nan=0.0)
```

iii. The notes say that `NaN` indicates a cell was not registered on that day and explicitly justify using all registered cells rather than place cells or activity-thresholded cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. In practice, the AI treats the start of each artificial 1-minute segment as the alignment point and records that synthetic event in metadata.

ii.
```python
for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)

'metadata': {
    'temporal_alignment_event': 'Start of 1-minute trial segment within 40-minute recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
}
```

iii. The notes say the recordings are continuous and the “trials” are imposed by the task, so the AI chose trial-start as the metadata alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. Yes, temporal rebinning is applied: 3 native 30 Hz frames are averaged into one bin after Gaussian smoothing.

ii.
```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_SIZE / FPS  # 100 ms

smoothed_trace = gaussian_filter1d(..., sigma=GAUSS_SIGMA, axis=1)
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` Step 5 says this was chosen to mirror the reference decoder’s preprocessing rather than the raw recording resolution.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives `input` from the `envs` variable, not from `blocked`. For each day, it reads the environment name string and converts it to a 3x3 geometry matrix.

ii.
```python
envs_all = dat[animal]['envs']          # (n_days, 1)
...
env_name = envs_all[day, 0] if envs_all.ndim > 1 else envs_all[day]
env_mat = get_env_mat(env_name)
env_flat = env_mat.flatten()
```

iii. In Step 5 of the notes, the AI’s variable mapping is explicitly `envs → get_env_mat()`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps each environment name to a hardcoded 3x3 binary matrix of open cells, flattens the matrix to length 9, casts it to `float32`, and reuses the same vector for every trial in that session.

ii.
```python
def get_env_mat(env):
    env_mats = {
        'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
        'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
        ...
    }
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)

input_trial = env_flat.astype(np.float32)  # (9,) static per trial
input_trials.append(input_trial)
```

iii. The notes justify this by pointing to the paper utility `get_env_mat` and by arguing that the 3x3 geometry is the relevant static decoder input.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `output` data is derived from the per-animal `position` array, specifically `position_all[day]` for each session.

ii.
```python
position_all = dat[animal]['position']  # (n_days, 2, n_frames)
...
neural_trials, input_trials, output_trials, n_registered = process_session(
    trace_all[day], position_all[day], env_name
)
```

iii. The notes identify `position` as the raw 2D `(x, y)` trajectory in centimeters.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI temporally averages position with the same 3-frame pooling used for neural data, computes per-axis 3-bin indices by dividing by a 25 cm bin width, clips them to `[0, 2]`, and converts the 2D bin into one categorical label per time bin.

ii.
```python
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()

bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)

pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. The notes say this was intended to keep position on the same 100 ms grid as neural activity while reducing the arena to a 3x3 decoding target.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI thresholds each axis into three bins spanning the 75 cm arena, using floor-division-like casting after dividing by `(75 + 1e-5) / 3`. It then combines the two 1D bins into a single category `0..8` using `x_bin * 3 + y_bin`.

ii.
```python
bin_size = (POSITION_MAX + BUFFER) / N_SPATIAL_BINS
pos_bins = (pos_binned_temporal / bin_size).astype(int)
pos_bins = np.clip(pos_bins, 0, N_SPATIAL_BINS - 1)
pos_category = pos_bins[0] * N_SPATIAL_BINS + pos_bins[1]
```

iii. The notes say the 3x3 categorization was chosen to match the environment’s 3x3 partition structure.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI uses the same pooling operation for neural and position streams and then slices both with the same trial indices. Alignment is therefore at the pooled 100 ms time base.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
neural_binned = pooling(torch.tensor(smoothed_trace)).numpy()
pos_binned_temporal = pooling(torch.tensor(position_day.astype(np.float32))).numpy()

for t in range(n_trials):
    t_start = t * TIME_BINS_PER_TRIAL
    t_end = (t + 1) * TIME_BINS_PER_TRIAL
    neural_trial = neural_binned[:, t_start:t_end].astype(np.float32)
    output_trial = pos_category[t_start:t_end].astype(np.int64).reshape(1, -1)
```

iii. The notes and trajectory repeatedly say that position should be binned “same as neural” so the two streams stay synchronized.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Unregistered cells are removed via the first-frame `NaN` mask, any leftover `NaN`s are replaced with zeros, sessions with zero registered cells are skipped, and incomplete final trial fragments are dropped by floor division.

ii.
```python
registered_mask = ~np.isnan(trace_day[:, 0])
registered_trace = trace_day[registered_mask]
registered_trace = np.nan_to_num(registered_trace, nan=0.0)

if n_registered == 0:
    return [], [], [], 0

n_trials = n_timebins_total // TIME_BINS_PER_TRIAL
```

iii. The notes justify this mainly by the interpretation that `NaN` marks a cell that was not registered that day; they do not describe any more elaborate missing-data repair.

## 6-a. What are the most time-consuming steps of the code?

i. According to the notes, the slowest steps are loading each large animal joblib file and then processing each session with smoothing and pooling. The code also times animal-level and day-level work explicitly.

ii.
```python
for animal_idx, animal in enumerate(animals):
    t_animal_start = time.time()
    t0 = time.time()
    dat = joblib.load(os.path.join(data_dir, animal))
    print(f"  Loaded in {time.time()-t0:.1f}s")
    ...
    t_day_start = time.time()
    neural_trials, input_trials, output_trials, n_registered = process_session(
        trace_all[day], position_all[day], env_name
    )
```

iii. `CONVERSION_NOTES.md` Step 7 gives runtime estimates of roughly 20-80 seconds per animal for loading and roughly 0.7-1.0 seconds per session for processing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the per-trial Python loop inside `process_session`, where regular contiguous slices are appended one by one. The outer animal/day loops reflect the dataset structure and are less avoidable.

ii.
```python
neural_trials = []
input_trials = []
output_trials = []

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

iii. The AI did not explicitly justify leaving this loop unvectorized in the notes or trajectory; it appears to be a straightforward implementation choice.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts the same static environment vector once per trial, rebuilds the `AvgPool1d` module once per session, and constructs trial lists via repeated slicing and appending even though the slices are regular.

ii.
```python
pooling = AvgPool1d(kernel_size=TEMPORAL_BIN_SIZE, stride=TEMPORAL_BIN_SIZE)
...
for t in range(n_trials):
    ...
    input_trial = env_flat.astype(np.float32)
    ...
    input_trials.append(input_trial)
```

iii. There is no explicit justification for these repeats in the notes; they look like convenience-oriented choices rather than deliberate optimizations.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main extra work is optional diagnostics: the script can generate large processing plots, prints timing summaries, and computes display-only summary information that is not stored in the final dataset.

ii.
```python
if show_processing and sessions_processed <= 2:
    save_processing_plot(trace_all[day], position_all[day], env_name,
                        neural_trials, input_trials, output_trials,
                        animal, day)

print(f"\n=== Conversion Summary ===")
print(f"Sessions: {total_sessions}")
print(f"Trials: {total_trials}")
print(f"Subjects: {len(subjects)}")
```

iii. The notes describe these as sanity checks and visual review steps rather than part of the exported analysis dataset itself.
