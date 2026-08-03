# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 7 animal IDs, loads one joblib file per animal from `/app/data/<animal>`, unwraps the nested dictionary with `dat[animal]`, and then iterates over every day in `d['trace']` as a session. Trials are not stored in the raw data; they are created later by slicing each session into fixed frame windows.

ii. 
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

def load_animal_data(animal):
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]

for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, ...)
```

iii. In `CONVERSION_NOTES.md`, the agent says the reference code uses `load_dat`/`joblib.load`, and the trajectory shows it inspected `main.py` where the same 7 animals are enumerated and `load_dat(animal, p, format="joblib")` is called.

## 1-b. How are the data split into subjects?

i. Subjects are split by file: each animal-specific joblib file corresponds to one mouse. Session-to-subject mapping is tracked by appending `ANIMALS.index(animal)` for each retained session.

ii. 
```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, ...)
    subject_id = ANIMALS.index(animal)
    for s_idx in range(len(neural)):
        ...
        subject_idx_list.append(subject_id)

data = {
    'subjects': ANIMALS,
    'subject_idx': np.array(subject_idx_list, dtype=np.int64),
}
```

iii. The notes state there are 7 mice stored as separate joblib files, and the trajectory excerpt from `main.py` shows the same animal list used by the reference code.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day as one session. It uses the first dimension of `d['trace']`, `d['position']`, and `d['envs']`, iterating `for day in range(n_days)`.

ii. 
```python
n_days = d['trace'].shape[0]
envs = d['envs'].squeeze()

for day in range(n_days):
    env_name = envs[day]
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. `CONVERSION_NOTES.md` documents `trace` as `(n_days, n_cells, n_frames)` and `position` as `(n_days, 2, n_frames)`. The trajectory shows the agent read reference functions that index sessions by day in the same way.

## 1-d. How are the data split into trials?

i. The agent creates synthetic trials by chopping each session into contiguous 1-minute windows at 30 Hz, starting at frames `0, 1800, 3600, ...`. It keeps full 1800-frame windows and also keeps the final shorter window if it is at least 900 frames long.

ii. 
```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
MIN_TRIAL_FRAMES = FPS * 30

trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The justification in the notes is that the decoder instructions said to split the long sessions into 1-minute trials. The notes list “Trial definition: 1-minute segments (1800 frames at 30Hz), ~40 trials per session.”

## 1-e. How are trials filtered based on quality controls?

i. There is almost no trial-quality filtering. The code only drops trailing partial trials shorter than 30 seconds and then drops any session left with fewer than 2 trials. It does not inspect movement, missing behavior, or any per-trial signal-quality metric.

ii. 
```python
if trial_len < MIN_TRIAL_FRAMES:
    continue

...
n_trials = len(neural_trials)
if n_trials < 2:
    print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
    continue
```

iii. The notes explicitly say “Trial curation rules: No trial filtering,” but the actual code does apply the two length/count checks above. The trajectory and notes do not mention any other trial QC decision.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from the raw `trace` array for each day, after masking out cells that are not registered on that day.

ii. 
```python
trace = d['trace'][day]
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The notes say `trace` contains binary calcium events with `NaN` for unregistered cells. The trajectory shows the agent inspected reference functions like `get_rate_maps(position, trace, ...)` and `decode_position_within(..., traces, ...)`, both of which use `trace` directly.

## 2-b. How is the `neural` data processed?

i. The agent does almost no additional neural preprocessing. It assumes the joblib `trace` is already the processed signal of interest, filters to registered cells, slices it into trials, and casts it to `float32`. It does not compute dF/F, smooth the traces, deconvolve further, or convert to rate maps.

ii. 
```python
# Identify registered cells (non-NaN on this day)
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]

# Neural: (n_registered, trial_len)
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says “Data is already preprocessed: binary calcium trace (0/1 for significant events from rising-phase extraction)” and “No delta F/F computation needed.” The trajectory shows the agent read the reference code around `get_rate_maps`, which treats `trace` as the already-prepared calcium-event signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC the code applies is removing cells whose first sample on a given day is `NaN`, treating those cells as unregistered for that session. It does not apply place-cell filtering, split-half reliability thresholds, movement-based frame filtering, or the decoder code’s `cell_threshold > 5` active-cell filter.

ii. 
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
if n_registered == 0:
    ...
trace_registered = trace[registered_mask]
```

iii. The notes justify this with “Use ALL registered cells on each day (non-NaN trace). No place-cell filtering.” The same notes also mention that the reference decoder uses a velocity filter and `cell_threshold > 5`, but the agent chose not to reproduce those filters in the converted dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data to the artificial trial boundary it created itself: frame 0 of each 1-minute chunk is treated as the alignment event. No task event from the paper is used.

ii. 
```python
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)

'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. The notes say the decoder task required 1-minute trials and record the alignment event as “Start of each 1-minute trial segment within a 40-minute recording session.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native imaging frame rate, 30 Hz, which the metadata reports as `1000 / 30 = 33.33 ms` per sample. No temporal rebinning is applied.

ii. 
```python
FPS = 30

'time_bin_size': 1000.0 / FPS,  # ~33.33 ms

neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes say “Time bin: 30Hz native sampling (33.33ms)” and the trajectory/reference excerpts mention the original data and reference decoder both operate at 30 Hz.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the session’s environment label string in `d['envs']`. The code does not use `blocked`; it maps the environment name directly to a hard-coded binary geometry template.

ii. 
```python
envs = d['envs'].squeeze()
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()
```

iii. The notes say the mapping is “Environment geometry 3x3 | `get_env_mat(env)` -> flatten to 9 values,” and the trajectory shows the agent copied `get_env_mat` from the reference `utils.py`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is converted to a binary 3x3 occupancy/blockage matrix with `get_env_mat`, then flattened to a 9-element vector and reused as a static per-trial input.

ii. 
```python
def get_env_mat(env):
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    ...

env_mat = get_env_mat(env_name).flatten()
input_trial = env_mat.astype(np.float32)
```

iii. The notes call this out as copied from the reference code. The trajectory confirms the agent inspected `/app/code/georepca1/src/utils.py` where `get_env_mat` is defined.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output comes directly from the raw `position` array for each day.

ii. 
```python
position = d['position'][day]  # (2, n_frames)
pos_bins = bin_position_3x3(position)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes document `position` as `(n_days, 2, n_frames)` in centimeters and identify it as the source for the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent bins continuous x and y coordinates into a 3x3 grid over a fixed 75 cm square arena, clips indices to `[0, 2]`, converts the two bin coordinates into a single class index, and then slices the resulting time series per trial. There is no smoothing, velocity filtering, or masking to accessible locations.

ii. 
```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx
```

iii. The notes justify this as the decoder-task-required discretization: “Position discretization: 3x3 grid, row-major ordering.” The trajectory shows the agent compared this to reference functions that bin position spatially, but the 3x3 scheme is its own adaptation.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The agent creates 9 categories by thresholding x and y into 3 equal-width spatial bins each and then combining them with row-major indexing: `0..8 = x_bin * 3 + y_bin`.

ii. 
```python
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin

output_values = [
    [f"x{r}y{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]
]
```

iii. The notes explicitly state “3x3 grid, row-major ordering (bin_idx = x_bin*3 + y_bin),” matching the output labels written into the dataset.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output and neural data are aligned by using the same day index and identical `start:end` frame slices. The code first bins the full session’s position time series and then slices each trial with the same boundaries used for neural traces.

ii. 
```python
trace = d['trace'][day]
position = d['position'][day]
pos_bins = bin_position_3x3(position)

for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes say the conversion should show “no temporal misalignments,” and the trajectory includes a sanity check where the agent recomputed the original `pos_bins[start:end]` and verified it matched the converted output exactly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is minimal. Missing neural data are treated as unregistered cells via `NaN` masking on `trace[:, 0]`. Entire days are skipped if no cells are registered. Very short trailing partial trials are skipped. There is no explicit repair or imputation of missing position samples, no gap-filling, and no special handling of `blocked` or other metadata inconsistencies.

ii. 
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)
if n_registered == 0:
    continue

if trial_len < MIN_TRIAL_FRAMES:
    continue
```

iii. The notes say the raw `trace` contains `NaN` for unregistered cells, and the rest of the writeup assumes the data are otherwise already clean. No stronger missing-data strategy appears in the trajectory or script.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading each large animal file with `joblib.load`, slicing and appending thousands of trial arrays to nested Python lists, and then serializing the final multi-gigabyte dataset with `pickle.dump`. The per-session timings in `conversion_full_out.txt` show the compute inside each day loop is relatively cheap compared with total per-animal runtime.

ii. 
```python
d = load_animal_data(animal)
...
for start in trial_starts:
    ...
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes mention timing output and runtime estimation, and `conversion_full_out.txt` shows 11-29 seconds per animal plus a 19+ GB final pickle output.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The repeated Python loop over trials could have been reduced by reshaping or pre-splitting full-session arrays once, especially for the full 1800-frame trials. The `build_dataset` loop that appends one session at a time and the output-distribution concatenation at the end also do avoidable Python-level work. The plotting loops are also non-vectorized, though they are optional.

ii. 
```python
for start in trial_starts:
    ...
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)

for s_idx in range(len(neural)):
    all_neural.append(neural[s_idx])
    all_input.append(inp[s_idx])
    all_output.append(out[s_idx])

all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
```

iii. The notes claim the code uses “efficient vectorized operations,” but the implementation still relies on several large Python loops for trial/session packaging and summary-stat computation.

## 6-c. What processing does the code repeat multiple times?

i. It repeatedly casts the same static environment vector to `float32` once per trial, repeatedly stores identical environment geometry for every trial in a session, and repeatedly slices arrays trial by trial even though most sessions share the same fixed trial structure. It also recomputes whole-dataset output summaries after the dataset has already been built.

ii. 
```python
env_mat = get_env_mat(env_name).flatten()

for start in trial_starts:
    ...
    input_trial = env_mat.astype(np.float32)
    ...

all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
```

iii. The notes focus on correctness and runtime, but the code itself shows this repetition clearly. The repeated per-trial copying of the same session-level input is the clearest example.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional plotting generates large visualization files but is not used by the decoder. The end-of-run output-distribution summary concatenates all outputs only for printing. `session_info` is stored in metadata even though the downstream decoder primarily uses `neural`, `input`, `output`, and indexing fields. These steps are useful for inspection but not for model training itself.

ii. 
```python
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)

'session_info': all_session_info,

all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
unique, counts = np.unique(all_out, return_counts=True)
```

iii. The notes emphasize sanity checks, plotting, and reporting. Those are reasonable for validation, but they are not needed by downstream decoding once the converted pickle has been produced.
