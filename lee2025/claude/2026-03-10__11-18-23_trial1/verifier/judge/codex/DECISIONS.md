# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads preprocessed per-animal `joblib` files from `/app/data/<animal>` using a hard-coded `ANIMALS` list. It does not read the `.mat` files. Each animal file is loaded into memory, then iterated day-by-day and split into trials in Python.

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
    neural, inp, out, sess_info = process_animal(
        animal, show_processing=show_processing
    )
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset is stored as joblib files and explicitly treats those as the working source: “Each animal stored as joblib file.” It also claims in the notes that `load_dat` in the reference uses `joblib.load`, so it considered this consistent with the reference pipeline.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the fixed `ANIMALS` list. The AI does not discover subjects from the filesystem at runtime except indirectly by assuming those named files exist.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

for animal in animals:
    ...
    subject_id = ANIMALS.index(animal)
```

iii. The notes say there are 7 mice with those IDs and treat them as the full subject list. The trajectory shows the AI enumerated directory contents early, saw exactly those seven animals, and then encoded them as constants.

## 1-c. How are the data split into sessions?

i. Within each animal, sessions are treated as recording days. The AI reads `d['trace']`, takes `n_days = d['trace'].shape[0]`, and iterates `for day in range(n_days)`, with each day becoming one session if it survives later filters.

ii.
```python
n_days = d['trace'].shape[0]
...
for day in range(n_days):
    env_name = envs[day]
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. In the notes, the AI equates “day” and “session” and reports 207 sessions total from the day axis of the joblib data.

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 1-minute windows of 1800 frames at 30 Hz. Unlike the reference, the AI keeps the final partial chunk if it is at least 30 seconds long.

ii.
```python
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
MIN_TRIAL_FRAMES = FPS * 30
...
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    if trial_len < MIN_TRIAL_FRAMES:
        continue
```

iii. The notes state “1-minute segments (1800 frames at 30Hz)” but also record an edge-case rule: “Partial trials <900 frames (30s) are dropped,” implying the AI deliberately preserved longer partial end segments.

## 1-e. How are trials filtered based on quality controls?

i. Trials shorter than 30 seconds are dropped. After trialization, sessions with fewer than 2 trials are also skipped entirely.

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

iii. The notes justify this as decoder-oriented curation: “All sessions produce >=2 trials (required for decoder evaluation).”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the `trace` array in each joblib animal dictionary, specifically `d['trace'][day]`.

ii.
```python
trace = d['trace'][day]
```

iii. The notes describe this as “binary calcium events (0/1)” and say no delta-F/F computation is needed because the data are already preprocessed.

## 2-b. How is the `neural` data processed?

i. The AI keeps registered cells only, leaves the trace values otherwise unchanged, and casts each trial to `float32`. There is no temporal smoothing or rebinning.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The notes repeatedly justify this by saying the traces are already binarized calcium events, so the intended neural representation is the raw event matrix for registered cells.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered by whether the first frame is non-NaN on that day. If a day has zero registered cells it is skipped.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)

if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue
```

iii. The notes say “Use ALL registered cells on each day (non-NaN trace).” The code operationalizes that as checking only the first frame for NaN status.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the start of each artificial 1-minute segment as the alignment event and records that in metadata.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
    ...
}
```

iii. The notes frame the data as continuous recordings split into trials, but the final metadata still invents an explicit alignment event at trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native 30 Hz sampling rate, so each bin is about 33.33 ms. No temporal rebinning is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS
```

iii. The notes explicitly say “Time bin: 30Hz native sampling (33.33ms).”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI does not use the raw `blocked` variable. Instead, it uses the categorical environment name `d['envs'][day]` and maps that name to a hand-coded 3x3 geometry matrix.

ii.
```python
envs = d['envs'].squeeze()
...
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()
```

iii. The notes say “Environment geometry 3x3” comes from ``get_env_mat(env)`` and emphasize geometry identity rather than blocked-position indices.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts each environment label such as `square`, `t`, `u`, or `glenn` into a binary 3x3 occupancy matrix, flattens it to length 9, and repeats that static vector for every trial in the session.

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

iii. The notes call this a direct reuse of the reference `get_env_mat` helper and justify it as representing “1=accessible, 0=blocked.”

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry vector is static per trial and copied into `input_trials` once for each neural trial; it is not time-varying and therefore has no frame-wise alignment step.

ii.
```python
input_trial = env_mat.astype(np.float32)
...
input_trials.append(input_trial)
```

iii. The notes say the environment geometry is “static per trial.”

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` array for each day.

ii.
```python
position = d['position'][day]  # (2, n_frames)
```

iii. The notes describe `position` as x-y coordinates in centimeters over time.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI discretizes the 2D position into a 3x3 grid using equal-width bins across a 75 cm arena, then stores the category index as a length-`T` vector reshaped to `(1, T)`.

ii.
```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx

output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes justify this as a decoder-specific reduction to 9 balanced spatial classes and describe it as “row-major ordering.”

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is thresholded by uniform 25 cm-wide bins over `[0, 75]`, clipped to the valid range `[0, 2]`, then combined into a single class index with `x_bin * 3 + y_bin`.

ii.
```python
bin_size = (env_size + buffer) / N_POS_BINS
x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
bin_idx = x_bin * N_POS_BINS + y_bin
```

iii. The notes say the discretization uses a 3x3 grid with “bin_idx = x_bin*3 + y_bin.”

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes `trace` and `position` are already frame-aligned within each day. It then slices both by the same `start:end` trial boundaries.

ii.
```python
trace = d['trace'][day]
position = d['position'][day]
...
neural_trial = trace_registered[:, start:end].astype(np.float32)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes state that position was converted trial-by-trial from the same session arrays, and the spot-check section says converted neural and position bins matched their original day/trial slices exactly.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native 30 Hz frames, corresponding to 33.33 ms bins. No rebinning is applied.

ii.
```python
FPS = 30
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S
...
'time_bin_size': 1000.0 / FPS
```

iii. The notes explicitly call this the “30Hz native sampling.”

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output data are aligned by slicing the same `start:end` frame ranges from a common session. The input is static geometry repeated once per trial, so it aligns at the trial level rather than per frame.

ii.
```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    ...
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    input_trial = env_mat.astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes summarize this as “Input: Environment geometry, static per trial” and “Position binned to 3x3 ... Time-varying.”

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI handles missing or unusable data by dropping days with zero registered cells, dropping trial fragments shorter than 30 seconds, and dropping sessions that end up with fewer than two trials. There is no broader malformed-entry handling beyond NaN-based registration filtering.

ii.
```python
registered_mask = ~np.isnan(trace[:, 0])
...
if n_registered == 0:
    ...
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
if n_trials < 2:
    ...
    continue
```

iii. The notes present these as edge-case checks and explicitly list short partial-trial removal and the `>=2` trial requirement.

## 7-a. What are the most time-consuming steps of the code?

i. The AI’s notes indicate the main costs are loading large joblib animal files and iterating through every day/session to build trialized arrays; optional plotting also adds cost.

ii.
```python
dat = joblib.load(filepath)
...
for day in range(n_days):
    ...
    for start in trial_starts:
        ...
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)
```

iii. `CONVERSION_NOTES.md` reports runtime per animal and for full conversion, and describes the implementation as using timing output for each session.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has explicit Python loops over animals, days, and trial windows. The per-trial slicing loop could have been vectorized or reshaped for full-length trials, and the repeated per-trial appends could have been reduced.

ii.
```python
for animal in animals:
    ...
    for day in range(n_days):
        ...
        for start in trial_starts:
            ...
            neural_trials.append(neural_trial)
            input_trials.append(input_trial)
            output_trials.append(output_trial)
```

iii. The notes claim “Efficient vectorized operations (no inner loops for trace/position),” but the actual code still uses nested loops for trial construction.

## 7-c. What processing does the code repeat multiple times?

i. The AI repeatedly casts the same static environment matrix to `float32` once per trial, repeatedly constructs per-trial arrays inside Python loops, and repeatedly looks up `ANIMALS.index(animal)` inside the subject loop.

ii.
```python
input_trial = env_mat.astype(np.float32)
...
input_trials.append(input_trial)
...
subject_id = ANIMALS.index(animal)
```

iii. The notes do not call this repetition out; they instead describe the implementation as efficient.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `session_info` only as metadata, calculates extra logging statistics, and includes an optional plotting pipeline that is unrelated to the final pickle contents.

ii.
```python
session_info.append({
    'animal': animal,
    'day': day,
    'env': env_name,
    'n_registered': int(n_registered),
    'n_trials': n_trials,
})
...
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)
...
unique, counts = np.unique(all_out, return_counts=True)
```

iii. The notes justify plotting and summaries as sanity checks and validation, not as essential conversion outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same handling as in 6: the AI removes sessions with zero registered cells, drops short end fragments under 30 seconds, and skips sessions with fewer than two resulting trials.

ii.
```python
if n_registered == 0:
    ...
if trial_len < MIN_TRIAL_FRAMES:
    continue
...
if n_trials < 2:
    ...
    continue
```

iii. The notes explicitly list these as edge-case checks.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading the joblib data and looping over all days and trial windows dominate the runtime, with optional plotting as extra overhead.

ii.
```python
dat = joblib.load(filepath)
...
for day in range(n_days):
    ...
    for start in trial_starts:
        ...
```

iii. The notes include measured runtime estimates for sample and full conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the loops over days and trial windows are the main vectorization opportunities.

ii.
```python
for day in range(n_days):
    ...
    for start in trial_starts:
        ...
```

iii. The notes describe the code as largely vectorized, but the implementation still relies on those loops.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: repeated `astype(np.float32)` on static inputs, repeated list appends per trial, and repeated subject-index lookup.

ii.
```python
input_trial = env_mat.astype(np.float32)
...
subject_id = ANIMALS.index(animal)
```

iii. This repetition is visible in the code rather than explicitly discussed in the notes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: optional plots, verbose summary computations, and session metadata are produced in addition to the actual converted arrays.

ii.
```python
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)
...
'session_info': all_session_info,
...
unique, counts = np.unique(all_out, return_counts=True)
```

iii. The notes frame these as verification and documentation rather than decoder inputs/outputs.
