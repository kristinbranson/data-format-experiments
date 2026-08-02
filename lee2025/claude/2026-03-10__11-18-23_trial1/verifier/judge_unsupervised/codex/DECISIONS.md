# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 7 mouse IDs, loads one preprocessed `joblib` file per mouse from `/app/data/{animal}`, unwraps the nested `{animal: data}` dict, and then iterates over all days and all generated trials to build the dataset.

ii. <Code snippets>

```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_animal_data(animal):
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]

for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)
```

iii. The agent justified this by matching the reference `load_dat(..., format="joblib")` path and by noting in `CONVERSION_NOTES.md` that the no-extension files in `data/` are the preprocessed Python/joblib versions of the dataset.

## 1-b. How are the data split into subjects?

i. Subjects are split by iterating through the fixed `ANIMALS` list. Each processed session is assigned a subject index using `ANIMALS.index(animal)`, and the dataset stores both the full subject name list and per-session `subject_idx`.

ii. <Code snippets>

```python
for animal in animals:
    neural, inp, out, sess_info = process_animal(animal, show_processing=show_processing)

    subject_id = ANIMALS.index(animal)

    for s_idx in range(len(neural)):
        all_neural.append(neural[s_idx])
        ...
        subject_idx_list.append(subject_id)

data = {
    ...
    'subjects': ANIMALS,
    'subject_idx': np.array(subject_idx_list, dtype=np.int64),
}
```

iii. The notes say the dataset contains 7 mice with known IDs, and the trajectory shows the agent chose to preserve that ordering exactly so session-to-subject mapping stayed explicit and simple.

## 1-c. How are the data split into sessions?

i. The agent treats each recording day as one session. It reads `n_days = d['trace'].shape[0]` and loops `for day in range(n_days)`, pulling one session’s `trace`, `position`, and `envs` entry per day.

ii. <Code snippets>

```python
n_days = d['trace'].shape[0]
envs = d['envs'].squeeze()

for day in range(n_days):
    env_name = envs[day]
    trace = d['trace'][day]
    position = d['position'][day]
```

iii. The agent’s notes explicitly say one session was recorded per day and that the reference paper reports 207 sessions total, so it mapped “day” to “session”.

## 1-d. How are the data split into trials?

i. Each day/session is split into consecutive 1-minute trial chunks at the native 30 Hz frame rate, so each full trial is 1800 frames. Trial boundaries are `[0, 1800, 3600, ...]` within each session.

ii. <Code snippets>

```python
FPS = 30
TRIAL_DURATION_S = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S

trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))

for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    trial_len = end - start
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The trajectory and notes say this came from the task instructions, which explicitly required splitting long sessions into 1-minute trials for decoder evaluation.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not use any paper-specific trial quality control. It only skips short trailing partial trials under 30 s and drops sessions that would end up with fewer than 2 trials.

ii. <Code snippets>

```python
MIN_TRIAL_FRAMES = FPS * 30

if trial_len < MIN_TRIAL_FRAMES:
    continue

n_trials = len(neural_trials)
if n_trials < 2:
    print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
    continue
```

iii. In `CONVERSION_NOTES.md` the agent says “No trial filtering” relative to the paper, but it added these checks to satisfy the decoder format requirements that need usable trials and at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the per-day calcium event matrix `d['trace'][day]`, after removing cells that are unregistered on that day.

ii. <Code snippets>

```python
trace = d['trace'][day]
registered_mask = ~np.isnan(trace[:, 0])
trace_registered = trace[registered_mask]
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The agent’s notes cite the paper’s preprocessing description: the stored `trace` is already the binarized rising-phase event vector used as the firing-rate-like signal in later analyses.

## 2-b. How is the `neural` data processed?

i. The agent leaves the neural signal essentially as-is: binary event traces are sliced by trial and cast to `float32`. It does not compute rate maps, smooth in time, rebin in time, or infer spikes.

ii. <Code snippets>

```python
# Extract registered cells' traces
trace_registered = trace[registered_mask]

# Neural: (n_registered, trial_len)
neural_trial = trace_registered[:, start:end].astype(np.float32)
```

iii. The notes state “No delta F/F computation needed - trace is already binarized” and “The final binarized rising-phase vector was then set to 1... This binary vector was treated as the firing rate in all further analyses.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent only filters out unregistered cells by testing whether the first frame is NaN. It does not apply place-cell filtering, movement filtering, or the reference decoder’s active-cell threshold.

ii. <Code snippets>

```python
registered_mask = ~np.isnan(trace[:, 0])
n_registered = np.sum(registered_mask)

if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue
```

iii. The notes say “Use ALL registered cells on each day (non-NaN trace). No place-cell filtering” and also explicitly record that the reference decoder had a velocity filter and `cell_threshold > 5`, but the agent chose not to apply them.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 1-minute segment. Within a trial, neural samples keep their original frame order relative to that segment start.

ii. <Code snippets>

```python
trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))

for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)

'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. The agent’s metadata names the alignment event directly. Its rationale was that the instructions define trials as 1-minute segments, not stimulus-locked epochs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz sampling rate, i.e. about 33.33 ms per time bin. No temporal rebinning is applied.

ii. <Code snippets>

```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The notes repeatedly state that recordings were acquired at 30 Hz and that the agent kept the native sampling because the stored traces and positions are already frame-aligned.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the session’s environment label, `d['envs'][day]`, not from the `blocked` array.

ii. <Code snippets>

```python
envs = d['envs'].squeeze()
env_name = envs[day]
env_mat = get_env_mat(env_name).flatten()
```

iii. The notes say the agent followed the reference `get_env_mat` helper and treated the string environment name as the authoritative input to that mapping.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent copies the reference `get_env_mat` lookup table, converts each named environment into a binary 3x3 occupancy matrix, then flattens it into a 9-element vector and stores it as `float32`.

ii. <Code snippets>

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

iii. The trajectory shows the agent read the reference `get_env_mat` code and reused it verbatim because the task input was “which part of the arena is blocked”.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. Input is static within a trial. For every neural trial slice from a session, the same 9-element environment vector is attached to that trial.

ii. <Code snippets>

```python
env_mat = get_env_mat(env_name).flatten()

for start in trial_starts:
    ...
    input_trial = env_mat.astype(np.float32)
    input_trials.append(input_trial)
```

iii. The notes call the environment input “static per trial”, and the trajectory shows the agent chose this because geometry does not change within a recording session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the per-day tracked position array `d['position'][day]`, which stores x and y coordinates over time.

ii. <Code snippets>

```python
position = d['position'][day]  # (2, n_frames)
pos_bins = bin_position_3x3(position)
output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes cite the paper’s methods stating that position was generated from DeepLabCut head tracking and was recorded in sync with calcium imaging.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent bins continuous x,y position into a 3x3 grid using a fixed 75 cm arena size, flooring each coordinate into one of 3 bins and clipping to `[0, 2]`.

ii. <Code snippets>

```python
def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx
```

iii. The trajectory shows the agent inspected the position range, confirmed it spans roughly `0-75 cm`, and justified the skewed occupancy histogram as natural behavior rather than a binning bug.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Categories are the 9 cells of the 3x3 grid. The category index is encoded row-major as `x_bin * 3 + y_bin`, yielding class IDs 0-8 labeled `x0y0` through `x2y2`.

ii. <Code snippets>

```python
bin_idx = x_bin * N_POS_BINS + y_bin
...
output_values = [
    [f"x{r}y{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]
]
```

iii. The notes explicitly list “Position discretization: 3x3 grid, row-major ordering” as a key design choice made to satisfy the decoder output specification.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Output uses the same frame indices and trial boundaries as the neural slices. For a given trial, `output_trial[t]` corresponds to the same original recording frame as `neural_trial[:, t]`.

ii. <Code snippets>

```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The agent’s rationale was that the methods say behavior and imaging were recorded simultaneously at 30 Hz and frame-timestamp aligned.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms per sample (30 Hz), and there is no temporal rebinning.

ii. <Code snippets>

```python
FPS = 30
'time_bin_size': 1000.0 / FPS,
```

iii. The notes cite the 30 Hz acquisition rate from the paper and state that the agent kept native sampling.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are aligned by shared session frame index and identical trial slices. Input is session-static and attached once per trial, so it is aligned at the trial level rather than frame-by-frame.

ii. <Code snippets>

```python
trace = d['trace'][day]
position = d['position'][day]
env_mat = get_env_mat(env_name).flatten()
pos_bins = bin_position_3x3(position)

for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    neural_trial = trace_registered[:, start:end].astype(np.float32)
    input_trial = env_mat.astype(np.float32)
    output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)
```

iii. The notes say the streams are all derived from the same synchronized 30 Hz session data; the environment geometry is constant for the whole session, so the agent stored it as a per-trial constant input.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. Missing/unregistered cells are handled by removing rows whose first trace sample is `NaN`. Sessions with no registered cells are skipped. Unknown environment labels raise an error. There is no broader repair logic for malformed positions or partially missing behavioral samples.

ii. <Code snippets>

```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0:
    print(f"  Day {day} ({env_name}): No registered cells, skipping")
    continue

else:
    raise ValueError(f"Unknown environment: {env}")
```

iii. The notes emphasize that NaNs indicate cells not registered on a given day, so the agent treated that as the main data-integrity case that needed explicit handling.

## 7-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading each large animal `joblib` file, iterating through every session and trial to slice data into Python lists, and writing the final ~20 GB pickle. Optional plotting also adds overhead when enabled.

ii. <Code snippets>

```python
dat = joblib.load(filepath)
...
for day in range(n_days):
    ...
    for start in trial_starts:
        ...
        neural_trials.append(neural_trial)
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The conversion logs and notes report per-animal runtimes and a large output file, which is consistent with I/O and repeated slicing dominating runtime rather than arithmetic.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop could be more vectorized by reshaping full-session arrays into `(n_trials, ..., 1800)` blocks when trials are full length, and the repeated per-trial appending of identical static inputs could be generated more directly. The subject/session accumulation loop also remains pure Python.

ii. <Code snippets>

```python
for start in trial_starts:
    end = min(start + FRAMES_PER_TRIAL, n_frames)
    ...
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)

for s_idx in range(len(neural)):
    all_neural.append(neural[s_idx])
    all_input.append(inp[s_idx])
    all_output.append(out[s_idx])
```

iii. The agent described the implementation as “efficient vectorized operations” in the notes, but these loops are still straightforward candidates for reshaping/broadcasting rather than repeated Python appends.

## 7-c. What processing does the code repeat multiple times?

i. It repeatedly casts the same environment vector to `float32` once per trial, repeats the same session-static input for every trial of a session, and repeatedly appends per-session structures into global lists. It also recomputes summary statistics by concatenating all outputs after building the dataset.

ii. <Code snippets>

```python
input_trial = env_mat.astype(np.float32)
input_trials.append(input_trial)

all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
```

iii. This follows directly from the code structure: even though `env_mat` is constant within a session, it is regenerated as a separate object for every trial.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional visualization in `plot_processing`, runtime/summary printing, and some metadata bookkeeping (`session_info`) are not used by `train_decoder.py` for model fitting. The post-save output-distribution summary is also purely diagnostic.

ii. <Code snippets>

```python
if show_processing and len(all_neural) > 0:
    plot_processing(animal, d, all_neural, all_input, all_output, session_info)

'session_info': all_session_info,

print(f"\nOutput distribution (position_bin):")
for u, c in zip(unique, counts):
    print(f"  Bin {int(u)} ({data['output_values'][0][int(u)]}): {c/len(all_out)*100:.1f}%")
```

iii. The notes describe these as validation and documentation aids rather than part of the converted representation itself.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Same as 6: NaN-marked unregistered cells are removed, zero-cell sessions are skipped, and unknown environment labels raise an exception. The code does not attempt interpolation or repair of behavioral traces.

ii. <Code snippets>

```python
registered_mask = ~np.isnan(trace[:, 0])
if n_registered == 0:
    ...
    continue

else:
    raise ValueError(f"Unknown environment: {env}")
```

iii. The agent consistently treated registration NaNs as the main expected data issue and otherwise assumed the preprocessed files were clean.

## 9-a. What are the most time-consuming steps of the code?

i. Same as 7-a: loading large `joblib` files, looping over all day/trial slices, and serializing the final pickle dominate runtime; plotting is extra overhead when turned on.

ii. <Code snippets>

```python
dat = joblib.load(filepath)
...
for day in range(n_days):
    ...
    for start in trial_starts:
        ...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The runtime logs in `CONVERSION_NOTES.md` and `conversion_full_out.txt` support that interpretation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Same as 7-b: the trial-splitting/appending loops and the cross-session accumulation loops could be reduced with reshaping or preallocation.

ii. <Code snippets>

```python
for start in trial_starts:
    ...
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)

for s_idx in range(len(neural)):
    all_neural.append(neural[s_idx])
    ...
```

iii. These are the main pure-Python loops left in the implementation.

## 9-c. What processing does the code repeat multiple times?

i. Same as 7-c: static environment inputs are regenerated per trial, session outputs are re-concatenated for summaries, and list appends repeat the same structural work across all sessions.

ii. <Code snippets>

```python
input_trial = env_mat.astype(np.float32)
...
all_out = np.concatenate([
    np.concatenate([t[0] for t in sess]) for sess in data['output']
])
```

iii. The repetition is a direct consequence of storing session-static inputs as one object per trial and doing an extra diagnostic pass after conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Same as 7-d: optional plots, verbose summaries, and extra metadata/diagnostic calculations are not consumed by downstream decoder training.

ii. <Code snippets>

```python
plot_processing(animal, d, all_neural, all_input, all_output, session_info)
...
'session_info': all_session_info,
...
print(f"\nOutput distribution (position_bin):")
```

iii. The agent added these for inspection and validation rather than because the target format required them.
