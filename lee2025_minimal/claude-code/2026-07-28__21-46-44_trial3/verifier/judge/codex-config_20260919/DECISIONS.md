# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes seven animal IDs, loads one extensionless joblib file per animal from `/app/data`, selects the dictionary entry keyed by that animal ID, and iterates through every day in its in-memory `trace`, `position`, and `envs` arrays. Full mode uses all animals; `--sample` uses the first two.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for animal_idx, animal in enumerate(animals):
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]
    n_days = d['trace'].shape[0]
    for day in range(n_days):
```

iii. The trajectory shows that the agent inspected every animal file with `joblib.load`, confirmed the array shapes, and concluded that each file contains one animal and all its recording days. It later cited the resulting 7 animals, 207 sessions, and 69,744 neuron-sessions as matching the paper.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is treated as one subject. `subjects` preserves that list, and each retained recording day receives the corresponding zero-based `animal_idx`.

ii.
```python
subjects = list(animals)
...
for animal_idx, animal in enumerate(animals):
    ...
    subject_idx_list.append(animal_idx)
```

iii. The agent inferred from inspecting the files that each top-level joblib file/dictionary key represents one mouse. The seven IDs also matched the paper’s animal count.

## 1-c. How are the data split into sessions?

i. Each recording day (`day` along axis 0) becomes one output session, unless it fails the agent’s session-level minimum-neuron or minimum-trial checks.

ii.
```python
n_days = d['trace'].shape[0]
for day in range(n_days):
    trace_day = d['trace'][day]
    pos_day = d['position'][day]
    env_name = str(d['envs'][day, 0])
    ...
    neural_all.append(neural_trials)
```

iii. The trajectory explicitly states “Sessions = days (one recording per day)” because each day has its own recording and environment geometry.

## 1-d. How are the data split into trials?

i. A session is divided into consecutive, non-overlapping 60-second trials at 30 Hz (1,800 frames). Only complete trials are retained; trailing frames are discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_traces = traces[:, start:end]
    trial_pos = pos_bins[start:end]
```

iii. The agent followed the explicit instruction to split long sessions into one-minute trials and noted that integer division intentionally drops the final incomplete minute.

## 1-e. How are trials filtered based on quality controls?

i. Individual trials receive no content-based quality filtering. A whole day/session is skipped if it has fewer than five registered neurons or yields fewer than two complete trials; otherwise all complete one-minute trials are retained.

ii.
```python
if n_registered < 5:
    continue
...
if n_trials < 2:
    continue
```

iii. The trajectory does not explain the five-neuron cutoff. The two-trial check implements the target-format requirement that every session have at least two trials. In this dataset, the agent reported roughly 40-minute sessions, so these checks apparently removed no sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the per-animal `trace` array, indexed by recording day.

ii.
```python
trace_day = d['trace'][day]  # (n_cells, n_timepoints)
```

iii. After inspecting sample values and the methods/reference code, the agent identified `trace` as binary calcium-transient events extracted from calcium fluorescence.

## 2-b. How is the `neural` data processed?

i. The already neuron-by-time daily trace is restricted to registered cells, remaining NaNs are changed to zero, and each trial slice is cast to `float32`. No smoothing, temporal aggregation, normalization, or deconvolution is applied by the converter.

ii.
```python
traces = trace_day[registered]
traces = np.nan_to_num(traces, nan=0.0)
...
neural_trials.append(trial_traces.astype(np.float32))
```

iii. The agent reasoned that the stored traces were already binary calcium transient events (0/1), so further signal processing was unnecessary. Its notes say the original rise-phase extraction used a z-score threshold greater than 2.5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells whose entire daily trace is NaN are removed as unregistered for that day. A session is also skipped if fewer than five such registered cells remain. Any partial NaNs in retained cells are replaced with zero.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
n_registered = registered.sum()
if n_registered < 5:
    continue
traces = np.nan_to_num(trace_day[registered], nan=0.0)
```

iii. The trajectory says NaN traces indicate cells not registered on a particular day. It also states that no additional biological filtering was desired because the paper motivated including all cells, but it gives no specific justification for the five-cell session threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event realignment. Trials are fixed windows aligned to successive one-minute boundaries from the beginning of each recording session.

ii.
```python
start = t * FRAMES_PER_TRIAL
end = (t + 1) * FRAMES_PER_TRIAL
trial_traces = traces[:, start:end]
```

iii. The agent described the underlying streams as natively synchronized at 30 Hz and documented the alignment event as the start of the recording session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz resolution is retained, giving `1000/30 = 33.33...` ms per frame. No temporal rebinning or resampling is performed.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The agent explicitly selected the native 30 Hz time bin after inspecting the data and paper and found no need to resample already aligned streams.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. It is derived from the string environment name in `d['envs'][day, 0]`, not from the raw `blocked` field. The name indexes a hard-coded mapping copied from the repository’s `get_env_mat` logic.

ii.
```python
env_name = str(d['envs'][day, 0])
...
env_mat = get_env_mat(env_name)
```

iii. The agent inspected `blocked`, checked its relationship to environment names, and then chose the named-environment matrices because it viewed the repository helper as the authoritative environment geometry representation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name maps to a 3×3 binary matrix where 1 means open and 0 means blocked. That matrix is flattened row-major into a static nine-element vector, cast to `float32`, and repeated for every trial in the session.

ii.
```python
env_mats = {
    'square': [[1,1,1],[1,1,1],[1,1,1]],
    'o':      [[1,1,1],[1,0,1],[1,1,1]],
    ...
}
...
env_input = get_env_mat(env_name).flatten()
input_trials.append(env_input.astype(np.float32))
```

iii. The agent justified this as a direct reuse of reference repository code, with nine independent grid locations and a constant geometry throughout each recording day. Its documented convention is explicitly open=1, blocked=0.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the day’s two-coordinate time series in `d['position']`.

ii.
```python
pos_day = d['position'][day]  # (2, n_timepoints)
pos_bins = discretize_position(pos_day)
```

iii. The agent identified `position` as the mouse’s x-y coordinates in the 75×75 cm arena and used it because the requested target is time-varying mouse position.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Coordinates are clipped to the arena, divided by 25 cm, floored into x and y bins, clipped to 0–2, and combined into one categorical label. Each trial output is stored with shape `(1, 1800)` and integer dtype.

ii.
```python
x = np.clip(position[0], 0, arena_size - 1e-10)
y = np.clip(position[1], 0, arena_size - 1e-10)
x_bin = np.floor(x / (arena_size / n_bins)).astype(int)
y_bin = np.floor(y / (arena_size / n_bins)).astype(int)
bin_idx = x_bin * n_bins + y_bin
...
output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))
```

iii. The agent chose a 3×3 grid because the task requires nine position classes and because 25×25 cm bins match the physical partition grid. It described the ordering as row-major with the first coordinate as the row.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis uses equal-width thresholds at 25 cm and 50 cm: `[0,25)`, `[25,50)`, and `[50,75]`, after clipping. The two axis bins form classes 0–8 via `x_bin * 3 + y_bin`.

ii.
```python
bin_size = arena_size / n_bins
x_bin = np.floor(x / bin_size).astype(int)
y_bin = np.floor(y / bin_size).astype(int)
bin_idx = x_bin * n_bins + y_bin
```

iii. The trajectory justifies equal 25 cm bins as matching both the requested 3×3 discretization and the physical arena partitions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position arrays are assumed frame-aligned and are sliced with exactly the same trial start/end indices. No interpolation or lag correction is applied.

ii.
```python
trial_traces = traces[:, start:end]
trial_pos = pos_bins[start:end]
```

iii. The agent states that behavioral and imaging streams were acquired synchronously at 30 Hz, so identical slicing preserves alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Entirely NaN cell-days are dropped, partial NaNs in retained traces are replaced by zero, coordinates are clipped to legal arena bounds, unknown environment names raise an error, incomplete final trial fragments are dropped, and short/low-neuron sessions are skipped.

ii.
```python
registered = ~np.all(np.isnan(trace_day), axis=1)
traces = np.nan_to_num(traces, nan=0.0)
x = np.clip(position[0], 0, arena_size - 1e-10)
...
raise ValueError(f"Unknown environment: {env}")
n_trials = n_timepoints // FRAMES_PER_TRIAL
```

iii. The agent regarded all-NaN traces as non-registration rather than observations, called zero-filling a safety measure for unexpected residual NaNs, clipped boundary values to valid classes, and accepted losing less than one minute when keeping only fixed-length trials.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant conversion costs are loading seven large joblib objects, scanning full traces for NaNs, copying/casting all retained neural trial arrays, and serializing the large result. The separate `sanity_checks` function also traverses every trial multiple times. The agent itself primarily identified data loading and full conversion as the expensive work.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
registered = ~np.all(np.isnan(trace_day), axis=1)
neural_trials.append(trial_traces.astype(np.float32))
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory inspected and converted large arrays for 207 sessions and used long timeouts for those operations. It did not provide a formal profile; its validation narrative focused on I/O-scale dataset statistics.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop could be replaced with reshape/split operations for neural and position data. The loops that create nine input/output names could be comprehensions. The animal/day loops are appropriate because neuron counts and session metadata vary.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    neural_trials.append(traces[:, start:end].astype(np.float32))
    input_trials.append(env_input.astype(np.float32))
    output_trials.append(pos_bins[start:end][np.newaxis, :].astype(np.int64))
```

iii. The trajectory gives no explicit vectorization rationale. The straightforward loop was evidently chosen for clarity and to directly construct the required nested list-of-trials format.

## 6-c. What processing does the code repeat multiple times?

i. It casts the identical static environment vector once per trial, repeatedly casts trial slices, computes trial boundaries in a loop, and the sanity checks make separate complete passes for dimensions and output ranges. Summary reporting also revisits sessions and trials.

ii.
```python
input_trials.append(env_input.astype(np.float32))
...
for s in range(n_sessions):
    for t in range(n_trials):
        ...
for s in range(n_sessions):
    for t in range(len(data['output'][s])):
        ...
```

iii. The trajectory does not discuss repeated processing. It emphasizes successful validation, suggesting these extra passes were accepted as inexpensive safeguards relative to loading and storing the neural arrays.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Conversion-time statistics and verbose summaries are computed only for printing. `sanity_checks` computes sample unique neural values, neuron-session totals, per-subject neuron summaries, and repeated validations without changing saved data. `sys` is imported but unused. The `blocked` raw field is described and inspected in the trajectory but is not used by the final converter.

ii.
```python
unique_vals = np.unique(sample_neural)
total_neurons = sum(data['neural'][s][0].shape[0] for s in range(n_sessions))
...
n_neurons_list = [data['neural'][s][0].shape[0] for s in session_indices]
```

iii. The agent intentionally used these calculations as sanity checks against paper statistics (207 sessions and 69,744 neuron-sessions) and to verify decoder compatibility. They are useful for validation, but their results are not consumed by downstream decoding.
