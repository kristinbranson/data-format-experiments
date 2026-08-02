# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the `.mat` files used by the human reference. It hardcodes a list of 7 animal IDs, loads one joblib file per animal from `data/`, then iterates through each session and splits each session into 1-minute trial chunks in Python.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
...
trace = d['trace']
position = d['position']
envs = d['envs']
```

iii. In `CONVERSION_NOTES.md` Step 1 and the trajectory, the AI justified this by concluding that the non-`.mat` files in `data/` were the correct loading target and that `load_dat()` in the reference code supported joblib loading.

## 1-b. How are the data split into subjects?

i. Each subject is one animal ID from the hardcoded `ANIMALS` list. The output `subjects` field is the list of processed animal IDs, and each session gets a subject index from the outer loop over animals.

ii.
```python
for a_idx, animal in enumerate(animals_to_process):
    ...
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
    for sess in sessions:
        ...
        all_subject_idx.append(a_idx)
...
'subjects': [a for a in animals_to_process],
'subject_idx': np.array(all_subject_idx, dtype=int),
```

iii. The AI stated in the trajectory that there were 7 mice and that “each animal contributes multiple sessions,” so it treated each animal-level file as one subject.

## 1-c. How are the data split into sessions?

i. Within each animal, the AI treats the first axis of `trace`, `position`, and `envs` as session/day index. Each `day` becomes one session in the converted output.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)
...
for day in range(n_sessions):
    env_name = envs[day, 0]
    tr = trace[day]
    pos = position[day]
```

iii. In the trajectory and Step 5 notes, the AI justified this as “each day/recording is one session.”

## 1-d. How are the data split into trials?

i. The AI defines trials as non-overlapping 60-second windows at 30 Hz, i.e. 1800 frames per trial. It computes `n_trials` with integer division and discards any remainder frames.

ii.
```python
FPS = 30
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
```

iii. Step 5 in `CONVERSION_NOTES.md` says the AI deliberately chose 1-minute trials because the instruction required splitting long sessions into 1-minute segments.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-level quality filtering. The only effective filtering is that incomplete trailing fragments shorter than 1800 frames are omitted because `n_trials` uses floor division.

ii.
```python
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
```

iii. In Step 3 and Step 5 notes, the AI explicitly said there was “no explicit trial structure in original data” and chose not to add extra trial curation during conversion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data comes from `d['trace']`, loaded from each animal’s joblib data structure.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
...
tr = trace[day]  # (n_neurons, n_timepoints)
```

iii. The AI repeatedly documented in `CONVERSION_NOTES.md` that `trace` is the neural variable and interpreted it as binary rising-phase calcium transient data.

## 2-b. How is the `neural` data processed?

i. For each session, the AI keeps neurons that are not all-NaN, leaves the session in `(neurons, time)` orientation, replaces any remaining NaNs with zero, and stores each trial as `float32`. It does not apply temporal binning, velocity filtering, or place-cell filtering during conversion.

ii.
```python
tr = trace[day]  # (n_neurons, n_timepoints)
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. Step 5 says: “Use raw binary trace data (0/1) for active neurons. No additional temporal binning at this stage.” The trajectory also says the AI believed decoder-time preprocessing such as velocity filtering and 3-frame pooling should not be applied in conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filter is removing neurons whose entire session trace is NaN. Remaining NaNs are not used as a filter; they are replaced with zero.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]
...
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. In Step 3 and Step 5 notes, the AI justified this by saying NaN neurons indicate neurons not tracked on that day and that decoding should include “all active (non-NaN) neurons.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not perform event-based alignment. Trials are contiguous chunks from a continuous recording, but the metadata claims the alignment event is the “Start of recording session.”

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
...
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
}
```

iii. In Step 5, the AI treated the data as continuous free exploration with no natural per-trial event, but later encoded “Start of recording session” as the metadata alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native 30 Hz frame rate, with one frame per bin. The AI records the time bin size as `1000.0 / FPS` milliseconds and does not rebin in conversion.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. Step 5 explicitly states: “Time bin size: 1 frame = 1/30 sec ≈ 33.33ms. Keep original 30Hz resolution.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input from `d['envs']`, not from the `blocked` variable. It looks up a canonical 3x3 geometry mask from the environment name string.

ii.
```python
envs = d['envs']         # (n_sessions, 1)
...
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. In the trajectory, the AI decided the `blocked` field had inconsistent orientations and concluded it should use `get_env_mat(env_name)` as the “canonical representation used in the paper’s analysis.”

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The input is a flattened 3x3 binary geometry template returned by `get_env_mat`. In this representation, 1 means accessible and 0 means blocked. The AI does not encode the actual `blocked` indices from raw data.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        ...
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
trial_input.append(env_mat.astype(np.float32))
```

iii. Step 4 and Step 5 notes say the AI believed `get_env_mat()` gave a better canonical environment representation than the raw `blocked` field.

## 3-c. How is the `input` *Environment geometry* aligned with the neural data?

i. The geometry input is static within a session. The AI stores the same 9-element vector once per trial, with no time axis, so it is aligned only at the trial/session level rather than frame-by-frame.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
trial_input.append(env_mat.astype(np.float32))
```

iii. Step 5 says the environment input should be “static per trial (same for all timepoints in trial).”

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position comes from `d['position']`, using the per-session `(2, n_timepoints)` x/y coordinates.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
...
pos = position[day]  # (2, n_timepoints)
```

iii. The AI documented in Step 2 and Step 5 that `position` is the continuous x/y tracking variable in centimeters.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI discretizes x and y into 3 bins each using `floor(position / 25 cm)`, clips to `[0, 2]`, and combines them into one label. The implementation uses `x_bin * 3 + y_bin`, even though the notes say “row*3 + col.”

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. Step 5 says the AI chose a 3x3 grid because the instruction required 9 spatial bins, and described each bin as a 25 cm by 25 cm region.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Continuous x and y are thresholded at 25 cm and 50 cm to form 3 bins per axis. The code uses floor-based binning and clipping, then combines the two axis bins into one categorical label from 0 to 8.

ii.
```python
bin_size = env_size / n_bins
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. In Step 5, the AI justified this as a direct implementation of the instruction to decode “3 x 3 = 9 spatial bins.”

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is assumed to be sampled on the same 30 Hz timeline as the neural trace. The AI slices position-derived labels using the same `start:end` frame windows used for neural trials.

ii.
```python
pos_bins = position_to_bin(pos)  # (n_timepoints,)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The trajectory says the AI believed neural and position streams were already aligned frame-by-frame in the source data.

## 5-a. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses the original frame rate, 30 Hz, corresponding to about 33.33 ms per bin. No temporal rebinning is applied in `convert_data.py`.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The AI stated in Step 5 that decoder-side temporal pooling from the reference code should not be copied into conversion.

## 5-b. How are the neural, input, and output data temporally aligned?

i. Neural and output are aligned by identical frame slicing within each session. Input is a static per-trial vector copied for every trial in that session.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. Step 5 describes this explicitly: neural remains frame-wise, input is static per trial, and output is time-varying over the same frames.

## 6. How are minor issues in the data (e.g., missing data, malformed entries) handled?

i. The AI removes neurons that are entirely NaN, fills any remaining NaNs in retained neurons with zero, clips out-of-range positions into valid bins, and silently falls back to an all-zero geometry if an environment name is missing from `get_env_mat`.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
x_bins = np.clip(..., 0, n_bins - 1)
y_bins = np.clip(..., 0, n_bins - 1)
...
return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

iii. The only explicit justification is in code comments and Step 5 notes, where the AI called `np.nan_to_num(..., nan=0.0)` a safety measure and described clipping as keeping positions valid.

## 7-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading large joblib files, looping over every session and trial to build Python lists, optionally plotting sample sessions, and pickling a very large output object to disk.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_sessions):
    ...
    for t in range(n_trials):
        ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The trajectory repeatedly mentions load times per animal, total conversion time, and concern about multi-gigabyte output size.

## 7-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop could have been replaced by reshape/chunk operations for neural and output arrays, and the repeated `trial_input.append(...)` could have been generated without Python iteration. The later summary loop that concatenates outputs for a histogram is also a pure Python accumulation loop.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
...
all_bins = []
for sess_outputs in all_output:
    for trial_out in sess_outputs:
        all_bins.append(trial_out[0])
```

iii. The AI’s notes claim it tried to write efficient code, but the implementation still uses nested Python loops for chunking and aggregation.

## 7-c. What processing does the code repeat multiple times?

i. It re-casts the same per-session `env_mat` to `float32` for every trial, repeatedly slices arrays inside trial loops, and stores the same static input vector separately for each trial instead of sharing or broadcasting it.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
for t in range(n_trials):
    ...
    trial_input.append(env_mat.astype(np.float32))
```

iii. This follows directly from the implementation; there is no separate justification beyond the AI’s choice to make each trial self-contained.

## 7-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `--show-processing` path generates plots that are not part of the dataset, the code builds and prints global output histograms only for logging, and it stores extensive `session_info` metadata that the downstream decoder does not require.

ii.
```python
if show_processing:
    plot_processing(d, animal, sessions)
...
session_info.append({
    'animal': sess['animal'],
    'env_name': sess['env_name'],
    'n_active': sess['n_active'],
    'n_trials': len(sess['neural']),
})
...
counts = np.bincount(all_bins.astype(int), minlength=9)
print(f"  Output distribution (9 bins): {counts / counts.sum()}")
```

iii. The trajectory shows the AI deliberately added these for inspection and reporting rather than for the converted data format itself.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Minor data issues are handled by dropping all-NaN neurons, zero-filling any remaining NaNs, clipping position-derived bins into valid range, and defaulting unknown environment names to an all-zero 3x3 mask.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
...
return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
```

iii. The AI’s justification is the same as in Section 6: “safety” handling rather than reference-matched curation.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant runtime costs are bulk joblib I/O, Python-level session/trial chunking, and writing the large pickle file.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_sessions):
    ...
    for t in range(n_trials):
        ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The trajectory records per-animal load times, total runtime, and repeated concern about file size, which is the AI’s own evidence for these bottlenecks.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The same vectorization opportunities as in 7-b remain: trial chunking for neural/output, repeated replication of static input vectors, and Python loops used for summary statistics and plotting.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. There is no separate later justification; this is simply a property of the final implementation.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly slices session arrays into trials, repeatedly casts the same static input vector for every trial, and repeatedly walks the nested output lists again to compute summary distributions.

ii.
```python
for t in range(n_trials):
    ...
    trial_input.append(env_mat.astype(np.float32))
...
for sess_outputs in all_output:
    for trial_out in sess_outputs:
        all_bins.append(trial_out[0])
```

iii. This repetition comes from the AI’s trial-by-trial construction strategy.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. As above, the plotting path, printed summary histograms, and extra session-level metadata are not used by the downstream decoder and mainly serve logging or manual inspection.

ii.
```python
if show_processing:
    plot_processing(d, animal, sessions)
...
'session_info': session_info,
...
print(f"  Output distribution (9 bins): {counts / counts.sum()}")
```

iii. The trajectory shows these were added for validation and reporting rather than to reproduce the reference conversion itself.
