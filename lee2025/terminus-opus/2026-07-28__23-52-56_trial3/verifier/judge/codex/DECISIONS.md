# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a list of seven animals, loads one serialized joblib file per animal from `data/`, extracts the nested dict for that animal, and then pulls the full `trace`, `position`, and `envs` arrays into memory. Trials are not loaded directly; they are created later by slicing each session into fixed 1-minute windows.

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
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)
```

iii. The justification in `CONVERSION_NOTES.md` says the reference repo’s `load_dat` path uses `joblib.load()` on non-`.mat` files, and the trajectory repeatedly states that the non-`.mat` joblib files in `data/` are the intended source format for loading.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. Each animal name becomes one subject, and `subject_idx` is assigned from the order of that list as sessions are appended.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
for a_idx, animal in enumerate(animals_to_process):
    ...
    for sess in sessions:
        ...
        all_subject_idx.append(a_idx)
...
'subjects': [a for a in animals_to_process],
'subject_idx': np.array(all_subject_idx, dtype=int),
```

iii. The notes say there are 7 animals and list them explicitly. The trajectory shows the agent inspected the files, concluded these seven mice define the dataset, and then encoded that list directly into the script.

## 1-c. How are the data split into sessions?

i. Each index along the first axis of the per-animal arrays is treated as one recording session or day. The code loops over `range(n_sessions)` and builds one session record per day.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)

n_sessions = trace.shape[0]
...
for day in range(n_sessions):
    env_name = envs[day, 0]
    tr = trace[day]
    pos = position[day]
```

iii. `CONVERSION_NOTES.md` says “Each day/recording is one session. Each animal contributes multiple sessions,” and the trajectory states the agent inferred session structure from the first dimension of `trace`, `position`, and `envs`.

## 1-d. How are the data split into trials?

i. Within each session, trials are defined as consecutive non-overlapping 60-second chunks at 30 Hz, so each trial is 1800 frames. The number of trials is the integer floor of `n_timepoints / 1800`; any remainder is dropped.

ii.
```python
FPS = 30
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC  # 1800 frames
...
n_timepoints = trace.shape[2]
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
...
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes explicitly call this out as a key decision: “Split each 40-min session into 1-minute trials (1800 frames at 30fps).”

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply any explicit trial-level quality control. Every full 1-minute chunk is kept, and only incomplete trailing chunks are excluded by integer division.

ii.
```python
n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    ...
```

iii. The notes say there is “No explicit trial structure in original data” and do not describe any trial rejection rule. The agent’s stated rationale is that trialing is an artificial segmentation step rather than a curated trial selection step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data comes from `d['trace']`, which the agent interprets as binarized rising-phase calcium-transient activity with shape `(n_sessions, n_neurons, n_timepoints)`.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
...
tr = trace[day]  # (n_neurons, n_timepoints)
```

iii. `CONVERSION_NOTES.md` states that `trace` contains “binary calcium transient data” and that the reference code uses the trace data directly because it is already preprocessed.

## 2-b. How is the `neural` data processed?

i. For each session, the agent selects neurons whose entire trace is not NaN, keeps the remaining `(n_active, n_timepoints)` matrix as-is, replaces any remaining NaNs with zero, and converts each trial slice to `float32`. It does not apply temporal smoothing, deconvolution, rebinning, or activity-threshold filtering during conversion.

ii.
```python
tr = trace[day]  # (n_neurons, n_timepoints)
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]  # (n_active, n_timepoints)
...
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The notes say “Use raw binary trace data (0/1) for active neurons. No additional temporal binning at this stage,” and explain that temporal binning in the paper’s decoder should not be duplicated in conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter is removal of neurons that are all NaN within a session, which the agent interprets as neurons not tracked that day. It explicitly does not apply the paper decoder’s velocity-based cell filtering or split-half place-cell filtering at conversion time.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]
```

iii. The notes say “Include all active (non-NaN) neurons” and “Cell filtering: NOT applied at conversion.” The trajectory records the agent discovering that NaN rows correspond to neurons not tracked on a given day and choosing to exclude only those.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. In the actual code, neural data is not aligned to a behavioral or stimulus event; it is just sliced into consecutive 60-second windows. However, the metadata labels the alignment event as the “Start of recording session” with `off_start = 0.0` and `off_end = 60`.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
...
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
    ...
},
```

iii. The notes describe the recordings as continuous and the trials as artificial 1-minute chunks, but the final metadata still names the start of the recording session as the alignment event. No stronger justification than that metadata choice appears in the notes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data remains at the native 30 Hz frame rate, so each bin is `1000 / 30` ms, about 33.33 ms. No temporal rebinning is applied during conversion.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. The notes explicitly say “Keep original 30Hz resolution” and “No additional temporal binning at this stage.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The agent derives environment geometry from `d['envs']`, the session-level environment-name strings. It does not use the raw `blocked` field to build the decoder input.

ii.
```python
envs = d['envs']         # (n_sessions, 1)
...
env_name = envs[day, 0]
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. The notes say “Use get_env_mat() for canonical environment representation,” and the trajectory shows the agent deciding against `blocked` because it believed the blocked indices used inconsistent orientations relative to `get_env_mat`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The environment name is mapped through a hard-coded lookup table of 10 geometries, producing a 3×3 binary matrix with 1 for accessible positions and 0 for blocked positions. That matrix is flattened to length 9 and reused as a static per-trial input vector for every trial in the session.

ii.
```python
def get_env_mat(env):
    envs = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        ...
    }
    return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
trial_input.append(env_mat.astype(np.float32))
```

iii. `CONVERSION_NOTES.md` calls this the “canonical environment representation” and says the decoder input should be a 9-element static geometry vector because the environment does not change within a session or trial.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from `d['position']`, the session-level 2D `(x, y)` coordinates in centimeters.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
...
pos = position[day]  # (2, n_timepoints)
```

iii. The notes identify `position` as the DeepLabCut-tracked mouse location in a 75 cm by 75 cm arena and map it directly to the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent converts continuous `(x, y)` position into a 3×3 grid. It divides each coordinate by 25 cm, floors to bin indices 0 to 2, clips to stay within range, and then combines them into one category index.

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
...
pos_bins = position_to_bin(pos)  # (n_timepoints,)
```

iii. The notes justify this as a task-driven change from the paper’s finer 15×15 decoding bins to the requested 3×3 output space.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis is thresholded into three 25 cm bins over the 0 to 75 cm arena. The final class label is `x_bin * 3 + y_bin`, yielding integer categories 0 through 8.

ii.
```python
BIN_SIZE_CM = ENV_SIZE_CM / N_SPATIAL_BINS  # 25cm per bin
...
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. The notes say “Position (0-75cm) / 25 = 0,1,2 for each axis. Combined bin = row*3 + col = 0-8,” although the code actually implements `x_bin * 3 + y_bin`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position is aligned frame-for-frame with neural data because both come from the same session arrays and are sliced with the same `start:end` trial boundaries.

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
...
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes say the data streams share the same 30 Hz sampling and that the same trial segmentation is applied to both neural activity and position.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing neurons are handled by removing rows that are all NaN within a session. Any residual NaNs in kept neurons are replaced with zero as a safety step. Position values at arena boundaries are clipped into valid bins, and incomplete trailing timepoints that do not fill a full minute are discarded.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
```

iii. The notes say the NaN rows correspond to neurons not tracked on a given day and that clipping prevents boundary issues. They also say the extra `nan_to_num` step is only a precaution.

## 6-a. What are the most time-consuming steps of the code?

i. The code’s slowest steps are loading each full-animal joblib file, iterating through all sessions and trials to materialize per-trial arrays, and saving the final very large pickle. Optional verification plotting also adds noticeable cost when enabled.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_sessions):
    ...
    for t in range(n_trials):
        ...
        trial_neural.append(active_neurons[:, start:end].astype(np.float32))
        trial_input.append(env_mat.astype(np.float32))
        trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes’ runtime table reports per-animal load time, per-animal processing time, and a large save time, which is the only explicit justification the agent recorded.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunity is the inner trial-building loop. The code slices and casts each 1-minute trial one at a time for neural data, position labels, and repeated copies of the same environment vector, all of which could have been reshaped or split more directly. The final output-distribution pass also revisits every trial after conversion.

ii.
```python
trial_neural = []
trial_input = []
trial_output = []

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
all_bins = np.concatenate(all_bins)
```

iii. No explicit optimization rationale was documented. This answer is inferred from the code structure and the runtime notes.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts the same session-level environment vector to `float32` once per trial, repeatedly reshapes and casts each output slice one trial at a time, and then loops over all outputs again later just to compute a printed distribution. It also recalculates per-session trial lists even though the session length is fixed within an animal.

ii.
```python
env_mat = get_env_mat(env_name).flatten()  # (9,)
...
for t in range(n_trials):
    ...
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
...
all_bins = []
for sess_outputs in all_output:
    for trial_out in sess_outputs:
        all_bins.append(trial_out[0])
```

iii. No explicit justification was found in the notes or trajectory; the repeated work is visible only in the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code optionally generates diagnostic plots, computes and prints a global output-bin distribution, prints extensive timing and summary statistics, and stores some intermediate per-session bookkeeping purely to assemble metadata or console output. None of that affects the converted neural/input/output arrays used downstream.

ii.
```python
if show_processing:
    plot_processing(d, animal, sessions)
...
all_bins = []
for sess_outputs in all_output:
    for trial_out in sess_outputs:
        all_bins.append(trial_out[0])
all_bins = np.concatenate(all_bins)
counts = np.bincount(all_bins.astype(int), minlength=9)
print(f"  Output distribution (9 bins): {counts / counts.sum()}")
...
session_info.append({
    'animal': sess['animal'],
    'env_name': sess['env_name'],
    'n_active': sess['n_active'],
    'n_trials': len(sess['neural']),
})
```

iii. The notes describe these steps as sanity checks, validation, or reporting. They are justified for debugging, but not required by the final decoder data format itself.
