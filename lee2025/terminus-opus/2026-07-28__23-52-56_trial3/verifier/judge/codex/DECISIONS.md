# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes seven animal IDs, loads one `joblib` file per animal from `data/<animal>`, then reads session-wise arrays from the nested dict. It does not scan the `.mat` files used by the human reference. Trials are not loaded directly; they are created later by slicing each session into fixed 1-minute chunks.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

def process_animal(animal, data_dir=DATA_DIR, show_processing=False):
    dat = joblib.load(os.path.join(data_dir, animal))
    d = dat[animal]

    trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
    position = d['position'] # (n_sessions, 2, n_timepoints)
    envs = d['envs']         # (n_sessions, 1)
```

iii. In `CONVERSION_NOTES.md` Step 1 and trajectory step 17/43, the AI says the non-`.mat` joblib files are the practical loading path and describes the nested dict structure (`trace`, `position`, `envs`, etc.) as the source it chose to convert.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded entries in `ANIMALS`. Each animal name is also used as the subject identifier in the output `subjects` list.

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
...
if args.sample:
    animals_to_process = ANIMALS[:2]
else:
    animals_to_process = ANIMALS
...
'subjects': [a for a in animals_to_process],
```

iii. `CONVERSION_NOTES.md` Step 2 says there are 7 animals and lists per-animal statistics. The trajectory states that “Animals: 7 mice” and treats those IDs as the subject split.

## 1-c. How are the data split into sessions?

i. Each day/session is one slice along axis 0 of the per-animal arrays. The script loops over `range(n_sessions)` and appends one output session per day.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
position = d['position'] # (n_sessions, 2, n_timepoints)
envs = d['envs']         # (n_sessions, 1)

n_sessions = trace.shape[0]

for day in range(n_sessions):
    env_name = envs[day, 0]
    tr = trace[day]
    pos = position[day]
    ...
    sessions.append({
        'neural': trial_neural,
        'input': trial_input,
        'output': trial_output,
        'env_name': env_name,
        'n_active': n_active,
        'animal': animal,
    })
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “Each day/recording is one session. Each animal contributes multiple sessions.”

## 1-d. How are the data split into trials?

i. Trials are artificial, non-overlapping 1-minute windows at 30 Hz, so each trial is 1800 frames. The number of trials is `n_timepoints // 1800`, and any remainder at the end of the session is discarded.

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
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
    trial_input.append(env_mat.astype(np.float32))
    trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. `CONVERSION_NOTES.md` Step 5 says: “Split each 40-min session into 1-minute trials (1800 frames at 30fps).”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply trial-level quality-control filtering. It keeps every full 1-minute chunk and only drops incomplete trailing frames. In the notes it explicitly says velocity filtering is not applied during conversion.

ii.
```python
n_trials = n_timepoints // FRAMES_PER_TRIAL

for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    ...
```

iii. `CONVERSION_NOTES.md` Step 5: “Velocity filtering: NOT applied at data conversion stage.” No other trial rejection logic appears in the script.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` array inside each animal dict.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
...
tr = trace[day]  # (n_neurons, n_timepoints)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `trace` directly to `neural`, and Step 2 describes `trace` as the calcium transient data field.

## 2-b. How is the `neural` data processed?

i. For each session, the AI takes the session slice from `trace`, removes all-NaN neurons, replaces any remaining NaNs with zero, keeps the native frame rate, casts each trial to `float32`, and stores trials as `(n_neurons, 1800)` matrices. It does not apply temporal rebinning or the decoder-side activity thresholding from the reference code.

ii.
```python
tr = trace[day]  # (n_neurons, n_timepoints)
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
n_active = active_neurons.shape[0]

active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. `CONVERSION_NOTES.md` Step 5 says: “Use raw binary trace data (0/1) for active neurons. No additional temporal binning at this stage.” It also says the decoder can handle later filtering/binnning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level QC applied in the conversion script is removing neurons that are all NaN within a session. The AI explicitly chose not to apply the reference decoder’s activity-threshold cell filtering during conversion. It also zero-fills any residual NaNs as a safety step.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)  # neurons that are not all-NaN
active_neurons = tr[active_mask]
...
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
```

iii. `CONVERSION_NOTES.md` Step 5: “Cell filtering: NOT applied at conversion. Include all active (non-NaN) neurons.” Step 4 notes the reference `cell_threshold=5` but says not to use it here.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the start of the recording session as the alignment event and defines each trial as a contiguous 60-second segment starting from that origin. There is no event-specific within-session realignment beyond fixed slicing.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
...
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
}
```

iii. The metadata in `convert_data.py` encodes this choice directly. The notes also describe the source sessions as continuous recordings with no native trial structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data are kept at the native 30 Hz frame rate, so each time bin is about 33.33 ms. No temporal rebinning is applied in the conversion.

ii.
```python
FPS = 30  # frames per second
...
'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “Keep original 30Hz resolution.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives `input` from the `envs` session label, not from the raw `blocked` field. It converts each environment name into a canonical 3x3 geometry template.

ii.
```python
envs = d['envs']         # (n_sessions, 1)
...
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()  # (9,)
```

iii. In trajectory steps 35-36 and `CONVERSION_NOTES.md` Step 4, the AI says the `blocked` field did not match `get_env_mat` in a single orientation, so it chose `get_env_mat(env_name)` as the “canonical representation.”

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI hard-codes a lookup table from environment name to a 3x3 matrix with `1=accessible` and `0=blocked`, flattens that matrix to length 9, and reuses the same vector for every trial in the session.

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
env_mat = get_env_mat(env_name).flatten()
...
trial_input.append(env_mat.astype(np.float32))
```

iii. The notes state “Use get_env_mat() for canonical environment representation,” and the trajectory says this was chosen because the raw blocked indices seemed orientation-dependent.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` is derived from the raw `position` array for each session.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
...
pos = position[day]  # (2, n_timepoints)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `position` to the decoder output and Step 2 describes it as x/y coordinates in centimeters.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI converts continuous x/y coordinates in a 75 cm square arena into 3 equal bins per axis using `floor(pos / 25)` with clipping to `[0, 2]`. It then combines the axis bins into a single class label. The notes say the class is `row*3 + col`, but the implemented function actually computes `x_bin*3 + y_bin`.

ii.
```python
def position_to_bin(pos_xy, env_size=ENV_SIZE_CM, n_bins=N_SPATIAL_BINS):
    bin_size = env_size / n_bins
    x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
    bin_indices = x_bins * n_bins + y_bins
    return bin_indices
```

iii. `CONVERSION_NOTES.md` Step 5 justifies the 3x3 discretization as required by the task: each bin covers `25 cm x 25 cm`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is thresholded into three equal-width bins spanning `0-75 cm`: `[0,25)`, `[25,50)`, and `[50,75]` after clipping. The two per-axis bins are then combined into one of 9 categories. Again, the notes describe `row*3 + col`, but the implemented code uses `x_bin*3 + y_bin`.

ii.
```python
bin_size = env_size / n_bins
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. `CONVERSION_NOTES.md` Step 5: “Position (0-75cm) / 25 = 0,1,2 for each axis. Combined bin = row*3 + col = 0-8.”

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI assumes `position` and `trace` are already frame-aligned within each session and slices both with the same `start:end` trial boundaries, so alignment is frame-for-frame.

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
...
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes repeatedly describe both signals as being sampled at 30 Hz from the same recording session and do not mention any offset correction.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing whole-neuron session entries are handled by dropping all-NaN neurons. Any residual NaNs inside kept neurons are replaced with zero. Incomplete trailing data shorter than 1 minute are silently discarded because `n_trials` uses floor division. There is no other explicit repair or imputation.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
...
n_trials = n_timepoints // FRAMES_PER_TRIAL
```

iii. `CONVERSION_NOTES.md` Step 2 notes that NaN neurons are “not tracked on a given day,” and Step 5 says to include all active non-NaN neurons. The inline code comment says the zero-fill is a safety measure.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify file loading and per-animal processing as the main cost, and the code structure suggests the biggest costs are `joblib.load(...)`, serializing the very large pickle, and the Python loop over all sessions/trials. Optional plotting is also expensive when enabled.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
for day in range(n_sessions):
    ...
    for t in range(n_trials):
        ...
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. `CONVERSION_NOTES.md` Step 7 includes runtime estimates for load/process/save and attributes substantial per-animal time to those stages.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop inside each session could have been replaced with a reshape/split-based implementation, and the final nested loop used only to aggregate output-bin counts could also have been reduced. The current code repeatedly slices and casts trial arrays in Python.

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
all_bins = np.concatenate(all_bins)
```

iii. The task instructions explicitly asked for vectorization where possible. The AI did not justify these loops beyond straightforward implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly casts the same session-constant environment vector to `float32` once per trial, repeatedly casts neural/output slices inside the inner loop, and reconstructs summary arrays after the full dataset is already assembled.

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_input.append(env_mat.astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
...
counts = np.bincount(all_bins.astype(int), minlength=9)
```

iii. There is no explicit justification in the notes beyond implementation convenience; the trajectory emphasizes getting a working converter rather than optimizing these repeated operations.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains optional plotting code that is not used by downstream decoder training, computes summary statistics and output distributions purely for logging, tracks some metadata fields not needed by the reference decoder, and stores/prints some intermediate timing information. It also defines `n_neurons_total` without using it.

ii.
```python
n_neurons_total = trace.shape[1]
...
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

iii. The notes justify these as sanity checks and documentation aids. They are useful for inspection, but they are discarded by downstream training/analysis.
