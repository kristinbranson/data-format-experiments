# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the raw `.mat` files. It hard-coded the seven subject IDs, loaded one extensionless joblib file per subject from `/app/data`, and then read the preprocessed per-subject dictionary entries `trace`, `position`, `envs`, and `blocked`. Trials were then created later by splitting each per-day recording into minute windows.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
for subj_i, subject in enumerate(SUBJECTS):
    print(f'Loading {subject}', flush=True)
    source = joblib.load(DATA_DIR / subject)[subject]
    traces = source['trace']       # day x union-neuron x frame
    positions = source['position'] # day x (x,y) x frame
    envs = source['envs']
    blocked = source['blocked']
```

iii. In the trajectory, the AI first inspected both the `.mat` files and the extensionless files, concluded that the extensionless files were compressed serialized outputs from the repository, and then explicitly decided to use them because they already contained “aligned 30-Hz binary rising-transient traces and DLC position” and were more tractable than reprocessing the larger `.mat` files. It described this choice again in the docstring it wrote into `convert_data.py`.

## 1-b. How are the data split into subjects?

i. The AI treated each hard-coded subject file as one mouse. Subject identity came directly from the hard-coded `SUBJECTS` list, and each session appended during that subject loop received the corresponding `subj_i`.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
for subj_i, subject in enumerate(SUBJECTS):
    ...
    for day in range(traces.shape[0]):
        ...
        subject_idx.append(subj_i)
...
'subjects': SUBJECTS,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. In the trajectory, the AI summarized the dataset as seven animals and chose to preserve those seven animals exactly. It justified the hard-coded list as matching the discovered repository contents and the paper totals.

## 1-c. How are the data split into sessions?

i. The AI treated each recording day within a subject’s `trace`/`position` arrays as one session. Every `day` index in `traces.shape[0]` produced one output session.

ii.
```python
traces = source['trace']       # day x union-neuron x frame
positions = source['position'] # day x (x,y) x frame
...
for day in range(traces.shape[0]):
    tr = np.asarray(traces[day])
    pos = np.asarray(positions[day])
    ...
    neural.append([counts[t] for t in range(n_trials)])
    inputs.append([geometry.copy() for _ in range(n_trials)])
    outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
```

iii. In the trajectory, the AI stated that “a recording day is a session” and that it would “preserve all 207 recording days as sessions.” It tied this to reproducing the paper’s reported 207 sessions.

## 1-d. How are the data split into trials?

i. The AI split each continuous session into complete, non-overlapping 60-second windows. With `FS = 30`, each trial started as `1800` source frames. Any short trailing fragment was dropped. After that, each minute was rebinned into 60 one-second bins.

ii.
```python
FS = 30
BIN_FRAMES = 30
TRIAL_SECONDS = 60
BINS_PER_TRIAL = TRIAL_SECONDS
TRIAL_FRAMES = FS * TRIAL_SECONDS
...
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f'too few complete trials: {subject} day {day}')
used = n_trials * TRIAL_FRAMES
...
counts = tr[:, :used].reshape(
    n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
```

iii. In the trajectory, the AI repeatedly said that the task instructions required one-minute trials and that only complete 60-second windows should be kept. It justified dropping the trailing remainder so all trials would have identical duration.

## 1-e. How are trials filtered based on quality controls?

i. The AI did not filter individual trials by behavioral or neural quality metrics. The only trial-level curation was implicit: it discarded incomplete trailing fragments that did not fill a full minute, and it would raise an error if a session had fewer than two complete one-minute trials.

ii.
```python
n_trials = n_frames // TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f'too few complete trials: {subject} day {day}')
used = n_trials * TRIAL_FRAMES
```

iii. In the trajectory, the AI justified the `n_trials < 2` check using the decoder requirement that each session have at least two trials. It did not cite any paper-specific trial rejection rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from the `trace` entry inside each subject’s joblib file, not from the raw HDF5-backed `.mat` file. It interpreted that joblib `trace` as already-processed, aligned binary calcium-event activity arranged as day by union-neuron by frame.

ii.
```python
source = joblib.load(DATA_DIR / subject)[subject]
traces = source['trace']       # day x union-neuron x frame
...
tr = np.asarray(traces[day])
```

iii. In the trajectory, the AI wrote that the repository’s extensionless files contained “the paper’s aligned 30-Hz binary rising-transient traces,” and therefore chose them as the source of neural data rather than loading the `.mat` file directly.

## 2-b. How is the `neural` data processed?

i. The AI assumed the source traces were binary event indicators. It removed absent neurons, chopped the continuous recording into full-minute chunks, and then rebinned each 30-frame second by summing the binary events within that second. The final neural trial arrays therefore have shape `(n_neurons, 60)` and represent per-second event counts.

ii.
```python
finite = np.isfinite(tr)
valid = finite.all(axis=1)
...
tr = tr[valid]
...
counts = tr[:, :used].reshape(
    n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
...
neural.append([counts[t] for t in range(n_trials)])
```

iii. In the trajectory, the AI explicitly argued that keeping 30 Hz data would make the decoder dataset impractically large, so it chose synchronized 1-second bins instead. It justified summing within each second as preserving “event mass” while keeping the dataset tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI kept only neurons whose entire session trace was finite. Rows that were all `NaN` were treated as cells absent from that day and removed. It also checked for a case where a neuron was only partially missing within a session and would raise an error if that happened.

ii.
```python
finite = np.isfinite(tr)
valid = finite.all(axis=1)
if np.any(finite.any(axis=1) != valid):
    raise ValueError(f'partially missing neuron: {subject} day {day}')
tr = tr[valid]
```

iii. In the trajectory, the AI ran an audit across subjects and concluded that each neuron row was either wholly finite or wholly `NaN` within a session. It used that result to justify session-wise removal of all-`NaN` rows and tied the resulting totals to the paper’s reported 69,744 rate maps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI did not align neural data to a biological or task event. Instead, it defined the alignment event as the start of each non-overlapping one-minute window, then represented neural activity in one-second bins counted from that artificial trial start.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'start of each non-overlapping one-minute window',
    'off_start': 0.0,
    'off_end': 60.0,
    ...
}
```

iii. In the trajectory, the AI reasoned that the recording was continuous and that the instructions nonetheless required a temporal alignment field, so it used trial-window start as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI used a `1000.0` ms time bin and rebinned the original 30 Hz recordings by aggregating each consecutive 30 frames into one second.

ii.
```python
FS = 30
BIN_FRAMES = 30
...
counts = tr[:, :used].reshape(
    n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
...
'time_bin_size': 1000.0,
'source_sampling_rate_hz': 30.0,
```

iii. In the trajectory, the AI explicitly chose one-second bins for decoder tractability and described the rebinned neural signal as event counts per second.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derived environment geometry from the per-day `blocked` entry in the joblib file. It did not use `envs` to build the decoder input, though it stored the environment label in metadata.

ii.
```python
envs = source['envs']
blocked = source['blocked']
...
block_values = np.asarray(blocked[day], dtype=float).reshape(-1)
```

iii. In the trajectory, the AI identified the geometry input as the “blocked-region descriptions” in the processed files and decided that the decoder input should be the blocked grid-cell pattern.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converted the blocked-cell indices into a length-9 binary vector, where each entry indicates whether that spatial bin is blocked. Values outside `0..8` were ignored, which effectively treats the square’s `-1` sentinel as “no blocked cells.” The same geometry vector was copied into every trial of the session.

ii.
```python
geometry = np.zeros(9, dtype=np.float32)
block_values = np.asarray(blocked[day], dtype=float).reshape(-1)
for b in block_values:
    if np.isfinite(b) and 0 <= int(b) < 9:
        geometry[int(b)] = 1.0

inputs.append([geometry.copy() for _ in range(n_trials)])
```

iii. In the trajectory, the AI justified this as a direct encoding of the supplied blocked-grid geometry and noted that the square’s `-1` code meant no blocked cells.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derived output position from the per-day `position` entry in the joblib files. It interpreted this as aligned DeepLabCut `x, y` positions in centimeters.

ii.
```python
positions = source['position'] # day x (x,y) x frame
...
pos = np.asarray(positions[day])
if pos.shape[0] != 2 and pos.shape[1] == 2:
    pos = pos.T
```

iii. In the trajectory, the AI said the repository files already contained aligned DLC position and that position was finite and scaled to the 0 to 75 cm arena coordinates.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first averaged `x` and `y` within each one-second bin, then discretized those mean positions into the 3 x 3 arena grid. The resulting output is a single categorical spatial-bin label at each of 60 one-second time bins per trial.

ii.
```python
mean_pos = pos[:, :used].reshape(
    2, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
xbin = np.clip(np.floor(mean_pos[0] / 25.0), 0, 2).astype(np.int64)
ybin = np.clip(np.floor(mean_pos[1] / 25.0), 0, 2).astype(np.int64)
spatial_class = ybin * 3 + xbin
...
outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
```

iii. In the trajectory, the AI justified averaging position over the same one-second bins used for neural counts so the two streams would remain synchronized after rebinning.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI divided the 75 cm square arena into three equal bins per axis using boundaries at 25 and 50 cm. The categorical label is `y_bin * 3 + x_bin`, i.e. row-major order with `x` selecting the column and `y` selecting the row.

ii.
```python
xbin = np.clip(np.floor(mean_pos[0] / 25.0), 0, 2).astype(np.int64)
ybin = np.clip(np.floor(mean_pos[1] / 25.0), 0, 2).astype(np.int64)
spatial_class = ybin * 3 + xbin
...
'output_values': [[f'row_{r}_column_{c}' for r in range(3) for c in range(3)]],
```

iii. In the trajectory, the AI cited the paper’s 3 x 3 partition of the 75 cm square and stated that the class should be `row * 3 + column`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligned position and neural activity by using the same complete source frames, the same one-minute trial boundaries, and the same one-second sub-bins for both. Neural bins were sums over 30 frames; position bins were means over those same 30 frames.

ii.
```python
counts = tr[:, :used].reshape(
    n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
mean_pos = pos[:, :used].reshape(
    2, n_trials, BINS_PER_TRIAL, BIN_FRAMES
).mean(axis=3)
```

iii. In the trajectory, the AI repeatedly stated that the streams were already aligned at 30 Hz and that it would apply identical second-long windows to both signals to preserve synchronization after rebinning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removed all-`NaN` neuron rows as absent cells, raised an error if a neuron was only partially missing within a session, and discarded any trailing frames that did not complete a full one-minute trial. It did not impute missing neural or position samples.

ii.
```python
finite = np.isfinite(tr)
valid = finite.all(axis=1)
if np.any(finite.any(axis=1) != valid):
    raise ValueError(f'partially missing neuron: {subject} day {day}')
tr = tr[valid]
...
n_trials = n_frames // TRIAL_FRAMES
used = n_trials * TRIAL_FRAMES
```

iii. In the trajectory, the AI justified the all-`NaN` removal by auditing the data and confirming that missing neurons appeared exactly as absent-session markers rather than partial corruption. It justified dropping leftover frames so all trials would have uniform duration.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by loading the large per-subject joblib files and then performing the session-wide reshape, sum, and mean operations needed for one-second binning. The explicit `gc.collect()` after each subject suggests it expected these stages to be the memory and time bottlenecks.

ii.
```python
for subj_i, subject in enumerate(SUBJECTS):
    print(f'Loading {subject}', flush=True)
    source = joblib.load(DATA_DIR / subject)[subject]
    ...
    counts = tr[:, :used].reshape(
        n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
    ).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
    mean_pos = pos[:, :used].reshape(
        2, n_trials, BINS_PER_TRIAL, BIN_FRAMES
    ).mean(axis=3)
...
del source, traces, positions
gc.collect()
```

iii. In the trajectory, the AI discussed the large file sizes, worried about decoder tractability, and chose one-second aggregation partly to reduce downstream computational cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most of the expensive numeric work is already vectorized with `reshape`, `sum`, and `mean`. The remaining explicit loops are the subject/session loops, the short loop over blocked indices, and the per-trial list constructions when appending `counts[t]`, `geometry.copy()`, and `spatial_class[t][None, :]`.

ii.
```python
for subj_i, subject in enumerate(SUBJECTS):
    ...
    for day in range(traces.shape[0]):
        ...
        for b in block_values:
            if np.isfinite(b) and 0 <= int(b) < 9:
                geometry[int(b)] = 1.0

        neural.append([counts[t] for t in range(n_trials)])
        inputs.append([geometry.copy() for _ in range(n_trials)])
        outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
```

iii. The trajectory shows that the AI intentionally used vectorized reshaping for the heavy temporal aggregation. It did not discuss further vectorization for the smaller loops.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats some session-constant or defensive processing: it rechecks and possibly transposes `pos` and `tr` every session, rebuilds the same geometry vector copy for every trial within a session, and stores repeated per-session metadata fields. It also reruns the same reshape-based aggregation separately for neural and position streams.

ii.
```python
if pos.shape[0] != 2 and pos.shape[1] == 2:
    pos = pos.T
if tr.shape[1] != pos.shape[1] and tr.shape[0] == pos.shape[1]:
    tr = tr.T
...
inputs.append([geometry.copy() for _ in range(n_trials)])
...
session_info.append({
    'subject': subject,
    'source_session_index': day,
    'environment': env,
    ...
})
```

iii. The trajectory indicates these repetitions came from the AI trying to be defensive about source layout and from its choice to represent geometry as a per-trial static input instead of a session-level field.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores metadata that the decoder does not use, including the stringified `envs` label and a detailed `session_info` record. It also enforces the 207-session and 69,744-session-neuron totals with a hard runtime assertion; that is useful for validation but not for downstream decoding.

ii.
```python
env = scalar_env(envs[day])
session_info.append({
    'subject': subject,
    'source_session_index': day,
    'environment': env,
    'blocked_grid_indices': [int(b) for b in block_values if b >= 0],
    'source_frames': int(n_frames),
    'used_frames': int(used),
    'n_complete_one_minute_trials': int(n_trials),
    'n_neurons_present': int(n_neurons),
})
...
if len(neural) != 207 or total_valid != 69744:
    raise RuntimeError(f'unexpected source totals: {len(neural)} sessions, '
                       f'{total_valid} session-neurons')
```

iii. In the trajectory, the AI explicitly used the paper totals as a sanity check and wanted rich metadata documenting its decisions, even though those fields are not consumed by the decoder.
