# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter uses a fixed list of seven animal IDs and loads each animal's extensionless joblib file. It extracts that record's `envs`, `trace`, `position`, and `blocked` arrays, then iterates through every day/session. In sample mode it instead takes the first two animals.

ii.
```python
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
    'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

def load_animal(data_dir, animal):
    return joblib.load(Path(data_dir) / animal)[animal]

animals = ANIMALS[:2] if sample else ANIMALS
for subj_idx, animal in enumerate(animals):
    rec = load_animal(data_dir, animal)
    envs = np.array(rec['envs']).reshape(-1)
    trace = np.asarray(rec['trace'])
    position = np.asarray(rec['position'])
    blocked = rec['blocked']
```

iii. The agent stated that the reference repository's `load_dat` directly consumes the preprocessed joblib animal files, so recomputing data from MATLAB was unnecessary. It documented seven animals and 207 sessions and used full mode for the final artifact.

## 1-b. How are the data split into subjects?

i. Each named joblib file is one subject. The fixed `ANIMALS` list becomes `subjects`, and its enumeration index is attached to every retained day/session for that animal.

ii.
```python
animals = ANIMALS[:2] if sample else ANIMALS
data = {'subjects': animals.copy(), 'subject_idx': [], ...}
for subj_idx, animal in enumerate(animals):
    ...
    data['subject_idx'].append(subj_idx)
```

iii. The notes identify the seven per-animal datasets and treat each animal identifier as the subject name, consistent with the dataset organization.

## 1-c. How are the data split into sessions?

i. Every animal day is an output session. The number of days is taken from `envs`; corresponding indexed slices from trace, position, and blocked geometry are processed together. Sessions with fewer than two complete minute-long trials are skipped.

ii.
```python
n_days = len(envs)
for day in range(n_days):
    session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
    blocked_vec = blocked_to_vec(blocked[day])
    trace_day = np.asarray(trace[day], dtype=np.float32)
    pos_day = np.asarray(position[day], dtype=np.float32)
    ...
    if n_trials < 2:
        continue
```

iii. The agent found that sessions correspond to recording days/environments and noted 207 total sessions, one session per day.

## 1-d. How are the data split into trials?

i. Each continuous session is split from session start into non-overlapping 60-second windows of 1,800 samples at 30 Hz. The frame count is first limited to the shorter of neural and position streams, and any final incomplete window is discarded.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_SECONDS
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
used = n_trials * FRAMES_PER_TRIAL
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. Native recordings have no trials; the task explicitly requires one-minute trials. The agent chose contiguous windows and documented trimming partial trailing minutes to preserve uniform shapes and alignment.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial filtering. A session is excluded only if it contains fewer than two full trials; partial trailing data are dropped.

ii.
```python
n_trials = n_frames // FRAMES_PER_TRIAL
...
if n_trials < 2:
    print(f'Skipping {session_id}: only {n_trials} full 1-minute trial(s)')
    continue
```

iii. The agent found no native trial-curation rule and added the two-trial guard to satisfy the decoder format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come directly from each animal record's per-day `trace`, described as the precomputed binarized rising-phase calcium-event vector.

ii.
```python
trace = np.asarray(rec['trace'])
trace_day = np.asarray(trace[day], dtype=np.float32)
```

iii. The paper/code exploration established that `trace` is already the binary rise-event signal treated as firing rate, rather than raw fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. Per-day traces are cast to float32 for validation, invalid neuron rows are removed, residual non-finite values are replaced with zero, and trial slices are stored as uint8. No smoothing, dF/F calculation, temporal pooling, or rate-map computation is applied.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
trace_day = np.nan_to_num(trace_day, nan=0.0, posinf=0.0, neginf=0.0)
...
neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
```

iii. The agent initially retained float traces, then changed to uint8 because they represent binary events and the sample pickle was very large. It intentionally preserved native framewise data rather than applying the paper decoder's optional smoothing/three-frame pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is retained in a session only if every value in its entire day trace is finite. All rows containing any NaN or infinity are removed. The number removed is recorded in metadata.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = trace_day[valid_neurons]
...
'n_neurons_dropped_nonfinite': int((~valid_neurons).sum()),
```

iii. Sample verification exposed NaN/Inf values from unregistered cells. The agent interpreted these as invalid per-session neuron rows and dropped them. It did not restrict data to statistically identified place cells because that curation was not required for this decoder conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Trial zero begins at session start, and subsequent trials are consecutive one-minute windows. Metadata describes that artificial alignment.

ii.
```python
'temporal_alignment_event': 'Continuous session split into contiguous 1-minute windows from session start.',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The agent noted that the recordings are continuous exploration sessions with no stimulus event, so session-start windowing is the applicable alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data remain at the native 30 Hz resolution, or 33.333 ms per sample. No temporal rebinning is performed.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. Behavior and imaging were acquired together at 30 Hz. Although the paper's within-session decoder sometimes pooled three frames, the agent chose to retain native samples for the downstream decoder format.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the per-day `blocked` field. The environment name in `envs` is used only in session IDs/metadata, not as a decoder-input dimension.

ii.
```python
blocked = rec['blocked']
...
blocked_vec = blocked_to_vec(blocked[day])
```

iii. The agent mapped blocked arena partitions to geometry context because the task describes geometry as which portions of the arena are blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Nested blocked indices are flattened, NaNs ignored, and valid indices 0–8 set corresponding entries of a nine-element float32 multi-hot vector. A sole `-1` produces an all-zero vector. The same static vector is copied into every trial of that session.

ii.
```python
vec = np.zeros(9, dtype=np.float32)
...
if len(vals) == 1 and vals[0] == -1:
    return vec
for v in vals:
    if 0 <= v <= 8:
        vec[v] = 1.0
...
input_trials.append(blocked_vec.copy())
```

iii. The notes say blocked geometry is session-constant and that `-1` means no blocked partition. A nine-dimensional encoding independently represents the arena's 3×3 partitions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the per-day `position` field, containing aligned x/y coordinates.

ii.
```python
position = np.asarray(rec['position'])
pos_day = np.asarray(position[day], dtype=np.float32)
```

iii. The agent identified `position` as frame-aligned DeepLabCut-derived mouse coordinates in the 75 cm square arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Non-finite coordinates are first replaced with zero. Coordinates are clipped to the arena, divided into three equal 25 cm intervals on each axis, floored to integer bins, and combined into one time-varying categorical label. Trial outputs have shape `(1, 1800)` and uint8 dtype.

ii.
```python
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
...
xbin = np.floor(np.clip(x, 0, ARENA_SIZE_CM - eps) /
                (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
ybin = np.floor(np.clip(y, 0, ARENA_SIZE_CM - eps) /
                (ARENA_SIZE_CM / N_POS_BINS)).astype(int)
return (ybin * N_POS_BINS + xbin).astype(np.int64)
```

iii. The task requires nine categorical spatial bins. Clipping was used to keep boundary/out-of-range coordinates in valid categories, and uint8 reduced output size.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholds are 25 cm and 50 cm independently for x and y. The final category is `3 * y_bin + x_bin`, yielding labels 0–8; 75 cm is clipped just below the upper edge.

ii.
```python
xbin = np.floor(np.clip(x, 0, 75.0 - 1e-6) / 25.0).astype(int)
ybin = np.floor(np.clip(y, 0, 75.0 - 1e-6) / 25.0).astype(int)
return (ybin * 3 + xbin).astype(np.int64)
```

iii. This implements the requested equal 3×3 subdivision of the 75×75 cm arena and matches the documented class ordering.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams are truncated to their shared minimum frame count and then sliced with exactly the same 1,800-frame boundaries. Assertions verify equal time lengths.

ii.
```python
n_frames = min(trace_day.shape[1], pos_day.shape[1])
trace_day = trace_day[:, :used]
pos_day = pos_day[:, :used]
...
assert data['neural'][s][tr].shape[1] == data['output'][s][tr].shape[1]
```

iii. The agent found the two streams were natively synchronized at 30 Hz and preserved frame-for-frame correspondence during trimming and trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural rows containing any non-finite value are dropped; remaining non-finite neural values are defensively zero-filled. All non-finite positions are replaced by zero, which maps them to bin 0. NaNs in blocked metadata are ignored, invalid blocked indices are ignored, mismatched stream tails are truncated to the shorter stream, and incomplete final trials are discarded.

ii.
```python
valid_neurons = np.all(np.isfinite(trace_day), axis=1)
trace_day = np.nan_to_num(trace_day[valid_neurons], nan=0.0, posinf=0.0, neginf=0.0)
pos_day = np.nan_to_num(pos_day, nan=0.0, posinf=0.0, neginf=0.0)
n_frames = min(trace_day.shape[1], pos_day.shape[1])
n_trials = n_frames // FRAMES_PER_TRIAL
```

iii. These safeguards were motivated by a failed sample verification containing NaN/Inf neural values. The agent documented dropping unregistered cells and trimming remainders as practical format-preserving fixes.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large per-animal joblib records, materializing/casting full trace and position arrays, building thousands of trial arrays, and serializing the multi-gigabyte pickle dominate. Optional matplotlib diagnostic plots and full decoder validation/training add further time outside the core conversion.

ii.
```python
rec = load_animal(data_dir, animal)
trace_day = np.asarray(trace[day], dtype=np.float32)
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The agent explicitly observed a 12 GB sample before dtype optimization and a 4.7 GB full artifact, leading it to store binary neural/output arrays as uint8. Its notes identify large-array I/O and verification/training as the practical bottlenecks.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial slicing is implemented as a Python loop and could be reshaped/split vectorially. Nested blocked-value flattening and per-session assertion loops could also be vectorized or simplified, though they are minor compared with data I/O.

ii.
```python
for i in range(n_trials):
    s = i * FRAMES_PER_TRIAL
    e = (i + 1) * FRAMES_PER_TRIAL
    neural_trials.append(trace_day[:, s:e].astype(np.uint8, copy=False))
    input_trials.append(blocked_vec.copy())
    output_trials.append(pos_bins[s:e][None, :].astype(np.uint8, copy=False))
```

iii. The agent did not explicitly justify these loops. They make the required nested list structure directly and are unlikely to dominate joblib/pickle I/O, but reshaping first could reduce Python overhead.

## 6-c. What processing does the code repeat multiple times?

i. It repeatedly casts/slices arrays for every trial, copies the same blocked vector for every trial, calls `flatten_env_name` twice per session, and—when plotting—recomputes position bins already computed during trial splitting. It also performs a second nested traversal for assertions.

ii.
```python
session_id = f'{animal}_day{day:02d}_{flatten_env_name(envs[day])}'
...
'environment': flatten_env_name(envs[day]),
...
pos_bins = position_to_bins_3x3(pos_day[:, :used])
```

iii. No explicit rationale was documented for these repetitions. Most support metadata, independent trial objects, validation, or optional diagnostics rather than scientific processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional processing plots compute occupancy histograms and render trajectories/neural images that are not stored in the converted dataset. Detailed `session_info`, runtime metadata, environment-name normalization, and verbose diagnostics are not needed by decoder training. The conversion also casts full traces to float32 before later casting retained trial data to uint8.

ii.
```python
trace_day = np.asarray(trace[day], dtype=np.float32)
...
if show_processing and plotted < 2:
    pos_bins = position_to_bins_3x3(pos_day[:, :used])
    make_processing_plot(...)
```

iii. The plotting path exists only for requested visual sanity checks. The agent used metadata and diagnostics to document validation; they are useful for auditing but discarded by downstream model fitting.
