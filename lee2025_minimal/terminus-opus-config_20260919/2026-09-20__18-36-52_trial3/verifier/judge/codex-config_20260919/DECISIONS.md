# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script glob-sorts every `.mat` file in the selected data directory, opens each MATLAB v7.3/HDF5 file with `h5py`, and dereferences its `trace`, `position`, and `blocked` datasets. Full mode processes every discovered file and recording session; sample mode is optional.

ii.
```python
mat_files = sorted(glob.glob(f'{args.datadir}/*.mat'))
...
f = h5py.File(filepath, 'r')
trace_refs = f['trace']
pos_refs = f['position']
blk_refs = f['blocked'][:][0]
```

iii. The trajectory initially recognized both `.mat` and extensionless joblib files, but does not justify the delivered choice of HDF5 over the joblib source used by the paper code. In fact, its final rationale describes loading joblib data and processing that is absent from the delivered script, so that rationale is inconsistent with this code.

## 1-b. How are the data split into subjects?

i. Each sorted `.mat` file is one mouse. Its basename without `.mat` is the subject ID, and every session read from that file gets the file-loop index in `subject_idx`.

ii.
```python
for subj_i, mat_file in enumerate(mat_files):
    name = mat_file.split('/')[-1].replace('.mat', '')
    subjects.append(name)
...
    subject_idx_list.append(subj_i)
```

iii. The data listing showed one `.mat` file per named mouse. The trajectory’s final summary also expected seven mice, although it describes a different loader.

## 1-c. How are the data split into sessions?

i. Each row of the HDF5 reference arrays is treated as one recording session. The script dereferences the trace, position, and blocked entry at the same index and appends one output session.

ii.
```python
n_recording_sessions = trace_refs.shape[0]
for i in range(n_recording_sessions):
    trace = f[trace_refs[i][0]][:]
    position = f[pos_refs[i][0]][:].T
    blk_indices = f[blk_refs[i]][:].flatten()
```

iii. The agent treated the parallel reference-array entries as session/day records. Its trajectory ultimately reported 207 sessions, consistent with the paper’s session count, though for a different version of its script.

## 1-d. How are the data split into trials?

i. Continuous sessions are cut into consecutive, non-overlapping 60-second trials. At 30 Hz this is 1,800 frames; any incomplete tail is dropped.

ii.
```python
TRIAL_LENGTH = SAMPLING_RATE * TRIAL_DURATION_SEC
...
n_trials = data.shape[1] // trial_length
return [data[:, i * trial_length:(i + 1) * trial_length]
        for i in range(n_trials)]
```

iii. This directly follows the instruction to make one-minute trials. No event-defined trials exist in these continuous foraging sessions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are not quality-filtered. Only incomplete final chunks are discarded; the script also does not enforce the required minimum of two trials per session.

ii.
```python
n_trials = data.shape[1] // trial_length
...
sessions.append({'neural': neural_trials, ...})
```

iii. The delivered code provides no quality-control justification. The trajectory’s alternative implementation said sessions shorter than two trials were skipped, which the delivered script does not do.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `trace` session dataset.

ii.
```python
trace = f[trace_refs[i][0]][:]  # (timepoints, neurons)
```

iii. The agent identified `trace` as the paper’s preprocessed, binarized rising-phase calcium-transient representation used as neural activity.

## 2-b. How is the `neural` data processed?

i. After cell filtering, traces are cast to `float32` and transposed from time-by-neuron to neuron-by-time. The delivered code performs no Gaussian smoothing or temporal pooling.

ii.
```python
trace = trace[:, active_mask].astype(np.float32).T
```

iii. There is no trajectory justification for omitting further processing. The trajectory instead explicitly claims Gaussian smoothing with sigma 3 frames and 3-frame average pooling, contradicting the delivered code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is removed only when its entire session column is NaN. There is no event-count/near-silent-cell filter, and a partially NaN neuron is retained.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, active_mask].astype(np.float32).T
```

iii. The code comments interpret all-NaN columns as cells not recorded that session. The trajectory additionally claims removal of cells with at most five transients and stricter registration filtering, but neither appears in this script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Frame zero of each consecutive one-minute slice is its alignment point, implicitly measured from session start.

ii.
```python
neural_trials = split_into_trials(trace)
```

iii. The recording is continuous free foraging and the requested trials are artificial one-minute windows, so no stimulus or behavioral event is available for alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output stays at the native 30 Hz sampling rate, approximately 33.33 ms per frame. No temporal rebinning is applied.

ii.
```python
SAMPLING_RATE = 30
TIME_BIN_SIZE = 1000.0 / SAMPLING_RATE
```

iii. The delivered script offers no paper-based justification. The trajectory says the paper-code decoder processing uses 3-frame pooling to 100 ms, again contradicting this implementation.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the session’s `blocked` indices.

ii.
```python
blk_refs = f['blocked'][:][0]
...
blk_indices = f[blk_refs[i]][:].flatten()
```

iii. These indices identify inaccessible partitions in the arena’s 3-by-3 design.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices become a 9-element float32 multi-hot vector, with one meaning blocked. A sole `-1` means no blocked partition. The same static vector object is placed in every trial of the session.

ii.
```python
blocked = np.zeros(n_positions, dtype=np.float32)
if not (len(blk_indices) == 1 and blk_indices[0] == -1):
    blocked[blk_indices.astype(int)] = 1
...
input_trials = [blocked] * len(neural_trials)
```

iii. A nine-dimensional encoding preserves each partition independently and reflects that geometry is constant throughout a session. The trajectory validated the index convention by checking that mice almost never occupied marked-blocked bins.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the session’s two-coordinate `position` dataset.

ii.
```python
position = f[pos_refs[i][0]][:].T  # (2, n_timepoints)
```

iii. The paper and methods describe simultaneously acquired x/y behavior at 30 Hz in a 75-by-75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Each raw position frame is directly digitized into x and y grid bins, combined as `y_bin * 3 + x_bin`, converted to `int8`, and given a singleton output dimension. Position is not temporally averaged first.

ii.
```python
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
return (y_bin * n_grid + x_bin).astype(np.int8)
```

iii. The 3-by-3 output matches the task and arena partition design. The trajectory supports the `y*3+x` convention, but describes averaging positions into 100 ms bins before categorization, absent here.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each 0–75 cm axis is split at 25 and 50 cm. `np.digitize` produces bins 0, 1, or 2 and clipping handles out-of-range/boundary values; the Cartesian bins become classes 0–8.

ii.
```python
edges = np.linspace(0, arena_size, n_grid + 1)[1:-1]
x_bin = np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
y_bin = np.clip(np.digitize(position[1], edges), 0, n_grid - 1)
```

iii. Equal 25 cm bins exactly follow the requested 3-by-3 discretization of the 75 cm arena.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural arrays are assumed frame-synchronous and are independently sliced with identical 1,800-frame boundaries. The code does not explicitly truncate them to a common minimum length or validate equal lengths.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
```

iii. The methods state that behavior and imaging were simultaneously acquired at 30 Hz and timestamp-aligned. The trajectory relied on this correspondence, though its alternative code defensively used the common minimum after rebinning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neuron columns are removed, spatial bins are clipped to 0–2, and incomplete trial tails are dropped. The code has no handling for partial NaNs, neural/position length disagreement, malformed blocked indices, zero-trial sessions, or fewer than two trials.

ii.
```python
active_mask = ~np.all(np.isnan(trace), axis=0)
...
np.clip(np.digitize(position[0], edges), 0, n_grid - 1)
...
n_trials = data.shape[1] // trial_length
```

iii. The comments justify all-NaN removal as excluding unrecorded cells and deliberately drop short tails. Other edge cases are neither discussed nor defended; the trajectory describes safeguards not present in the delivered script.

## 6-a. What are the most time-consuming steps of the code?

i. Reading the large HDF5 trace and position datasets and serializing the full pickle are likely dominant. Full-array NaN scans and the large diagnostic concatenations also cost time and memory.

ii.
```python
trace = f[trace_refs[i][0]][:]
position = f[pos_refs[i][0]][:].T
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent’s documented view was that loading the large neural arrays is I/O-bound and dominates the relatively simple array processing. It did not discuss serialization or diagnostics.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial construction uses Python list comprehensions and the outer file/session traversal is serial. Trial splitting could be represented by reshape/views before conversion to the required list, although the final nested-list format still requires per-trial objects; file/session loops are appropriate for variable-sized data.

ii.
```python
return [data[:, i * trial_length:(i + 1) * trial_length]
        for i in range(n_trials)]
```

iii. The agent supplied no efficiency justification or vectorization analysis.

## 6-c. What processing does the code repeat multiple times?

i. Trial-boundary computation and slicing are repeated separately for neural and output data. Each session also concatenates all output trials solely to compute diagnostic classes, and identical blocked vectors are repeatedly referenced in trial lists.

ii.
```python
neural_trials = split_into_trials(trace)
output_trials = split_into_trials(output)
...
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
```

iii. The trajectory did not identify repeated processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Per-session diagnostic values—unique output classes, trial length, blocked display vector, and formatted status text—are computed only for printing and are not saved or used downstream. The seeded random generator is also never used.

ii.
```python
np.random.seed(sum(ord(c) for c in _lee2025_seed) % 2**31)
...
output_classes = np.unique(np.concatenate([t.flatten() for t in sess['output']]))
blocked_str = sess['input'][0].astype(int) if n_trials > 0 else []
print(f"    session {sess_i}: ...")
```

iii. The agent gave no justification for these discarded diagnostics and characterized unnecessary processing as not applicable.
