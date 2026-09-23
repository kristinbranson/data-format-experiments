# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data one animal at a time from seven hard-coded per-animal joblib files in `/app/data`, then iterates through every day/session in each loaded animal. Trials are not loaded directly from disk; they are created later by splitting each session in memory.

ii. ```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
for subj,animal in enumerate(selected):
    t0=time.time(); root=joblib.load('/app/data/'+animal); dat=root[animal]
    print(f'Loaded {animal} in {time.time()-t0:.1f}s ({len(dat["trace"])} sessions)',flush=True)
    for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
        ...
```

iii. In `CONVERSION_NOTES.md` Step 1/Step 10, the agent justifies this by saying the reference repository's `load_dat` uses the joblib representation by default, the joblib files are the preferred native representation in `/app/data`, and loading one animal at a time reduces memory pressure.

## 1-b. How are the data split into subjects?

i. The agent treats each entry in the hard-coded `ANIMALS` list as one mouse/subject. The subject list in the output is exactly that list, and `subject_idx` is built from the loop index over that list.

ii. ```python
ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
...
selected=ANIMALS[:2] if sample else ANIMALS
...
for subj,animal in enumerate(selected):
    ...
    neural.append(nt); inputs.append(it); outputs.append(ot); subject_idx.append(subj)
...
data={
    ...
    'subjects':selected,'subject_idx':np.asarray(subject_idx,dtype=np.int64),
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent states that one subject corresponds to one animal file and that the seven subject IDs should stay in the paper/repository order.

## 1-c. How are the data split into sessions?

i. The agent defines each animal-day as one session. It iterates over synchronized per-day entries from `trace`, `position`, `blocked`, and `envs`, and each loop iteration becomes one output session.

ii. ```python
for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
    ts=time.time()
    nt,it,ot,ncells,nfixed,tail=process_session(tr,pos,blk,animal,day)
    neural.append(nt); inputs.append(it); outputs.append(ot); subject_idx.append(subj)
```

iii. In Step 5, the agent explicitly says “each animal-day is one session,” matching the paper's use of day-level recording sessions.

## 1-d. How are the data split into trials?

i. The agent splits each continuous session into consecutive non-overlapping 60-second windows. At 30 Hz this is 1,800 source frames per trial; after the agent's 3-frame pooling, each trial contains 600 bins. Any incomplete tail at the end of a session is discarded.

ii. ```python
FPS = 30
POOL = 3
TRIAL_FRAMES = 60 * FPS
TRIAL_BINS = TRIAL_FRAMES // POOL
...
ntrials = trace.shape[1] // TRIAL_FRAMES
nframes = ntrials * TRIAL_FRAMES
...
neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
input_trials = [bmask.copy() for _ in range(ntrials)]
output_trials = [np.ascontiguousarray(labels[None, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. In Step 5, the agent justifies this as following the task instruction to create 1-minute trials from continuous sessions while keeping trial length fixed across all sessions.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply behavioral or statistical trial-quality filtering. It only enforces structural constraints: sessions must contain at least two complete 60-second trials, and any incomplete final remainder is dropped.

ii. ```python
ntrials = trace.shape[1] // TRIAL_FRAMES
nframes = ntrials * TRIAL_FRAMES
if ntrials < 2:
    raise ValueError(f'{animal} day {day}: fewer than two complete trials')
...
return neural_trials, input_trials, output_trials, int(finite_all.sum()), nfixed, trace.shape[1]-nframes
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent argues that the paper's speed filter is analysis-specific and should not be used here because the target format requires continuous regular one-minute trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from the raw `trace` variable for each day/session in the joblib animal structure.

ii. ```python
for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
    ...
    nt,it,ot,ncells,nfixed,tail=process_session(tr,pos,blk,animal,day)
```

iii. In Step 1 and Step 5, the agent says `trace` is already the released binary rise-extracted calcium-event representation and therefore should be used directly as the source neural stream.

## 2-b. How is the `neural` data processed?

i. The agent removes non-finite rows, checks that finite values are binary 0/1, applies Gaussian smoothing with `sigma=3` frames, then mean-pools non-overlapping groups of 3 frames to produce 100 ms neural bins. The result is stored as `float32`.

ii. ```python
finite_all = np.all(np.isfinite(trace), axis=1)
finite_any = np.any(np.isfinite(trace), axis=1)
if np.any(finite_any != finite_all):
    raise ValueError(f'{animal} day {day}: partially missing neural row')
raw = trace[finite_all]
vals = np.unique(raw)
if not np.all(np.isin(vals, [0, 1])):
    raise ValueError(f'{animal} day {day}: nonbinary finite trace values {vals[:10]}')
...
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
pooled = pooled.astype(np.float32)
```

iii. The agent's Step 5 notes justify this as matching the paper's within-session decoder preprocessing: “Gaussian smoothing sigma=3 source frames followed by non-overlapping 3-frame mean pooling.” It treats that decoder preprocessing as the reference processing most applicable to the new decoder task.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only neuron rows that are finite at every timepoint in that session and rejects any session containing a partially missing neural row. In practice this removes all-NaN unregistered rows and keeps fully recorded rows.

ii. ```python
finite_all = np.all(np.isfinite(trace), axis=1)
finite_any = np.any(np.isfinite(trace), axis=1)
if np.any(finite_any != finite_all):
    raise ValueError(f'{animal} day {day}: partially missing neural row')
raw = trace[finite_all]
```

iii. In Step 2 and Step 5, the agent says absent registrations appear as whole-row NaNs, and that all finite cells should be retained because the paper included all cells unless an analysis explicitly introduced another filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent does not align neural activity to an external task event. Instead, it defines the alignment event as the start of each consecutive non-overlapping 60-second segment and slices trials from those segment boundaries.

ii. ```python
'metadata':{
    ...
    'temporal_alignment_event':'start of each consecutive non-overlapping 60-second segment within a recording session',
    'off_start':0.0,'off_end':60.0,
    ...
}
```

iii. In Step 5, the agent justifies this by noting that the recordings are continuous and have no natural trial onset; the trial starts are artificial segmentation points introduced by the task specification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. Yes: the agent rebins the 30 Hz source data by Gaussian-smoothing and then mean-pooling non-overlapping 3-frame windows.

ii. ```python
FPS = 30
POOL = 3
BIN_MS = 100.0
...
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
...
'metadata':{
    'time_bin_size':BIN_MS,
    ...
}
```

iii. The justification in Step 3/Step 5 is that the paper's position decoder uses 3-frame temporal binning at 30 Hz, so the agent adopted 100 ms bins as the reference temporal resolution most relevant to decoder training.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment-geometry input is derived from the raw `blocked` variable for each session.

ii. ```python
def blocked_vector(blocked):
    ...
    idx = np.asarray(blocked).reshape(-1).astype(int)
    idx = idx[idx >= 0]
    ...
    out[idx] = 1
    return out
...
for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
    ...
```

iii. In Step 1 and Step 5, the agent says the source `blocked` field directly stores omitted 3x3 arena partitions in row-major order, with `-1` meaning nothing is blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The agent converts the blocked partition indices into a 9-element row-major binary vector where `1` means blocked and `0` means accessible. The same static vector is copied to every trial in that session.

ii. ```python
def blocked_vector(blocked):
    """Return row-major 3x3 mask (1 blocked, 0 accessible)."""
    out = np.zeros(9, dtype=np.float32)
    idx = np.asarray(blocked).reshape(-1).astype(int)
    idx = idx[idx >= 0]
    if len(idx):
        if np.any(idx > 8):
            raise ValueError(f'Invalid blocked indices: {idx}')
        out[idx] = 1
    return out
...
bmask = blocked_vector(blocked)
...
input_trials = [bmask.copy() for _ in range(ntrials)]
```

iii. In Step 5, the agent justifies this as the most direct geometry representation for the decoder because geometry is constant within a session and the paper/code already define blocked partitions on the native 3x3 arena layout.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output mouse-position labels are derived from the raw per-frame `position` variable.

ii. ```python
for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
    ...
    nt,it,ot,ncells,nfixed,tail=process_session(tr,pos,blk,animal,day)
```

iii. In Step 1 and Step 5, the agent identifies `position` as the aligned framewise x-y behavioral stream recorded alongside the neural traces.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The agent truncates position to complete trials, mean-pools x and y over the same 3-frame windows used for neural data, discretizes pooled coordinates into 25 cm bins, converts them to row-major class labels, and then snaps any labels that fall into blocked cells to the nearest accessible cell center.

ii. ```python
def pool_position(position, nframes):
    p = np.asarray(position, dtype=np.float64)[:, :nframes]
    ...
    return p.reshape(2, -1, POOL).mean(axis=2)


def position_labels(xy, blocked_mask):
    col = np.clip(np.floor(xy[0] / 25.0), 0, 2).astype(np.int64)
    row = np.clip(np.floor(xy[1] / 25.0), 0, 2).astype(np.int64)
    labels = row * 3 + col
    invalid = blocked_mask[labels].astype(bool)
    ...
    labels[invalid] = open_labels[nearest]
    return labels.astype(np.uint8), nfixed
```

iii. In Step 5 and Step 10, the agent justifies this by saying the target output must be a 3x3 categorical position, that temporal pooling should match neural pooling, and that rare pooled blocked labels are tracking/interpolation artifacts that should be reassigned to the nearest accessible bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The agent thresholds x and y independently using 25 cm boundaries across the 75 cm arena, clips each axis to indices `0..2`, and combines them as `row * 3 + col`.

ii. ```python
col = np.clip(np.floor(xy[0] / 25.0), 0, 2).astype(np.int64)
row = np.clip(np.floor(xy[1] / 25.0), 0, 2).astype(np.int64)
labels = row * 3 + col
```

iii. In Step 5, the agent justifies this as using the arena's natural 3x3 partition structure required by the task and matching the row-major indexing used by the blocked geometry masks.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The agent aligns output position with neural data by using the same source frame count, the same 3-frame pooling groups, and the same 60-second trial boundaries for both streams.

ii. ```python
if trace.shape[1] != position.shape[1]:
    raise ValueError(f'{animal} day {day}: neural-position length mismatch')
...
smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
xy = pool_position(position, nframes)
...
neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
output_trials = [np.ascontiguousarray(labels[None, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. In Steps 2, 5, 10, and 12, the agent repeatedly justifies this by saying the source streams are frame-aligned in the release and therefore should be pooled and sliced using identical indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing/unexpected data conservatively: whole-row missing neurons are removed, partially missing neural rows cause an error, non-finite position values cause an error, length mismatches cause an error, non-binary neural values cause an error, and incomplete trailing frames are discarded rather than padded.

ii. ```python
if trace.ndim != 2 or position.ndim != 2 or position.shape[0] != 2:
    raise ValueError(f'{animal} day {day}: bad trace/position dimensions')
if trace.shape[1] != position.shape[1]:
    raise ValueError(f'{animal} day {day}: neural-position length mismatch')
finite_all = np.all(np.isfinite(trace), axis=1)
finite_any = np.any(np.isfinite(trace), axis=1)
if np.any(finite_any != finite_all):
    raise ValueError(f'{animal} day {day}: partially missing neural row')
...
if not np.all(np.isin(vals, [0, 1])):
    raise ValueError(f'{animal} day {day}: nonbinary finite trace values {vals[:10]}')
...
return neural_trials, input_trials, output_trials, int(finite_all.sum()), nfixed, trace.shape[1]-nframes
```

iii. The agent's notes justify this by saying the released data already uses all-NaN rows for absent cells, positions are expected to be finite, and it is safer to fail loudly on unexpected corruption while only discarding the incomplete final tail that cannot form a full trial.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading each large per-animal joblib file, smoothing/pooling the full-session neural arrays, and serializing the large nested float32 output dataset to pickle.

ii. ```python
for subj,animal in enumerate(selected):
    t0=time.time(); root=joblib.load('/app/data/'+animal); dat=root[animal]
    print(f'Loaded {animal} in {time.time()-t0:.1f}s ({len(dat["trace"])} sessions)',flush=True)
    for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
        ...
        smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
        pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
        ...
with open(outfile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Step 6, the agent explicitly discusses full-dataset size, per-animal loading, vectorized whole-session processing, and conversion runtime, indicating these are the major cost centers.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy numeric work is already vectorized, but the code still constructs trial lists with Python-level per-trial comprehensions and repeats Python loops over animals and sessions. The per-trial slicing/copying of `neural_trials`, `input_trials`, and `output_trials` is the clearest remaining non-vectorized part.

ii. ```python
for subj,animal in enumerate(selected):
    ...
    for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
        ...
        neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
        input_trials = [bmask.copy() for _ in range(ntrials)]
        output_trials = [np.ascontiguousarray(labels[None, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
```

iii. The agent does not call out these exact loops in its notes, but Step 6 says it intentionally vectorized filtering, pooling, and discretization and left the nested-list trial construction required by the target format.

## 6-c. What processing does the code repeat multiple times?

i. The code recomputes `blocked_vector(blk)` for the same session more than once, copies the same static geometry vector into every trial of a session, and repeatedly materializes per-trial array slices after already processing whole sessions.

ii. ```python
bmask = blocked_vector(blocked)
...
input_trials = [bmask.copy() for _ in range(ntrials)]
...
session_info.append({
    ...
    'blocked_indices':np.flatnonzero(blocked_vector(blk)).tolist(),
    ...
})
...
if show_processing and plot_count < 2:
    plot_processing(..., blocked_vector(blk))
```

iii. The notes only partially justify this. Step 6 says the target schema requires repeated per-trial copies and nested lists, but it does not justify the repeated recomputation of the blocked mask within a session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs substantial validation and bookkeeping that are not needed by downstream decoders: checking value domains with `np.unique`, counting corrected blocked bins, recording `session_info`, timing/logging every session, and supporting optional plotting. It also computes `env_name` only for metadata/logging.

ii. ```python
vals = np.unique(raw)
if not np.all(np.isin(vals, [0, 1])):
    raise ValueError(...)
...
labels, nfixed = position_labels(xy, bmask)
...
env_name=str(np.asarray(env).reshape(-1)[0])
session_info.append({... 'environment':env_name, ... 'corrected_blocked_bins':nfixed})
...
if show_processing and plot_count < 2:
    plot_processing(...)
print(f'  day {day:02d} {env_name:10s}: cells={ncells:3d} trials={len(nt):2d} tail={tail:4d} fixed={nfixed:2d} ({time.time()-ts:.2f}s)',flush=True)
```

iii. The agent partly justifies this in the notes as sanity checking and documentation support, but these steps do not contribute to the final neural/input/output arrays consumed by downstream analyses.
