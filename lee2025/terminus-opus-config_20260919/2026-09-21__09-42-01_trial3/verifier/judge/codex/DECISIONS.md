# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, opens one MATLAB v7.3 / HDF5 file per animal with `h5py`, reads session metadata (`envs`, `blocked`) for all days, and then lazily reads each day/session’s `trace` and `position` arrays. Trials are not loaded from disk directly; they are created later from each processed session.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def open_animal(animal):
    return h5py.File(os.path.join(DATA_DIR, f'{animal}.mat'), 'r')

def read_meta(f):
    envs = [''.join(chr(c) for c in f[r][:].ravel()) for r in f['envs'][0]]
    blocked = [np.atleast_1d(f[r][:].ravel()).astype(int) for r in f['blocked'][0]]
    return envs, blocked

def read_session(f, day):
    trace = f[f['trace'][day, 0]][:]
    position = f[f['position'][day, 0]][:]
    return trace, position
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose lazy HDF5 reads because the joblib files load a whole animal into memory, while the HDF5 route reads one session at a time and keeps peak memory lower.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats each hard-coded animal ID / `.mat` file as one subject. `subjects` is initialized as the full `ANIMALS` list, and each session stores `subject_idx = ANIMALS.index(animal)`.

ii.
```python
data = {
    'subjects': list(ANIMALS), 'subject_idx': [],
    ...
}

for ai, animal in enumerate(animals):
    subj_idx = ANIMALS.index(animal)
    ...
    data['subject_idx'].append(subj_idx)
```

iii. The notes describe the dataset as seven per-animal files and explicitly list the same seven mice as the subject set.

## 1-c. How are the data split into sessions?

i. Each recording day in an animal file is treated as one session. The AI loops over `range(n_days)`, reads that day’s `trace` and `position`, processes them, and appends one session entry to the output lists.

ii.
```python
envs, blocked = read_meta(f)
n_days = len(envs)
...
days = SAMPLE_SESSIONS[animal] if sample else range(n_days)
for day in days:
    trace, position = read_session(f, day)
    ...
    data['neural'].append(ntr)
    data['input'].append(inp)
    data['output'].append(outp)
```

iii. The AI’s notes say the per-animal data contain one entry per day/session and report 207 total sessions across the seven mice.

## 1-d. How are the data split into trials?

i. The AI creates trials by cutting each processed session into consecutive 60 s blocks after converting the data to 100 ms bins. Because it first removes non-locomotion bins within each block, resulting trial lengths are variable. It also keeps a final partial block if it is at least 50% of a full trial.

ii.
```python
TRIAL_SEC = 60.0
MIN_TRIAL_FRAC = 0.5
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN_FRAMES))

start = 0
while start < n_bins:
    stop = min(start + BINS_PER_TRIAL, n_bins)
    if (stop - start) < MIN_TRIAL_FRAC * BINS_PER_TRIAL:
        break
    sel = np.where(moving_binned[start:stop])[0] + start
    if sel.size >= MIN_BINS_PER_TRIAL:
        neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
        input_trials.append(input_vec.copy())
        output_trials.append(out_class[sel][None, :].copy())
    start = stop
```

iii. In the notes, the AI says trials are consecutive 60 s blocks because the decoder task requested 1-minute trials, and says it keeps the final partial block if it is at least 30 s long to avoid discarding the ~55 s remainder.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial/session filters. Within each 60 s block it keeps only bins where the smoothed speed exceeds 5 cm/s, discards trial blocks with fewer than 30 retained bins, discards a session entirely if fewer than two usable trials remain, and drops a final remainder block if it is shorter than half a trial.

ii.
```python
MIN_TRIAL_FRAC = 0.5
MIN_BINS_PER_TRIAL = 30
...
moving_binned = speed_binned > V_THRESH
...
if (stop - start) < MIN_TRIAL_FRAC * BINS_PER_TRIAL:
    break
sel = np.where(moving_binned[start:stop])[0] + start
if sel.size >= MIN_BINS_PER_TRIAL:
    ...
...
if len(ntr) < 2:
    print(f'  SKIP {sid}: only {len(ntr)} usable trials', flush=True)
    continue
```

iii. The AI justifies this in the notes by appealing to the paper’s within-session decoder, which excludes low-speed periods, and by the target-format requirement that each session have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived from the raw session `trace` array only.

ii.
```python
def read_session(f, day):
    trace = f[f['trace'][day, 0]][:]
    position = f[f['position'][day, 0]][:]
    return trace, position
```

iii. The notes identify `trace` as the paper’s binarized rising-phase calcium-event signal and state that no dF/F computation is needed because the authors already provided the processed event trace.

## 2-b. How is the `neural` data processed?

i. The AI computes speed, keeps only selected cells, smooths each kept cell’s `trace` over time with `gaussian_filter1d(sigma=3)`, average-pools over non-overlapping 3-frame windows (100 ms bins), casts to `float32`, and multiplies by `FPS` to express the pooled values as events/s. Trials are then stored as `(neurons, time)` arrays.

ii.
```python
tr = trace[:, keep_cells]
...
tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SMOOTH_SIGMA, axis=0)
neural = pool_mean(tr_smooth).astype(np.float32) * FPS
...
neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
```

iii. The AI’s justification, stated both in the file header and notes, is that this reproduces the paper code’s decoder preprocessing (`fit_decoder` / `test_decoder`): Gaussian smoothing followed by `AvgPool1d(kernel=3, stride=3)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells that are registered on that day and have more than 5 events during locomotion periods. Unregistered cells are detected from NaNs; low-event cells are filtered by event count during movement.

ii.
```python
speed = compute_speed(position)
moving = speed > V_THRESH

registered = ~np.isnan(trace[0, :])
events_moving = np.nansum(trace[moving, :], axis=0)
keep_cells = registered & (events_moving > CELL_THRESHOLD)

tr = trace[:, keep_cells]
if tr.size == 0:
    return [], [], [], keep_cells
```

iii. The notes explicitly cite `decode_position_within` from the reference code and its `cell_threshold=5` rule as the basis for this curation choice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align neural activity to any task event. It treats the experiment as continuous free exploration, bins the continuous trace into 100 ms bins, and then slices the session into consecutive 60 s trial windows referenced to session time.

ii.
```python
'temporal_alignment_event':
    'Start of each 60 s trial, measured from the start of the recording '
    'session (free exploration; the task has no discrete events). ...',
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. The notes say the task has no discrete events and therefore uses trial start within the continuous recording as the alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. The AI rebins the 30 Hz source data by averaging every 3 frames into one bin; no overlapping windows are used.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS

def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    n = (x.shape[0] // k) * k
    x = x[:n]
    return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)
```

iii. The AI’s stated rationale is to mirror the reference decoder’s temporal preprocessing rather than keep the original 30 Hz frame rate.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives the decoder input primarily from the raw `envs` variable, not directly from `blocked`. It uses `envs[day]` to look up one of 10 hard-coded 3x3 geometry templates, then converts that template into a blocked-partition vector. It does, however, cross-check that this derived vector matches the raw `blocked` field for every session.

ii.
```python
def read_meta(f):
    envs = [''.join(chr(c) for c in f[r][:].ravel()) for r in f['envs'][0]]
    blocked = [np.atleast_1d(f[r][:].ravel()).astype(int) for r in f['blocked'][0]]
    return envs, blocked
...
expected = np.where(env_blocked_vector(envs[day]))[0]
got = np.sort(blocked[day][blocked[day] >= 0])
assert np.array_equal(expected, got)
```

iii. The notes justify this by saying the geometry can be reconstructed from `utils.get_env_mat` and that the reconstruction was verified against the dataset’s `blocked` field across all 207 sessions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI maps each environment name to a hard-coded 3x3 open/blocked matrix, flips it vertically, flattens it, and stores a 9-element float vector where `1` means blocked. The same static vector is copied into every trial for that session.

ii.
```python
ENV_MATS = {
    'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ...
}

def env_blocked_vector(env_name):
    m = np.array(ENV_MATS[env_name], dtype=float)
    return (np.flipud(m).ravel() == 0).astype(np.float32)
...
input_vec = env_blocked_vector(env_name)
...
input_trials.append(input_vec.copy())
```

iii. The AI’s notes say this reproduces the orientation convention implied by `utils.clean_rate_maps` and matches the session-level `blocked` annotations.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw session `position` array only.

ii.
```python
def read_session(f, day):
    trace = f[f['trace'][day, 0]][:]
    position = f[f['position'][day, 0]][:]
    return trace, position
```

iii. The notes describe `position` as DeepLabCut-tracked `x,y` coordinates in centimeters over the 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first averages the continuous `position` signal into the same 100 ms bins used for neural data, then discretizes each binned `x,y` coordinate into a 3x3 arena partition. If a binned sample falls inside a geometrically blocked partition, the code snaps it to the nearest open partition.

ii.
```python
pos_binned = pool_mean(position)
...
def position_to_class(position, blocked_vec=None):
    part = ARENA_SIZE / N_PART
    bins = np.clip((position / part).astype(int), 0, N_PART - 1)
    cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
    if blocked_vec is not None and blocked_vec.any():
        blocked_ids = np.where(blocked_vec > 0)[0]
        bad = np.isin(cls, blocked_ids)
        if bad.any():
            open_ids = np.where(blocked_vec == 0)[0]
            centres = np.stack([(open_ids % N_PART) * part + part / 2,
                                (open_ids // N_PART) * part + part / 2], axis=1)
            d = np.linalg.norm(position[bad][:, None, :] - centres[None], axis=2)
            cls[bad] = open_ids[np.argmin(d, axis=1)]
    return cls
```

iii. The header comment and metadata say the snapping step is meant to mimic the reference decoder’s cleaning of out-of-geometry positions. The notes are inconsistent here: one part says blocked-partition samples are kept as-is, but the code and metadata clearly implement snapping.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI divides each axis of the 75 cm arena into three equal 25 cm bins, clips the bin indices to `[0, 2]`, and computes the categorical output class as `3 * ybin + xbin`, yielding classes `0` through `8`.

ii.
```python
part = ARENA_SIZE / N_PART
bins = np.clip((position / part).astype(int), 0, N_PART - 1)
cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
```

iii. The notes justify this as matching the dataset’s blocked-partition indexing and the decoder task’s required 3x3 spatial categorization.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output and neural data by putting both streams onto the same 100 ms grid, then using the same retained locomotion-bin indices `sel` for both signals inside each trial block.

ii.
```python
neural = pool_mean(tr_smooth).astype(np.float32) * FPS
pos_binned = pool_mean(position)
speed_binned = pool_mean(speed[:, None])[:, 0]
...
sel = np.where(moving_binned[start:stop])[0] + start
if sel.size >= MIN_BINS_PER_TRIAL:
    neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
    output_trials.append(out_class[sel][None, :].copy())
```

iii. The notes say behavior and calcium were acquired together at 30 Hz and that pooling/slicing them identically preserves their alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues: unregistered cells are removed via NaN checks; sessions with no kept cells return no trials; low-speed bins are discarded; short remainder blocks are dropped; blocks with too few retained bins are dropped; sessions with fewer than two usable trials are skipped; and rare position samples inside blocked partitions are reassigned to the nearest open partition.

ii.
```python
registered = ~np.isnan(trace[0, :])
...
if tr.size == 0:
    return [], [], [], keep_cells
...
if (stop - start) < MIN_TRIAL_FRAC * BINS_PER_TRIAL:
    break
...
if sel.size >= MIN_BINS_PER_TRIAL:
    ...
...
if len(ntr) < 2:
    ...
    continue
...
cls[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. The AI’s notes justify the NaN handling from the dataset structure, the movement-related filters from the paper’s decoder code, and the blocked-partition reassignment as a way to correct apparent tracking noise near walls.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies data loading as a major cost and says the main expensive operations are per-session HDF5 reads plus session processing over the large `trace` arrays. It explicitly contrasts this with the even more expensive joblib-based whole-animal loading approach.

ii.
```python
f = open_animal(animal)
...
trace, position = read_session(f, day)
...
timings.append((t_read, t_proc))
print(f'  {sid} ... (read {t_read:.2f}s proc {t_proc:.2f}s)', flush=True)
```

iii. In the notes, the AI says total runtime on the full dataset is about 3 minutes and attributes the main cost to HDF5 I/O and per-session processing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s own script is mostly already vectorized. The notes say it intentionally avoided reference-style Python loops over frames for binning and replaced average pooling with a reshape-and-mean implementation. The remaining explicit loop over trials/sessions was not identified as something to vectorize further.

ii.
```python
def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    n = (x.shape[0] // k) * k
    x = x[:n]
    return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)
```

iii. The notes explicitly call out “Python loops over frames for binning” as an inefficiency in the reference-style approach that this script avoided.

## 6-c. What processing does the code repeat multiple times?

i. The script repeats the same per-session sequence for every day: read metadata, verify the geometry against `blocked`, compute speed, apply cell filtering, smooth/pool, discretize position, and cut into trials. No major duplicated helper computations beyond this session-wise repetition are singled out in the notes.

ii.
```python
for day in days:
    trace, position = read_session(f, day)
    expected = np.where(env_blocked_vector(envs[day]))[0]
    got = np.sort(blocked[day][blocked[day] >= 0])
    assert np.array_equal(expected, got)
    ntr, inp, outp, keep = process_session(trace, position, envs[day],
                                           diagnostics=diag)
```

iii. The notes mainly discuss repeated work in the rejected joblib-based approach; they do not highlight any major unnecessary repetition remaining in the final script.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the default conversion path, very little processing is performed solely to be discarded. The clearest extra work is the per-session `expected` versus `blocked` consistency assertion and the optional diagnostics/plotting path used only for `--show-processing`. The metadata and `session_info` bookkeeping are also not used by downstream decoding.

ii.
```python
expected = np.where(env_blocked_vector(envs[day]))[0]
got = np.sort(blocked[day][blocked[day] >= 0])
assert np.array_equal(expected, got)
...
diag = {} if (args.show_processing and n_plotted < 2) else None
...
if diag is not None:
    plot_processing(diag, sid, envs[day])
```

iii. The notes frame these as sanity checks and visualization aids rather than as required parts of the saved dataset.
