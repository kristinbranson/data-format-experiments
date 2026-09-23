# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a fixed list of seven animal `.mat` files from `/app/data`, opens each with `h5py`, and lazily reads one session/day at a time from HDF5 object references. Trials are not stored in the raw files; they are created later inside `process_session`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

with h5py.File(os.path.join(DATA_DIR, f"{animal}.mat"), 'r') as f:
    n_days = f['trace'].shape[0]
    day_list = range(n_days) if days is None else days
    for day in day_list:
        position, trace, env, blocked = read_session(f, day)
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose the HDF5 `.mat` files because they are byte-equivalent to the joblib exports but allow lazy per-day reads and much lower memory use than loading whole animals at once.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats each named animal file as one subject. Subject names come from the hardcoded `ANIMALS` list, and `build_dataset` converts per-session animal labels into `subjects` and `subject_idx`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

subjects = sorted({s['animal'] for s in all_sessions})
'subject_idx': np.array([subjects.index(s['animal']) for s in all_sessions], dtype=np.int64),
```

iii. The notes say these seven IDs came from the reference `main.py` and match the seven mice reported in the paper.

## 1-c. How are the data split into sessions?

i. Each recording day inside an animal file is treated as one session. `process_animal` iterates over all `day` indices, and `read_session` pulls that day's `position`, `trace`, `env`, and `blocked` data.

ii.
```python
def read_session(f, day):
    position = f[f['position'][day, 0]][()].astype(np.float64)
    trace = f[f['trace'][day, 0]][()].astype(np.float32)
    env = _h5_str(f, f['envs'][0, day])
    blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
    blocked = blocked[blocked >= 0].astype(int)
    return position, trace, env, blocked

for day in day_list:
    position, trace, env, blocked = read_session(f, day)
```

iii. In the notes, the AI explicitly states "Session = recording day" and reports 207 total day-sessions across the seven animals.

## 1-d. How are the data split into trials?

i. The AI first rebins each continuous session to 100 ms bins, then defines trials as contiguous 60 s blocks of 600 bins. It also keeps a final partial tail block if it is at least 100 bins (10 s), and within each block it only retains bins marked as running, so saved trial lengths are variable rather than fixed.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / BIN_FRAMES))   # 600 bins of 100 ms
MIN_TAIL_BINS = 100

starts = list(range(0, n_bins - BINS_PER_TRIAL + 1, BINS_PER_TRIAL))
bounds = [(s, s + BINS_PER_TRIAL) for s in starts]
tail_start = len(starts) * BINS_PER_TRIAL
if n_bins - tail_start >= MIN_TAIL_BINS:
    bounds.append((tail_start, n_bins))

for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    ...
    neural_trials.append(np.ascontiguousarray(neural[:, idx]))
```

iii. The notes justify this as imposing the decoder task's 1-minute trials on a continuous 40-minute task, while preserving an extra trailing block when it covers enough recording time.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops any trial block that retains fewer than 30 running bins after motion filtering, and drops any session that ends up with fewer than two usable trials.

ii.
```python
MIN_TRIAL_BINS = 30

for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    if idx.size < MIN_TRIAL_BINS:
        continue
    ...

if len(neural_trials) < 2:
    return None
```

iii. In the notes, the AI says this follows from excluding immobility and from the validator's requirement that each session have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data comes from the raw `trace` field for each day/session.

ii.
```python
def read_session(f, day):
    ...
    trace = f[f['trace'][day, 0]][()].astype(np.float32)
    ...
    return position, trace, env, blocked
```

iii. The notes describe `trace` as the already-binarized calcium-transient rising-phase signal distributed with the dataset.

## 2-b. How is the `neural` data processed?

i. The AI treats `trace` as already-processed neural events, then further processes it by removing unregistered and low-event cells, smoothing with `gaussian_filter1d` (`sigma=3` frames), average-pooling every 3 frames to 100 ms bins, transposing to `(neurons, time)`, and later slicing each trial down to retained running bins.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]

events_running = trace[moving].sum(axis=0)
keep_cell = events_running > CELL_THRESH
trace = trace[:, keep_cell]

smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
```

iii. The notes say no dF/F computation is needed because the dataset already stores binarized transient-rising events, and that the extra smoothing and 100 ms binning were copied from the paper's `fit_decoder` preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neural QC filters: first it removes all-NaN columns, interpreted as cells not registered that day; then it keeps only cells with more than 5 events during running periods.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.any(np.isnan(trace)), "a registered cell has partial NaNs -- unexpected"

events_running = trace[moving].sum(axis=0)
keep_cell = events_running > CELL_THRESH
trace = trace[:, keep_cell]
```

iii. In the notes, the AI says this copies the reference decoder's curation rule: registered that day and `> 5` events while running, with no place-cell filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to any behavioral or stimulus event. Instead, it defines the temporal anchor as the start of each artificial 1-minute trial block within the continuous session, while relying on the source data's native frame alignment between behavior and imaging.

ii.
```python
'temporal_alignment_event':
    'Start of each 1-minute trial block, measured from the start of the (continuous, 40 min) '
    'recording session. The task is continuous free foraging, so there is no stimulus or '
    'behavioural event to align to; imaging and behaviour were acquired on the same DAQ at '
    '30 Hz and are frame-aligned in the source data.',
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. The notes repeatedly state that this experiment has no true trial event and that `position` and `trace` are already timestamp-aligned by the authors' DAQ, so the only imposed alignment is to the start of each synthetic trial block.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. Yes, the AI rebins 30 Hz data by smoothing and then averaging every 3 frames.

ii.
```python
BIN_FRAMES = 3
TIME_BIN_MS = 1000.0 * BIN_FRAMES / FPS  # 100 ms

smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
```

iii. The notes justify this as matching the paper's `fit_decoder(..., temporal_bin_size=3)` preprocessing.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The decoder input is derived from the raw `blocked` field, with `envs` used only as a cross-check that the blocked indices agree with the named geometry.

ii.
```python
env = _h5_str(f, f['envs'][0, day])
blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
blocked = blocked[blocked >= 0].astype(int)

expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected)
```

iii. The notes say the stored `blocked` list is the actual source for the decoder input, and `env` is used to re-derive and verify the same mask via the reference `get_env_mat`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the blocked indices into a 9-element binary vector where `1` means that partition is blocked. The vector is static for the whole session and copied into every retained trial.

ii.
```python
geometry = np.zeros(NBINS * NBINS, dtype=np.float32)
geometry[blocked] = 1.0
...
input_trials.append(geometry.copy())
```

iii. The notes justify this as the decoder-task-specific representation of arena geometry and note that `blocked = -1` becomes an all-zero vector after filtering to an empty blocked list.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the raw `position` field for each session.

ii.
```python
def read_session(f, day):
    position = f[f['position'][day, 0]][()].astype(np.float64)
    ...
    return position, trace, env, blocked
```

iii. The notes describe `position` as 30 Hz DeepLabCut head position in centimeters over the 75 cm by 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first average-pools position over the same 3-frame windows used for neural data, converts each pooled point to a 3x3 arena bin using fixed 25 cm edges, snaps any sample that falls into a blocked partition to the nearest open partition, and finally restricts outputs to the running-bin indices retained for each trial.

ii.
```python
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
pos_bin = position_to_bin(position_binned_cm).astype(np.int64)
n_snapped = int(np.isin(pos_bin, blocked).sum())
pos_bin = snap_to_open_bins(pos_bin, blocked)
...
output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

iii. The notes justify the 25 cm discretization from the 75 cm arena size and justify snapping as a fix for rare tracking or pooling artifacts that otherwise place the animal just inside a wall.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI thresholds continuous position into nine categories by flooring each axis into one of three 25 cm bins, clipping to `[0, 2]`, and combining them as `3 * ybin + xbin`. Samples that initially land in blocked bins are remapped afterward to the nearest open category.

ii.
```python
def position_to_bin(position_cm):
    b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
    return NBINS * b[:, 1] + b[:, 0]

def snap_to_open_bins(pos_bin, blocked):
    ...
    return lut[pos_bin]
```

iii. The notes say this matches the dataset's `[[0,1,2],[3,4,5],[6,7,8]]` bin convention and that blocked-bin remapping is only a cleanup for a tiny fraction of samples.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output and neural data by pooling both over the same 3-frame windows and then applying the same retained-bin indices `idx` within each trial block, so the saved `output` and `neural` trials are time-matched bin by bin.

ii.
```python
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)

position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
pos_bin = position_to_bin(position_binned_cm).astype(np.int64)

for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    neural_trials.append(np.ascontiguousarray(neural[:, idx]))
    output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

iii. The notes explicitly say position and neural data are already DAQ-aligned in the source data and that the converted output shows no lag relative to the pooled position traces.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: it drops all-NaN unregistered cells, treats `blocked = -1` as no blocked partitions, clips edge positions into valid bins, drops leftover 1-2 frames that cannot form a 3-frame bin, optionally keeps a long enough final partial trial block, drops trials with too little running data, and snaps rare blocked-bin position samples to the nearest open bin.

ii.
```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.any(np.isnan(trace)), "a registered cell has partial NaNs -- unexpected"

blocked = blocked[blocked >= 0].astype(int)
...
b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
...
if n_bins - tail_start >= MIN_TAIL_BINS:
    bounds.append((tail_start, n_bins))
...
pos_bin = snap_to_open_bins(pos_bin, blocked)
```

iii. The notes justify these as practical fixes for dataset quirks, especially rare DeepLabCut boundary overshoots and session/trial formatting issues introduced by the decoder-specific 100 ms binning.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies loading and processing whole sessions as the expensive part, especially HDF5 reads, `gaussian_filter1d` smoothing, pooling, and final pickling. It therefore parallelizes at the animal level and avoids whole-animal joblib loads.

ii.
```python
from multiprocessing import Pool
...
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
...
if len(tasks) > 1 and args.nproc > 1:
    with Pool(min(args.nproc, len(tasks))) as pool:
        results = pool.map(process_animal, tasks)
...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Step 6 and Step 7 of the notes, the AI calls out whole-animal loading, filtering/pooling, multiprocessing, and pickle writing as the main runtime considerations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI mostly vectorized the heavy numerical work already. The main remaining explicit Python loops are the small loop over blocked bins in `snap_to_open_bins` and the per-trial loop that assembles trial arrays; the notes emphasize that slower per-frame loops from the reference code were deliberately replaced with vectorized operations.

ii.
```python
for b in blocked:
    d = np.linalg.norm(coords[open_bins] - coords[b], axis=1)
    lut[b] = open_bins[np.argmin(d)]

for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    ...
```

iii. The notes explicitly say the script replaced naive Python loops with `reshape+mean`, `np.floor`, and `np.clip`, leaving only these relatively small control-flow loops.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats some lightweight processing per session or per trial: it re-derives `blocked` from `env` as a cross-check for every session, copies the same geometry vector into every retained trial, and recomputes summary/bookkeeping fields for every session. If plotting is enabled, it also stores extra intermediates solely for diagnostic figures.

ii.
```python
expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected)
...
input_trials.append(geometry.copy())
...
if show_processing:
    result['_plot'] = dict(position=position, speed=speed, moving=moving,
                           neural=neural, pos_bin=pos_bin, moving_bins=moving_bins,
                           position_binned_cm=position_binned_cm)
```

iii. The notes frame these repetitions as built-in sanity checks and diagnostics rather than core conversion logic.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores several validation-oriented quantities that the downstream decoder does not need: running speed/masks, snapped-sample counts, trial-bound bookkeeping, rich session metadata, and optional plotting state. It also re-derives blocked geometry from `env` only to assert consistency.

ii.
```python
speed = running_speed(position)
moving = speed > V_THRESH
...
result = {
    ...
    'frac_moving': float(moving_bins.mean()),
    'n_snapped': n_snapped,
    'trial_bounds': trial_bounds,
}
if show_processing:
    result['_plot'] = dict(...)
```

iii. The notes describe these as sanity-check and documentation features added to validate the conversion, not as information required by the final decoder inputs or outputs.
