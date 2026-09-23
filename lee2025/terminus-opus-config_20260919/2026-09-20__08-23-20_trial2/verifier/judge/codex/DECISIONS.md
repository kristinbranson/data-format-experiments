# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data animal-by-animal from a hard-coded list of seven animal IDs and reads the extensionless per-animal `joblib` files in `/app/data`. For each animal it loads the full dict, then uses the `trace`, `position`, `envs`, and `blocked` arrays to process each session. It does not load the `.mat` files.

ii.
```python
import joblib

DATA_DIR = "/app/data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

animals = ANIMALS[3:4] if sample else ANIMALS
for ai, animal in enumerate(animals):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    envs = np.asarray(dat['envs']).ravel()
    n_days = dat['trace'].shape[0]
```

iii. In `CONVERSION_NOTES.md`, the AI says the joblib and `.mat` files contain identical content, that the reference code path `load_dat(..., format="joblib")` uses the joblib files, and that joblib loading is about 5x faster than reading the large `.mat` files.

## 1-b. How are the data split into subjects (mice)?

i. Each animal ID in the hard-coded `ANIMALS` list becomes one subject. The subject string is appended once to `data['subjects']`, and every session from that animal gets the same `subject_idx`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

subject_idx = len(data['subjects'])
data['subjects'].append(animal)
...
data['subject_idx'].append(subject_idx)
```

iii. The notes describe the dataset as seven per-animal files and treat “animal” as the subject unit, matching the paper’s per-mouse organization.

## 1-c. How are the data split into sessions?

i. Each recording day within an animal file is treated as one session. The AI uses the first axis of `dat['trace']` to count sessions and iterates `d` from `0` to `n_days - 1`, appending one converted session per day.

ii.
```python
n_days = dat['trace'].shape[0]
...
for d in range(n_days):
    res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
```

iii. In the notes, the AI states that the paper’s unit of analysis is the animal-day session and that all 207 day-level sessions are kept.

## 1-d. How are the data split into trials?

i. The AI first rebins each continuous session to 100 ms bins, then slices it into non-overlapping 1-minute trials of 600 bins each. Unlike the human reference, it keeps a final partial trial if it is at least 30 seconds long.

ii.
```python
TRIAL_SECONDS = 60
TRIAL_BINS = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES
MIN_PARTIAL_TRIAL_BINS = TRIAL_BINS // 2

def trial_slices(n_bins):
    slices = []
    n_full = n_bins // TRIAL_BINS
    for t in range(n_full):
        slices.append((t * TRIAL_BINS, (t + 1) * TRIAL_BINS))
    rem = n_bins - n_full * TRIAL_BINS
    if rem >= MIN_PARTIAL_TRIAL_BINS:
        slices.append((n_full * TRIAL_BINS, n_bins))
    return slices
```

iii. `CONVERSION_NOTES.md` says this preserves more data and that the format allows variable trial lengths, so sessions are represented as 39 full 60 s trials plus a final 55.5-60 s trial.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filtering. The AI does not remove trials for low speed or poor quality. It only drops leftover frames that cannot fill a 100 ms bin, discards sub-30-second terminal fragments, and asserts that each session still has at least two trials.

ii.
```python
def trial_slices(n_bins):
    ...
    if rem >= MIN_PARTIAL_TRIAL_BINS:
        slices.append((n_full * TRIAL_BINS, n_bins))

...
assert len(data['neural'][s]) >= 2, f'session {s} has < 2 trials'
```

iii. The notes explicitly justify not using the reference decoder’s speed filter because the paper’s rate maps use all frames and removing stationary frames would break the required contiguous 1-minute trial structure.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data comes from the per-session `trace` array in each animal dict, specifically `dat['trace'][d]`.

ii.
```python
for d in range(n_days):
    res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
```

```python
def process_session(trace_day, position_day, env_name, blocked_field):
    # trace_day: (n_cells_all, n_frames) binary events, NaN rows for unregistered cells
```

iii. The AI’s notes say `trace` is already the authors’ binarized calcium-event rising-phase signal and should be used directly rather than recomputing anything like dF/F.

## 2-b. How is the `neural` data processed?

i. Within each session, the AI keeps registered cells, smooths the binary event trains with `gaussian_filter1d(sigma=3)` over time, average-pools non-overlapping 3-frame windows to 100 ms bins, multiplies by 30 to express values in events/s, and then cuts the result into trials.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
tr = trace_day[registered].astype(np.float32)

tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (bin_time_series(tr_s) * FPS).astype(np.float32)

for (a, b) in trial_slices(n_bins):
    neural.append(np.ascontiguousarray(rates[:, a:b]))
```

iii. The notes justify this as matching the paper’s decoder preprocessing (`fit_decoder`), specifically Gaussian smoothing and 3-frame pooling, rather than keeping the native 30 Hz binary traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI removes cells that were not registered in a session by excluding rows that are NaN for the session. It does not apply place-cell, split-half reliability, or minimum-event thresholds.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
assert np.all(np.isnan(trace_day[~registered]).all(axis=1)), \
    "unregistered cells must be NaN for the entire session"
tr = trace_day[registered].astype(np.float32)
```

iii. The notes say this follows the paper’s “include all cells” choice for downstream analyses and intentionally omits the `>5` event filter used in one reference decoder because that rule is decoder-specific and would only remove about 1.3% of cell-sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. There is no experimental event alignment. The AI treats each session as a continuous recording, bins neural and behavioral streams on the same timeline, and defines alignment relative to the start of each synthetic 1-minute trial.

ii.
```python
for (a, b) in trial_slices(n_bins):
    neural.append(np.ascontiguousarray(rates[:, a:b]))
    output.append(part[a:b][None, :].astype(np.int64))
    inputs.append(blocked_vec.copy())
```

```python
'temporal_alignment_event': (
    'start of each 1-minute trial; trial k starts 60*k s after the start of the continuous '
    '40-min recording session (there is no explicit task event - free foraging)'),
```

iii. The notes cite the free-foraging task and the paper’s statement that behavior and calcium were recorded on the same 30 Hz DAQ clock, so no additional event alignment was needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI converts the data to 100 ms bins by pooling every 3 original 30 Hz frames after Gaussian smoothing. Yes, temporal rebinning is applied.

ii.
```python
FPS = 30
TEMPORAL_BIN_FRAMES = 3
SMOOTH_SIGMA_FRAMES = 3

def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)

'time_bin_size': 1000.0 * TEMPORAL_BIN_FRAMES / FPS,
```

iii. The AI says this choice was taken from the reference decoder’s `temporal_bin_size=3` and was intended to make neural and position streams share 100 ms bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry primarily from the session’s `envs` label via a hard-coded map of environment names to 3x3 open/blocked matrices. It also reads the raw `blocked` field, but only as a consistency check.

ii.
```python
ENV_MATS = {
    'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'o':         [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    ...
}

blocked_vec = env_blocked_vector(env_name)

bl = np.atleast_1d(np.asarray(blocked_field[0]).ravel()).astype(int)
from_field = np.zeros(9, dtype=np.float32)
if not (bl.size == 1 and bl[0] == -1):
    from_field[bl] = 1
assert np.array_equal(from_field, blocked_vec)
```

iii. The notes say this was chosen because the decoder input is “environment geometry,” while the `blocked` field was used to verify that the geometry-derived indexing matched the dataset for all sessions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the named environment into a 3x3 matrix, flips it vertically to match dataset indexing, ravels it into a length-9 vector, and marks blocked partitions with `1`. That static vector is copied into every trial of the session.

ii.
```python
def env_blocked_vector(env_name):
    mat = np.array(ENV_MATS[env_name], dtype=float)
    open_flat = np.flipud(mat).ravel()
    return (open_flat == 0).astype(np.float32)

...
inputs.append(blocked_vec.copy())
```

iii. The notes say this yields the same 0-8 partition labeling used for the output position categories and matches the constant-within-session geometry assumption.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position comes from the per-session `position` array, i.e. `dat['position'][d]`, which contains 2D `(x, y)` coordinates over time.

ii.
```python
for d in range(n_days):
    res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
```

```python
def process_session(trace_day, position_day, env_name, blocked_field):
    # position_day: (2, n_frames) x, y position in cm
```

iii. The notes describe these as DeepLabCut-derived x/y positions in centimeters over the 75 x 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI average-pools x and y into the same 100 ms bins as the neural data, discretizes each binned position into a 3x3 arena partition, clips boundary values into the valid range, and snaps bins that fall inside blocked partitions to the nearest open partition.

ii.
```python
pos_b = bin_time_series(position_day.astype(np.float64))
part, n_snapped = position_to_partition(pos_b, blocked_vec)
```

```python
def position_to_partition(pos_binned, blocked_vec):
    col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
    row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
    p = (N_PART * row + col).astype(np.int64)
    bad = blocked_vec[p] > 0
    if np.any(bad):
        open_ids = np.flatnonzero(blocked_vec == 0)
        d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None], axis=2)
        p[bad] = open_ids[np.argmin(d, axis=1)]
    return p, int(np.sum(bad))
```

iii. The notes justify 100 ms pooling as matching the rebinned neural data and justify the snapping step as handling rare tracking noise that places the animal inside blocked partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI uses a fixed 75 cm arena split into `25 cm x 25 cm` bins. It computes `col = floor(x/25)`, `row = floor(y/25)`, clips both to `[0, 2]`, and encodes the category as `p = 3 * row + col`, giving labels `0-8`. Any blocked category is reassigned to the nearest open category.

ii.
```python
col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
p = (N_PART * row + col).astype(np.int64)
bad = blocked_vec[p] > 0
...
p[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. The notes say this thresholding matches the dataset’s blocked-partition indexing and avoids producing invalid blocked-position labels from a very small number of noisy samples.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI bins position with the same 3-frame windows used for neural data, asserts that the resulting bin counts match, and then slices neural and output arrays with identical trial boundaries.

ii.
```python
tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (bin_time_series(tr_s) * FPS).astype(np.float32)
pos_b = bin_time_series(position_day.astype(np.float64))

n_bins = rates.shape[1]
assert pos_b.shape[1] == n_bins

for (a, b) in trial_slices(n_bins):
    neural.append(np.ascontiguousarray(rates[:, a:b]))
    output.append(part[a:b][None, :].astype(np.int64))
```

iii. The AI’s notes state that behavior and calcium are recorded on the same frame clock, so shared pooling windows and identical slices are sufficient to preserve alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes unregistered cells whose traces are NaN for an entire session, asserts that NaN cells are fully NaN rather than partially missing, clips boundary positions at the edge of the arena into the last spatial bin, snaps the rare binned positions that land in blocked compartments to the nearest open compartment, drops the last 0-2 raw frames if they cannot make a complete 100 ms bin, and drops any final trial fragment shorter than 30 s.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
assert np.all(np.isnan(trace_day[~registered]).all(axis=1))

def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)

col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
...
if np.any(bad):
    ...
    p[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. The notes justify these as sanity-protective edge-case handlers, especially the blocked-partition snapping and the decision to preserve long terminal fragments instead of discarding them entirely.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies file loading and full-session neural processing as the main costs, with explicit timing around per-animal `joblib.load`, per-session processing, and final pickle writing.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
t_load = time.time() - t0

t1 = time.time()
for d in range(n_days):
    res = process_session(...)
...
print(f'  processed {n_days} sessions in {time.time() - t1:.1f}s '
      f'({(time.time() - t1) / n_days:.2f}s/session)')

t0 = time.time()
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In `CONVERSION_NOTES.md`, the AI reports full-run timings of about 89 s for loading, 93 s for processing, and 5 s for pickle writing, and describes the neural smoothing/pooling over large matrices as the dominant compute step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s main decision was to vectorize the expensive temporal processing rather than iterate frame-by-frame. The remaining Python loops are over animals, sessions, trials, and final summary statistics; these could be compressed further, but they are not the main bottleneck.

ii.
```python
def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)

tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (bin_time_series(tr_s) * FPS).astype(np.float32)
```

iii. The notes explicitly say the reference per-frame loops were replaced by vectorized smoothing and reshape/mean pooling, and they present this as a deliberate speed optimization.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats a few lightweight operations: it copies the same static `blocked_vec` into every trial of a session, re-traverses all trials to compute summary occupancy statistics, and scans first-trial neural arrays again to print a neural-value range summary.

ii.
```python
for (a, b) in trial_slices(n_bins):
    ...
    inputs.append(blocked_vec.copy())
```

```python
occ = np.zeros(9)
for s in range(ns):
    for o in data['output'][s]:
        occ += np.bincount(o[0], minlength=9)

allr = np.concatenate([data['neural'][s][0].ravel() for s in range(ns)])
```

iii. The notes frame the code as largely avoiding repeated heavy processing, so the repeated work that remains is mainly for bookkeeping, summaries, and duplicating the static session input across trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code carries extra diagnostics for validation and plotting that are not needed by downstream decoding, including `n_snapped`, `blocked_vec`, `rates`, `pos_b`, `part`, and `tr` in the `process_session` return dict, plus optional visualization generation and summary-statistic passes. These are used for checks/documentation and then discarded.

ii.
```python
return {
    'neural': neural, 'output': output, 'input': inputs,
    'n_neurons': int(registered.sum()),
    'n_bins': n_bins,
    'n_snapped': n_snapped,
    'blocked_vec': blocked_vec,
    'rates': rates, 'pos_b': pos_b, 'part': part, 'tr': tr,
}
```

```python
if args.show_processing and n_plotted < 2:
    plot_processing(session_id, res, dat['position'][d], dat['trace'][d])
```

iii. The notes make clear that these extras were intentionally included to support sanity checks, figures, and documentation rather than the final decoder input/output tensors themselves.
