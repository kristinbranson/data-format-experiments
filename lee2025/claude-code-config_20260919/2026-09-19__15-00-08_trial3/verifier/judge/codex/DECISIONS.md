# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the MATLAB `.mat` files used by the human reference. Instead, it hard-codes the 7 animal IDs, loads one joblib file per animal from `/app/data`, extracts `trace`, `position`, `envs`, and `blocked`, and then processes each session/day inside that in-memory animal object.

ii.
```python
DATA_DIR = '/app/data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def process_animal(animal, days=None, show_processing=False, plot_dir='/app', verbose=True):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
```

iii. In `CONVERSION_NOTES.md`, the AI says the joblib and `.mat` files contain the same content, that the reference code already supports joblib loading, and that loading one animal once is a performance optimization because the decompressed arrays are large.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by animal file. The AI uses the 7 hard-coded animal IDs as the `subjects` list and treats each loaded joblib file as one mouse.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

data = {
    ...
    'subjects': list(ANIMALS), 'subject_idx': [],
    ...
}
```

iii. The notes state that `/app/data/<ANIMAL>` and `/app/data/<ANIMAL>.mat` hold the same content and that there are 7 animals total, so the AI uses those animal names directly as subject IDs.

## 1-c. How are the data split into sessions?

i. Within each subject, the AI treats each day/session along axis 0 of `trace`, `position`, `envs`, and `blocked` as one session. It iterates `day` over all available sessions for that animal.

ii.
```python
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
if days is None:
    days = list(range(trace.shape[0]))
...
for day in days:
    res = process_session(trace[day], position[day], blocked[day], str(envs[day]),
                          session_id=f'{animal}_day{day:02d}',
                          show_processing=show_processing,
                          plot_dir=plot_dir)
```

iii. The notes describe the joblib structure as `(n_days, ...)` arrays and report 207 total sessions, so the AI interprets each day as one recording session.

## 1-d. How are the data split into trials?

i. The AI first rebins each session from 30 Hz frames into 100 ms bins (`TBIN = 3` frames), then cuts the binned session into consecutive 1-minute blocks of 600 bins. Within each block it keeps only the bins whose average speed exceeds 5 cm/s, keeps the final partial block, and drops a block entirely if fewer than 30 moving bins remain.

ii.
```python
TBIN = 3
TRIAL_SECONDS = 60.0
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TBIN))    # 600 bins of 100 ms
MIN_TRIAL_BINS = 30
...
for start in range(0, nb, TRIAL_BINS):
    idx = np.arange(start, min(start + TRIAL_BINS, nb))
    sel = idx[moving[idx]]
    if len(sel) < MIN_TRIAL_BINS:
        n_dropped += 1
        continue
    trials_neural.append(np.ascontiguousarray(neural[:, sel]))
    trials_input.append(blocked_vec.copy())
    trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. The AI justifies this in the notes by saying the task requires 1-minute trials, the reference decoder uses 100 ms bins, and keeping only moving bins is consistent with the paper’s position-decoding pipeline.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by movement content: a 1-minute block is dropped if it has fewer than 30 retained 100 ms bins after the speed filter. Later, a session is also skipped from the final dataset if it ends up with fewer than 2 usable trials.

ii.
```python
MIN_TRIAL_BINS = 30
...
if len(sel) < MIN_TRIAL_BINS:
    n_dropped += 1
    continue
...
for s in sessions:
    if len(s['neural']) < 2:
        print(f"  !! skipping session {s['session_id']}: only {len(s['neural'])} usable trials")
        continue
```

iii. The notes say this was added because the target format requires at least 2 trials per session and because very short low-movement blocks were considered unusable after speed filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the `trace` variable in the per-animal joblib file. Each session uses `trace[day]`, which the AI interprets as binarized calcium-transient rising-phase events with shape `(n_cells, n_frames)`.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
...
res = process_session(trace[day], position[day], blocked[day], str(envs[day]), ...)
```

iii. The notes explicitly state that the dataset’s `trace` field is already the paper’s binarized rising-phase event train and should be treated as the firing-rate-like neural signal rather than recomputing dF/F.

## 2-b. How is the `neural` data processed?

i. The AI keeps only registered cells, smooths each cell’s binary trace with a Gaussian (`sigma=3` frames), averages non-overlapping 3-frame windows to 100 ms bins, and multiplies by 30 to express the result in events/s.

ii.
```python
registered = ~np.isnan(trace[:, 0])
raw = trace[registered][:, :nb * TBIN].astype(np.float32)
events_b = bin_time(raw, nb, how='sum')

smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
```

iii. The AI cites the reference decoder (`fit_decoder`) as justification: in the notes it argues that the paper’s position-decoding analysis smooths and temporally bins traces to 100 ms, so it mirrors that pipeline and expresses the result in Hz-like units.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered twice: first to registered cells only (`~np.isnan(trace[:, 0])`), then to cells with more than 5 events across the retained moving bins of that session.

ii.
```python
registered = ~np.isnan(trace[:, 0])
raw = trace[registered][:, :nb * TBIN].astype(np.float32)
events_b = bin_time(raw, nb, how='sum')
...
active = events_b[:, moving].sum(axis=1) > CELL_THRESH
cell_idx = np.where(registered)[0][active]
neural = neural[active]
```

iii. The notes say this matches `decode_position_within` from the reference code: unregistered cells are NaN and cells with `<= 5` events during moving periods are excluded by the reference decoder’s `cell_threshold=5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to a behavioral stimulus event. Instead, it defines an artificial event: the start of each 1-minute trial block. Within each block, only the time bins passing the speed filter are kept.

ii.
```python
for start in range(0, nb, TRIAL_BINS):
    idx = np.arange(start, min(start + TRIAL_BINS, nb))
    sel = idx[moving[idx]]
    ...
    trials_neural.append(np.ascontiguousarray(neural[:, sel]))
...
'temporal_alignment_event':
    'Start of each 1-minute trial. Sessions are continuous 40-min recordings cut into consecutive '
    '1-min blocks; within a block only the time bins in which the mouse ran faster than 5 cm/s are '
    'kept, so trials contain <= 600 time bins.',
```

iii. The notes justify this by saying there is no natural trial structure in the experiment, so sessions must be cut into consecutive 1-minute blocks to satisfy the decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data has 100 ms bins. The AI rebins the original 30 Hz data by averaging every 3 frames after Gaussian smoothing.

ii.
```python
FPS = 30.0
TBIN = 3
...
def bin_time(x, nbins, how='mean'):
    x = x[..., :nbins * TBIN]
    x = x.reshape(x.shape[:-1] + (nbins, TBIN))
    return x.mean(axis=-1) if how == 'mean' else x.sum(axis=-1)
...
'time_bin_size': 1000.0 * TBIN / FPS,
```

iii. The AI’s notes repeatedly justify 100 ms bins by pointing to `fit_decoder(..., temporal_bin_size=3)` in the paper’s decoder code.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived from the raw `blocked` field for each session/day.

ii.
```python
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
...
blocked_vec, blocked_ids = blocked_vector(blocked_entry)
```

iii. The notes say the `blocked` field directly stores which 3x3 partitions are occluded and that it was preferred over reconstructing geometry from environment names.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts each session’s `blocked` entry into a 9-dimensional binary vector with `1` for blocked partitions and `0` otherwise. This vector is static within a session and copied once per kept trial.

ii.
```python
def blocked_vector(blocked_entry):
    e = blocked_entry
    if isinstance(e, (list, tuple)):
        e = e[0]
    idx = np.atleast_1d(np.asarray(e, dtype=float)).ravel()
    idx = idx[idx >= 0].astype(int)
    vec = np.zeros(N_GRID * N_GRID, dtype=np.float32)
    vec[idx] = 1.0
    return vec, tuple(sorted(idx.tolist()))
...
trials_input.append(blocked_vec.copy())
```

iii. The notes justify this as the decoder-ready representation of environment geometry and say it matches the semantics of the session-level `blocked` field.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Mouse position is derived from the raw `position` field for each session/day.

ii.
```python
trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
...
res = process_session(trace[day], position[day], blocked[day], str(envs[day]), ...)
```

iii. The notes describe `position` as a `(n_days, 2, n_frames)` array of head position in centimeters with no missing values.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI computes smoothed speed from framewise position, averages position into 100 ms bins, discretizes each binned position into one of 9 partitions, repairs any labels that fall into a blocked partition by assigning the nearest open partition, keeps only bins above the speed threshold, and stores the resulting labels per trial.

ii.
```python
speed = compute_speed(position)
speed_b = bin_time(speed, nb)
moving = speed_b > V_THRESH
pos_b = bin_time(position, nb)
part, n_fixed = discretize_position(pos_b, blocked_ids)
...
trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. The notes say this follows the paper’s decoder for speed filtering and temporal binning, while the nearest-open reassignment is justified as a cleanup for rare tracking samples that land inside blocked partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI uses a fixed 3x3 grid over the 75 cm arena. It computes `x_bin = floor(x / 25)`, `y_bin = floor(y / 25)`, clips each to `0..2`, and encodes the category as `3 * y_bin + x_bin`.

ii.
```python
xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
part = N_GRID * yb + xb
```

iii. The notes justify this by saying the physical arena is 75 x 75 cm and the decoder task asks for a 3 x 3 partitioning, so each category is a 25 x 25 cm bin.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output and neural data by applying the same 100 ms binning to both streams and then using the same retained-bin indices `sel` when cutting trials. Neural and output trials therefore contain the same time bins.

ii.
```python
pos_b = bin_time(position, nb)
...
smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
...
sel = idx[moving[idx]]
...
trials_neural.append(np.ascontiguousarray(neural[:, sel]))
trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. The notes say the imaging and behavior streams are frame-aligned in the raw data and that using identical bin edges preserves that alignment at 100 ms resolution.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: it drops unregistered NaN cells, clips arena-edge positions into valid bins, reassigns rare samples that land in blocked partitions to the nearest open partition, drops trailing frames that do not fill a full 3-frame bin, keeps the final partial 1-minute block, and drops blocks with too little retained movement.

ii.
```python
registered = ~np.isnan(trace[:, 0])
...
x = x[..., :nbins * TBIN]
...
xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
...
bad = np.isin(part, blocked_ids)
...
part[bad] = open_ids[np.argmin(d, axis=1)]
...
if len(sel) < MIN_TRIAL_BINS:
    n_dropped += 1
    continue
```

iii. The notes justify these as defensive fixes for rare tracking or boundary issues and as necessary curation once the speed filter creates variable-length trial content.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies two major costs: loading/decompressing the large joblib animal files and smoothing the neural traces before binning. It also notes that these are mitigated by loading each animal once and processing animals in parallel.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
...
from joblib import Parallel, delayed
results = Parallel(n_jobs=n_jobs, verbose=0)(
    delayed(process_animal)(a, n, args.show_processing, plot_dir) for a, n in jobs)
```

iii. In the notes, the AI says file loading takes 6-16 s per animal and that Gaussian smoothing is the dominant compute step once the extra neural preprocessing is added.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s position is effectively that the important per-frame work was already vectorized. It replaces naive framewise binning with reshape-based pooling, so the remaining explicit loops are mostly over animals, sessions, and 1-minute blocks.

ii.
```python
def bin_time(x, nbins, how='mean'):
    x = x[..., :nbins * TBIN]
    x = x.reshape(x.shape[:-1] + (nbins, TBIN))
    return x.mean(axis=-1) if how == 'mean' else x.sum(axis=-1)
...
for day in days:
    res = process_session(...)
...
for start in range(0, nb, TRIAL_BINS):
    ...
```

iii. The notes explicitly say that naive frame loops were avoided and that vectorized reshape-and-reduce was the main speedup.

## 6-c. What processing does the code repeat multiple times?

i. The AI tries to avoid major repeated work by loading each animal once, but it still repeats some session- and trial-level bookkeeping: it copies the same static blocked vector into every trial, recomputes per-session metadata, and later computes summary statistics by iterating over all saved trials.

ii.
```python
trials_input.append(blocked_vec.copy())
...
session_info.append({
    'session_id': s['session_id'], 'animal': s['animal'], 'day': int(s['day']),
    'env': s['env'], 'blocked': list(s['blocked_ids']),
    'n_neurons': s['n_neurons'], 'n_registered': s['n_registered'],
    'n_bins_total': s['n_bins'], 'n_bins_kept': s['n_moving'],
    'n_trials': len(s['neural']),
    'cell_idx': s['cell_idx'].tolist(),
})
...
Ts = np.concatenate([[t.shape[1] for t in sess] for sess in data['neural']])
allout = np.concatenate([t.ravel() for sess in data['output'] for t in sess])
```

iii. The notes mostly frame this negatively only in contrast to what was avoided: they stress that repeated file loading was eliminated by processing all sessions for an animal after a single `joblib.load`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs extra diagnostics and bookkeeping that downstream decoder training does not use: optional processing plots, `session_info` metadata, `cell_idx`, `n_fixed`, and end-of-run summary statistics. It also carries some QC counters whose only purpose is reporting.

ii.
```python
if show_processing:
    _plot_processing(...)
...
result = {
    ...
    'cell_idx': cell_idx,
    'n_bins': int(nb),
    'n_moving': int(moving.sum()),
    'n_dropped_trials': n_dropped,
    'n_blocked_fixed': n_fixed,
}
...
session_info.append({
    ...
    'cell_idx': s['cell_idx'].tolist(),
})
...
print(f'Output distribution: {np.round(np.bincount(allout, minlength=9) / len(allout), 4).tolist()}')
```

iii. The notes justify these additions as sanity checks and validation support rather than as part of the core converted representation.
