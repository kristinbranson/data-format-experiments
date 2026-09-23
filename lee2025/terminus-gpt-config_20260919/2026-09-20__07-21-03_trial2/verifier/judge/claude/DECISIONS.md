# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from preprocessed joblib files (extensionless files in `/app/data`) rather than the `.mat` files. Each animal's data is loaded as a dictionary using `joblib.load(f)[f.name]`, which contains `trace`, `position`, `blocked`, and `envs` arrays among other fields. The data is loaded one animal at a time to manage memory.

ii.
```python
def animal_files():
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.name.startswith('QLAK-CA1-') and not p.suffix and '_' not in p.name)
# ...
d=joblib.load(f)[f.name]
```

iii. The AI noted that the data directory contains both joblib (extensionless) and MATLAB `.mat` formats. The reference code's `load_dat` function uses joblib loading, so the AI followed that convention. This is documented in CONVERSION_NOTES.md Step 1, which identifies `load_dat` as using joblib.

## 1-b. How are the data split into subjects?

i. Each extensionless joblib file corresponds to one subject (mouse). The subject name is derived from the filename (e.g., `QLAK-CA1-08`). Seven subjects are identified.

ii.
```python
files=animal_files()
subjects=[p.name for p in files]
# ...
for si, f in enumerate(files):
    d=joblib.load(f)[f.name]
```

iii. Each file contains all recording sessions for one animal, matching the data organization described in the paper and reference code.

## 1-c. How are the data split into sessions?

i. Each recording day within a subject file becomes a separate session. The AI iterates over `d['trace'].shape[0]` days per animal.

ii.
```python
for day in range(d['trace'].shape[0]):
    # ...
    nmat, out, geom, present, ncorrect = process_day(
        d['trace'][day], d['position'][day], d['blocked'][day])
```

iii. The data structure stores `trace` as `(n_days, n_registered_cells, n_frames)`, so each day index corresponds to one recording session. This matches the paper's "one session was recorded per day" protocol.

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 60-second segments. After 3-frame temporal pooling (see 2-b), each trial has 600 time bins (60s / 0.1s). Remainder bins that don't fill a complete trial are discarded. Sessions with fewer than 2 full trials are skipped.

ii.
```python
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)  # 600
# ...
ntrials=nmat.shape[1]//TRIAL_BINS
if ntrials < 2:
    print(f'WARNING skipping {f.name} day {day}: only {ntrials} full trials')
    continue
keep=ntrials*TRIAL_BINS
ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
```

iii. The instructions specify "1-minute trials" and require "at least two trials within each session." The AI implements both requirements. The temporal pooling means trials have 600 bins rather than 1800 frames.

## 1-e. How are trials filtered based on quality controls?

i. No per-trial quality filtering is applied beyond the minimum-2-trials-per-session check. All complete 60-second segments are retained.

ii.
```python
if ntrials < 2:
    print(f'WARNING skipping {f.name} day {day}: only {ntrials} full trials')
    continue
```

iii. The paper describes continuous 40-minute recording sessions with no trial-level quality criteria. The only filtering is structural: sessions must have at least 2 complete trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binary calcium event traces (0/1 rising-phase events) with shape `(n_days, n_registered_cells, n_frames)`.

ii.
```python
def process_day(trace_day, position_day, blocked_raw):
    present = np.isfinite(trace_day[:, 0])
    # ...
    raw = np.asarray(trace_day[present].T, dtype=np.float32)  # time x neurons
```

iii. The `trace` variable contains pre-processed binary calcium events as documented in the paper: "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5, and 0 otherwise."

## 2-b. How is the `neural` data processed?

i. The AI applies two temporal processing steps matching the reference paper's decoder code: (1) Gaussian smoothing with sigma=3 native frames, and (2) non-overlapping 3-frame mean pooling. This produces 100ms time bins from the native 30Hz data.

ii.
```python
POOL = 3
# ...
# Match reference fit_decoder/test_decoder: temporal Gaussian sigma=3 then AvgPool stride=3.
smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
```

iii. The AI explicitly references the `fit_decoder`/`test_decoder` functions in the reference code, which apply Gaussian smoothing (sigma=3) and `AvgPool1d(3,3)`. This is documented in CONVERSION_NOTES.md Steps 1, 4, and 5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons that are not present (registered) on a given day are removed. Presence is determined by checking if the first frame is finite (non-NaN). The AI verifies the mask is consistent across frames. No place-cell filtering is applied.

ii.
```python
present = np.isfinite(trace_day[:, 0])
if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
    raise ValueError('Cell registration mask changes within a session')
raw = np.asarray(trace_day[present].T, dtype=np.float32)
if not np.isfinite(raw).all():
    raise ValueError('Non-finite value in a day-present neural trace')
```

iii. All-NaN rows represent cells not registered/present on that day. The AI noted from the paper that "reliability results motivated the inclusion of all cells in subsequent analyses," justifying not filtering to place cells only.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No event-based alignment is applied. The recording is continuous, and trials are artificial 60-second non-overlapping segments starting from the beginning of each session.

ii. N/A - alignment is implicit in the temporal processing and trial slicing.

iii. There are no stimulus events in this free-navigation paradigm. The alignment event is described as "Start of each contiguous non-overlapping 1-minute segment within a recording day."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has 100ms time bins (10 Hz effective rate). Temporal rebinning is applied: Gaussian smoothing (sigma=3 native frames) followed by non-overlapping 3-frame mean pooling, reducing from 30 Hz to 10 Hz.

ii.
```python
BIN_MS = 100.0
POOL = 3
# ...
smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
```

iii. The AI justified this by matching the reference code's `fit_decoder`/`test_decoder` functions, which apply the same temporal processing pipeline. The metadata records `time_bin_size=100.0`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` variable, which contains indices of blocked reward/partition locations for each recording session.

ii.
```python
def blocked_indices(raw):
    vals = np.asarray(raw[0]).ravel().astype(int)
    return vals[vals >= 0]

def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom
```

iii. The `blocked` field stores which of the 9 possible positions (in row-major 3x3 order) are blocked. A value of `[-1]` indicates no positions blocked.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI creates a 9-element float32 vector where 1=accessible and 0=blocked. This is the opposite convention from the human reference (which uses 1=blocked, 0=not-blocked). The vector is static per session, replicated for each trial.

ii.
```python
def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom
# ...
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The AI followed the reference code's `get_env_mat` function convention, which represents accessible partitions as 1 and blocked as 0. Input names are `accessible_<location>` to match this encoding.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable, which contains 2D (x, y) coordinates of the animal in the arena at each timepoint.

ii.
```python
pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
```

iii. The `position` variable records the animal's head location from DeepLabCut tracking at 30 Hz in a 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The position is first temporally averaged in 3-frame windows (matching the neural pooling), then discretized into a 3x3 grid using 25cm bins: `x_bin = clip(floor(x/25), 0, 2)`, `y_bin = clip(floor(y/25), 0, 2)`, and the class label is `y_bin * 3 + x_bin`. Rare positions in blocked bins are snapped to the nearest accessible bin.

ii.
```python
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
classes = (ybin * 3 + xbin).astype(np.int8)
geom = geometry_vector(blocked_raw)
classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
```

iii. The 25cm bin width comes from the 75cm arena divided into 3 equal parts. The blocked-bin correction handles rare tracking artifacts (0.003% of frames) that place the animal in inaccessible partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized by dividing each coordinate by 25cm, truncating to integer, and clipping to [0,2]. This creates a 3x3 grid with 9 classes (0-8) using `y_bin * 3 + x_bin`.

ii.
```python
xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
classes = (ybin * 3 + xbin).astype(np.int8)
```

iii. The 3x3 discretization is specified in the task instructions. The `y*3+x` row-major convention was validated by the AI against blocked partition data (only 0.003% violations vs 17% with wrong convention).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are processed with identical 3-frame temporal pooling windows before trial slicing. Both are split at the same bin boundaries.

ii.
```python
# In process_day():
n_pool = raw.shape[0] // POOL
n_used = n_pool * POOL
smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
```

iii. Both streams use the same `n_used = n_pool * POOL` frames and the same 3-frame groupings, ensuring temporal alignment is preserved.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of data issues are handled: (1) Neurons not present on a day (all-NaN) are excluded; (2) Remainder frames that don't fill complete 3-frame pools or complete trials are discarded; (3) Rare tracking positions in blocked bins are snapped to the nearest accessible bin.

ii.
```python
present = np.isfinite(trace_day[:, 0])
raw = np.asarray(trace_day[present].T, dtype=np.float32)
# ...
classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
```

iii. The blocked-bin correction addresses 152 pooled-bin tracking artifacts across the full dataset. The AI verified that only 445/14,903,019 native frames (0.003%) fell in blocked bins, indicating tracking noise rather than systematic error.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the compressed joblib animal files, taking approximately 9-23 seconds per animal. Session-level processing is fast (~0.2s/session). Full conversion takes about 170 seconds total.

ii. N/A (timing is reported in print statements)

iii. Documented in CONVERSION_NOTES.md Step 7: source loading dominates at ~2 min for seven animals, while 207 session conversions take <1 min total.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `nearest_accessible` function uses a loop-free vectorized approach with broadcasting to find nearest accessible bins. The main processing uses vectorized NumPy operations (reshape-based pooling, vectorized discretization). The per-trial splitting loop creates list slices but is unavoidable for the output format.

ii.
```python
# Vectorized nearest accessible computation:
valid = np.flatnonzero(geom).astype(np.int8)
vx, vy = valid % 3, valid // 3
bx, by = xbin[bad, None], ybin[bad, None]
nearest = valid[np.argmin((bx-vx[None, :])**2 + (by-vy[None, :])**2, axis=1)]
```

iii. The AI designed the code with vectorization in mind from the start, using reshape-based pooling rather than explicit loops.

## 6-c. What processing does the code repeat multiple times?

i. The geometry vector is copied for each trial within a session (`geom.copy() for _ in range(ntrials)`), but this is trivial (9-element array). No substantial processing is repeated.

ii.
```python
ins=[geom.copy() for _ in range(ntrials)]
```

iii. The geometry is constant per session, so this is minimal overhead.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `session_info` metadata including environment names, blocked indices, native frame counts, and discarded-frame accounting for each session. This metadata is stored but not used by the decoder. The `envs` field is loaded but only used for metadata/plotting. The blocked-bin correction tracking (`n_corrected`) is informational only.

ii.
```python
env=str(np.asarray(d['envs']).ravel()[day])
info=dict(session_id=f'{f.name}_day{day:02d}', subject=f.name, source_day=day,
          environment=env, blocked=blocked_indices(d['blocked'][day]).tolist(),
          native_frames=int(d['trace'].shape[2]), present_neurons=int(present.sum()),
          n_trials=int(ntrials), retained_time_bins=int(keep),
          discarded_native_equivalent_frames=int(d['trace'].shape[2]-keep*POOL),
          corrected_blocked_pooled_bins=int(ncorrect))
```

iii. This metadata supports provenance tracking and debugging but is not used by the downstream decoder.
