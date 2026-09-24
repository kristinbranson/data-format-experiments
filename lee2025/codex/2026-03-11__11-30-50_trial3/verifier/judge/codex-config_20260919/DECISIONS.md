# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers the seven extensionless `QLAK-CA1-*` files, loads each primary joblib object, enumerates every day from `envs`, and later caches one animal at a time while processing its sessions. In sample mode it enumerates only the first two sessions.

ii.
```python
def get_animal_ids(data_dir: str) -> list[str]:
    return sorted(filename for filename in os.listdir(data_dir)
                  if filename.startswith("QLAK-CA1-") and "." not in filename)

def load_animal_dataset(data_dir: str, animal: str) -> dict:
    return joblib.load(os.path.join(data_dir, animal))[animal]
```

iii. The notes say joblib is the default path in the authors' `load_dat`, avoids cached analysis products, and contains the primary per-animal data. The one-animal cache limits memory and avoids reloading for every session.

## 1-b. How are the data split into subjects?

i. Each extensionless joblib file/key is one subject; IDs are sorted and mapped to integer indices.

ii.
```python
animals = get_animal_ids(data_dir)
subject_lookup = {animal: idx for idx, animal in enumerate(animals)}
```

iii. The notes identify seven primary subject datasets and state that one animal file contains all sessions for that subject.

## 1-c. How are the data split into sessions?

i. Each recording day in an animal's `envs` array becomes one output session, ordered by subject and then day.

ii.
```python
for animal in animals:
    dat = load_animal_dataset(data_dir, animal)
    for day_index in range(dat["envs"].shape[0]):
        session_refs.append(SessionRef(animal=animal, day_index=day_index))
```

iii. The agent states that the source is organized by day/session and the target supports multiple trials per session, so it preserves one target session per original day.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into non-overlapping 60-second slices of 1,800 frames at 30 Hz. A final incomplete slice is discarded, and sessions with fewer than two complete trials raise an error.

ii.
```python
def trial_slices(n_frames: int, trial_frames: int = TRIAL_FRAMES) -> list[slice]:
    n_trials = n_frames // trial_frames
    return [slice(i * trial_frames, (i + 1) * trial_frames) for i in range(n_trials)]
```

iii. The task explicitly requires one-minute trials. The notes say non-overlapping windows preserve within-session context and fixed-length alignment; the small remainder cannot form a complete trial.

## 1-e. How are trials filtered based on quality controls?

i. No individual full trial is filtered. Only incomplete tail frames are dropped; a session is rejected if it has fewer than two full trials.

ii.
```python
slices = trial_slices(n_frames)
if len(slices) < 2:
    raise ValueError(f"{session_ref.session_id}: fewer than 2 full 1-minute trials available")
```

iii. The paper has continuous recordings rather than native trials, and the notes report no trial-level curation rule. The two-trial check enforces the target-format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each day's released `trace` matrix in the joblib animal object.

ii.
```python
trace_day = np.asarray(dat["trace"][day], dtype=np.float64)
```

iii. The README and methods identify `trace` as the released rise-extracted binary calcium-event signal used by the paper's analyses.

## 2-b. How is the `neural` data processed?

i. After selecting session-present rows, traces are cast to `float16` and sliced into neuron-by-time trial matrices. No fluorescence, smoothing, deconvolution, or temporal aggregation is performed.

ii.
```python
session_trace = trace_day[present_mask].astype(np.float16, copy=False)
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice].astype(np.float16, copy=False)
```

iii. The agent concluded that the released trace is already the paper's binary rising-phase representation. It chose `float16` to halve a very large pickle; binary 0/1 values are exactly representable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cell is retained when its first frame is not NaN. The session is rejected if this leaves no cells. There is no place-cell, activity, or velocity filtering.

ii.
```python
def session_present_cell_mask(trace_day: np.ndarray) -> np.ndarray:
    return ~np.isnan(trace_day[:, 0])

if not np.any(present_mask):
    raise ValueError(f"{session_ref.session_id}: no present cells after NaN filtering")
```

iii. The notes say absent cells have NaN rows, all cells were included in the main analyses, and place-cell status was analysis-specific rather than required curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Frame zero of each non-overlapping one-minute slice is declared its alignment event.

ii.
```python
"temporal_alignment_event": "start of each non-overlapping 1-minute within-session segment",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The source sessions are continuous and have no trial event. The agent preserved their synchronized frame clock and used segment starts as the only applicable alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution remains the native 30 Hz, or 33.333 ms per frame. No temporal rebinning is applied.

ii.
```python
FPS = 30.0
"time_bin_size": 1000.0 / FPS,
```

iii. The methods say behavior and imaging were acquired at 30 Hz. The task does not require the reference decoder's optional three-frame pooling.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived primarily from the per-day `blocked` partition indices. The agent additionally loads `maps["smoothed"]` to validate the resulting footprint.

ii.
```python
geometry_grid, geometry_vector = blocked_to_geometry_vector(dat["blocked"], day)
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
```

iii. The notes call `blocked` the authoritative raw description and use the map support as an all-session orientation cross-check.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A nine-element vector begins as all ones (open), blocked indices are set to zero, the 3x3 grid is transposed, and it is flattened. A copy of this static vector is stored for every trial. Geometry must equal a 3x3 downsampling of valid map support or conversion aborts.

ii.
```python
geometry = np.ones(GEOMETRY_BINS * GEOMETRY_BINS, dtype=np.float32)
if not (blocked.size == 1 and blocked[0] == -1):
    geometry[blocked] = 0.0
geometry_grid = geometry.reshape(GEOMETRY_BINS, GEOMETRY_BINS).T.astype(np.float32)
input_trials.append(geometry_vector.copy())
```

iii. The agent interpreted “environment geometry” as accessible versus blocked arena support. It says transposition was necessary for asymmetric environments and matched the non-NaN map footprint in all 207 sessions.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output comes from each day's two-coordinate `position` stream.

ii.
```python
position_day = np.asarray(dat["position"][day], dtype=np.float64)
position_bins, output_class = compute_position_bins(position_day)
```

iii. The notes identify this as the synchronized continuous mouse/head position used by the paper's decoder.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The array is transposed to time-by-coordinate, each coordinate is floor-binned into three divisions using that session's coordinate maximum plus a small buffer, clipped to 0–2, and flattened into one class as `x_bin * 3 + y_bin`.

ii.
```python
coords = np.asarray(position_day, dtype=np.float64).T
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The agent says this adapts the reference code's floor-division rule from 15x15 to the required 3x3 and uses session-wide normalization so all derived trials share boundaries.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Per coordinate, thresholds are one-third and two-thirds of that session's observed maximum (with a `1e-5` buffer). The two indices produce classes 0–8 in x-major order.

ii.
```python
POSITION_BUFFER = 1e-5
scale = (np.nanmax(coords, axis=0) + POSITION_BUFFER) / float(n_bins)
binned = np.floor(coords / scale).astype(np.int64)
class_idx = (binned[:, 0] * n_bins + binned[:, 1]).astype(np.int64)
```

iii. The buffer keeps the maximum inside the last bin; clipping handles numerical boundaries. The notes explicitly prefer one nine-class output over separate x/y outputs.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position classes are computed once for the whole synchronized session and sliced with exactly the same trial slices as trace.

ii.
```python
for trial_slice in slices:
    neural_trial = session_trace[:, trial_slice]
    output_trial = output_class[trial_slice][np.newaxis, :].astype(np.int8, copy=False)
```

iii. The notes state that behavioral and imaging streams share the released 30 Hz aligned time base; direct identical slicing preserves framewise alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Absent-cell NaN rows are removed; no-present-cell, geometry-mismatch, or too-short sessions raise errors. Incomplete tail frames are recorded in metadata and dropped. Position maxima use `nanmax`; final validations checked that no NaN/Inf remained.

ii.
```python
if not np.array_equal(geometry_grid, valid_grid):
    raise ValueError(f"{session_ref.session_id}: geometry derived from blocked field does not match maps valid mask")
"discarded_tail_frames": int(n_frames - len(slices) * TRIAL_FRAMES),
```

iii. The agent favored explicit failures for structural inconsistencies and documented the only intentional loss, partial-minute tails. It reports exact checks on representative sessions and all-session summary checks.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large full-animal joblib files is identified as the dominant step; serialization of the 9.3 GB result is also substantial, while per-session slicing is simple.

ii.
```python
animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
with open(outpicklefile, "wb") as f:
    pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes measured multi-second effective session conversion and explicitly identify the 68–145 MB full-animal loads as the main runtime cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Core conversion is already array-based; the trial loop mainly creates the required list objects. The optional plotting occupancy loop could be replaced by `np.add.at`/`bincount`, and trial lists could be formed from reshaped full-minute prefixes, though lists would still be needed for output format.

ii.
```python
for xbin, ybin in position_bins:
    occupancy[xbin, ybin] += 1.0

for trial_slice in slices:
    neural_trials.append(session_trace[:, trial_slice])
```

iii. The agent's notes do not explicitly call out vectorizable loops; they instead emphasize that full-session binning is already vectorized and avoids per-trial recomputation.

## 6-c. What processing does the code repeat multiple times?

i. Every animal is loaded once during session-reference enumeration and again during conversion. Static geometry is copied once per trial. In optional plots, class labels are recomputed from already available position bins.

ii.
```python
dat = load_animal_dataset(data_dir, animal)  # iter_session_refs
...
animal_cache[session_ref.animal] = load_animal_dataset(data_dir, session_ref.animal)
...
session_output = (position_bins[:, 0] * POSITION_BINS + position_bins[:, 1]).astype(int)
```

iii. The notes claim reuse prevents per-session reloads and that session-wide position binning avoids trial-level repetition, but do not acknowledge the separate enumeration load or optional plotting recomputation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads each day's full smoothed rate-map tensor only to derive a geometry consistency mask; the mask is not saved. Optional processing figures compute occupancy and plots that are not used by the decoder. Timing/session metadata is retained but not consumed by training.

ii.
```python
smoothed_day = np.asarray(dat["maps"]["smoothed"][:, :, :, day], dtype=np.float64)
valid_grid = aggregate_valid_map(smoothed_day)
if show_processing and session_ref.session_id in plot_session_ids:
    plot_processing_figure(...)
```

iii. The map work is deliberate validation of the chosen geometry orientation, and plotting is opt-in diagnostic work. The notes present these as sanity checks, not decoder features.
