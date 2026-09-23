# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data one animal at a time from the preconverted `joblib` archives in `/app/data`, not from the `.mat` files. It uses a fixed animal list, opens each archive with `joblib.load`, and then reads that animal's `trace`, `position`, `envs`, and `blocked` arrays for all days/sessions.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def convert_animal(animal, days=None, show_processing=False, plot_dir="."):
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    ...
    trace = dat["trace"]
    position = dat["position"]
    envs = np.asarray(dat["envs"]).ravel()
    blocked = dat["blocked"]
```

iii. In `CONVERSION_NOTES.md`, the AI says the shipped joblib files are a lossless conversion of the `.mat` files and load much faster, so it chose them as an efficiency optimization while claiming the contents are identical.

## 1-b. How are the data split into subjects (mice)?

i. Each animal archive is treated as one subject. Subject identities come from the fixed `ANIMALS` list, and `subjects` is built from the keys of `total_cells` after each animal is processed.

ii.
```python
plan = OrderedDict((a, None) for a in ANIMALS)
...
for animal, days in plan.items():
    out, ncells = convert_animal(animal, days=days, show_processing=show,
                                 plot_dir=args.plot_dir)
    sessions.extend(out)
    total_cells[animal] = int(ncells)

subjects = list(total_cells.keys())
data = build_dataset(sessions, subjects)
```

iii. The notes state that the dataset is organized per animal, with one file per animal and 7 total animals, so the AI used the animal IDs directly as subject identifiers.

## 1-c. How are the data split into sessions?

i. Each day within an animal is treated as one session. The AI loops over `range(n_days)` and converts each day separately.

ii.
```python
trace = dat["trace"]                      # (n_days, n_cells, T)
position = dat["position"]                # (n_days, 2, T)
...
n_days = trace.shape[0]
...
for d in days:
    sid = f"{animal}_day{d:02d}_{str(envs[d]).replace(' ', '')}"
    res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down,
                          sid, show_processing=show_processing, plot_dir=plot_dir)
```

iii. The notes explicitly say “Session = one animal-day (40-min recording)” and justify this as matching the paper’s unit of analysis.

## 1-d. How are the data split into trials?

i. The AI creates artificial trials by cutting each continuous session into consecutive, non-overlapping 60 s chunks. It computes `n_trials` by floor division on frame count and drops the trailing partial chunk.

ii.
```python
FPS = 30.0
TRIAL_SECONDS = 60.0
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FPS))          # 1800
...
n_cells_total, n_frames = trace_day.shape
n_trials = n_frames // TRIAL_FRAMES          # complete 1-minute trials only
if n_trials < MIN_TRIALS_PER_SESSION:
    return None
n_used = n_trials * TRIAL_FRAMES
```

iii. The notes say trials are not part of the original experiment, but the decoder task requires them, so it used the task-specified 1-minute non-overlapping segmentation.

## 1-e. How are trials filtered based on quality controls?

i. After trial splitting, the AI drops any trial with fewer than 10 retained 500 ms bins after its running-speed and blocked-position filtering. It also drops sessions with fewer than 2 surviving trials.

ii.
```python
MIN_BINS_PER_TRIAL = 10
MIN_TRIALS_PER_SESSION = 2
...
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
    input_trials.append(geometry.copy())
    output_trials.append(partition[t, sel][None, :].astype(np.int64))
...
if len(neural_trials) < MIN_TRIALS_PER_SESSION:
    return None
```

iii. The notes justify this as removing “empty/degenerate trials” and enforcing the decoder’s need for at least two trials per session for train/validation splitting.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-animal `trace` array, which the AI interprets as binarized calcium event traces with shape `(n_days, n_cells, T)`.

ii.
```python
trace = dat["trace"]                      # (n_days, n_cells, T)
...
def convert_session(trace_day, position_day, blocked_day, env_name, bin_down,
                    session_id, show_processing=False, plot_dir="."):
    ...
    registered = ~np.isnan(trace_day[:, 0])
    raw = trace_day[registered][:, :n_used].astype(np.float32)
```

iii. The notes say the paper’s preprocessing has already produced the final binary rising-phase event signal, so no dF/F computation is needed and `trace` is the raw source for neural decoding input.

## 2-b. How is the `neural` data processed?

i. The AI keeps only registered cells, smooths active traces with `gaussian_filter1d`, average-pools them into 500 ms bins, scales by `FPS` to express rates in Hz, and then slices out only the kept bins for each retained trial.

ii.
```python
raw = trace_day[registered][:, :n_used].astype(np.float32)
...
frame_keep = np.repeat(keep.ravel(), TEMPORAL_BIN_FRAMES)
n_events = raw[:, frame_keep].sum(axis=1)
active = n_events > CELL_EVENT_THRESHOLD
...
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```

iii. The notes say this follows the paper’s “smooth then average-pool” decoder recipe, but deliberately changes the temporal bin from 100 ms to 500 ms to improve SNR for the provided downstream decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neural QC filters: it removes cells not registered on that day (`NaN` traces), then removes cells with `<= 5` events among retained samples.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
raw = trace_day[registered][:, :n_used].astype(np.float32)
...
n_events = raw[:, frame_keep].sum(axis=1)
active = n_events > CELL_EVENT_THRESHOLD
if active.sum() == 0:
    return None
```

iii. The notes cite the paper’s decoding pipeline for the `> 5` event threshold and say there is no place-cell selection, only registration and event-count filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not align to any experimental stimulus event. It treats the start of each artificial 1-minute trial as the alignment event and says the source experiment has no discrete trial structure.

ii.
```python
"temporal_alignment_event":
    "start of each 1-minute trial; trials are consecutive non-overlapping 60 s "
    "segments of the continuous 40-min free-foraging session (the experiment has "
    "no discrete trial structure or task events)",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The notes state that this is continuous free foraging, so there is no genuine event to align to; the trial start is only a bookkeeping alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data to 500 ms bins (`15` frames at 30 Hz). It uses Gaussian smoothing plus non-overlapping mean pooling within each 1-minute trial.

ii.
```python
TEMPORAL_BIN_FRAMES = 15   # 500 ms
BINS_PER_TRIAL = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES
TIME_BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0
...
def bin_time(x, n_trials, axis=-1, reduce="mean"):
    x = np.moveaxis(x, axis, -1)
    x = x[..., : n_trials * TRIAL_FRAMES]
    shp = x.shape[:-1] + (n_trials, BINS_PER_TRIAL, TEMPORAL_BIN_FRAMES)
    out = x.reshape(shp).mean(axis=-1)
    return out
```

iii. The notes explicitly justify changing from the reference’s 100 ms bins to 500 ms bins on the grounds of sparse calcium events and better decoder accuracy with larger bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The environment-geometry input is derived from the raw `blocked` field for each session/day.

ii.
```python
blocked = dat["blocked"]
...
def parse_blocked(blocked_day):
    idx = np.atleast_1d(np.asarray(blocked_day, dtype=float).ravel())
    geometry = np.zeros(SPATIAL_BINS * SPATIAL_BINS, dtype=np.float32)
    if idx.size == 1 and idx[0] < 0:
        return geometry
    geometry[idx.astype(int)] = 1.0
    return geometry
```

iii. The notes say `blocked` is the direct dataset encoding of which 3x3 partitions are walled off, and they preferred it over reconstructing geometry from environment names.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts `blocked` indices into a length-9 binary vector with `1` for blocked partitions and `0` otherwise. For `-1` it returns an all-zero vector. That vector is copied once per surviving trial and is static within each trial.

ii.
```python
def parse_blocked(blocked_day):
    idx = np.atleast_1d(np.asarray(blocked_day, dtype=float).ravel())
    geometry = np.zeros(SPATIAL_BINS * SPATIAL_BINS, dtype=np.float32)
    if idx.size == 1 and idx[0] < 0:
        return geometry
    geometry[idx.astype(int)] = 1.0
    return geometry
...
geometry = parse_blocked(blocked_day)
...
input_trials.append(geometry.copy())
```

iii. The notes justify this as the requested “environment geometry” input, and say they verified it matches the paper’s geometry definitions and blocked occupancy structure.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from the per-session `position` array.

ii.
```python
position = dat["position"]                # (n_days, 2, T)
...
pos_binned = bin_time(position_day, n_trials)
```

iii. The notes describe `position` as x-y coordinates in cm, already aligned to the neural stream and spanning the 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first average-pools position into 500 ms bins, computes an animal-wide spatial bin size `bin_down`, converts each pooled x and y value to one of 3 bins by floor-division and clipping, combines the x/y bins into a single partition label, and finally keeps only samples surviving its movement/reachability mask.

ii.
```python
bin_down = (np.nanmax(position) + POS_BUFFER) / SPATIAL_BINS
...
pos_binned = bin_time(position_day, n_trials)
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]
...
sel = keep[t]
...
output_trials.append(partition[t, sel][None, :].astype(np.int64))
```

iii. The notes say this is meant to mirror the paper’s spatial binning logic while adapting it to the task’s 3x3 output, and they additionally remove samples in blocked partitions as tracking artifacts.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI discretizes x and y into 3 bins each and maps them to 9 categories using `category = y_bin * 3 + x_bin`.

ii.
```python
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]             # class = y_bin*3 + x_bin
```

iii. The notes justify this labeling because it matches the dataset’s own partition numbering in `blocked`, so the geometry input and position output use the same index convention.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output and neural data by applying the same trial boundaries, the same 500 ms temporal pooling, and the same per-bin keep mask to both streams.

ii.
```python
speed_binned = bin_time(speed[None, :], n_trials)[0]
pos_binned = bin_time(position_day, n_trials)
...
keep = moving & reachable
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
output_trials.append(partition[t, sel][None, :].astype(np.int64))
```

iii. The notes say the original `trace` and `position` streams are already frame-aligned, and they checked that after pooling the bin transitions and neural bins still line up without lag.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes unregistered cells stored as all-`NaN`, drops trailing partial minutes, removes bins inside physically blocked partitions as tracking artifacts, filters out low-speed bins, and discards trials/sessions with too little retained data.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
...
keep = moving & reachable
keep[0, 0] = False
...
if sel.sum() < MIN_BINS_PER_TRIAL:
    continue
...
if len(neural_trials) < MIN_TRIALS_PER_SESSION:
    return None
```

iii. The notes frame NaN columns as unregistered cells, blocked-bin samples as physically impossible tracking artifacts, and short retained trials as degenerate cases not useful for decoder training.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies loading each animal archive as the main wall-clock cost, with Gaussian filtering and per-session processing as the next-largest cost.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
t_load = time.time() - t0
...
t1 = time.time()
...
t_proc = time.time() - t1
...
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
```

iii. The notes report 6–17 s to load each animal and about 0.57–1.07 s to process each session, and explicitly say full conversion time is dominated by archive loading plus filtering.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized most heavy operations, especially temporal binning and event counting, but it still keeps a Python loop over trials when assembling variable-length per-trial outputs after masking.

ii.
```python
def bin_time(x, n_trials, axis=-1, reduce="mean"):
    ...
    out = x.reshape(shp).mean(axis=-1)
    return out
...
frame_keep = np.repeat(keep.ravel(), TEMPORAL_BIN_FRAMES)
n_events = raw[:, frame_keep].sum(axis=1)
...
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```

iii. The notes explicitly say they replaced per-trial binning loops with reshape-based vectorization and reduced cell counting to a single masked sum, but they kept trial assembly as a loop because each trial has a different retained-bin mask.

## 6-c. What processing does the code repeat multiple times?

i. The code avoids repeated file I/O by loading each animal only once, but it does repeat some small operations per retained trial, most obviously copying the same static geometry vector into every trial and iterating once per trial to slice kept bins.

ii.
```python
for t in range(n_trials):
    sel = keep[t]
    ...
    input_trials.append(geometry.copy())
    output_trials.append(partition[t, sel][None, :].astype(np.int64))
```

iii. The notes emphasize that each animal is loaded exactly once and most heavy work is vectorized, so the repeated work that remains is mostly lightweight per-trial assembly.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive diagnostics and bookkeeping that are not used by the downstream decoder, including plotting, per-session summary counters, and metadata fields such as `trial_index`, `sequence`, and retained-bin accounting.

ii.
```python
if show_processing:
    _plot_processing(session_id, raw, active, neural_binned, position_day, pos_binned,
                     xy_bin, partition, speed, speed_binned, keep, geometry, env_name,
                     bin_down, n_trials, plot_dir)
...
result = {
    ...
    "n_bins_total": int(keep.size),
    "n_bins_kept": int(keep.sum()),
    "n_bins_moving": int(moving.sum()),
    "n_bins_unreachable": int((moving & ~reachable).sum()),
    "trial_index": trial_index,
    "env": env_name,
    "geometry": geometry,
    "bin_down": bin_down,
}
```

iii. The notes say these extra diagnostics were added for sanity checking and validation, not because the decoder requires them.
