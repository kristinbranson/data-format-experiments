# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses a fixed seven-animal list and loads each animal's extensionless joblib archive from `/app/data` once. Each archive contains all days, traces, positions, environments, and blocked-partition records. Full mode iterates all seven animals; sample mode selects two days.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace = dat["trace"]
position = dat["position"]
envs = np.asarray(dat["envs"]).ravel()
blocked = dat["blocked"]
```

iii. The notes say the joblib files are lossless conversions of the `.mat` files, were spot-checked against them, and load much faster. Loading one whole animal at a time limits peak memory and avoids repeated per-session I/O.

## 1-b. How are the data split into subjects?

i. Each named joblib archive is one mouse. The final subject list follows the fixed animal order, and each retained session receives the corresponding subject index.

ii.
```python
plan = OrderedDict((a, None) for a in ANIMALS)
subjects = list(total_cells.keys())
subj_index = {s: i for i, s in enumerate(subjects)}
"subject_idx": np.array([subj_index[s["animal"]] for s in sessions], dtype=np.int64),
```

iii. The notes identify exactly seven mice and verify their per-animal cell totals and the paper's total of 5,413 unique cells.

## 1-c. How are the data split into sessions?

i. One animal-day (one row along the archive's day axis) becomes one output session. All days are iterated unless sample mode requests selected day indices.

ii.
```python
n_days = trace.shape[0]
if days is None:
    days = range(n_days)
for d in days:
    res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down,
                          sid, show_processing=show_processing, plot_dir=plot_dir)
```

iii. The notes state that the paper treats an animal-day/40-minute recording as its unit of analysis and verify 207 sessions (31,31,31,21,31,31,31).

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 60-second blocks (1,800 native frames). The trailing incomplete minute is dropped. After sample filtering, trials with fewer than ten retained 500 ms bins are omitted.

ii.
```python
n_trials = n_frames // TRIAL_FRAMES
n_used = n_trials * TRIAL_FRAMES
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
```

iii. Sixty-second artificial trials are required by the task. The notes quantify trailing-frame loss and say the ten-bin minimum ensures at least five seconds of running data.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if fewer than ten 500 ms bins survive the running-speed and reachable-position mask. A session is dropped if fewer than two trials survive.

ii.
```python
if sel.sum() < MIN_BINS_PER_TRIAL:
    continue
...
if len(neural_trials) < MIN_TRIALS_PER_SESSION:
    return None
```

iii. The agent chose five seconds of retained running as a minimum and two trials because the supplied train/validation decoder requires at least two. The notes report 168 of 8,187 candidate trials removed and no full sessions removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the archive's `trace` array: the paper's binary rising-phase calcium-event signal.

ii.
```python
trace = dat["trace"]
raw = trace_day[registered][:, :n_used].astype(np.float32)
```

iii. The notes cite the paper's statement that this already processed binary vector is treated as firing rate, so raw fluorescence or dF/F is not recomputed.

## 2-b. How is the `neural` data processed?

i. Registered traces are cast to float32, Gaussian-smoothed along time with sigma 15 frames, average-pooled over non-overlapping 15-frame blocks, multiplied by 30 to express values in Hz, masked to retained samples, and stored neuron-by-time per trial.

ii.
```python
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```

iii. The agent says smooth-then-average-pool follows the paper decoder, but chose 500 ms rather than its 100 ms setting based on empirical decoder accuracy, signal sparsity, runtime, and output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells unregistered on a day are removed using NaN at the first frame. Of the remaining cells, only those with more than five raw events in retained running/reachable bins are kept. Sessions with no qualifying cell are dropped.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
...
frame_keep = np.repeat(keep.ravel(), TEMPORAL_BIN_FRAMES)
n_events = raw[:, frame_keep].sum(axis=1)
active = n_events > CELL_EVENT_THRESHOLD
if active.sum() == 0:
    return None
```

iii. The notes verify that NaNs always span a whole unregistered session and attribute the strict `>5` event rule to `decode_position_within`; no place-cell selection is performed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trial zero-time is the start of each artificial one-minute segment. Neural and behavior are binned within the same minute and receive the same `keep[t]` mask.

ii.
```python
"temporal_alignment_event":
    "start of each 1-minute trial; trials are consecutive non-overlapping 60 s "
    "segments of the continuous 40-min free-foraging session...",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The recordings are continuous free foraging with no stimulus or task event; the distributed neural and position streams are already synchronized frame-by-frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 500 ms (15 frames at 30 Hz). Gaussian smoothing with sigma 15 frames is followed by non-overlapping 15-frame average pooling.

ii.
```python
TEMPORAL_BIN_FRAMES = 15
TIME_BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS
```

iii. The notes acknowledge the paper decoder used 100 ms and justify 500 ms through small empirical comparisons showing improved or near-optimal balanced accuracy and reduced compute/storage.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Environment geometry is derived directly from each day's `blocked` field, which lists blocked physical 3×3 partitions or `-1` for the open square.

ii.
```python
blocked = dat["blocked"]
geometry = parse_blocked(blocked_day)
```

iii. The notes verified `blocked` against the environment matrices and actual occupancy for every session, resolving an orientation ambiguity in plotting code.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices are converted to a static nine-element float32 multi-hot vector, with 1 for blocked and 0 for open; `-1` yields all zeros. A copy is attached to each retained trial.

ii.
```python
geometry = np.zeros(SPATIAL_BINS * SPATIAL_BINS, dtype=np.float32)
if idx.size == 1 and idx[0] < 0:
    return geometry
geometry[idx.astype(int)] = 1.0
...
input_trials.append(geometry.copy())
```

iii. The nine dimensions correspond directly to the apparatus partitions and share indexing with output classes, making the session's blocked geometry explicit to the decoder.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output derives from each day's two-coordinate `position` array in centimeters.

ii.
```python
position = dat["position"]
res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down, ...)
```

iii. The notes describe position as synchronized x/y head position in a 75×75 cm arena with no NaNs.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is averaged in each 15-frame temporal bin, divided by an animal-wide spatial bin width, converted to integer x/y bins, clipped to 0–2, and flattened as `y*3+x`. Only running and physically reachable bins remain.

ii.
```python
pos_binned = bin_time(position_day, n_trials)
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]
...
output_trials.append(partition[t, sel][None, :].astype(np.int64))
```

iii. The agent follows the paper's average-pooling and global spatial-scale approach, changes spatial resolution to the required 3×3, and uses the dataset's blocked-index orientation.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. A single isotropic width is computed as `(maximum position across all days and axes for that animal + 1e-5)/3`, approximately 25 cm. Integer truncation/floor assigns each axis to 0, 1, or 2, clipping edge cases; class is `y_bin*3+x_bin` (0–8).

ii.
```python
bin_down = (np.nanmax(position) + POS_BUFFER) / SPATIAL_BINS
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]
```

iii. The buffer and clipping handle exact 75 cm boundary samples. The orientation was checked against blocked partitions and occupancy across all sessions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural, position, and speed use the same complete-minute boundaries and 15-frame temporal bins. The identical Boolean selection `sel = keep[t]` is applied to neural columns and position labels.

ii.
```python
sel = keep[t]
neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
output_trials.append(partition[t, sel][None, :].astype(np.int64))
```

iii. The source streams share acquisition timing, and the notes report visual and programmatic spot checks showing no lead/lag or shape mismatch.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Whole-session NaN cell traces are treated as unregistered and removed. Trailing partial minutes are discarded. Undefined initial speed is excluded; positions are clipped at spatial boundaries; bins apparently tracked inside blocked partitions are removed. Too-short trials and degenerate sessions are dropped.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
n_trials = n_frames // TRIAL_FRAMES
keep[0, 0] = False
reachable = geometry[partition] == 0
keep = moving & reachable
xy_bin = np.clip(..., 0, SPATIAL_BINS - 1)
```

iii. The notes validate whole-session NaN behavior, call blocked-position samples tracking artifacts, and give complete counts for every discarded category. Guards for empty cells, trials, and sessions were included even though no session was ultimately lost.

## 6-a. What are the most time-consuming steps of the code?

i. Decompressing/loading each large animal joblib archive is the major I/O and memory cost; per-session Gaussian filtering is the main processing cost. The full conversion took about 4.7 minutes.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
```

iii. The notes report 6–17 seconds to load an animal and roughly 0.57–1.6 seconds processing each session, noting that archives can decompress to roughly 17 GB.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial assembly remains a Python loop because each trial has a variable Boolean mask and output length. Animal/day loops are inherently per-file/per-session. Temporal binning and event counting, which could otherwise have been nested loops, were already vectorized.

ii.
```python
for t in range(n_trials):
    sel = keep[t]
    ...
out = x.reshape(shp).mean(axis=-1)
n_events = raw[:, frame_keep].sum(axis=1)
```

iii. The notes highlight reshape-based binning as avoiding about 8,000 Python trial loops for the numerical reduction. The remaining assembly loop is needed for ragged retained time axes.

## 6-c. What processing does the code repeat multiple times?

i. Speed computation, temporal binning, spatial classification, cell filtering, and smoothing are repeated once for each session; static geometry is copied once per retained trial. Data loading is not repeated within an animal.

ii.
```python
for d in days:
    res = convert_session(trace[d], position[d], blocked[d], ...)
...
input_trials.append(geometry.copy())
```

iii. The notes emphasize that each animal is loaded exactly once. Session-specific traces and positions require per-day processing, while geometry copies prevent aliasing between trial records.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes speed and several bookkeeping/diagnostic quantities (`moving`, reachable artifacts, trial indices, counts, environment and sequence metadata) that are not decoder arrays; optional plotting performs substantial visualization work. It also smooths the entire complete-session trace before selecting retained bins, so results at discarded bins are computed and then thrown away.

ii.
```python
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS
...
"n_bins_moving": int(moving.sum()),
"n_bins_unreachable": int((moving & ~reachable).sum()),
"trial_index": trial_index,
```

iii. Speed is required for the chosen quality mask and full-session smoothing avoids boundary artifacts, so these are deliberate costs. The extra statistics and plots support validation and documentation but are not consumed by decoder training.
