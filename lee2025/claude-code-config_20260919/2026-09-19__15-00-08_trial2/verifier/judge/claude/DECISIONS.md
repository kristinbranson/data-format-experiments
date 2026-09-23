# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from per-animal joblib archive files in `/app/data/`. Each file is loaded with `joblib.load()` and contains a dictionary keyed by the animal ID with fields `trace`, `position`, `envs`, `blocked`, etc. The AI hardcodes the 7 animal names in a list (`ANIMALS`) and iterates over them.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace = dat["trace"]          # (n_days, n_cells, T)
position = dat["position"]    # (n_days, 2, T)
envs = np.asarray(dat["envs"]).ravel()
blocked = dat["blocked"]
```

iii. The AI chose joblib over .mat files because, as documented in CONVERSION_NOTES.md Step 4, the joblib files are a lossless conversion of the .mat files (via the reference code's `mat2joblib`) and load much faster. The AI verified that values match between both formats.

## 1-b. How are the data split into subjects?

i. Each animal has its own joblib file. The AI iterates over a hardcoded list of 7 animal names. Each animal becomes a separate subject.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for animal, days in plan.items():
    out, ncells = convert_animal(animal, days=days, ...)
    sessions.extend(out)
    total_cells[animal] = int(ncells)
subjects = list(total_cells.keys())
```

iii. Each file corresponds to one subject (animal). The subject identifier is the animal name (e.g. "QLAK-CA1-08").

## 1-c. How are the data split into sessions?

i. Each recording day for an animal is a separate session. Within each animal's data, the `trace` array has shape `(n_days, n_cells, T)`, and each day index becomes a session.

ii.
```python
n_days = trace.shape[0]
if days is None:
    days = range(n_days)
for d in days:
    sid = f"{animal}_day{d:02d}_{str(envs[d]).replace(' ', '')}"
    res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down, sid, ...)
```

iii. This matches the paper's unit of analysis: 207 sessions across 7 animals (31 days for 6 animals, 21 for QLAK-CA1-51).

## 1-d. How are the data split into trials?

i. Each ~40-minute recording session is split into consecutive, non-overlapping 60-second segments (1800 frames at 30 Hz). The trailing partial segment is discarded. This yields 39 or 40 trials per session. After speed filtering and temporal binning, each trial contains a variable number of 500ms time bins (only bins where the mouse is running are retained).

ii.
```python
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FPS))   # 1800
n_trials = n_frames // TRIAL_FRAMES
n_used = n_trials * TRIAL_FRAMES
...
# After binning and filtering, per-trial assembly:
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```

iii. The 1-minute trial definition matches the task instructions. The variable-length trials result from the speed filter applied per-bin.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 10 retained time bins (i.e., less than 5 seconds of running) are dropped. Sessions with fewer than 2 surviving trials are also dropped (though this never occurs in practice).

ii.
```python
MIN_BINS_PER_TRIAL = 10
MIN_TRIALS_PER_SESSION = 2
...
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
...
if len(neural_trials) < MIN_TRIALS_PER_SESSION:
    return None
```

iii. The AI documented that 168 of 8,187 possible trials were dropped (2.05%) because the mouse was almost entirely immobile. This filtering is not present in the reference code but is a practical necessity given the speed filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `trace` variable, which contains binarized rising-phase calcium events with values in {0, 1}.

ii.
```python
trace = dat["trace"]   # (n_days, n_cells, T)
...
raw = trace_day[registered][:, :n_used].astype(np.float32)  # (n_reg, n_used), {0,1}
```

iii. The AI verified that `trace` is already the final binarized rising-phase signal (the paper's "firing rate"), so no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The AI applies a multi-step processing pipeline: (1) remove unregistered (NaN) cells, (2) apply gaussian smoothing with sigma = 15 frames along time, (3) average-pool into 500ms (15-frame) bins, (4) multiply by FPS (30) to convert to Hz. Only bins where the mouse is running are retained.

ii.
```python
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS  # (n_act, n_trials, BINS)
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```

iii. The processing follows the reference code's `fit_decoder` recipe: gaussian smooth then average pool. The AI changed the bin size from 3 frames (100ms) to 15 frames (500ms) with empirical justification that 500ms gives better decoder accuracy for the sparse calcium events (~0.24 Hz per cell).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) cells not registered on a given day (all-NaN traces) are removed, and (2) cells with 5 or fewer events among the retained (running) frames are removed.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
raw = trace_day[registered][:, :n_used].astype(np.float32)
...
frame_keep = np.repeat(keep.ravel(), TEMPORAL_BIN_FRAMES)
n_events = raw[:, frame_keep].sum(axis=1)
active = n_events > CELL_EVENT_THRESHOLD   # CELL_EVENT_THRESHOLD = 5
```

iii. Both filters match the reference code `decode_position_within`: unregistered cells are NaN and excluded; cells with <= 5 events during running frames are excluded via `cell_threshold=5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment. The experiment is continuous free foraging with no discrete trial structure or task events. Trials are consecutive 60-second segments starting from the beginning of the recording. `off_start = 0`, `off_end = 60`.

ii.
```python
"temporal_alignment_event":
    "start of each 1-minute trial; trials are consecutive non-overlapping 60 s "
    "segments of the continuous 40-min free-foraging session (the experiment has "
    "no discrete trial structure or task events)",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```

iii. The calcium and behavior streams are acquired by the same DAQ at 30 Hz and are already frame-aligned in the distributed data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins from 30 Hz (33.33ms per frame) to 500ms bins (15 frames). The rebinning uses gaussian smoothing (sigma=15 frames) followed by average pooling over 15-frame blocks.

ii.
```python
TEMPORAL_BIN_FRAMES = 15   # 500 ms
TIME_BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0  # 500.0
...
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS
```

iii. The reference code uses 3-frame (100ms) bins. The AI chose 500ms based on empirical testing showing it gives optimal decoder accuracy for sparse calcium events. This is documented with a comparison table in CONVERSION_NOTES.md Step 5.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the `blocked` field in each animal's data, which lists the indices of blocked (walled-off) partitions in the 3x3 grid for each session.

ii.
```python
blocked = dat["blocked"]
...
blk_indices = blocked[d]
geometry = parse_blocked(blocked_day)
```

iii. The `blocked` field directly encodes which of the 9 partitions are inaccessible, matching the "environment geometry" decoder input specification.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted to a 9-dimensional binary vector where 1 indicates a blocked partition and 0 indicates an open partition. If the session has `blocked = -1` (the "square" geometry with nothing blocked), the vector is all zeros. The input is static per trial (same for all trials within a session).

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
input_trials.append(geometry.copy())
```

iii. The AI verified that this encoding matches `flipud(get_env_mat(env)) == 0` for all 207 sessions, confirming consistency with the reference code's geometry representation.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output is derived from the `position` variable, which contains 2D (x, y) coordinates of the mouse in the arena at each timepoint.

ii.
```python
position = dat["position"]   # (n_days, 2, T)
...
pos_binned = bin_time(position_day, n_trials)  # (2, n_trials, BINS)
```

iii. The position is recorded at 30 Hz in centimeters within the 75x75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is first temporally binned (average-pooled into 500ms bins), then spatially discretized into a 3x3 grid. The spatial bin size is computed per-animal as `bin_down = (max_position + 1e-5) / 3`. Each axis is binned with `floor(position / bin_down)` clipped to [0, 2]. The partition index is `y_bin * 3 + x_bin`.

ii.
```python
bin_down = (np.nanmax(position) + POS_BUFFER) / SPATIAL_BINS
...
pos_binned = bin_time(position_day, n_trials)
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]   # class = y_bin*3 + x_bin
```

iii. The spatial binning follows `decode_position_within`'s `bin_down` rule exactly, adapted from 15 bins to 3. The clip guards the edge case where position = 75.0 cm exactly.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is discretized into 9 categories (3x3 grid). Each axis divided into 3 equal bins of ~25 cm. The category is computed as `y_bin * 3 + x_bin`, giving values 0-8.

ii.
```python
SPATIAL_BINS = 3
bin_down = (np.nanmax(position) + POS_BUFFER) / SPATIAL_BINS  # ~25.0 cm
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]
```

iii. The 3x3 grid matches the physical partition design of the arena (25 cm partitions). The AI chose `y_bin * 3 + x_bin` ordering to match the dataset's `blocked` partition numbering, which differs from the reference code's `x_bin * n_bins + y_bin` but is a pure relabeling that doesn't affect accuracy.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data are from the same DAQ at 30 Hz and are frame-aligned in the raw data. Both undergo the same temporal binning (average pooling over 15-frame windows) and are split into trials at the same boundaries. The same speed-based sample mask is applied to both.

ii.
```python
pos_binned = bin_time(position_day, n_trials)
...
neural_binned = bin_time(smoothed, n_trials) * FPS
...
# Same mask applied to both:
neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
output_trials.append(partition[t, sel][None, :].astype(np.int64))
```

iii. Alignment is maintained because both streams share the same time axis and are processed with the same trial boundaries and sample selection mask.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of data issues are handled: (1) NaN neurons (unregistered cells) are removed per session. (2) Trailing frames that don't fill a complete 60s trial are discarded. (3) Tracking artifacts — 28 time bins (443 raw frames) where the mouse was tracked inside a physically blocked partition are removed via a "reachable" mask. (4) Frame 0 is excluded since it has no defined speed.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
...
reachable = geometry[partition] == 0
keep = moving & reachable
keep[0, 0] = False   # frame/bin 0 has no defined speed
```

iii. The AI documented that 428 of the 443 offending raw frames come from a single session (CA1-51 day 4), likely DeepLabCut tracking errors. Removing these prevents unlearnable label noise.

## 6-a. What are the most time-consuming steps of the code?

i. Loading the animal joblib archives is the most time-consuming step (6-17 seconds per animal, ~100s total). Per-session processing takes 0.57-1.07s per session. Total conversion time is ~4.7 minutes.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
t_load = time.time() - t0
```

iii. The AI measured and documented timing for each step. Loading dominates because each animal's data decompresses ~17 GB of float64 arrays.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial assembly loop iterates over trials to apply the speed mask and assemble variable-length arrays. This cannot easily be vectorized because trials have different numbers of retained bins after filtering.

ii.
```python
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```

iii. The AI vectorized most operations (binning via reshape-and-mean, cell counting via boolean-mask sum) and noted this in CONVERSION_NOTES.md.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each animal is loaded once, each session is processed once, and each stream (neural, position, speed) is computed once per session.

ii. N/A

iii. The AI explicitly designed the code to avoid redundant computation: "Loading a whole animal decompresses ~17 GB of float64; only one animal is held at a time and each is loaded exactly once."

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes several bookkeeping statistics per session (`n_bins_total`, `n_bins_kept`, `n_bins_unreachable`, `trial_index`, `env`, `bin_down`, `n_registered`, etc.) that are stored in intermediate dictionaries but not all end up in the final pickle. The `session_info` metadata field stores detailed per-session information. The gaussian smoothing is applied to all cells before the activity filter could further reduce them, though the filter removes very few cells (1.2%).

ii.
```python
result = {
    "neural": neural_trials,
    ...
    "n_bins_total": int(keep.size),
    "n_bins_kept": int(keep.sum()),
    "n_bins_moving": int(moving.sum()),
    "n_bins_unreachable": int((moving & ~reachable).sum()),
    "trial_index": trial_index,
    "env": env_name,
    ...
}
```

iii. The extra bookkeeping is used for the conversion summary statistics and sanity checks, and some is stored in metadata for downstream interpretability.
