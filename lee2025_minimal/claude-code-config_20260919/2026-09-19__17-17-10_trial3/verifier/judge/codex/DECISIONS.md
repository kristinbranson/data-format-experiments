# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads per-animal joblib files from `/app/data` rather than opening the `.mat` files directly. It uses a hard-coded list of seven animal IDs, opens each `/app/data/<animal>` file with `joblib.load`, then extracts the `trace`, `position`, `envs`, and `blocked` arrays before iterating through every day/session and later splitting each session into trials.

ii.
```python
DATA_DIR = "/app/data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for subject, animal in enumerate(animals):
    print(f"Loading {animal} ...", flush=True)
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace, position = dat["trace"], dat["position"]
    envs = [str(e) for e in np.asarray(dat["envs"]).ravel()]
    blocked_all = dat["blocked"]
    n_days = trace.shape[0]
```

iii. In the trajectory, the agent inspected the repository README and `utils.load_dat`/`mat2joblib`, saw that the paper code already supports joblib copies of the dataset, and then probed a joblib file to confirm its structure. In its final summary it explicitly justified this as using files "identical content to the `.mat` files, produced by the paper's own `mat2joblib`."

## 1-b. How are the data split into subjects (mice)?

i. Each named animal in the hard-coded `ANIMALS` list is treated as one mouse/subject. The loop index from `enumerate(animals)` becomes the stored subject index for that animal's sessions.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

for subject, animal in enumerate(animals):
    ...
    subject_idx.append(subject)

data = {
    ...
    "subjects": list(animals),
    "subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. In the trajectory, the agent enumerated the seven animal files, checked their session counts, and treated each per-animal file as the natural mouse boundary.

## 1-c. How are the data split into sessions?

i. Sessions are taken from the first axis of the per-animal arrays. `trace.shape[0]` defines the number of recording days, and each `day` becomes one session in the output.

ii.
```python
n_days = trace.shape[0]

days = range(n_days) if max_sessions is None else range(min(n_days, max_sessions))
for day in days:
    blocked = blocked_indices(blocked_all[day])
    ...
    neural_all.append(neural_trials)
    input_all.append(input_trials)
    output_all.append(output_trials)
```

iii. The trajectory shows the agent inspecting the shapes of the joblib arrays and confirming counts like 31 days for most animals and 21 for `QLAK-CA1-51`, then carrying those day-level slices through as sessions.

## 1-d. How are the data split into trials?

i. Each continuous recording session is divided into consecutive non-overlapping 1-minute trials after temporal pooling. Because the AI rebins to 100 ms, each trial is `600` pooled bins, and any trailing partial minute is dropped.

ii.
```python
FPS = 30
FRAMES_PER_BIN = 3
TRIAL_SECONDS = 60.0
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FPS / FRAMES_PER_BIN))

n_bins = min(rates.shape[1], part.shape[0])
n_trials = n_bins // BINS_PER_TRIAL

for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. In its final summary, the agent stated that each session is a single continuous 40-minute foraging session with no native trial structure, so it cut sessions into consecutive 1-minute segments and dropped any trailing partial minute.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a dedicated trial-quality filter. The only trial-level exclusion is dropping the trailing incomplete minute when the session length is not an exact multiple of the trial size. It explicitly considered, then rejected, filtering to running periods.

ii.
```python
* Running speed.  The paper's Bayesian decoder additionally restricts fitting and testing to
  frames faster than 5 cm/s.  That criterion belongs to their decoding analysis rather than
  to the preparation of the dataset, and applying it here would delete ~48% of the recording
  and leave ragged, non-contiguous 1 min trials, so all time points are kept and the choice
  of whether to condition on running is left to the analysis.

n_trials = n_bins // BINS_PER_TRIAL
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
```

iii. The trajectory shows the agent testing a velocity-filtered variant on a small subset, then deciding not to use it because it would remove a large fraction of the recording and create irregular trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the `trace` array in the per-animal dataset. It interprets that array as binarized rising-phase calcium-transient events, with NaNs marking cells not registered on a given day.

ii.
```python
trace    : (n_days, n_cells, n_frames) binarised rising phase of calcium transients,
           NaN for a cell that was not registered on that day.

dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
trace, position = dat["trace"], dat["position"]

tr = trace[day]
```

iii. The agent read both `methods.txt` and the repository README, which describe the trace as the binarized rising phase of calcium transients and state that this binary vector is treated as firing rate in the paper's analyses. The final summary repeats that rationale.

## 2-b. How is the `neural` data processed?

i. The AI applies substantial additional neural processing. For each session it removes NaN cells and low-event cells, smooths each cell's binary trace along time with a Gaussian (`sigma=3` frames), average-pools over non-overlapping 3-frame windows, scales the pooled values by `FPS`, and stores the result as `float32` event rates.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered]
active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
tr = tr[active]

tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
```

iii. In the trajectory, the agent inspected `utils.fit_decoder` and `utils.test_decoder`, saw that those functions Gaussian-smooth and temporally pool traces before decoding, and decided to mirror that preprocessing. In the final summary it justified this as matching the paper's decoder and converting the stored quantity into an event rate in Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neuron filters. First, it removes cells that are unregistered on the day, detected by NaNs. Second, it removes cells with `5` or fewer events in the session using `CELL_EVENT_THRESHOLD = 5`.

ii.
```python
CELL_EVENT_THRESHOLD = 5

tr = trace[day]
registered = ~np.isnan(tr[:, 0])
tr = tr[registered]
n_registered = int(registered.sum())

active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
tr = tr[active]
```

iii. The trajectory shows the agent checking that the NaN pattern was all-or-none per cell/day and computing how many low-event cells each animal had. Its final summary explicitly cites the paper decoder's `cell_threshold = 5` sparsity criterion as the reason for the second filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not perform event-triggered alignment. It treats the recordings as already synchronized continuous 30 Hz streams, pools neural and behavior into the same 3-frame windows, and defines each trial by slicing from recording onset into consecutive 1-minute chunks.

ii.
```python
* Temporal alignment.  The two streams are acquired by the same DAQ at 30 Hz and are already
  frame-aligned in the released arrays (trace and position have identical frame counts), so
  position is pooled over the *same* 3-frame windows as the neural data and no further
  alignment is needed.

tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
pos = pool_mean(position[day], FRAMES_PER_BIN)

"temporal_alignment_event":
    "start of each 1-min trial; the session is a single continuous recording with "
    "no trial structure, cut into consecutive 1-min segments from recording onset",
```

iii. The justification came from the methods text and final summary: the agent noted that neural and behavioral streams were acquired by the same DAQ at 30 Hz and were already frame-matched in the released arrays, so no additional alignment step was needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. The AI rebins the original 30 Hz data by averaging every 3 frames after smoothing, so there is explicit temporal rebinning.

ii.
```python
FPS = 30
FRAMES_PER_BIN = 3
TIME_BIN_MS = 1000.0 * FRAMES_PER_BIN / FPS

def pool_mean(x, factor):
    n = (x.shape[-1] // factor) * factor
    return x[..., :n].reshape(*x.shape[:-1], n // factor, factor).mean(axis=-1)

rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)

"time_bin_size": float(TIME_BIN_MS),
```

iii. The agent explicitly tied this to the paper's decoder code in the trajectory and final summary, stating that `temporal_bin_size=3` at 30 Hz yields 100 ms bins.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the session's `blocked` field, not from the `envs` label.

ii.
```python
blocked  : per day, the indices of the occluded partitions in the 3x3 layout
           [[0, 1, 2], [3, 4, 5], [6, 7, 8]], or -1 when nothing is blocked.

blocked_all = dat["blocked"]
...
blocked = blocked_indices(blocked_all[day])
geom = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
geom[blocked] = 1.0
```

iii. In the trajectory, the agent inspected both `get_env_mat(env)` and the actual `blocked` contents, then justified using `blocked` because mirrored environments existed for some mice and `blocked` captured the actual partition configuration used in that session.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI converts the blocked-partition list into a 9-dimensional binary vector. `-1` means nothing is blocked, so all entries stay zero. The resulting vector is static within a session and copied once per trial.

ii.
```python
def blocked_indices(blocked_entry):
    vals = np.atleast_1d(np.asarray(blocked_entry[0], dtype=float)).ravel()
    vals = vals[vals >= 0]
    return vals.astype(int)

geom = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
geom[blocked] = 1.0

for t in range(n_trials):
    ...
    input_trials.append(geom.copy())
```

iii. The trajectory shows the agent examining example `blocked` values and then describing the final input as a static 9-dimensional binary geometry vector repeated across all trials of the session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position is derived from the `position` array in the dataset, which stores 2D head position over time.

ii.
```python
position : (n_days, 2, n_frames) head position from DeepLabCut, in cm, 0-75 cm in both
           x and y, already in a common frame across days.

trace, position = dat["trace"], dat["position"]
...
pos = pool_mean(position[day], FRAMES_PER_BIN)
```

iii. The trajectory shows the agent reading the README and probing the joblib shapes and ranges, confirming that `position` is a `(n_days, 2, n_frames)` array spanning `0` to `75` cm.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first average-pools position over the same 3-frame windows used for neural data. It then computes spatial bins by floor-dividing by an animal-wide `bin_size`, combines the x/y bins into a single class index, and finally reassigns any rare samples that landed in blocked partitions to the nearest open partition.

ii.
```python
bin_size = (np.nanmax(position) + POSITION_BUFFER) / N_SPATIAL_BINS

pos = pool_mean(position[day], FRAMES_PER_BIN)
xb = np.clip((pos[0] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
part = N_SPATIAL_BINS * yb + xb

stray = np.isin(part, blocked)
if stray.any():
    d = np.linalg.norm(_PART_CENTRES[part[stray]][:, None, :]
                       - _PART_CENTRES[open_parts][None, :, :], axis=2)
    part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. In the trajectory, the agent inspected `decode_position_within`, checked actual occupancy of blocked partitions, and then justified the snapping step as a way to clean up rare tracking-noise samples near partition walls while keeping the paper's partition indexing.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Position is thresholded into a 3-by-3 categorical grid. The AI uses floor-division by `bin_size` for both axes, clips each axis to `0..2`, and combines them as `3 * y_bin + x_bin`. After that, blocked-partition assignments are remapped to the nearest open category.

ii.
```python
xb = np.clip((pos[0] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
yb = np.clip((pos[1] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
part = N_SPATIAL_BINS * yb + xb

stray = np.isin(part, blocked)
if stray.any():
    d = np.linalg.norm(_PART_CENTRES[part[stray]][:, None, :]
                       - _PART_CENTRES[open_parts][None, :, :], axis=2)
    part[stray] = open_parts[np.argmin(d, axis=1)]
```

iii. The agent justified `3 * y + x` in the trajectory by empirically checking that this indexing agreed with the `blocked` field and led to almost no occupancy in blocked partitions before cleanup.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output and neural data by processing both streams with the same 3-frame temporal pooling, truncating them to a shared number of pooled bins, and then slicing both with identical 1-minute windows.

ii.
```python
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)

pos = pool_mean(position[day], FRAMES_PER_BIN)
...
n_bins = min(rates.shape[1], part.shape[0])

for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. The trajectory justification was explicit: because both data streams are already frame-aligned in the source data, the agent pooled them over the same windows and then cut matching trial slices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI removes cells with NaNs, interprets those NaNs as "not registered this day," drops near-silent cells, discards trailing partial-minute leftovers, and treats rare occupancy of blocked partitions as tracking noise by snapping those samples to the nearest open partition. It does not do position imputation because it found no position NaNs.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered]
active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
tr = tr[active]

stray = np.isin(part, blocked)
if stray.any():
    d = np.linalg.norm(_PART_CENTRES[part[stray]][:, None, :]
                       - _PART_CENTRES[open_parts][None, :, :], axis=2)
    part[stray] = open_parts[np.argmin(d, axis=1)]

n_trials = n_bins // BINS_PER_TRIAL
```

iii. In the trajectory, the agent explicitly checked that position had no NaNs, verified that neuron NaNs were all-or-none by cell/day, and measured that occupancy in blocked partitions was extremely rare before deciding to clean those samples rather than keep them.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work in this script is reading the large per-animal joblib arrays, smoothing the full-session neural traces with `gaussian_filter1d`, and writing the multi-gigabyte pickle. The trial-building loop itself is comparatively lightweight.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)
...
with open(out_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory does not contain a separate runtime analysis, but it does show the agent timing a six-session test conversion and then reporting that the full run wrote a 6.65 GB file, which is consistent with those being the dominant costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial assembly loop could have been vectorized or replaced with a reshape/split-based construction, because it repeatedly slices the same arrays one trial at a time and appends Python lists. The subject/day loops are structural, but the inner trial loop is the clearest vectorization target.

ii.
```python
neural_trials, input_trials, output_trials = [], [], []
for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])
```

iii. The agent did not explicitly discuss vectorization in the trajectory. This is a reconstruction from the final code structure.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats per-trial copying and slicing for every session: the same static geometry vector is copied for each trial, and the neural/output arrays are repeatedly sliced trial-by-trial even though trial boundaries are regular. It also repeats per-session metadata bookkeeping and logging.

ii.
```python
geom = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
geom[blocked] = 1.0

for t in range(n_trials):
    sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
    neural_trials.append(np.ascontiguousarray(rates[:, sl]))
    input_trials.append(geom.copy())
    output_trials.append(part[sl].astype(np.int64)[None, :])

session_info.append({
    "subject": animal,
    "day": int(day),
    "environment": envs[day],
    "blocked_partitions": [int(b) for b in blocked],
    "n_neurons": int(rates.shape[0]),
    "n_neurons_registered": n_registered,
    "n_trials": int(n_trials),
})
```

iii. The trajectory does not give an explicit justification here. The repetition is visible directly in the code.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script builds and stores rich session metadata and environment labels that the downstream decoder does not use. It also spends work on human-readable naming and reporting that are irrelevant to decoder training. Those extras are retained in the pickle or printed, but they are discarded by the actual downstream analysis path.

ii.
```python
session_info.append({
    "subject": animal,
    "day": int(day),
    "environment": envs[day],
    "blocked_partitions": [int(b) for b in blocked],
    "n_neurons": int(rates.shape[0]),
    "n_neurons_registered": n_registered,
    "n_trials": int(n_trials),
})

"environments": sorted({s["environment"] for s in session_info}),
"session_info": session_info,

print(f"  day {day:2d} {envs[day]:>10s}  cells {rates.shape[0]:4d}"
      f" (registered {n_registered:4d})  trials {n_trials}", flush=True)
```

iii. The trajectory suggests these additions were deliberate documentation and bookkeeping choices rather than requirements of the decoder format. The agent emphasized detailed metadata and reporting in its final summary.
