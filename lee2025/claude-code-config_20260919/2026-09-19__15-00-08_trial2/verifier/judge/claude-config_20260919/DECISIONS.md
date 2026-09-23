# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads one **joblib archive per animal** from `/app/data/<animal>` (the `QLAK-CA1-XX`
files, not the `.mat` twins), exactly as the reference code's `load_dat(..., format="joblib")`
does. The hard-coded animal list (7 IDs) is copied from the reference `main.py`. Each archive is
a nested dict `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}`; the AI pulls
out `trace` (n_days, n_cells, T), `position` (n_days, 2, T), `envs` (n_days,) and `blocked`
(n_days). The whole animal is held in memory once, all of its days are converted, and then the
arrays are deleted before the next animal is loaded (one file read per animal, no per-session
I/O). Sessions are accumulated into flat lists across animals and assembled at the end.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]          # main.py

def convert_animal(animal, days=None, show_processing=False, plot_dir="."):
    """Load one animal's joblib file and convert the requested days."""
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0

    trace = dat["trace"]                      # (n_days, n_cells, T)
    position = dat["position"]                # (n_days, 2, T)
    envs = np.asarray(dat["envs"]).ravel()
    blocked = dat["blocked"]
    n_days = trace.shape[0]
    ...
    for d in days:
        sid = f"{animal}_day{d:02d}_{str(envs[d]).replace(' ', '')}"
        res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down, sid, ...)
    ...
    del dat, trace, position
```
```python
plan = OrderedDict((a, None) for a in ANIMALS)   # --full
for animal, days in plan.items():
    out, ncells = convert_animal(animal, days=days, ...)
    sessions.extend(out)
    total_cells[animal] = int(ncells)
```

iii. From CONVERSION_NOTES Step 4/Step 1: the reference `mat2joblib`/`save_dat` shows the joblib
files are a lossless conversion of the `.mat` files, and `load_dat` defaults to `format="joblib"`,
so using joblib is what the original authors' own pipeline does and it loads much faster. The AI
did not take this on trust — sanity check [5] reopened `QLAK-CA1-51.mat` through `h5py` and
confirmed `position` and `trace` for day 7 are identical to the joblib arrays. Counts after
loading (7 subjects, 207 sessions, 5,413 unique cells, 69,744 registered cell-sessions) reproduce
the paper's headline numbers exactly.

## 1-b. How are the data split into subjects?

i. One subject per animal file / animal ID. Seven subjects, taken from the explicit `ANIMALS`
list copied from the reference `main.py`, in that (sorted) order. `subjects` holds the 7 ID
strings and `subject_idx` maps each session to its animal's index.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]          # main.py
...
res["animal"] = animal
...
subjects = list(total_cells.keys())
...
"subjects": list(subjects),
"subject_idx": np.array([subj_index[s["animal"]] for s in sessions], dtype=np.int64),
```

iii. Each data file is one mouse; the reference code iterates over exactly this animal list. The
AI cross-checked the resulting per-animal cell counts (515, 875, 942, 554, 862, 713, 952) against
the paper's "mean number of cells per animal = 773 ± 68 SE, minimum 515, maximum 952" and against
5,413/7 = 773.3 (CONVERSION_NOTES Step 3/Step 9).

## 1-c. How are the data split into sessions?

i. One session = one **animal-day** (one 40-min recording), which is the paper's own unit of
analysis. The leading axis of `trace`/`position`/`envs`/`blocked` is the day index, so the AI
simply loops `for d in range(n_days)`. This gives 31 sessions for six animals and 21 for
QLAK-CA1-51 = **207 sessions**, matching the paper. Neuron identity/count differs per session
because cell registration differs by day; each session therefore carries its own neuron count and
its own `brain_region_idx`.

ii.
```python
n_days = trace.shape[0]
...
if days is None:
    days = range(n_days)
for d in days:
    sid = f"{animal}_day{d:02d}_{str(envs[d]).replace(' ', '')}"
    res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down, sid, ...)
    ...
    res["day"] = int(d)
    res["session_id"] = sid
    res["sequence"] = int(min(d // 10, max(n_days // 10 - 1, 0)) + 1)
```

iii. CONVERSION_NOTES Step 5 decision 1: "Session = one animal-day (40-min recording). 207
sessions, matching the paper's unit of analysis. Different days have different registered cells,
so neuron count varies per session (allowed by the format)." The geometry (`envs`/`blocked`) is a
per-day property, so a session is also the natural unit for the decoder input. Per-session
metadata (geometry name, day, sequence number, blocked partitions, spatial bin size) is written
into `metadata['session_info']`.

## 1-d. How are the data split into trials?

i. Trials are consecutive, **non-overlapping 60-s segments** of the continuous recording:
1800 frames at 30 Hz, i.e. 120 bins of 500 ms. `n_trials = n_frames // 1800`, and the trailing
partial minute is dropped. Recordings are 71,866–72,219 frames, so this gives 40 trials/session
for five animals and 39 for QLAK-CA1-30 and -50 (8,187 possible trials). All three streams
(neural, position/output, speed) are cut on the same frame index by the same `bin_time` reshape,
so trials are identical across streams by construction.

ii.
```python
TRIAL_SECONDS = 60.0       # decoder task: 1-minute trials
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FPS))          # 1800
TEMPORAL_BIN_FRAMES = 15   # 500 ms
BINS_PER_TRIAL = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES    # 120

def bin_time(x, n_trials, axis=-1, reduce="mean"):
    x = np.moveaxis(x, axis, -1)
    x = x[..., : n_trials * TRIAL_FRAMES]
    shp = x.shape[:-1] + (n_trials, BINS_PER_TRIAL, TEMPORAL_BIN_FRAMES)
    out = x.reshape(shp).mean(axis=-1)
    return out
```
```python
n_cells_total, n_frames = trace_day.shape
n_trials = n_frames // TRIAL_FRAMES          # complete 1-minute trials only
if n_trials < MIN_TRIALS_PER_SESSION:
    return None
n_used = n_trials * TRIAL_FRAMES
```

iii. CONVERSION_NOTES Step 5 decision 2: the experiment has no trial structure (40 min of
continuous free foraging), so trials are the artificial 1-minute segments the decoder task
prescribes. The trailing partial segment is dropped "so that every trial is a full minute";
the AI quantified the loss (1,666 frames ≈ 55 s for animals 30 and 50 = 2.3 %, ≤ 219 frames
< 0.3 % for the other five; 98.88 % of all frames land in complete trials). Step 5 decision 11
records that there is no alignment event, so `off_start = 0`, `off_end = 60`.

## 1-e. How are trials filtered based on quality controls?

i. Three layers of curation touch trials:
1. **Within-trial sample curation** — a 500-ms bin is kept only if (a) the mean smoothed running
   speed in that bin is > 5 cm/s (the reference `decode_position_within` `v_thresh=5` criterion)
   and (b) the mouse is not tracked inside a partition that is physically walled off on that
   session. Bin 0 of the session is always dropped because frame 0 has no defined speed
   (matching the reference, where `vel_idx[0]` stays False). Trials therefore have **variable
   length** (mean 65.9 of 120 bins; 53.8 % of bins retained).
2. **Trial dropping** — a trial with fewer than 10 retained bins (5 s of running) is discarded:
   168 of 8,187 trials (2.05 %), leaving **8,019 trials**.
3. **Session dropping** — a session is kept only if ≥ 2 trials survive (needed for the
   train/validation split) and at least one cell survives the activity filter. **0 of 207
   sessions were actually dropped.**

ii.
```python
MIN_BINS_PER_TRIAL = 10    # 5 s of running; trials with less are dropped
MIN_TRIALS_PER_SESSION = 2 # train_validate_decoder needs >= 2 trials to split a session
...
moving = speed_binned > V_THRESH
reachable = geometry[partition] == 0
keep = moving & reachable
keep[0, 0] = False          # frame/bin 0 has no defined speed (reference excludes it)
...
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
    input_trials.append(geometry.copy())
    output_trials.append(partition[t, sel][None, :].astype(np.int64))
    trial_index.append(t)

if len(neural_trials) < MIN_TRIALS_PER_SESSION:
    return None
```

iii. CONVERSION_NOTES Step 5 decisions 5 and 10, and Step 10 Check 4. The speed filter is taken
verbatim from the paper's own decoding function (`v_thresh=5`, speed smoothed with sigma = 5
frames); the AI also ran `cache/speed_test.py` and found the filter *improves* 3×3 balanced
accuracy (0.652 → 0.757 on animal 51 day 18) "because position coding is unreliable during
immobility". The AI deliberately applies the filter *after* binning so a bin never averages
frames across a temporal gap, calling this "a strict improvement over the reference, which pools
after filtering and therefore mixes non-contiguous frames". The unreachable-partition rule removes
28 bins (0.003 %) of DeepLabCut tracking artifacts that "cannot happen physically" and would be
"unlearnable label noise". The <10-bin trial rule removes minutes in which the mouse was almost
entirely immobile (mean 2.8 running bins each); the ≥2-trial session rule is driven by the
provided decoder's per-session train/validation split.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely the `trace` field, `(n_days, n_cells, T)`, which is the paper's **binarized
rising-phase calcium event** signal (values exactly {0,1}, NaN for a whole day when a cell was
not registered that day). No other neural field (`SFPs`, `centroids`, `maps`) is used.

ii.
```python
trace = dat["trace"]                      # (n_days, n_cells, T)
...
res = convert_session(trace[d], position[d], blocked[d], ...)
...
# trace_day: (n_cells_total, T) binarized calcium events, NaN for cells that were
#            not registered on this day.
registered = ~np.isnan(trace_day[:, 0])
raw = trace_day[registered][:, :n_used].astype(np.float32)   # (n_reg, n_used), {0,1}
```

iii. CONVERSION_NOTES Step 1/Step 5 decision 3: "No dF/F is needed: the pipeline in the paper
already produced this binary event vector and 'treated [it] as the firing rate in all further
analyses'" (methods.txt). The AI verified the values are exactly {0,1} and that the reference's
`get_rate_maps`/`fit_decoder` consume this same `trace` array directly.

## 2-b. How is the `neural` data processed?

i. Pipeline, following the reference decoder front-end `fit_decoder`:
1. Drop unregistered (all-NaN) cells and truncate to complete trials.
2. Drop cells with ≤ 5 events among the retained (running) frames (see 2-c).
3. `gaussian_filter1d` along time with **sigma = 15 frames (= one 500 ms bin)** over the whole
   session.
4. Average-pool non-overlapping 15-frame blocks (`reshape(...).mean(-1)`), giving (n_cells,
   n_trials, 120).
5. Multiply by FPS = 30 to express the result as an **event rate in Hz** (as `get_rate_maps`
   does). Mean value in the converted data is 0.298 Hz, consistent with the 0.21–0.27 Hz raw
   per-cell event rates.
6. Select only the retained (running, reachable) bins of each trial; cast to contiguous float32.

ii.
```python
# ---- 7. temporal binning of the neural data --------------------------------------
#   Reference recipe: gaussian smooth along time with sigma = bin size, then average
#   pool.  Expressed in Hz (x FPS) as in get_rate_maps.
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS           # (n_act, n_trials, BINS)
del smoothed
...
neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```
Reference being mirrored (`utils.py:fit_decoder`):
```python
pooling = AvgPool1d(kernel_size=temporal_bin_size, stride=temporal_bin_size)
behav, traces = pooling(torch.tensor(behav.T)).numpy().astype(int).T, \
    pooling(torch.tensor(gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0).T)).numpy().T
```

iii. CONVERSION_NOTES Step 5 decision 4 and Step 10 Check 3: the smooth-then-average-pool recipe
is copied from `fit_decoder`; the only change is the kernel width (15 frames instead of 3, see
2-e). The ×FPS scaling makes the values interpretable as Hz, matching `get_rate_maps`. The AI kept
the native `(n_cells, T)` memory layout rather than the reference's `(T, n_cells)` "because
filtering `axis=-1` is ~15 % faster and avoids a transpose copy", and used float32 throughout.
Sanity check [1] independently recomputed *every* neural value of 4 randomly chosen sessions
straight from the raw joblib files and matched with `np.allclose(atol=1e-6)`, plus named
spot-checks (e.g. `QLAK-CA1-74_day28_glenn` `neural[trial=5, neuron=6, t=20] = 26.283426 Hz`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both taken from the reference:
1. **Registration filter** — cells not registered on that day are stored as all-NaN and are
   dropped (`~np.isnan(trace_day[:, 0])`). 69,744 registered cell-sessions survive, exactly the
   paper's "69,744 rate maps".
2. **Activity filter** — of the registered cells, those with **≤ 5 events among the retained
   (running) frames** are dropped, i.e. `decode_position_within`'s `cell_threshold=5`. This
   removes 833 cell-sessions (1.19 %), leaving 68,911 neurons (mean 332.9/session, min 111,
   max 563). If no cell survives, the session returns `None` (never triggered).
3. **No place-cell selection** is applied.

ii.
```python
CELL_EVENT_THRESHOLD = 5   # decode_position_within(cell_threshold=5) events
...
registered = ~np.isnan(trace_day[:, 0])
raw = trace_day[registered][:, :n_used].astype(np.float32)   # (n_reg, n_used), {0,1}
...
# ---- 6. cell activity filter ------------------------------------------------------
#   decode_position_within: cells with more than `cell_threshold` events among the
#   retained samples.  Counted on the raw binary events of the retained frames.
frame_keep = np.repeat(keep.ravel(), TEMPORAL_BIN_FRAMES)
n_events = raw[:, frame_keep].sum(axis=1)
active = n_events > CELL_EVENT_THRESHOLD
if active.sum() == 0:
    return None
```
Reference being mirrored (`utils.py:decode_position_within`):
```python
cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold
```

iii. CONVERSION_NOTES Step 3 "Neuron curation rules" and Step 5 decision 6. The AI notes the
paper explicitly "motivated the inclusion of all cells in subsequent analyses", so place-cell
status (`get_place_cells`) is deliberately *not* used to select cells — only the two criteria the
paper's own decoding function applies. The AI verified that NaNs are always whole-session
(`isnan(trace).all(axis=time) == isnan(trace[:, :, 0])`), which is what makes the cheap
first-frame test valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is **no experimental alignment event** — the recording is 40 min of continuous free
foraging. Trial *t* is defined to start at `t × 60 s` from the start of the recording, and the
neural, position and speed streams are cut on the same frame index by the same reshape, so they
are aligned frame-for-frame by construction. The DAQ acquired behaviour and imaging at 30 Hz with
hardware timestamps, and the distributed `trace` and `position` arrays already share one frame
index (identical T within a day), so no resampling or shifting is needed. Metadata records
`off_start = 0.0`, `off_end = 60.0`.

ii.
```python
"temporal_alignment_event":
    "start of each 1-minute trial; trials are consecutive non-overlapping 60 s "
    "segments of the continuous 40-min free-foraging session (the experiment has "
    "no discrete trial structure or task events)",
"off_start": 0.0,
"off_end": float(TRIAL_SECONDS),
```
```python
speed_binned = bin_time(speed[None, :], n_trials)[0]         # (n_trials, BINS_PER_TRIAL)
pos_binned   = bin_time(position_day, n_trials)              # (2, n_trials, BINS)
neural_binned = bin_time(smoothed, n_trials) * FPS           # (n_act, n_trials, BINS)
```

iii. CONVERSION_NOTES Step 3 "Temporal alignment" and Step 5 decision 11. The AI verified
alignment visually in `--show-processing` panel 2: the derived x/y bin index steps "exactly where
the position trace crosses a border. No lead or lag ⇒ no temporal misalignment", and in panel
4/5 "every raster tick lines up with a bright bin at the same time".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — rebinned from 30 Hz to 500 ms bins** (15 frames per bin, 120 bins per 1-min trial).
`metadata['time_bin_size'] = 500.0` ms. The rebinning is the reference's smooth-then-average-pool
recipe applied with a kernel of 15 frames instead of the reference's 3 frames (100 ms). Position
and speed are pooled with the same kernel so all streams share the 500 ms grid.

ii.
```python
FPS = 30.0
TEMPORAL_BIN_FRAMES = 15   # 500 ms; reference uses 3 (see CONVERSION_NOTES Step 5 decision 4)
BINS_PER_TRIAL = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES    # 120
TIME_BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0        # 500.0
...
"time_bin_size": float(TIME_BIN_MS),
```

iii. CONVERSION_NOTES Step 5 decision 4 and Step 10 Check 3 (flagged there as one of two
deliberate deviations). Rationale: calcium events are very sparse (≈0.24 Hz/cell), so a 100 ms bin
holds ~0.024 events/cell; the provided decoder is a per-timepoint linear read-out of 100 PCs, not
a Bayesian accumulator, so per-bin SNR matters. The AI ran an explicit sweep
(`cache/binsize_test.py`, PCA + multinomial logistic regression, per session, 80/20 split):

| session | 100 ms | 500 ms | 1 s | 2 s |
|---|---|---|---|---|
| 51 day 0 (square) | 0.254 | 0.331 | 0.379 | 0.308 |
| 51 day 5 (rectangle) | 0.464 | **0.595** | 0.560 | 0.552 |
| 51 day 12 (l) | 0.654 | **0.767** | 0.709 | 0.635 |

and concluded 500 ms is "at or near the optimum and is still short relative to the time the mouse
spends crossing a 25 cm partition, so no positional information is smeared away", while also
keeping the pickle under 1 GB.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The per-day `blocked` field: the indices of the occluded partitions in the 3×3 grid numbered
`[[0,1,2],[3,4,5],[6,7,8]]`, or `-1` when nothing is blocked (the open square). `envs` (the
geometry's string name) is carried into metadata and used for cross-validation against
`get_env_mat`, but the input vector itself comes from `blocked`.

ii.
```python
envs = np.asarray(dat["envs"]).ravel()
blocked = dat["blocked"]
...
res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down, sid, ...)
```

iii. CONVERSION_NOTES Step 1: "`blocked` is not used anywhere in the reference code (geometry is
handled through the `envs` string names and `get_env_mat`), but it encodes exactly the decoder
input we need." Step 4 resolves an orientation ambiguity: `get_env_mat` is in *plot* orientation
(row 0 = bottom), whereas `blocked` is in array orientation; the AI verified on all 207 sessions
that the `blocked` vector equals `flipud(get_env_mat(env)) == 0` and adopted the `blocked`
convention so input and output share one index.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked` is converted to a **9-dim binary (float32) vector**, 1.0 = that partition is walled
off; `-1` (square) → all zeros. The vector is **static per session**: the same array is attached
to every trial of the session (shape `(9,)`, the per-trial form the format allows). `input_names`
are `geometry_partition_{p}_blocked`, indexed identically to the 9 output classes. Partition 7 is
never blocked in any of the 10 geometries, so that dimension is constant 0 across the dataset;
the AI kept it deliberately to preserve index correspondence with the output.

ii.
```python
def parse_blocked(blocked_day):
    """Return the 9-dim binary geometry vector for one session. ..."""
    idx = np.atleast_1d(np.asarray(blocked_day, dtype=float).ravel())
    geometry = np.zeros(SPATIAL_BINS * SPATIAL_BINS, dtype=np.float32)
    if idx.size == 1 and idx[0] < 0:          # -1 => open square, nothing blocked
        return geometry
    geometry[idx.astype(int)] = 1.0
    return geometry
...
geometry = parse_blocked(blocked_day)                        # (9,), 1 = blocked
...
input_trials.append(geometry.copy())
...
"input_names": [f"geometry_partition_{p}_blocked" for p in range(9)],
```

iii. CONVERSION_NOTES Step 5 decision 9: a 9-dim binary vector is the literal "environment
geometry, representing which parts of the arena are blocked" the task asks for, it is static
within a session (geometry only changes between days), and sharing the index with the output
classes lets the decoder exploit the fact that a blocked partition can never be the answer. Step
10 Check 1 explains the constant dimension 7 (listing all 10 geometries' blocked sets) as "constant
by design", not a bug. Verified by sanity check [2] on all 207 sessions against both `blocked` and
`flipud(get_env_mat(env))==0`, and by the decoder's behaviour (only 6 of 108,146 held-out
predictions land in a walled-off partition).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, `(n_days, 2, T)` — DeepLabCut head-tracking x,y in cm, range exactly
[0, 75], no NaNs. It is also used (via its time derivative) to compute the running speed that
drives the sample mask.

ii.
```python
position = dat["position"]                # (n_days, 2, T)
...
res = convert_session(trace[d], position[d], blocked[d], ...)   # position_day: (2, T) in cm
```

iii. CONVERSION_NOTES Step 2/Step 5: `position` is the only behavioural stream and is the same
array the reference `get_rate_maps` and `decode_position_within` consume. The AI verified its
range is exactly [0, 75] cm for all animals, matching the paper's 75 × 75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. 1. Average-pool x and y over the same 500 ms bins as the neural data (`bin_time`).
2. Divide by `bin_down` and take the floor to get an x-bin and a y-bin in {0,1,2}, where
   `bin_down = (max position over both axes and all days of the animal + 1e-5) / 3` — literally
   `decode_position_within`'s spatial-bin rule with `n_bins = 3`. The max is 75.0 cm for every
   animal, so `bin_down` = 25.000003 cm, the true physical partition width.
3. Clip to [0, 2] to guard the exact-boundary case.
4. Combine into a single categorical class `partition = y_bin * 3 + x_bin`.
5. Keep only the bins that survive the running/reachability mask; store as `(1, n_kept)` int64.

ii.
```python
POS_BUFFER = 1e-5          # get_rate_maps(buffer=1e-5)
SPATIAL_BINS = 3           # decoder task: 3 x 3 partitions (reference uses 15 for 5 cm bins)
...
# One isotropic spatial bin size per animal, from the maximum position across all
# days and both axes -- exactly decode_position_within's `bin_down`.
bin_down = (np.nanmax(position) + POS_BUFFER) / SPATIAL_BINS
...
# ---- 3. position -> 3x3 partition index ----------------------------------------
pos_binned = bin_time(position_day, n_trials)                # (2, n_trials, BINS)
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]             # class = y_bin*3 + x_bin
...
output_trials.append(partition[t, sel][None, :].astype(np.int64))
```
Reference being mirrored (`utils.py:decode_position_within`):
```python
behav_max = behav.max(axis=0).max(axis=1)
bin_down = (behav_max.max() + buffer) / n_bins
behav /= bin_down
```

iii. CONVERSION_NOTES Step 5 decision 7: the `bin_down` rule is taken from the reference so the
grid is isotropic and shared across all sessions of an animal (important because the reachable
area shrinks in the deformed geometries — a per-session max would move the grid). The buffer is
`get_rate_maps`'s 1e-5. The clip is justified in Step 10 Check 5: without it the reference formula
puts a sample at exactly `n_bins` (out of range) when the position is exactly at the maximum.
Averaging position within the bin before discretizing mirrors `fit_decoder`, which average-pools
position with the identical `AvgPool1d` before `.astype(int)`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Into **9 categories** — the 3 × 3 = 9 partitions, edges at 25 cm and 50 cm (i.e. the physical
partition walls), one categorical value per 500 ms bin. Class index = `y_bin * 3 + x_bin`, which is
the transpose of the index `fit_decoder` builds internally (`x_bin * n_bins + y_bin`) but is the
numbering the dataset's own `blocked` field and README use, so input and output share an index.
`output_names = ['position_partition']`, `output_values = ['p0_x0_y0', ..., 'p8_x2_y2']`. Realised
class distribution: [.095 .108 .114 .088 .070 .088 .108 .158 .171].

ii.
```python
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]             # class = y_bin*3 + x_bin
...
"output_names": ["position_partition"],
"output_values": [[f"p{p}_x{p % 3}_y{p // 3}" for p in range(9)]],
...
"spatial_discretization":
    "floor(position / bin_down) per axis, clipped to [0,2]; partition index = "
    "y_bin * 3 + x_bin, matching the dataset's `blocked` partition numbering "
    "[[0,1,2],[3,4,5],[6,7,8]]. bin_down = (max position over all sessions of the "
    "animal + 1e-5) / 3 = 25 cm, as in decode_position_within.",
```

iii. CONVERSION_NOTES Step 4 and Step 5 decision 8: the 3×3 grid is dictated by the decoder task
and happens to coincide with the arena's real partition walls, so the classes are physically
meaningful. The `y*3+x` choice is documented as a deliberate, accuracy-neutral relabelling —
"the transpose of `fit_decoder`'s internal class index — an arbitrary relabelling of the 9 classes
that leaves decoding accuracy unchanged, but makes the decoder *input* (geometry) and *output*
(partition) share one index". It was validated empirically rather than assumed: `cache/geo_check.py`
confirmed on all 207 sessions that occupancy is essentially zero in exactly the partitions listed
in `blocked` under this indexing (443 offending raw frames out of 14.9 M, all tracking artifacts),
which would fail under the transposed convention. The mild non-uniformity of the class
distribution is explained (thigmotaxis; the centre partition 4 is blocked in 5 of 10 geometries)
and is handled by the decoder's balanced loss/balanced accuracy.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame: `trace` and `position` share one 30 Hz index in the source file, both are
truncated to `n_trials × 1800` frames, both are pooled with the identical `bin_time` reshape into
the same (n_trials, 120) grid, and the identical boolean `keep` mask selects the same bins from
both. No shift, interpolation or resampling is applied anywhere.

ii.
```python
pos_binned = bin_time(position_day, n_trials)                # (2, n_trials, BINS)
...
neural_binned = bin_time(smoothed, n_trials) * FPS           # (n_act, n_trials, BINS)
...
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
    input_trials.append(geometry.copy())
    output_trials.append(partition[t, sel][None, :].astype(np.int64))
```

iii. CONVERSION_NOTES Step 3/Step 10 Check 3 row (c): "the DAQ simultaneously acquired behavioral
and cellular imaging streams at 30 Hz ... and all recorded frames were timestamped for post-hoc
alignment", and T is identical for `trace` and `position` within a day, so no further alignment is
required; "trials are cut on the same frame index for all three streams". Verified visually
(`--show-processing` panels 2, 4–6) and by sanity check [1], which recomputed both streams from
the raw files for 4 sessions and matched every value.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI enumerates and handles seven edge cases:
- **Unregistered cells** appear as all-NaN for a whole day → dropped per session (verified NaNs
  are never partial, so testing frame 0 is sufficient; no NaN survives into the output).
- **Recordings not an exact multiple of 60 s** → trailing partial minute dropped (98.88 % of
  frames retained).
- **Frame 0 has no defined speed** → bin 0 of the session excluded, matching the reference.
- **Position exactly at the arena maximum (75.0 cm)** → buffered `bin_down` plus an explicit
  `np.clip(..., 0, 2)` guard; without it the reference formula produces an out-of-range bin.
- **Tracking artifacts inside walled-off partitions** (443 raw frames dataset-wide, 428 of them
  from CA1-51 day 4; plus exactly 1 frame per `rectangle` session at the x = 25.0 boundary) →
  those bins are dropped (28 bins, 0.003 %).
- **Near-immobile minutes** → trials with < 10 retained bins dropped.
- **A session where every cell fails the activity filter** → `convert_session` returns `None`
  (never triggered).
Differing session lengths (71,866–72,219 frames) are handled per session. No RNG is used, so the
conversion is deterministic.

ii.
```python
registered = ~np.isnan(trace_day[:, 0])
raw = trace_day[registered][:, :n_used].astype(np.float32)
...
n_trials = n_frames // TRIAL_FRAMES          # complete 1-minute trials only
...
xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
...
#   The second term removes rare tracking artifacts: a handful of frames (and, on
#   two sessions, a few hundred) are tracked inside a partition that is walled off,
#   which cannot happen physically.  These would be unlearnable label noise.
moving = speed_binned > V_THRESH
reachable = geometry[partition] == 0
keep = moving & reachable
keep[0, 0] = False          # frame/bin 0 has no defined speed (reference excludes it)
...
if active.sum() == 0:
    return None
...
if len(neural_trials) < MIN_TRIALS_PER_SESSION:
    return None
```

iii. CONVERSION_NOTES Step 10 Check 5 ("Check for edge cases") tabulates each case with its
handling, and Check 4 gives a full frame-by-frame accounting of where every sample went
(14,903,019 raw frames → 528,802 retained 500 ms bins, with each drop category quantified). The
AI also records that the "unreachable" rule was added as a fix during Critical Review 1 (issue 2:
"Samples tracked inside walled-off partitions ... originally passed through as label noise; now
dropped, and counted in the summary"), and that a `sequence` metadata off-by-one (issue 3) was
found and fixed with a full re-run of the conversion and all checks.

## 6-a. What are the most time-consuming steps of the code?

i. The script prints per-animal load and process timings. Measured on the full run: **loading the
7 joblib archives took 96.2 s total (6.5–16.3 s each) and per-session processing took 185.4 s
total (0.57–1.07 s/session)**, for 284 s (4.7 min) end to end plus ~1 s to write the 709 MB
pickle. So the dominant cost is per-session processing — specifically the `gaussian_filter1d` pass
over the full (n_cells ≈ 340, 72,000) float32 trace and the boolean-masked event count — with
archive decompression a close second. The AI explicitly estimated the full run time from the
sample run (~5–6 min predicted vs 4.7 min actual) and concluded no parallelisation was needed.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
t_load = time.time() - t0
...
t1 = time.time()
... # per-day loop
t_proc = time.time() - t1
...
print(f"  {animal}: {len(out)} sessions, {total_cells} registered cells total, "
      f"load {t_load:.1f}s, process {t_proc:.1f}s ({t_proc / max(len(list(days)), 1):.2f}s/session)",
      flush=True)
```

iii. CONVERSION_NOTES Step 6 and Step 7 "Run Time Estimates": "Measured: 0.57–1.07 s of processing
per session, 6–17 s to load each animal. Full conversion = 4.7 min wall clock, well under the
15-minute budget, so no multiprocessing was needed (it would have multiplied the 17 GB peak per
worker)."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the ones that mattered — all temporal binning is a single
`reshape(...).mean(-1)` instead of per-trial loops, and the cell-event count is one boolean-mask
sum over the whole session. The loops that remain, and could in principle be reduced further:
- `for t in range(n_trials)` in `convert_session` (~8,200 iterations total) builds one
  variable-length array per trial. This cannot be fully vectorized because the running mask leaves
  a different number of bins per trial, but the whole session could be sliced once with
  `np.split` on the flattened kept-bin index instead of fancy-indexing `neural_binned[:, t, sel]`
  per trial (each of those is a copy of a (n_cells × ~66) block).
- `for d in days` in `convert_animal` — sequential over 207 sessions; embarrassingly parallel
  across animals (the AI considered and rejected multiprocessing on memory grounds).
- In `_plot_processing`, `for i, c in enumerate(show)` (30 `vlines` calls) and
  `for i, k in enumerate(keep[t_show])` (120 `axvspan` calls per trial) are plotting-only loops
  that could be single calls; they run only under `--show-processing`.
None of these is a bottleneck relative to the `gaussian_filter1d` pass.

ii.
```python
# already vectorized:
def bin_time(x, n_trials, axis=-1, reduce="mean"):
    x = np.moveaxis(x, axis, -1)
    x = x[..., : n_trials * TRIAL_FRAMES]
    shp = x.shape[:-1] + (n_trials, BINS_PER_TRIAL, TEMPORAL_BIN_FRAMES)
    out = x.reshape(shp).mean(axis=-1)
    return out

frame_keep = np.repeat(keep.ravel(), TEMPORAL_BIN_FRAMES)
n_events = raw[:, frame_keep].sum(axis=1)

# remaining per-trial loop:
for t in range(n_trials):
    sel = keep[t]
    if sel.sum() < MIN_BINS_PER_TRIAL:
        continue
    neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "All binning done with a single
`reshape(...).mean(-1)` instead of per-trial loops" (noted as a "large" saving — "avoids 8,000
python-level loops"); "Cell-activity counting done with one boolean-mask sum over the whole
session"; "float32 for the neural stream throughout (halves memory and filter cost)". The
remaining per-trial loop is inherent to producing the required list-of-per-trial-arrays output
format with variable trial lengths.

## 6-c. What processing does the code repeat multiple times?

i. Very little. Each animal archive is read exactly once (`convert_animal` explicitly avoids
per-session file I/O), `bin_down` is computed once per animal and reused for all its days, and
`bin_time` is called three times per session but on three different arrays (speed, position,
neural) rather than redundantly. The small repetitions that do exist:
- `geometry.copy()` is stored once per trial (~8,000 tiny 9-element copies of a session-constant
  vector) instead of sharing one array per session.
- `keep.sum()`/`sel.sum()` is recomputed in the per-trial loop and again in the summary
  bookkeeping fields.
- `len(list(days))` in the timing print re-materialises the day range.
- Across the *workflow* (not the script), the conversion itself was re-run in full after the
  Critical Review 1 fixes, and `cache/sanity_checks.py` re-loads each animal archive a second time
  to recompute the pipeline independently — but that duplication is deliberate: the checks must not
  import `convert_data.py`.

ii.
```python
# one load per animal, reused for every day:
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
bin_down = (np.nanmax(position) + POS_BUFFER) / SPATIAL_BINS
for d in days:
    res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down, sid, ...)

# per-trial copies of a session-constant vector:
input_trials.append(geometry.copy())
```

iii. CONVERSION_NOTES Step 6: "Loading a whole animal decompresses ~17 GB of float64; only one
animal is held at a time and each is loaded exactly once (no per-session file I/O)." The AI's
stated design goal was to avoid repeated I/O, which it achieves; the remaining repetitions are
negligible in cost.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three genuine pieces of wasted work, none of them large enough to matter at 4.7 min total:
1. **The whole joblib archive is deserialised** even though only `trace`, `position`, `envs` and
   `blocked` are used. The archives also carry `SFPs` (35×35×n_cells×n_days ≈ 290 MB uncompressed
   for the largest animal), `centroids` and the precomputed `maps`, all decompressed and then
   dropped. This is most of the 96 s of load time.
2. **Smoothing and pooling are computed for every time bin, then ~46 % are thrown away.** The
   running/reachability mask is known before step 7, so `gaussian_filter1d` + `bin_time` produce
   453,170 bins of neural data that are immediately discarded (though, because the gaussian filter
   needs contiguous frames, filtering the full session first is the *correct* order — the waste is
   inherent to the "smooth before masking" choice, not an oversight).
3. **Per-session bookkeeping** (`n_bins_moving`, `n_bins_unreachable`, `n_frames_total`,
   `trial_index`, …) is computed for every session purely for the printed summary; `trial_index`
   and several counters never reach the pickle.
Also: `--show-processing` in `--full` mode would render a 7-panel figure for *every* day of the
first two animals (62 figures) rather than 2, because `show_processing` is passed to each session
of an animal while the plot budget is decremented per animal — the flag was only ever exercised in
`--sample` mode, where it correctly produces 2 figures.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]   # also loads SFPs, centroids, maps
trace = dat["trace"]; position = dat["position"]
envs = np.asarray(dat["envs"]).ravel(); blocked = dat["blocked"]
```
```python
smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
neural_binned = bin_time(smoothed, n_trials) * FPS   # all 120 bins/trial, ~46% later discarded
```
```python
n_plots_left = 2 if args.show_processing else 0
for animal, days in plan.items():
    show = n_plots_left > 0
    out, ncells = convert_animal(animal, days=days, show_processing=show, plot_dir=args.plot_dir)
    # only plot the first session of each animal we visit, up to 2 in total
    if show:
        n_plots_left -= 1
```

iii. The AI did not flag items 1–3 explicitly; its Step 6 efficiency notes concentrate on what it
*did* optimise (float32, contiguous-axis filtering, reshape-based binning, one load per animal)
and on the conclusion that "Full conversion = 4.7 min wall clock, well under the 15-minute budget,
so no multiprocessing was needed". The comment in `main` shows the intent was "only plot the first
session of each animal we visit, up to 2 in total", which is what happens in `--sample` mode.
