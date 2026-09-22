# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven per-animal **joblib** files in `/app/data` (the extension-less `QLAK-CA1-*` files), which is the default path of the reference code's `load_dat(..., format="joblib")`. Each file is a dict `{animal_id: payload}` whose payload holds `trace` (days x globally-registered cells x frames), `position` (days x 2 x frames), `blocked` (per-day partition indices), `envs`, plus unused `SFPs`, `centroids`, and `maps`. The animal list is hardcoded in the order used by the reference `main.py`. One animal is loaded at a time and the payload is deleted after its days are processed. All 207 sessions / 8,187 trials / 69,744 session-neurons are loaded in one pass (`--sample` restricts to the first two days of the first animal).

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75",
]
...
for animal_idx, animal in enumerate(ANIMALS):
    if animal not in targets:
        continue
    payload = joblib.load(DATA_DIR / animal)[animal]
    trace = np.asarray(payload["trace"])
    position = np.asarray(payload["position"])
    envs = np.asarray(payload["envs"]).squeeze()
    selected_days = range(trace.shape[0]) if targets[animal] is None else sorted(targets[animal])
    for day in selected_days:
        ...
    del payload, trace, position
```

iii. From CONVERSION_NOTES Step 1/Step 10: "Loading | `load_dat` uses joblib nested by animal | Same joblib payload and animal list | Match". The AI documented that the `.mat` and joblib files are duplicates of the same dataset and that the reference code's default loader is the joblib branch, so it used that branch and the reference's canonical animal ordering. Loading one animal at a time and freeing it was documented as the memory-control measure for arrays of this size.

## 1-b. How are the data split into subjects (mice)?

i. One subject per animal file. `subjects` is the fixed seven-element `ANIMALS` list (kept complete even in `--sample` mode) and `subject_idx` records the animal index for every emitted session, so sessions are animal-major and then day-ordered. Verified in the converted pickle: 7 subjects, `subject_idx` running 0...6 with 31/31/31/21/31/31/31 sessions.

ii.
```python
subject_idx.append(animal_idx)
...
"subjects": ANIMALS.copy(),
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2/Step 5: "Seven extensionless compressed joblib files (`QLAK-CA1-*`) ... one file per mouse", mapped as "Seven IDs in reference order; one subject index per day-session", cross-checked against the paper's per-animal unique-cell statistics (mean 773.29, range 515-952, matching "773 +/- 68 SE, minimum 515, maximum 952").

## 1-c. How are the data split into sessions?

i. Each recording **day** within an animal file (axis 0 of `trace`/`position`/`blocked`/`envs`) becomes one output session. Six animals contribute 31 days and `QLAK-CA1-51` contributes 21, giving exactly 207 sessions. Day order is preserved (it encodes each mouse's randomized geometry sequence), and each session records its environment name, day index, frame counts, and blocked indices in `metadata['session_info']`.

ii.
```python
selected_days = range(trace.shape[0]) if targets[animal] is None else sorted(targets[animal])
for day in selected_days:
    ...
    session_id = f"{animal}_day{day:02d}"
    session_info.append({
        "session_id": session_id, "subject": animal, "day_index": int(day),
        "environment": str(envs[day]), "original_frames": n_raw_frames,
        "used_frames": n_used_frames, "discarded_tail_frames": n_raw_frames - n_used_frames,
        "n_trials": n_trials, "n_neurons": int(present.sum()),
        "blocked_indices": np.flatnonzero(geometry).astype(int).tolist(),
    })
```

iii. CONVERSION_NOTES Step 2/Step 4: "one session was recorded per day", "Six animals have 31 days ... while QLAK-CA1-51 has 21 days"; Step 4 resolution: "Preserve each stored session/day order" because geometry sequences are randomized per mouse. The 207-session total was checked against the methods quote "5,413 unique neurons across 207 sessions".

## 1-d. How are the data split into trials?

i. The instructions define trials as 1-minute windows. Each continuous ~40-min session is cut into consecutive, non-overlapping 1,800-frame (60 s at 30 Hz) windows; the incomplete tail is dropped. Because the AI pools 3 frames per time bin, each trial is 600 bins. Sessions with <2 complete trials would raise (none do). Result: 39 trials for 93 sessions, 40 for 114, total 8,187 — independently reproduced here from the raw `.mat` files.

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS            # 1800
TIMEPOINTS_PER_TRIAL = RAW_TRIAL_FRAMES // POOL_FRAMES  # 600
...
n_raw_frames = int(position.shape[2])
n_trials = n_raw_frames // RAW_TRIAL_FRAMES
n_used_frames = n_trials * RAW_TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f"{animal} day {day} has fewer than two complete trials")
...
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
    session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
    session_input.append(geometry.copy())
    session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "Consecutive non-overlapping 1,800-frame windows are exactly 60 s and become trials. Only the short incomplete tail is discarded. This yields 8,187 trials". Step 10 edge-case check explicitly confirmed the trial-0/trial-1 boundary (frames 1797:1800 vs 1800:1803) has "no overlap or gap", and per-animal tails (1,666 / 219 / 91 / 60 / 71 frames) are recorded in metadata.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering. Only incomplete tail windows are dropped, and there is a hard requirement of >=2 complete trials per session. The AI explicitly considered and rejected the reference decoder's >5 cm/s speed mask (`decode_position_within`), because dropping frames would break the required contiguous 60-second trials. It kept the blocked-bin occupancy check as a *diagnostic* only, not a rejection rule.

ii.
```python
if n_trials < 2:
    raise ValueError(f"{animal} day {day} has fewer than two complete trials")
...
# A diagnostic rather than a rejection rule: tracking at walls can place
# occasional samples on a blocked-bin boundary.
occupied_geometry = geometry[labels]
blocked_position_count += int(occupied_geometry.sum())
total_position_count += int(labels.size)
```

iii. CONVERSION_NOTES Step 3: "No event trials and no trial rejection are described"; Step 5 Key Decision 3: "The reference's >5-moving-event cell and >5 cm/s frame masks are classifier-specific optimizations. Dropping frames would violate contiguous one-minute trial timing." The 152/4,912,200 (0.0031%) samples that land in blocked bins were attributed to head-tracking/bin-edge noise and left in place rather than silently relabelled.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely `payload["trace"]` — the authors' rise-extracted binary calcium event matrix (1 = significant transient rising phase), indexed `[day, globally_registered_cell, frame]` at 30 Hz. No dF/F recomputation, no deconvolution, no use of `maps`, `SFPs`, or `centroids`.

ii.
```python
trace = np.asarray(payload["trace"])
...
binned_neural = process_neural(trace[day], present, n_used_frames)
```

iii. CONVERSION_NOTES Step 1/Step 4: "The native neural stream is `trace`, a rise-extracted binary calcium-event series sampled at 30 Hz. Delta-F/F has already been processed upstream; it must not be recomputed", matching the methods statement "All analyses were conducted using the binary vector of the rising phases of transients, treating this vector as if it were the firing rate of the cell." `maps` were classified as "reference products for checks, not decoder inputs."

## 2-b. How is the `neural` data processed?

i. For each session: select the cells registered that day, cast to float32, Gaussian-smooth along time with sigma = 3 native frames **over the whole continuous session**, crop to the complete-trial frames, then mean-pool disjoint groups of 3 frames. This reproduces the smoothing+pooling that the reference `fit_decoder`/`test_decoder` apply before Bayesian position decoding. Values end up continuous in [0, 1]. Smoothing is done in place (`output=`) to avoid a second session-sized array; smoothing before cropping avoids a synthetic filter edge at the last retained minute (a bug the AI found and fixed in Step 10).

ii.
```python
def process_neural(raw_trace, present, n_used_frames):
    """Select registered cells, Gaussian-smooth, and mean-pool aligned frames."""
    selected = np.asarray(raw_trace[present, :], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Registered neural traces contain intermittent NaN/Inf")
    gaussian_filter1d(selected, sigma=POOL_FRAMES, axis=1, output=selected)
    selected = selected[:, :n_used_frames]
    return selected.reshape(selected.shape[0], -1, POOL_FRAMES).mean(axis=2)
```
(compare reference `utils.fit_decoder`: `pooling(torch.tensor(gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0).T))` with `temporal_bin_size=3`.)

iii. CONVERSION_NOTES Step 5 Key Decision 2: "Use the reference position decoder's `temporal_bin_size=3` at 30 Hz. Gaussian smoothing (sigma 3 native frames) followed by 3-frame average pooling is applied to traces ... This preserves reference processing and makes every trial 600 timepoints." Step 10 documents the fix that smoothing now runs over the complete physical session "preventing a synthetic filter boundary at the final retained minute".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells that were **registered on that day** are kept: a cell absent from a day is stored as an all-NaN trace, and the AI tests finiteness of the day's first frame to build the `present` mask, then asserts that the retained block contains no NaN/Inf at all. No place-cell selection, no minimum-event threshold (the 45 session-neurons with zero events are deliberately retained), no speed-gated activity threshold. The retained count is exactly 69,744 session-neurons, the number of rate maps reported in the paper.

ii.
```python
present = np.isfinite(trace[day, :, 0])
if not present.any():
    raise ValueError(f"{animal} day {day} has no registered neurons")
...
selected = np.asarray(raw_trace[present, :], dtype=np.float32)
if not np.isfinite(selected).all():
    raise ValueError("Registered neural traces contain intermittent NaN/Inf")
...
brain_region_idx.append(np.zeros(int(present.sum()), dtype=np.int64))
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "All manually curated, day-registered cells are retained, matching the paper's stated inclusion of all cells and preserving the reported 69,744 rate maps." Step 4 records the paper's justification ("high population reliability motivated the inclusion of all cells in subsequent analyses"), so place-cell filtering (p<0.01 split-half) was judged not to be the reference's general rule, and the decoder-specific >5-event mask was judged classifier-specific.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the recording is continuous free exploration. Trials are therefore aligned to the start of each contiguous 60-second window measured from frame 0 of the session; `off_start=0.0`, `off_end=60.0`, `temporal_alignment_event` is filled in accordingly. Neural and position streams share the same frame index, so alignment is index-for-index, with both streams pooled over identical frame triplets.

ii.
```python
"temporal_alignment_event": "start of each contiguous one-minute window within the recording session",
"off_start": 0.0,
"off_end": 60.0,
...
if binned_neural.shape[1] != labels.size:
    raise AssertionError("Neural and position bins are temporally misaligned")
if binned_neural.shape[1] != n_trials * TIMEPOINTS_PER_TRIAL:
    raise AssertionError("Unexpected number of pooled bins")
```

iii. CONVERSION_NOTES Step 4/Step 5 Key Decision 6: "Neural and behavioral samples already align one-to-one. Temporal binning begins at session frame 0"; the methods state both streams were "simultaneously acquired ... at 30 Hz" and timestamped, so "No resynchronization or interpolation needed." Trial-boundary frames were explicitly checked in Step 10.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning is applied. The native 30 Hz (33.3 ms) stream is Gaussian-smoothed (sigma = 3 frames) and mean-pooled in disjoint 3-frame groups, giving a **100 ms** bin (`time_bin_size = 100.0`) and a uniform 600 bins per 60-second trial for every trial and session. Position is pooled over exactly the same frame triplets.

ii.
```python
FPS = 30
POOL_FRAMES = 3
TIME_BIN_MS = 100.0
TIMEPOINTS_PER_TRIAL = RAW_TRIAL_FRAMES // POOL_FRAMES   # 600
...
"time_bin_size": TIME_BIN_MS,
"neural_processing": "binary rising-phase events; Gaussian sigma=3 source frames; non-overlapping 3-frame mean pooling",
```

iii. CONVERSION_NOTES Step 5 Key Decision 2 and the Step 10 comparison table: "Temporal binning | `fit_decoder`/`test_decoder`: Gaussian sigma=3 then 3-frame AvgPool; position 3-frame AvgPool | Same sigma, SciPy default boundary mode, and non-overlapping 3-frame mean ... | Match". I.e. the bin size was chosen to equal the reference paper's own position-decoding temporal bin rather than being invented.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From `payload["blocked"][day]` — the list of blocked 3x3 partition indices for that day (`-1` meaning nothing blocked). The alternative, deriving geometry from the `envs` string via the reference `get_env_mat()`, was explicitly rejected because the stored `blocked` field disagrees with `get_env_mat` for the orientation-sensitive shapes (`t`, `l`, `bit donut`, `glenn`).

ii.
```python
geometry = blocked_vector(payload["blocked"][day][0])
...
values = np.asarray(raw_blocked).reshape(-1)
```

iii. CONVERSION_NOTES Step 4: "`blocked` differs from `get_env_mat` for `t`, `l`, `bit donut`, and `glenn` ... Use native `blocked` as authoritative"; Step 5: "Native field preserves orientation". The code README defines `blocked` as "location of blocked (occluded) partitions in 3x3 design ... organized in the following way - [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If no partitions are blocked, value is -1."

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A 9-dim binary one-hot-style vector (1 = blocked, 0 = accessible), zero everywhere when the day's value is `-1`, validated to lie in 0..8. Because the AI labels position as `class = x*3 + y` while the native `blocked` matrix is indexed `(row=y, col=x)`, it fills the native (y,x) matrix and then **transposes** it before flattening, so that input element *j* refers to the same physical partition as output class *j*. The vector is static and repeated for every trial of the session; `input_names` are `blocked_x{x}_y{y}`.

ii.
```python
def blocked_vector(raw_blocked):
    """Return x-first 3x3 geometry with 1 for blocked, 0 for accessible."""
    values = np.asarray(raw_blocked).reshape(-1)
    native_yx = np.zeros(9, dtype=np.float32)
    if values.size == 1 and float(values[0]) == -1.0:
        return native_yx
    indices = values.astype(np.int64)
    if np.any(indices < 0) or np.any(indices > 8):
        raise ValueError(f"Invalid blocked partition index: {values}")
    native_yx[indices] = 1.0
    return np.ascontiguousarray(native_yx.reshape(3, 3).T.ravel())
...
session_input.append(geometry.copy())
```

iii. CONVERSION_NOTES Step 5 Key Decision 4 / Step 9-10: the first full run showed 17.14% of position samples falling inside supposedly blocked partitions; "Testing all eight grid symmetries made this unambiguous: transpose reduced position/block overlap from 17.14% to 0.0031%; the residual 152 boundary samples are tracking/bin-edge noise." The occupancy orientation of `position` itself was separately confirmed against the authors' supplied `maps.sampling` (r = 0.99991 direct vs 0.52980 transposed).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. From `payload["position"]`, indexed `[day, xy, frame]` with row 0 = x and row 1 = y in the reference code's convention, in centimetres spanning exactly 0-75 cm. No other behavioural field is used.

ii.
```python
position = np.asarray(payload["position"])
...
binned_position = pool_position(position[day], n_used_frames)
labels = position_classes(binned_position)
```

iii. CONVERSION_NOTES Step 1/Step 10: positions are the DeepLabCut head-tracking stream, "author calls use `position[day].T` (time x 2)", and the x-first ordering was verified against the authors' occupancy map (`maps.sampling`) with correlation 0.99991 in the direct orientation.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The 2 x time position stream is truncated to complete-trial frames and mean-pooled over the same disjoint 3-frame groups used for the neural data (asserting finiteness first), then discretized (4-c). The pooled continuous position is kept only as an intermediate; the saved output is the integer class, shape (1, 600) per trial, dtype int64.

ii.
```python
def pool_position(raw_position, n_used_frames):
    """Mean-pool an aligned 2 x time position stream into 100 ms bins."""
    selected = np.asarray(raw_position[:, :n_used_frames], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Position contains NaN/Inf in a retained complete trial")
    return selected.reshape(2, -1, POOL_FRAMES).mean(axis=2)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Average each non-overlapping 3-frame group; clip coordinates to [0,75); floor-divide by 25 cm", listed against reference function "`fit_decoder` position pooling and row-major one-hot construction" — the reference likewise averages the position over 3 frames before assigning a spatial bin.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Fixed arena bounds: each axis is clipped to [0, 75) using `nextafter` (so an exact 75.0 sample falls in bin 2 rather than creating a 10th class), floor-divided by 25 cm into 3 bins per axis, and flattened as `class = x_bin*3 + y_bin` (the reference code's x-first `rate_maps[:, x, y]` convention), giving 9 classes named `x0_y0 ... x2_y2`. An assertion rejects any class outside 0..8. All nine classes occur; the global distribution is [0.100, 0.075, 0.116, 0.098, 0.057, 0.141, 0.135, 0.077, 0.200].

ii.
```python
ARENA_CM = 75.0; SPATIAL_BINS = 3; SPATIAL_BIN_CM = ARENA_CM / SPATIAL_BINS

def position_classes(position_binned):
    """Convert 2-D centimeters to x-first, row-major 3x3 categorical labels."""
    upper = np.nextafter(np.float32(ARENA_CM), np.float32(0.0))
    xy = np.floor(np.clip(position_binned, 0.0, upper) / SPATIAL_BIN_CM).astype(np.int64)
    labels = xy[0] * SPATIAL_BINS + xy[1]
    if labels.min() < 0 or labels.max() > 8:
        raise AssertionError("Position discretization produced a class outside 0..8")
    return labels
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Use the reference's x-first NumPy row-major flattening. Bins cover [0,25), [25,50), and [50,75] cm on each axis. This is the requested coarse 3x3 analogue of the reference 15x15 decoder." Step 10 edge-case check: "Exact 75 cm coordinates are clipped into bin 2 rather than producing class 9/12."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and neural data share the same 30 Hz frame index (simultaneous timestamped acquisition), the same crop to `n_used_frames`, the same 3-frame pooling grid, and the same trial slicing, so bin *t* of a trial covers the same three raw frames in both streams. Shape assertions guard against drift, and the `--show-processing` plots overlay raw and pooled position to show no lag.

ii.
```python
binned_position = pool_position(position[day], n_used_frames)
labels = position_classes(binned_position)
binned_neural = process_neural(trace[day], present, n_used_frames)
if binned_neural.shape[1] != labels.size:
    raise AssertionError("Neural and position bins are temporally misaligned")
...
sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
session_neural.append(...binned_neural[:, sl]...)
session_output.append(...labels[sl][None, :]...)
```

iii. CONVERSION_NOTES Step 4/Step 10: "Identical time axes; no position NaNs/intermittent trace NaNs ... No resynchronization or interpolation needed", and the independent Step 10 sanity checks "`np.allclose`" the converted neural bin to the smoothed mean of its *exact three raw frames* and the converted class to the same three raw position samples, for three animals/sessions including a trial edge.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) Cells not registered on a day appear as all-NaN traces and are dropped via the `present` mask; (b) any *intermittent* NaN/Inf in a retained trace, or in position, raises immediately rather than propagating silently (verified never to trigger); (c) the incomplete final <60 s window is discarded and the discarded tail size is recorded per session in metadata; (d) sessions with no registered cells or fewer than two complete trials raise; (e) position samples exactly at the 75 cm wall are clipped into the last bin; (f) the 45 zero-event session-neurons are knowingly retained; (g) the ~0.003% of samples that land in a blocked partition (wall-adjacent tracking noise) are logged as a diagnostic, not removed or relabelled; (h) `validate_before_save` re-checks shapes, finiteness, and the expected 207/8,187/69,744 totals before writing the pickle.

ii.
```python
present = np.isfinite(trace[day, :, 0])
if not present.any():
    raise ValueError(f"{animal} day {day} has no registered neurons")
...
if not np.isfinite(selected).all():
    raise ValueError("Position contains NaN/Inf in a retained complete trial")
...
"discarded_tail_frames": n_raw_frames - n_used_frames,
...
def validate_before_save(data, sample):
    ...
    if not np.isfinite(n).all() or not np.isfinite(i).all() or not np.isfinite(o).all():
        raise AssertionError(f"NaN/Inf in session {s}")
    if sum(map(len, data["neural"])) != 8187:
        raise AssertionError("Full dataset trial count is not 8,187")
    if sum(len(x) for x in data["brain_region_idx"]) != 69744:
        raise AssertionError("Full dataset session-neuron count is not 69,744")
```

iii. CONVERSION_NOTES Step 2/Step 10: "Missing day-specific cell registrations are all-NaN traces. Position is finite for every stored frame; trace/position frame axes are exactly aligned and have no intermittent missing samples", and Step 10 edge cases: "All-NaN unregistered cells are excluded, finite registered cells contain no intermittent NaNs, and zero-event registered cells remain intentionally." Fail-fast assertions were preferred over imputation because the data were verified to contain no partial gaps.

## 6-a. What are the most time-consuming steps of the code?

i. Loading/decompressing the joblib payloads dominates: 8.84 / 14.73 / 15.88 / 6.44 / 14.58 / 12.24 / 15.97 s = ~88.7 s of the 161.29 s conversion, versus 0.16-0.65 s per session for smoothing+pooling+slicing (~72 s total for 207 sessions), plus 5.63 s to write the 6.2 GiB pickle. Within a session, the Gaussian smoothing of the (n_neurons x ~72,000) float32 matrix is the main compute cost.

ii.
```python
load_start = time.perf_counter()
payload = joblib.load(DATA_DIR / animal)[animal]
print(f"Loaded {animal} in {time.perf_counter()-load_start:.2f} s", flush=True)
...
print(f"  {session_id}: ... {time.perf_counter()-session_start:.2f} s", flush=True)
...
print(f"Conversion computation time: {time.perf_counter()-started:.2f} s", flush=True)
```

iii. CONVERSION_NOTES Step 7 run-time table attributes the cost to "Native load (amortized) ~4.44 s in sample; ~0.5-0.7 s/day when amortized" vs "Session conversion + optional plot 0.70 s", estimating 3-5 min in total; Step 9 reports the realized 161.29 s + 5.39 s save, i.e. well under the 15-minute budget, so no further optimization was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Essentially everything heavy is already vectorized: smoothing is a single SciPy call per session, pooling is a `reshape(...).mean(axis=2)`, and discretization is a vectorized `floor/clip`. The only remaining Python loops are over animals (7), days (207), and trials within a day (39-40). The per-trial loop is the one that could be avoided — the 8,187 `np.ascontiguousarray` slice copies (and the 8,187 `geometry.copy()` calls) could instead be produced by a single reshape/`np.split` over the pooled session arrays, since the required trials are contiguous equal-length blocks. This is a small cost (a few seconds and ~2x transient memory on the neural array) relative to I/O.

ii.
```python
for trial in range(n_trials):
    sl = slice(trial * TIMEPOINTS_PER_TRIAL, (trial + 1) * TIMEPOINTS_PER_TRIAL)
    session_neural.append(np.ascontiguousarray(binned_neural[:, sl], dtype=np.float32))
    session_input.append(geometry.copy())
    session_output.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Native float64 animal arrays are large ... Repeated trial-by-trial filtering would also repeat Gaussian work. Code speedups added: Each animal is loaded once; each selected session is cast once to float32; SciPy smoothing overwrites that working buffer via `output=`; vectorized reshape/mean performs pooling". The AI deliberately hoisted all per-trial processing out of the trial loop, leaving only slicing/copying inside it.

## 6-c. What processing does the code repeat multiple times?

i. Only minor repetition: `int(present.sum())` is recomputed four times per session; `n_raw_frames`/`n_trials`/`n_used_frames` are recomputed inside the day loop although they are constant within an animal (all days of an animal share the same frame count); `geometry` is copied once per trial although all trials in a session share the same static vector; `np.ascontiguousarray` re-copies data that is already contiguous along the slicing axis for the output labels. No expensive step (load, smoothing, pooling, discretization) is executed more than once per session. Separately, the validation/sanity-check scripts re-read the raw files, but that is by design (independence from the conversion code).

ii.
```python
brain_region_idx.append(np.zeros(int(present.sum()), dtype=np.int64))
...
"n_neurons": int(present.sum()),
...
print(f"  {session_id}: {int(present.sum())} neurons, ...")
```

iii. The notes do not flag these as problems; CONVERSION_NOTES Step 6 states the design goal that "Each animal is loaded once; each selected session is cast once to float32", and Step 9/10 report that total runtime (161 s) was comfortably within budget, so no further deduplication was pursued.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `joblib.load` materialises the entire animal payload — `SFPs`, `centroids`, and `maps` are decompressed into memory but never used; only `trace`, `position`, `blocked`, and `envs` are needed (an h5py read of the parallel `.mat` files would touch only those). This is the largest avoidable cost, inside the step that dominates runtime. (2) Diagnostic bookkeeping — `occupied_geometry`, `blocked_position_count`, `class_counts`, per-session `session_info` — is computed for every session and is not consumed by the decoder. (3) The continuous pooled position (`binned_position`) is only an intermediate for discretization. (4) `envs[day]` strings are read only for metadata. (5) The `--show-processing` plotting path (capped at 2 sessions). (6) Storing a separate 9-float copy of the identical geometry vector for each of 8,187 trials. All of these are small or serve documentation/validation purposes; none changes the converted values.

ii.
```python
payload = joblib.load(DATA_DIR / animal)[animal]   # also loads SFPs/centroids/maps
...
occupied_geometry = geometry[labels]
blocked_position_count += int(occupied_geometry.sum())
total_position_count += int(labels.size)
class_counts += np.bincount(labels, minlength=9)
...
session_info.append({... "environment": str(envs[day]), ...})
```

iii. CONVERSION_NOTES Step 1 explains that `maps`/`SFPs` are "reference products for checks, not decoder inputs" (they are loaded only because the joblib file is a single object), and Step 5/Step 10 justify the diagnostics: the blocked-bin overlap statistic is what exposed the geometry-orientation bug, and `session_info` is retained as per-session provenance ("metadata records discarded tail frames", environment name, blocked indices).
