# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the **joblib** version of the dataset (the files in `/app/data` with no extension), not the `.mat` twins. It hard-codes the list of the 7 animal IDs taken from the reference repo's `main.py`, and for each animal calls `joblib.load(<path>)[animal]`, which returns the dict documented in the reference README (`SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`). Only four fields are used: `trace`, `position`, `blocked`, `envs`. The file is loaded once per animal (not once per session) and `del`-ed after the animal's sessions are processed. Inside the animal, sessions are the first axis of `position`/`trace` (`ndays = positions.shape[0]`), and each session is cut into trials afterwards.

ii.
```python
DATA_DIR = '/app/data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for ai, animal in enumerate(animals):
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0
    traces = dat['trace']
    positions = dat['position']
    envs = np.array(dat['envs']).ravel()
    blocked = dat['blocked']
    ndays = positions.shape[0]
    ...
    for d in range(ndays):
        trace_day = np.asarray(traces[d])
        position_day = np.asarray(positions[d])
        neural_trials, input_trials, output_trials, stats, aux = process_session(
            trace_day, position_day, blocked[d])
    ...
    del dat, traces, positions
```

iii. From CONVERSION_NOTES Step 1/2: the reference loader is `load_dat(animal, p, to_convert=["envs","position","trace"], format="joblib")` in `src/utils.py:61`, whose **default format is joblib**; the `.mat` files are stated by the repo README to be "identical content in MATLAB v7.3 format". The AI wrote: "I use the joblib files (same data, faster, the repo default)." It verified equivalence to the paper at load time: 207 sessions, 5,413 unique cells, 69,744 registered (non-NaN) cell-sessions — all exactly the numbers quoted in the paper.

## 1-b. How are the data split into subjects (mice)?

i. One subject per data file / animal ID. `subjects` is the fixed list of 7 IDs; `subject_idx` gets one entry (the animal's index) appended for every session emitted by that animal. All 7 mice are kept; none are excluded.

ii.
```python
data = {
    ...
    'subjects': ANIMALS if not sample else ANIMALS[:1],
    'subject_idx': [],
    ...
}
for ai, animal in enumerate(animals):
    ...
    for d in range(ndays):
        ...
        data['subject_idx'].append(ai)
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=int)
```

iii. The reference repo's `main.py` lists exactly these 7 animals, and each joblib file contains all of one animal's recording days. The AI cross-checked the per-animal cell counts against the paper ("mean cells per animal = 773 ± 68 SE, min 515, max 952") and got an exact match, confirming the file↔animal mapping.

## 1-c. How are the data split into sessions?

i. One output session = one recording day of one animal (the first axis of `trace`/`position`). No sessions are merged or dropped: 31/31/31/21/31/31/31 = **207** sessions. The session's geometry name (`envs[d]`) and blocked-partition list are recorded in `metadata['session_info']`.

ii.
```python
ndays = positions.shape[0]
...
for d in range(ndays):
    ...
    session_info.append(dict(animal=animal, day=int(d), env=str(envs[d]),
                             blocked=np.where(stats['geometry'] > 0)[0].tolist(),
                             n_registered=stats['n_registered'],
                             n_neurons=stats['n_kept_cells'],
                             n_trials=stats['n_trials'], ...))
```

iii. CONVERSION_NOTES Step 5, decision 1: "**Session = one recording day** (207 sessions). Matches the reference, where all analyses are per-day." All reference functions (`get_rate_maps`, `decode_position_within`, `get_shr_within`) operate day-by-day, cell registration is per-day, and the paper states "All sessions were 40 min, and one session was recorded per day". The resulting count (207) was checked against the paper's "5,413 unique neurons across 207 sessions".

## 1-d. How are the data split into trials?

i. There are no experimenter-defined trials (continuous 40-min free foraging), so the AI follows the task instruction and cuts each session into consecutive, non-overlapping **1-minute windows of session time**. Because it rebins to 100 ms, a trial is 600 bins (`TRIAL_BINS = 60 s × 30 Hz / 3 frames`). The trailing partial minute of each session is dropped. Sessions are 71,866–72,219 frames, giving 39–40 windows per session (8,056 trials total after the trial QC of 1-e).

ii.
```python
TRIAL_SECONDS = 60.0                                               # task requirement
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES)) # 600 bins
...
n_full_trials = nbins // TRIAL_BINS       # drop trailing partial minute
for t in range(n_full_trials):
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    ...
    neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
    output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
    input_trials.append(geometry.copy())
```

iii. CONVERSION_NOTES Step 5, decision 6: "Trials = consecutive 1-min windows of session time (600 bins of 100 ms), as specified by the task. The trailing partial window (<1 min) of each session is dropped so that all trials cover the same amount of real time." Step 10 Check 5 documents the edge case explicitly (sessions are 39.9–40.1 one-minute windows long).

## 1-e. How are trials filtered based on quality controls?

i. Two trial/session-level filters, both consequences of the timepoint-level speed filter:
- A 1-minute window that retains **fewer than 30 moving bins (3 s of locomotion)** is dropped entirely.
- A session that ends up with **fewer than 2 trials** is skipped (the format spec requires ≥2 trials per session). This never actually fired — the minimum retained session has 22 trials.
Mean 38.9 trials/session (min 22, max 40).

ii.
```python
MIN_TRIAL_BINS = 30        # drop trials with < 3 s of moving data
...
    m = moving_bins[sl]
    if m.sum() < MIN_TRIAL_BINS:
        continue
...
    if len(neural_trials) < 2:
        print(f'  WARNING session {animal} day {d} has < 2 trials, skipped')
        continue
```

iii. CONVERSION_NOTES Step 5, decision 6 and Step 10 Check 5: "Trials retaining < 30 moving bins (3 s) are dropped as too short/unreliable; sessions keep ~40 trials each, far above the minimum of 2." The reference paper/code has no trial concept and therefore no trial QC; this rule exists purely to avoid degenerate, near-empty trials produced by the reference immobility filter, and is documented as such.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely from `dat[animal]['trace'][day]` — the released **rise-extracted, binary** calcium event traces, shape `(n_cells, n_frames)`, with all-NaN rows for cells not registered that day. No other neural field (`SFPs`, `centroids`, `maps`) is used.

ii.
```python
traces = dat['trace']
...
trace_day = np.asarray(traces[d])   # (ncells, T) binary, NaN rows = unregistered
...
tr = trace_day[keep_cells].astype(np.float32)
```

iii. CONVERSION_NOTES Step 1/3: the repo README defines `trace` as "rise-extracted calcium traces, where '1' indicates a significant event", and the paper says "This binary vector was treated as the firing rate in all further analyses". The AI verified the values are exactly {0, 1} (plus NaN) and concluded explicitly that **no ΔF/F computation is needed**, because the released data are already past motion correction, segmentation, trace extraction and rising-phase thresholding.

## 2-b. How is the `neural` data processed?

i. Exactly the preprocessing the reference decoder applies (`fit_decoder` / `test_decoder`, `utils.py:1776`):
1. keep only curated cells (2-c);
2. `gaussian_filter1d(trace, sigma = 3 frames, axis = time)` on the **continuous** session trace;
3. average-pool over non-overlapping 3-frame windows → 100 ms bins (numpy reshape+mean, the equivalent of the reference `AvgPool1d(kernel_size=3, stride=3)`);
4. multiply by 30 Hz so the value is an event rate in Hz;
5. drop bins whose mean speed ≤ 5 cm/s;
6. slice into 1-minute trials, stored as `float32` `(n_neurons, T)`.

ii.
```python
def bin_frames(x, nframes_per_bin=TEMPORAL_BIN_FRAMES):
    """Average-pool the last axis of x in non-overlapping windows (drops the remainder).
    Equivalent to torch.nn.AvgPool1d(kernel_size=k, stride=k) used in the reference
    `fit_decoder`, but vectorised in numpy."""
    T = x.shape[-1]
    nbins = T // nframes_per_bin
    x = x[..., :nbins * nframes_per_bin]
    return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)
...
    tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
    neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS  # event rate in Hz
```

iii. CONVERSION_NOTES Step 5, decision 3: "Neural processing = reference decoder preprocessing: `gaussian_filter1d(trace, sigma=3 frames, axis=time)` then average-pool over 3 frames." One deliberate deviation is documented: "I apply the smoothing to the *continuous* session trace before dropping immobile frames (the reference smooths after concatenating retained frames); smoothing before filtering avoids mixing activity across temporal discontinuities and is strictly more correct." The ×30 scaling is noted as "a constant scale factor; irrelevant to the linear decoder but interpretable". The AI validated the whole chain by re-implementing the reference 15-bin Bayesian decoder on raw data and reproducing the authors' own `precomputed_results/within_decoding` errors to within 0.15 cm on 11 sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two quality filters, both copied from the reference `decode_position_within`:
- **Cell registration**: cells whose trace is all-NaN on that day are unregistered and dropped. The code asserts that NaN is never partial.
- **Activity sparsity**: of the registered cells, only those with **more than 5 events during moving (>5 cm/s) frames** of that session are kept (`cell_threshold = 5`). This removes 882 / 69,744 = 1.3 % of cell-sessions, leaving 68,862 (mean 332.7 neurons/session, range 112–562).
- **Timepoint QC**: bins whose mean smoothed speed ≤ 5 cm/s are removed from the neural (and output) stream, retaining 51.4 % of the recording.

ii.
```python
    nan_any = np.isnan(trace_day).any(axis=1)
    nan_all = np.isnan(trace_day).all(axis=1)
    assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
    registered = ~nan_all

    speed = compute_speed(position_day)
    moving = speed > V_THRESH

    events_moving = np.nansum(trace_day[:, moving], axis=1)
    keep_cells = registered & (events_moving > CELL_THRESHOLD)
...
    speed_binned = bin_frames(speed[np.newaxis, :])[0]
    moving_bins = speed_binned > V_THRESH
```
with
```python
def compute_speed(position_xy):
    """Speed in cm/s, smoothed exactly as in `decode_position_within`."""
    d = np.linalg.norm(np.diff(position_xy, axis=1), axis=0) * FPS
    speed = np.zeros(position_xy.shape[1], dtype=np.float64)
    speed[1:] = gaussian_filter1d(d, sigma=V_FILT_SIGMA)
    return speed
```

iii. CONVERSION_NOTES Step 5, decisions 4–5, and Step 10 Check 3(b)/(e): the reference line is `cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold`, and `vel_idx[d,1:] = gaussian_filter1d(||diff(behav)||*fps, sigma=v_filt_size) > v_thresh/bin_down`. The AI notes it uses the reference **decoding** curation rather than the paper's "inclusion of all cells" rule, "since our task is decoding". It justifies immobility removal both as the reference rule and on neuroscientific grounds: "hippocampal activity during immobility reflects replay/SWR rather than current position", adding that "the decoder used here classifies each timepoint independently, so removing timepoints does not break anything".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus/behavioural alignment event. Neural and behavioural streams were acquired on the same 30 Hz DAQ clock and the released arrays have identical frame counts per session, so they are aligned 1:1 with no shift. Trials are aligned to the start of each 1-minute window of session time; `metadata['temporal_alignment_event']` records this, with `off_start = 0.0`, `off_end = 60.0` s. Neural, input and output are sliced with the *same* `slice` object and the *same* moving-bin mask, so alignment is preserved by construction.

ii.
```python
    T = position_day.shape[1]
    assert trace_day.shape[1] == T, 'trace and position frame counts differ'
...
    sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
    m = moving_bins[sl]
    neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
    output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
...
    'temporal_alignment_event': (
        'Start of each 1-minute trial window, measured from the start of the '
        'continuous 40-min recording session (there are no experimenter-defined '
        'trials in this free-foraging task).'),
    'off_start': 0.0,
    'off_end': TRIAL_SECONDS,
```

iii. CONVERSION_NOTES Step 3/4: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz"; the reference `get_rate_maps` and `decode_position_within` index `position[day]` and `trace[day]` frame-for-frame. The AI concluded "streams are already aligned 1:1; **no additional alignment needed**", and verified it visually (panel (1,1) of `processing_*.png`, raw vs binned position superimposed with no lag) and indirectly through the 0.15 cm reproduction of the published decoding error, noting "a one-bin misalignment would inflate the error markedly".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms bins** (`time_bin_size = 100.0`). Yes — the native 30 Hz data are rebinned by a factor of 3 (Gaussian smoothing σ = 3 frames, then average-pooling over 3 frames), exactly the reference `temporal_bin_size = 3`. The bin size is identical for every trial and session; the trailing frames that do not fill a bin are dropped by `bin_frames`.

ii.
```python
FPS = 30.0                 # acquisition rate of both imaging and behaviour streams
TEMPORAL_BIN_FRAMES = 3    # reference `temporal_bin_size=3` -> 100 ms bins
TRACE_SIGMA_FRAMES = 3     # reference gaussian_filter1d(traces, sigma=temporal_bin_size)
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100 ms
...
    'time_bin_size': TIME_BIN_MS,
```

iii. CONVERSION_NOTES Step 5, decision 2: "Time bin = 100 ms (3 frames at 30 Hz): exactly the reference decoding bin (`temporal_bin_size=3` in `fit_decoder`/`test_decoder`). Identical for every trial and session." Step 3 lists it in the expected-statistics table with the code citation. The AI also noted the practical benefit (Step 2 of the trajectory): at 30 Hz the dataset would be ~10 GB, at 100 ms with the speed filter it is 3.4 GB.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. From the per-session `dat[animal]['blocked'][day]` field — the list of blocked (occluded) partition indices, or `[-1]` when nothing is blocked. The AI explicitly rejected deriving geometry from `envs` + the reference `get_env_mat(env)`, because several geometries were presented vertically flipped. `envs[d]` is still stored in `metadata['session_info']` for reference, but is not a decoder input.

ii.
```python
    blocked = dat['blocked']
    ...
    neural_trials, input_trials, output_trials, stats, aux = process_session(
        trace_day, position_day, blocked[d])
```

iii. CONVERSION_NOTES Step 4: "`blocked` is the per-session ground truth; several geometries were presented as the vertically flipped variant (cf. the `flipud` options in `get_environment_label`), so `get_env_mat` alone would mislabel those sessions. **Use `blocked`.**" Sanity check A2 confirms that every session's geometry equals `get_env_mat(env)` *or its flipud* for all 207 sessions.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked` is converted to a **9-dimensional binary vector** (1 = partition blocked, 0 = open) under the convention `p = 3·ybin + xbin`; `[-1]` maps to an all-zero vector (open square, 46 sessions). The vector is `float32` shape `(9,)`, constant within a session, and a copy is attached to every trial (static per trial, as the task requires). `input_names` are `blocked_partition_0 … _8`.

ii.
```python
def blocked_to_vector(blocked_day):
    """Convert the per-session `blocked` entry into a 9-d binary vector.
    1 = partition blocked (occluded), 0 = open.  Partition index p = 3*ybin + xbin,
    matching the `[[0, 1, 2], [3, 4, 5], [6, 7, 8]]` layout documented in the code
    README (verified against occupancy: mice are never in a blocked partition)."""
    b = np.atleast_1d(np.asarray(blocked_day[0]).ravel()).astype(float)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if b.size and b[0] >= 0:
        vec[b.astype(int)] = 1.0
    return vec
...
    geometry = blocked_to_vector(blocked_day)
    ...
        input_trials.append(geometry.copy())
```

iii. CONVERSION_NOTES Step 5, decision 7, plus the index-convention investigation (Step 4 and Step 10 "Issues found"): "Initially ambiguous whether the `blocked` index is `3*y+x` or `3*x+y`. Resolved empirically: occupancy in blocked partitions is 0.0000 for `3*y+x` versus ~0.19 for the alternative, across all 189 non-square sessions." The AI also flags that `blocked_partition_7` has range [0,0] in the converted data, and explains it: the only geometry that blocks partition 7 canonically (`l`) was always presented flipped.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. From `dat[animal]['position'][day]`, shape `(2, n_frames)` — DeepLabCut head-tracking x–y in cm, range exactly [0, 75]. The AI verified there are no NaNs in `position` for any animal.

ii.
```python
    positions = dat['position']
    ...
    position_day = np.asarray(positions[d])   # (2, T) in cm
    ...
    pos_binned = bin_frames(position_day)     # (2, nbins), cm
```

iii. CONVERSION_NOTES Step 2/4: "`position`: float64 (n_days, 2, n_frames), x-y in **cm**, range [0, 75]. **No NaNs anywhere** (verified for all animals in the scan)." Same field the reference `get_rate_maps` and `decode_position_within` use.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. (1) x and y are **averaged within each 100 ms bin** (the same `bin_frames` average-pool the reference applies to `behav` in `fit_decoder`); (2) the binned position is discretized into 9 classes (4-c); (3) bins failing the 5 cm/s speed criterion are dropped, using the same mask as the neural data; (4) the result is sliced into the same 1-minute trials and stored as `int64` of shape `(1, T)`. `output_names = ['position_bin']`, `output_values = [['x0y0', 'x1y0', ..., 'x2y2']]`.

ii.
```python
    pos_binned = bin_frames(position_day)            # (2, nbins), cm
    ...
    out_class = position_to_class(pos_binned)
    ...
        output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
```

iii. CONVERSION_NOTES Step 5 mapping table: "average x,y within each 100 ms bin, then discretize with 25 cm edges -> class `3*ybin+xbin` in 0..8", citing `decode_position_within`'s spatial binning and `fit_decoder`'s `AvgPool1d(3,3)` on `behav` followed by `.astype(int)`. Step 10 Check 3(d) records this as "identical" to the reference except for `n_bins = 3` instead of 15, which is mandated by the decoder task.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Floor division by a 25 cm bin width, using the reference `bin_down = (max + buffer) / n_bins` formula with `n_bins = 3` and `max = 75`, then clipped to 0..2 per axis, then flattened as `class = 3·ybin + xbin` ∈ {0..8}. The 25/50 cm edges coincide exactly with the physical partition walls, so the 9 output classes *are* the 9 arena partitions.

ii.
```python
N_SPATIAL_BINS = 3         # task requirement: 3 x 3 = 9 spatial bins
ARENA_SIZE = 75.0          # cm (paper: 75 x 75 cm)
BUFFER = 1e-15             # reference buffer in decode_position_within

def position_to_class(position_binned_xy):
    """Discretize (2, nbins) position in cm into 9 classes (3*ybin + xbin)."""
    bin_down = (ARENA_SIZE + BUFFER) / N_SPATIAL_BINS  # 25 cm, = reference bin_down formula
    xy = np.floor(position_binned_xy / bin_down).astype(int)
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)
```

iii. CONVERSION_NOTES Step 10 Check 3, note 3: "The reference bins position with `(max over all days + buffer)/n_bins`; the max is exactly 75.0 cm for every animal (verified), so my fixed 25 cm edges are identical to the reference formula and coincide with the physical partition walls." Decision 8: "Bin edges at 25 and 50 cm exactly coincide with the physical partition walls, so the output classes are the arena partitions (verified: essentially zero occupancy in blocked partitions)." The `clip` is documented in Step 10 Check 5 as a guard against a position of exactly 75.0 cm. Independent verification: sanity check C reproduced the classes with `np.digitize(pos, [25, 50])` and `np.array_equal`.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Perfectly, by construction: position and trace share the same 30 Hz frame index, are average-pooled with the same `bin_frames` (so bin *k* of the output covers frames 3k…3k+2, exactly like bin *k* of the neural data), are masked with the *same* `moving_bins` boolean, and are sliced with the *same* trial `slice`. The code asserts equal frame counts up front and asserts per trial that `neural.shape[1] == output.shape[1]`.

ii.
```python
    assert trace_day.shape[1] == T, 'trace and position frame counts differ'
...
    neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS
    pos_binned    = bin_frames(position_day)
    speed_binned  = bin_frames(speed[np.newaxis, :])[0]
...
        m = moving_bins[sl]
        neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
        output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
...
        for tr_i, tr in enumerate(sess):
            assert tr.shape[1] == data['output'][si_i][tr_i].shape[1]
```

iii. CONVERSION_NOTES Step 4: "streams are already aligned 1:1; no additional alignment needed", and Step 12 Check 2: "neural and output share the identical bin index by construction". Backed by the `processing_*.png` panels showing raw vs binned position with no lag, and by the 0.15 cm reproduction of the published within-session decoding errors.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Enumerated in CONVERSION_NOTES Step 10 Check 5:
- **Unregistered cells** appear as all-NaN rows and are dropped; an assertion checks NaN is never *partial* (it never fired over 207 sessions). `np.nansum` is used for the event count so NaN rows score 0 and fail the threshold anyway.
- **`blocked = [-1]`** (no partitions blocked, 46 sessions) → all-zero geometry vector.
- **First frame speed** is undefined by differencing → set to 0 (not moving), matching the reference, which leaves `vel_idx[0] = False`.
- **Position exactly at 75.0 cm** → `np.clip(..., 0, 2)` prevents a 4th bin (the `+BUFFER` in `bin_down` already covers it).
- **Trailing partial minute / partial time bin** → dropped (`nbins // TRIAL_BINS`, and `bin_frames` truncates the remainder).
- **Tracking jitter into a blocked partition** (DeepLabCut head marker tracked a few cm over a 25 cm insert) → detected, reported as a warning, and deliberately **not** corrected.
- **Degenerate trials/sessions** → trials with <30 moving bins dropped; sessions with <2 trials skipped.
- Final assertions verify no NaN/Inf anywhere and consistent shapes.

ii.
```python
    nan_any = np.isnan(trace_day).any(axis=1)
    nan_all = np.isnan(trace_day).all(axis=1)
    assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
    registered = ~nan_all
...
    events_moving = np.nansum(trace_day[:, moving], axis=1)
...
    if b.size and b[0] >= 0:
        vec[b.astype(int)] = 1.0
...
    speed = np.zeros(position_xy.shape[1], dtype=np.float64)
    speed[1:] = gaussian_filter1d(d, sigma=V_FILT_SIGMA)
...
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
...
        for tr_i, tr in enumerate(sess):
            assert tr.shape[0] == n
            assert tr.shape[1] == data['output'][si_i][tr_i].shape[1]
            assert np.isfinite(tr).all()
        geom = data['input'][si_i][0]
        classes = np.unique(np.concatenate([o.ravel() for o in data['output'][si_i]]))
        bad = [c for c in classes if geom[c] > 0]
        if bad:
            print(f'  WARNING session {si_i}: output classes {bad} are blocked partitions ...')
```

iii. CONVERSION_NOTES Step 10 Check 1 gives the reasoning for leaving the jitter in place: "(a) it is <0.008 of timepoints in the worst session and 4e-5 on average, (b) deleting timepoints on the basis of the output label would bias the dataset, and (c) the reference code applies no such correction (it 'cleans' predictions to the nearest visited bin only for reporting the Bayesian error)." The NaN/registration rule is validated against the paper: the non-NaN cell-session count is exactly 69,744, the paper's reported number of rate maps.

## 6-a. What are the most time-consuming steps of the code?

i. Measured by the script itself (`conversion_full_out.txt`): total **219.9 s** for the full dataset, split roughly into
- **joblib loading, ~91 s** (6.6–16.3 s per animal, 7 loads) — I/O + decompression of the whole animal dict, which includes `SFPs`, `centroids` and `maps` that are never used;
- **per-session processing, ~128 s** (mean 0.62 s/session × 207), dominated by `gaussian_filter1d` over the `(n_cells, ~72,000)` float array and the NaN scans;
- pickling the 3.41 GB output at the end.
The script prints per-animal load time, per-session processing time, the mean per-session time and total elapsed.

ii.
```python
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0
    ...
    print(f'{animal}: loaded in {t_load:.1f} s, {ndays} sessions, ...')
    ...
        ts = time.time()
        ...
        per_session_times.append(time.time() - ts)
    ...
    if per_session_times:
        print(f'mean processing time per session: {np.mean(per_session_times):.2f} s')
    print(f'total elapsed: {time.time() - t_start:.1f} s')
```

iii. CONVERSION_NOTES Steps 6/7: "Each animal file is loaded exactly once (9-16 s each); the per-session processing is 0.3-1 s"; estimated total ~4 min, actual 3.7 min — "well under the 15 min budget", so no further optimisation was pursued.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Essentially none of substance — all per-frame work is vectorized. The AI explicitly replaced reference-style per-frame Python loops (as in `get_rate_maps`/`fit_decoder`'s one-hot loop) with numpy `reshape`/`mean`/`floor`. What remains are cheap loops:
- the per-trial loop in `process_session` (≤40 iterations/session; it only slices views and applies a boolean mask, and cannot be fully vectorized because the mask makes trials ragged);
- the per-animal and per-session loops in `main()` — these are not vectorizable but *could* have been parallelized across the 128 available cores (e.g. `joblib.Parallel` over animals), which would have cut the 220 s roughly 7-fold;
- the per-class occupancy loop in `plot_processing` (9 iterations, plotting only);
- the final per-session/per-trial assertion loop, and the `bad`-class warning line which re-concatenates the whole session's outputs once per offending class.

ii.
```python
    # vectorized replacement for per-frame loops:
    return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)
    ...
    xy = np.floor(position_binned_xy / bin_down).astype(int)
    ...
    # remaining cheap loop:
    for t in range(n_full_trials):
        sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
        m = moving_bins[sl]
        ...
```

iii. CONVERSION_NOTES Step 6: "Inefficiency identified: naive per-frame python loops (as in the reference `get_rate_maps`) would be far too slow for 207 x 72,000 frames -> all binning/discretization is vectorised with numpy reshape/mean." Step 7 lists the speed-ups: "Vectorised binning/discretisation instead of per-frame loops (~100x vs reference-style loops)" and "One joblib load per animal (not per session) (~30x fewer loads)". Parallelism was considered unnecessary once the runtime estimate came in under the 15-minute budget.

## 6-c. What processing does the code repeat multiple times?

i. Only small, cheap duplications:
- `np.isnan(trace_day)` is evaluated **twice** (`.any(axis=1)` and `.all(axis=1)`) for the assertion, i.e. two full passes over an `(n_cells, 72,000)` array per session.
- `bin_frames` is called three times per session (trace, position, speed) — necessary, since the arrays differ.
- Speed is thresholded twice, once at frame resolution (`moving`, used for the cell filter) and once at bin resolution (`moving_bins`, used for timepoint selection) — deliberate, since the reference cell filter is defined on frames.
- In the final sanity-check block, the session's outputs are concatenated once for `classes` and then again inside the f-string for each offending class.
- `geometry.copy()` is stored once per trial (≈39 identical 9-float arrays per session) — required by the target format, which asks for one input array per trial.
No expensive computation is repeated: each file is read once, each session is smoothed and binned once.

ii.
```python
    nan_any = np.isnan(trace_day).any(axis=1)
    nan_all = np.isnan(trace_day).all(axis=1)
    assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
...
    moving = speed > V_THRESH            # frame resolution (cell filter)
    ...
    moving_bins = speed_binned > V_THRESH  # bin resolution (timepoint filter)
...
        bad = [c for c in classes if geom[c] > 0]
        if bad:
            print(f'  WARNING session {si_i}: output classes {bad} are blocked partitions '
                  f'(occupancy {[float(np.mean(np.concatenate([o.ravel() for o in data["output"][si_i]]) == c)) for c in bad]})')
```

iii. Not discussed explicitly in CONVERSION_NOTES beyond the general efficiency notes in Step 6 ("Each animal file is loaded exactly once…"). The double `isnan` is there to support the documented edge-case assertion ("Cells only partially NaN: asserted never to occur"), i.e. it is a deliberate correctness check rather than an oversight; the two speed thresholds are justified in Step 10 Check 3, note 2.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, all small relative to the 220 s runtime but real:
- **Whole-file joblib loading**: `joblib.load(...)` materialises the entire animal dict, including `SFPs` `(35,35,n_cells,n_days)`, `centroids` and the 15×15 `maps`, none of which are used. The reference `load_dat` has a `to_convert` argument precisely to avoid this, and an HDF5 reader would read only the needed datasets lazily. This accounts for most of the ~91 s of load time and a large transient memory footprint.
- **The `aux` dict** (`speed`, `moving`, `speed_binned`, `moving_bins`, `pos_binned`, `out_class`, `neural_binned`, `keep_cells`, `registered`) is assembled and returned for **every** session even though it is only consumed by `plot_processing` for at most 2 sessions in `--show-processing` mode.
- **`* FPS`** rescaling of the binned traces to Hz: a global constant factor that the PCA + linear decoder is invariant to; purely cosmetic.
- **`stats['frac_moving_bins']`** is computed and never used; `envs` is loaded and stored in `session_info` but is not a decoder variable (kept for provenance).
- `np.ascontiguousarray` on each neural trial forces an extra copy (it does make the pickle compact and the decoder's strided reads faster, so this one is arguably worth it).

ii.
```python
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]   # loads SFPs, centroids, maps too
...
    neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS  # constant rescale
...
    stats['frac_moving_bins'] = float(moving_bins.mean())           # never used
...
    aux = dict(speed=speed, moving=moving, speed_binned=speed_binned,
               moving_bins=moving_bins, pos_binned=pos_binned, out_class=out_class,
               neural_binned=neural_binned, keep_cells=keep_cells, registered=registered)
    return neural_trials, input_trials, output_trials, stats, aux
```

iii. CONVERSION_NOTES does not flag these as waste. The ×30 scaling is defended in Step 5 decision 3 as "a constant scale factor; irrelevant to the linear decoder but interpretable"; `envs` is kept deliberately ("kept for reference, not a decoder input"). The AI's efficiency justification is that the total runtime (220 s) was far inside the 15-minute budget, so it optimised only the asymptotically important parts (vectorised binning, one load per animal) and left the constant-factor waste in place. Memory pressure was checked and dismissed early in the trajectory ("System: 1 TB RAM, 128 cores, L4 GPU").
