# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the **joblib** version of the dataset (one file per animal, `/app/data/QLAK-CA1-XX`, no extension), which is the format the reference code's `load_dat(..., format="joblib")` uses by default. The seven animal IDs are hard-coded in a module-level constant (taken from `main.py`'s `animals` list). Each file is a nested dict `{animal: {...}}`; the AI pulls four fields out of it — `trace` (n_days, n_cells, n_frames), `position` (n_days, 2, n_frames), `envs` (n_days,) and `blocked` (list of n_days). Each animal file is loaded exactly once and **all** of that animal's sessions are processed from that single load; the seven animals are processed in parallel with `joblib.Parallel`. The AI separately verified (Step 2 / Step 10 of CONVERSION_NOTES.md) that the `.mat` (MATLAB v7.3) files contain the same content, and used the `.mat` files read through `h5py` as an *independent* path for its sanity checks.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def process_animal(animal, days=None, show_processing=False, plot_dir='/app', verbose=True):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
    if days is None:
        days = list(range(trace.shape[0]))
    ...
    for day in days:
        res = process_session(trace[day], position[day], blocked[day], str(envs[day]),
                              session_id=f'{animal}_day{day:02d}', ...)
```
```python
    jobs = [(a, None) for a in ANIMALS]
    ...
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_animal)(a, n, args.show_processing, plot_dir) for a, n in jobs)
```

iii. From CONVERSION_NOTES.md Step 1/Step 10 Check 3: `load_dat` in `georepca1/src/utils.py:61` is the reference loader and its default format is `joblib`; `mat2joblib`/`save_dat` show the joblib file is produced *from* the `.mat` file, so "the joblib and `.mat` files hold the *same* data". Loading each animal once (rather than per session) was an explicit efficiency decision: "`joblib.load` of one animal takes 6-16 s and the decompressed arrays are 9-17 GB, so each animal is loaded exactly once and all of its sessions are processed from that one load." The animal list matches the paper's 7 mice / 207 sessions / 5,413 cells, which the AI verified against the methods text.

## 1-b. How are the data split into subjects?

i. One subject = one animal file = one entry of the hard-coded `ANIMALS` list. `data['subjects']` is that list (all 7 animals, in file order), and `subject_idx` for each session is `ANIMALS.index(animal)`. The animal ID is also carried in each session's `session_id` (`QLAK-CA1-08_day02`) and in `metadata['session_info']`.

ii.
```python
'subjects': list(ANIMALS), 'subject_idx': [],
...
for sessions in results:
    for s in sessions:
        ...
        data['subject_idx'].append(ANIMALS.index(s['animal']))
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. Each dataset file corresponds to one mouse (code/README.md: "The dataset ... are given names of animal IDs"). CONVERSION_NOTES.md Step 2 verifies 7 animals and that per-animal cell counts (mean 773.3, min 515, max 952) reproduce the paper's "mean number of cells per animal = 773 ± 68 SE, minimum 515, maximum 952".

## 1-c. How are the data split into sessions?

i. One session = one recording day = one index along axis 0 of `trace`/`position`/`envs`/`blocked`. All days of an animal are processed (`days = list(range(trace.shape[0]))`), giving 31 sessions for six animals and 21 for `QLAK-CA1-51` = 207 sessions. No session-level exclusion rule is applied, except a defensive guard that would skip a session with fewer than 2 usable trials (which never fired: all 207 sessions are kept). Each session's geometry name and blocked set are recorded in `metadata['session_info']`.

ii.
```python
    if days is None:
        days = list(range(trace.shape[0]))
    ...
    res = process_session(trace[day], position[day], blocked[day], str(envs[day]),
                          session_id=f'{animal}_day{day:02d}', ...)
```
```python
        for s in sessions:
            if len(s['neural']) < 2:
                print(f"  !! skipping session {s['session_id']}: only {len(s['neural'])} usable trials")
                continue
```

iii. CONVERSION_NOTES.md Step 4: "6 animals × 31 + 1 animal (QLAK-CA1-51) × 21 ... 6*31 + 21 = **207** exactly" matching the paper's "207 sessions"; "All sessions were 40 min, and one session was recorded per day". Key Decision 10: "All 7 animals / 207 sessions are kept - no session-level exclusions exist in the reference." The `<2 trials` guard exists because the target format requires "at least two trials within each session".

## 1-d. How are the data split into trials?

i. The source recordings are continuous 40-min sessions with no trial structure. Following the task specification, the AI cuts each session into **consecutive 1-minute blocks**. Because it works in 100 ms bins (see 2-e), a full trial is `TRIAL_BINS = 600` bins. The trailing partial block (~18 s, 555 bins) is **kept** as a shorter trial rather than discarded. Within each block only the bins passing the speed filter are retained (see 1-e), so trials are variable length (mean 315 bins, min 30, max 561). Result: 8,163 trials, 39.4 per session (range 22-41).

ii.
```python
TRIAL_SECONDS = 60.0                                   # 1-minute trials (task specification)
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TBIN))    # 600 bins of 100 ms
MIN_TRIAL_BINS = 30                                    # drop trials with < 3 s of movement
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

iii. CONVERSION_NOTES.md Key Decision 6: "**Trials = consecutive 1 min blocks (600 bins of 100 ms) of the session**, as required by the task. The final partial block of each session (~18 s) is kept as a shorter trial so that no data is discarded." Step 10 Check 5 notes that the ≤2 trailing frames that do not fill a complete 100 ms bin are dropped (<0.003% of a session) so that all bins are complete. The metadata records the trial convention: `temporal_alignment_event` = "Start of each 1-minute trial ... within a block only the time bins in which the mouse ran faster than 5 cm/s are kept, so trials contain <= 600 time bins."

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both derived from the reference decoding analysis `decode_position_within`:
  1. **Speed filter (sample level)**: speed is computed as `gaussian_filter1d(‖diff(position)·30‖, sigma=5 frames)`, averaged within each 100 ms bin, and only bins with speed **> 5 cm/s** are kept. This removes ~48% of bins (51.9% retained, matching the measured fraction of time moving). Bins are kept/dropped whole, so retained samples are contiguous 100 ms chunks of recording, but a trial's retained samples are not contiguous in wall-clock time.
  2. **Trial-level drop**: a 1-min block that retains fewer than **30 bins (3 s) of movement** is discarded as unusable. This drops 2,162 bins (0.08% of retained bins) and is why some sessions have 22-39 rather than 40-41 trials.
No place-cell / session-level / geometry-level exclusions are applied.

ii.
```python
V_SIGMA = 5.0          # frames, Gaussian smoothing of the speed trace (v_filt_size)
V_THRESH = 5.0         # cm/s, speed threshold (v_thresh)

def compute_speed(position):
    speed = np.zeros(position.shape[1], dtype=np.float64)
    speed[1:] = gaussian_filter1d(np.linalg.norm(np.diff(position, axis=1) * FPS, axis=0),
                                  sigma=V_SIGMA)
    return speed
...
    speed = compute_speed(position)
    speed_b = bin_time(speed, nb)                                  # (nb,)
    moving = speed_b > V_THRESH                                    # velocity filter (v_thresh=5)
...
        sel = idx[moving[idx]]
        if len(sel) < MIN_TRIAL_BINS:
            n_dropped += 1
            continue
```

iii. CONVERSION_NOTES.md Key Decision 4: "**Speed > 5 cm/s filter (`v_thresh=5`, speed smoothed with sigma = 5 frames).** Matches the reference decoding analysis. Verified in a pilot to improve balanced accuracy (0.488 vs 0.424 without). Bins are kept or dropped as whole 100 ms bins ... so every retained sample is a contiguous 100 ms of recording. This makes trials variable-length, which the target format allows." Step 1 documents the source: `vel_idx[d,1:] = gaussian_filter1d(norm(diff(behav)*fps), sigma=5) > v_thresh/bin_down`, i.e. speed > 5 cm/s, "Only these 'moving' frames are used for fitting *and* testing." The one acknowledged deviation (Step 10 Check 3, row b'): the threshold is applied per 100 ms bin instead of per frame.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively the `trace` field: `(n_days, n_cells, n_frames)`, values in {0, 1} where the cell was registered that day and NaN otherwise. These are the binarised calcium-transient **rising-phase event trains** shipped with the dataset. No ΔF/F is computed, and no other neural field (`SFPs`, `centroids`, `maps`) is used for the neural stream.

ii.
```python
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
    ...
    res = process_session(trace[day], position[day], ...)
...
def process_session(trace, position, blocked_entry, env, session_id, ...):
    """
        trace : (n_cells, n_frames) binarised event trains, NaN for cells not registered that day
    """
```

iii. CONVERSION_NOTES.md Key Decision 1: "**Neural signal = binarized event train, not dF/F.** The shipped `trace` field *is* the rising-phase-binarized signal that the paper 'treated as the firing rate'; no dF/F step is needed or possible from these files." Step 3 quotes the methods: the binary rising-phase vector "was treated as the firing rate in all further analyses". Step 2 verified traces are exactly {0,1} where registered.

## 2-b. How is the `neural` data processed?

i. Reproducing the reference decoder's `fit_decoder`/`test_decoder` preprocessing: (1) select registered cells; (2) truncate to a whole number of 3-frame bins; (3) cast to float32; (4) smooth along time with a **Gaussian, sigma = 3 frames**; (5) **average-pool over 3 frames** (100 ms); (6) multiply by 30 to express the result in **events/s (Hz)**; (7) apply the cell curation mask; (8) index the moving bins per trial. The one deliberate ordering difference from the reference: the AI smooths the *continuous* session trace and then selects moving bins, whereas the reference smooths after concatenating the moving frames.

ii.
```python
TBIN = 3               # frames per time bin -> 100 ms  (fit_decoder: temporal_bin_size=3)
TRACE_SIGMA = 3.0      # frames, Gaussian smoothing of the traces (fit_decoder)

def bin_time(x, nbins, how='mean'):
    x = x[..., :nbins * TBIN]
    x = x.reshape(x.shape[:-1] + (nbins, TBIN))
    return x.mean(axis=-1) if how == 'mean' else x.sum(axis=-1)
...
    registered = ~np.isnan(trace[:, 0])
    raw = trace[registered][:, :nb * TBIN].astype(np.float32)      # (n_reg, nb*TBIN) binary
    events_b = bin_time(raw, nb, how='sum')                        # events per 100 ms bin (curation)

    smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
    neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)     # (n_reg, nb) in events/s
```

iii. CONVERSION_NOTES.md Key Decisions 2 and 3: "**Time bin = 100 ms (3 frames at 30 Hz), with a 3-frame Gaussian smoothing before pooling.** This is exactly the reference decoder's `temporal_bin_size=3` + `gaussian_filter1d(sigma=3)`. I apply the smoothing to the *continuous* session trace before the velocity selection (the reference smooths after concatenating the moving frames); smoothing before selection is strictly more correct temporally and is the only defensible order once the data are cut into trials." And: "**Units: events/s (Hz)**, i.e. the pooled mean binary value × 30 fps, as `get_rate_maps` does." A scale check is reported in Step 8 (rescaling by 0.2× / 5× changed validation accuracy by ≤0.02). Step 10 Check 2 verifies that total event mass is conserved by smoothing+binning (`sum(rate)/30*3 == #events`, e.g. 273,060.0 vs 273,060.0) and that every trial matches an independent recomputation from the `.mat` files under `np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two criteria, both from `decode_position_within`:
  1. **Registration**: cells whose trace is NaN on that day (not registered by CellReg) are dropped. NaN is all-or-none per (cell, day), which the AI verified, so it tests only frame 0.
  2. **Activity**: of the registered cells, only those emitting **more than 5 events during the retained (moving) bins** are kept (`cell_threshold=5`).
Result: 68,860 of the 69,744 registered cell-sessions kept (98.7%); mean 332.7 neurons/session (112-562); 5,374 of the 5,413 unique cells appear somewhere. No place-cell / split-half selection is applied.

ii.
```python
CELL_THRESH = 5        # events during the retained bins (decode_position_within: cell_threshold)
...
    registered = ~np.isnan(trace[:, 0])                            # NaN is all-or-none per (cell, day)
    raw = trace[registered][:, :nb * TBIN].astype(np.float32)
    events_b = bin_time(raw, nb, how='sum')
    ...
    # cell curation: > 5 events during the retained (moving) bins  (reference: cell_threshold=5)
    active = events_b[:, moving].sum(axis=1) > CELL_THRESH
    cell_idx = np.where(registered)[0][active]
    neural = neural[active]
```

iii. CONVERSION_NOTES.md Key Decision 5: "**Cell curation: registered that day AND > 5 events during the retained moving bins**, matching `decode_position_within`'s `cell_idx`. No place-cell selection (the paper includes all cells)." Step 1 note 3 spells out the reference rule: `cell_idx[d] = np.sum(traces[:,:,d][vel_idx[d]], axis=0) > cell_threshold = 5`, and observes "NaN traces (cells not registered that day) fail this test automatically, so unregistered cells are dropped" — the AI makes the NaN handling explicit instead of relying on NaN comparison semantics. Step 9/Step 10 Check 4 report the resulting counts against the paper's 69,744 / 5,413.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural event to align to — the recordings are continuous 40-min free-exploration sessions with no trial structure. The AI therefore defines the alignment event as the **start of each 1-minute block**, with `off_start = 0.0 s` and `off_end = 60.0 s`, and states this explicitly in the metadata. Alignment *between streams* is frame-for-frame: the imaging and behaviour streams were acquired simultaneously at 30 Hz and timestamped, both arrays have identical length (asserted in code), and both are binned with the same bin edges and indexed with the same bin selector `sel`.

ii.
```python
    n_cells, n_frames = trace.shape
    nb = n_frames // TBIN                      # number of complete 100 ms bins
    assert position.shape[1] == n_frames       # imaging and behaviour streams are frame-aligned
...
        trials_neural.append(np.ascontiguousarray(neural[:, sel]))
        trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```
```python
        'temporal_alignment_event':
            'Start of each 1-minute trial. Sessions are continuous 40-min recordings cut into consecutive '
            '1-min blocks; within a block only the time bins in which the mouse ran faster than 5 cm/s are '
            'kept, so trials contain <= 600 time bins.',
        'off_start': 0.0,
        'off_end': TRIAL_SECONDS,
```

iii. CONVERSION_NOTES.md Step 3: "Position from DeepLabCut head tracking, acquired at 30 Hz simultaneously with the imaging stream and timestamped for post-hoc alignment -> **position frame i is aligned to trace frame i** (both streams in the files have identical length)." Step 10 Check 3 row (c): "position frame i <-> trace frame i; both avg-pooled with `AvgPool1d(3,3)`" — same in the AI's code via `bin_time()` on both streams with the same bin edges. Alignment was also verified visually (processing plots panels 1, 5, 6, 7) and numerically against the `.mat` files.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms** (`time_bin_size = 100.0`). Yes — rebinning is applied: the native 30 Hz (33.33 ms) streams are Gaussian-smoothed (sigma = 3 frames) and then average-pooled over 3 frames, exactly the reference decoder's `temporal_bin_size=3`. Position is average-pooled over the same 3-frame windows before discretisation, and the speed trace is averaged within each bin before thresholding, so all three streams share the same bin edges. The ≤2 trailing frames that do not fill a bin are dropped.

ii.
```python
FPS = 30.0             # acquisition rate of both the imaging and the behaviour stream (methods)
TBIN = 3               # frames per time bin -> 100 ms  (fit_decoder: temporal_bin_size=3)
...
    nb = n_frames // TBIN                      # number of complete 100 ms bins
    speed_b = bin_time(speed, nb)
    pos_b = bin_time(position, nb)
    neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
...
        'time_bin_size': 1000.0 * TBIN / FPS,          # 100.0 ms
```

iii. CONVERSION_NOTES.md Key Decision 2 and Step 3's statistics table: "Neural data time bin (decoding) | 3 frames = 100 ms | `fit_decoder(..., temporal_bin_size=3)`" and "Behaviour data time bin (decoding) | same 100 ms | `test_decoder` avg-pools position identically". Step 10 Check 3 row (d) confirms the match with the reference. A secondary benefit noted in Step 6/7 is the 3× reduction in stored data (3.44 GB).

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The per-session `blocked` field — the indices of the occluded partitions in the 3×3 grid laid out `[[0,1,2],[3,4,5],[6,7,8]]`, with `-1` meaning nothing is blocked (square). The AI cross-checked this against the `envs` string labels and against `get_env_mat(env)` from the reference code, and validated it against the animals' actual occupancy.

ii.
```python
    trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
    ...
        res = process_session(trace[day], position[day], blocked[day], str(envs[day]), ...)
...
def blocked_vector(blocked_entry):
    """
    `blocked` holds the indices of the occluded partitions of the 3x3 grid, laid out as
    [[0,1,2],[3,4,5],[6,7,8]] (code/README.md).  A value of -1 means nothing is blocked (square).
    """
```

iii. CONVERSION_NOTES.md Step 4 documents a genuine discrepancy the AI resolved: `get_env_mat('l')` implies blocked {4,5,7,8} but the data's `blocked` field for 'l' is {1,2,4,5}; "`blocked` equals the zeros of **`flipud(get_env_mat(env))`** for all 10 geometries ... **I use the per-session `blocked` field**, which I verified directly against occupancy." The verification: with `partition = 3*floor(y/25) + floor(x/25)` the occupancy of every blocked partition is essentially zero (14 of 4.46 M frames), and the 3×3 block sums of the precomputed 15×15 `maps['sampling']` reproduce the open/blocked mask.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked` indices are converted to a **9-dimensional binary vector** (1 = blocked, 0 = open), with the `-1` sentinel filtered out so the square geometry gives an all-zero vector. The vector is constant within a session and is replicated (copied) into every trial of that session as a static `(9,)` input. `input_names` are `blocked_r0c0 ... blocked_r2c2`, matching the output value names. The dataset contains exactly 10 distinct input vectors, one per geometry; dimension `r2c1` (partition 7) is 0 everywhere because it is open in all 10 geometries.

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
    blocked_vec, blocked_ids = blocked_vector(blocked_entry)
    ...
        trials_input.append(blocked_vec.copy())
```

iii. CONVERSION_NOTES.md Key Decision 8: "**Input = 9-dim binary geometry vector (1 = blocked)**, static per trial, exactly the Decoder Task specification. It is the same information as the geometry name but in the decoder-usable form that tells the decoder which partitions are impossible." Step 10 Check 2 confirms the input "equals the `blocked` vector read from the `.mat` file, identical in every trial", and Check 4 confirms "Exactly 10 unique geometry input vectors, each corresponding to exactly one environment name."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field: `(n_days, 2, n_frames)` float64 head-tracking coordinates in cm (DeepLabCut), range exactly [0, 75] in both dimensions with no NaNs. Dimension 0 is treated as x (West→East, column) and dimension 1 as y (North→South, row); this axis convention was verified against the `blocked` field and the precomputed occupancy maps.

ii.
```python
    trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
    ...
    res = process_session(trace[day], position[day], blocked[day], str(envs[day]), ...)
...
def process_session(trace, position, ...):
    """ position : (2, n_frames) head position in cm """
```

iii. CONVERSION_NOTES.md Step 2/Step 4: "`position` | (n_days, 2, n_frames) float64 | x-y head position in cm, range exactly [0, 75] in both dims, **no NaNs**"; and the verified consistency "This confirms both the axis convention (dim 0 = West-East = column, dim 1 = North-South = row) and the `blocked` semantics."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is average-pooled over the same 3-frame (100 ms) windows as the neural data, then converted to a partition index (see 4-c), then indexed by the moving-bin selector so that each trial's output is `(1, T)` int64 with the same `T` as the trial's neural matrix. No smoothing, interpolation or unit conversion is applied to the position itself (it is already in cm).

ii.
```python
    pos_b = bin_time(position, nb)                                 # (2, nb), cm
    part, n_fixed = discretize_position(pos_b, blocked_ids)        # (nb,) partition index 0..8
    ...
        trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "`position` -> `output[session][trial]`, shape (1, T): average-pool 3 frames -> `partition = 3*floor(y/25) + floor(x/25)` ... Reference: `test_decoder` (avg-pool then int)". The AI's Step 10 Check 3 row (f) compares this to the reference `decode_position_within`, which divides by a global `bin_down` and average-pools before `astype(int)`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. A **fixed physical 3×3 grid** on the 75×75 cm arena: `col = clip(floor(x/25), 0, 2)`, `row = clip(floor(y/25), 0, 2)`, `partition = 3*row + col`, giving 9 categories named `r0c0 ... r2c2`. The `clip` handles the 198 frames (of 14.9 M) sitting exactly at 75.0 cm, which would otherwise index bin 3. Additionally, binned positions that land in a partition marked blocked for that session are **reassigned to the nearest open partition** (by Euclidean distance to partition centres), mirroring the reference decoder's "cleaning" of positions onto valid bins; this affected 152 bins across the whole dataset (145 of them in a single session, `QLAK-CA1-51` day 04, 'bit donut').

ii.
```python
ARENA_SIZE = 75.0; N_GRID = 3; PART_SIZE = ARENA_SIZE / N_GRID   # 25 cm

def discretize_position(pos_binned, blocked_ids):
    xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    part = N_GRID * yb + xb

    n_fixed = 0
    if len(blocked_ids):
        bad = np.isin(part, blocked_ids)
        n_fixed = int(bad.sum())
        if n_fixed:
            open_ids = np.array([p for p in range(N_GRID * N_GRID) if p not in blocked_ids])
            d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None, :, :],
                               axis=2)
            part[bad] = open_ids[np.argmin(d, axis=1)]
    return part, n_fixed
```

iii. CONVERSION_NOTES.md Key Decisions 7 and 9: "**Output = single categorical variable with 9 values** (the 3x3 partition), time-varying, as required by the Decoder Task. Value names `r0c0 ... r2c2` (row = North-South = position dim 1, column = West-East = position dim 0), matching the paper's partition ordering"; and "**Spatial binning uses the fixed physical grid (25 cm)** rather than a per-session position maximum, so that bin identity is comparable across sessions and consistent with `blocked`." Step 10 Check 3 row (f) argues equivalence with the reference: "the global position max is *exactly* 75.0 for all 7 animals, so `(75+buffer)/3 = 25.0 cm` is identical. The only difference is the `clip`, which fixes a genuine edge case". Check 4 reports the strongest validation: without the velocity filter, the per-session output distribution correlates **1.0000 in all 207 sessions** with the 3×3 block sums of the authors' own precomputed occupancy maps. (Note: the notes' claim that the blocked-partition reassignment affects "14 frames in the whole dataset, all in `QLAK-CA1-74` day 9" is stale — the conversion log shows 152 binned samples across 4 sessions.)

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, then bin-for-bin. Both streams have identical frame counts (asserted), are truncated to the same `nb*TBIN` frames, are binned with identical bin edges by `bin_time`, and within each trial are indexed with the *same* selector array `sel` of retained moving bins. There is no lag, shift or interpolation. This guarantees `neural[t]` and `output[t]` describe the same 100 ms of recording.

ii.
```python
    assert position.shape[1] == n_frames       # imaging and behaviour streams are frame-aligned
    nb = n_frames // TBIN
    ...
    pos_b = bin_time(position, nb)
    neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
    ...
        idx = np.arange(start, min(start + TRIAL_BINS, nb))
        sel = idx[moving[idx]]
        ...
        trials_neural.append(np.ascontiguousarray(neural[:, sel]))
        trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
```

iii. CONVERSION_NOTES.md Step 3: both streams were acquired by the same DAQ at 30 Hz and timestamped for post-hoc alignment, so frame i ↔ frame i. Step 7's plot review states "raw 30 Hz position overlaid with the 100 ms binned position - no offset, no lag" and "raw binary events of 3 cells overlaid with their smoothed/binned rate - peaks coincide exactly". Step 10 Check 2 verifies every trial's output array equals an independent recomputation from the `.mat` files (`np.array_equal`) and that trial lengths match between neural and output.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI enumerated and handled six edge cases:
  - **Unregistered cells (NaN traces, ~58% of cell-days)**: dropped per session via `~np.isnan(trace[:, 0])`, after verifying NaN is all-or-none within a (cell, day).
  - **Frame counts not divisible by 3**: the ≤2 trailing frames are dropped so every 100 ms bin is complete.
  - **Positions exactly at the 75 cm arena edge** (198 of 14.9 M frames): `np.clip` keeps them in bin 2 instead of an out-of-range bin 3.
  - **Binned positions falling inside a blocked partition** (152 bins): reassigned to the nearest open partition.
  - **Blocks with almost no movement**: 1-min blocks retaining <30 bins are dropped; sessions with <2 usable trials would be skipped (none occurred).
  - **Frame 0 has no velocity estimate**: left at 0 so it is never "moving", matching the reference's `vel_idx[0] = False`.
  A known residual: 2 sessions contain one neuron that is silent in every retained trial (its >5 events all fell in dropped blocks); the AI left it in rather than adding a non-reference second filtering pass.

ii.
```python
    registered = ~np.isnan(trace[:, 0])                            # NaN is all-or-none per (cell, day)
    raw = trace[registered][:, :nb * TBIN].astype(np.float32)
...
def compute_speed(position):
    speed = np.zeros(position.shape[1], dtype=np.float64)
    speed[1:] = gaussian_filter1d(...)   # frame 0 has no speed estimate and is left at 0
...
    xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    ...
            part[bad] = open_ids[np.argmin(d, axis=1)]
...
        if len(sel) < MIN_TRIAL_BINS:
            n_dropped += 1
            continue
...
            if len(s['neural']) < 2:
                print(f"  !! skipping session {s['session_id']}: only {len(s['neural'])} usable trials")
                continue
```

iii. CONVERSION_NOTES.md Step 10 Check 5 ("Edge cases") lists each of these with counts, and Step 2 records the data-quality screening that motivated them: "traces are exactly {0,1} where registered; position has no NaN and no frozen (dropout) segments longer than 2 frames; recordings are not zero-padded at either end." On the silent-neuron case: "This is harmless - the format checker reports no warning and the decoder's SVD initialisation handles zero rows - so it was left as is rather than adding a second, non-reference filtering pass."

## 6-a. What are the most time-consuming steps of the code?

i. Measured (printed by the script itself): **loading the animal joblib files** — 6.5-16.4 s per animal, 7 animals — and **per-session processing** at 0.54-0.59 s/session, dominated by the `gaussian_filter1d` over the (n_registered × n_frames) trace, plus **pickling the 3.44 GB output**. Total full-dataset wall clock: **40.7 s** with 7 parallel workers.

ii.
```python
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0
    ...
    t_proc = time.time() - t1
    print(f'[{animal}] processed {len(days)} sessions in {t_proc:.1f}s '
          f'({t_proc / max(len(days), 1):.2f}s/session), load {t_load:.1f}s', flush=True)
...
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) '
          f'in {time.time() - t_start:.1f}s total')
```

iii. CONVERSION_NOTES.md Step 6: "`joblib.load` of one animal takes 6-16 s and the decompressed arrays are 9-17 GB, so each animal is loaded exactly once and all of its sessions are processed from that one load. The traces are float64 in the file; they are cast to float32 before the Gaussian smoothing (**the dominant cost**), and only the *registered* cells (about 40%) are smoothed." Step 7's timing table breaks the 41 s down into ~16 s load (parallel), ~20 s processing, ~20 s assemble+pickle.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Essentially none remain in the hot path. All per-frame work is vectorised: `bin_time` is a single `reshape(...).mean(-1)`; speed is one `np.diff` + `np.linalg.norm` + `gaussian_filter1d`; discretisation is pure array arithmetic; the nearest-open-partition fix is a single broadcast distance computation plus `argmin`. The remaining Python loops are all O(sessions) or O(trials-per-session): the `for day in days:` loop (≤31 iterations, each doing heavy vectorised work), the `for start in range(0, nb, TRIAL_BINS)` trial loop (≤41 iterations per session), and the assembly loops in `main()`. The trial loop could in principle be replaced by `np.split`-style index bucketing, but it costs a negligible fraction of the runtime. The `PART_CENTERS` comprehension runs once at import.

ii.
```python
def bin_time(x, nbins, how='mean'):
    x = x[..., :nbins * TBIN]
    x = x.reshape(x.shape[:-1] + (nbins, TBIN))
    return x.mean(axis=-1) if how == 'mean' else x.sum(axis=-1)
...
            d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None, :, :],
                               axis=2)
            part[bad] = open_ids[np.argmin(d, axis=1)]
...
    for start in range(0, nb, TRIAL_BINS):        # <= 41 iterations/session
        idx = np.arange(start, min(start + TRIAL_BINS, nb))
        sel = idx[moving[idx]]
```

iii. CONVERSION_NOTES.md Step 6, "Code inefficiencies identified": "The naive implementation would loop over frames (as the reference's `get_rate_maps`/`fit_decoder` do); all binning is instead done with a single `reshape(...).mean(-1)`." The Step 7 speed-up table credits "vectorised binning instead of per-frame loops" with "~50x on the binning step".

## 6-c. What processing does the code repeat multiple times?

i. Minor, bounded repetition only:
  - The raw trace is binned **twice**: once with `how='sum'` (`events_b`, used only for cell curation) and once with `how='mean'` after smoothing (`neural`). The two serve different purposes but traverse the same array.
  - Gaussian smoothing is applied to **all registered cells before curation**, so the ~1.3% of cells later dropped by the >5-event rule are smoothed needlessly.
  - `np.ascontiguousarray(neural[:, sel])` copies each trial's slice out of the session array, so the retained neural data is materialised twice at peak.
  - In `--show-processing` mode, `np.nansum(trace, axis=1)` re-traverses the full raw trace for the curation histogram.
  - `blocked_vec.copy()` is stored once per trial (~40 copies of a 9-float vector per session) rather than shared — trivial in cost.
  Nothing is recomputed across sessions or animals: each file is read once and each session's speed/position/trace pipeline runs exactly once.

ii.
```python
    events_b = bin_time(raw, nb, how='sum')                        # pass 1 over `raw`
    smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)   # pass 2 (all registered cells)
    neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)
    ...
    neural = neural[active]                                        # curation applied after smoothing
    ...
        trials_neural.append(np.ascontiguousarray(neural[:, sel]))
        trials_input.append(blocked_vec.copy())
```

iii. The AI did not flag these specifically; CONVERSION_NOTES.md Step 6 documents the repetitions it *did* eliminate ("one file load per animal instead of per session ... ~20x fewer seconds of I/O", "smooth only registered cells, in float32 ... ~3x on the dominant smoothing step"). It judged the remaining 41 s total runtime as well inside the instructions' 15-minute budget, so no further deduplication was pursued.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small amounts of computed-then-discarded work:
  - **Smoothing and binning are computed for all bins of the session, but ~48% of bins are then dropped by the speed filter** and a further 0.08% by the <30-bin trial rule. This is the largest discarded fraction — though it is unavoidable given the deliberate decision to smooth the continuous trace *before* selecting moving bins.
  - Smoothing and binning of the ~1.3% of registered cells subsequently removed by the `cell_threshold=5` rule.
  - `cell_idx` (the source cell index of every kept neuron, 68,860 integers) is computed and stored in `metadata['session_info']`; the decoder never reads it. Likewise `n_registered`, `n_bins_total`, `n_bins_kept`, `n_dropped_trials`, `n_blocked_fixed` and `env` are bookkeeping only.
  - `trial_bin_idx` is accumulated in every run but only consumed by the `--show-processing` plotting path.
  - The `(blocked_vec, blocked_ids)` tuple recomputes `sorted(...)` per session; the returned `n_fixed` counter is diagnostic only.
  None of this is on the critical path: the whole conversion takes 41 s.

ii.
```python
    trials_neural, trials_input, trials_output, trial_bin_idx = [], [], [], []
    ...
        trial_bin_idx.append(sel)          # used only by _plot_processing
    ...
    result = {..., 'cell_idx': cell_idx, 'n_bins': int(nb), 'n_moving': int(moving.sum()),
              'n_dropped_trials': n_dropped, 'n_blocked_fixed': n_fixed}
...
            session_info.append({... 'cell_idx': s['cell_idx'].tolist()})
```

iii. Not flagged as waste by the AI; the extra fields are deliberate provenance for the sanity checks. CONVERSION_NOTES.md Step 10 Check 2 relies on `session_info` (`cell_idx`, `n_neurons`, geometry, trial counts) to re-derive the pipeline independently from the `.mat` files, and `trial_bin_idx` underpins the `--show-processing` trial-structure panel required by the instructions. The AI's stated position on smoothing order (Key Decision 2) explains why the full-session smoothing — and hence the discarded ~48% — is intentional: "smoothing across a discontinuous concatenation mixes samples minutes apart. Smoothing first is the temporally correct order and is required once the data is cut into trials."
