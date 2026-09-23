# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the authors' **joblib** files (the un-suffixed `QLAK-CA1-*` files in `/app/data`, i.e. the format the paper's own `utils.load_dat` uses by default) rather than the `.mat` files. The seven animal IDs are hard-coded in an `ANIMALS` list. For each animal, `joblib.load(...)[animal]` returns a dict from which the AI takes `trace` (n_days, n_cells, T), `position` (n_days, 2, T), `envs` (n_days, 1) and `blocked` (list of n_days entries). It then iterates over every recording day of every animal; nothing is subsampled. The resulting dataset covers all 7 mice and all 207 sessions (31/31/31/21/31/31/31), which the AI explicitly cross-checked against the paper's reported "5,413 unique neurons across 207 sessions ... forming 69,744 rate maps". `del dat, traces, positions` is called after each animal to release the (multi-GB) per-animal arrays.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
    for ai, animal in enumerate(ANIMALS):
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        traces, positions = dat['trace'], dat['position']
        envs = [str(e[0]) for e in dat['envs']]
        blocked = dat['blocked']
        n_days = traces.shape[0]

        for day in range(n_days):
            ...
        del dat, traces, positions
```

iii. From the module docstring: "data loaded from the authors' joblib files (identical content to the .mat files)" and "one session per recording day (40 min, 30 Hz), all 7 mice / 207 sessions kept". In the trajectory the AI read `/app/code/georepca1/README.md` and `utils.load_dat`, which documents that the joblib and MATLAB files hold the same fields and that `format="joblib"` is the default loader. It then ran an audit script over all seven files (step 20–21) confirming 207 sessions, 69,744 registered cell-sessions and ~72,000 frames/session, matching the paper exactly, before committing to this loader.

## 1-b. How are the data split into subjects (mice)?

i. One `joblib` file = one mouse. The animal ID (which is both the filename and the single top-level key inside the file) is used verbatim as the subject name. `subjects` is the fixed `ANIMALS` list of 7 IDs; `subject_idx` records the animal index for each emitted session.

ii.
```python
    for ai, animal in enumerate(ANIMALS):
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        ...
            subject_idx.append(ai)
...
        'subjects': ANIMALS,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. The AI verified in the trajectory that each file is a dict with a single key equal to the animal ID (`<class 'dict'> ['QLAK-CA1-08']`), and that each file holds all of that animal's recording days. No mice were excluded — the docstring states "all 7 mice / 207 sessions kept".

## 1-c. How are the data split into sessions?

i. One session per **recording day**. The leading axis of `trace`/`position`/`envs`/`blocked` indexes days, and the AI loops `for day in range(n_days)`, emitting one entry in `neural`/`input`/`output` per day. Each day is a 40-minute continuous recording in one environment geometry. All 207 day-sessions survive (the "fewer than 2 usable trials" guard never fires). Per-session provenance is stored in `metadata['session_info']`.

ii.
```python
        n_days = traces.shape[0]

        for day in range(n_days):
            env = envs[day]
            inp = blocked_vector(env, blocked[day])
            pos = np.asarray(positions[day], dtype=np.float64)     # (2, T)
            tr = np.asarray(traces[day])                           # (n_cells, T)
            ...
            neural_all.append(sess_neural)
            ...
            session_info.append({'subject': animal, 'day': int(day), 'environment': env,
                                 'blocked_partitions': np.nonzero(inp)[0].tolist(), ...})
```

iii. Methods: "All sessions were 40 min, and one session was recorded per day to avoid photobleaching." Each day has its own geometry, its own set of registered cells and its own cell-registration mapping, so a day is the natural session unit; the AI's docstring says "one session per recording day (40 min, 30 Hz)". Keeping days separate is also required by the target format, since `brain_region_idx`/neuron identity is per-session.

## 1-d. How are the data split into trials?

i. Per the task instructions, each session is cut into consecutive, non-overlapping **1-minute** trials. Because the AI rebins to 100 ms, one minute = `BINS_PER_TRIAL = 600` bins. The loop `range(0, nbins, 600)` walks the session start-to-end; the final chunk is kept even if shorter than 600 bins (so sessions yield 40 or 41 chunks from ~23,955–24,073 bins). Within each 1-minute window only the bins that pass the running-speed filter are retained, so the stored trials are variable-length (T mean 315, min 30, max 561 bins) sub-samples of a fixed 60 s window.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN))  # 600 bins = 1 min
...
            for start in range(0, nbins, BINS_PER_TRIAL):
                sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
                keep = run_b[sl]
                if keep.sum() < MIN_BINS_PER_TRIAL:
                    continue
                sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
                sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
                sess_input.append(inp.copy())
```

iii. Docstring: "long sessions are split into consecutive 1-minute trials (600 bins)". The instructions state "The experiment consists of long recording sessions, which will be split into 1-minute trials within each session", so the trial boundary is imposed by the downstream analysis, not by the experiment (the recording is continuous free foraging with no trial structure). The AI verified the result: 8,163 trials across 207 sessions (~39.4/session).

## 1-e. How are trials filtered based on quality controls?

i. Two quality gates, both driven by the running-speed criterion taken from the paper's decoding code:
- A trial is dropped if fewer than `MIN_BINS_PER_TRIAL = 30` of its 600 bins pass the >5 cm/s running filter (i.e. < 3 s of running data in that minute).
- A whole session is dropped if fewer than 2 usable trials remain (this guard never actually fires — all 207 sessions are kept).

24 of 8,187 possible 1-minute windows are removed this way (8,163 trials retained).

ii.
```python
MIN_BINS_PER_TRIAL = 30   # >= 3 s of running data required to keep a trial
...
                keep = run_b[sl]
                if keep.sum() < MIN_BINS_PER_TRIAL:
                    continue
...
            if len(sess_neural) < 2:
                print(f'  day {day} ({env}): < 2 usable trials, skipped')
                continue
```
```python
            'trial_curation': (
                f'trials with fewer than {MIN_BINS_PER_TRIAL} running time bins and '
                'sessions with fewer than 2 usable trials were dropped'),
```

iii. The minimum-trial-length rule exists because the AI adopted the paper's speed filter (`utils.decode_position_within`, `v_thresh=5`), which can leave a minute nearly empty if the mouse rested; such a trial would contribute almost no usable samples and could make a session degenerate. The 2-trial session rule comes straight from the format spec: "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively the `trace` field, `trace[day]` of shape (n_cells, T). Per the repository README this is the "rise-extracted calcium trace, where '1' indicates a significant event", with all-NaN rows for cells not registered on that day. The `position` field is used only indirectly (to build the running mask and the cell-activity criterion that select which trace samples/cells survive). `SFPs`, `centroids` and `maps` are not used.

ii.
```python
            tr = np.asarray(traces[day])                           # (n_cells, T)
            # keep only cells registered (tracked) on this day
            registered = ~np.isnan(tr[:, 0])
            tr = tr[registered].astype(np.float32)
```

iii. Methods: "All analyses were conducted using the binary vector of the rising phases of transients, treating this vector as if it were the firing rate of the cell". The AI confirmed empirically that the trace values are exactly `{0., 1.}` for registered cells (trajectory step 16), so `trace` is already the paper's firing-rate proxy and no re-derivation from raw fluorescence is required or possible.

## 2-b. How is the `neural` data processed?

i. Three steps, reproducing `utils.fit_decoder`:
1. Cast to `float32` after dropping unregistered / low-activity cells (2-c).
2. Gaussian smoothing along time with `sigma = 3` frames (`gaussian_filter1d(tr, sigma=3, axis=1)`).
3. Non-overlapping average pooling over 3 frames (`pool_mean`), giving a mean event rate per 100 ms bin. The remainder (<3 frames at the end of the session) is dropped.

Finally the running mask is applied and the array is sliced per trial. The result is a float32 (n_cells, n_bins) matrix in [0, 1]. Note the AI smooths the **full continuous** trace and only then applies the running mask, whereas the paper concatenates running frames first and smooths across the resulting gaps.

ii.
```python
TEMPORAL_BIN = 3      # frames per time bin -> 100 ms (as in utils.fit_decoder)
...
def pool_mean(x, k):
    """Non-overlapping average pooling over the last axis (drops the remainder)."""
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(x.shape[:-1] + (n // k, k)).mean(axis=-1)
...
            # ---- temporal smoothing + 100 ms binning (as in fit_decoder)
            tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
            neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, nbins)
```

iii. Docstring: "traces smoothed with a gaussian (sigma = 3 frames) and average-pooled over 3 frames -> 100 ms time bins (utils.fit_decoder, temporal_bin_size=3)". The AI read `fit_decoder` in the trajectory (step 8) and transcribed its two operations — `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` followed by `AvgPool1d(kernel_size=3, stride=3)` — which is precisely the neural preprocessing the paper uses for its own within-session position decoder. `metadata['neural_data_type']` records this: "binarized calcium-transient rising phases (events), gaussian smoothed (sigma = 3 frames) and averaged within 100 ms bins (mean event rate)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two neuron-level filters applied per session, plus one sample-level filter:
1. **Registration**: cells not tracked on that day are all-NaN and are removed (`~np.isnan(tr[:, 0])`).
2. **Activity**: of the registered cells, those with **≤ 5 events during running** in that session are removed (`cell_threshold=5` in `utils.decode_position_within`). This drops 69,744 → 68,862 neuron-sessions (~1.3%); per-session counts are 112–562 neurons (mean 332.7).
3. **Running-speed sample filter**: speed is computed from the frame-to-frame position displacement × 30 Hz, smoothed with a gaussian of sigma 5 frames, and thresholded at 5 cm/s; only bins above threshold are kept. This removes ~45% of all time bins (2.57 M of ~4.79 M bins retained).

If a session ends up with zero surviving cells it is skipped (never triggered).

ii.
```python
V_FILT_SIGMA = 5      # gaussian sigma (frames) for speed estimate
V_THRESH = 5.0        # cm/s, running threshold
CELL_THRESH = 5       # minimum number of events during running per session
...
            # keep only cells registered (tracked) on this day
            registered = ~np.isnan(tr[:, 0])
            tr = tr[registered].astype(np.float32)

            # ---- running speed (cm/s), smoothed as in decode_position_within
            speed = np.zeros(pos.shape[1])
            speed[1:] = np.linalg.norm(np.diff(pos, axis=1), axis=0) * FPS
            speed = gaussian_filter1d(speed, sigma=V_FILT_SIGMA)
            running_frames = speed > V_THRESH

            # ---- exclude poorly active cells (>5 events while running)
            active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
            tr = tr[active]
            if tr.shape[0] == 0:
                print(f'  day {day} ({env}): no cells pass criteria, skipped')
                continue
```

iii. Docstring and `metadata['neuron_curation']`: "per session: only cells registered (tracked) on that day and with more than 5 events during running were kept"; `metadata['speed_filter']`: "time bins with running speed <= 5.0 cm/s excluded ... as in the papers within-session position decoding". All three thresholds are the default arguments of `utils.decode_position_within(..., v_filt_size=5, v_thresh=5, cell_threshold=5)`, the function the paper uses for exactly this decoding analysis. Before committing, the AI measured the cost of these filters on two animals (trajectory step 26): ~50–61% of frames pass the speed filter and only 0.5–1.8% of cells fail the activity criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural alignment event — the recording is 40 minutes of continuous free foraging. The AI aligns trials to the **start of the recording session**: trial *k* covers bins `[600k, 600(k+1))`, i.e. seconds `[60k, 60(k+1))` of the session. This is recorded in metadata as `temporal_alignment_event = 'start of the recording session; each session is cut into consecutive 1-minute trials'`, with `off_start = 0.0` and `off_end = 60.0` (signed seconds from the alignment event to trial start/end). Neural, input and output streams are cut with identical slices and identical running masks, so they remain sample-for-sample aligned.

ii.
```python
            for start in range(0, nbins, BINS_PER_TRIAL):
                sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
                keep = run_b[sl]
                ...
                sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
                sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
```
```python
            'temporal_alignment_event': (
                'start of the recording session; each session is cut into consecutive '
                '1-minute trials'),
            'off_start': 0.0,
            'off_end': 60.0,
```

iii. The task is continuous spatial navigation with no trial structure or discrete events, so the only meaningful anchor is the session onset; the AI states this directly in the metadata rather than inventing an event. The gaussian smoothing applied to the traces is symmetric (`scipy.ndimage.gaussian_filter1d`), so it introduces no lag between the neural stream and the position stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms** (`metadata['time_bin_size'] = 100.0`). Yes — the native 30 Hz (33.3 ms) data is rebinned by a factor of 3 via gaussian smoothing (sigma = 3 frames) + non-overlapping 3-frame average pooling, exactly as in `utils.fit_decoder(temporal_bin_size=3)`. Position and speed are pooled over the same 3-frame windows, so all streams share the bin grid. The bin size is identical for every trial and every session (only the *number* of bins per trial varies, because of the running filter).

ii.
```python
FPS = 30.0            # acquisition rate of miniscope / behaviour (Hz)
TEMPORAL_BIN = 3      # frames per time bin -> 100 ms (as in utils.fit_decoder)
...
            neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, nbins)
            pos_b = pool_mean(pos, TEMPORAL_BIN)                        # (2, nbins)
            speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]     # (nbins,)
            run_b = speed_b > V_THRESH
...
            'time_bin_size': 100.0,
            'sampling_rate_original_hz': FPS,
```

iii. Docstring: "traces smoothed ... and average-pooled over 3 frames -> 100 ms time bins (utils.fit_decoder, temporal_bin_size=3)". The paper's own position decoder operates on 100 ms bins, so this reproduces the reference analysis' temporal resolution; it also raises the per-bin signal-to-noise of the sparse binary event trains (mean event rate ≈ 0.21–0.24 Hz, measured by the AI in step 26) and cuts the dataset size threefold.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field — a per-day list of the indices of occluded partitions in the 3×3 grid, with the sentinel `[-1]` for the unblocked square. The `envs` field (the geometry's string name) is used as a cross-check only: the AI reconstructs the same 3×3 geometry from `utils.get_env_mat(env)` and asserts the two agree. `envs` is also stored in `metadata['session_info']` for provenance.

ii.
```python
            envs = [str(e[0]) for e in dat['envs']]
            blocked = dat['blocked']
            ...
            env = envs[day]
            inp = blocked_vector(env, blocked[day])
```
```python
    b = np.atleast_1d(np.asarray(blocked_field, dtype=float).ravel())
    vec = np.zeros(9, dtype=np.float32)
    if not (b.size == 1 and b[0] == -1):
        vec[b.astype(int)] = 1.0
    from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
    assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
```

iii. The repository README defines the field: "**blocked**: location of blocked (occluded) partitions in 3x3 design of environment ... organized in the following way – [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If no partitions are blocked, value is -1." The AI used `get_env_mat` (the paper's own geometry table) as an independent second source so that any indexing error would be caught — and it was: the first run raised `AssertionError: blocked mismatch for env t`, revealing that `get_env_mat` draws rows top-down while `blocked` indexes `3*y_bin + x_bin`, i.e. the two are vertical flips of each other (trajectory steps 31–32).

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `blocked` is turned into a 9-dimensional binary (float32) vector, 1 = partition blocked, 0 = open; `[-1]` yields the all-zero vector. The vector is **static per session** and stored as a 1-D `(9,)` array repeated (by `.copy()`) for every trial of that session — the format spec permits `(d_input,)` per trial. Input names are `blocked_{i}_x{i%3}y{i//3}`. Partition 7 (`x1y2`) is never blocked in any of the 10 geometries, so that channel is constant 0. Two assertions validate the encoding: the `get_env_mat` cross-check, and a per-session occupancy check requiring that the mouse spends < 2% of bins in partitions marked blocked.

ii.
```python
def blocked_vector(env_name, blocked_field):
    b = np.atleast_1d(np.asarray(blocked_field, dtype=float).ravel())
    vec = np.zeros(9, dtype=np.float32)
    if not (b.size == 1 and b[0] == -1):
        vec[b.astype(int)] = 1.0
    from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
    assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
    return vec
...
                sess_input.append(inp.copy())
...
            # sanity check: the animal should essentially never be in a blocked partition
            occ = np.bincount(out_bins[0], minlength=9) / nbins
            assert occ[inp.astype(bool)].sum() < 0.02, (
                f'{animal} day {day} ({env}): occupancy in blocked partitions '
                f'{occ[inp.astype(bool)].sum():.3f}')
...
        'input_names': [f'blocked_{i}_{PART_NAMES[i]}' for i in range(9)],
```

iii. Docstring / metadata: geometry is "a static per-trial input", with `partition_layout` = "partition index = 3 * y_bin + x_bin, i.e. [[0, 1, 2], [3, 4, 5], [6, 7, 8]] as in the datasets blocked field". A binary indicator over the 9 partitions is the complete description of a geometry in this paradigm (the 10 geometries are exactly 10 distinct blocking patterns), it is the encoding the instructions ask for ("For discrete inputs, use a one-hot encoding"), and using the same index convention as the output labels lets the decoder relate the geometry channel *i* directly to position class *i*. The geometry is fixed for a whole day, so it is correctly constant within and across trials of a session.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, `position[day]` of shape (2, T): the mouse's x-y head position in cm, from DeepLabCut tracking, sampled at 30 Hz in register with the imaging frames. The AI verified there are no NaNs anywhere in `position` for any animal and that the values span 0–75 cm.

ii.
```python
            pos = np.asarray(positions[day], dtype=np.float64)     # (2, T)
```

iii. README: "**position**: x-y position data for all days ... x-y position in first dimension, and number of temporal bins / frames in second dimension." Methods: "Position data were generated from tracking the head with DeepLabCut". The trajectory audit (step 20–21) reported `posnan 0` for all seven animals and x/y ranges of 0.0–75.0 cm, confirming the units are centimetres in the arena frame and that no gap-filling is required.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Position is average-pooled over the same 3-frame (100 ms) windows as the neural data, then each pooled sample is mapped to one of 9 partition labels via `3 * y_bin + x_bin`. Samples falling outside the running mask are dropped together with the corresponding neural samples. The output is stored as a `(1, n_bins)` int64 array (`output_names = ['position_bin']`, `output_values` = the 9 partition labels `0_x0y0 … 8_x2y2`). Observed class fractions are 0.070–0.171, i.e. reasonably balanced with the usual corner/wall over-representation.

ii.
```python
            pos_b = pool_mean(pos, TEMPORAL_BIN)                        # (2, nbins)
            ...
            out_bins = position_to_bin(pos_b)[np.newaxis, :]            # (1, nbins)
...
        'output_names': ['position_bin'],
        'output_values': [[f'{i}_{PART_NAMES[i]}' for i in range(9)]],
```

iii. `fit_decoder` in the paper pools the behavioural stream with the same `AvgPool1d(kernel_size=3, stride=3)` it applies to the traces and only then casts to an integer spatial bin, so pooling-then-discretizing (rather than discretizing-then-pooling) follows the reference implementation and keeps behaviour and neural activity on one bin grid. The index convention `3*y_bin + x_bin` was chosen to match the dataset's own `blocked` layout and was empirically verified: the AI computed 3×3 occupancy maps for 10 sessions and confirmed that the bins the mouse never visits are exactly the indices listed in `blocked` (trajectory steps 19–20, 32).

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each axis of the 75 cm arena is cut into 3 equal 25 cm bins by `floor(coordinate / 25)`, clipped to [0, 2], giving the 3×3 = 9 partitions required by the decoder task. The two bin indices are combined into a single 9-way categorical label as `3 * y_bin + x_bin`. The clip handles the exact-maximum coordinate (x or y = 75.0 would otherwise floor to bin 3).

ii.
```python
ARENA_SIZE = 75.0     # cm (75 x 75 cm square)
N_SPACE_BINS = 3      # 3 x 3 partitions
...
def position_to_bin(pos_xy):
    edge = ARENA_SIZE / N_SPACE_BINS
    xb = np.clip(np.floor(pos_xy[0] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
    yb = np.clip(np.floor(pos_xy[1] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
    return 3 * yb + xb
...
            'spatial_bin_size_cm': ARENA_SIZE / N_SPACE_BINS,
```

iii. The instructions specify "Mouse position discretized into 3 x 3 = 9 spatial bins", and the 3×3 partition grid is the paper's own experimental unit — "We partitioned an open square (75 × 75 cm) into a 3 × 3 grid space" — so the bin edges are the physical partition walls, not an arbitrary grid. Using the same index convention as `blocked` is what makes the occupancy assertion (`< 2%` of bins in blocked partitions) a meaningful validation of the discretization; it passes for all 207 sessions.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and calcium are acquired by the same DAQ at 30 Hz with a shared frame index, so they are aligned sample-for-sample in the raw data. The AI preserves that alignment end-to-end: both streams are pooled with the same `pool_mean(..., 3)` (so they share the same bin boundaries and the same dropped remainder), both are sliced with the same `sl` trial slice, and both are masked with the same boolean `keep` running mask. The stored `output[s][t]` therefore has exactly the same number of columns as `neural[s][t]`, bin for bin.

ii.
```python
            neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, nbins)
            pos_b = pool_mean(pos, TEMPORAL_BIN)                        # (2, nbins)
            speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]     # (nbins,)
            run_b = speed_b > V_THRESH
            out_bins = position_to_bin(pos_b)[np.newaxis, :]            # (1, nbins)
...
                keep = run_b[sl]
                sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
                sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
```

iii. Methods: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz as uncompressed AVI files and all recorded frames were timestamped for post-hoc alignment", and both arrays in the dataset have identical length T per day — so no resampling or time-shifting is needed, only identical downstream operations. Applying one shared `keep` mask (rather than filtering the streams independently) is what guarantees the correspondence survives the speed filter; `train_decoder.py --verify-only` reported no dimension warnings.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i.
- **Unregistered cells** (the only NaNs in the dataset) are dropped per session via `~np.isnan(tr[:, 0])`. The AI first verified that NaN-ness is all-or-nothing per cell per day ("days with partial-nan cells: 0" for all 7 animals), so testing the first frame is sufficient.
- **Missing position**: none — verified `posnan 0` for every animal — so no interpolation is implemented.
- **Remainder frames**: the 0–2 frames at the end of a session that do not fill a 100 ms bin are silently dropped by `pool_mean`.
- **Degenerate sessions/trials**: sessions with no surviving cells, trials with < 30 running bins, and sessions with < 2 trials are skipped with an explanatory print.
- **Silent-corruption guards**: two assertions fail loudly rather than producing wrong labels — the `blocked` vs `flipud(get_env_mat)` geometry check, and the per-session check that < 2% of occupancy falls in blocked partitions.

ii.
```python
            registered = ~np.isnan(tr[:, 0])
            tr = tr[registered].astype(np.float32)
...
            if tr.shape[0] == 0:
                print(f'  day {day} ({env}): no cells pass criteria, skipped')
                continue
...
    assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
...
            assert occ[inp.astype(bool)].sum() < 0.02, (
                f'{animal} day {day} ({env}): occupancy in blocked partitions '
                f'{occ[inp.astype(bool)].sum():.3f}')
...
def pool_mean(x, k):
    """Non-overlapping average pooling over the last axis (drops the remainder)."""
    n = (x.shape[-1] // k) * k
```

iii. README: "**trace**: ... If cell is not registered on given day, will appear as nan the same shape." The AI audited the NaN structure and the position completeness before writing the converter (trajectory steps 16, 20–21), so each handler addresses a property it had actually measured rather than a hypothetical. The geometry assertion caught a real indexing bug on the first run, which is the AI's stated reason for keeping both checks in the shipped code.

## 6-a. What are the most time-consuming steps of the code?

i. In descending order:
1. **`joblib.load` of each animal file** — by far the dominant cost. The compressed files are 71–151 MB but expand to the full float64 `trace` array (e.g. 31 × 515 × 71,866 ≈ 9.2 GB for QLAK-CA1-08), plus `SFPs`, `centroids` and `maps`, all of which must be decompressed even though only `trace`/`position`/`envs`/`blocked` are used.
2. **`gaussian_filter1d(tr, sigma=3, axis=1)`** — a 1-D convolution over ~72,000 frames for every one of ~330 cells, once per session, 207 times.
3. **`pickle.dump` of the 3.3 GB output** plus the `np.ascontiguousarray` copies that materialise ~8,163 trial arrays (2.57 M bins × ~333 neurons of float32).
4. The boolean fancy-indexing `neural[:, sl][:, keep]`, which makes two copies per trial.

The whole conversion ran in roughly 6 minutes end-to-end in the trajectory.

ii.
```python
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
            tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
...
                sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
...
    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
```

iii. The AI did not profile the code, but it did check the machine's resources up front (`df -h`, `free -g`, `nproc`) before committing to a data size, and it added `del dat, traces, positions` after each animal specifically because the per-animal arrays are the memory/IO bottleneck. Its choice of 100 ms bins was partly motivated by keeping the output pickle to a manageable size.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Essentially none of the hot paths — all per-sample work (speed, smoothing, pooling, discretization, event counting) is already fully vectorized over cells and time. The remaining Python loops are:
- `for ai, animal in ANIMALS` and `for day in range(n_days)` — not vectorizable (each day has a different cell count and geometry), though they are *parallelizable*: the 7 animals are independent and the machine had 128 cores, so a `multiprocessing` fan-out over animals would have cut wall-clock several-fold.
- `for start in range(0, nbins, BINS_PER_TRIAL)` — ~40 iterations per session doing only slicing; could be expressed as a reshape for the full-length chunks, but the variable-length `keep` masks make the gain negligible.
- `envs = [str(e[0]) for e in dat['envs']]` — 31 elements, negligible.

ii.
```python
    for ai, animal in enumerate(ANIMALS):
        ...
        for day in range(n_days):
            ...
            for start in range(0, nbins, BINS_PER_TRIAL):
```

iii. Not discussed by the AI. The design implicitly relies on NumPy/SciPy for all per-element work, which is why the Python-level loops iterate only over animals (7), days (31) and trials (~40) rather than over samples or cells.

## 6-c. What processing does the code repeat multiple times?

i. Only small, cheap repetitions:
- `get_env_mat` rebuilds its 10-entry geometry dictionary on every call (207 calls), and `blocked_vector` recomputes `1 - flipud(get_env_mat(env))` for each day even though only 10 distinct geometries exist — this could be cached once.
- `pool_mean` is invoked three times per session (traces, position, speed) — necessary, not redundant.
- `speed` is thresholded twice: once at frame resolution (`running_frames`, used for the cell-activity criterion) and once at bin resolution (`run_b`, used to select bins).
- `inp.copy()` allocates a fresh 9-element array for each of the ~8,163 trials.
- `np.ascontiguousarray(neural[:, sl][:, keep])` copies each trial twice (once for the slice, once for the mask).

None of these is material next to the file I/O.

ii.
```python
    from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
...
            running_frames = speed > V_THRESH
            ...
            run_b = speed_b > V_THRESH
...
                sess_input.append(inp.copy())
```

iii. Not discussed by the AI. The duplicated speed thresholding is deliberate: the frame-resolution mask is needed to reproduce the paper's per-cell event count during running, while the bin-resolution mask is needed to select 100 ms bins; the per-trial `inp.copy()` avoids aliasing a single array across trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Loading unused fields.** `joblib.load` deserializes the whole animal dict, including `SFPs` (e.g. 35 × 35 × 515 × 31 ≈ 157 MB for one animal), `centroids`, and `maps['sampling'/'smoothed'/'unsmoothed']` — none of which the converter touches. This is the largest piece of wasted work, though it is hard to avoid with `joblib.load`.
- **Smoothing and pooling data that is then thrown away.** `gaussian_filter1d` and `pool_mean` are applied to the entire session, but ~45% of the resulting bins are immediately removed by the running filter, and the bins belonging to the 24 discarded short trials are wasted too.
- **Validation-only computation.** `get_env_mat` + the `flipud` comparison, and the per-session `np.bincount` occupancy assertion, produce nothing that is saved; they exist purely as correctness checks.
- **Metadata not consumed by the decoder.** `metadata['session_info']` (207 dicts) and the per-day progress prints are written for provenance but are ignored by `train_decoder.py`.
- **`np.float64` upcast of position** (`np.asarray(positions[day], dtype=np.float64)`) when the data is already float64 and only 3 bins of precision are ultimately needed.

ii.
```python
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]   # also loads SFPs, centroids, maps
...
            tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)   # ~45% of output later masked out
            neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
...
    from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
    assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
...
            occ = np.bincount(out_bins[0], minlength=9) / nbins
            assert occ[inp.astype(bool)].sum() < 0.02, (...)
...
            session_info.append({'subject': animal, 'day': int(day), 'environment': env, ...})
```

iii. The AI never states these as costs, but the ordering is deliberate on one point: smoothing the *contiguous* trace before masking (rather than masking first, as the paper does) avoids smoothing across the temporal gaps the speed filter creates, at the price of computing values that are then discarded. The assertions and `session_info` are explicitly defensive/provenance features — the geometry assertion caught a real bug during development — so the AI traded a small amount of redundant computation for verifiability.
