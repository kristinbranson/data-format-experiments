# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven extension-less **joblib** files in `/app/data` (`QLAK-CA1-08`, `-30`, `-50`, `-51`, `-56`, `-74`, `-75`) rather than the parallel `.mat` files. These are the repository's own converted copies (`utils.mat2joblib` / `utils.load_dat(..., format="joblib")`, which is the repo default). Each file is a dict keyed by the animal ID containing `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`. The subject list is hard-coded, and for each subject the whole dict is loaded into memory at once, then iterated day-by-day. Every animal, every recording day, and every complete 60-s window is kept; nothing is subsampled. After the loop the AI asserts the paper's totals (207 sessions, 69,744 session-neurons) and aborts if they are not reproduced.

ii.
```python
DATA_DIR = Path('/app/data')
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
for subj_i, subject in enumerate(SUBJECTS):
    source = joblib.load(DATA_DIR / subject)[subject]
    traces = source['trace']       # day x union-neuron x frame
    positions = source['position'] # day x (x,y) x frame
    envs = source['envs']
    blocked = source['blocked']
    for day in range(traces.shape[0]):
        ...
    del source, traces, positions
    gc.collect()
...
if len(neural) != 207 or total_valid != 69744:
    raise RuntimeError(f'unexpected source totals: {len(neural)} sessions, '
                       f'{total_valid} session-neurons')
```

iii. From the trajectory: "The extensionless files are compressed serialization rather than plain pickle, likely joblib files used by the repository" (step 5) and "create `convert_data.py` using the smaller joblib-converted source files" (step 8). The script docstring states: "The repository's joblib files are used rather than reprocessing calcium video. They contain the paper's aligned 30-Hz binary rising-transient traces and DLC position, produced with the processing described in the paper." The AI first verified that all seven animals together give "exactly the paper's 207 sessions and 5,413 tracked neurons" (step 6) and later that the valid session-neuron count "exactly reproduces the paper's 69,744 session-specific rate maps" (step 11), and hard-coded both numbers as a guard.

## 1-b. How are the data split into subjects (mice)?

i. One subject per data file / per animal ID. The seven IDs are hard-coded in `SUBJECTS` (verified beforehand by listing `/app/data`), written verbatim into `data['subjects']`, and each session appends the enclosing subject's index to `subject_idx`.

ii.
```python
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
...
        subject_idx.append(subj_i)
...
    'subjects': SUBJECTS,
    'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. Step 3: "The dataset contains seven animals (QLAK-CA1-08, 30, 50, 51, 56, 74, 75), each with a large MAT file and an extensionless processed file." The README states each file is named for an animal ID, and cells are tracked across days *within* an animal, so the file boundary is the subject boundary. The AI cross-checked the per-animal day counts against the paper's 207 sessions.

## 1-c. How are the data split into sessions?

i. One session per recording day, i.e. per index along the first axis of `trace` / `position` / `envs` / `blocked` (31 days for six animals, 21 for QLAK-CA1-51 → 207 sessions). Each day becomes one entry of `neural`/`input`/`output`/`brain_region_idx`, plus a `session_info` record holding subject, day index, environment name, blocked indices and frame counts.

ii.
```python
    traces = source['trace']       # day x union-neuron x frame
    for day in range(traces.shape[0]):
        tr = np.asarray(traces[day])
        pos = np.asarray(positions[day])
        ...
        env = scalar_env(envs[day])
        session_info.append({
            'subject': subject,
            'source_session_index': day,
            'environment': env,
            'blocked_grid_indices': [int(b) for b in block_values if b >= 0],
            'source_frames': int(n_frames),
            'used_frames': int(used),
            'n_complete_one_minute_trials': int(n_trials),
            'n_neurons_present': int(n_neurons),
        })
```

iii. The script docstring: "A recording day is a session." The paper states "All sessions were 40 min, and one session was recorded per day", and each day is a different environment geometry, so day = session. Step 6: "each recording day should remain a session." The AI confirmed the total equals the paper's 207 sessions.

## 1-d. How are the data split into trials?

i. Each session's continuous ~40-min recording is cut into **non-overlapping complete 60-s windows** (1800 frames at 30 Hz, stored as 60 one-second bins). The trailing partial minute is dropped (sessions have 71,866–72,219 frames → 39 or 40 complete trials; 8,187 trials in total). Splitting is done by a single `reshape` of the truncated arrays.

ii.
```python
FS = 30
BIN_FRAMES = 30
TRIAL_SECONDS = 60
BINS_PER_TRIAL = TRIAL_SECONDS
TRIAL_FRAMES = FS * TRIAL_SECONDS      # 1800
...
            n_trials = n_frames // TRIAL_FRAMES
            if n_trials < 2:
                raise ValueError(f'too few complete trials: {subject} day {day}')
            used = n_trials * TRIAL_FRAMES
            counts = tr[:, :used].reshape(
                n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
            ).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
```

iii. The instructions define the trial: "long recording sessions, which will be split into 1-minute trials within each session." The docstring: "Complete, non-overlapping 60-s windows are trials. A short trailing fragment is omitted so all trials have identical duration." Step 8: "Sessions differ slightly in frame count, so only complete 1,800-frame (60 s at 30 Hz) trials should be retained; this yields 39 or 40 trials depending on actual recorded duration."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. The only exclusion is structural — an incomplete trailing window is dropped — and a defensive check aborts a session with fewer than two complete trials (never triggered; every session has 39–40). No velocity/immobility filter is applied, even though the paper's own Bayesian decoder (`utils.decode_position_within`, `v_thresh=5` cm/s) discards low-speed samples.

ii.
```python
            n_trials = n_frames // TRIAL_FRAMES
            if n_trials < 2:
                raise ValueError(f'too few complete trials: {subject} day {day}')
            used = n_trials * TRIAL_FRAMES
```

iii. Trials are an artificial construct imposed by the downstream decoder, not an experimental unit, so there is no behavioural criterion by which one minute of free foraging is "bad". The format requirement "There needs to be at least two trials within each session" is enforced explicitly. A speed filter would break the contiguity and fixed length of the 60-s windows, so it is not used.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely the `trace` field: the paper's rise-extracted binary calcium event traces, stored per animal as a `day × union-neuron × frame` array (values 0/1, NaN where a tracked cell was not registered that day). `SFPs`, `centroids` and `maps` are loaded with the file but not used.

ii.
```python
    traces = source['trace']       # day x union-neuron x frame
    ...
        tr = np.asarray(traces[day])
```

iii. README: "**trace**: rise-extracted calcium traces, where '1' indicates a significant event." Methods: "This binary vector was treated as the firing rate in all further analyses." Step 4: "Neural activity is already preprocessed into binary rising-phase calcium-event vectors"; step 6 confirmed empirically that "traces are binary".

## 2-b. How is the `neural` data processed?

i. No re-derivation from fluorescence — the paper's binary event vector is taken as given. Processing is: (a) orientation check so the array is `(neuron, frame)`; (b) removal of absent (all-NaN) neurons (see 2-c); (c) **summation of the binary events within each non-overlapping 30-frame (1 s) window**, producing an integer-valued event count per neuron per second stored as `float32`; (d) transposition to per-trial `(n_neurons, 60)` matrices. Neural values therefore range 0–30 rather than {0,1}.

ii.
```python
            if tr.shape[1] != pos.shape[1] and tr.shape[0] == pos.shape[1]:
                tr = tr.T
            if tr.shape[1] != pos.shape[1]:
                raise ValueError(f'unaligned streams: {subject} day {day}')
            ...
            counts = tr[:, :used].reshape(
                n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
            ).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
            ...
            neural.append([counts[t] for t in range(n_trials)])
```
with metadata `'neural_measurement': 'count per 1-s bin of paper-preprocessed binary rising-phase calcium transient events'`.

iii. Docstring: "Thirty synchronized frames are aggregated per 1-s decoder bin. Binary neural events are summed (event counts), while x/y is averaged. This preserves event mass and avoids an impractically large 30-Hz decoder dataset." Step 6: "raw 30 Hz would produce a prohibitively large decoder dataset, so we need … a justified common bin size (likely 1 second with event counts and mean position)." Summation (rather than averaging) is chosen because the paper treats the binary vector as a firing rate, so a per-second sum is an event rate in Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Per session, only neurons whose trace is entirely finite are retained; rows that are entirely NaN (cells from the animal's cross-day union that were not registered that day) are dropped. The code additionally asserts that no neuron is *partially* NaN. No other neuron-level curation is applied — no minimum event count, no place-cell selection — so session neuron counts run 113–564 and total 69,744 session-neurons, matching the paper's 69,744 rate maps. All neurons are assigned to the single brain region `CA1`.

ii.
```python
            finite = np.isfinite(tr)
            valid = finite.all(axis=1)
            if np.any(finite.any(axis=1) != valid):
                raise ValueError(f'partially missing neuron: {subject} day {day}')
            tr = tr[valid]
            n_neurons, n_frames = tr.shape
            ...
            brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
            total_valid += n_neurons
...
if len(neural) != 207 or total_valid != 69744:
    raise RuntimeError(...)
```

iii. Docstring: "Neurons tracked across animals' days occupy a union array; an all-NaN row means that cell was not observed that day. Such rows are removed session-wise. This gives the reported 69,744 rate maps." Step 8: "Those columns must be removed separately per session; retaining them would introduce invalid values and would incorrectly treat unrecorded neurons as silent." The AI ran an audit over all seven animals before writing the converter (steps 9–11) and confirmed both the all-or-nothing NaN pattern and the 69,744 total. The paper applies no further neuron filter to its main analyses (its `cell_threshold=5` applies only inside its Bayesian decoder), so no activity threshold was imposed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the recording is continuous free foraging. The AI defines the alignment event as the start of each 60-s window and records it explicitly, with `off_start = 0.0 s` and `off_end = 60.0 s`. Neural, input and output streams are cut with the same frame indices, so all trials cover exactly the same 60 s of wall-clock recording.

ii.
```python
    'metadata': {
        ...
        'temporal_alignment_event': 'start of each non-overlapping one-minute window',
        'off_start': 0.0,
        'off_end': 60.0,
        'trial_definition': ('complete non-overlapping 60-s windows; trailing '
                             'partial minute omitted'),
```

iii. The instructions require `temporal_alignment_event`, `off_start` and `off_end`; with no stimulus or trial-onset event in this paradigm, the only well-defined reference point is the window boundary, and the AI states it rather than leaving the fields unset. Step 7/8 note that all streams are already timestamp-aligned at 30 Hz by the acquisition DAQ ("The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz … all recorded frames were timestamped for post-hoc alignment"), so a common frame index is sufficient for alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — the data are rebinned from the native 30 Hz to 1 Hz.** Thirty consecutive frames are pooled into one bin: neural events are summed, x/y position is averaged. Each trial is therefore `(n_neurons, 60)` instead of `(n_neurons, 1800)`, and `metadata['time_bin_size'] = 1000.0` ms (vs. the native 33.33 ms). The bin size is identical for every trial and session. This makes the saved dataset ~667 MB rather than ~20 GB.

ii.
```python
FS = 30
BIN_FRAMES = 30
BINS_PER_TRIAL = TRIAL_SECONDS          # 60 bins per trial
...
            counts = tr[:, :used].reshape(
                n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
            ).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
            mean_pos = pos[:, :used].reshape(
                2, n_trials, BINS_PER_TRIAL, BIN_FRAMES
            ).mean(axis=3)
...
        'time_bin_size': 1000.0,
        'source_sampling_rate_hz': 30.0,
```

iii. Docstring: "Thirty synchronized frames are aggregated per 1-s decoder bin … This preserves event mass and avoids an impractically large 30-Hz decoder dataset." Step 5: "retaining 30 Hz across all sessions would create a very large pickle and decoder workload, while the paper's maps use spatial rather than temporal binning." Step 11: "use synchronized 1-second bins to keep decoder size tractable." The choice is motivated purely by dataset size / decoder tractability; the AI notes the paper itself temporally bins for decoding, though the paper's own decoder uses a 3-frame (100 ms) bin, 10× finer than the AI's.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field — a per-day list of indices of the occluded cells of the 3×3 partition grid, with the sentinel `[-1]` for the unpartitioned square. `envs` (the geometry's string name, e.g. `square`, `o`, `t`, `l`, `glenn`) is read only for the `session_info` metadata, not for building the input.

ii.
```python
    blocked = source['blocked']
    envs = source['envs']
    ...
            block_values = np.asarray(blocked[day], dtype=float).reshape(-1)
            ...
            env = scalar_env(envs[day])
```

iii. README: "**blocked**: location of blocked (occluded) partitions in 3x3 design of environment … organized in the following way – [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If no partitions are blocked, value is -1." Step 6: "Environment geometry is explicitly supplied as blocked indices in a flattened 3x3 grid; square uses [-1] meaning no blocked bins" — i.e. the geometry is available directly and need not be reconstructed from the environment name via `utils.get_env_mat`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The blocked indices are converted to a fixed 9-element binary indicator vector in the same row-major `[[0,1,2],[3,4,5],[6,7,8]]` order as the source, with 1 = blocked. The `-1` sentinel (and any out-of-range value) is skipped, giving an all-zero vector for the open square. The vector is static per session and is replicated (a copy per trial) as each trial's `input`, shape `(9,)`; `input_names` are `blocked_spatial_bin_0..8`. Bin 7 is never blocked in this dataset, so that input is constant 0.

ii.
```python
            geometry = np.zeros(9, dtype=np.float32)
            block_values = np.asarray(blocked[day], dtype=float).reshape(-1)
            for b in block_values:
                if np.isfinite(b) and 0 <= int(b) < 9:
                    geometry[int(b)] = 1.0
            ...
            inputs.append([geometry.copy() for _ in range(n_trials)])
...
    'input_names': [f'blocked_spatial_bin_{i}' for i in range(9)],
    'geometry_encoding': ('nine binary blocked-cell indicators in the same '
                          'row-major index convention as source blocked lists'),
```

iii. Docstring: "Geometry is the supplied `blocked` list, encoded as nine binary indicators. The square's sentinel -1 means no blocked cells." The instructions ask for "Environment geometry, representing which parts of the arena are blocked. Static per-trial", so a nine-way binary code is the direct, lossless representation of all ten geometries; keeping the source's own index convention guarantees that input index *k* refers to the same physical cell as output class *k*.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field — DeepLabCut head-tracking x/y in centimetres in the 75 × 75 cm arena, stored `day × (x, y) × frame` at 30 Hz, one sample per imaging frame.

ii.
```python
    positions = source['position'] # day x (x,y) x frame
    ...
        pos = np.asarray(positions[day])
        if pos.shape[0] != 2 and pos.shape[1] == 2:
            pos = pos.T
```

iii. README: "**position**: x-y position data for all days … x-y position in first dimension, and number of temporal bins / frames in second dimension." Methods: "Position data were generated from tracking the head with DeepLabCut." Step 8: "Position is already aligned, finite, and scaled to 0–75 cm" — the AI checked the range and NaN prevalence before using it, so no rescaling or interpolation is applied.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Orientation is normalised to `(2, n_frames)` and the stream is checked to have the same frame count as the trace (hard error otherwise). Positions are then averaged within each 30-frame (1 s) bin and the *mean* position is discretised (see 4-c). No smoothing, no speed filtering, no unit conversion, and no interpolation of missing samples (there are none). The result is one integer class per second, stored per trial as shape `(1, 60)`.

ii.
```python
            mean_pos = pos[:, :used].reshape(
                2, n_trials, BINS_PER_TRIAL, BIN_FRAMES
            ).mean(axis=3)
            xbin = np.clip(np.floor(mean_pos[0] / 25.0), 0, 2).astype(np.int64)
            ybin = np.clip(np.floor(mean_pos[1] / 25.0), 0, 2).astype(np.int64)
            spatial_class = ybin * 3 + xbin
            ...
            outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
...
        'position_processing': ('mean aligned DLC x/y per 1-s bin, discretized '
                                'at 25 and 50 cm into class y_bin*3+x_bin'),
```

iii. Docstring and metadata: position is averaged over exactly the same 30 frames used to sum the neural events, so the behavioural and neural streams remain sample-for-sample synchronous. This mirrors the paper's own decoding pipeline, which average-pools position over frames (`AvgPool1d`) and then floor-divides into spatial bins — the AI uses a 30-frame pool where the paper uses 3.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 75 cm arena is cut at 25 cm and 50 cm on each axis, giving a 3 × 3 grid of 25 cm cells. `x_bin = clip(floor(x/25), 0, 2)`, `y_bin = clip(floor(y/25), 0, 2)` (the clip catches the exact boundary value x = 75), and the class is `y_bin * 3 + x_bin`, i.e. row-major with y as row — the same convention as the `blocked` indices. There is one categorical output, `spatial_bin`, with 9 values named `row_r_column_c`. Class frequencies in the converted file range from 0.057 to 0.200.

ii.
```python
            xbin = np.clip(np.floor(mean_pos[0] / 25.0), 0, 2).astype(np.int64)
            ybin = np.clip(np.floor(mean_pos[1] / 25.0), 0, 2).astype(np.int64)
            spatial_class = ybin * 3 + xbin
...
    'output_names': ['spatial_bin'],
    'output_values': [[f'row_{r}_column_{c}' for r in range(3) for c in range(3)]],
```

iii. Docstring: "The 75-cm square is divided at 25 and 50 cm, exactly matching the experiment's 3x3 construction. Class is row*3+column, with x selecting column and y row." The instructions require "Mouse position discretized into 3 x 3 = 9 spatial bins", and the paper built the arena as a 3 × 3 partition grid of 25 cm cells, so the decoder's bins coincide with the physical partition cells and with the geometry input's index convention. (Checking the data confirms the convention: occupancy is exactly zero in precisely the `blocked` indices under `y_bin*3+x_bin`.)

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and calcium frames are acquired and stored frame-synchronously, so alignment is by shared frame index: the same `used = n_trials * 1800` frames are taken from both streams, reshaped with the same `(n_trials, 60, 30)` layout, and the neural sum and position mean for bin *t* cover exactly the same 30 frames. A shape check aborts the conversion if the two streams ever disagree in frame count. The resulting per-trial arrays are `(n_neurons, 60)` and `(1, 60)`.

ii.
```python
            if tr.shape[1] != pos.shape[1] and tr.shape[0] == pos.shape[1]:
                tr = tr.T
            if tr.shape[1] != pos.shape[1]:
                raise ValueError(f'unaligned streams: {subject} day {day}')
            ...
            counts = tr[:, :used].reshape(n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES).sum(axis=3, ...)
            mean_pos = pos[:, :used].reshape(2, n_trials, BINS_PER_TRIAL, BIN_FRAMES).mean(axis=3)
```

iii. Methods: "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz … and all recorded frames were timestamped for post-hoc alignment", and step 8 confirmed the two arrays have identical frame counts per session, so no resampling or lag correction is needed; the docstring notes the 30 frames are "synchronized".

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled. (1) *Unregistered cells*: all-NaN neuron rows are dropped per session; partially-NaN rows would be a genuine anomaly and raise an error (verified never to occur across all 207 sessions). (2) *Ragged recording length*: sessions have 71,866–72,219 frames, so the trailing <60 s fragment is discarded (1,666–2,019 frames per session) to keep all trials exactly equal in length. (3) *Structural anomalies*: transposed arrays are silently corrected, but mismatched neural/position lengths, sessions with <2 complete trials, and any departure from the paper's 207/69,744 totals raise and abort the run rather than being papered over. Position has no missing samples (checked) and boundary values at exactly 75 cm are absorbed by `np.clip`; out-of-range `blocked` codes (the `-1` sentinel) are skipped.

ii.
```python
            finite = np.isfinite(tr)
            valid = finite.all(axis=1)
            if np.any(finite.any(axis=1) != valid):
                raise ValueError(f'partially missing neuron: {subject} day {day}')
            tr = tr[valid]
            ...
            n_trials = n_frames // TRIAL_FRAMES          # trailing fragment dropped
            if n_trials < 2:
                raise ValueError(f'too few complete trials: {subject} day {day}')
            ...
            xbin = np.clip(np.floor(mean_pos[0] / 25.0), 0, 2)
            for b in block_values:
                if np.isfinite(b) and 0 <= int(b) < 9:
                    geometry[int(b)] = 1.0
...
if len(neural) != 207 or total_valid != 69744:
    raise RuntimeError(...)
```

iii. Step 8: NaN rows "must be removed separately per session; retaining them would introduce invalid values and would incorrectly treat unrecorded neurons as silent", and "only complete 1,800-frame … trials should be retained". The AI ran an explicit audit (steps 9–11) to establish that NaNs are strictly all-or-nothing per cell before relying on it, and encoded every assumption it had verified as a hard assertion so that a violation would fail loudly rather than corrupt the dataset. `session_info` records `source_frames` vs `used_frames` so the discarded remainder is auditable.

## 6-a. What are the most time-consuming steps of the code?

i. The arithmetic (one `reshape`+`sum`, one `reshape`+`mean`, a few `clip`/`floor` calls per session) is negligible; the run is dominated by I/O and serialisation: (1) `joblib.load` of each animal file, which decompresses the *entire* dict — including `SFPs`, `centroids` and `maps` — and materialises the full `day × union-neuron × frame` float64 trace array (9.2–17.0 GB per animal, ~93 GB across the seven if all were held at once); (2) `pickle.dump` of the 667 MB output. Per-session work is `O(n_neurons × n_frames)` and single-pass. Memory is managed only coarsely, by `del` + `gc.collect()` between animals.

ii.
```python
        source = joblib.load(DATA_DIR / subject)[subject]   # decompresses whole animal dict
        ...
        del source, traces, positions
        gc.collect()
...
    with OUT.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 7 originally planned "a streaming HDF5 converter to avoid loading multi-GB MAT files", but step 8 switched to "the smaller joblib-converted source files" on the grounds that they are smaller on disk (99–151 MB compressed vs 296–794 MB for `.mat`). The docstring keeps only the scientific justification ("They contain the paper's aligned 30-Hz binary rising-transient traces and DLC position"); the peak-memory consequence of whole-file loading is not discussed. The AI did note (steps 12–13) that loading and serialising were the phases it had to wait on.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The numerically heavy work is already fully vectorised: binning and trial-splitting are one `reshape`/`sum` and one `reshape`/`mean` per session; discretisation is vectorised over the whole session at once. The remaining Python loops are shallow and unavoidable or trivial: the 7-subject × ~30-day outer loops (each iteration is an independent session), the ≤4-element loop over `block_values`, and three per-trial list comprehensions (`counts[t]`, `geometry.copy()`, `spatial_class[t][None, :]`) that only slice an already-computed array because the target format demands a Python list of per-trial arrays. None of these is worth vectorising further.

ii.
```python
            for b in block_values:
                if np.isfinite(b) and 0 <= int(b) < 9:
                    geometry[int(b)] = 1.0
            neural.append([counts[t] for t in range(n_trials)])
            inputs.append([geometry.copy() for _ in range(n_trials)])
            outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
```

iii. Not discussed explicitly in the trajectory; the vectorised formulation follows from the AI's decision to express trial-splitting and binning as a single 4-D reshape ("(neuron, trial, second, frame-within-second), then trial first", per the inline comment), which performs splitting, binning and reordering in one pass instead of a per-trial loop.

## 6-c. What processing does the code repeat multiple times?

i. Essentially nothing is recomputed. Each session's trace and position are read once and reduced in a single pass; `n_neurons`/`n_frames`/`n_trials` are computed once and reused. The only duplication is representational: `geometry.copy()` creates `n_trials` identical 9-float vectors per session (≈8,187 copies, ~300 KB total) where one shared array would do, and `np.asarray(traces[day])` / `np.asarray(positions[day])` wrap already-materialised slices. `envs[day]` is parsed per session but only for metadata.

ii.
```python
            inputs.append([geometry.copy() for _ in range(n_trials)])
            ...
        tr = np.asarray(traces[day])
        pos = np.asarray(positions[day])
```

iii. Not discussed in the trajectory. The per-trial copy follows from the format requirement that `input` be a list with one entry per trial; copying rather than aliasing avoids any risk that a downstream consumer mutating one trial's input would silently alter every other trial in the session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three items. (1) `joblib.load` deserialises the whole per-animal dict, so `SFPs` (35 × 35 × n_cells × n_days), `centroids` and the precomputed `maps` are decompressed and held in memory although the code only ever uses `trace`, `position`, `envs` and `blocked` — an h5py/`.mat` path would read only the needed datasets, session by session. (2) The trace is loaded for *all* union neurons including the all-NaN rows that are then discarded (roughly 65% of rows on a typical day for QLAK-CA1-08), and the full trace is truncated only after loading. (3) A verbose `session_info` record (207 dicts) and `envs` parsing are computed for metadata that the decoder never reads — cheap and useful for provenance, but not used downstream. The defensive orientation checks and the hard-coded 207/69,744 assertion likewise cost nothing but are pure validation.

ii.
```python
        source = joblib.load(DATA_DIR / subject)[subject]   # also loads SFPs, centroids, maps
        ...
            tr = tr[valid]                                  # after the whole union array is in memory
            ...
            session_info.append({...})                      # metadata only
...
if len(neural) != 207 or total_valid != 69744:
    raise RuntimeError(...)
```

iii. The AI's stated reason for the joblib route was that those files are "smaller" on disk (step 8) and are the repository's own converted copies; it did not weigh the cost of eagerly deserialising the unused fields. The `session_info` block is deliberate — the instructions invite extra metadata fields such as `session_info`, and the AI used it to record `source_frames`/`used_frames` so the discarded trailing minute and the per-session neuron counts remain auditable.
