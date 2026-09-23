# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven deposited per-animal **joblib** files in `/app/data` (the extension-less `QLAK-CA1-*` files), not the duplicate `.mat` files. `animal_files()` discovers them by filtering `/app/data` for names starting with `QLAK-CA1-`, having no suffix and no underscore (this excludes `QLAK-CA1-08.mat`, `behav_dict` and `precomputed_results`). Each file is loaded whole with `joblib.load(f)[f.name]`, yielding the nested animal dictionary documented in the reference README (`SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`). Animals are processed one at a time and the dictionary is deleted + `gc.collect()`ed before the next, to bound memory. Within an animal, every recording day is iterated (`for day in range(d['trace'].shape[0])`), and each day is split into contiguous 1-minute trials. Result: 7 subjects, 207 sessions, 8,187 trials, 69,744 session-neuron instances.

ii.
```python
DATA_DIR = Path('/app/data')

def animal_files():
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.name.startswith('QLAK-CA1-') and not p.suffix and '_' not in p.name)
...
    for si, f in enumerate(files):
        t_load=time.time(); print(f'Loading {f.name}...', flush=True)
        d=joblib.load(f)[f.name]
        print(f'  loaded in {time.time()-t_load:.2f}s', flush=True)
        for day in range(d['trace'].shape[0]):
            ...
            nmat, out, geom, present, ncorrect = process_day(
                d['trace'][day], d['position'][day], d['blocked'][day])
        del d; gc.collect()
```

iii. From CONVERSION_NOTES Step 1/2/10: the reference `utils.load_dat` has two branches — a MATLAB branch (`mat73.loadmat`) and a **joblib branch that is the documented default** (`format="joblib"`), and `mat2joblib` shows the joblib files are simply the converted `.mat` files. The AI chose the joblib branch as "the efficient reference-code input" (70–151 MB compressed vs 300–800 MB `.mat`), noting `/app/data` "contains seven animal datasets in both compressed joblib (extensionless, used by reference `load_dat`) and duplicate MATLAB v7.3 `.mat` formats". Its Step 10 reference-code comparison records: "Data loading | `joblib.load(file)[animal]` | `load_dat` joblib branch | Same source files and nested animal dictionary."

---

## 1-b. How are the data split into subjects?

i. One `.joblib` file == one mouse. The subject ID is the file name (`QLAK-CA1-08`, `-30`, `-50`, `-51`, `-56`, `-74`, `-75`), which is also the top-level key inside the file. `subjects` is the full sorted list of all seven IDs, and `subject_idx` stores the index of the owning animal for each emitted session. The full 7-subject vocabulary is deliberately retained even in `--sample` mode (where only subject 0 is referenced).

ii.
```python
files=animal_files()
subjects=[p.name for p in files]
...
for si, f in enumerate(files):
    d=joblib.load(f)[f.name]
    ...
        subject_idx.append(si)
...
used_subjects=sorted(set(subject_idx))
# Keep full subject vocabulary as required IDs of all subjects; sample may reference only index 0.
data=dict(..., subjects=subjects, subject_idx=np.asarray(subject_idx,dtype=np.int64), ...)
```

iii. CONVERSION_NOTES Step 2: "Each joblib root is `{animal_id: dataset}`" and "Animal IDs: QLAK-CA1-08, …, QLAK-CA1-75" — i.e. the file name is the animal identity used by the reference `load_dat(animal, p)` API. Step 9 verifies 7 subjects with 31/31/31/21/31/31/31 sessions, matching the deposited data. Cross-day cell registration is per-animal, which is why the AI treats each file as one subject (Step 2: "`n_registered_cells` is the cross-day union per animal (515–952)").

---

## 1-c. How are the data split into sessions?

i. Each **recording day** inside an animal file becomes one target session (`d['trace'].shape[0]` days). This gives 207 sessions. Each session carries a constant, simultaneously-recorded neuron set, its own environment geometry, and its own `session_info` provenance record (`session_id`, `subject`, `source_day`, `environment`, `blocked`, `native_frames`, `present_neurons`, `n_trials`, `retained_time_bins`, `discarded_native_equivalent_frames`, `corrected_blocked_pooled_bins`).

ii.
```python
for day in range(d['trace'].shape[0]):
    ...
    nmat, out, geom, present, ncorrect = process_day(
        d['trace'][day], d['position'][day], d['blocked'][day])
    ...
    env=str(np.asarray(d['envs']).ravel()[day])
    info=dict(session_id=f'{f.name}_day{day:02d}', subject=f.name, source_day=day,
              environment=env, blocked=blocked_indices(d['blocked'][day]).tolist(),
              native_frames=int(d['trace'].shape[2]), present_neurons=int(present.sum()),
              n_trials=int(ntrials), retained_time_bins=int(keep),
              discarded_native_equivalent_frames=int(d['trace'].shape[2]-keep*POOL),
              corrected_blocked_pooled_bins=int(ncorrect))
    session_info.append(info)
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "**Target session definition**: Each original recording day is one target session. This preserves a constant simultaneously recorded neuron set and the paper's within-session decoder framing." This is backed by the methods ("All sessions were 40 min, and one session was recorded per day") and by the reference `decode_position_within`, which fits/tests decoders *within* each day `d`. Step 9 confirms 207 sessions against the paper's "207 sessions" figure in `methods.txt`.

---

## 1-d. How are the data split into trials?

i. The native recordings are continuous ~40-minute sessions with no native trial structure. Per the task instruction, each session is cut into **contiguous, non-overlapping 60-second segments**. Because the AI rebins to 100 ms (see 2-e), one trial = `TRIAL_BINS = 600` time bins. Trials are cut *after* whole-session smoothing/pooling, by `np.split` on the pooled matrix. The trailing sub-minute remainder is discarded (never padded, never emitted as a short trial). Result: 39 trials for sessions of 71,866 native frames, 40 for the longer ones; 8,187 trials total.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)   # 600
...
    ntrials=nmat.shape[1]//TRIAL_BINS
    if ntrials < 2:
        print(f'WARNING skipping {f.name} day {day}: only {ntrials} full trials')
        continue
    keep=ntrials*TRIAL_BINS
    ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
    os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
    ins=[geom.copy() for _ in range(ntrials)]
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "**One-minute trials**: Use contiguous, non-overlapping exact minutes. At 100 ms/bin each trial has 600 timepoints. Drop only the final sub-minute tail of each day; never pad or create a short 'one-minute' trial." Key Decision 3 adds that smoothing/pooling is done on the **whole day before trial slicing** "to avoid artificial filter boundaries". Step 2 pre-computed the expected yield ("8,187 complete non-overlapping minutes … 166,419 trailing frames would remain") and Step 9 confirms the converted data reproduces exactly 8,187.

---

## 1-e. How are trials filtered based on quality controls?

i. Essentially no trial-level quality filtering is applied, by design. The only trial/session rejections are structural: (a) the trailing partial minute of every session is dropped (and the discarded frame count is logged per session), and (b) a session producing fewer than 2 complete trials would be skipped with a warning (this never triggers — every session yields 39 or 40). The AI explicitly decided **not** to apply the reference decoder's running-speed gate (`v_thresh=5` cm/s with sigma-5-frame speed smoothing) or its per-cell activity gate, and not to apply place-cell selection.

ii.
```python
    ntrials=nmat.shape[1]//TRIAL_BINS
    if ntrials < 2:
        print(f'WARNING skipping {f.name} day {day}: only {ntrials} full trials')
        continue
    keep=ntrials*TRIAL_BINS
...
    info=dict(..., discarded_native_equivalent_frames=int(d['trace'].shape[2]-keep*POOL), ...)
...
    # structural checks before saving
    for sidx in range(len(neural)):
        assert len(neural[sidx])==len(inputs[sidx])==len(outputs[sidx])>=2
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules"): "Native recordings are continuous 40-minute sessions, not event-aligned trials. No invalid-time mask or omitted days is described. Deposited position is finite and day-present neural traces are finite throughout." Step 10 reference comparison: "All day-present curated cells retained … Only incomplete tails removed to satisfy exact 1-min trials." The speed/activity gates were judged to be *decoder-internal* selection in `decode_position_within` rather than dataset curation, and applying them would destroy the contiguous fixed-length trial structure the target format requires. The `>= 2` guard directly implements the format spec's "There needs to be at least two trials within each session in order to evaluate the decoder performance."

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely `trace` — the deposited rise-extracted binary calcium-event matrix, indexed as `d['trace'][day]` with shape `(n_registered_cells, n_frames)`. No dF/F, no deconvolution, no rate-map product, no `SFPs`/`centroids`/`maps` are used for the neural stream.

ii.
```python
        nmat, out, geom, present, ncorrect = process_day(
            d['trace'][day], d['position'][day], d['blocked'][day])
...
def process_day(trace_day, position_day, blocked_raw):
    present = np.isfinite(trace_day[:, 0])
    ...
    raw = np.asarray(trace_day[present].T, dtype=np.float32)  # time x neurons
```

iii. CONVERSION_NOTES Step 1: "`trace` is already a rise-extracted binary calcium-event series (1 = significant event), so delta-F/F must **not** be recomputed for this conversion." Step 3 quotes the methods: "The final binarized rising-phase vector was then set to 1 whenever this z-scored … vector exceeded 2.5, and 0 otherwise" and "All analyses were conducted using the binary vector … treating this vector as if it were the firing rate." Step 2 verified empirically that "all finite sampled traces are exactly 0 or 1."

---

## 2-b. How is the `neural` data processed?

i. Three steps, applied per session over the whole day before trialization:
1. Select day-present neurons (drop all-NaN unregistered cells) and transpose to `(time, neurons)`, cast float32.
2. `gaussian_filter1d(..., sigma=3 native frames, axis=time)` — the reference decoder's temporal smoothing.
3. Non-overlapping mean pooling over groups of 3 native frames (reshape-mean, equivalent to `AvgPool1d(kernel_size=3, stride=3)`), then transpose back to `(neurons, n_bins)`.

The stored values are therefore mean smoothed event probability per 100 ms bin (float32), not events/s. No place-cell filtering, no z-scoring, no normalisation.

ii.
```python
POOL = 3
...
def process_day(trace_day, position_day, blocked_raw):
    present = np.isfinite(trace_day[:, 0])
    if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
        raise ValueError('Cell registration mask changes within a session')
    raw = np.asarray(trace_day[present].T, dtype=np.float32)  # time x neurons
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite value in a day-present neural trace')
    n_pool = raw.shape[0] // POOL
    n_used = n_pool * POOL
    # Match reference fit_decoder/test_decoder: temporal Gaussian sigma=3 then AvgPool stride=3.
    smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
    pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
    ...
    return pooled_neural.T, classes[None, :], geom, present, n_corrected
```
Reference code it mirrors (`utils.py:1786-1788`):
```python
pooling = AvgPool1d(kernel_size=temporal_bin_size, stride=temporal_bin_size)  # temporal_bin_size=3
behav, traces = pooling(torch.tensor(behav.T)).numpy().astype(int).T, \
    pooling(torch.tensor(gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0).T)).numpy().T
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "**Temporal processing**: Match reference position decoder: Gaussian-filter neural traces with sigma 3 native frames, then average neural and position over stride-3 windows. Process the whole day before trial slicing to avoid artificial filter boundaries." Key Decision 8: "Store smoothed mean binary-event activity as float32. Values represent event probability per 100 ms frame grouping, not events/s; this exactly follows the reference decoder's filtering/pooling rather than rate-map scaling." Step 10's comparison table records "Logic matched" against `fit_decoder`/`test_decoder`, and Step 10 Check 2/3 verified a reconstruction of three trials from the original joblib with `np.allclose(rtol=1e-6, atol=1e-7)`.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filter is removal of cells **not registered on that day**, which the deposited data encodes as all-NaN rows. The AI detects them with `np.isfinite(trace_day[:, 0])` and hardens this with two assertions: the registration mask must be identical at the first and last frame of the session (proving NaN encodes absence, not an invalid time window), and every retained value must be finite. All cells present on the day are retained — no place-cell selection, no minimum-event threshold. Yields 113–564 neurons/session, mean 336.93, 69,744 session-neuron instances.

ii.
```python
    present = np.isfinite(trace_day[:, 0])
    if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
        raise ValueError('Cell registration mask changes within a session')
    raw = np.asarray(trace_day[present].T, dtype=np.float32)  # time x neurons
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite value in a day-present neural trace')
...
    region_idx.append(np.zeros(nmat.shape[0],dtype=np.int8))
```

iii. CONVERSION_NOTES Step 3 "Neuron curation rules": "Motion correction and spatial footprints were manually inspected; lens artifacts were manually removed before deposited preprocessing. Cells are registered across sessions. A cell absent on a day is all-NaN and must not be included in that session. Paper explicitly says reliability results 'motivated the inclusion of all cells in subsequent analyses.' Therefore place-cell filtering is analysis-specific, not a general dataset filter." Step 5, Key Decision 4 adds: "Do not restrict to place cells because the paper explicitly includes all cells and the reference position decoder uses an activity threshold rather than place-cell labels." Step 2 empirically verified "Registration masks at first, middle, and last frames are identical for every animal/day, showing NaNs encode absent cells rather than invalid temporal periods."

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental alignment event — the paradigm is continuous free foraging. The AI aligns each trial to the **start of its own contiguous 60-second segment** within the recording day, and records this explicitly in metadata with `off_start = 0.0` and `off_end = 60.0`. Neural, input and output for a trial are all cut from the identical bin window, so alignment is exact by construction (`np.split` of arrays already truncated to the same `n_used` length).

ii.
```python
    keep=ntrials*TRIAL_BINS
    ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
    os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
...
    metadata=dict(
        ...
        temporal_alignment_event='Start of each contiguous non-overlapping 1-minute segment within a recording day',
        off_start=0.0, off_end=60.0, source_frame_rate_hz=FPS,
        trialization='Complete 60 s segments only; trailing sub-minute data discarded.',
        ...)
```

iii. CONVERSION_NOTES Step 4: "Position alignment | `position` and `trace` loaded together | Exact frame-dimension equality all 207 sessions | Streams simultaneously acquired/timestamped | Preserve same frame indices for neural and output; no interpolation needed." Step 5, Key Decision 10: "**Alignment**: Apply identical 3-frame windows and minute slices to neural and position arrays. No interpolation is required because source dimensions match exactly." Backed by the methods quote "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz … all recorded frames were timestamped for post-hoc alignment."

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — rebinned from 30 Hz (33.33 ms) to 100 ms bins.** Three native frames are Gaussian-smoothed (sigma = 3 frames) and then mean-pooled into one output bin, giving `time_bin_size = 100.0` ms and 600 bins per 60-second trial. This is applied identically to the neural and the position streams, and uniformly to every trial and session.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)   # 600
...
    n_pool = raw.shape[0] // POOL
    n_used = n_pool * POOL
    smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
    pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
    pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
    pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
...
    metadata=dict(..., time_bin_size=BIN_MS,
        neural_processing='Deposited binary rising-phase events; Gaussian smoothing sigma=3 native frames followed by non-overlapping 3-frame mean pooling.', ...)
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "Decoder temporal bin | Reference decoder aggregates 3 frames (0.1 s) | Native 30 Hz | Acquisition 30 Hz; paper does not explicitly state aggregation | **Use reference decoder's 3-frame binning for neural/output to match decoding processing and reduce dataset size.**" Step 1 recorded the reference default `temporal_bin_size=3` in `fit_decoder`/`test_decoder`. Secondary benefit noted in Step 6: float32 at 100 ms yields a 6.17 GiB pickle rather than ~18 GiB at native rate.

---

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The `blocked` field, read per day as `d['blocked'][day]`, whose entries are the indices of occluded partitions on the row-major 3x3 grid, or `-1` when nothing is blocked. `d['envs'][day]` (the named geometry, e.g. `square`, `o`, `bit donut`) is *not* used to build the vector but is stored in `session_info` and was used as an independent cross-check against the reference `get_env_mat`.

ii.
```python
def blocked_indices(raw):
    vals = np.asarray(raw[0]).ravel().astype(int)
    return vals[vals >= 0]

def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom
...
    geom = geometry_vector(blocked_raw)   # from d['blocked'][day]
...
    env=str(np.asarray(d['envs']).ravel()[day])
```

iii. CONVERSION_NOTES Step 1: "`blocked` encodes blocked partition indices in row-major 3x3 order `[[0,1,2],[3,4,5],[6,7,8]]`; -1 means none blocked. `get_env_mat` provides equivalent named-geometry occupancy matrices." Step 4: "Geometry | `get_env_mat` maps names to accessible 3x3 matrices | `blocked` indices exactly match names | … | Construct static 9-vector … directly from `blocked`; cross-check against `get_env_mat`." Using `blocked` directly avoids depending on the reference's string-to-matrix lookup table while remaining verifiably identical to it.

---

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A static 9-element **accessibility** vector per session: initialise to ones, set the blocked indices to zero, so `1 = accessible, 0 = blocked`, row-major (`index = y*3 + x`). The `-1` sentinel is filtered out by `vals[vals >= 0]`, so an unobstructed square yields all-ones. The vector is float32 (changed from int8 after the validator warned about dtype) and is copied once per trial so every trial carries the same static `(9,)` input. `input_names` are `accessible_bottom-left` … `accessible_top-right`.

ii.
```python
def blocked_indices(raw):
    vals = np.asarray(raw[0]).ravel().astype(int)
    return vals[vals >= 0]

def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom
...
    ins=[geom.copy() for _ in range(ntrials)]
...
    input_names=[f'accessible_{name}' for name in OUTPUT_LABELS],
...
        assert n.shape==(nn,TRIAL_BINS) and i.shape==(9,) and o.shape==(1,TRIAL_BINS)
        assert np.isfinite(n).all() and np.all((o>=0)&(o<9)) and np.all(i[o]==1)
```

iii. CONVERSION_NOTES Step 5, Key Decision 7: "**Geometry encoding**: Use 1=accessible and 0=blocked, matching `get_env_mat`. Input names explicitly state accessibility to prevent semantic ambiguity." The geometry is constant within a recording day (one geometry per day per the methods), so it is a static per-trial input as the Decoder Task specifies. Step 10 Check 4: "Original nested `blocked` values were independently converted to a nine-element float32 accessibility mask. Exact equality passed." Step 9/10 also record the float32 dtype fix after the validator warned that int8 inputs would be converted during training.

---

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field, `d['position'][day]`, shape `(2, n_frames)` giving DeepLabCut-tracked head x and y in cm within the 75 x 75 cm arena. The geometry (`blocked`) is used secondarily, only to snap physically impossible samples (see 4-c). No other variable contributes.

ii.
```python
        nmat, out, geom, present, ncorrect = process_day(
            d['trace'][day], d['position'][day], d['blocked'][day])
...
def process_day(trace_day, position_day, blocked_raw):
    ...
    pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 1/2: "`position` is x-y by frame for each recording day/session" and "`position`: float64 `(n_days,2,n_frames)` x-y coordinates, range approximately 0--75." Step 3 quotes the methods: "Position data were generated from tracking the head with DeepLabCut pose-estimation software." Step 2 verified "All positions are finite" and "Neural and position frame counts match exactly for every session."

---

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The x and y traces are truncated to the same `n_used` frames as the neural data and **mean-pooled over the same non-overlapping 3-frame windows** (in float64), producing one (x, y) pair per 100 ms bin. No smoothing, interpolation, or velocity filtering is applied to position. The pooled coordinates are then discretized (4-c).

ii.
```python
    n_pool = raw.shape[0] // POOL
    n_used = n_pool * POOL
    ...
    pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
    pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
```
Reference equivalent (`utils.py:1787`, `behav` already divided by `bin_down`):
```python
behav = pooling(torch.tensor(behav.T)).numpy().astype(int).T
```

iii. CONVERSION_NOTES Step 5 mapping table: "`position[day, x/y, :]` -> `output[session][trial]` | Average matching 3-frame groups; clip x,y to [0,75]; divide by 25 and floor/clip to bins 0--2; class=`y_bin*3+x_bin`; slice 600-bin minutes | `fit_decoder`, `test_decoder`; task-required 3x3 override". Key Decision 3 makes the pooling of neural and position a single joint decision so that the two streams stay frame-locked. Step 10: "Same temporal aggregation and valid-bin principle; coarser 3x3 classes required by task."

---

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each pooled coordinate is divided by 25 cm (= 75/3) and truncated to an integer bin, clipped to `[0, 2]`; the class is `y_bin * 3 + x_bin`, giving 9 row-major classes 0–8 labelled `bottom-left` … `top-right` by increasing source y. Additionally, any bin whose class falls in a **blocked** partition for that session's geometry is snapped to the nearest accessible grid cell by squared Euclidean distance in grid coordinates (`nearest_accessible`). Over the whole dataset this affected **152 of ~4.97 million pooled bins (0.003%)**, 145 of them in a single `bit donut` session.

ii.
```python
    xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
    ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
    classes = (ybin * 3 + xbin).astype(np.int8)
    geom = geometry_vector(blocked_raw)
    classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)

def nearest_accessible(classes, xbin, ybin, geom):
    """Snap classes in blocked partitions to nearest accessible grid center."""
    bad = geom[classes] == 0
    if not np.any(bad):
        return classes, 0
    valid = np.flatnonzero(geom).astype(np.int8)
    vx, vy = valid % 3, valid // 3
    bx, by = xbin[bad, None], ybin[bad, None]
    nearest = valid[np.argmin((bx-vx[None, :])**2 + (by-vy[None, :])**2, axis=1)]
    out = classes.copy()
    out[bad] = nearest
    return out, int(np.count_nonzero(bad))
```

iii. CONVERSION_NOTES Step 5, Key Decision 5: "**Spatial discretization/orientation**: Use 25 cm bins and row-major `y*3+x`. Direct testing against all original blocked masks found 445/14,903,019 frames (0.0030%) in blocked classes versus 17.12% under the wrong x-major convention, decisively confirming orientation." Key Decision 6: "**Blocked-bin artifacts**: Snap the very rare blocked output to the nearest accessible 3x3 cell after temporal averaging, matching the spirit of reference `decode_position_within`, which snaps actual/predicted positions to valid map bins. This avoids contradictory geometry/output labels caused by boundary/tracking noise." The reference precedent is `utils.py:1912-1915`, which replaces both actual and predicted bins with the nearest occupied map bin. Bin size 25 cm comes from the methods ("The full square environment was 75 cm x 75 cm … inserted partition walls" on a 3x3 grid).

---

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, by construction. `position` and `trace` have identical frame counts in every one of the 207 sessions (verified directly from source). Both are truncated to the same `n_used = (n_frames // 3) * 3`, pooled with the same stride-3 windows, and then cut with the same `[:keep]` truncation and the same `np.split` into 600-bin trials. No interpolation, no resampling, no lag or shift is introduced. Final shapes per trial: neural `(n_neurons, 600)`, output `(1, 600)`.

ii.
```python
    n_pool = raw.shape[0] // POOL
    n_used = n_pool * POOL
    smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
    pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
    pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
    pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
    ...
    return pooled_neural.T, classes[None, :], geom, present, n_corrected
...
    keep=ntrials*TRIAL_BINS
    ns=[... for x in np.split(nmat[:,:keep],ntrials,axis=1)]
    os=[... for x in np.split(out[:,:keep],ntrials,axis=1)]
...
        assert n.shape==(nn,TRIAL_BINS) and o.shape==(1,TRIAL_BINS)
```

iii. CONVERSION_NOTES Step 2: "Neural and position frame counts match exactly for every session." Step 5, Key Decision 10: "Apply identical 3-frame windows and minute slices to neural and position arrays. No interpolation is required because source dimensions match exactly." Step 10 Check 5: "Original x-y position was independently 3-frame averaged, mapped with `floor(coord/25)`, clipped, flattened as `y*3+x`, and valid-bin corrected. Exact equality passed for beginning, middle, and last sample trials." Step 12 Check 5 adds visual alignment evidence via `processing_*.png`, `sample_trials.png` and `predictions.png`.

---

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Five categories, each handled explicitly and logged:
1. **Unregistered (all-NaN) cells** — removed per day via the finite mask, with assertions that the mask is time-invariant and that no NaN survives.
2. **Trailing sub-minute frames** — dropped; the exact discarded native-frame count is stored per session (`discarded_native_equivalent_frames`, 0–1,799 frames).
3. **Position samples in physically blocked partitions** — snapped to the nearest accessible bin, count logged per session and in aggregate (152 total).
4. **Out-of-range coordinates / boundary values** — `np.clip(..., 0, 2)` guarantees valid bins (e.g. x = 75.0 maps to bin 2).
5. **`blocked == -1` sentinel** — filtered by `vals[vals >= 0]` so "nothing blocked" yields an all-accessible vector.
Anything unexpected is a hard failure rather than a silent pass: the registration-mask change and non-finite-trace checks `raise ValueError`, and a full structural assert sweep runs over every session and trial before pickling.

ii.
```python
    present = np.isfinite(trace_day[:, 0])
    if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
        raise ValueError('Cell registration mask changes within a session')
    ...
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite value in a day-present neural trace')
    ...
    xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
    ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
    classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
...
def blocked_indices(raw):
    vals = np.asarray(raw[0]).ravel().astype(int)
    return vals[vals >= 0]
...
    # Internal structural checks.
    assert len(neural)==len(inputs)==len(outputs)==len(subject_idx)==len(region_idx)==len(session_info)
    for sidx in range(len(neural)):
        assert len(neural[sidx])==len(inputs[sidx])==len(outputs[sidx])>=2
        nn=neural[sidx][0].shape[0]
        for n,i,o in zip(neural[sidx],inputs[sidx],outputs[sidx]):
            assert n.shape==(nn,TRIAL_BINS) and i.shape==(9,) and o.shape==(1,TRIAL_BINS)
            assert np.isfinite(n).all() and np.all((o>=0)&(o<9)) and np.all(i[o]==1)
```

iii. CONVERSION_NOTES Step 10 Check 7 "Edge cases": "Checked coordinate value 75 clipping to bin 2; `blocked=-1`; 39-vs-40-trial sessions; sub-minute tails (all 0--1,799 native frames); first/last trial slicing; animal boundary and 21-session animal; all-NaN absent cells; and rare blocked tracking points. No off-by-one failure found." Step 10 "Issues Found and Resolved" documents the two real issues found: the int8-input dtype validator warning (fixed by emitting float32 geometry and regenerating both pickles) and the 152 blocked-bin tracking artifacts (snapped, with a global check confirming zero remaining contradictions).

---

## 6-a. What are the most time-consuming steps of the code?

i. The code prints per-stage timing. Full conversion took **170.32 s** total, decomposed as: **decompressing the joblib source files, ~9–23 s per animal (~100 s total, by far the largest share)**; per-session processing (finite mask, Gaussian filter, pooling, discretization, trial splitting) 0.17–0.61 s x 207 sessions (~60 s, scaling with neuron count); and the final `pickle.dump` of the 6.17 GiB output, 5.13 s. With `--show-processing`, plotting adds ~0.4–0.8 s per plotted session.

ii.
```python
    for si, f in enumerate(files):
        t_load=time.time(); print(f'Loading {f.name}...', flush=True)
        d=joblib.load(f)[f.name]
        print(f'  loaded in {time.time()-t_load:.2f}s', flush=True)
        for day in range(d['trace'].shape[0]):
            t0=time.time()
            ...
            print(f'  session {session_count:03d} {info["session_id"]}: {present.sum()} neurons, '
                  f'{ntrials} trials, corrected={ncorrect}, {time.time()-t0:.2f}s', flush=True)
...
    t_save=time.time()
    with open(outfile,'wb') as fh: pickle.dump(data,fh,protocol=5)
    print(f'Saved in {time.time()-t_save:.2f}s; total {time.time()-t_all:.2f}s; ...')
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Compressed source loading is slower than per-session processing. Full output is intrinsically large because it retains all day-present cells across 8,187 one-minute trials." Step 7's run-time table estimates "Source loading ~9--23 s/animal, ~2 min for seven animals" vs "Session conversion ~0.2 s … <1 min for 207 sessions", concluding "Total conservatively <10 min, below 15-min threshold"; Step 9 reports the measured 170.32 s, consistent with the estimate.

---

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's position is that the numerically heavy work is already vectorized, and this is borne out by the code: Gaussian filtering is a single `gaussian_filter1d` over the whole session, pooling is a `reshape(...).mean(axis=1)` rather than a per-bin loop, discretization is array arithmetic, and the blocked-bin correction is a single broadcast `argmin` over all offending bins at once. The loops that remain are: the per-animal and per-day loops (inherently sequential and I/O bound; could be parallelized across animals with multiprocessing at the cost of ~1 animal's RAM per worker); the per-trial list comprehensions building `ns`/`os`/`ins` (these are copies required by the target list-of-arrays format, not avoidable computation); and the final O(8,187-trial) assertion sweep.

ii.
```python
    smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
    pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
    pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
    xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
    ...
    nearest = valid[np.argmin((bx-vx[None, :])**2 + (by-vy[None, :])**2, axis=1)]
...
    ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
    ins=[geom.copy() for _ in range(ntrials)]
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "Vectorized Gaussian filtering, reshape-based pooling, discretization, and blocked-bin correction"; "Session-wide filtering occurs once before slicing, avoiding repeated trial computations and filter-boundary artifacts"; "Sequential per-animal loading with garbage collection bounds source-memory use." Step 7 reports the result: "~0.2 s/session without plotting", i.e. processing is not the bottleneck, so further loop vectorization would not materially change the 170 s runtime.

---

## 6-c. What processing does the code repeat multiple times?

i. Very little, and nothing on a hot path. The genuine repeats are: (a) `blocked_indices()` is computed twice per session — once inside `geometry_vector()` and once again to populate `session_info['blocked']`; (b) `d['trace'].shape` is re-read several times per session for `session_info`; (c) in `--show-processing` mode `d['trace'][day, present]` is re-extracted from the source array purely for plotting; (d) the closing validation sweep re-walks all 207 sessions x ~40 trials, re-checking shapes, finiteness and value ranges that were already guaranteed at construction. Each of these is microseconds-to-milliseconds against a 170 s run. Critically, the expensive operations are *not* repeated: smoothing/pooling/discretization run exactly once per session, before trialization, rather than once per trial.

ii.
```python
def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0          # first call
    return geom
...
    info=dict(session_id=..., blocked=blocked_indices(d['blocked'][day]).tolist(),   # second call
              native_frames=int(d['trace'].shape[2]), ...,
              discarded_native_equivalent_frames=int(d['trace'].shape[2]-keep*POOL), ...)
...
            if show_processing and plot_count < 2:
                plot_processing(..., d['trace'][day,present], d['position'][day], nmat, out, geom)
...
    for sidx in range(len(neural)):
        for n,i,o in zip(neural[sidx],inputs[sidx],outputs[sidx]):
            assert np.isfinite(n).all() and np.all((o>=0)&(o<9)) and np.all(i[o]==1)
```

iii. CONVERSION_NOTES Step 6 addresses this as a design principle rather than an enumerated defect: "Session-wide filtering occurs once before slicing, avoiding repeated trial computations and filter-boundary artifacts." The AI did not flag the duplicate `blocked_indices` call or the final assertion sweep; the latter is deliberate defensive validation (Step 10 relies on it), not accidental duplication.

---

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items, all cheap relative to the run but real:
1. **`used_subjects` is dead code** — `sorted(set(subject_idx))` is computed and never referenced; `subjects` (the full 7-name vocabulary) is what gets stored.
2. **Whole-file loading pulls unused fields.** `joblib.load(f)[f.name]` materialises the entire animal dictionary, including `SFPs` (35x35xN_cellsxN_days), `centroids` and `maps` (`smoothed`/`unsmoothed`, 15x15xN_cellsxN_days) — the largest arrays in the file — even though only `trace`, `position`, `blocked` and `envs` are ever used. This is the dominant cost of the dominant step (6-a). The `.mat`/h5py path used by the human reference reads datasets lazily and would avoid it.
3. **Provenance metadata never consumed downstream.** The per-session `session_info` records (`retained_time_bins`, `discarded_native_equivalent_frames`, `corrected_blocked_pooled_bins`, `native_frames`, `environment`, …) are written into the pickle but ignored by `train_decoder.py`.
4. **Final assertion sweep and per-session console statistics** are validation-only and discarded.
Conversely, the storage-relevant decisions actively *reduce* waste: float32 neural / int8 output, and the 3x temporal reduction that makes the pickle 6.17 GiB rather than roughly three times that at the native rate.

ii.
```python
    used_subjects=sorted(set(subject_idx))     # computed, never used
    # Keep full subject vocabulary as required IDs of all subjects; sample may reference only index 0.
    data=dict(..., subjects=subjects, ...)
...
        d=joblib.load(f)[f.name]               # loads SFPs / centroids / maps as well
...
            info=dict(session_id=..., retained_time_bins=int(keep),
                      discarded_native_equivalent_frames=int(d['trace'].shape[2]-keep*POOL),
                      corrected_blocked_pooled_bins=int(ncorrect))
            session_info.append(info)
```

iii. The AI documented the storage/efficiency side of this in CONVERSION_NOTES Step 6 ("Compressed source loading is slower than per-session processing"; "float32/int8 output" listed under speed-ups with "Approximately halves neural storage versus float64") and Step 7 ("One source load for multiple days | Avoids repeated ~9--23 s decompression per session"). It did **not** identify the dead `used_subjects` variable or the fact that `SFPs`/`centroids`/`maps` are decompressed and discarded. The `session_info` provenance is a deliberate choice, justified in Step 5's mapping table as "Enables provenance and tail accounting", and is used by the AI's own Step 10 edge-case audit even though the decoder ignores it.
