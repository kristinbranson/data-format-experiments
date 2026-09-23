# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven per-animal **joblib** files in `/app/data` (the extensionless `QLAK-CA1-*` files), explicitly excluding the larger `.mat` duplicates, the derived `behav_dict`, and `precomputed_results/`. Each joblib object is a one-key dict `{subject_id: data_dict}`; the AI unwraps it with `joblib.load(f)[f.name]` and then iterates over the day axis of `raw['trace']`, pulling `position[day]`, `trace[day]`, `blocked[day]`, and `envs[day]` for each animal-day. Animals are processed one at a time and released (`del raw; gc.collect()`) before the next is loaded, to bound peak memory. This yields 7 subjects × 207 animal-days, 69,744 day-present cell recordings, and 8,187 one-minute trials — all of which reproduce the paper's reported totals.

ii.
```python
DATA_DIR = Path('/app/data')

def animal_files():
    """Return primary joblib animal files, excluding larger .mat duplicates."""
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.is_file() and p.name.startswith('QLAK-CA1-') and p.suffix != '.mat')
```
```python
    for si, f in enumerate(files):
        load_t=time.time(); raw=joblib.load(f)[f.name]
        print(f'Loaded {f.name} in {time.time()-load_t:.2f}s',flush=True)
        for day in range(raw['trace'].shape[0]):
            ...
            ntr,itr,otr,info,payload=process_session(
                raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
            ...
        del raw; gc.collect()
```

iii. From CONVERSION_NOTES Step 2/Step 10: the joblib files are "what the reference notebooks load and they contain all required aligned streams." The reference `utils.load_dat()` defaults to `format="joblib"`, and Figure 1 calls `decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, ...)`, so the AI mirrors the reference loading path exactly and treats the `.mat` files as redundant, larger duplicates of the same content. It documented the axis conventions it inferred (`trace` = (day, registered-cell union, frame), `position` = (day, 2, frame)) and confirmed `position` and `trace` are exactly frame-aligned.

## 1-b. How are the data split into subjects?

i. One file = one subject. The subject ID is the filename (`QLAK-CA1-08`, `-30`, `-50`, `-51`, `-56`, `-74`, `-75`), and files are sorted for deterministic ordering. `subjects` is the list of all seven names; `subject_idx` records the file index for each emitted session.

ii.
```python
start=time.time(); files=animal_files(); subjects=[p.name for p in files]
...
    for si,f in enumerate(files):
        ...
            subject_idx.append(si)
...
data={... 'subjects':subjects, 'subject_idx':np.asarray(subject_idx,dtype=np.int64), ...}
```

iii. CONVERSION_NOTES Step 2 documents that each joblib object is keyed by the animal ID and contains all of that animal's days; the README of the reference repo states files "are given names of animal IDs from the original study." The verification log confirms 31 sessions each for six animals and 21 for `QLAK-CA1-51`.

## 1-c. How are the data split into sessions?

i. One target session = one **animal-day** (equivalently, one environment/geometry exposure). The AI iterates `for day in range(raw['trace'].shape[0])` and emits a separate session per day, giving 207 sessions total. Per-session provenance (`subject`, `source_day_index`, `environment`, `blocked_indices`, frame counts, neuron counts, trial count) is recorded in `metadata['session_info']`.

ii.
```python
        for day in range(raw['trace'].shape[0]):
            st=time.time()
            ntr,itr,otr,info,payload=process_session(
                raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
            neural.append(ntr); inputs.append(itr); outputs.append(otr); subject_idx.append(si)
            region_idx.append(np.zeros(info['retained_neurons'],dtype=np.int64)); session_info.append(info)
```
```python
    info = {
        'subject': subject, 'source_day_index': int(source_day), 'environment': str(environment),
        'blocked_indices': [int(x) for x in np.asarray(blocked).ravel() if x >= 0],
        'source_frames': int(position.shape[1]), 'used_source_frames': int(used_raw),
        'discarded_tail_frames': int(position.shape[1] - used_raw), 'n_trials': int(n_trials), ...}
```

iii. The methods state "All sessions were 40 min, and one session was recorded per day"; the AI verified 71,866–72,219 frames per day (~39.9–40.1 min at 30 Hz) and that the geometry/day counts are 27 open-square plus 20 of each of the other nine geometries = 207, matching the paper's "207 sessions." Each day is a distinct environment geometry, so a day is the natural session unit.

## 1-d. How are the data split into trials?

i. There are no native trials (continuous 40-min free exploration). Per the decoder-task instruction, each session is cut from time zero into **non-overlapping 60-second trials**. Because the AI pools to 100 ms bins, one trial = 600 bins = 1,800 source frames. The incomplete tail is discarded — never padded or wrapped. This yields 39 trials/session for animals with 71,866 frames and 40 for sessions ≥ 72,000 frames; 8,187 trials total.

ii.
```python
FPS = 30; POOL = 3; BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)   # 600
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS            # 1800
```
```python
    n_trials = position.shape[1] // RAW_TRIAL_FRAMES
    used_raw = n_trials * RAW_TRIAL_FRAMES
    if n_trials < 2:
        raise ValueError(f'{subject} day {source_day}: fewer than two complete trials')
    ...
    for tr in range(n_trials):
        sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
        neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
        input_trials.append(geometry.copy())
        output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. CONVERSION_NOTES Step 4/Step 5: "Required one-minute trials are an explicit downstream modification; segment from session start into complete contiguous windows." The AI noted the nominal 40-min duration varies by −4.5 to +7.3 s across sessions and resolved this by "retaining complete 60-s windows and recording every discarded remainder," so that all trials have identical length (a hard requirement of the target format). The `n_trials < 2` guard enforces the format spec's "at least two trials within each session"; it never fired.

## 1-e. How are trials filtered based on quality controls?

i. **No trials are dropped.** The AI explicitly considered and rejected applying the reference decoder's speed criterion (`decode_position_within`: keep only frames whose Gaussian-smoothed speed > 5 cm/s) as a *timepoint* filter, because deleting frames would destroy the contiguous fixed-duration one-minute trial structure and bias the position-occupancy distribution being decoded. The speed mask is used *only* for neuron selection (see 2-c). Structural guards are applied instead: position must be 2×T and fully finite, trace and position frame counts must match, at least one valid cell must survive, at least two complete trials must exist, and all emitted trials are asserted to have the right shapes, finite non-negative neural values, and labels in 0–8.

ii.
```python
    if position.ndim != 2 or position.shape[0] != 2:
        raise ValueError(f'{subject} day {source_day}: bad position shape {position.shape}')
    if trace.ndim != 2 or trace.shape[1] != position.shape[1]:
        raise ValueError(f'{subject} day {source_day}: trace/position mismatch')
    if not np.isfinite(position).all():
        raise ValueError(f'{subject} day {source_day}: nonfinite position')
    ...
    if selected.shape[0] < 1 or not np.isfinite(selected).all():
        raise ValueError(f'{subject} day {source_day}: no valid selected cells')
    if n_trials < 2:
        raise ValueError(f'{subject} day {source_day}: fewer than two complete trials')
    ...
    assert all(x.shape == (selected.shape[0], TRIAL_BINS) for x in neural_trials)
    assert all(np.isfinite(x).all() and np.min(x) >= 0 for x in neural_trials)
    assert labels.min() >= 0 and labels.max() <= 8
```

iii. CONVERSION_NOTES Step 4/Step 5 Decision 4 ("Timepoint curation"): "Preserve all timepoints, including immobility, because deleting frames would violate contiguous one-minute trial timing and bias the requested location distribution. The speed mask is used only for reference cell-quality curation." Step 3 also notes there are no rewarded/failed trials to curate — mice freely explored — and the distributed `within_decoding` results contain 1,035 fold errors (207 × 5), confirming every source session was analyzable in the original paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Solely `trace[day]` — the authors' rise-extracted calcium event traces, shape (registered-cell union, frames), binary 0/1 with `NaN` for cells not registered on that day. No dF/F is computed and no deconvolution is applied. `position[day]` is used only as an auxiliary input to the neuron-selection criterion (to derive the moving-frame mask).

ii.
```python
            ntr,itr,otr,info,payload=process_session(
                raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
```
```python
def process_session(position, trace, blocked, subject, source_day, environment):
    ...
    keep, moving, event_sum = select_cells(position, trace)
    selected = np.asarray(trace[keep], dtype=np.float32)
```

iii. CONVERSION_NOTES Step 1: "Neural data are calcium-derived trace/event activity already provided by the authors, not raw fluorescence. No reference function computes dF/F during analysis; therefore conversion should not recompute dF/F." Step 4 confirms against methods.txt: the released trace is the binarized rising-phase vector (derivative of median-subtracted signal, Gaussian-smoothed σ=5 frames, noise-normalized, thresholded at z > 2.5), which the paper states "was treated as the firing rate in all further analyses."

## 2-b. How is the `neural` data processed?

i. After selecting cells (2-c), the AI reproduces the reference decoder's temporal processing from `fit_decoder`/`test_decoder`: (1) `scipy.ndimage.gaussian_filter1d` along the time axis with **σ = 3 source frames**; (2) crop to the last complete minute; (3) non-overlapping **3-frame mean pooling** (`reshape(..., -1, 3).mean(axis=2)`), giving 100 ms bins; (4) cast to float32; (5) slice into per-trial `(n_neurons, 600)` C-contiguous arrays. Smoothing is deliberately done on the whole continuous session *before* cropping/splitting, so minute boundaries do not create edge artifacts. Values are consequently fractional event rates in [0, 1], not binary.

ii.
```python
    # Match fit_decoder: smooth event vectors continuously, then non-overlapping
    # average pooling in groups of three frames. Smooth before session cropping so
    # minute boundaries do not introduce edge artifacts.
    smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
    neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                                   dtype=np.float32)
```
Reference being matched (`/app/code/georepca1/src/utils.py:1786`):
```python
    pooling = AvgPool1d(kernel_size=temporal_bin_size, stride=temporal_bin_size)
    behav, traces = pooling(torch.tensor(behav.T)).numpy().astype(int).T, \
        pooling(torch.tensor(gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0).T)).numpy().T
```

iii. CONVERSION_NOTES Step 5 Decision 2 ("Temporal processing"): "Match reference `fit_decoder` by applying `gaussian_filter1d(..., sigma=3, axis=time)` to neural traces and averaging non-overlapping 3-frame groups… Smooth across the continuous session before splitting to avoid artificial minute-boundary edge effects." Step 4 adds that "the reference code temporally smooths/pools three source frames (100 ms) for decoding. This is the most directly applicable common neural/behavior time bin and reduces storage while preserving exact alignment." Step 10 check 9 records the line-by-line comparison to `utils.py:1776/1806`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two criteria, combined with AND, reproducing `decode_position_within`:
1. **Presence**: the cell must have at least one finite value on that day (`np.isfinite(trace).any(axis=1)`) — i.e. it was registered/recorded that day. This drops the NaN rows for unregistered cells.
2. **Activity**: summed binary events **during moving frames must exceed 5**, where the moving mask is `gaussian_filter1d(speed, sigma=5) > 5 cm/s` with speed computed as `‖diff(position)‖ × 30`, and the first sample forced to `False` (matching the reference's `vel_idx[d, 1:] = ...`).

This removes 882 of 69,744 day-level cell recordings (1.26%), leaving 68,862 (mean 332.67/session, range 112–562). No place-cell filter is applied.

ii.
```python
def select_cells(position, trace):
    """Match reference position decoder's moving-frame activity curation."""
    present = np.isfinite(trace).any(axis=1)
    # First velocity sample is false, as in decode_position_within.
    displacement_speed = np.linalg.norm(np.diff(position.T, axis=0) * FPS, axis=1)
    moving = np.zeros(position.shape[1], dtype=bool)
    moving[1:] = gaussian_filter1d(displacement_speed, sigma=5, axis=0) > 5.0
    event_sum = np.nansum(trace[:, moving], axis=1)
    keep = present & (event_sum > 5)
    return keep, moving, event_sum
```
Reference being matched (`utils.py:1901`):
```python
        vel_idx[d, 1:] = gaussian_filter1d(np.linalg.norm((behav[1:, :, d] - behav[:-1, :, d]) * fps, axis=1),
                                           axis=0, sigma=v_filt_size) > (v_thresh / bin_down)
        cell_idx[d] = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold
```

iii. CONVERSION_NOTES Step 5 Decision 3 ("Cell curation"): "Start from finite day-specific cells (the paper-curated population), then match `decode_position_within`… Do not apply place-cell-only filtering because reference position decoding uses the population." Step 3 justifies excluding the place-cell criterion: "Place-cell significance… is an analysis-specific classification and was not required for population position decoding; therefore it should not be imposed globally." Step 2 also documents *why* finite-trace presence is the right presence indicator: "Per-day finite trace presence exactly equals finite centroid presence in every animal. SFP presence does not always match and is not a suitable inclusion indicator."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event to align to — recordings are continuous free exploration. The alignment event is therefore defined as the **start of each non-overlapping 1-minute segment**, measured from frame 0 of the animal-day. Trials are cut at exact multiples of 1,800 source frames (600 pooled bins) with no offset, no overlap, no interpolation, and no shift; neural, input, and output are sliced from identical frame indices. Metadata records this explicitly: `temporal_alignment_event` = "Start of each non-overlapping 1-minute segment within a continuous animal-day recording", `off_start = 0.0`, `off_end = 60.0`.

ii.
```python
    for tr in range(n_trials):
        sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
        neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
        input_trials.append(geometry.copy())
        output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```
```python
        'temporal_alignment_event':'Start of each non-overlapping 1-minute segment within a continuous animal-day recording',
        'off_start':0.0,
        'off_end':60.0,
```

iii. CONVERSION_NOTES Step 4: "Reference passes trace and behavior from identical frames. `process_session` pools identical 3-frame slices and raw-source checks prove exact trial alignment. There is no interpolation, shift, or independent truncation." The paper's Figure-1 call `decode_position_within(position.T, trace.T, ...)` establishes that the released streams are already frame-aligned (both acquired at 30 Hz by the same DAQ and timestamped for post-hoc alignment), so preserving common frame indices is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms** (`metadata['time_bin_size'] = 100.0`), i.e. 600 bins per 60-s trial. Yes — the AI rebins from the native 30 Hz (33.33 ms) by a factor of 3, using Gaussian smoothing (σ = 3 frames) followed by non-overlapping 3-frame mean pooling, applied identically to neural and position streams. This matches the reference decoder's `temporal_bin_size=3`.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)  # 600
```
```python
    neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, dtype=np.float32)
    pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
```
```python
        'time_bin_size':BIN_MS,
        'source_frame_rate_hz':FPS,
```

iii. CONVERSION_NOTES Step 4: "The reference code temporally smooths/pools three source frames (100 ms) for decoding. This is the most directly applicable common neural/behavior time bin and reduces storage while preserving exact alignment." Step 5 Decision 7 notes the storage benefit (the full pickle is 6.13 GiB at 100 ms; at 30 Hz it would be ~3× larger). Step 10 check 9 documents the direct match to `fit_decoder`/`test_decoder`.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Solely `blocked[day]` — a ragged per-day list of row-major indices into the 3×3 partition layout `[[0,1,2],[3,4,5],[6,7,8]]`, with `-1` meaning no partition blocked. `envs[day]` (the geometry name string) is read but stored only as descriptive metadata, not used to build the input.

ii.
```python
            ntr,itr,otr,info,payload=process_session(
                raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
```
```python
        geometry = blocked_mask(blocked)
```

iii. CONVERSION_NOTES Step 1/Step 2: the reference repo README defines `blocked` as "location of blocked (occluded) partitions in 3x3 design of environment… organized in the following way – [[0, 1, 2], [3, 4, 5], [6, 7, 8]]. If no partitions are blocked, value is -1." The AI enumerated the ten geometries and their index sets (square `-1`, o `4`, l `1,2,4,5`, u `4,5`, bit donut `0,4`, rectangle `0,3,6`, + `0,2,6,8`, glenn `0,8`, i `3,5`, t `3,5,6,8`) and confirmed the 27/20×9 = 207 session split.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The ragged index list is expanded into a fixed **length-9 float32 binary mask** in the same row-major order: 1 = that partition is blocked, 0 = accessible. Negative sentinels (`-1`) are filtered out, so the open square maps to an all-zero vector; indices > 8 raise a `ValueError`. The mask is static per session and a `.copy()` is appended for every trial, giving shape `(9,)` per trial (the allowed "static per trial" form). Inputs are named `blocked_NW … blocked_SE`.

ii.
```python
def blocked_mask(raw):
    """Expand MATLAB-style blocked indices to a static row-major 3x3 mask."""
    idx = np.asarray(raw).ravel().astype(np.int64)
    mask = np.zeros(9, dtype=np.float32)
    idx = idx[idx >= 0]  # -1 denotes no blocked partition
    if idx.size:
        if np.any(idx > 8):
            raise ValueError(f'Invalid blocked indices: {idx}')
        mask[idx] = 1.0
    return mask
```
```python
INPUT_NAMES = ['blocked_NW','blocked_N','blocked_NE','blocked_W','blocked_center',
               'blocked_E','blocked_SW','blocked_S','blocked_SE']
...
        input_trials.append(geometry.copy())
```

iii. CONVERSION_NOTES Step 5 Decision 6 ("Input representation"): "Geometry is a static 9-vector per trial, exactly as requested. A binary blocked mask is more explicit and decoder-compatible than variable-length blocked-index lists or environment-name categories." Step 10 check 4 verifies mask sums equal raw blocked-index counts, that `-1` maps to all zeros, and that mask frequencies reproduce 27 square and 20 of each other design.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Solely `position[day]`, shape (2, frames): DeepLabCut head-tracked x (row 0) and y (row 1) coordinates in cm. The AI verified these are fully finite and span 0–75 cm on both axes with no NaN/zero tail padding.

ii.
```python
            ntr,itr,otr,info,payload=process_session(
                raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
```
```python
    if not np.isfinite(position).all():
        raise ValueError(f'{subject} day {source_day}: nonfinite position')
    ...
    pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
```

iii. CONVERSION_NOTES Step 2: "Position is finite and ranges 0–75 cm on both axes. Sampled final frames remain active, so arrays are not zero/NaN tail padded." Methods confirm "Position data were generated from tracking the head with DeepLabCut pose-estimation software" at 30 Hz, simultaneously acquired with the imaging stream.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. x and y are **mean-pooled over the same non-overlapping 3-frame groups used for the neural data** (matching the reference's `AvgPool1d` on `behav`), then discretized to a 3×3 grid and flattened to a single row-major class label `y_bin*3 + x_bin`, stored as `int64` with shape `(1, 600)` per trial — i.e. time-varying, as the format spec prefers.

ii.
```python
    pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
    xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
    labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
```
```python
OUTPUT_VALUES = ['NW','N','NE','W','center','E','SW','S','SE']
...
        output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
...
'output_names':['position_bin'],'output_values':[OUTPUT_VALUES]
```

iii. CONVERSION_NOTES Step 5 Decision 5: "Pool continuous position before categorization, matching reference temporal behavior pooling." The row-major convention is taken from the reference README's `blocked` layout so that position class *k* refers to the same physical partition as `blocked_k`; `output_values` names the nine bins NW…SE consistently with `input_names`.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. **Fixed physical boundaries at 0 / 25 / 50 / 75 cm**, taken from the experimental geometry rather than from per-session data min/max: `bin = clip(floor(coord / 25), 0, 2)` on each axis, then `class = y_bin*3 + x_bin` ∈ {0..8}. The `clip` exists only to fold the exact 75.0 cm endpoint (which does occur in the data) into bin 2. The resulting class distribution across all 8,187 trials is `[.0997, .0985, .1351, .0754, .0573, .0768, .1157, .1412, .2003]` — all nine classes present, none dominant.

ii.
```python
ARENA partitioning (in process_session):
    xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
    labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
    ...
    assert labels.min() >= 0 and labels.max() <= 8
```

iii. CONVERSION_NOTES Step 5 Decision 5 ("Output discretization"): "Use physical fixed boundaries 0,25,50,75 cm from the experimental 3×3 geometry, not per-session min/max." Step 10 check 11: "Paper defines nine 25×25 cm partitions of the 75×75 cm arena. Conversion uses fixed physical boundaries and row-major `y*3+x`, clipping only exact upper endpoints. This differs from reference decoder's 15×15 one-hot solely because the requested output is 3×3." Step 10 check 14 additionally cross-validates the x/y→index orientation by measuring occupancy of *blocked* partitions, which must be ~zero if the orientation is right.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame. `position` and `trace` share the same frame index on each day, are cropped to the same `used_raw` frames, pooled over the *same* 3-frame groups, and sliced with the *same* `slice(tr*600, (tr+1)*600)`. No interpolation, resampling, lag, or independent truncation is applied. The `--show-processing` plots overlay pooled x/y with the class labels on a shared time axis and plot the trajectory colored by class over the 3×3 boundaries to make any misalignment visible.

ii.
```python
    smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
    neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, dtype=np.float32)
    pos_pooled    = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
    ...
    for tr in range(n_trials):
        sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
        neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
        output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. CONVERSION_NOTES Step 4/Step 10 check 8: the DAQ "simultaneously acquired behavioral and cellular imaging streams at 30 Hz… all recorded frames were timestamped for post-hoc alignment", and the reference passes both streams from identical frames, so "preserve common frame indices" is the correct alignment rule. The AI's independent `raw_sanity_checks.py` re-derives both streams from the original files for sessions/trials (0,5), (103,17), (206,39) and confirms exact agreement.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Cases handled:
- **NaN traces for unregistered cells** — removed by the `present` mask before any smoothing, so NaN never propagates. `np.nansum` is used for the activity criterion so NaN cells cannot corrupt it.
- **`blocked = -1` sentinel and ragged blocked lists** — `.ravel()` plus `idx[idx >= 0]` handle both; out-of-range indices raise.
- **Variable session length (71,866–72,219 frames)** — only complete 60-s windows are kept; the remainder (0–1,799 frames) is discarded and *recorded* per session as `source_frames`, `used_source_frames`, `discarded_tail_frames`, with `source == used + discarded` asserted.
- **Exact 75.0 cm coordinate at the arena edge** — `np.clip(..., 0, 2)` folds it into the last bin instead of producing class 3.
- **Corrupt output on interruption** — the pickle is written to `<out>.tmp` and atomically `os.replace`d.
- **Hard failures** are raised rather than silently skipped (bad position shape, nonfinite position, trace/position frame mismatch, zero surviving cells, fewer than two complete trials).

ii.
```python
    present = np.isfinite(trace).any(axis=1)
    ...
    event_sum = np.nansum(trace[:, moving], axis=1)
```
```python
    idx = idx[idx >= 0]  # -1 denotes no blocked partition
```
```python
    n_trials = position.shape[1] // RAW_TRIAL_FRAMES
    used_raw = n_trials * RAW_TRIAL_FRAMES
    ...
        'source_frames': int(position.shape[1]),
        'used_source_frames': int(used_raw),
        'discarded_tail_frames': int(position.shape[1] - used_raw),
```
```python
    tmp=Path(str(outfile)+'.tmp')
    with open(tmp,'wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,outfile)
```

iii. CONVERSION_NOTES Step 5 Decision 8: "Position is fully finite. NaN traces occur only for absent registered cells and are removed before smoothing. Coordinate values are clipped after division solely to handle exact 75-cm endpoints. Every source session yields at least 39 trials." Step 10 "Issues Found and Resolved" documents the ragged-`blocked` handling fix and the variable frame-count resolution; Step 6 documents the atomic write "to avoid leaving a corrupt target if interrupted."

## 6-a. What are the most time-consuming steps of the code?

i. The AI instrumented and printed per-animal load time and per-session processing time. Measured from `conversion_full_out.txt`: total 219.18 s, of which **joblib decompression of the seven compressed animal files = 87.7 s (40%)**, **per-session processing ≈ 0.31–0.92 s × 207 ≈ 126 s (57%)** — dominated by `gaussian_filter1d` over the (n_cells × ~72,000) float64 trace and the reshape/mean pooling — and **pickle serialization of the 6.13 GiB output = 5.33 s (2%)**. In its notes the AI ranks the compressed-file decompression as the top inefficiency and states processing "scales with neuron count."

ii.
```python
        load_t=time.time(); raw=joblib.load(f)[f.name]
        print(f'Loaded {f.name} in {time.time()-load_t:.2f}s',flush=True)
        ...
            print(f"  {sid}: neurons {info['source_present_neurons']}->{info['retained_neurons']}, "
                  f"trials {info['n_trials']}, {time.time()-st:.2f}s",flush=True)
        ...
    print(f'Saved {outfile} ({Path(outfile).stat().st_size/2**30:.3f} GiB) in {time.time()-save_t:.2f}s; total {time.time()-start:.2f}s',flush=True)
```

iii. CONVERSION_NOTES Step 6 "Code inefficiencies identified": "Source joblib files are compressed and must each be decompressed once. The target list-of-trial-arrays format necessarily copies each 60-second trial. Class-count accumulation… is negligible compared with neural processing." Step 7 estimated the full run conservatively at <10 min, below the 15-minute optimization threshold; the actual 3.65 min beat the estimate, so no further optimization was required.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI states it already replaced per-frame loops with vectorized operations: Gaussian filtering over the whole (cells × time) array at once, `reshape(..., -1, 3).mean(axis=2)` instead of a binning loop, and arithmetic label construction (`y*3+x` on whole arrays) instead of the reference's per-sample one-hot loop (`for i in range(n_samples): empty_map[behav[i][0], behav[i][1]] += 1 ...`). The remaining loops are the per-animal loop, the per-day loop (both inherently sequential, and needed to bound memory), and the per-trial append loop. The per-trial loop is a materialization of contiguous slices into the required list-of-arrays format, so it cannot be removed without changing the output structure, though it could be written as a single `reshape` + `np.split`/`ascontiguousarray` over the neuron axis rather than 39–40 separate slice copies. `validate_complete` also re-walks every trial in a Python double loop after conversion, and `class_counts` concatenates the outputs of each session a second time — both are cheap relative to the neural work.

ii.
```python
    smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
    neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2, dtype=np.float32)
    pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
    xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
    labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
```
```python
    for tr in range(n_trials):                      # remaining per-trial copy loop
        sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
        neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
        ...
            class_counts += np.bincount(np.concatenate([x.ravel() for x in otr]),minlength=9)
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "Vectorized Gaussian filtering, reshape-based pooling, and label construction replace per-frame loops. Float32 neural storage halves memory/disk relative to source float64. One-animal-at-a-time loading bounds source memory… Smoothing is performed once per continuous session before slicing all trials." It also acknowledges that "the target list-of-trial-arrays format necessarily copies each 60-second trial."

## 6-c. What processing does the code repeat multiple times?

i. The AI documents no significant repeated processing, and the major steps (load, smooth, pool, discretize) are each performed once per session. Reviewing the code, three small repeats exist that the AI did not flag:
1. **`np.isfinite(trace).any(axis=1)` is computed twice per session** — once inside `select_cells` as `present`, and again in `process_session` when building `info['source_present_neurons']`. This is a full pass over the (n_cells × ~72,000) float64 source array, so it is the largest of the redundancies (though still small relative to the Gaussian filter).
2. **`np.asarray(blocked).ravel()` is parsed twice** — once in `blocked_mask()` and again for `info['blocked_indices']`.
3. **Every trial is re-walked twice after conversion** — once by `class_counts` (concatenating each session's outputs) and once by `validate_complete`, which re-checks shapes/lengths that were already asserted inside `process_session`.

ii.
```python
    keep, moving, event_sum = select_cells(position, trace)   # computes `present` internally
    ...
    info = {
        ...
        'blocked_indices': [int(x) for x in np.asarray(blocked).ravel() if x >= 0],   # re-parse
        'source_present_neurons': int(np.isfinite(trace).any(axis=1).sum()),          # recomputed
```
```python
    assert all(x.shape == (selected.shape[0], TRIAL_BINS) for x in neural_trials)   # in process_session
...
def validate_complete(data):                                                        # again at the end
    for s in range(ns):
        for a,b,c in zip(data['neural'][s],data['input'][s],data['output'][s]):
            if a.shape != (n,TRIAL_BINS) or b.shape != (9,) or c.shape != (1,TRIAL_BINS):
```

iii. CONVERSION_NOTES Step 6 lists only "the target list-of-trial-arrays format necessarily copies each 60-second trial" and notes class-count accumulation is "negligible compared with neural processing." The redundant `isfinite` pass is not mentioned; `select_cells` already returns `keep`, `moving`, and `event_sum`, so returning `present` too would have removed it at zero cost. None of these changes the output, and the full conversion still runs in 3.65 min.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, and all of it cheap:
- **`plot_payload` is built on every one of the 207 sessions** even when `--show-processing` is not passed (and after the first two sessions, when no further plots are made). It is only array slices, so the cost is negligible, but it is unconditional work whose result is discarded 205+ times.
- **`event_sum` is returned from `select_cells` but never used** outside the `keep` computation; `moving` is retained only to compute the descriptive `moving_frame_fraction` in `session_info`.
- **Gaussian smoothing is applied to the discarded tail frames** (up to 1,799 frames per session) before cropping to `used_raw` — deliberate, since smoothing before cropping avoids boundary artifacts at the last retained trial, but the tail's smoothed values are thrown away.
- **`validate_complete` and the `class_counts` accumulation** produce reporting/QC values that the decoder never consumes (see 6-c).
- The `envs` string and the extensive `session_info` provenance are stored in metadata and unused by `train_decoder.py`, but they are required for traceability and are trivial in size relative to the 6.13 GiB of neural data.

Notably, the code avoids the largest potential waste: it does **not** regenerate the precomputed 15×15 rate maps, load `SFPs`/`centroids`/`maps`, or read the redundant `.mat` duplicates.

ii.
```python
    plot_payload = (pos_pooled[:, :TRIAL_BINS], labels[:TRIAL_BINS],
                    neural_pooled[:min(30, selected.shape[0]), :TRIAL_BINS], geometry)
    return neural_trials, input_trials, output_trials, info, plot_payload
```
```python
            if show_processing and plots < 2:
                make_processing_plot(payload,sid); plots += 1
```
```python
    smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
    neural_pooled = smooth[:, :used_raw].reshape(...)   # tail smoothing discarded here
```

iii. CONVERSION_NOTES Step 10 check 6: "precomputed maps are intentionally not regenerated because direct aligned streams define the requested decoder data." Step 6 justifies smoothing before cropping: "Smoothing is performed once per continuous session before slicing all trials" and (in `convert_data.py`'s inline comment) "Smooth before session cropping so minute boundaries do not introduce edge artifacts." Step 5 Decision 7 justifies retaining `session_info`: it "enables source traceability and edge-case review."
