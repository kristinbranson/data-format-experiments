# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every first-level directory under `/app/data` as a subject and every second-level directory as a session. For each session it memory-maps `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, and `motion_energy_glob.npy`; full mode processes all discovered sessions sequentially.

ii.
```python
for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        out.append((subject_dir.name, session_dir.name, session_dir))
...
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r')
```

iii. The notes identify `/app/data/<mouse>/<date_session>/` as the hierarchy, with six mice and 41 sessions. Memory mapping and sequential processing were chosen to limit memory use.

## 1-b. How are the data split into subjects?

i. Each first-level directory is treated as one mouse. Subject names are sorted, deduplicated from the discovered session tuples, and mapped to integer indices.

ii.
```python
subjects = sorted({x[0] for x in sessions})
subject_lookup = {x: i for i, x in enumerate(subjects)}
'subject_idx': np.asarray([subject_lookup[x[0]] for x in sessions], dtype=np.int64),
```

iii. The notes report that the six `jm...` directories correspond to the six experimental mice and that within-mouse neuron counts/order are constant longitudinally.

## 1-c. How are the data split into sessions?

i. Each sorted second-level directory under a mouse is one session and becomes one element of each top-level session list.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
    out.append((subject_dir.name, session_dir.name, session_dir))
...
for i, (subject, sid, path) in enumerate(sessions):
    n, x, y, info, dt = process_session(subject, sid, path, show)
    neural.append(n); inputs.append(x); outputs.append(y)
```

iii. The agent interpreted each dated directory as a daily recording and preserved deterministic lexical ordering.

## 1-d. How are the data split into trials?

i. Continuous recordings are divided into non-overlapping complete 60-second trials (1,800 raw frames or 180 ten-frame bins). The usable duration is the neural/motion shared prefix; any incomplete terminal trial is dropped.

ii.
```python
shared_frames = min(F.shape[1], motion.shape[0])
n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
...
for trial in range(n_trials):
    a = trial * PROCESSED_POINTS_PER_TRIAL
    b = a + PROCESSED_POINTS_PER_TRIAL
    neural_trials.append(neural_binned[:, a:b])
```

iii. The data have no native trials, so the requested 60-second segmentation was imposed. The agent explicitly chose shared-prefix trimming rather than inventing behavior values for missing terminal video frames.

## 1-e. How are trials filtered based on quality controls?

i. Only complete trials in the shared neural/behavior prefix are retained, and a session is rejected if fewer than two remain. Trials are also validated for shape, dtype, finite values, and label range.

ii.
```python
if n_trials < 2:
    raise ValueError(f'{subject}/{session_id}: fewer than two complete 60-s trials')
...
assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
assert y.min() >= 0 and y.max() <= 4
```

iii. This enforces the decoder’s two-trial minimum and avoids partial or unmatched trials. The notes report 1,081 retained trials after nine motion-short sessions lost their last trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from Suite2p `F.npy` and `Fneu.npy`, with processing parameters from `ops.npy`. `iscell.npy` is checked as a curation invariant.

ii.
```python
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```

iii. The agent concluded that `F` is raw fluorescence and selected the paper/Suite2p-style neuropil-corrected, baseline-subtracted fluorescence rather than `spks` or a plotting z-score.

## 2-b. How is the `neural` data processed?

i. It computes `Fc = F - neucoeff*Fneu`, estimates a Suite2p maximin baseline using Gaussian smoothing followed by minimum and maximum filters, subtracts that baseline, then averages non-overlapping groups of 10 frames.

ii.
```python
Fc = np.asarray(F, dtype=np.float32) - np.float32(neucoeff) * np.asarray(Fneu, dtype=np.float32)
Flow = gaussian_filter(Fc, [0.0, sig])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
return (Fc - Flow).astype(np.float32, copy=False), Fc, Flow
...
return x.reshape(x.shape[0], n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=2, dtype=np.float32)
```

iii. The notes reason that the paper’s “baseline corrected fluorescence” corresponds to Suite2p preprocessing: default 0.7 neuropil subtraction, maximin baseline subtraction, and the paper’s 10-timestamp denoising.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron is removed during conversion. The agent requires every supplied ROI already to have `iscell[:,0] == 1`, checks matching row counts, and validates finite converted values.

ii.
```python
if F.shape[0] != iscell.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(f'{subject}/{session_id}: supplied matched-cell curation is inconsistent')
```

iii. The notes infer that the supplied arrays are already Track2p-matched, accepted-cell exports because all flags are one and cell counts/order are constant within mouse.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural and motion samples are aligned by common frame index. Saved trials are aligned to the start of each artificial 60-second segment; metadata declares offsets 0 to 60 seconds.

ii.
```python
'temporal_alignment_event': 'start of each non-overlapping 60-second trial; source video was hardware-triggered by microscope',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The agent cites microscope-triggered 30 Hz video acquisition as support for frame-index alignment and corrected its metadata during review so offsets describe trial start consistently.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 30 Hz data are averaged in non-overlapping 10-frame bins, yielding 3 Hz data, 333.333 ms bins, and 180 samples per 60-second trial.

ii.
```python
TEMPORAL_AVG = 10
PROCESSED_POINTS_PER_TRIAL = RAW_FRAMES_PER_TRIAL // TEMPORAL_AVG
...
'time_bin_size': 1000.0 * TEMPORAL_AVG / 30.0,
```

iii. This directly follows the methods statement that both dF/F and behavior traces were averaged over 10 consecutive timestamps for decoding.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from the processed-bin index, the fixed 10 raw frames per bin, and the session’s `ops['fs']`; no timestamp file is used.

ii.
```python
fs = float(ops['fs'])
elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
            + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)
```

iii. Hardware synchronization and verified 30 Hz sampling led the agent to use frame indices. It describes this as elapsed seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Each ten-frame bin is assigned its center time: `(10*b + 4.5)/fs`. The continuous session-level vector is cast to float32 and later sliced into trials.

ii.
```python
elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
            + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)
```

iii. The notes explicitly document the bin-center convention, e.g. the first trial spans approximately 0.15 to 59.8167 seconds.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector has one value per jointly binned sample and is sliced with exactly the same `[a:b]` trial indices as neural and output data. It remains absolute session time rather than resetting each trial.

ii.
```python
neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
```

iii. The agent’s independent checks verified continuity across trial boundaries and alignment of all three streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived directly from each session’s `move_deve/motion_energy_glob.npy` global motion-energy trace.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r')
```

iii. The notes describe this source as the precomputed sum of squared pixel differences between consecutive behavioral-video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent truncates to the complete shared neural/motion prefix, averages motion over non-overlapping 10-frame bins, computes session-local quintile thresholds over retained bins, and converts values to integer labels 0–4.

ii.
```python
motion_binned = mean_bins_1d(motion, used_frames)
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. Ten-frame averaging matches the decoding methods, and per-session percentiles implement the requested “selected per session” equal-percentile categories. Trimming was justified as avoiding fabricated behavioral data.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20/40/60/80% session-specific quantiles are calculated and `searchsorted(..., side='right')` assigns five zero-indexed categories.

ii.
```python
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. This yields approximately balanced classes within each session and is compatible with cross-entropy labels 0–4; right-side tie handling is documented by the implementation.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is by the hardware-synchronized frame index, but the agent uses only `min(neural length, motion length)` and drops any unmatched terminal neural data rather than interpolating missing motion frames. Both retained streams are then binned and sliced identically.

ii.
```python
shared_frames = min(F.shape[1], motion.shape[0])
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
neural_binned = mean_bins_2d(neural_bc, used_frames)
motion_binned = mean_bins_1d(motion, used_frames)
```

iii. The agent observed all motion deficits at session ends and therefore considered shared-prefix trimming safer than inventing values. Independent source checks confirmed its saved arrays follow this policy.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Shape/rate/curation violations raise errors. Missing terminal motion samples are handled by restricting conversion to the shared prefix and dropping incomplete 60-second trials. No interpolation or padding is performed; finiteness and categorical bounds are asserted.

ii.
```python
if F.shape != Fneu.shape:
    raise ValueError(...)
shared_frames = min(F.shape[1], motion.shape[0])
n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
...
assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
```

iii. The notes identify nine motion-short sessions and state that missing samples are terminal, motivating use of the shared valid prefix. This produces 1,081 trials.

## 6-a. What are the most time-consuming steps of the code?

i. Full-session maximin baseline estimation is the dominant computation; loading/serializing the large arrays is the other material cost. The script records per-session and wall-clock timing.

ii.
```python
Flow = gaussian_filter(Fc, [0.0, sig])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
...
dt = time.perf_counter() - t0
```

iii. The notes anticipated sliding filters over every neuron and frame as the costly stage and report a full conversion time around 36 seconds for 41 sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop mainly slices arrays and creates the required nested lists; it could be partially reshaped/vectorized before conversion to lists. Session discovery/processing must still produce per-session objects, while validation’s nested trial loop could operate on concatenated arrays.

ii.
```python
for trial in range(n_trials):
    a = trial * PROCESSED_POINTS_PER_TRIAL
    b = a + PROCESSED_POINTS_PER_TRIAL
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. The agent did not identify this as a bottleneck; expensive numerical filtering and averaging are already vectorized, and the loop directly constructs the mandated list-of-trials format.

## 6-c. What processing does the code repeat multiple times?

i. Each session repeats loading, baseline filtering, 10-frame averaging, quantile computation, trial slicing, and validation. Validation then revisits every saved trial for shapes, dtypes, finiteness, and label bounds.

ii.
```python
for i, (subject, sid, path) in enumerate(sessions):
    n, x, y, info, dt = process_session(subject, sid, path, show)
...
for s in range(ns):
    for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
        assert np.isfinite(n).all() ...
```

iii. Per-session repetition is required because parameters and quintiles are session-specific. The second validation pass is intentional defensive checking rather than part of the scientific transform.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `Fc` and `Flow` are retained solely for optional audit plots, and `iscell` is loaded solely to assert prior curation. Baseline correction is computed over the complete neural recording even when terminal samples are later discarded; this preserves full-session baseline estimation but processes some unsaved samples. Optional plotting also performs visualization-only work.

ii.
```python
neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
...
if show_processing:
    make_processing_plot(..., Fc, Flow, ...)
```

iii. The agent deliberately estimates baseline over the full neural session to match Suite2p and prevent behavior availability from altering neural preprocessing. Plotting is gated by `--show-processing` and limited to two sessions, so it is an audit aid rather than required downstream data.
