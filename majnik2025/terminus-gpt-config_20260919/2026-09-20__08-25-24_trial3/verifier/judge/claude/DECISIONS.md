# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers data by walking the two-level directory tree under `/app/data`: every top-level directory is treated as a subject and every sub-directory of a subject as a session, sorted alphabetically at both levels. This yields 6 subjects and 41 sessions (jm031=7, jm032=7, jm038=7, jm039=7, jm040=6, jm046=7). There is no trial structure in the raw data, so trials are constructed later (see 1-d). For each session it loads, with `mmap_mode='r'` where possible, `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `suite2p/plane0/iscell.npy`, `suite2p/plane0/ops.npy` (for `fs`, `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`) and `move_deve/motion_energy_glob.npy`. It does **not** load `move_deve/tstamps.npy` or `move_deve/interframe_int.npy`, and it does not read `/app/data/README.md` (Step 2 of CONVERSION_NOTES.md states, incorrectly, "No README, table, Track2p match matrix, or other metadata file is present in the data tree").

ii.
```python
DATA_ROOT = Path('/app/data')

def discover_sessions(root=DATA_ROOT):
    """Return sorted (subject, session_id, session_path) tuples."""
    out = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            out.append((subject_dir.name, session_dir.name, session_dir))
    return out
```
```python
    plane = path / 'suite2p' / 'plane0'
    move = path / 'move_deve'
    F = np.load(plane / 'F.npy', mmap_mode='r')
    Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r')
```

iii. From CONVERSION_NOTES.md Steps 2/4/5: the hierarchy is `/app/data/<mouse>/<YYYY-MM-DD_a>/`, each session holding one `suite2p/plane0/` folder and one `move_deve/` folder. Sorting both levels gives deterministic, chronological ordering. Suite2p parameters are read from each session's own `ops.npy` rather than hard-coded "so as to reproduce default Suite2p analysis processing" exactly as it was configured for that recording. `iscell.npy` is loaded only to *verify* that the supplied export is already curated (all flags equal 1), not to re-filter.

## 1-b. How are the data split into subjects?

i. Each top-level directory under `/app/data` is one mouse. Subject names are the sorted unique set of directory names over the discovered sessions, and `subject_idx` is built by looking each session's parent folder name up in that list. Six subjects result: `jm031, jm032, jm038, jm039, jm040, jm046`.

ii.
```python
    sessions = discover_sessions()
    if args.sample:
        sessions = sessions[:2]
    subjects = sorted({x[0] for x in sessions})
    subject_lookup = {x: i for i, x in enumerate(subjects)}
    ...
        'subjects': subjects,
        'subject_idx': np.asarray([subject_lookup[x[0]] for x in sessions], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 2: "Hierarchy: `/app/data/<mouse>/<YYYY-MM-DD_a>/`. Six mice: jm031, jm032, jm038, jm039, jm040, jm046." The paper's "A total of 6 mice were used in the study" is quoted in Step 3 as the consistency check, and Step 9 records 6 subjects in the converted data, matching.

## 1-c. How are the data split into sessions?

i. Every sub-directory of a subject folder is one session (one daily recording), sorted alphabetically, i.e. chronologically because the folder names are `YYYY-MM-DD_a`. All 41 sessions are kept; none are excluded. Sessions of both lengths found in the data (36,000 frames = 20 min and 54,000 frames = 30 min at 30 Hz) are retained. Each converted session becomes one entry of `neural`/`input`/`output`.

ii.
```python
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            out.append((subject_dir.name, session_dir.name, session_dir))
```
```python
    info = {
        'subject': subject, 'session_id': session_id, 'n_neurons': int(F.shape[0]),
        'neural_source_frames': int(F.shape[1]), 'motion_source_frames': int(motion.shape[0]),
        ...
    }
```

iii. Step 2: "Session counts: jm031=7, jm032=7, jm038=7, jm039=7, jm040=6, jm046=7 (41 total)", consistent with the paper's "at least 6 consecutive days". Step 4 explicitly decided to keep the 30-minute sessions despite the Methods stating 20-minute recordings: "Preserve all valid supplied data. The 30-minute sessions are internally consistent and likely reflect an expanded/exported cohort; arbitrary truncation would discard valid synchronized data."

## 1-d. How are the data split into trials?

i. The raw recordings are continuous with no native trial structure, so trials are defined as instructed: non-overlapping 60-second segments = 1,800 raw frames = 180 averaged bins. The number of trials per session is `shared_frames // 1800`, where `shared_frames = min(n_neural_frames, n_motion_frames)`; any partial trailing minute is discarded. This gives 1,081 trials (19–20 per 20-min session, 29–30 per 30-min session). Note that because `shared_frames` is capped by the (sometimes shorter) motion array rather than by the neural array, the nine sessions with dropped camera frames lose one *whole* extra trial each (19 instead of 20, 29 instead of 30); with the reference's interpolation they would have yielded 1,090 trials.

ii.
```python
RAW_FRAMES_PER_TRIAL = 1800  # 60 s * 30 Hz
TEMPORAL_AVG = 10
PROCESSED_POINTS_PER_TRIAL = RAW_FRAMES_PER_TRIAL // TEMPORAL_AVG
...
    shared_frames = min(F.shape[1], motion.shape[0])
    n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
    if n_trials < 2:
        raise ValueError(f'{subject}/{session_id}: fewer than two complete 60-s trials')
    used_frames = n_trials * RAW_FRAMES_PER_TRIAL
...
    for trial in range(n_trials):
        a = trial * PROCESSED_POINTS_PER_TRIAL
        b = a + PROCESSED_POINTS_PER_TRIAL
        neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
        input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
        output_trials.append(np.ascontiguousarray(labels[None, a:b], dtype=np.int64))
```

iii. Step 4: "Trial definition — reference decoder splits on consecutive 2-minute blocks for cross-validation; native recordings are continuous. Explicit downstream instruction overrides this: create non-overlapping 60-second trials." Step 5 Key Decision 4: "Each trial is exactly 1,800 raw frames or 180 processed samples. Only complete trials are retained." Trials are taken from the "shared neural/behavior prefix" because the AI believed missing camera frames were terminal (see 4-d).

## 1-e. How are trials filtered based on quality controls?

i. No quality-based trial filtering is applied; every complete 60-second segment is kept. Two structural guards exist: a session with fewer than two complete trials raises an error (never triggered — the shortest session yields 19 trials), and `validate_converted` asserts that every session has ≥2 trials, that all trials are exactly 180 bins, that dtypes are as expected, that all values are finite, and that labels lie in 0–4.

ii.
```python
    if n_trials < 2:
        raise ValueError(f'{subject}/{session_id}: fewer than two complete 60-s trials')
```
```python
def validate_converted(data):
    ns = len(data['neural'])
    assert ns == len(data['input']) == len(data['output']) == len(data['subject_idx'])
    assert ns == len(data['brain_region_idx'])
    for s in range(ns):
        assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s]) >= 2
        assert len(data['brain_region_idx'][s]) == data['neural'][s][0].shape[0]
        for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
            assert n.shape[1] == x.shape[1] == y.shape[1] == PROCESSED_POINTS_PER_TRIAL
            assert n.dtype == np.float32 and x.dtype == np.float32 and y.dtype == np.int64
            assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
            assert y.min() >= 0 and y.max() <= 4
```

iii. There is no behavioural/task criterion in this spontaneous-activity dataset by which a trial could be judged bad; Step 2 records that "All scanned neural and motion arrays contain finite values." The ≥2-trials guard exists because the format specification requires "at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `suite2p/plane0/F.npy` (ROI fluorescence, n_neurons × n_frames, float32) and `suite2p/plane0/Fneu.npy` (neuropil fluorescence, same shape), with `suite2p/plane0/ops.npy` supplying the processing parameters (`fs=30`, `neucoeff=0.7`, `baseline='maximin'`, `win_baseline=60.0`, `sig_baseline=10.0`). `spks.npy` (Suite2p deconvolved activity) was examined during exploration but deliberately not used.

ii.
```python
    F = np.load(plane / 'F.npy', mmap_mode='r')
    Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```
```python
    neucoeff = float(ops.get('neucoeff', 0.7))
    fs = float(ops['fs'])
    sig = float(ops.get('sig_baseline', 10.0))
    win = int(float(ops.get('win_baseline', 60.0)) * fs)
    baseline_mode = ops.get('baseline', 'maximin')
```

iii. Step 4: the paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses", and Step 2 confirmed `F.npy` is still on a raw fluorescence scale (roughly tens to thousands), not normalised. Therefore the neural signal must be rebuilt from `F` and `Fneu` using the Suite2p parameters stored in `ops`.

## 2-b. How is the `neural` data processed?

i. The AI reproduces Suite2p's default preprocessing in-line rather than importing `suite2p.extraction.dcnv`: (1) neuropil subtraction `Fc = F - neucoeff*Fneu` with `neucoeff = 0.7` from `ops`; (2) maximin baseline — Gaussian smoothing along time with `sigma = sig_baseline = 10` frames, then a rolling `minimum_filter1d` followed by a rolling `maximum_filter1d` with window `win_baseline*fs = 60*30 = 1800` frames; (3) baseline **subtraction** (no division). Baseline estimation is run on the complete session before any trimming. The result is then averaged in non-overlapping 10-frame bins and cast to float32. No z-scoring, no dF/F division, no per-neuron normalisation. (I verified numerically that this reimplementation reproduces `dcnv.preprocess(..., baseline='maximin')` on jm031/2023-10-18: r = 0.99996, mean |difference| 0.06 against a signal scale of ~29.6, the residual being float32/edge-handling differences between the scipy and torch implementations.)

ii.
```python
def baseline_correct(F, Fneu, ops):
    """Reproduce default Suite2p/Track2p maximin baseline subtraction."""
    neucoeff = float(ops.get('neucoeff', 0.7))
    fs = float(ops['fs'])
    sig = float(ops.get('sig_baseline', 10.0))
    win = int(float(ops.get('win_baseline', 60.0)) * fs)
    baseline_mode = ops.get('baseline', 'maximin')
    Fc = np.asarray(F, dtype=np.float32) - np.float32(neucoeff) * np.asarray(Fneu, dtype=np.float32)
    if baseline_mode == 'maximin':
        Flow = gaussian_filter(Fc, [0.0, sig])
        Flow = minimum_filter1d(Flow, win, axis=1)
        Flow = maximum_filter1d(Flow, win, axis=1)
    ...
    return (Fc - Flow).astype(np.float32, copy=False), Fc, Flow
```
```python
    # Baseline estimation uses the complete neural recording, matching Suite2p processing.
    neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
    neural_binned = mean_bins_2d(neural_bc, used_frames)
```

iii. Step 4: "Reproduce default Suite2p analysis processing: `Fc=F-neucoeff*Fneu`, smooth with `sig_baseline`, rolling minimum then maximum over `win_baseline*fs`, and subtract baseline. Do not divide by baseline." Step 5 Key Decision 1: "Literal division by the baseline was rejected because the repository and Suite2p both subtract, and division creates extreme artifacts around nonpositive baselines." Key Decision 9: "No z-scoring: the reference uses z-score only for plotting, not saved analysis traces" (the Track2p demo notebook z-scores only for raster display).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. Instead the AI *verifies* that the supplied export is already curated: it asserts `F` and `Fneu` have identical shapes, that `iscell` has one row per ROI and that **all** `iscell[:,0] == 1`, and that `ops['fs']` is 30 Hz; any violation raises. All 20,445 neuron-session rows pass. All neurons are assigned to a single brain region, `barrel cortex L2/3`.

ii.
```python
    if F.shape != Fneu.shape:
        raise ValueError(f'{subject}/{session_id}: F and Fneu shapes differ')
    if F.shape[0] != iscell.shape[0] or not np.all(iscell[:, 0] == 1):
        raise ValueError(f'{subject}/{session_id}: supplied matched-cell curation is inconsistent')
    fs = float(ops['fs'])
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{subject}/{session_id}: expected 30 Hz, found {fs}')
```
```python
        'brain_regions': ['barrel cortex L2/3'],
        'brain_region_idx': [np.zeros(n[0].shape[0], dtype=np.int64) for n in neural],
```

iii. Step 4: "Track2p defaults to Suite2p binary iscell ...; matched exports remove rows missing any day. All provided rows have `iscell[:,0]=1`; each mouse has a constant neuron count and ordering across days ... Data are already cell-filtered and all-day matched. Retain every supplied row; do not filter a second time." Step 5 Key Decision 2: "a second filter would incorrectly discard curated cells." Step 3 records the paper's curation rules (classifier probability > 0.5, then Track2p all-day matching) as already applied upstream.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus event: trials are contiguous, non-overlapping 60-second blocks of the continuous recording, starting at session start. Trial *k* covers processed bins `[180k, 180k+180)`, i.e. raw frames `[1800k, 1800k+1800)`. Neural, input and output are sliced with exactly the same indices, so they are aligned by construction. Metadata records the alignment event as the start of each 60-second trial, with `off_start = 0.0` and `off_end = 60.0`.

ii.
```python
    for trial in range(n_trials):
        a = trial * PROCESSED_POINTS_PER_TRIAL
        b = a + PROCESSED_POINTS_PER_TRIAL
        neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
        input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
        output_trials.append(np.ascontiguousarray(labels[None, a:b], dtype=np.int64))
```
```python
            'temporal_alignment_event': 'start of each non-overlapping 60-second trial; source video was hardware-triggered by microscope',
            'off_start': 0.0,
            'off_end': 60.0,
            'trial_duration_seconds': 60.0,
```

iii. Step 10 "Issues Found and Resolved": "Initial metadata named session start as the alignment event while `off_start=0`, `off_end=60` referred to trial boundaries. Fixed `convert_data.py` to define the event as each 60-second trial start while retaining the explicit session-elapsed input description." Step 7 verified there is no off-by-one at the boundary: "Trial-1 bin centers are 0.15–59.8167 s; trial 2 starts at 60.15 s, confirming no overlap/gap beyond the expected 1/3-s sample spacing."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes. Native acquisition is 30 Hz for both the microscope and the camera; the AI averages non-overlapping blocks of 10 consecutive frames, giving 180 bins per 60-s trial, a bin size of 10/30 s = 333.333 ms (3 Hz). The same binning is applied to the neural traces and to the motion-energy trace, and it is applied to motion energy **before** discretisation. `metadata['time_bin_size']` is 333.333 ms. The binning is vectorised by reshaping to (..., n_bins, 10) and taking the mean over the last axis.

ii.
```python
TEMPORAL_AVG = 10

def mean_bins_2d(x, n_frames):
    """Average a neurons/features x time array in nonoverlapping 10-frame bins."""
    x = np.asarray(x[:, :n_frames])
    return x.reshape(x.shape[0], n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=2, dtype=np.float32)

def mean_bins_1d(x, n_frames):
    """Average a time vector in nonoverlapping 10-frame bins."""
    x = np.asarray(x[:n_frames], dtype=np.float64)
    return x.reshape(n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=1)
```
```python
    neural_binned = mean_bins_2d(neural_bc, used_frames)
    motion_binned = mean_bins_1d(motion, used_frames)
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
...
            'time_bin_size': 1000.0 * TEMPORAL_AVG / 30.0,
            'temporal_averaging_frames': TEMPORAL_AVG,
```

iii. Step 3: "Decoder analysis likewise averaged both dF/F and behavior in non-overlapping bins of 10 consecutive timestamps (effective approximately 3 Hz)." Step 4: "Apply non-overlapping 10-frame means to both streams, yielding 10/30 s = 333.333 ms bins (3 Hz)." Step 5 Key Decision 3: "Required to match paper decoding." Discretisation deliberately follows binning (Key Decision 6) so that thresholds describe exactly the saved samples.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from any stored time variable. It is computed analytically from the bin index and the nominal frame rate `ops['fs'] = 30 Hz`. The camera clock files `tstamps.npy` / `interframe_int.npy` were inspected during Step 2 (their units were found to need a ×1000 rescaling to give ~0.0336 s intervals) but were explicitly rejected as the time source.

ii.
```python
    fs = float(ops['fs'])
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{subject}/{session_id}: expected 30 Hz, found {fs}')
...
    elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
                + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)
```

iii. Step 5 mapping table: "Frame index and `ops['fs']` → `input[0]` ... Nominal 30 Hz is used because fixed bins are required; timestamps are a camera diagnostic and show only small clock-rate deviation." Step 4: "Use nominal 30 Hz for bin duration and elapsed-time construction."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A single vectorised expression yields the **centre** of each 10-frame averaging window in seconds since the start of that session: `t_k = (10k + 4.5)/30`, so 0.15, 0.4833, 0.8167, … s, step 1/3 s. The vector is built once for the whole session and then sliced per trial, so time runs continuously across trials within a session and is *not* reset to zero at each trial boundary: trial 0 spans 0.15–59.8167 s, trial 1 starts at 60.15 s, and the last bin of a 30-minute session is 1799.8167 s. Stored as float32 with shape (1, 180) per trial, named `time elapsed from session start (s)`. (The reference instead uses the left edge of each bin, `10k/30`, i.e. 0.0–1799.667 s — a constant 0.15 s offset.)

ii.
```python
    elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
                + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)
```
```python
        input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
...
        'input_names': ['time elapsed from session start (s)'],
```

iii. Step 5 Key Decision 7: "Use session-elapsed bin-center times: averaged observations represent the centers of their 10-frame windows. Each trial retains absolute session time rather than resetting to zero, as explicitly requested." Step 10 check 3 independently recomputed the bin-centre times from raw frame indices and 30 Hz and confirmed `np.allclose` against the saved input.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: `elapsed` is indexed by exactly the same binned-time axis as `neural_binned` and `labels`, has the same length, and is sliced with the identical `[a:b]` trial indices, so element *j* of the input is the same 333.3 ms bin as column *j* of the neural matrix. Trial-length equality is asserted in `validate_converted`.

ii.
```python
    neural_binned = mean_bins_2d(neural_bc, used_frames)
    motion_binned = mean_bins_1d(motion, used_frames)
    ...
    elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
                + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)
    for trial in range(n_trials):
        a = trial * PROCESSED_POINTS_PER_TRIAL
        b = a + PROCESSED_POINTS_PER_TRIAL
        neural_trials.append(...)
        input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
```
```python
            assert n.shape[1] == x.shape[1] == y.shape[1] == PROCESSED_POINTS_PER_TRIAL
```

iii. Step 10 checks 3 and 7: bin centres were independently reconstructed and matched with `np.allclose`; "Every trial has 180 bins; first/last centers are trial start +0.15/+59.8167 s; adjacent bins differ by 1/3 s", confirming no off-by-one at trial or session boundaries. The `--show-processing` plots put neural and time/motion panels on a shared session-time axis as a visual check.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Solely `move_deve/motion_energy_glob.npy` (uint64, one value per video frame; the paper's sum over pixels of squared differences between consecutive frames). The companion files `move_deve/tstamps.npy` and `move_deve/interframe_int.npy` — which the dataset README identifies as the way to locate dropped camera frames — are **not** loaded by `convert_data.py`, and `/app/data/README.md` was never read.

ii.
```python
    move = path / 'move_deve'
    motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r')
```

iii. Step 2: "`motion_energy_glob` | frames, uint64 | Nonnegative global framewise motion energy; finite, global range 0 to 65,795,161." Step 4: "Motion energy has already been computed exactly as described in the paper and only needs the same 10-frame averaging." The timestamp files were characterised in Step 2 (`tstamps` "requires multiplication by 1000 to yield seconds"; `interframe_int` = `diff(tstamps)`) but were then treated as a clock diagnostic only, and the AI concluded the missing frames were terminal so no index information was needed.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The raw motion-energy values are used as supplied — no smoothing, rescaling, log transform or normalisation. Two steps are applied: (1) the trace is truncated to `used_frames` (the shared neural/motion prefix rounded down to whole 60-s trials) and averaged in the same non-overlapping 10-frame bins as the neural data, in float64; (2) the binned trace is discretised into five per-session quintile classes (see 4-c). The interpolation of dropped camera frames performed by the reference is absent.

ii.
```python
def mean_bins_1d(x, n_frames):
    """Average a time vector in nonoverlapping 10-frame bins."""
    x = np.asarray(x[:n_frames], dtype=np.float64)
    return x.reshape(n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=1)
```
```python
    motion_binned = mean_bins_1d(motion, used_frames)
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```
```python
            'motion_processing': 'global squared frame-difference energy; 10-frame means; per-session quintiles',
```

iii. Step 4: "Motion energy has already been computed exactly as described in the paper and only needs the same 10-frame averaging." Step 5 mapping table: "`motion_energy_glob.npy` → `output[0]`: shared-prefix alignment; non-overlapping 10-frame mean; compute 20/40/60/80% quantiles per session over retained complete trials." Step 10 check 4 independently reloaded the raw motion array, re-binned and re-discretised it outside the conversion code and confirmed exact agreement for three spot-checked trials.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, after binning and after restricting to the retained complete trials, the 20/40/60/80th percentiles of the binned motion trace are computed with `np.quantile`, and each sample is assigned a class 0–4 by `np.searchsorted(edges, x, side='right')` (ties fall in the higher bin). Edges are recomputed independently for every session — never pooled across sessions or mice — and are stored per session in `metadata['session_info']`. This yields exactly 20.0% of samples per class in every session (the only exception is jm031/2023-10-22, where two tied values give [683, 685, 684, 684, 684]). Class names are `quintile 1 (lowest) … quintile 5 (highest)`.

ii.
```python
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
    ...
    counts = np.bincount(labels, minlength=5)
```
```python
        'output_names': ['motion energy quintile'],
        'output_values': [['quintile 1 (lowest)', 'quintile 2', 'quintile 3',
                           'quintile 4', 'quintile 5 (highest)']],
```

iii. Step 5 Key Decision 6: "Discretize per session after temporal averaging and complete-trial selection: this follows 'five equal-percentile bins, selected per session' and ensures thresholds describe exactly the saved output population." Step 5 mapping table: "Ties at boundaries deterministically enter the higher bin; motion is sufficiently continuous that ties should be rare." Step 10: "One session has two tied samples crossing an edge ... exact equal-size groups are impossible without arbitrarily separating identical motion values ... global imbalance is two of 194,580 samples."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns the two streams purely by frame index from the start of the recording, and handles the nine sessions where the motion array is shorter than the neural array by truncating both to the common prefix `shared_frames = min(n_neural_frames, n_motion_frames)` and then to whole 60-s trials. No interpolation, padding or masking is performed, and the drop indices in `interframe_int.npy` are never consulted. The justification rests on the claim that the missing camera frames occur at the *end* of the session. That claim is false for this dataset: the drops are interior (e.g. jm031/2023-10-22 loses 116 frames starting at frame 653; jm032/2023-10-22 loses 148 starting at frame 212), so from the first drop onward the motion trace is shifted earlier relative to the neural trace by a progressively growing amount (up to ~4–5 s, i.e. 12–15 bins). Recomputing those two sessions with the reference's interpolation gives correlations of only 0.59 and 0.79 against the AI's binned trace; the other seven affected sessions are off by ≤3 frames (<0.1 s) and are effectively unaffected.

ii.
```python
    shared_frames = min(F.shape[1], motion.shape[0])
    n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
    ...
    used_frames = n_trials * RAW_FRAMES_PER_TRIAL
    neural_binned = mean_bins_2d(neural_bc, used_frames)
    motion_binned = mean_bins_1d(motion, used_frames)
```
```python
        'motion_source_frames': int(motion.shape[0]),
        'shared_frames': int(shared_frames), 'used_frames': int(used_frames),
        'unused_shared_terminal_frames': int(shared_frames-used_frames),
        'motion_frames_missing_vs_neural': int(F.shape[1]-motion.shape[0]),
```
```python
            'temporal_alignment_event': 'start of each non-overlapping 60-second trial; source video was hardware-triggered by microscope',
            'trial_policy': 'non-overlapping complete 60-s trials from shared neural/behavior prefix',
```

iii. Step 3: "Videos were recorded at 30 Hz, with the microscope acquisition acting as a trigger", so the AI treats frame index *i* of the video as simultaneous with frame *i* of the imaging. Step 2: "Nine sessions have fewer motion samples: deficits 1 frame (3 sessions), 2 (3), 3 (1), 116 (1), and 148 (1). These deficits occur at the session end because each movement series begins at timestamp zero; conversion must use only the shared valid prefix rather than invent behavioral values." Step 4: "Align by frame index and use only the shared valid prefix. Never interpolate or fabricate missing terminal behavior." Step 5 Key Decision 5: "Motion deficits occur only at the end. No interpolation, extrapolation, or padding is justified."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms: (a) missing camera frames are handled by shared-prefix truncation, as described in 4-d — nine sessions are affected and each loses one whole extra 60-s trial (1,081 trials instead of 1,090), and the two sessions with 116/148 interior drops end up with a misaligned motion trace; (b) the trailing partial minute of every session is discarded; (c) structural mistakes are turned into hard failures — mismatched `F`/`Fneu` shapes, any `iscell` flag ≠ 1, a frame rate ≠ 30 Hz, or a session with <2 complete trials all raise `ValueError`; (d) `validate_converted` asserts shapes, dtypes, finiteness and label range for every trial before the pickle is written, and per-session provenance (source/used frame counts, drop count, quintile edges, class counts) is recorded in `metadata['session_info']`. The AI never read `/app/data/README.md`, which states that "in some recordings there might be some missing frames from the camera ... the indices of missing frames can be obtained by looking at `tstamps.npy` or `interframe_int.npy` and treated as missing values for motion energy or they can be interpolated over".

ii.
```python
    shared_frames = min(F.shape[1], motion.shape[0])
    n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
    if n_trials < 2:
        raise ValueError(f'{subject}/{session_id}: fewer than two complete 60-s trials')
    used_frames = n_trials * RAW_FRAMES_PER_TRIAL
```
```python
    if F.shape != Fneu.shape:
        raise ValueError(f'{subject}/{session_id}: F and Fneu shapes differ')
    if F.shape[0] != iscell.shape[0] or not np.all(iscell[:, 0] == 1):
        raise ValueError(f'{subject}/{session_id}: supplied matched-cell curation is inconsistent')
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{subject}/{session_id}: expected 30 Hz, found {fs}')
```
```python
            assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
            assert y.min() >= 0 and y.max() <= 4
```

iii. Step 5 Key Decision 5 (quoted above) and Step 10 check 7: "Sessions short by 1, 2, 3, 116, or 148 motion frames drop only the incomplete terminal minute." Step 9: "No data were silently lost: only incomplete terminal 60-second intervals are excluded, as documented per session." The AI's rationale for refusing interpolation is that inventing behavioural values is less defensible than dropping them — sound in principle, but applied under a mistaken belief about *where* the frames are missing.

## 6-a. What are the most time-consuming steps of the code?

i. The maximin baseline estimation in `baseline_correct` — a Gaussian filter plus a rolling minimum and a rolling maximum over a 1,800-frame window on the full n_neurons × n_frames matrix — dominates; it is the only step that touches every one of the 1.96 × 10⁹ neuron-frame samples several times. Measured cost is 0.29 s/session for the 221-neuron 20-min sessions and ~0.81–1.1 s for the 746-neuron 30-min sessions, with the whole 41-session conversion (including pickling ~800 MB of float32 trials) finishing in 36.1 s. Disk I/O of `F.npy`/`Fneu.npy` is second; motion processing, discretisation and trial slicing are negligible.

ii.
```python
    if baseline_mode == 'maximin':
        Flow = gaussian_filter(Fc, [0.0, sig])
        Flow = minimum_filter1d(Flow, win, axis=1)
        Flow = maximum_filter1d(Flow, win, axis=1)
```
```python
    dt = time.perf_counter() - t0
    print(f'Processed {subject}/{session_id}: {F.shape[0]} neurons, {n_trials} trials, '
          f'class counts {counts.tolist()}, {dt:.2f}s', flush=True)
    ...
    print(f'Saved {args.outpicklefile}: {len(sessions)} sessions, '
          f'{sum(len(x) for x in neural)} trials in {wall:.2f}s; '
          f'mean processing {np.mean(times):.2f}s/session', flush=True)
```

iii. Step 6: "Suite2p maximin baseline filtering necessarily touches each full fluorescence matrix and is the primary compute/memory cost. Storing all final trial arrays is required by the target pickle format." Step 7 extrapolated the 2-session sample (1.13 s/session) to "~47 s for 41 sessions", concluding "Full conversion is safely below the 15-minute optimization threshold"; the actual full run took 36.4 s, within the estimate.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI reports that no significant loop remains: filtering and binning are fully vectorised over neurons and time (`reshape(...).mean(axis=2)` rather than per-neuron or per-bin loops), and the only Python loops are over sessions and over trials. The per-trial loop (`for trial in range(n_trials)`) could in principle be replaced by a single reshape/`np.split`, but it executes only 1,081 times in total and merely copies contiguous slices, so the gain would be immaterial. The per-session loop is the one place where real wall-clock could be recovered — the 41 sessions are completely independent and could be processed with `multiprocessing`; the AI chose sequential execution deliberately to bound peak memory. Note there is no dropped-frame interpolation loop in this code (the reference's `np.insert`-per-drop loop, which is the reference's own vectorisation candidate) because the AI does not interpolate at all.

ii.
```python
def mean_bins_2d(x, n_frames):
    x = np.asarray(x[:, :n_frames])
    return x.reshape(x.shape[0], n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=2, dtype=np.float32)
```
```python
    for trial in range(n_trials):
        a = trial * PROCESSED_POINTS_PER_TRIAL
        b = a + PROCESSED_POINTS_PER_TRIAL
        neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
```
```python
    for i, (subject, sid, path) in enumerate(sessions):
        show = args.show_processing and i < 2
        n, x, y, info, dt = process_session(subject, sid, path, show)
```

iii. Step 6 "Code speedups added": "Source fluorescence arrays are initially memory-mapped. Filtering and temporal binning are vectorized over neurons/time. Data are cast to float32/int64 once and sliced into contiguous trial arrays. Sessions are processed sequentially so temporary full-session matrices are released before the next session." Step 7 concluded the runtime was far under the 15-minute threshold, so no further optimisation was pursued.

## 6-c. What processing does the code repeat multiple times?

i. Essentially nothing is recomputed: baseline correction, binning and quantile estimation each run exactly once per session, and each raw file is opened once. Three minor repetitions exist: `validate_converted` walks every trial array a second time after they have already been built and shape-checked implicitly; `np.ascontiguousarray(..., dtype=np.float32)` re-casts arrays that `mean_bins_2d` already produced as float32 (a redundant copy of every trial); and per-session bookkeeping (`np.bincount`, the `info` dict) duplicates statistics that are also printed to the log. None is measurable against the baseline filter.

ii.
```python
        neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
```
```python
def validate_converted(data):
    ...
    for s in range(ns):
        for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
            assert n.shape[1] == x.shape[1] == y.shape[1] == PROCESSED_POINTS_PER_TRIAL
            assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
```

iii. The AI does not flag any repeated processing in CONVERSION_NOTES.md; Step 6 only claims "Data are cast to float32/int64 once and sliced into contiguous trial arrays." The revalidation pass is a deliberate correctness cost, consistent with the instruction to "Include sanity checks (e.g., trial counts match across arrays)" and "Validate data shapes and types at each step".

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four small items. (1) `baseline_correct` always returns the intermediate `Fc` and the full baseline `Flow` in addition to the corrected trace, and `process_session` always binds them — two extra full-session float32 arrays (≈161 MB each for a 746 × 54,000 session) that are used only when `--show-processing` is set, tripling peak memory in the default path. (2) Baseline correction is computed over the *entire* recording, including the trailing frames beyond `used_frames` that are then thrown away (up to 2,116 frames in jm031/2023-10-22) — though this is the correct choice, since restricting the rolling window to the retained prefix would change the baseline everywhere. (3) `iscell.npy` is loaded on every session purely to assert a property, and is not otherwise used. (4) A fairly large `session_info` list (16 fields × 41 sessions, including full quintile-edge and class-count lists) is pickled into the metadata and ignored by the decoder. None of these affects the converted values.

ii.
```python
    return (Fc - Flow).astype(np.float32, copy=False), Fc, Flow
```
```python
    # Baseline estimation uses the complete neural recording, matching Suite2p processing.
    neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
```
```python
    iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
    ...
    if F.shape[0] != iscell.shape[0] or not np.all(iscell[:, 0] == 1):
        raise ValueError(f'{subject}/{session_id}: supplied matched-cell curation is inconsistent')
```
```python
        'motion_quintile_edges': edges.tolist(), 'output_class_counts': counts.tolist(),
        ...
            'session_info': infos,
```

iii. The AI does not identify any of these as waste. Its stated rationale for the items is elsewhere in the notes: Step 6 — "Baseline estimation uses each complete neural session, matching Suite2p preprocessing; behavior availability affects only the retained shared trial interval"; Step 4 — the `iscell` read exists because "Data are already cell-filtered and all-day matched ... the converter verifies rather than reapplies filtering"; Step 5 mapping table — `session_info` is kept because it "Enables auditing all curation". The `Fc`/`Flow` return values exist to feed `make_processing_plot` under `--show-processing`.
