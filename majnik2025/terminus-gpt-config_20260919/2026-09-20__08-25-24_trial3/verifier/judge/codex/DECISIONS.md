# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories, then scans every session subdirectory under each subject. For each session it loads Suite2p calcium files `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`, plus behavior `motion_energy_glob.npy`. Trials are not loaded directly from disk because the source data are continuous; trials are constructed later in memory from the loaded session arrays.

ii. 
```python
def discover_sessions(root=DATA_ROOT):
    out = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            out.append((subject_dir.name, session_dir.name, session_dir))
    return out

...

F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r')
```

iii. In `CONVERSION_NOTES.md`, the AI says the data tree is `/app/data/<mouse>/<session>/`, that each session contains one `suite2p/plane0` folder and one `move_deve` folder, and that the source is a preprocessed Track2p export. It therefore chose directory walking plus direct `.npy` loading.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the top-level directories under `/app/data`. The subject ID saved in the output is the directory name, and the final `subjects` list is the sorted unique set of those names.

ii.
```python
for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    ...
    out.append((subject_dir.name, session_dir.name, session_dir))

...

subjects = sorted({x[0] for x in sessions})
subject_lookup = {x: i for i, x in enumerate(subjects)}
...
'subject_idx': np.asarray([subject_lookup[x[0]] for x in sessions], dtype=np.int64),
```

iii. The notes state that the hierarchy is organized by mouse folders and list six mice (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), so the AI treated each top-level folder as one mouse.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject directory. Sessions are sorted lexicographically within each subject and then processed one by one.

ii.
```python
for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        out.append((subject_dir.name, session_dir.name, session_dir))
```

iii. In the notes, the AI identifies the hierarchy `/app/data/<mouse>/<YYYY-MM-DD_a>/` and treats each dated folder as a daily recording session.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous and constructs artificial trials as non-overlapping 60-second chunks. It first finds the shared neural/behavior prefix length, converts that to a number of complete 60-second windows at 30 Hz (`1800` raw frames), keeps only whole windows, then after 10-frame averaging slices each session into `180` binned samples per trial.

ii.
```python
RAW_FRAMES_PER_TRIAL = 1800  # 60 s * 30 Hz
TEMPORAL_AVG = 10
PROCESSED_POINTS_PER_TRIAL = RAW_FRAMES_PER_TRIAL // TEMPORAL_AVG

shared_frames = min(F.shape[1], motion.shape[0])
n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL

for trial in range(n_trials):
    a = trial * PROCESSED_POINTS_PER_TRIAL
    b = a + PROCESSED_POINTS_PER_TRIAL
    neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
    input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
    output_trials.append(np.ascontiguousarray(labels[None, a:b], dtype=np.int64))
```

iii. The notes explicitly say the source dataset has no native trials and that the decoder instruction overrides the paper’s continuous/block analysis, so the AI chose non-overlapping 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a trial-quality metric such as motion/artifact rejection. Its effective filtering is structural: it requires at least two complete 60-second trials per session, uses only the shared neural/behavior prefix, and drops incomplete trailing data that do not fill a full trial.

ii.
```python
n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
if n_trials < 2:
    raise ValueError(f'{subject}/{session_id}: fewer than two complete 60-s trials')
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
```

iii. In the notes, the AI frames this as satisfying decoder-format constraints rather than biological quality control. It also argues that only complete trials from synchronized data should be retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The saved neural signal is derived primarily from `F.npy` and `Fneu.npy`, with preprocessing parameters read from `ops.npy`. `iscell.npy` is used only as a consistency check, not as the numerical source of the neural trace.

ii.
```python
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
...
neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI maps `F.npy`, `Fneu.npy`, and `ops.npy` to the output `neural` field and says these are the variables needed to reconstruct the paper’s analysis signal.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil correction and then a manual reimplementation of Suite2p-style maximin baseline subtraction. Specifically, it computes `Fc = F - neucoeff * Fneu`, smooths in time with a Gaussian, applies a rolling minimum then rolling maximum over a 60-second window, subtracts that baseline, and finally averages non-overlapping 10-frame bins.

ii.
```python
def baseline_correct(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    fs = float(ops['fs'])
    sig = float(ops.get('sig_baseline', 10.0))
    win = int(float(ops.get('win_baseline', 60.0)) * fs)
    Fc = np.asarray(F, dtype=np.float32) - np.float32(neucoeff) * np.asarray(Fneu, dtype=np.float32)
    if baseline_mode == 'maximin':
        Flow = gaussian_filter(Fc, [0.0, sig])
        Flow = minimum_filter1d(Flow, win, axis=1)
        Flow = maximum_filter1d(Flow, win, axis=1)
    ...
    return (Fc - Flow).astype(np.float32, copy=False), Fc, Flow

...

neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
neural_binned = mean_bins_2d(neural_bc, used_frames)
```

iii. The notes say the paper uses “baseline corrected fluorescence traces” with default Suite2p parameters and explicitly reject dividing by baseline. The AI therefore chose neuropil correction plus baseline subtraction, then the paper’s 10-frame averaging used for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not remove any neurons beyond checking that the supplied export is already fully curated. It verifies that `iscell[:, 0] == 1` for all rows and that the neuron count matches `F`, then keeps every row.

ii.
```python
iscell = np.load(plane / 'iscell.npy', mmap_mode='r')
...
if F.shape[0] != iscell.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(f'{subject}/{session_id}: supplied matched-cell curation is inconsistent')
```

iii. In Step 4 and Step 5 of the notes, the AI argues that the provided arrays are already Track2p-matched and Suite2p-cell-filtered across all days, so a second neuron filter would be wrong.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats the start of each constructed 60-second trial as the alignment event. The neural array for each trial begins at that trial boundary, and the metadata record the event as the start of each non-overlapping 60-second trial.

ii.
```python
for trial in range(n_trials):
    a = trial * PROCESSED_POINTS_PER_TRIAL
    b = a + PROCESSED_POINTS_PER_TRIAL
    neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))

...

'metadata': {
    ...
    'temporal_alignment_event': 'start of each non-overlapping 60-second trial; source video was hardware-triggered by microscope',
    'off_start': 0.0,
    'off_end': 60.0,
}
```

iii. The notes say the recordings are continuous and that the task-required trialization is a deliberate deviation from the paper. The AI therefore used the synthetic trial boundary as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral time series by averaging non-overlapping windows of 10 raw frames at 30 Hz. This yields one sample every `10 / 30 = 0.333...` s, i.e. a `333.333 ms` bin size.

ii.
```python
TEMPORAL_AVG = 10

def mean_bins_2d(x, n_frames):
    x = np.asarray(x[:, :n_frames])
    return x.reshape(x.shape[0], n_frames // TEMPORAL_AVG, TEMPORAL_AVG).mean(axis=2, dtype=np.float32)

...

'time_bin_size': 1000.0 * TEMPORAL_AVG / 30.0,
```

iii. The notes repeatedly cite the paper’s decoding analysis as averaging both dF/F and behavior in 10-frame bins, so the AI chose to match that preprocessing.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not use a saved timestamp vector. It derives elapsed time from the sample index after temporal binning and from the imaging frame rate in `ops['fs']`.

ii.
```python
fs = float(ops['fs'])
...
elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
            + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)
```

iii. In Step 5 of the notes, the AI says the timestamps are only a diagnostic and that nominal 30 Hz plus frame index is the cleaner way to generate fixed-bin elapsed time for the decoder.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each 10-frame averaged bin, not the left edge. The AI forms a regularly spaced float32 vector in seconds, then slices that vector into the same 60-second trial windows as the neural data.

ii.
```python
elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
            + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)

...

input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
```

iii. The notes justify this by saying that each averaged sample represents a 10-frame window, so the bin center is the most faithful scalar time tag.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time to neural data by computing the elapsed vector at the same 10-frame binning used for neural and behavior, then slicing it with the same trial boundaries. The time values remain absolute within the session rather than resetting to zero each trial.

ii.
```python
neural_binned = mean_bins_2d(neural_bc, used_frames)
motion_binned = mean_bins_1d(motion, used_frames)
elapsed = ((np.arange(len(motion_binned), dtype=np.float64) * TEMPORAL_AVG
            + (TEMPORAL_AVG - 1) / 2) / fs).astype(np.float32)

for trial in range(n_trials):
    a = trial * PROCESSED_POINTS_PER_TRIAL
    b = a + PROCESSED_POINTS_PER_TRIAL
    neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
    input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
```

iii. The notes explicitly say each trial should retain absolute session time and that the same 10-frame bins and trial cuts should be used for all streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from the per-session `motion_energy_glob.npy` vector. The AI does not use `tstamps.npy` or `interframe_int.npy` in the final conversion logic.

ii.
```python
move = path / 'move_deve'
motion = np.load(move / 'motion_energy_glob.npy', mmap_mode='r')
...
motion_binned = mean_bins_1d(motion, used_frames)
```

iii. The notes say motion energy has already been computed in the supplied data exactly as described in the paper, so the AI treats the global motion-energy vector itself as the raw source signal.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI keeps only the shared neural/behavior prefix, averages the motion-energy vector into non-overlapping 10-frame bins, and then discretizes the binned values into five classes using per-session quintile thresholds. It also computes those thresholds only on the retained complete-trial interval.

ii.
```python
shared_frames = min(F.shape[1], motion.shape[0])
n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL

motion_binned = mean_bins_1d(motion, used_frames)

edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. In the notes, the AI argues that it should “never interpolate or fabricate missing terminal behavior,” so it trims to the shared valid interval before binning and discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses session-specific quintiles. It takes the 20th, 40th, 60th, and 80th percentiles of the binned motion-energy trace and assigns labels `0` through `4` with `searchsorted(..., side='right')`.

ii.
```python
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
...
'output_values': [['quintile 1 (lowest)', 'quintile 2', 'quintile 3',
                   'quintile 4', 'quintile 5 (highest)']],
```

iii. The notes cite the decoder requirement “five equal-percentile bins, selected per session” and say the thresholds should be computed after 10-frame averaging so that categorization is applied to the actual saved signal.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes the two streams are already synchronized frame-by-frame at 30 Hz and aligns them by truncating both to the shared prefix length `min(F.shape[1], motion.shape[0])`. It does not attempt interpolation or dropped-frame correction.

ii.
```python
shared_frames = min(F.shape[1], motion.shape[0])
n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL

neural_binned = mean_bins_2d(neural_bc, used_frames)
motion_binned = mean_bins_1d(motion, used_frames)
```

iii. The notes argue that camera acquisition was hardware-triggered by the microscope and that observed motion deficits should be handled by trimming the shared valid prefix rather than inventing values for missing frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles inconsistencies conservatively. It raises errors for mismatched `F`/`Fneu`, inconsistent `iscell`, unexpected frame rate, or too few complete trials. For neural/behavior length mismatches, it drops data beyond the shared prefix rather than padding or interpolating. It also discards any trailing frames that do not complete a full 60-second trial.

ii.
```python
if F.shape != Fneu.shape:
    raise ValueError(f'{subject}/{session_id}: F and Fneu shapes differ')
if F.shape[0] != iscell.shape[0] or not np.all(iscell[:, 0] == 1):
    raise ValueError(f'{subject}/{session_id}: supplied matched-cell curation is inconsistent')
if not np.isclose(fs, 30.0):
    raise ValueError(f'{subject}/{session_id}: expected 30 Hz, found {fs}')

shared_frames = min(F.shape[1], motion.shape[0])
n_trials = shared_frames // RAW_FRAMES_PER_TRIAL
if n_trials < 2:
    raise ValueError(f'{subject}/{session_id}: fewer than two complete 60-s trials')
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
```

iii. In the notes, the AI repeatedly says it should avoid fabricating values and should only keep the synchronized continuous interval supported by the files.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies the maximin baseline filtering on full-session fluorescence matrices as the main compute and memory cost. Secondarily, full-array loading and storing all trial arrays are unavoidable overheads.

ii.
```python
neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
neural_binned = mean_bins_2d(neural_bc, used_frames)
...
times.append(dt)
...
print(f'Saved {args.outpicklefile}: {len(sessions)} sessions, '
      f'{sum(len(x) for x in neural)} trials in {wall:.2f}s; '
      f'mean processing {np.mean(times):.2f}s/session', flush=True)
```

iii. Step 6 of `CONVERSION_NOTES.md` explicitly says “Suite2p maximin baseline filtering necessarily touches each full fluorescence matrix and is the primary compute/memory cost.”

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the heavy numerical operations over neurons and time. The remaining explicit loops are mostly unavoidable orchestration loops over sessions and trial packaging loops that populate the required nested Python-list format.

ii.
```python
for i, (subject, sid, path) in enumerate(sessions):
    ...
    n, x, y, info, dt = process_session(subject, sid, path, show)

...

for trial in range(n_trials):
    a = trial * PROCESSED_POINTS_PER_TRIAL
    b = a + PROCESSED_POINTS_PER_TRIAL
    neural_trials.append(np.ascontiguousarray(neural_binned[:, a:b], dtype=np.float32))
    input_trials.append(np.ascontiguousarray(elapsed[None, a:b], dtype=np.float32))
    output_trials.append(np.ascontiguousarray(labels[None, a:b], dtype=np.int64))
```

iii. The notes say “Filtering and temporal binning are vectorized over neurons/time,” so the AI viewed the remaining loops as mostly data-structure assembly rather than missed vectorization opportunities.

## 6-c. What processing does the code repeat multiple times?

i. The AI does not repeat a major numerical transform unnecessarily across the whole dataset. Each session is processed once. The closest repeated work is that the same per-trial slicing pattern is applied independently to neural, input, and output arrays, and optional plotting uses intermediate arrays already computed for conversion.

ii.
```python
for trial in range(n_trials):
    ...
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)

if show_processing:
    make_processing_plot(...)
```

iii. In the notes, the AI emphasizes sequential one-pass session processing and vectorized transforms, which suggests it believed duplicate processing had largely been avoided.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main extra work is computing and returning intermediate arrays (`Fc`, `Flow`) that are only needed for optional audit plots, plus generating per-session provenance counts and quantile metadata not used by the decoder itself. When `--show-processing` is enabled, plotting is entirely diagnostic and not used downstream.

ii.
```python
return (Fc - Flow).astype(np.float32, copy=False), Fc, Flow

...

info = {
    ...
    'motion_quintile_edges': edges.tolist(), 'output_class_counts': counts.tolist(),
    ...
}
if show_processing:
    make_processing_plot(Path('/app') / f'processing_{safe}.png', ...)
```

iii. The notes describe these as processing-audit and provenance features added for validation rather than for the final decoder inputs or outputs.
