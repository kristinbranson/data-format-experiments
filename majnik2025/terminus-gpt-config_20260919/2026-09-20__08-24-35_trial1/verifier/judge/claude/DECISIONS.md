# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing for `suite2p/plane0` directories under the data root. For each session, it loads `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy` from the suite2p directory, plus `motion_energy_glob.npy` and `tstamps.npy` from the `move_deve` subdirectory. All six subject folders (`jm031`–`jm046`) are included.

ii.
```python
def discover_sessions() -> list[Path]:
    return sorted(DATA_ROOT.glob('*/*/suite2p/plane0'))

# In process_session:
F = np.load(sp / 'F.npy', mmap_mode='r')
Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
iscell = np.load(sp / 'iscell.npy')
ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
```

iii. The AI documented this approach in CONVERSION_NOTES.md Steps 1–2, noting the standard suite2p directory structure with `F.npy` and `Fneu.npy` for calcium imaging and `motion_energy_glob.npy` for behavior. Memory-mapped loading was chosen for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are identified as the unique parent-of-parent directory names from the discovered session paths (i.e. the `jm*` directories). They are sorted alphabetically, and each session is mapped to its subject index.

ii.
```python
subjects = sorted({p.parent.parent.parent.name for p in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
subject_idx.append(subject_to_idx[info['subject']])
```

iii. The AI noted in Step 2 that six subject directories (`jm031`–`jm046`) each contain dated session subdirectories. Sorted unique directory names provide deterministic subject ordering.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a dated subdirectory within a subject folder, discovered via `suite2p/plane0` paths. Sessions are sorted lexicographically, which is chronological within each subject.

ii.
```python
def discover_sessions() -> list[Path]:
    return sorted(DATA_ROOT.glob('*/*/suite2p/plane0'))
```

iii. The AI documented in Step 5 that sessions are ordered by subject ID then ISO-format date, giving chronological order.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive non-overlapping 60-second windows of the continuous recording. After 10-frame averaging (30 Hz → 3 Hz), each trial contains exactly 180 time bins. The AI truncates to the minimum common length across neural and behavioral streams, then retains only complete 1800-frame (60s) blocks at native resolution before binning.

ii.
```python
NATIVE_FRAMES_PER_TRIAL = 1800  # 60 s * 30 Hz
N_BINS_PER_TRIAL = NATIVE_FRAMES_PER_TRIAL // AVERAGE_FRAMES  # 180

common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL

n_trials = keep_native // NATIVE_FRAMES_PER_TRIAL
neural_trials = [... for x in neural_binned.reshape(..., n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
```

iii. The instructions specify "split sessions into 60-second trials." The AI's CONVERSION_NOTES Step 5 documents this as yielding 1,081 total trials (9 fewer than the nominal 1,090 due to endpoint truncation in sessions with short behavior streams).

## 1-e. How are trials filtered based on quality controls?

i. No explicit quality-based trial filtering is applied. However, the truncation to common stream length (taking the minimum of neural and behavioral lengths) implicitly discards one trial at the end of 9 sessions where the behavior stream is shorter than the neural stream.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
    raise ValueError(f'{sid}: fewer than two complete 60-s trials')
```

iii. The AI noted in CONVERSION_NOTES Step 4 that 8 of 41 behavior streams are shorter than neural data by 1–148 samples. Rather than interpolating the missing frames, the AI chose to truncate to the common length, which loses 9 complete endpoint trials across the dataset (1,081 vs 1,090 trials).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw ROI fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The `ops.npy` file provides preprocessing parameters, and `iscell.npy` provides the cell classification probabilities for neuron filtering.

ii.
```python
F = np.load(sp / 'F.npy', mmap_mode='r')
Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
iscell = np.load(sp / 'iscell.npy')
ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
```

iii. These are standard suite2p output files. The paper states baseline-corrected fluorescence traces were used as dF/F, not deconvolved spikes.

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) neuropil subtraction (`F - neucoeff * Fneu` where neucoeff is read from ops, typically 0.7), (2) suite2p-style baseline correction reimplemented locally using Gaussian smoothing, minimum filtering, and maximum filtering (maximin method with 60s window), (3) 10-frame non-overlapping averaging to downsample from 30 Hz to 3 Hz. The baseline correction is applied to the full recording before truncation.

ii.
```python
neucoeff = float(ops.get('neucoeff', 0.7))
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
corrected_full = suite2p_preprocess(raw_corr_full, ops)
corrected = corrected_full[:, :keep_native]
neural_binned = average_blocks_2d(corrected)

# suite2p_preprocess reimplements dcnv.preprocess:
def suite2p_preprocess(F, ops):
    flow = gaussian_filter(x, sigma=(0.0, sig), mode='reflect')
    flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
    flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
    x -= flow
    return x
```

iii. The AI documented in CONVERSION_NOTES Step 3 that the paper uses "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The local reimplementation was verified against suite2p's dcnv.preprocess with `np.allclose(rtol=1e-6, atol=1e-6)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `iscell[:, 1] > 0.5` (the suite2p cell classification probability threshold). However, all neurons in the distributed dataset pass this threshold since the data was already curated by Track2p.

ii.
```python
cell_mask = iscell[:, 1] > 0.5
if not np.all(cell_mask):
    print(f'  {sid}: filtering {np.sum(~cell_mask)} ROIs below iscell threshold')
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
```

iii. The AI documented in Steps 2 and 4 that all 20,445 ROI entries pass `iscell > 0.5`, confirming the data is already Track2p-curated. The filter is applied as a safety check matching the paper's stated criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each 60-second window within the continuous recording. There is no stimulus-based event alignment since the experiment involves spontaneous behavior. The first trial starts at the beginning of the session.

ii.
```python
'temporal_alignment_event': 'Start of each consecutive non-overlapping 60-second window in a continuous session',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The AI noted in CONVERSION_NOTES Step 4 that recordings are continuous with no stimulus events, so trial alignment is to the session start with consecutive 60-second windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged in non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms per bin). This yields 180 bins per 60-second trial.

ii.
```python
AVERAGE_FRAMES = 10
N_BINS_PER_TRIAL = NATIVE_FRAMES_PER_TRIAL // AVERAGE_FRAMES  # 180

def average_blocks_2d(x, block=AVERAGE_FRAMES):
    return x.reshape(x.shape[0], -1, block).mean(axis=2, dtype=np.float32)

'time_bin_size': 1000.0 * AVERAGE_FRAMES / 30.0,  # 333.33 ms
```

iii. The AI documented in Step 3 that the paper's decoding methods "averaged in bins of 10 consecutive timestamps" for both neural and behavioral signals. This is implemented identically.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index, the number of averaged frames (10), and the sampling rate (30 Hz). The time represents the center of each averaged bin, giving seconds elapsed from session start.

ii.
```python
elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
            + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
```

iii. The AI documented in Step 5 that "absolute elapsed session time in seconds at each 10-frame averaged sample: mean native sample index / 30." The first value is 0.15 s (center of the first 10-frame bin spanning frames 0–9).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A simple arithmetic computation: for each bin index `i`, time = `(i * 10 + 4.5) / 30.0`. This uses the center of each 10-frame bin. Time continues across trial boundaries within a session (does not reset per trial).

ii.
```python
elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
            + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. Documented in Step 5 as a direct computation from synchronized frame ordinal and known acquisition rate.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is inherently aligned with the neural data because both are derived from the same frame indices. Each time bin corresponds exactly to the same 10-frame averaging window used for neural activity.

ii. N/A (alignment is implicit through shared indexing)

iii. The AI notes in Step 5 that all streams share the same frame-derived time axis.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. The `tstamps.npy` file is also loaded to determine the common synchronized length with the neural data.

ii.
```python
motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
```

iii. The paper describes motion energy as "pixel-wise frame difference, square each pixel difference, then sum across pixels" — this is the pre-computed `motion_energy_glob.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) truncation to common observed length across neural and behavioral streams (no interpolation of missing frames), (2) 10-frame non-overlapping averaging, (3) discretization into 5 percentile-based bins computed within each session using quantiles at [0.2, 0.4, 0.6, 0.8] and `np.searchsorted`.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
motion_binned = average_blocks_1d(motion_native)

edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. The AI documented in Step 5 that quintile edges are computed separately per session after 10-frame averaging and complete-trial restriction, matching the "selected per session" instruction. The truncation approach was chosen to avoid extrapolating missing behavior.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After 10-frame averaging, the binned motion energy for each session is split into 5 equal-frequency bins using quantile edges at the 20th, 40th, 60th, and 80th percentiles. `np.searchsorted` with `side='right'` assigns each value to a class (0–4). The AI verifies that all four edges are distinct.

ii.
```python
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
if np.unique(edges).size != 4:
    raise ValueError(f'{sid}: non-distinct quintile edges {edges}')
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
assert labels.min() == 0 and labels.max() == 4
```

iii. The instructions specify "five equal-percentile bins, selected per session." The AI implements this with session-local quantiles and reports approximately 20% per class across the full dataset.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns neural and behavioral data by taking the minimum common length across F, Fneu, motion energy, and timestamps, then truncating all streams to the nearest complete trial boundary. No interpolation of dropped video frames is performed.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
# ...
corrected = corrected_full[:, :keep_native]
motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
```

iii. The AI documented in Step 4 that "Camera is microscope-triggered; both modalities at 30 Hz" enabling alignment by sample ordinal. Rather than interpolating dropped frames (as the reference code does with `interframe_int.npy`), the AI truncates to the shorter stream, reasoning that "never extrapolate missing endpoint behavior" is safer.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing behavior data (dropped video frames making motion energy shorter than neural data) by truncating all streams to the minimum common length, then further truncating to complete 60-second trial boundaries. This loses 9 trials across the dataset. The AI verifies that at least 2 complete trials remain per session.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
    raise ValueError(f'{sid}: fewer than two complete 60-s trials')
dropped_neural = F.shape[1] - keep_native
dropped_motion = motion.size - keep_native
```

iii. The AI documented in CONVERSION_NOTES Steps 2 and 4 that 8 of 41 behavior streams are shorter by 1–148 samples. The truncation approach avoids fabricating data but loses one complete trial per affected session.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the `suite2p_preprocess` baseline correction, which applies Gaussian smoothing, minimum filtering, and maximum filtering across the full recording length for every neuron. The full conversion runs in ~37 seconds for 41 sessions.

ii. N/A (timing is measured but the baseline correction dominates)

iii. The AI documented timing in Step 7, estimating ~1 second per session, with the scipy-based baseline correction being the dominant cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is already well-vectorized. The main session-level for-loop cannot be vectorized (sessions have different neuron counts). Trial splitting uses reshape/transpose rather than per-trial loops.

ii.
```python
# Vectorized trial splitting:
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                 for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
```

iii. The AI noted that per-session processing is bounded by memory rather than loop overhead. No obvious vectorization opportunities were identified.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once in a single pass. The baseline correction is applied to the full recording once, and the truncated prefix is then used for trial assembly.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies baseline correction to the full neural recording (including frames beyond the common truncation point), then discards the endpoint frames. This is intentional to avoid boundary artifacts in the baseline estimation, but the correction of discarded frames is technically unnecessary computation.

ii.
```python
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
corrected_full = suite2p_preprocess(raw_corr_full, ops)
corrected = corrected_full[:, :keep_native]  # truncate after correction
```

iii. The AI documented in Step 10 that this was a deliberate fix: initial code only baseline-corrected the truncated prefix, which introduced boundary artifacts. Correcting the full recording first is the correct approach, matching suite2p's behavior.
