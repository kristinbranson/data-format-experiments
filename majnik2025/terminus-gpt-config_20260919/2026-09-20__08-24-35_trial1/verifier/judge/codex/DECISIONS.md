# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every imaging session by globbing for `*/*/suite2p/plane0`, sorts those paths, and processes each one. Per session it memory-maps `F.npy`, `Fneu.npy`, motion energy, and timestamps, and loads `iscell.npy` and `ops.npy` normally. Full mode uses all discovered sessions; sample mode keeps the first two.

ii.
```python
def discover_sessions() -> list[Path]:
    return sorted(DATA_ROOT.glob('*/*/suite2p/plane0'))

F = np.load(sp / 'F.npy', mmap_mode='r')
Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
iscell = np.load(sp / 'iscell.npy')
ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
```

iii. The notes say the native hierarchy contains six subject directories and 41 daily session directories. Sorted globbing gives deterministic subject/date order; memory mapping and one-session-at-a-time processing bound memory use.

## 1-b. How are the data split into subjects?

i. The subject is the parent directory of each session. Unique subject names are sorted, mapped to integer indices, and each session receives the index for its subject.

ii.
```python
subject = session_dir.parent.name
subjects = sorted({p.parent.parent.parent.name for p in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
subject_idx.append(subject_to_idx[info['subject']])
```

iii. The agent found six mouse folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and treated directory names as mouse IDs, consistent with the dataset hierarchy.

## 1-c. How are the data split into sessions?

i. Every sorted `suite2p/plane0` path defines one session; its session directory is two parents above that path. Sessions remain separate entries in all target lists.

ii.
```python
sessions = discover_sessions()
for i, sp in enumerate(sessions):
    n, x, y, r, info = process_session(sp, show)
    neural.append(n); inputs.append(x); outputs.append(y)
```

iii. The notes identify each dated subdirectory as one daily recording and state that lexicographic sorting is chronological because session names use ISO dates.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping 60-second trials. At 30 Hz this is 1,800 native frames, or 180 samples after 10-frame averaging. Only complete trials within the common stream length are retained.

ii.
```python
NATIVE_FRAMES_PER_TRIAL = 1800
N_BINS_PER_TRIAL = NATIVE_FRAMES_PER_TRIAL // AVERAGE_FRAMES
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
n_trials = keep_native // NATIVE_FRAMES_PER_TRIAL
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
    for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
```

iii. The experiment is continuous and has no natural trials; the requested decoder task explicitly requires 60-second trials. Complete-window truncation was chosen to avoid fabricating unavailable behavior.

## 1-e. How are trials filtered based on quality controls?

i. A session is rejected if it has fewer than two complete synchronized 60-second trials. Otherwise all complete trials are retained; the incomplete endpoint is discarded. Per-trial shape and finite-value assertions are also enforced.

ii.
```python
if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
    raise ValueError(f'{sid}: fewer than two complete 60-s trials')
for ni, ii, oo in zip(neural_trials, input_trials, output_trials):
    assert ii.shape == oo.shape == (1, N_BINS_PER_TRIAL)
    assert np.isfinite(ni).all() and np.isfinite(ii).all() and np.isfinite(oo).all()
```

iii. The target format requires at least two trials per session. The notes report that every session retained at least 19 trials, so no complete trial was removed by an additional quality metric.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p `F.npy` and `Fneu.npy`, with the neuropil coefficient and baseline parameters read from `ops.npy`; `iscell.npy` supplies the ROI mask.

ii.
```python
F = np.load(sp / 'F.npy', mmap_mode='r')
Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
iscell = np.load(sp / 'iscell.npy')
ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
neucoeff = float(ops.get('neucoeff', 0.7))
```

iii. The agent concluded from the methods that the decoded neural representation is neuropil- and baseline-corrected fluorescence, not `spks.npy`.

## 2-b. How is the `neural` data processed?

i. Selected fluorescence is converted to float32, neuropil corrected as `F - neucoeff*Fneu`, and baseline corrected over the full recording using a local reproduction of Suite2p preprocessing. The retained prefix is then averaged in non-overlapping blocks of 10 frames and reshaped into trials.

ii.
```python
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
corrected_full = suite2p_preprocess(raw_corr_full, ops)
corrected = corrected_full[:, :keep_native]
neural_binned = average_blocks_2d(corrected)
```

iii. The notes say the local SciPy implementation reproduced `suite2p.extraction.dcnv.preprocess` on real data with `np.allclose`, avoids a slow Suite2p import, and preserves the important ordering of full-recording baseline estimation before endpoint restriction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs with `iscell[:,1] <= 0.5` are removed. Shape consistency is checked, and final neural samples must be finite. In the supplied data all ROIs pass, so this removes none in practice.

ii.
```python
cell_mask = iscell[:, 1] > 0.5
if len(cell_mask) != F.shape[0]:
    raise ValueError(f'{sid}: iscell/F neuron mismatch')
raw_corr_full = np.asarray(F[cell_mask, :], dtype=np.float32) - ...
assert np.isfinite(ni).all()
```

iii. The notes cite Track2p's 0.5 cell-probability criterion but also conclude that the distributed arrays are already curated and reindexed, making this an enforcement/verification step rather than additional effective filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Neural bins are aligned by frame ordinal to the start of each consecutive 60-second window in the continuous session; each trial covers offsets 0 to 60 seconds.

ii.
```python
'temporal_alignment_event': 'Start of each consecutive non-overlapping 60-second window in a continuous session',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The notes explain that trial boundaries are artificial because the recordings are continuous. Hardware-synchronized sample ordinal is used rather than an event marker.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz streams are averaged in non-overlapping groups of 10, yielding 3 Hz data with 333.333 ms bins and 180 bins per trial.

ii.
```python
AVERAGE_FRAMES = 10
return x.reshape(x.shape[0], -1, block).mean(axis=2, dtype=np.float32)
'time_bin_size': 1000.0 * AVERAGE_FRAMES / 30.0,
```

iii. This matches the paper's decoding preprocessing, which averages neural and behavioral traces over 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the retained averaged-bin ordinal, the fixed block size of 10 frames, and `ops['fs']`; timestamp values are loaded for length/alignment checks but are not used to calculate time.

ii.
```python
fs = float(ops['fs'])
elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
            + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
```

iii. The agent found timestamp magnitudes had unclear units, whereas acquisition was documented as microscope-triggered at 30 Hz. It therefore considered frame ordinal divided by 30 the reliable elapsed-time basis.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each 10-frame average, the agent uses the center frame `(10k + 4.5)/fs`, converts to float32 seconds, and reshapes the continuous vector into trial arrays without resetting at trial boundaries.

ii.
```python
elapsed = ((np.arange(...) * AVERAGE_FRAMES + 4.5) / fs).astype(np.float32)
input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. The notes justify the 0.15-second first value as the mean timestamp of native samples 0–9 and preserve absolute within-session time because the requested variable is elapsed time from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One elapsed-time center is generated for every neural 10-frame mean from the same ordinal block, then both arrays are reshaped with identical 180-bin trial boundaries.

ii.
```python
neural_binned = average_blocks_2d(corrected)
elapsed = ((np.arange(keep_native // AVERAGE_FRAMES) * AVERAGE_FRAMES + 4.5) / fs)
input_trials = [... for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. Shared ordinal construction guarantees one-to-one alignment; time intentionally continues between consecutive trials.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from `move_deve/motion_energy_glob.npy`. `tstamps.npy` is also loaded and its length participates in choosing the synchronized common endpoint.

ii.
```python
motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
```

iii. The notes identify `motion_energy_glob.npy` as the provided global squared frame-difference behavior signal and timestamps as evidence of the observed behavior endpoint.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion is restricted to the common complete-trial prefix, converted to float64, averaged in non-overlapping blocks of 10 frames, and then converted to session-specific categorical labels.

ii.
```python
motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
motion_binned = average_blocks_1d(motion_native)
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. Ten-frame averaging follows the paper. Truncation was selected over interpolation because missing samples occur at endpoints and the agent did not want to invent behavior labels.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20/40/60/80% quantiles are computed separately for every retained session. Right-sided search assigns integer classes 0–4; non-distinct edges cause an error.

ii.
```python
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
if np.unique(edges).size != 4:
    raise ValueError(f'{sid}: non-distinct quintile edges {edges}')
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. This implements the requested five equal-percentile bins selected per session. Right-sided handling keeps tied values in the same category.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Imaging and video are aligned by sample ordinal. The code takes the minimum neural, motion, and timestamp length, rounds it down to a whole 60-second trial, and applies identical 10-frame bins and trial boundaries. Missing endpoint video frames are not interpolated.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
corrected = corrected_full[:, :keep_native]
motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
```

iii. The agent relied on documented hardware synchronization and judged ordinal alignment more reliable than the timestamp units. Its notes explicitly say incomplete endpoint windows are discarded without interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Length mismatches are handled by truncating all streams to the shortest observed length and then to the last complete trial. Sessions with too little data, inconsistent shapes/rates, non-distinct class edges, or nonfinite final data raise errors. No missing value is imputed.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
if not np.isclose(fs, 30.0):
    raise ValueError(...)
assert np.isfinite(ni).all() and np.isfinite(ii).all() and np.isfinite(oo).all()
```

iii. The notes report eight or nine shortened behavior endpoints and explain that truncation avoids extrapolating unavailable motion. This reduced the full result from 1,090 nominal trials to 1,081 retained trials.

## 6-a. What are the most time-consuming steps of the code?

i. Full-recording baseline correction for every neuron is the main computation; loading/materializing fluorescence and pickle serialization are the other substantial operations.

ii.
```python
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
corrected_full = suite2p_preprocess(raw_corr_full, ops)
```

iii. The notes identify full fluorescence materialization and baseline filtering as unavoidable per-session work and report about one second per session and 37.39 seconds for the full conversion.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The important numerical operations are already vectorized. Python remains around sessions and in list comprehensions that turn reshaped trial views into contiguous arrays; these could potentially be replaced with batched arrays, but the required nested-list output ultimately needs per-trial objects.

ii.
```python
for i, sp in enumerate(sessions):
    n, x, y, r, info = process_session(sp, show)
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                 for x in neural_binned.reshape(...).transpose(1, 0, 2)]
```

iii. The notes emphasize that neuropil subtraction, filters, block means, quantiles, and trial reshape are vectorized; one-session-at-a-time iteration is intentional for bounded memory.

## 6-c. What processing does the code repeat multiple times?

i. The same loading, validation, baseline correction, binning, quantile, and trialization pipeline is repeated once per session. It also copies each final trial into a contiguous array after forming reshaped views.

ii.
```python
for i, sp in enumerate(sessions):
    n, x, y, r, info = process_session(sp, show)
neural_trials = [np.ascontiguousarray(x, dtype=np.float32) for x in ...]
```

iii. This repetition is by design because baseline parameters, neuron counts, endpoints, and percentile edges are session-specific, and processing all sessions together would consume excessive memory.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Baseline correction is computed over each full neural recording even though endpoint frames beyond `keep_native` are then discarded. The script also loads timestamps but uses only their length, calculates extensive per-session metadata, and optionally creates plots that are not decoder inputs.

ii.
```python
corrected_full = suite2p_preprocess(raw_corr_full, ops)
corrected = corrected_full[:, :keep_native]
tstamps = np.load(..., mmap_mode='r')
info = {'quintile_edges': edges.tolist(), 'class_counts': ...}
```

iii. Full-recording baseline processing is intentional: the notes say truncating before the maximin filter would create an artificial endpoint baseline. Timestamp loading supports safe alignment, while metadata and optional plots support validation rather than training.
