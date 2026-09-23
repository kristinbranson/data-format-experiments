# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all sessions by globbing `/app/data/*/*/suite2p/plane0`, then processes each discovered session independently. For each session it loads calcium arrays `F.npy` and `Fneu.npy`, ROI metadata `iscell.npy` and `ops.npy`, and behavioral arrays `motion_energy_glob.npy` and `tstamps.npy`. Trials are not loaded directly; they are created later by reshaping the continuous session data into consecutive 60-second windows.

ii.
```python
DATA_ROOT = Path('/app/data')

def discover_sessions() -> list[Path]:
    return sorted(DATA_ROOT.glob('*/*/suite2p/plane0'))

def process_session(sp: Path, show_processing: bool = False) -> tuple[list, list, list, np.ndarray, dict]:
    session_dir = sp.parent.parent
    F = np.load(sp / 'F.npy', mmap_mode='r')
    Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(sp / 'iscell.npy')
    ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
    motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
    tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
```

iii. In `CONVERSION_NOTES.md`, the AI says the data tree consists of subject/session directories and that Track2p uses Suite2p-style outputs. In Step 5 it justifies session-wise processing as the natural unit for conversion and says one session should become one decoder session, with trials synthesized afterward from continuous recordings.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory names of the discovered session paths. The subject list is the sorted set of those names, and each converted session gets a `subject_idx` by mapping its subject name into that sorted list.

ii.
```python
subjects = sorted({p.parent.parent.parent.name for p in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[info['subject']])
```

iii. In Step 2 and Step 5 notes, the AI states that the dataset contains six subject directories and that subject directory names are the authoritative mouse identifiers. It also says lexicographic ordering is deterministic and chronological enough because the session folders use ISO-style dates within each subject.

## 1-c. How are the data split into sessions?

i. Each session is one daily recording directory identified by the presence of `suite2p/plane0`. The AI sorts those paths and treats each as one session in the converted dataset.

ii.
```python
def discover_sessions() -> list[Path]:
    return sorted(DATA_ROOT.glob('*/*/suite2p/plane0'))

for i, sp in enumerate(sessions):
    show = show_processing and i < 2
    n, x, y, r, info = process_session(sp, show)
    neural.append(n); inputs.append(x); outputs.append(y); regions.append(r); infos.append(info)
```

iii. The Step 5 notes say sessions are ordered lexicographically by subject ID and then session directory name, and that each recording day remains a separate decoder session. The AI explicitly rejected using Track2p’s longitudinal identity tracking to merge multiple days into one session.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous data with no native trial structure. It creates artificial trials as consecutive non-overlapping 60-second windows, corresponding to `1800` native 30 Hz frames or `180` post-binning time bins. Only complete windows are retained.

ii.
```python
NATIVE_FRAMES_PER_TRIAL = 1800  # 60 s * 30 Hz
AVERAGE_FRAMES = 10
N_BINS_PER_TRIAL = NATIVE_FRAMES_PER_TRIAL // AVERAGE_FRAMES
...
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
...
n_trials = keep_native // NATIVE_FRAMES_PER_TRIAL
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                 for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
output_trials = [np.ascontiguousarray(x[None, :], dtype=np.int64)
                 for x in labels.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. In Step 3 and Step 5 notes, the AI says the original experiment is continuous rather than trial-based, so 60-second windows are a task-imposed downstream format. It justifies using non-overlapping windows because that is the simplest interpretation of “Split sessions into 60-second trials.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply content-based trial QC such as behavioral or neural rejection criteria, but it does retain only complete synchronized 60-second windows. Any session with fewer than two complete 60-second windows is rejected outright.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
    raise ValueError(f'{sid}: fewer than two complete 60-s trials')
```

iii. In Step 3 and Step 5 notes, the AI says there is no paper-defined trial exclusion because the recordings are continuous, but argues that the decoder format requires at least two trials and that only complete windows with all synchronized streams should be kept. It explicitly preferred dropping incomplete endpoint windows over fabricating behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural data is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`, together with `ops.npy` for sampling rate and baseline-preprocessing parameters. `iscell.npy` is used to define which ROIs are kept.

ii.
```python
F = np.load(sp / 'F.npy', mmap_mode='r')
Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
iscell = np.load(sp / 'iscell.npy')
ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
...
cell_mask = iscell[:, 1] > 0.5
neucoeff = float(ops.get('neucoeff', 0.7))
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
```

iii. The AI’s Step 4 and Step 5 notes say the paper’s reference neural stream is baseline-corrected fluorescence, not Suite2p `spks.npy`. It therefore chose `F` and `Fneu` as the primary raw variables, using `ops.npy` to reproduce the saved Suite2p baseline settings and `iscell.npy` to enforce the paper’s cell criterion.

## 2-b. How is the `neural` data processed?

i. Neural processing consists of neuropil subtraction followed by a local reimplementation of Suite2p’s baseline correction, then 10-frame averaging. The local function supports the same baseline modes but the intended case is `maximin` with the session’s saved parameters from `ops.npy`.

ii.
```python
def suite2p_preprocess(F: np.ndarray, ops: dict) -> np.ndarray:
    x = np.asarray(F, dtype=np.float32).copy()
    baseline = ops.get('baseline', 'maximin')
    sig = float(ops.get('sig_baseline', 10.0))
    fs = float(ops['fs'])
    win = max(1, int(float(ops.get('win_baseline', 60.0)) * fs))
    prct = float(ops.get('prctile_baseline', 8.0))
    if baseline == 'maximin':
        flow = gaussian_filter(x, sigma=(0.0, sig), mode='reflect')
        flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
        flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
    ...
    x -= flow
    return x
...
corrected_full = suite2p_preprocess(raw_corr_full, ops)
neural_binned = average_blocks_2d(corrected)
```

iii. In Step 4 through Step 6 notes, the AI justifies this as matching the paper’s statement that baseline-corrected fluorescence traces were used “as dF/F” with default Suite2p parameters. It also says it intentionally did not divide by the baseline because its inspection of Suite2p indicated that `dcnv.preprocess` subtracts the estimated baseline rather than computing a literal ratio.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters ROIs by the Suite2p cell-probability criterion `iscell[:, 1] > 0.5`. In the distributed dataset this apparently removes nothing, because the notes say all provided ROIs already satisfy the threshold.

ii.
```python
cell_mask = iscell[:, 1] > 0.5
if len(cell_mask) != F.shape[0]:
    raise ValueError(f'{sid}: iscell/F neuron mismatch')
# Distributed data are already curated; still enforce the paper criterion.
if not np.all(cell_mask):
    print(f'  {sid}: filtering {np.sum(~cell_mask)} ROIs below iscell threshold', flush=True)
...
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
```

iii. The Step 2 and Step 4 notes say that Track2p had already produced a curated longitudinal ROI set and that every supplied ROI passes the `> 0.5` criterion. The AI still enforced the threshold because the paper and Track2p defaults both use it, and it wanted the code to remain faithful if a session ever contained rejected ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI effectively aligns each extracted neural trial to the start of its own consecutive 60-second window, while also keeping an input channel that carries absolute elapsed session time. In metadata it describes the alignment event as the start of each non-overlapping 60-second window, with `off_start = 0.0` and `off_end = 60.0`.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Start of each consecutive non-overlapping 60-second window in a continuous session',
    'off_start': 0.0,
    'off_end': 60.0,
    ...
}
...
n_trials = keep_native // NATIVE_FRAMES_PER_TRIAL
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                 for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
```

iii. In Step 5 notes, the AI says the recordings have no native behavioral event, so the meaningful alignment event for each trial is simply the start of each task-imposed 60-second window. It distinguishes that from the absolute-time decoder input, which continues across window boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion data by averaging non-overlapping blocks of 10 native frames, reducing 30 Hz to 3 Hz. This yields a converted time bin size of `1000 * 10 / 30 = 333.333...` ms.

ii.
```python
AVERAGE_FRAMES = 10
...
def average_blocks_2d(x: np.ndarray, block: int = AVERAGE_FRAMES) -> np.ndarray:
    return x.reshape(x.shape[0], -1, block).mean(axis=2, dtype=np.float32)

def average_blocks_1d(x: np.ndarray, block: int = AVERAGE_FRAMES) -> np.ndarray:
    return x.reshape(-1, block).mean(axis=1, dtype=np.float64)
...
'time_bin_size': 1000.0 * AVERAGE_FRAMES / 30.0,
```

iii. The Step 3 through Step 5 notes say the paper averaged both neural and behavior traces in bins of 10 timestamps for decoding, so the AI treated that as a reference-preserving requirement rather than an optional denoising step.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a dedicated raw timestamp channel in seconds. It is derived from the synchronized sample ordinal and the imaging frame rate `fs` from `ops.npy`.

ii.
```python
fs = float(ops['fs'])
...
elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
            + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
```

iii. In Step 4 and Step 5 notes, the AI says the stored behavioral timestamps have unclear units, so elapsed time should instead be derived from frame index at the documented 30 Hz sampling rate. It treats the microscope-triggered 30 Hz acquisition as the trustworthy timing source.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one time value per 10-frame averaged bin using the mean native-frame position of that bin, then casts to `float32` and reshapes to a `(1, 180)` trial time series. Time continues monotonically across all trials within a session.

ii.
```python
elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
            + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
...
input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. In Step 5 notes, the AI explicitly justifies using the mean time of each averaged 10-frame block: because neural and behavior are averaged, the bin-center time is the most representative coordinate for the resulting sample. It also notes that time should not reset within a session because the task asks for elapsed time from session start.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the time input to neural data by constructing both from the same retained native-frame prefix, using the same 10-frame binning, and then reshaping both into the same sequence of 60-second trials.

ii.
```python
corrected = corrected_full[:, :keep_native]
neural_binned = average_blocks_2d(corrected)
...
elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
            + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
...
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                 for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. The Step 5 notes say alignment is by shared sample ordinal after common-length truncation, not by interpreting `tstamps.npy` as seconds. The AI considered this safer because the paper reports hardware synchronization and the raw timestamp magnitudes looked unreliable.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The motion-energy output is derived from `move_deve/motion_energy_glob.npy`. The AI also loads `tstamps.npy` to determine the synchronized common length, but it does not recompute motion energy from video frames and does not use `interframe_int.npy`.

ii.
```python
motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
...
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
```

iii. In Step 3 notes, the AI says the paper’s behavioral quantity is global motion energy already computed from squared frame differences, so `motion_energy_glob.npy` should be treated as the raw behavioral signal for conversion. In Step 4 it explains that `tstamps.npy` is used only as a consistency bound because its units did not look reliable as seconds.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI truncates motion energy to the synchronized common prefix shared with calcium and timestamps, averages it in non-overlapping 10-frame bins, computes within-session quintile edges, and converts the binned motion values into class labels 0-4. It does not interpolate missing endpoint samples.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
...
motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
motion_binned = average_blocks_1d(motion_native)

edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. In Step 4 and Step 5 notes, the AI argues that incomplete endpoint behavior should be dropped rather than extrapolated, because the timestamps already show those samples are genuinely unavailable. It frames the 10-frame averaging as reference-preserving and the quintile discretization as required by the new decoder task.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes session-specific quintile thresholds at the 20th, 40th, 60th, and 80th percentiles of the 10-frame-averaged motion trace, then assigns category labels with right-sided thresholding so outputs take integer values `0` through `4`.

ii.
```python
edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
if np.unique(edges).size != 4:
    raise ValueError(f'{sid}: non-distinct quintile edges {edges}')
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
...
assert labels.min() == 0 and labels.max() == 4
```

iii. The Step 5 notes say “selected per session” means the percentile edges must be computed separately within each session. The AI also justifies right-sided thresholding as a deterministic way to keep tied values in the same class rather than forcing artificial splits.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by sample ordinal at 30 Hz, using the common observed prefix across `F`, `Fneu`, motion, and timestamps. It then bins both streams in the same 10-frame blocks and reshapes them into the same trials. Missing endpoint behavior is handled by truncation, not interpolation.

ii.
```python
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
...
corrected = corrected_full[:, :keep_native]
motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
neural_binned = average_blocks_2d(corrected)
motion_binned = average_blocks_1d(motion_native)
```

iii. In Step 4 notes, the AI says microscope-triggered video means frame-ordinal alignment is the safest interpretation. It explicitly rejects interpolation because it interprets the short behavior streams as genuinely missing endpoint observations rather than isolated dropped frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles mismatched stream lengths by truncating all modalities to the shortest common prefix and then discarding any incomplete final 60-second window. It also checks for invalid sampling rate, shape mismatches, non-distinct motion quantile edges, and insufficient trial count, and raises errors rather than silently proceeding.

ii.
```python
if not np.isclose(fs, 30.0):
    raise ValueError(f'{sid}: expected 30 Hz, found {fs}')
if F.shape != Fneu.shape or F.ndim != 2:
    raise ValueError(f'{sid}: incompatible F/Fneu shapes {F.shape}, {Fneu.shape}')
...
common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
    raise ValueError(f'{sid}: fewer than two complete 60-s trials')
...
if np.unique(edges).size != 4:
    raise ValueError(f'{sid}: non-distinct quintile edges {edges}')
```

iii. In Step 4 and Step 5 notes, the AI says it intentionally avoids interpolation or extrapolation for missing behavior because it considers that fabricated data. In Step 10 notes it also records a bug it found in its own first implementation: it initially baseline-corrected only the retained prefix, then fixed this to preprocess the full neural recording before truncation.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify full-session neural preprocessing as the main expensive step, especially materializing the full neuropil-corrected fluorescence array and running the Gaussian/minimum/maximum baseline filters. Session-wise file loading is also nontrivial, but less dominant.

ii.
```python
raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                 - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
corrected_full = suite2p_preprocess(raw_corr_full, ops)
```

iii. In Step 6 notes, the AI says this work is the unavoidable cost of reproducing Suite2p preprocessing and that processing all sessions simultaneously would use excessive memory. It created a local `suite2p_preprocess` specifically to avoid the overhead of importing the whole Suite2p package during each conversion run.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the heavy numeric processing, so the remaining non-vectorized parts are mostly Python-level list comprehensions that package trials and per-trial validation loops. Optional plotting also iterates over a few example neurons.

ii.
```python
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                 for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
output_trials = [np.ascontiguousarray(x[None, :], dtype=np.int64)
                 for x in labels.reshape(n_trials, N_BINS_PER_TRIAL)]

for ni, ii, oo in zip(neural_trials, input_trials, output_trials):
    assert ni.shape == (int(cell_mask.sum()), N_BINS_PER_TRIAL)
    assert ii.shape == oo.shape == (1, N_BINS_PER_TRIAL)
```

iii. In Step 6 notes, the AI says it intentionally used vectorized neuropil subtraction, filtering, block averaging, percentile thresholding, and trial reshaping. It treats the remaining Python loops as packaging and validation overhead, not the main bottleneck.

## 6-c. What processing does the code repeat multiple times?

i. The main conversion path does not repeat large numerical transforms unnecessarily across sessions, but it does repeatedly copy each trial into contiguous arrays and repeatedly perform per-trial shape/finite assertions after trialization.

ii.
```python
neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                 for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
output_trials = [np.ascontiguousarray(x[None, :], dtype=np.int64)
                 for x in labels.reshape(n_trials, N_BINS_PER_TRIAL)]

for ni, ii, oo in zip(neural_trials, input_trials, output_trials):
    assert np.isfinite(ni).all() and np.isfinite(ii).all() and np.isfinite(oo).all()
```

iii. The Step 6 notes argue that the repeated work it kept is deliberate: one-session-at-a-time processing bounds memory, and trial-level contiguous copies/assertions are used to make serialization and downstream validation predictable.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does a few pieces of work that are helpful for validation or documentation but not required by the downstream decoder: loading `tstamps.npy` only to constrain common length, building rich `session_info` metadata, optional diagnostic plotting, and per-trial assertion checks. These do not change the converted neural/input/output arrays used for training.

ii.
```python
tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
...
if show_processing:
    plot_processing(sid, raw_corr, corrected, neural_binned, motion_native,
                    motion_binned, labels, edges)
...
'metadata': {
    ...
    'session_info': infos,
},
```

iii. In Step 6 and Step 10 notes, the AI says these extra steps are there to support sanity checks, debugging, and reproducibility. It explicitly separates them from the core conversion path and limits plotting to at most two sessions.
