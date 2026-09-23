# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every dated recording that contains `suite2p/plane0/F.npy`, sorts those paths deterministically, and processes all of them unless `--sample` is requested. Per session it loads/memory-maps `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy`, and loads motion energy and camera timestamps. The full run included 41 sessions from six mice.

ii.
```python
def discover_sessions():
    return sorted(p.parents[2] for p in DATA_ROOT.glob('*/*/suite2p/plane0/F.npy'))

F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
Fneu = np.load(session/'suite2p/plane0/Fneu.npy', mmap_mode='r')
iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')
ops = np.load(session/'suite2p/plane0/ops.npy', allow_pickle=True).item()
motion = np.load(session/'move_deve/motion_energy_glob.npy').astype(np.float64)
stamps = np.load(session/'move_deve/tstamps.npy').astype(np.float64)
```

iii. The agent justified file-based discovery as a robust way to include every supplied dated recording, including valid 30-minute recordings that differ from the paper's stated 20-minute duration. Memory mapping and per-session processing limit memory use.

## 1-b. How are the data split into subjects?

i. A subject is the parent directory of each dated session. Unique names are sorted, and each session receives the corresponding integer index.

ii.
```python
subjects=sorted({s.parent.name for s in sessions})
subject_idx.append(subjects.index(s.parent.name))
```

iii. The notes identify the six `jm*` directories as mice and report that rows are consistently tracked within each mouse.

## 1-c. How are the data split into sessions?

i. Each dated recording directory containing a Suite2p fluorescence file is one session; sessions remain separate and sorted by path.

ii.
```python
sessions=discover_sessions()
for i,s in enumerate(sessions):
    n,x,y,info=convert_session(s,args.show_processing and i<2)
```

iii. Keeping each recording day separate preserves daily neuron matrices and permits the required motion-energy percentiles to be selected independently per session.

## 1-d. How are the data split into trials?

i. Each 3-Hz session is divided into consecutive, non-overlapping 60-second trials of 180 bins. An incomplete final segment is dropped.

ii.
```python
TRIAL_T = int(TRIAL_SECONDS * ANALYSIS_HZ)
n_trials = n_bins // TRIAL_T
for tr in range(n_trials):
    sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
    neural_trials.append(np.ascontiguousarray(neural_binned[:,sl], dtype=np.float32))
```

iii. The raw recordings are continuous, so the agent followed the explicit instruction to manufacture 60-second trials. All supplied session lengths happened to form complete trials after binning.

## 1-e. How are trials filtered based on quality controls?

i. There is no behavioral trial-quality exclusion because there are no native trials. Only complete 60-second windows are retained, and shape/finiteness assertions are applied to every retained window.

ii.
```python
n_trials = n_bins // TRIAL_T
for n,x,y in zip(neural_trials,input_trials,output_trials):
    assert n.shape==(n_neurons,TRIAL_T) and x.shape==(1,TRIAL_T) and y.shape==(1,TRIAL_T)
    assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
```

iii. The notes state that the paper has continuous sessions and supplies no native trial QC rule; completeness and validity checks are therefore the applicable controls.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from raw ROI fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`; `ops.npy` supplies preprocessing parameters and `iscell.npy` is used to validate curation.

ii.
```python
F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
Fneu = np.load(session/'suite2p/plane0/Fneu.npy', mmap_mode='r')
iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')
```

iii. The agent found that the paper's “dF/F” corresponds to the supplied Track2p/Suite2p baseline-corrected fluorescence implementation, rather than `spks.npy` or conventional division by baseline.

## 2-b. How is the `neural` data processed?

i. In 64-neuron chunks, the agent computes `F - neucoeff*Fneu`, Gaussian-smooths it, applies a 60-second minimum then maximum filter to obtain the maximin baseline, subtracts that baseline, and averages non-overlapping groups of 10 frames.

ii.
```python
fc = np.asarray(F[lo:hi], dtype=np.float32) - neucoeff * np.asarray(Fneu[lo:hi], dtype=np.float32)
flow = gaussian_filter1d(fc, sigma=sigma, axis=1, mode='reflect')
flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
corrected = fc - flow
out[lo:hi] = corrected.reshape(hi-lo, -1, AVERAGE_FRAMES).mean(axis=2)
```

iii. This was chosen to reproduce the Track2p GUI/Suite2p maximin baseline code and the paper's decoder denoising by averaging 10 consecutive timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No rows are removed during conversion. The code asserts that every already-distributed ROI has Suite2p cell probability above 0.5; the supplied rows were already cell-filtered and Track2p-matched across days.

ii.
```python
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f'Distributed rows fail reference iscell criterion in {session}')
```

iii. The constant row count/order within a mouse and all probabilities passing the reference threshold supported treating the files as pre-curated products rather than filtering them again.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trials are consecutive windows on a common session-start grid, with trial slices applied identically to neural, input, and output arrays.

ii.
```python
sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
neural_trials.append(np.ascontiguousarray(neural_binned[:,sl], dtype=np.float32))
# metadata: 'temporal_alignment_event':
# 'session start; trials are consecutive non-overlapping 60-second windows'
```

iii. The recordings are continuous and the requested time variable is relative to session start, so artificial windows rather than stimulus-event alignment are appropriate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 30-Hz samples are averaged in non-overlapping groups of 10, producing 3 Hz data and 333.333 ms bins for both neural and behavior streams.

ii.
```python
AVERAGE_FRAMES = 10
ANALYSIS_HZ = NATIVE_HZ / AVERAGE_FRAMES
out[lo:hi] = corrected.reshape(hi-lo, -1, AVERAGE_FRAMES).mean(axis=2)
'time_bin_size':1000.0/ANALYSIS_HZ
```

iii. This directly follows the paper's stated denoising for decoding and preserves synchronized bin boundaries.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is derived from analysis-bin indices and the known 30-Hz sampling rate, not from a stored raw variable.

ii.
```python
elapsed = ((np.arange(n_bins, dtype=np.float64)*AVERAGE_FRAMES
            + (AVERAGE_FRAMES-1)/2) / NATIVE_HZ).astype(np.float32)
```

iii. The agent considered index-derived physical time reliable because acquisition is at a validated constant 30 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Each 10-frame bin is represented by its center time, `(10*i + 4.5)/30` seconds, cast to float32. Time continues across trial boundaries rather than resetting.

ii.
```python
elapsed = ((np.arange(n_bins, dtype=np.float64)*AVERAGE_FRAMES + 4.5)
           / NATIVE_HZ).astype(np.float32)
```

iii. Bin centers were chosen as the physical time represented by a mean over frames 0 through 9; continuous session-relative time implements “time elapsed from the beginning of the session.”

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The elapsed array has one value per post-binning neural column and is sliced with exactly the same trial slice.

ii.
```python
sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
input_trials.append(np.ascontiguousarray(elapsed[None,sl], dtype=np.float32))
```

iii. Shared bin indices and slices avoid resampling or trial-boundary offsets.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses the supplied one-dimensional `move_deve/motion_energy_glob.npy`; `tstamps.npy` is used to locate missing camera samples.

ii.
```python
motion = np.load(session/'move_deve/motion_energy_glob.npy').astype(np.float64)
stamps = np.load(session/'move_deve/tstamps.npy').astype(np.float64)
```

iii. The paper defines this supplied global trace as squared consecutive-frame pixel differences; recomputing it from video is unnecessary.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If behavior is shorter than neural data, timestamp gaps are converted to inferred frame positions and missing values are linearly interpolated. The repaired trace is then averaged over the same 10-frame groups as neural data.

ii.
```python
dt = np.diff(stamps)
med = float(np.median(dt))
steps[dt > 1.5 * med] = np.maximum(1, np.rint(dt[dt > 1.5 * med] / med).astype(np.int64))
positions = np.r_[0, np.cumsum(steps)]
repaired = np.interp(np.arange(n_neural), positions, motion)
motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1)
```

iii. The agent wanted to preserve trigger/frame-order synchronization while repairing only local gaps that exactly explain a neural/behavior length deficit, avoiding drift from absolute timestamp rounding.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four edges are the 20th, 40th, 60th, and 80th percentiles of the full binned trace for that session. Right-sided insertion encodes integer classes 0–4.

ii.
```python
edges = np.quantile(motion_binned.astype(np.float64), [.2, .4, .6, .8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. This directly implements five equal-percentile bins selected independently per session; right-sided handling gives deterministic behavior for ties without jittering the data.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Missing camera samples are repaired to the native neural length, both streams use identical 10-frame bins, equality of binned lengths is asserted, and identical trial slices are used.

ii.
```python
motion_repaired, motion_raw, stamps, inserted = repair_motion(session, n_native)
if neural_binned.shape[1] != len(motion_binned):
    raise AssertionError(f'Post-bin alignment mismatch in {sid}')
output_trials.append(np.ascontiguousarray(labels[None,sl], dtype=np.int64))
```

iii. The camera was microscope-triggered, so frame order is the alignment basis. The notes report independent exact checks of repaired samples, common bins, and output labels.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid shapes/non-finite behavior, neural shape mismatches, failed cell criteria, unexpected rates/baselines, and unexplained behavior deficits raise errors. Explained missing camera frames are interpolated. Behavior longer than neural is defensively truncated, and incomplete final trials are dropped.

ii.
```python
if inserted != deficit or positions[-1] != n_neural - 1:
    raise ValueError(...)
elif len(motion) == n_neural:
    repaired = motion
else:
    repaired = motion[:n_neural]
```

iii. The agent emphasized fail-fast validation and only repairing deficits supported exactly by local timestamp gaps; observed motion values are asserted unchanged at reconstructed positions.

## 6-a. What are the most time-consuming steps of the code?

i. The Gaussian and 60-second min/max filters across every neuron and native frame dominate computation. Loading large NPY arrays and serializing the roughly 395-MiB pickle are secondary costs.

ii.
```python
flow = gaussian_filter1d(fc, sigma=sigma, axis=1, mode='reflect')
flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
```

iii. The notes identify full native-rate baseline intermediates as large and report chunking/memory mapping plus vectorized SciPy filters as the mitigation; full conversion took about 41 seconds.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Session iteration is necessary for variable neuron counts and session-local quantiles. The trial loop mostly creates the required nested list and contiguous copies, but slicing could be reshaped/batched before list conversion. `subjects.index(...)` could be replaced by a lookup dictionary. The chunk loop is deliberate memory control rather than an obvious inefficiency.

ii.
```python
for tr in range(n_trials):
    sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
    ...
for lo in range(0, n_neurons, CHUNK_NEURONS):
    ...
subject_idx.append(subjects.index(s.parent.name))
```

iii. The agent explicitly avoided per-neuron Python loops and vectorized filtering and binning within chunks. It regarded chunk iteration as necessary to avoid excessive intermediate memory.

## 6-c. What processing does the code repeat multiple times?

i. The `F.npy` header is opened once in baseline processing and again to obtain `n_native`; trial arrays are then scanned again for shape/finiteness assertions. With plotting enabled, already-computed processing products are traversed for figures. Per-session validation and preprocessing are otherwise intentionally repeated for each independent recording.

ii.
```python
F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
...
n_native = int(np.load(session/'suite2p/plane0/F.npy', mmap_mode='r').shape[1])
for n,x,y in zip(neural_trials,input_trials,output_trials):
    assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
```

iii. The notes prioritize strict independent sanity checks and bounded memory over removing these small repeated passes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Source timestamps are returned from `repair_motion` but not otherwise consumed after repair. Optional plotting retains detailed native-rate traces and generates diagnostic figures that are not used by the decoder. Per-session diagnostic metadata and class counts aid auditing but are not decoder features.

ii.
```python
return repaired.astype(np.float32), motion, stamps, inserted
...
motion_repaired, motion_raw, stamps, inserted = repair_motion(session, n_native)
if show_processing:
    make_processing_plot(...)
```

iii. The agent intentionally included these diagnostics for required processing visualization, reproducibility, and sanity checking; normal full conversion without `--show-processing` avoids construction of plot details.
