# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sessions by globbing every `F.npy` under `/app/data/*/*/suite2p/plane0/` and treats the session directory two levels up as the recording to convert. Within each session it loads calcium data from `suite2p/plane0` and behavior from `move_deve`.

ii. 
```python
def discover_sessions():
    """Return dated session paths in subject/date order."""
    return sorted(p.parents[2] for p in DATA_ROOT.glob('*/*/suite2p/plane0/F.npy'))

F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
Fneu = np.load(session/'suite2p/plane0/Fneu.npy', mmap_mode='r')
iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')
ops = np.load(session/'suite2p/plane0/ops.npy', allow_pickle=True).item()
motion = np.load(session/'move_deve/motion_energy_glob.npy').astype(np.float64)
stamps = np.load(session/'move_deve/tstamps.npy').astype(np.float64)
```

iii. In `CONVERSION_NOTES.md` Step 5 the agent states that each dated recording should remain a separate session, and in Step 31 of the trajectory it says the script should “discover all dated sessions deterministically” and load the neural and behavioral arrays needed for conversion.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each discovered session path. The final `subjects` list is the sorted unique set of those names, and each session gets a `subject_idx`.

ii. 
```python
subjects=sorted({s.parent.name for s in sessions})
...
subject_idx.append(subjects.index(s.parent.name))
```

iii. In Step 5 notes the agent justified keeping each dated recording separate while preserving subject identity from the directory structure.

## 1-c. How are the data split into sessions?

i. Each dated recording directory is one session. The code keeps them separate rather than merging days within a mouse.

ii. 
```python
def discover_sessions():
    """Return dated session paths in subject/date order."""
    return sorted(p.parents[2] for p in DATA_ROOT.glob('*/*/suite2p/plane0/F.npy'))
...
for i,s in enumerate(sessions):
    n,x,y,info=convert_session(s,args.show_processing and i<2)
    neural.append(n); inputs.append(x); outputs.append(y); infos.append(info)
```

iii. Step 5 notes say: “Session definition: Each dated recording remains a separate session. This preserves per-day neuron matrices and obeys ‘selected per session’ target quantiles.”

## 1-d. How are the data split into trials?

i. The dataset is treated as continuous, with no native trial structure. After temporal averaging to 3 Hz, each session is split into consecutive non-overlapping 60-second trials of 180 bins. Any incomplete tail would be dropped.

ii. 
```python
TRIAL_SECONDS = 60
TRIAL_T = int(TRIAL_SECONDS * ANALYSIS_HZ)
...
n_bins = neural_binned.shape[1]
n_trials = n_bins // TRIAL_T
use = n_trials * TRIAL_T
if use != n_bins:
    print(f'  WARNING dropping {n_bins-use} incomplete analysis bins from {sid}')
...
for tr in range(n_trials):
    sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
    neural_trials.append(np.ascontiguousarray(neural_binned[:,sl], dtype=np.float32))
    input_trials.append(np.ascontiguousarray(elapsed[None,sl], dtype=np.float32))
    output_trials.append(np.ascontiguousarray(labels[None,sl], dtype=np.int64))
```

iii. Step 5 notes say: “Trials: Split post-binned data into consecutive non-overlapping 60-s trials, 180 points each.” Step 29 of the trajectory repeats that the conversion spec uses “60-second trials.”

## 1-e. How are trials filtered based on quality controls?

i. The code does not apply a trial-quality filter. It only excludes incomplete trailing trial fragments if a session length is not an exact multiple of 60 seconds after binning, and it asserts final per-trial shapes and finiteness.

ii. 
```python
if use != n_bins:
    print(f'  WARNING dropping {n_bins-use} incomplete analysis bins from {sid}')
...
for n,x,y in zip(neural_trials,input_trials,output_trials):
    assert n.shape==(n_neurons,TRIAL_T) and x.shape==(1,TRIAL_T) and y.shape==(1,TRIAL_T)
    assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
```

iii. The notes and trajectory do not describe any additional trial curation rule; the agent consistently treats the sessions as continuous recordings that just need fixed-length segmentation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from `F.npy` and `Fneu.npy`, with `iscell.npy` and `ops.npy` used for validation and processing parameters.

ii. 
```python
F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
Fneu = np.load(session/'suite2p/plane0/Fneu.npy', mmap_mode='r')
iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')
...
neucoeff = float(ops.get('neucoeff', 0.7))
baseline = ops.get('baseline', 'maximin')
sigma = float(ops.get('sig_baseline', 10.0))
win = int(float(ops.get('win_baseline', 60.0)) * fs)
```

iii. Step 4 notes say the correct neural activity is the paper/reference baseline-corrected fluorescence from `F` and `Fneu`, not `spks.npy` or raw `F` alone.

## 2-b. How is the `neural` data processed?

i. The code reproduces the Track2p/Suite2p baseline-corrected fluorescence pipeline: neuropil subtraction `F - neucoeff * Fneu`, Gaussian smoothing, 60-second min and max filters for the maximin baseline, subtraction of that baseline, then averaging every 10 frames.

ii. 
```python
fc = np.asarray(F[lo:hi], dtype=np.float32) - neucoeff * np.asarray(Fneu[lo:hi], dtype=np.float32)
flow = gaussian_filter1d(fc, sigma=sigma, axis=1, mode='reflect')
flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
corrected = fc - flow
out[lo:hi] = corrected.reshape(hi-lo, -1, AVERAGE_FRAMES).mean(axis=2)
```

iii. Step 28 of the trajectory says the “reference neural preprocessing is now exact: neuropil subtraction followed by Suite2p maximin baseline subtraction, with no baseline division.” Step 4 notes also state that the paper’s “dF/F” corresponds to `Fc - Flow`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not re-filter neurons; it asserts that every distributed row already satisfies the Suite2p cell criterion `iscell[:,1] > 0.5` and then keeps all rows.

ii. 
```python
iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')
...
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f'Distributed rows fail reference iscell criterion in {session}')
```

iii. Step 5 notes say: “Already curated cells: Keep all rows after asserting `iscell[:,1] > 0.5`; files are already all-day matched and consistently ordered within each mouse.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to session start, then divided into consecutive 60-second windows. There is no event-triggered alignment inside the session.

ii. 
```python
'metadata':{
    ...
    'temporal_alignment_event':'session start; trials are consecutive non-overlapping 60-second windows',
    'off_start':0.0,'off_end':60.0,
    ...
}
```

iii. Step 5 notes say the alignment event is session start / consecutive 60-second window start, because the recordings are continuous and the trials are artificial windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output is at 3 Hz, i.e. 333.33 ms bins, produced by averaging non-overlapping 10-frame blocks from the native 30 Hz streams. This binning is applied to both neural and motion signals.

ii. 
```python
NATIVE_HZ = 30
AVERAGE_FRAMES = 10
ANALYSIS_HZ = NATIVE_HZ / AVERAGE_FRAMES
...
out[lo:hi] = corrected.reshape(hi-lo, -1, AVERAGE_FRAMES).mean(axis=2)
...
motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1, dtype=np.float64).astype(np.float32)
...
'time_bin_size':1000.0/ANALYSIS_HZ,
```

iii. Step 4 notes say the paper averages both streams in “10 consecutive timestamps,” and Step 29 of the trajectory lists “10-frame synchronous averaging” as one of the fixed conversion choices.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not taken from a raw file variable. The code derives it from the analysis-bin index using the known 30 Hz sampling rate and 10-frame bin width.

ii. 
```python
elapsed = ((np.arange(n_bins, dtype=np.float64)*AVERAGE_FRAMES + (AVERAGE_FRAMES-1)/2) / NATIVE_HZ).astype(np.float32)
...
input_trials.append(np.ascontiguousarray(elapsed[None,sl], dtype=np.float32))
```

iii. Step 5 notes say: “Time input: Use physical bin-center elapsed seconds from session start.” The agent treated frame timing as fully determined by the acquisition constants.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as session-relative elapsed seconds at the center of each 10-frame bin, then sliced into trials without resetting within trial.

ii. 
```python
elapsed = ((np.arange(n_bins, dtype=np.float64)*AVERAGE_FRAMES + (AVERAGE_FRAMES-1)/2) / NATIVE_HZ).astype(np.float32)
for tr in range(n_trials):
    sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
    input_trials.append(np.ascontiguousarray(elapsed[None,sl], dtype=np.float32))
```

iii. Step 5 notes explicitly justify using bin-center time and say it “does not reset each 60-s trial because the requested variable is elapsed time from session beginning.”

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time vector is defined on the same binned timeline as the neural data and is sliced with the same per-trial slice indices.

ii. 
```python
n_bins = neural_binned.shape[1]
...
elapsed = ((np.arange(n_bins, dtype=np.float64)*AVERAGE_FRAMES + (AVERAGE_FRAMES-1)/2) / NATIVE_HZ).astype(np.float32)
for tr in range(n_trials):
    sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
    neural_trials.append(np.ascontiguousarray(neural_binned[:,sl], dtype=np.float32))
    input_trials.append(np.ascontiguousarray(elapsed[None,sl], dtype=np.float32))
```

iii. The agent’s general justification in Step 5 is that both streams should share a synchronized 3 Hz grid after common 10-frame averaging.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `motion_energy_glob.npy`. The code uses `tstamps.npy` to infer where missing camera samples should be inserted before alignment to neural frames.

ii. 
```python
motion = np.load(session/'move_deve/motion_energy_glob.npy').astype(np.float64)
stamps = np.load(session/'move_deve/tstamps.npy').astype(np.float64)
```

iii. In Step 28 and Step 30 of the trajectory, the agent argues that local timestamp gaps are the reliable signal for missing behavior samples, while equal-length streams with timing glitches should be left unchanged.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If behavior is shorter than neural, the code detects large local timestamp gaps, reconstructs the intended native-frame positions, and linearly interpolates missing samples. It then averages motion over non-overlapping 10-frame bins and computes within-session quantile thresholds.

ii. 
```python
dt = np.diff(stamps)
med = float(np.median(dt))
steps = np.ones(len(dt), dtype=np.int64)
large = dt > 1.5 * med
steps[large] = np.maximum(1, np.rint(dt[large] / med).astype(np.int64))
positions = np.r_[0, np.cumsum(steps)]
...
repaired = np.interp(np.arange(n_neural), positions, motion)
...
motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1, dtype=np.float64).astype(np.float32)
edges = np.quantile(motion_binned.astype(np.float64), [.2, .4, .6, .8])
```

iii. Step 28 says absolute timestamp rounding accumulates drift, so interpolation should only occur at local gaps; Step 30 adds the rule that repair is allowed only when the inferred missing count exactly matches the neural-behavior deficit.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The binned motion-energy trace is discretized within each session into five categories using the 20th, 40th, 60th, and 80th percentiles, with right-inclusive thresholding via `np.searchsorted`.

ii. 
```python
edges = np.quantile(motion_binned.astype(np.float64), [.2, .4, .6, .8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
...
'output_values':[['lowest (0-20%)','low (20-40%)','middle (40-60%)',
                  'high (60-80%)','highest (80-100%)']]
```

iii. Step 5 notes say: “Quantile target: Derive four percentile boundaries from the full repaired/3-Hz session before trial splitting.”

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The behavior trace is repaired to native neural length when needed, then motion and neural are binned with the same 10-frame scheme, checked for equal post-bin length, and sliced into trials with the same indices.

ii. 
```python
motion_repaired, motion_raw, stamps, inserted = repair_motion(session, n_native)
motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1, dtype=np.float64).astype(np.float32)
if neural_binned.shape[1] != len(motion_binned):
    raise AssertionError(f'Post-bin alignment mismatch in {sid}')
...
for tr in range(n_trials):
    sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
    neural_trials.append(np.ascontiguousarray(neural_binned[:,sl], dtype=np.float32))
    output_trials.append(np.ascontiguousarray(labels[None,sl], dtype=np.int64))
```

iii. Step 29 of the trajectory lists “timestamp-aware repair of internal video drops” and “10-frame synchronous averaging” as fixed conversion choices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases defensively: invalid shapes or non-finite values raise errors; shorter behavior arrays are repaired only when timestamp gaps fully explain the deficit; longer behavior arrays are truncated defensively; incomplete analysis tails are dropped with a warning; all final trial arrays are asserted to have valid shape and finite values.

ii. 
```python
if motion.ndim != 1 or stamps.ndim != 1 or len(motion) != len(stamps):
    raise ValueError(f'Invalid behavior arrays in {session}')
if not (np.isfinite(motion).all() and np.isfinite(stamps).all()):
    raise ValueError(f'Non-finite behavior in {session}')
...
if inserted != deficit or positions[-1] != n_neural - 1:
    raise ValueError(f'Behavior gaps do not explain deficit in {session}: '
                     f'deficit={deficit}, inferred={inserted}')
...
elif len(motion) == n_neural:
    repaired = motion
else:
    repaired = motion[:n_neural]
...
if use != n_bins:
    print(f'  WARNING dropping {n_bins-use} incomplete analysis bins from {sid}')
```

iii. Step 30 of the trajectory gives the explicit justification for conservative repair: only interpolate when the stream is short and the inferred inserted count closes the exact deficit.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive part is neural preprocessing: loading the fluorescence arrays and applying Gaussian/minimum/maximum filtering for baseline subtraction across all neurons. Optional plotting is extra overhead but not part of the main conversion path.

ii. 
```python
for lo in range(0, n_neurons, CHUNK_NEURONS):
    hi = min(n_neurons, lo + CHUNK_NEURONS)
    fc = np.asarray(F[lo:hi], dtype=np.float32) - neucoeff * np.asarray(Fneu[lo:hi], dtype=np.float32)
    flow = gaussian_filter1d(fc, sigma=sigma, axis=1, mode='reflect')
    flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
    flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
```

iii. Step 31 of the trajectory says the implementation should use “efficient chunked baseline processing” specifically to control the cost of fluorescence preprocessing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are the chunk loop over neurons and the trial-construction loop. The motion-repair logic is already vectorized compared with a frame-by-frame insertion approach, but trial assembly still appends one trial at a time.

ii. 
```python
for lo in range(0, n_neurons, CHUNK_NEURONS):
    ...
for tr in range(n_trials):
    sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
    neural_trials.append(np.ascontiguousarray(neural_binned[:,sl], dtype=np.float32))
    input_trials.append(np.ascontiguousarray(elapsed[None,sl], dtype=np.float32))
    output_trials.append(np.ascontiguousarray(labels[None,sl], dtype=np.int64))
```

iii. In Step 31 the agent explicitly chose chunking as a memory/speed tradeoff; the trajectory also makes clear that the dropped-frame logic was redesigned to avoid iterative `np.insert` operations.

## 6-c. What processing does the code repeat multiple times?

i. The code re-opens `F.npy` after preprocessing to recover the native frame count, repeatedly does `subjects.index(...)` in the session loop, and performs per-trial shape/finite assertions after having already constructed all trials from common session-level arrays.

ii. 
```python
n_native = int(np.load(session/'suite2p/plane0/F.npy', mmap_mode='r').shape[1])
...
subject_idx.append(subjects.index(s.parent.name))
...
for n,x,y in zip(neural_trials,input_trials,output_trials):
    assert n.shape==(n_neurons,TRIAL_T) and x.shape==(1,TRIAL_T) and y.shape==(1,TRIAL_T)
    assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
```

iii. The notes emphasize validation and defensiveness, so these repeated operations appear to have been chosen for safety and bookkeeping rather than minimum work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion can optionally generate detailed processing plots and keep diagnostic traces for plotting. It also returns `stamps` from `repair_motion` even though downstream conversion uses only the repaired motion and the insert count. The metadata stores extensive per-session diagnostics that are useful for auditing but not for decoder training.

ii. 
```python
def make_processing_plot(...):
    ...

motion_repaired, motion_raw, stamps, inserted = repair_motion(session, n_native)
...
if show_processing:
    make_processing_plot(sid, details, motion_raw, motion_repaired, motion_binned, edges, labels)
...
'metadata':{
    ...
    'session_info':infos,
    ...
}
```

iii. Step 31 says diagnostic plots were intentionally included, and Step 45 says the remaining work included user-facing documentation and cached investigation artifacts, showing that the agent deliberately kept extra audit material beyond the minimal decoder inputs.
