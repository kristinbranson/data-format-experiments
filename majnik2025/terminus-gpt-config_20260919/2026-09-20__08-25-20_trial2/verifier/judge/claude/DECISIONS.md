# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all sessions by globbing for `*/*/suite2p/plane0/F.npy` under the data root, returning sorted session paths. For each session, it loads: `F.npy` and `Fneu.npy` (raw and neuropil fluorescence) from `suite2p/plane0/`, `ops.npy` (Suite2p parameters), `iscell.npy` (cell classification), `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. Neural data is memory-mapped for efficiency.

ii.
```python
def discover_sessions():
    """Return dated session paths in subject/date order."""
    return sorted(p.parents[2] for p in DATA_ROOT.glob('*/*/suite2p/plane0/F.npy'))

# In baseline_correct_and_bin:
F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
Fneu = np.load(session/'suite2p/plane0/Fneu.npy', mmap_mode='r')
iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')

# In repair_motion:
motion = np.load(session/'move_deve/motion_energy_glob.npy').astype(np.float64)
stamps = np.load(session/'move_deve/tstamps.npy').astype(np.float64)
```

iii. The AI's trajectory shows it explored the data directory structure in Step 2, identifying Suite2p outputs and behavioral files. It chose to load `tstamps.npy` instead of `interframe_int.npy` because it found that local timestamp differences (from `np.diff(tstamps)`) were more reliable for detecting dropped frames than the raw interframe intervals, since the latter had a confusing unit scale.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the parent directory names of discovered session paths, sorted alphabetically as a unique set.

ii.
```python
subjects = sorted({s.parent.name for s in sessions})
# ...
subject_idx.append(subjects.index(s.parent.name))
```

iii. Each `jm*` directory represents one mouse. The AI identified 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046) from the directory structure.

## 1-c. How are the data split into sessions?

i. Each dated subdirectory within a subject folder is one session. Sessions are discovered via the glob pattern and sorted by path (which sorts by subject then date).

ii.
```python
def discover_sessions():
    return sorted(p.parents[2] for p in DATA_ROOT.glob('*/*/suite2p/plane0/F.npy'))
```

iii. The AI noted that each subdirectory contains one daily recording with suite2p output and motion energy files. Sorting by path ensures deterministic subject-then-date ordering.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive non-overlapping 60-second segments of the continuous recording. After 10-frame binning (30 Hz → 3 Hz), each trial is 180 time bins. Any remainder bins that don't fill a complete trial are discarded (though in practice all sessions divide evenly).

ii.
```python
TRIAL_T = int(TRIAL_SECONDS * ANALYSIS_HZ)  # 60 * 3 = 180

n_trials = n_bins // TRIAL_T
use = n_trials * TRIAL_T
# ...
for tr in range(n_trials):
    sl = slice(tr*TRIAL_T, (tr+1)*TRIAL_T)
    neural_trials.append(np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32))
```

iii. Per the task instructions ("Split sessions into 60-second trials"), the AI constructs fixed-length trials from the continuous recording. The CONVERSION_NOTES document that all sessions divide exactly into 20 or 30 complete trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second trials are retained. The only data loss is any remainder bins at the end of a session that don't fill a complete trial (which is zero for all sessions in practice).

ii. N/A — no filtering code.

iii. The AI noted there is no natural trial structure or trial-level quality metric in this continuous recording dataset. The CONVERSION_NOTES confirm all native frame counts divide evenly by 10×180, so no incomplete trials exist.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and processing parameters from `ops.npy` (neucoeff, baseline method, sigma, window size, sampling rate).

ii.
```python
F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
Fneu = np.load(session/'suite2p/plane0/Fneu.npy', mmap_mode='r')
# ...
ops = np.load(session/'suite2p/plane0/ops.npy', allow_pickle=True).item()
neucoeff = float(ops.get('neucoeff', 0.7))
```

iii. These are the standard Suite2p output files. The AI also loads `iscell.npy` for validation (asserting all cells pass the >0.5 threshold) but does not use it for filtering since the distributed data is already pre-curated.

## 2-b. How is the `neural` data processed?

i. The AI reimplements the Track2p/Suite2p baseline correction pipeline using scipy filters:
1. Neuropil subtraction: `Fc = F - 0.7 * Fneu`
2. Gaussian smoothing: `gaussian_filter1d(fc, sigma=10, axis=1, mode='reflect')`
3. Minimum filter: `minimum_filter1d(flow, size=1800, axis=1, mode='reflect')` (60s × 30Hz = 1800 frames)
4. Maximum filter: `maximum_filter1d(flow, size=1800, axis=1, mode='reflect')`
5. Baseline subtraction: `corrected = fc - flow`
6. Non-overlapping 10-frame averaging to 3 Hz

ii.
```python
for lo in range(0, n_neurons, CHUNK_NEURONS):
    hi = min(n_neurons, lo + CHUNK_NEURONS)
    fc = np.asarray(F[lo:hi], dtype=np.float32) - neucoeff * np.asarray(Fneu[lo:hi], dtype=np.float32)
    flow = gaussian_filter1d(fc, sigma=sigma, axis=1, mode='reflect')
    flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
    flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
    corrected = fc - flow
    out[lo:hi] = corrected.reshape(hi-lo, -1, AVERAGE_FRAMES).mean(axis=2)
```

iii. The AI inspected the Track2p GUI's `F_processing` method (in `track2p/gui/data_management.py`) and found it uses scipy filters directly rather than calling `dcnv.preprocess`. The AI chose to replicate this exact implementation because: (1) it follows the actual reference code provided, (2) `dcnv.preprocess` might include additional steps (like division by baseline) that the reference code does not perform, and (3) the paper states "baseline corrected fluorescence traces using default Suite2p parameters" which Track2p implements as subtraction only, not division. Processing parameters are read from each session's `ops.npy` rather than hardcoded.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI loads `iscell.npy` and asserts all distributed rows have probability > 0.5, but does not filter (since they all pass). The data files are already pre-curated by Track2p's all-day matching pipeline.

ii.
```python
iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')
if not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f'Distributed rows fail reference iscell criterion in {session}')
```

iii. The AI found that the distributed data is already filtered and all-day matched by Track2p. Every row in every session passes the >0.5 iscell threshold. The assertion serves as a validation check rather than a filter, ensuring the assumption holds.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The first trial starts at the beginning of the session.

ii.
```python
'temporal_alignment_event': 'session start; trials are consecutive non-overlapping 60-second windows',
'off_start': 0.0, 'off_end': 60.0,
```

iii. There is no stimulus event to align to in this spontaneous activity dataset. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and behavioral data are averaged in non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bins). This matches the paper's description: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

ii.
```python
AVERAGE_FRAMES = 10
ANALYSIS_HZ = NATIVE_HZ / AVERAGE_FRAMES  # 3 Hz
# ...
out[lo:hi] = corrected.reshape(hi-lo, -1, AVERAGE_FRAMES).mean(axis=2)
# ...
motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1)
# ...
'time_bin_size': 1000.0 / ANALYSIS_HZ,  # 333.333 ms
```

iii. The 10-frame averaging is explicitly described in the paper's methods. Both streams are binned together so they remain aligned. Binning precedes discretization because averaging class labels would be meaningless.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed analytically from the bin index, the number of frames per bin (10), and the sampling rate (30 Hz). The value represents the bin center in seconds from session start.

ii.
```python
elapsed = ((np.arange(n_bins, dtype=np.float64) * AVERAGE_FRAMES + (AVERAGE_FRAMES-1)/2) / NATIVE_HZ).astype(np.float32)
```

iii. Since the frame rate is constant at 30 Hz, computing time from bin indices is equivalent to using stored timestamps, and avoids any clock drift issues.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes elapsed time as the **bin center** of each 10-frame averaging window. For bin `i`, the center is at native frame `i*10 + 4.5`, giving `(i*10 + 4.5)/30` seconds. This produces values starting at 0.15 s and incrementing by 1/3 s. Time does not reset per trial — it continues from session start across all trials.

ii.
```python
elapsed = ((np.arange(n_bins, dtype=np.float64) * AVERAGE_FRAMES + (AVERAGE_FRAMES-1)/2) / NATIVE_HZ).astype(np.float32)
# For trial slicing:
input_trials.append(np.ascontiguousarray(elapsed[None, sl], dtype=np.float32))
```

iii. The AI chose bin centers rather than left edges because they represent the actual temporal midpoint of the averaged data. The 4.5-frame offset (half of the 10-frame bin) avoids a systematic 167 ms bias.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices used for neural and behavioral data, so alignment is inherent. Each time value corresponds exactly to one analysis bin.

ii. N/A — alignment is implicit in the shared indexing.

iii. Since all three data streams (neural, time input, motion output) share the same bin indices, no separate alignment step is needed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (pre-computed global motion energy) and `tstamps.npy` (camera timestamps for detecting dropped frames), both from the `move_deve/` subdirectory of each session.

ii.
```python
motion = np.load(session/'move_deve/motion_energy_glob.npy').astype(np.float64)
stamps = np.load(session/'move_deve/tstamps.npy').astype(np.float64)
```

iii. The motion energy file contains a pre-computed signal (sum of squared pixel differences between consecutive video frames). The timestamps are used to detect and repair dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps:
1. **Dropped frame repair**: Detect missing frames via timestamp gaps > 1.5× median interval; insert linearly interpolated values using `np.interp` to match neural data length
2. **10-frame averaging**: Same binning as neural data
3. **Discretization**: Compute session-specific 20th/40th/60th/80th percentile edges using `np.quantile`, then assign class labels 0-4 using `np.searchsorted(..., side='right')`

ii.
```python
# Repair:
dt = np.diff(stamps)
med = float(np.median(dt))
large = dt > 1.5 * med
steps[large] = np.maximum(1, np.rint(dt[large] / med).astype(np.int64))
positions = np.r_[0, np.cumsum(steps)]
repaired = np.interp(np.arange(n_neural), positions, motion)

# Bin:
motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1)

# Discretize:
edges = np.quantile(motion_binned.astype(np.float64), [.2, .4, .6, .8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. The AI chose the timestamp-based repair approach because it found (through empirical testing) that absolute timestamp rounding accumulated clock drift, while local interval ratios reliably identified actual dropped frames. The interpolation using `np.interp` handles multi-frame gaps correctly and is vectorized. The discretization uses per-session quantiles as specified in the instructions ("five equal-percentile bins, selected per session").

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes four quantile edges at the 20th, 40th, 60th, and 80th percentiles of each session's binned motion energy, then assigns integer class labels 0-4 using `np.searchsorted` with `side='right'`.

ii.
```python
edges = np.quantile(motion_binned.astype(np.float64), [.2, .4, .6, .8])
labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. The `side='right'` parameter means values equal to a quantile boundary go into the higher bin. The AI verified that every session produces exactly 20% per class, confirming the equal-percentile requirement. The AI noted that ties at boundaries could cause imbalance in principle but chose to preserve exact values rather than jitter.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Video and neural data are acquired synchronously at 30 Hz (microscope triggers camera). When video frames are dropped, the motion energy array is shorter than neural data. The AI detects dropped frames via timestamp gaps exceeding 1.5× the session median interval and inserts linearly interpolated values. An assertion verifies the repaired length matches neural length exactly. Both streams then undergo identical 10-frame averaging.

ii.
```python
motion_repaired, motion_raw, stamps, inserted = repair_motion(session, n_native)
motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1)
if neural_binned.shape[1] != len(motion_binned):
    raise AssertionError(f'Post-bin alignment mismatch in {sid}')
```

iii. The AI verified that for every session with a behavior deficit, the timestamp-inferred insertions exactly equal the neural-behavior frame count difference. Equal-length sessions are left unchanged. The AI specifically tested that original observed values are preserved at their reconstructed indices using `np.allclose`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data issue is dropped camera frames (behavior shorter than neural data in 9 of 41 sessions). The AI handles this by:
1. Detecting gaps via local timestamp intervals > 1.5× median
2. Computing exact insertion positions and counts from the interval ratios
3. Asserting that inferred insertions equal the exact deficit
4. Linearly interpolating missing values using `np.interp`
5. Asserting repaired length matches neural length

Additionally, remainder bins that don't fill a complete 60-second trial would be discarded, though in practice all sessions divide evenly.

ii.
```python
if len(motion) < n_neural:
    # ... detect and repair
    if inserted != deficit or positions[-1] != n_neural - 1:
        raise ValueError(f'Behavior gaps do not explain deficit in {session}')
    repaired = np.interp(np.arange(n_neural), positions, motion)
elif len(motion) == n_neural:
    repaired = motion
else:
    repaired = motion[:n_neural]  # defensive; no session takes this branch
```

iii. The AI's approach is conservative: it only inserts frames when the timestamp gaps exactly account for the deficit. If they don't match, it raises an error rather than silently producing misaligned data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the scipy-based baseline correction (Gaussian filter + min/max filters over the full session length for every neuron). The full conversion of 41 sessions completed in 40.67 seconds.

ii. N/A — timing is printed per-session in the conversion output, showing 0.29-1.58 seconds per session.

iii. The baseline correction involves three sequential 1D filter passes over each neuron's full time series. Processing in 64-neuron chunks with memory mapping mitigates memory overhead but the computation remains the bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop creates individual numpy arrays for each trial. This could theoretically be replaced with a single reshape operation for sessions where all trials have the same length (which they do). However, the output format requires a list of separate arrays, so this is more of a structural constraint than a performance issue.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr*TRIAL_T, (tr+1)*TRIAL_T)
    neural_trials.append(np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32))
    input_trials.append(np.ascontiguousarray(elapsed[None, sl], dtype=np.float32))
    output_trials.append(np.ascontiguousarray(labels[None, sl], dtype=np.int64))
```

iii. The AI's code is already well-vectorized: baseline correction processes 64 neurons at once, binning uses reshape+mean, and behavior repair uses `np.interp`. The main remaining loop is trial slicing which is inherent to the output format.

## 6-c. What processing does the code repeat multiple times?

i. The AI loads `F.npy` twice per session: once in `baseline_correct_and_bin` (memory-mapped for processing) and once in `convert_session` to get the native frame count.

ii.
```python
# In baseline_correct_and_bin:
F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
# In convert_session:
n_native = int(np.load(session/'suite2p/plane0/F.npy', mmap_mode='r').shape[1])
```

iii. The second load is only to get the shape (memory-mapped, so minimal overhead), but the frame count could have been returned from `baseline_correct_and_bin` instead.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes detailed per-session metadata (`session_info`) including repair counts, quantile edges, class counts, and duration information that is stored in the pickle but not used by the decoder. The `iscell.npy` loading and assertion is also not strictly necessary for the conversion output but serves as validation.

ii.
```python
info = {'session_id': sid, 'subject': session.parent.name, 'native_frames': n_native,
        'native_behavior_samples': int(len(motion_raw)), 'inserted_behavior_samples': int(inserted),
        'analysis_bins': int(n_bins), 'n_trials': int(n_trials), 'n_neurons': int(n_neurons),
        'motion_quantile_edges': [float(v) for v in edges],
        'motion_class_counts': [int(v) for v in counts],
        'duration_seconds_nominal': float(n_native/NATIVE_HZ)}
```

iii. While this metadata is not used by the decoder, it provides valuable audit information and was recommended by the task instructions ("Add other relevant fields").
