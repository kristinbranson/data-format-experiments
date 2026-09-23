# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every sorted `jm*` subject directory and every sorted subdirectory beneath each subject. For each daily session it loads `F.npy`, `Fneu.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy`, processes the session, and appends it to the output.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
for si, subject in enumerate(subjects):
    sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
    for session in sessions:
        activity = suite2p_baseline_corrected(session)
        motion = aligned_motion(session, activity.shape[1])
```

iii. The trajectory says the README and inventory established that subject folders contain daily sessions with Suite2p tracked-cell data and behavior arrays. It explicitly chose to preserve all six mice and all 41 supplied sessions.

## 1-b. How are the data split into subjects?

i. A subject is a directory whose name starts with `jm`; subject names are sorted, and the enumeration index is recorded for every session in `subject_idx`.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
for si, subject in enumerate(subjects):
    ...
    subject_idx.append(si)
```

iii. The agent inferred from the documented directory structure and names that each `jm*` folder is one mouse. Sorting supplies deterministic ordering.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject is one session and becomes one element of each session-level output list.

ii.
```python
sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
for session in sessions:
    ...
    neural.append(...)
    inputs.append(...)
    outputs.append(...)
```

iii. The trajectory identifies these as daily recordings and reports 41 sessions total. This follows the supplied data organization.

## 1-d. How are the data split into trials?

i. After 10-frame averaging, each session is divided into consecutive, non-overlapping 60-second trials of 180 samples. Only complete trials are retained; the tail is truncated before `np.split`.

ii.
```python
TRIAL_SAMPLES = int(TRIAL_SECONDS * BIN_FS)
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
neural.append([np.ascontiguousarray(x, dtype=np.float32)
               for x in np.split(activity, ntrials, axis=1)])
```

iii. The agent states that the task explicitly requires 60-second trials and that 30 Hz data averaged by 10 gives 3 Hz, hence 180 samples per trial.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. All complete 60-second segments are retained, while any incomplete final segment is discarded.

ii.
```python
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
```

iii. No natural trials or trial-quality flags were found. The trajectory describes retaining all supplied data and only enforcing complete fixed-length segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from Suite2p `F.npy` and `Fneu.npy`; `ops.npy` supplies the neuropil and baseline parameters.

ii.
```python
F = np.load(p / 'F.npy').astype(np.float32, copy=False)
Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
neucoeff = float(ops.get('neucoeff', 0.7))
corrected = F - neucoeff * Fneu
```

iii. The agent inspected the notebook, methods, and Suite2p source and concluded that the paper used Suite2p-default baseline-corrected fluorescence, formed from neuropil-corrected `F` rather than the available `spks.npy` alternative.

## 2-b. How is the `neural` data processed?

i. The agent subtracts neuropil (`F - neucoeff*Fneu`), reproduces Suite2p's maximin baseline (Gaussian smoothing, rolling minimum, then rolling maximum) and subtracts it, then averages non-overlapping groups of 10 frames.

ii.
```python
corrected = F - neucoeff * Fneu
smooth = gaussian_filter1d(corrected, float(ops.get('sig_baseline', 10.0)),
                           axis=1, mode='reflect')
window = int(float(ops.get('win_baseline', 60.0)) * float(ops.get('fs', FS)))
base = minimum_filter1d(smooth, size=window, axis=1, mode='reflect')
base = maximum_filter1d(base, size=window, axis=1, mode='reflect')
corrected -= base
...
activity = average_ten(activity).astype(np.float32)
```

iii. The trajectory says the methods specified Suite2p-default baseline-corrected fluorescence and 10-timestamp denoising; inspection of Suite2p source confirmed that maximin is baseline subtraction rather than division.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional ROI filter is applied: all supplied rows are retained.

ii.
```python
# Every supplied subject/session is retained. Track2p's Suite2p exports already
# contain only cells tracked across every day of a subject. Their iscell scores
# exceed the paper's default 0.5 threshold, so no second ROI filter is applied.
```

iii. The agent checked the data documentation and `iscell` values and found that exports already contain cells tracked across days and exceed the paper's default cell-probability threshold, so another filter would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Samples remain on the session timeline and are cut into consecutive trials beginning at session start. Metadata calls this “session start; consecutive 60-second trial segmentation” and reports offsets 0 to 60 seconds.

ii.
```python
'temporal_alignment_event': 'session start; consecutive 60-second trial segmentation',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The agent reasoned that these are artificial segments of a continuous recording, so session start is the only alignment origin. It did not specifically justify why every trial receives the same metadata offsets despite elapsed time remaining session-relative.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Source data at 30 Hz are averaged in non-overlapping blocks of 10 frames, producing 3 Hz data and 333.33 ms bins for both neural and behavioral streams.

ii.
```python
AVERAGE_FRAMES = 10
BIN_FS = FS / AVERAGE_FRAMES
...
return x.reshape(*x.shape[:-1], n, AVERAGE_FRAMES).mean(axis=-1)
...
'time_bin_size': 1000.0 / BIN_FS,
```

iii. The trajectory quotes the paper's decoding procedure as averaging both dF/F and behavior over 10 consecutive timestamps and notes this keeps the streams synchronized.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from the binned sample index and the known 3 Hz sampling rate, not loaded from a raw variable.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
```

iii. The agent used the confirmed constant 30 Hz imaging rate and 10-frame averaging; a synthetic index clock is therefore equivalent to elapsed imaging time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A zero-based index for every binned session sample is divided by 3 Hz, then truncated with the other streams to complete trials and cast/retained as `float32`.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
...
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
```

iii. The trajectory emphasizes that elapsed time remains relative to session start rather than resetting at each artificial trial.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is created after neural averaging with exactly one value per neural bin, receives the identical usable-length truncation, and is split at the identical 180-sample boundaries.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
               for x in np.split(elapsed, ntrials)])
```

iii. The common sample count and split boundaries were chosen to maintain one-to-one temporal alignment with neural bins.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses the precomputed global motion energy in `motion_energy_glob.npy` and camera timestamps in `tstamps.npy` for alignment.

ii.
```python
motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(p / 'tstamps.npy').astype(np.float64)
```

iii. The trajectory says motion energy is the paper's sum of squared inter-frame pixel differences and that timestamp gaps identify dropped behavior-camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The precomputed trace is interpolated onto a regular imaging-length grid derived from camera timestamps, averaged over 10 frames, and converted to session-specific quintile labels.

ii.
```python
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
...
motion = average_ten(motion)
labels = quintiles(motion)
```

iii. The agent found dropped frames, including timestamp gaps in some equal-length arrays, and therefore preferred timestamp interpolation over padding. Ten-frame averaging follows the paper and occurs before categorical discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20th-percentile boundaries are calculated independently for each complete binned session. `searchsorted(..., side='right')` assigns integer classes 0–4.

ii.
```python
def quintiles(x: np.ndarray) -> np.ndarray:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. This directly implements the requested five equal-percentile bins selected per session. The agent checked that percentile ties were not problematic.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The median positive camera-timestamp increment defines the regular frame step. Motion is linearly interpolated at exactly `nframes` target locations beginning at the first camera timestamp, then neural and motion streams are averaged by the same factor and split identically.

ii.
```python
positive_dt = np.diff(stamps)
step = np.median(positive_dt[positive_dt > 0])
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
```

iii. The agent reasoned that timestamps encode exact missing-frame locations and observed gaps even where raw lengths matched. It judged timestamp interpolation onto the neural grid safer than shifting the remainder of the trace or padding at the end.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Motion/timestamp length mismatches raise an error. Dropped-camera gaps are repaired by timestamp-based linear interpolation; extrapolated endpoints use the nearest motion value. Incomplete trial tails are discarded.

ii.
```python
if len(motion) != len(stamps):
    raise ValueError(f'motion/timestamp mismatch in {session}')
...
np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
...
usable = ntrials * TRIAL_SAMPLES
```

iii. The README reportedly permitted treating dropped frames as missing or interpolating them. The agent selected interpolation to preserve framewise alignment and explicit failure for internally inconsistent source arrays.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session, per-neuron Gaussian/minimum/maximum baseline correction is the dominant computation; loading large arrays and materializing the converted dataset are secondary costs.

ii.
```python
smooth = gaussian_filter1d(corrected, ..., axis=1, mode='reflect')
base = minimum_filter1d(smooth, size=window, axis=1, mode='reflect')
base = maximum_filter1d(base, size=window, axis=1, mode='reflect')
```

iii. The trajectory inspected whether Suite2p preprocessing was available and describes baseline preprocessing over every neuron and full session as the key processing operation. The final implementation uses SciPy on CPU.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Session/subject loops are needed because shapes and files differ. Trial conversion uses Python list comprehensions around already vectorized `np.split`; these could be reduced by retaining views or restructuring storage, but the required nested-list format still requires per-trial objects. Dropped-frame repair itself is vectorized through `np.interp`.

ii.
```python
neural.append([np.ascontiguousarray(x, dtype=np.float32)
               for x in np.split(activity, ntrials, axis=1)])
inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
               for x in np.split(elapsed, ntrials)])
```

iii. The trajectory deliberately chose timestamp-based interpolation, avoiding a repeated `np.insert` loop. It did not identify further loops as bottlenecks.

## 6-c. What processing does the code repeat multiple times?

i. Loading, baseline correction, timestamp alignment, averaging, percentile calculation, and splitting repeat once per session. Contiguous conversion repeats once for each trial and each of the three streams. These repetitions reflect session-specific files, neuron counts, alignment, and quintile thresholds.

ii.
```python
for session in sessions:
    activity = suite2p_baseline_corrected(session)
    motion = aligned_motion(session, activity.shape[1])
    activity = average_ten(activity).astype(np.float32)
    motion = average_ten(motion)
    labels = quintiles(motion)
```

iii. The trajectory does not flag avoidable repeated processing; it stresses that quintiles must be selected per session and all sessions must be processed.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No major scientific processing is computed and then discarded. There are potentially unnecessary copies from `np.ascontiguousarray`, and rich `session_info` metadata is not required by decoder training, but both make the saved output robust and auditable.

ii.
```python
np.ascontiguousarray(x, dtype=np.float32)
...
session_info.append({
    'subject': subject, 'session': session.name,
    'n_neurons': int(activity.shape[0]), 'n_trials': ntrials,
    ...
})
```

iii. The trajectory reports validating and training the decoder but does not identify discarded conversion work. The copies ensure predictable layout/dtypes, and metadata documents provenance even if the decoder ignores it.
