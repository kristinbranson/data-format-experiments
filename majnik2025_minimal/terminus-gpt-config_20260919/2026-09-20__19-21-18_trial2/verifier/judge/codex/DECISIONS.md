# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all subjects by scanning `/app/data` for directories whose names start with `jm`, then loads every subdirectory under each subject as a session. Within each session it loads calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy`, Suite2p parameters from `ops.npy`, and behavior from `move_deve/motion_energy_glob.npy` plus `tstamps.npy`. Trials are not loaded directly; the session-level arrays are loaded first and trialized later.

ii. 
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
...
sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
...
F = np.load(p / 'F.npy').astype(np.float32, copy=False)
Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
...
motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(p / 'tstamps.npy').astype(np.float64)
```

iii. In trajectory steps 4-5 and 7-11, the agent said it inspected the README, notebook, methods, and file inventory to determine whether to use `F`, `Fneu`, `spks`, timestamps, and motion-energy files. It concluded that the reference processing should start from raw fluorescence, that camera timestamps were needed for dropped-frame handling, and that all supplied `jm*` subject/session folders should be retained.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined as top-level directories under `/app/data` whose names begin with `jm`. They are sorted alphabetically and stored in that order in `subjects`.

ii. 
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
```

iii. In steps 4, 8, 9, and 12, the agent stated that the dataset contains six subject folders and that all of them should be kept. The alphabetical sort gives a deterministic mouse ordering.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory under a subject directory. Sessions are sorted alphabetically within each subject and appended in that order to the output lists.

ii. 
```python
for si, subject in enumerate(subjects):
    sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
    for session in sessions:
        print(f'Processing {subject}/{session.name}', flush=True)
```

iii. In steps 4, 8, 9, and 12, the agent described each daily recording subdirectory as one session and noted that there were 41 sessions total. The sorted traversal is the mechanism it used to realize that decision.

## 1-d. How are the data split into trials?

i. The AI treats the recording as continuous and creates artificial trials by chopping each session into consecutive non-overlapping 60-second segments after temporal averaging. Because the averaged sampling rate is 3 Hz, each trial contains 180 time bins. Any leftover bins at the end of a session are dropped by truncating to the largest multiple of 180.

ii. 
```python
AVERAGE_FRAMES = 10
BIN_FS = FS / AVERAGE_FRAMES
TRIAL_SECONDS = 60
TRIAL_SAMPLES = int(TRIAL_SECONDS * BIN_FS)
...
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
neural.append([np.ascontiguousarray(x, dtype=np.float32)
               for x in np.split(activity, ntrials, axis=1)])
inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
               for x in np.split(elapsed, ntrials)])
outputs.append([np.ascontiguousarray(x[None, :], dtype=np.int64)
                for x in np.split(labels, ntrials)])
```

iii. In steps 8-12, the agent said the reference decoding analysis uses non-overlapping 10-frame averages, which makes 60-second trials equal to 180 samples. It also said sessions should be split into complete 60-second trials only.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-level quality-control filter. The only effective exclusion is that trailing bins that do not fill a complete 60-second trial are discarded before splitting.

ii. 
```python
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
```

iii. In steps 7-11, the agent focused quality control on neuron inclusion and dropped camera frames, not on per-trial rejection. No trajectory step mentions any trial-quality filter beyond keeping complete trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`. `ops.npy` is also read to recover the Suite2p preprocessing parameters such as `neucoeff`, `sig_baseline`, `win_baseline`, and `fs`.

ii. 
```python
F = np.load(p / 'F.npy').astype(np.float32, copy=False)
Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
neucoeff = float(ops.get('neucoeff', 0.7))
corrected = F - neucoeff * Fneu
```

iii. In steps 4-5 and 10-11, the agent said it intentionally chose raw fluorescence plus neuropil fluorescence rather than `spks.npy`, because the task asked for the same preprocessing described in the paper rather than using already deconvolved traces.

## 2-b. How is the `neural` data processed?

i. Neural activity is processed by computing neuropil-corrected fluorescence `F - neucoeff * Fneu`, then applying a manual reimplementation of Suite2p's maximin baseline subtraction: Gaussian smoothing, rolling minimum, rolling maximum, and baseline subtraction. After that, the trace is averaged in non-overlapping groups of 10 frames.

ii. 
```python
neucoeff = float(ops.get('neucoeff', 0.7))
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

iii. In steps 7-11, the agent said the methods called for "Suite2p-default baseline-corrected dF/F" and that inspecting Suite2p source confirmed this meant baseline subtraction after neuropil correction, not division by baseline. It also noted that both neural and behavioral traces should be averaged over 10 consecutive timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural filtering is applied in code. All neurons present in the provided Suite2p outputs are retained. The AI explicitly decided not to apply a second `iscell`-based filter.

ii. 
```python
for session in sessions:
    ...
    activity = suite2p_baseline_corrected(session)
    ...
    region_idx.append(np.zeros(activity.shape[0], dtype=np.int64))
```

iii. In steps 7, 10, and 11, the agent said the Track2p exports already contain tracked cells and that the observed `iscell` scores exceed the paper's default 0.5 threshold, so additional ROI filtering was unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to session start rather than to any stimulus or behavioral event. Trials are consecutive windows along that session-start-aligned timeline.

ii. 
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
...
'metadata': {
    ...
    'temporal_alignment_event': 'session start; consecutive 60-second trial segmentation',
    'off_start': 0.0,
    'off_end': float(TRIAL_SECONDS),
    ...
}
```

iii. In steps 8-12, the agent said the recordings are continuous and that behavior should be aligned to the neural frame grid, then split into fixed 60-second trials. That implies session-start alignment rather than event-triggered alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 333.333... ms bins, obtained by averaging non-overlapping groups of 10 frames from the original 30 Hz sampling. This rebinning is applied to both neural and motion data before trialization and before motion-energy discretization.

ii. 
```python
FS = 30.0
AVERAGE_FRAMES = 10
BIN_FS = FS / AVERAGE_FRAMES
...
activity = average_ten(activity).astype(np.float32)
motion = average_ten(motion)
...
'time_bin_size': 1000.0 / BIN_FS,
```

iii. In steps 7-12, the agent repeatedly justified this with the methods statement that decoding analyses denoised both neural and behavioral traces by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a raw file. It is synthesized from the index of each binned time point and the post-binning sample rate, yielding elapsed seconds from the start of the session.

ii. 
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
               for x in np.split(elapsed, ntrials)])
```

iii. In steps 8, 11, and 12, the agent said it would "retain global elapsed session time as input" and that this variable should remain relative to session start rather than resetting on each trial.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After neural activity is binned to 3 Hz, the AI constructs a uniformly spaced time vector with `np.arange(...) / BIN_FS`. It then truncates the vector to complete trials and splits it with the same trial boundaries as the neural data.

ii. 
```python
activity = average_ten(activity).astype(np.float32)
...
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
...
elapsed = elapsed[:usable]
inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
               for x in np.split(elapsed, ntrials)])
```

iii. In steps 8 and 11, the agent justified this as the natural consequence of 10-frame averaging and of keeping elapsed time on the same 3 Hz grid as the other signals.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is exactly aligned with neural data because it is generated after the neural trace has been binned, then truncated to the same usable length, and split with the same `np.split` boundaries.

ii. 
```python
activity = average_ten(activity).astype(np.float32)
motion = average_ten(motion)
labels = quintiles(motion)
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
...
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
...
for x in np.split(activity, ntrials, axis=1)
for x in np.split(elapsed, ntrials)
```

iii. In step 11, the agent explicitly said it would "retain global elapsed session time as input" while splitting into the same 60-second trials as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, and its temporal alignment is derived from `move_deve/tstamps.npy`. The AI does not use `interframe_int.npy`.

ii. 
```python
motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(p / 'tstamps.npy').astype(np.float64)
```

iii. In steps 4, 5, 8, and 11, the agent said that dropped camera frames should be handled from the timestamp stream because the timestamp gaps locate missing frames directly, and because some sessions showed gap structure that made timestamp-based alignment preferable to simple length padding.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates the motion-energy trace from irregular camera timestamps onto a regular imaging-frame grid whose spacing is the median positive timestamp difference. It then averages the aligned trace in groups of 10 frames and discretizes the averaged values into five within-session quintiles.

ii. 
```python
positive_dt = np.diff(stamps)
step = np.median(positive_dt[positive_dt > 0])
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
...
motion = average_ten(motion)
labels = quintiles(motion)
```

iii. In steps 8-11, the agent justified this by saying the timestamp clock is not in seconds but still provides the correct relative frame grid, that interpolation is preferable to naive padding, and that the paper's decoding analysis averages neural and behavioral traces before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes the 20th, 40th, 60th, and 80th percentiles of the binned motion-energy trace separately within each session, then assigns quintile labels 0-4 using those four cut points.

ii. 
```python
def quintiles(x: np.ndarray) -> np.ndarray:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. In steps 9-12, the agent said motion-energy quantiles were well separated and that the requested categorical target should be built from five per-session percentile bins after averaging.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by interpolating the camera-derived motion trace onto a target vector of length `nframes`, where `nframes` is the raw neural frame count. After that raw-frame alignment, both streams are averaged with the same 10-frame bins and split into trials with the same boundaries.

ii. 
```python
activity = suite2p_baseline_corrected(session)
motion = aligned_motion(session, activity.shape[1])
activity = average_ten(activity).astype(np.float32)
motion = average_ten(motion)
...
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
```

iii. In steps 8 and 11, the agent explicitly said it would "interpolate motion energy onto the 30 Hz neural grid using camera timestamps" and then average both streams in non-overlapping 10-frame bins so that they stay aligned.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles camera irregularities by using timestamps and interpolation rather than assuming the motion-energy array is already on the correct frame grid. It raises an error if motion and timestamp arrays disagree in length. It also discards any end-of-session remainder that does not fit a full 60-second trial.

ii. 
```python
if len(motion) != len(stamps):
    raise ValueError(f'motion/timestamp mismatch in {session}')
...
positive_dt = np.diff(stamps)
step = np.median(positive_dt[positive_dt > 0])
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
...
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
```

iii. In steps 4, 8, and 11, the agent said the dataset contains occasional dropped camera frames and that timestamp-based interpolation is the right way to correct them. In steps 8 and 12, it also emphasized keeping only complete 60-second trials.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work is the session-level neural preprocessing: loading the large fluorescence arrays and running the Suite2p-like baseline correction (Gaussian smoothing plus rolling min/max filters) across all neurons and all frames. Motion interpolation and per-session averaging are also linear-time over long session traces, but smaller than the calcium preprocessing.

ii. 
```python
F = np.load(p / 'F.npy').astype(np.float32, copy=False)
Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
...
smooth = gaussian_filter1d(corrected, float(ops.get('sig_baseline', 10.0)),
                           axis=1, mode='reflect')
base = minimum_filter1d(smooth, size=window, axis=1, mode='reflect')
base = maximum_filter1d(base, size=window, axis=1, mode='reflect')
...
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
```

iii. The trajectory does not contain an explicit runtime profile, but steps 10-11 show the agent concentrating on exact reproduction of Suite2p preprocessing as the main algorithmic component, which is the clear computational bottleneck in the written code.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already kept the heavy per-time operations vectorized. The remaining Python loops are at the subject/session level and for packaging session arrays into per-trial lists. Compared with the human reference, the AI specifically avoided a Python loop over dropped frames by using one `np.interp` call.

ii. 
```python
for si, subject in enumerate(subjects):
    sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
    for session in sessions:
        ...
        neural.append([np.ascontiguousarray(x, dtype=np.float32)
                       for x in np.split(activity, ntrials, axis=1)])
        inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
                       for x in np.split(elapsed, ntrials)])
        outputs.append([np.ascontiguousarray(x[None, :], dtype=np.int64)
                        for x in np.split(labels, ntrials)])
...
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
```

iii. In steps 8 and 11, the agent explicitly chose timestamp-based interpolation onto the whole neural grid, which is the reason the code avoids a per-dropped-frame insertion loop.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated processing beyond the intended per-session repetition of the same preprocessing pipeline. Within each session, the code also creates contiguous per-trial copies after session-level arrays have already been computed, but it does not rerun neural or motion preprocessing multiple times on the same data.

ii. 
```python
for session in sessions:
    activity = suite2p_baseline_corrected(session)
    motion = aligned_motion(session, activity.shape[1])
    activity = average_ten(activity).astype(np.float32)
    motion = average_ten(motion)
    labels = quintiles(motion)
    ...
    neural.append([np.ascontiguousarray(x, dtype=np.float32)
                   for x in np.split(activity, ntrials, axis=1)])
```

iii. No trajectory step identifies any deliberate repeated computation as a design choice. The repetition that exists is the intended once-per-session processing pipeline.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is no large obviously wasted computation, but the AI does do some extra bookkeeping that is not needed for the decoder: it loads `ops.npy` only to recover preprocessing parameters, builds a `session_info` metadata list, and computes full-session arrays before truncating the leftover tail that does not make a complete trial.

ii. 
```python
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
...
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
...
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
...
session_info.append({
    'subject': subject, 'session': session.name,
    'n_neurons': int(activity.shape[0]), 'n_trials': ntrials,
    'duration_seconds': float(usable / BIN_FS),
    'source_imaging_rate_hz': FS,
})
```

iii. The trajectory does not justify these as downstream requirements. They appear to be convenience or provenance choices added by the AI while implementing the converter.
