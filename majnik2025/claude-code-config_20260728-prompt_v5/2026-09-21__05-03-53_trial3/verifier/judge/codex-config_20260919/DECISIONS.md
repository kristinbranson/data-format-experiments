# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory under `/app/data` as a subject and every directory below each subject as a session. For each session it loads Suite2p `F.npy` and `Fneu.npy`, plus `motion_energy_glob.npy` and `tstamps.npy`. Full mode processes every discovered session; sample mode limits processing.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))])
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The notes describe the known hierarchy as six mouse directories containing daily Suite2p and behavioral recordings and report that full conversion found all 6 mice and 41 sessions.

## 1-b. How are the data split into subjects?

i. Each sorted top-level directory is treated as one mouse. Its index in the full `subjects` list is assigned to every session from that directory.

ii.
```python
for subj in subjects_to_process:
    subj_idx = subjects.index(subj)
    ...
    all_subject_idx.append(subj_idx)
```

iii. The agent identified the six directories `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` as mice and validated their per-mouse neuron and session counts.

## 1-c. How are the data split into sessions?

i. Sorted subdirectories beneath each mouse are daily sessions. Each becomes one outer-list entry in `neural`, `input`, and `output`.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))])
...
for sess_name in sessions:
    result = process_session(...)
    all_neural.append(result['neural'])
```

iii. The notes say these directories are daily recordings and that the resulting 41 sessions reproduce the source counts (7, 7, 7, 7, 6, 7).

## 1-d. How are the data split into trials?

i. After 10-frame binning, each session is divided into non-overlapping 60-second trials of 180 bins. A final incomplete segment is omitted by integer division.

ii.
```python
TRIAL_BINS = int(TRIAL_DURATION_S / (BIN_SIZE / FRAME_RATE))
n_trials = n // trial_length
for i in range(n_trials):
    slices[axis] = slice(i * trial_length, (i + 1) * trial_length)
    trials.append(data[tuple(slices)])
```

iii. The decoder task explicitly requests 60-second trials. The agent checked that 20- and 30-minute sessions yield 20 and 30 trials, respectively.

## 1-e. How are trials filtered based on quality controls?

i. No complete trial is filtered. Only a trailing segment shorter than 60 seconds would be discarded.

ii.
```python
n_trials = n // trial_length
```

iii. The notes found no trial-curation rule in the spontaneous-behavior experiment, so the agent retained all full-length trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from Suite2p raw somatic fluorescence `F.npy` and neuropil fluorescence `Fneu.npy` from `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The agent chose baseline-corrected fluorescence rather than `spks.npy` because the paper describes that signal for decoding.

## 2-b. How is the `neural` data processed?

i. The code subtracts 0.7 times neuropil fluorescence, estimates a Suite2p-style maximin baseline after Gaussian smoothing (60-second window, sigma 10), subtracts that baseline, casts through float32, and averages every 10 consecutive frames.

ii.
```python
Fc = (F - NEUCOEFF * Fneu).astype(np.float32)
...
data = conv1d(...)
data = -max_pool1d(-data, kernel_size=win, stride=1, padding=0)
data = max_pool1d(data, kernel_size=win, stride=1, padding=0)
...
dff = Fc - Flow
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. The trajectory explicitly resolves that Suite2p `preprocess` returns baseline-subtracted fluorescence, not division by baseline, and the notes report an exact check against Suite2p preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No runtime ROI filtering is applied; every row in `F.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
```

iii. The agent found that the supplied matrices already contain neurons tracked across all days and already passing the Suite2p `iscell > 0.5` selection, so further filtering would duplicate upstream curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Trials are consecutive windows anchored to session start, described in metadata as spontaneous activity with no task events.

ii.
```python
'temporal_alignment_event': 'Session start (spontaneous activity, no task events)',
'off_start': None,
'off_end': None,
```

iii. The notes explain that these are artificial fixed-duration trials in continuous spontaneous recordings.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are averaged per non-overlapping bin, producing 3-Hz samples and a 333.33-ms bin size. Short tails below ten frames are truncated.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000
return data_trunc.reshape(new_shape).mean(axis=-1)
```

iii. This follows the paper's decoding analysis, which averages both fluorescence and behavior over 10 timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the binned sample index, the 10-frame bin width, and the assumed 30-Hz frame rate; no raw timestamp array is used.

ii.
```python
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)
```

iii. The agent reasoned that a constant acquisition rate makes index-derived elapsed time sufficient and avoids the unusual units seen in behavioral timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Zero-based bin indices are multiplied by 10/30 seconds, reshaped to one feature by time, split into 180-bin trials, and cast to float32. Values continue across trial boundaries rather than resetting.

ii.
```python
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)
time_trials = split_into_trials(time_s.reshape(1, -1), TRIAL_BINS, axis=1)
'input': [t.astype(np.float32) for t in time_trials]
```

iii. The notes define the required input as elapsed seconds from session start and validate trial 1 ending at 59.67 s and trial 2 starting at 60 s.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One time value is generated for each binned neural column, and both arrays are sliced by the same 180-bin trial boundaries.

ii.
```python
n_timebins = dff_binned.shape[1]
time_s = np.arange(n_timebins) * (BIN_SIZE / FRAME_RATE)
neural_trials = split_into_trials(dff_binned, TRIAL_BINS, axis=1)
time_trials = split_into_trials(time_s.reshape(1, -1), TRIAL_BINS, axis=1)
```

iii. The agent's sanity checks report correct trial boundary times and equal trial dimensions.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output comes from precomputed `move_deve/motion_energy_glob.npy`. The code also loads `tstamps.npy`, but only passes it into a repair function that does not actually use its values.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
me_interp = interpolate_motion_energy(me, tstamps, n_frames)
```

iii. The notes identify global motion energy as the behavioral signal and timestamps as relevant to occasional missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If lengths differ, the code linearly resamples the entire motion-energy trace from evenly spaced source indices to the neural frame count. It then averages 10 frames per bin, calculates session-local percentile edges, digitizes into five classes, splits into trials, and casts to int64.

ii.
```python
camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
me_interp = np.interp(neural_indices, camera_indices, me_float)
...
me_binned = bin_data(me_interp.reshape(1, -1), BIN_SIZE, axis=1).flatten()
edges = np.percentile(me_binned, np.linspace(0, 100, n_bins + 1))
binned = np.digitize(me_binned, edges[1:-1], right=False)
```

iii. The agent intended interpolation to restore dropped frames before joint binning. It states that this preserves matching lengths, although its implementation globally stretches the signal rather than locating individual drops.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The already binned continuous values are thresholded separately per session at the 20th, 40th, 60th, and 80th percentiles. `np.digitize` produces labels 0–4, followed by clipping.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(me_binned, percentiles)
binned = np.digitize(me_binned, edges[1:-1], right=False)
binned = np.clip(binned, 0, n_bins - 1)
```

iii. This directly implements the requested five equal-percentile bins selected per session; validation found approximately 20% in each class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Equal-length recordings pass through unchanged. Short motion traces are globally linearly resampled to the neural frame count using `linspace`, after which both streams are separately averaged in equal 10-frame windows and split at identical trial lengths. Actual timestamps and missing-frame locations are ignored.

ii.
```python
neural_indices = np.arange(n_neural_frames)
camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
me_interp = np.interp(neural_indices, camera_indices, me_float)
```

iii. The agent justified this from synchronous 30-Hz microscope-triggered video and considered the streams approximately one-to-one, but chose a simple global interpolation for mismatches.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion-energy arrays are interpolated to the neural length; equal-length arrays are cast and retained. Incomplete temporal bins and incomplete 60-second trials are truncated. There is no explicit behavior for a motion array longer than neural data, NaNs, absent files, or corrupt sessions.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
return np.interp(neural_indices, camera_indices, me_float)
```

iii. The notes report camera-frame mismatches and say interpolation was selected to restore alignment. They also validated dimensions and one two-frame-short session.

## 6-a. What are the most time-consuming steps of the code?

i. Maximin baseline computation is the dominant transform; it applies convolution plus two 1,801-frame pooling passes to every neuron on CPU. Loading and serializing the roughly 395-MB result are secondary costs, and optional plotting adds work.

ii.
```python
for n in range(n_batches):
    ...
    data = conv1d(...)
    data = -max_pool1d(-data, kernel_size=win, stride=1, padding=0)
    data = max_pool1d(data, kernel_size=win, stride=1, padding=0)
```

iii. The notes identify dF/F computation as the principal per-session cost and report batching 100 neurons at a time; CPU was selected for reliability.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial extraction loops over trials even though a truncate-and-reshape operation could create trial views in one operation. The session and neuron-batch loops are less readily removable because of file boundaries and memory. Summary lookup also repeatedly scans session indices.

ii.
```python
for i in range(n_trials):
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(i * trial_length, (i + 1) * trial_length)
    trials.append(data[tuple(slices)])
```

iii. The notes only call out vectorized binning and batched baseline processing as implemented speedups; they do not discuss the remaining trial loop.

## 6-c. What processing does the code repeat multiple times?

i. Each session independently imports/initializes the PyTorch baseline machinery, reconstructs the Gaussian kernel, and computes identical maximin parameters. Neural, output, and time streams also traverse the same trial slicing loop separately.

ii.
```python
def compute_dfof(...):
    import torch
    from torch.nn.functional import conv1d, max_pool1d, pad
    ...
    gaussian = torch.exp(...)
...
neural_trials = split_into_trials(...)
me_trials = split_into_trials(...)
time_trials = split_into_trials(...)
```

iii. The agent's documentation does not identify repeated processing; it emphasizes per-session operation and reports the runtime as acceptable.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `tstamps.npy` is loaded but its values are never used. `n_trials` is computed in `process_session` only for logging even though splitting recomputes it. With `--show-processing`, large diagnostic figures and intermediate traces are produced but not used by the saved dataset or decoder. The summary variable `total_neurons` is calculated but never printed or saved.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
...
camera_indices = np.linspace(0, n_neural_frames - 1, len(me))
...
total_neurons = sum(len(br) for br in all_brain_region_idx)
```

iii. The notes justify optional plots as visual sanity checks, but do not acknowledge the unused timestamp contents, duplicate count, or unused summary value.
