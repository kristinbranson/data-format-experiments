# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six mouse IDs, enumerates and sorts every subdirectory under each mouse as a session, and loads each session's `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. It then processes every listed session and splits it into trials.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
sessions = sorted([
    d for d in os.listdir(mouse_dir)
    if os.path.isdir(os.path.join(mouse_dir, d))
])
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. In the trajectory, the agent inspected the directory tree, identified these six mice and 41 sessions, and intended to use all available recordings. It chose explicit mouse IDs after inspecting the dataset rather than discovering `jm*` directories dynamically.

## 1-b. How are the data split into subjects?

i. Each hard-coded entry in `MICE` is treated as one subject. Its integer position is stored for every session in `subject_idx`.

ii.
```python
for mouse_i, mouse in enumerate(MICE):
    ...
    subject_idx.append(mouse_i)
...
'subjects': MICE,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. The agent concluded from the directory naming and its inspection that `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` are the six mice.

## 1-c. How are the data split into sessions?

i. Every directory immediately below a mouse directory is one session; session names are sorted lexicographically. One element is appended to each top-level session list per directory.

ii.
```python
sessions = sorted([
    d for d in os.listdir(mouse_dir)
    if os.path.isdir(os.path.join(mouse_dir, d))
])
for sess in sessions:
    session_dir = os.path.join(mouse_dir, sess)
    ...
    all_neural.append(neural_trials)
```

iii. The trajectory identified the dated subdirectories as daily recordings and used the source directory hierarchy as the session boundary.

## 1-d. How are the data split into trials?

i. Each continuous session is split into consecutive, non-overlapping 60-second trials. At 30 Hz with ten-frame averaging, each trial has 180 bins. A final incomplete trial is discarded.

ii.
```python
TRIAL_DUR = 60
TRIAL_BINS = int(TRIAL_DUR / BIN_DUR)
...
n_trials = n_bins // trial_bins
for t in range(n_trials):
    start = t * trial_bins
    end = start + trial_bins
    neural_trials.append(neural[:, start:end])
    me_trials.append(me_binned[start:end])
```

iii. The explicit decoder instruction required 60-second trials. The agent reasoned that ten-frame bins produce 180 bins per trial and reported 20 trials for 20-minute sessions and 30 for 30-minute sessions.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality-control filtering is applied. Only a trailing segment too short to form a complete 60-second trial is omitted.

ii.
```python
n_trials = n_bins // trial_bins
for t in range(n_trials):
    ...
```

iii. The trajectory found no natural trials or trial-quality annotations and therefore used all complete fixed-duration segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p plane-0 fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. The agent inspected the paper and repository and chose fluorescence rather than `spks.npy` because the paper described dF/F for decoding.

## 2-b. How is the `neural` data processed?

i. The agent subtracts 0.7 times neuropil fluorescence, estimates a maximin baseline by Gaussian smoothing followed by a 60-second minimum and maximum filter, subtracts that baseline, casts to `float32`, and averages non-overlapping groups of ten frames.

ii.
```python
Fc = F - NEUCOEFF * Fneu
win = int(WIN_BASELINE * fs)
Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
dff = Fc - Flow
return dff.astype(np.float32)
...
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The trajectory explicitly selected Suite2p defaults (`neucoeff=0.7`, maximin baseline, `sig_baseline=10`, `win_baseline=60`, 30 Hz) and the paper's decoding-time averaging of ten consecutive timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron/ROI filtering is performed; every row in `F.npy` is retained.

ii.
```python
n_neurons = dff_binned.shape[0]
brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The agent inspected `iscell` and stated it was uniformly one, so it included all tracked cells rather than applying another ROI mask.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. The continuous recording is cut from session start into consecutive 60-second windows, and metadata calls session start the alignment event.

ii.
```python
start = t * trial_bins
end = start + trial_bins
neural_trials.append(neural[:, start:end])
...
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
```

iii. The agent reasoned that the data are continuous and have no stimulus-driven trial event, so session start is the natural origin. It did not discuss that later artificial trials do not themselves begin at session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten adjacent 30-Hz frames are averaged without overlap, producing 3-Hz data and 333.33-ms bins. Short tails below ten frames are truncated.

ii.
```python
BIN_SIZE = 10
BIN_DUR = BIN_SIZE / FS
...
return data[:, :n].reshape(data.shape[0], -1, bin_size).mean(axis=2)
...
'time_bin_size': BIN_DUR * 1000,
```

iii. The agent cited the paper's statement that dF/F and behavior traces were denoised by averaging ten consecutive timestamps for decoding.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from the trial number, bin index, fixed 30-Hz sampling rate, and ten-frame bin duration, not read from a raw timestamp variable. The generated value represents each averaged bin's center.

ii.
```python
start_bin = t * TRIAL_BINS
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
```

iii. The agent reasoned that elapsed session time can be computed from the known constant frame rate. It deliberately described the result as time from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the code forms 180 sequential bin indices, offsets them by the trial's starting bin, adds 0.5 to select bin centers, converts to seconds using `10/30`, reshapes to `(1, 180)`, and casts to `float32`.

ii.
```python
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

iii. The trajectory noted that 10-frame bins last one third of a second and chose a continuously increasing elapsed-session-time decoder input.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural arrays use the same trial index and the same 180 within-trial bin positions. A neural average over frames `10k..10k+9` is labeled by its center time `(k+0.5)/3` seconds.

ii.
```python
neural_trials.append(neural[:, start:end])
...
start_bin = t * TRIAL_BINS
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
```

iii. The agent's stated intent was a time-varying elapsed-session input with exactly the same 180 samples as each neural trial.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived from precomputed `move_deve/motion_energy_glob.npy`. The code also loads `tstamps.npy`, nominally for alignment, although the timestamp values are never actually used.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
me_aligned = align_motion_energy(me, n_neural, tstamps)
```

iii. The trajectory identified global motion energy as the behavioral target. It explored timestamp units and concluded that relative positions would suffice, but the eventual implementation did not use those timestamp positions.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If its length differs from neural data, motion energy is linearly resampled from a uniform normalized grid to the neural frame count. It is then averaged in ten-frame bins, split into complete trials, pooled within the retained portion of each session, and digitized using the session's 20th, 40th, 60th, and 80th percentiles.

ii.
```python
camera_pos = np.linspace(0, 1, len(me))
neural_pos = np.linspace(0, 1, n_neural_frames)
me_aligned = np.interp(neural_pos, camera_pos, me)
...
me_binned = bin_data(me_aligned, BIN_SIZE)
...
all_me = np.concatenate(me_trials)
boundaries = np.percentile(all_me, percentiles)
binned = np.digitize(me, boundaries)
```

iii. The agent intended to fill dropped camera frames, apply the paper's ten-frame behavior averaging, and satisfy the requested five equal-percentile categories per session. It judged normalized interpolation adequate because length mismatches were small.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four percentile boundaries (20, 40, 60, and 80) are computed separately for each session from all complete-trial motion-energy bins. `np.digitize` maps values to integer classes 0 through 4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
boundaries = np.percentile(all_me, percentiles)
...
binned = np.digitize(me, boundaries)
```

iii. The agent followed the decoder instruction to create five equal-percentile bins selected per session and emphasized that discretization occurs after temporal averaging.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Equal-length streams are treated as frame-aligned. For unequal lengths, the agent globally resamples motion energy between normalized session endpoints, rather than locating individual dropped frames. Neural and motion energy are then independently averaged in matching ten-frame bins and sliced with identical trial boundaries.

ii.
```python
if len(me) == n_neural_frames:
    return me
camera_pos = np.linspace(0, 1, len(me))
neural_pos = np.linspace(0, 1, n_neural_frames)
me_aligned = np.interp(neural_pos, camera_pos, me)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The trajectory recognized dropped camera frames and originally said timestamps should locate them, but then decided proportional interpolation was sufficient because the mismatch was small. This stretches all intervals and does not use actual drop locations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. A short motion-energy stream is resampled to the neural length with `np.interp`; equal lengths are accepted unchanged. Incomplete ten-frame bins and incomplete 60-second trials are silently discarded. There are no explicit assertions, NaN handling, or missing-file recovery.

ii.
```python
if len(me) == n_neural_frames:
    return me
...
me_aligned = np.interp(neural_pos, camera_pos, me)
...
n = len(data) // bin_size * bin_size
...
n_trials = n_bins // trial_bins
```

iii. The agent identified dropped camera frames as the relevant data defect and chose interpolation. It considered the small mismatch sufficient justification for global resampling and relied on successful format validation afterward.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant computation is per-session neural baseline estimation over every neuron and frame: Gaussian smoothing plus 1,800-frame minimum and maximum filters. Loading large arrays and casting/copying `Fc` to `float64` are also substantial.

ii.
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The trajectory investigated Suite2p baseline processing and understood it as the main full-session operation, though it did not provide a formal runtime profile.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial slicing, per-trial time construction, and per-trial digitization could be replaced by truncation followed by reshape/broadcast operations. Directory/session loops remain appropriate because neuron and session sizes vary.

ii.
```python
for t in range(n_trials):
    neural_trials.append(neural[:, start:end])
    me_trials.append(me_binned[start:end])
...
for me in me_trials:
    binned = np.digitize(me, boundaries)
...
for t in range(len(neural_trials)):
    time_points = ...
```

iii. The trajectory did not explicitly justify these Python loops; they are straightforward assembly code, and their cost is likely small compared with baseline filtering.

## 6-c. What processing does the code repeat multiple times?

i. Neural filtering, motion loading/alignment, temporal binning, percentile computation, time-array construction, and trial-list assembly are repeated independently for every session. Motion-energy trial lists are traversed once to concatenate them and again to digitize them.

ii.
```python
for sess in sessions:
    dff_binned, me_binned = process_session(session_dir)
    neural_trials, me_trials = split_trials(dff_binned, me_binned)
    me_discrete = discretize_me(me_trials)
```

iii. The agent designed processing at session scope because baseline correction and output percentile boundaries are session-specific. It did not identify reusable cross-session results.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `tstamps.npy` is loaded but its values are ignored. Samples in incomplete final temporal bins/trials are processed through neural baseline correction and motion alignment before being discarded. The code also creates `me_trials`, concatenates them, then builds a second set of discretized trial arrays.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
...
camera_pos = np.linspace(0, 1, len(me))  # does not use tstamps
...
n_trials = n_bins // trial_bins
```

iii. The trajectory intended timestamps to support alignment but switched to normalized positions without removing the load. It did not discuss the computational cost of processing eventual tail data; the final dataset happened to contain recordings with complete trial lengths.
