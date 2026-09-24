# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six subject IDs, discovers every immediate session directory under each subject in sorted order, and processes every session unless `--sample` stops after two. For each session it loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`. The full run produced 41 sessions and 1,090 trials.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_session_dirs(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))])
    return [os.path.join(subj_dir, s) for s in sessions]

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes say the data survey found exactly six mice and 41 sessions, with 6–7 sessions per mouse. The agent included all available data despite the paper describing 20-minute sessions while some files contain 30 minutes.

## 1-b. How are the data split into subjects?

i. A subject is one of six explicitly listed mouse-directory names. The outer loop index becomes `subject_idx` for every session belonging to that mouse.

ii.
```python
for subj_i, subject in enumerate(SUBJECTS):
    ...
    all_subject_idx.append(subj_i)
```

iii. The notes identify these directories as the six experimental mice and verify that `subject_idx` matches the expected mapping. Hard-coding reflects the surveyed dataset, though it is less robust than discovering `jm*` folders.

## 1-c. How are the data split into sessions?

i. Each immediate subdirectory of a subject directory is treated as a session; paths are lexically sorted, then each processed session is appended as one entry in the target lists.

ii.
```python
sess_dirs = get_session_dirs(subject)
for sess_dir in sess_dirs:
    neural_trials, input_trials, output_trials, n_neurons = process_session(...)
    all_neural.append(neural_trials)
```

iii. The notes describe these as daily recordings and report the expected 7,7,7,7,6,7 sessions across mice. Sorting gives deterministic chronological order for the date-prefixed directory names.

## 1-d. How are the data split into trials?

i. After 10-frame averaging, each continuous session is divided into non-overlapping 60-second blocks: 180 bins at 3 Hz. Only complete blocks are emitted; any incomplete tail is implicitly discarded.

ii.
```python
TRIAL_BINS = int(TRIAL_DURATION_SEC * BINNED_FS)
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. The 60-second duration comes directly from the task. The notes verify 180 time bins per trial, no overlap, and correct last-bin indexing.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. Every complete 60-second block is retained, and only a final partial block would be omitted.

ii.
```python
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    ...
```

iii. The notes state that this spontaneous-behavior dataset has no natural task trials and no explicit trial-curation rules.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p plane-0 raw fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. The agent identified these as the inputs used by the paper/reference processing, rather than using the available deconvolved `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The agent subtracts 0.7 times neuropil fluorescence, estimates a maximin baseline by Gaussian smoothing followed by minimum and maximum filters, subtracts that baseline (without division), converts to float32, and averages non-overlapping groups of 10 frames.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
dff = Fc - Flow
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. The notes resolve two ambiguities: 0.7 is the Suite2p default stored in `ops.npy`, and the project code implements “dF/F” as baseline subtraction rather than division. The agent reports fixing an initial division implementation after it produced extreme values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional filtering is performed in the converter; all rows in the provided fluorescence arrays are retained.

ii.
```python
n_neurons, n_frames = F.shape
# no iscell mask is applied
```

iii. The notes say all supplied `iscell` entries were checked and equal 1, and that the files already contain Track2p cells tracked across all days. Therefore another mask would not change the data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Neural bins are contiguous slices beginning at session time zero, and the agent labels the alignment event `session_start`. It records `off_start=0` and `off_end=60`, although those offsets only describe the first trial; later trial slices occur later in the session.

ii.
```python
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS
neural_trials.append(dff_binned[:, start:end].astype(np.float32))

'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),
```

iii. The notes justify session start because the recording is continuous and the trials are artificial. They verify non-overlapping slicing, but do not discuss the metadata-offset inconsistency for trials after the first.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are averaged per bin, yielding 3 Hz data and 333.33 ms bins. Short tails below ten frames are trimmed.

ii.
```python
BIN_SIZE = 10
BINNED_FS = FS / BIN_SIZE
TIME_BIN_MS = (BIN_SIZE / FS) * 1000
return data_trimmed.reshape(new_shape).mean(axis=axis + 1).astype(np.float32)
```

iii. The paper explicitly says decoding traces were denoised by averaging ten consecutive timestamps, so the agent applies the same operation to neural and behavioral streams.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from a raw timestamp variable. It is derived from the post-binning sample index, the ten-frame bin width, and the fixed 30-Hz rate.

ii.
```python
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
```

iii. The notes say this represents seconds from session start at the binned resolution and verify starts such as 0, 300, and 1,140 seconds for selected trials.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A zero-based bin index is multiplied by 10/30 seconds. The resulting float32 vector runs continuously across trial boundaries and is reshaped to one input row when sliced.

ii.
```python
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
```

iii. This follows from the known constant acquisition rate and meets the task requirement for time elapsed from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector has one value per binned neural sample, and both are sliced with exactly the same `start:end` indices for each trial.

ii.
```python
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
```

iii. The agent’s spot checks confirmed expected trial start times and correct bin endpoints.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived primarily from `move_deve/motion_energy_glob.npy`; `interframe_int.npy` controls missing-frame placement. `tstamps.npy` is loaded and passed to the interpolation function but is not actually used.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
```

iii. The notes identify the saved global motion-energy trace as the paper’s precomputed behavioral signal and the inter-frame intervals as the evidence for dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing camera samples are restored by mapping camera frames onto neural-frame indices inferred from interval/median-interval ratios and linearly interpolating. The aligned trace is then averaged in groups of ten and discretized per session.

ii.
```python
median_ifi = np.median(ifi)
for i in range(len(ifi)):
    n_skip = max(1, round(ifi[i] / median_ifi))
    neural_idx += n_skip
    if i + 1 < len(me):
        neural_indices[i + 1] = neural_idx
me_full = np.interp(np.arange(n_neural_frames), neural_indices, me.astype(np.float64))
me_binned = bin_data(me_full.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
```

iii. The notes explain that behavioral and neural data must be aligned before identical temporal averaging. Gap interpolation was validated on the session with 116 missing frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. For each session independently, the agent computes the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of binned motion energy. The four internal boundaries are passed to `np.digitize`, producing integer labels 0–4, then clipped defensively.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(me_binned, percentiles)
me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
me_discrete = np.clip(me_discrete, 0, n_bins - 1)
```

iii. Five equal-percentile bins selected per session are required by the task. The agent verified approximately 20% occupancy in every class (exact here because ties did not disturb the cuts).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera gaps are first expanded onto the neural frame grid. Neural and motion traces are then independently averaged over identical ten-frame blocks and sliced with identical trial boundaries.

ii.
```python
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_full.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The agent reasons that microscope-triggered video is nominally frame-synchronous and that detected interval gaps locate missing frames. Its inferred indices end on the expected neural frame for the supplied missing-frame sessions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion-energy arrays are linearly interpolated to the neural length using inter-frame gaps. Complete ten-frame bins and complete 60-second trials are retained; incomplete tails are dropped. The code does not explicitly check NaNs, but the final data were checked for NaN/Inf and none were found.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
me_full = np.interp(np.arange(n_neural_frames), neural_indices, me.astype(np.float64))
n_bins = n // bin_size
n_trials = n_total_bins // TRIAL_BINS
```

iii. The notes document all sessions with missing video frames, verify the worst case, and report clean final validation. Interpolation is preferred to shifting all subsequent samples or dropping matched neural data.

## 6-a. What are the most time-consuming steps of the code?

i. Maximin baseline estimation dominates conversion time; file loading is the secondary cost. Per-session timing in the notes shows baseline processing at roughly 0.25–0.95 seconds, while binning is about 0.01–0.06 seconds.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
t_dff = time.time() - t1
```

iii. Gaussian and long-window min/max filters operate over every neuron and frame. The agent measured rather than merely inferred the timing and estimated about 41 seconds for the full conversion.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop constructing `neural_indices` from inter-frame intervals could be replaced by a vectorized rounded-ratio calculation and cumulative sum. The trial loop could be reshaped/vectorized, though it builds the required list objects and is cheap relative to filtering.

ii.
```python
for i in range(len(ifi)):
    n_skip = max(1, round(ifi[i] / median_ifi))
    neural_idx += n_skip
    if i + 1 < len(me):
        neural_indices[i + 1] = neural_idx
```

iii. The agent did not document this optimization explicitly. Its timing shows interpolation and trial assembly are negligible compared with fluorescence filtering, so vectorization would have little end-to-end effect.

## 6-c. What processing does the code repeat multiple times?

i. Loading, maximin filtering, motion interpolation, binning, and percentile computation are intentionally repeated once per session. Trial slices are also cast to dtypes they generally already have, causing repeated copies.

ii.
```python
for sess_dir in sess_dirs:
    neural_trials, input_trials, output_trials, n_neurons = process_session(...)
...
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. Per-session repetition is necessary because sessions have separate files, neuron counts, and output percentile boundaries. The notes do not claim avoidable repeated scientific processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `tstamps.npy` is loaded and passed through but never used. Trial slices are copied by redundant `astype` calls. With `--show-processing`, large diagnostic plots and intermediate traces are produced but are not part of the converted dataset; in the full recorded run plotting was off.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
# interpolate_missing_frames never reads tstamps

neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. The notes justify optional plots as sample-stage visual validation. They do not justify the unused timestamp load or redundant casts; these are minor efficiency issues and do not alter converted values.
