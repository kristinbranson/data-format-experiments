# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six mouse IDs, finds sorted session directories whose names begin with a digit, and loads each session's `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. It processes every discovered session and later divides each continuous recording into trials.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. The trajectory says the agent inspected the project, data, paper, and reference code, identified six mice and 41 daily sessions, and described these files as the Suite2p fluorescence and precomputed global motion-energy products required by the task.

## 1-b. How are the data split into subjects?

i. Each hard-coded `MICE` entry is treated as one subject. Its list position becomes the subject index for every session belonging to that mouse.

ii.
```python
for mouse_idx, mouse in enumerate(MICE):
    ...
    subject_idx_list.append(mouse_idx)
```

iii. The agent concluded from the directory layout that the six `jm...` folders are the six mice. It reported six mice in its final summary.

## 1-c. How are the data split into sessions?

i. A session is each digit-prefixed directory directly below a mouse directory. Names are sorted, and one output session is appended per directory.

ii.
```python
sessions = [d for d in os.listdir(mouse_dir)
            if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
sessions.sort()
...
for session_name in sessions:
    ...
    neural_all.append(session_neural)
```

iii. The agent interpreted the 6–7 dated directories per mouse as daily recording sessions and reported 41 sessions total.

## 1-d. How are the data split into trials?

i. Each continuous session is split into consecutive, non-overlapping 60-second trials. At 30 Hz with 10-frame bins, a trial is 180 bins. Any incomplete final segment is omitted.

ii.
```python
bin_duration = BIN_SIZE / fs
bins_per_trial = int(TRIAL_DURATION / bin_duration)
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
```

iii. The task explicitly requests 60-second trials. The agent noted that the source is continuous rather than naturally trial-based, so it used fixed consecutive segments.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. Only complete 60-second trials are retained; the leftover tail is dropped.

ii.
```python
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    ...
```

iii. The trajectory gives no separate trial-quality criterion. The only stated selection rule is completeness of the fixed-length segment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy`; processing parameters and sampling rate are read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
fs = ops['fs']
```

iii. The agent identified `F` as fluorescence and `Fneu` as neuropil fluorescence, and used the saved Suite2p settings to reproduce the paper/code processing.

## 2-b. How is the `neural` data processed?

i. The agent subtracts 0.7-scaled neuropil, estimates a maximin baseline by Gaussian smoothing followed by minimum and maximum filters, subtracts that baseline (without division), then averages every 10 consecutive frames. It casts each saved trial to `float32`.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
dff = Fc - Flow
...
dff_binned = bin_data(dff, BIN_SIZE)
session_neural.append(dff_binned[:, start:end].astype(np.float32))
```

iii. The trajectory records that the first implementation divided by near-zero baselines, creating extreme values and poor decoding. After re-reading Track2p's `F_processing`, the agent changed to baseline subtraction, reasoning that this matches the actual code despite the paper's “dF/F” terminology. Decoder accuracy then improved.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit ROI or neuron filtering is performed; every row in `F.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
...
session_neural.append(dff_binned[:, start:end].astype(np.float32))
```

iii. The agent stated that the supplied ROIs had already passed the Suite2p `iscell > 0.5` criterion and were tracked by Track2p, so it considered another filter unnecessary. The script does not load or verify `iscell.npy`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Trials are consecutive slices from the start of the continuous recording; metadata calls the start of the recording session the alignment event.

ii.
```python
start = t * bins_per_trial
end = start + bins_per_trial
session_neural.append(dff_binned[:, start:end].astype(np.float32))
...
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION),
```

iii. The agent reasoned that the data contain continuous spontaneous behavior and no stimulus-driven trial event, so session-relative segmentation is the applicable alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are averaged per non-overlapping bin, producing a nominal 3-Hz effective rate and 333.33-ms bins. A tail shorter than ten frames is discarded.

ii.
```python
BIN_SIZE = 10
FS = 30
...
return data.reshape(new_shape).mean(axis=-1)
...
time_bin_ms = (BIN_SIZE / FS) * 1000
```

iii. The agent cited the paper's decoding method, which averages neural and behavior traces in bins of 10 consecutive timestamps for denoising.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated rather than loaded. It is derived from the binned sample index and the session sampling rate from `ops.npy`.

ii.
```python
fs = ops['fs']
bin_duration = BIN_SIZE / fs
time_vec = np.arange(n_bins) * bin_duration
```

iii. The agent treated constant-rate sample indices as elapsed seconds because no separate neural timestamp vector was needed.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The zero-based bin index is multiplied by `10 / fs`, producing seconds from session start at bin left edges. Trial slices are reshaped to `(1, time)` and cast to `float32`.

ii.
```python
time_vec = np.arange(n_bins) * bin_duration
session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
```

iii. The agent described this as time elapsed from session start, continuous across successive trials, with a one-third-second step.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector has one value per binned neural column and is sliced with exactly the same `start:end` indices as neural data.

ii.
```python
session_neural.append(dff_binned[:, start:end].astype(np.float32))
session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
```

iii. The trajectory does not give an additional alignment justification; common indexing follows directly from constructing time from neural bin indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived solely from each session's precomputed `move_deve/motion_energy_glob.npy`. The agent does not load `interframe_int.npy`.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. The agent identified this file as the paper's global motion-energy behavioral trace. It did not justify omitting the available interframe-interval data beyond choosing common-length truncation for mismatches.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is truncated to the neural/video common length, averaged in non-overlapping 10-frame bins, then discretized using session-specific percentiles.

ii.
```python
common_len = min(n_frames, len(me))
me = me[:common_len]
...
me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]
bin_edges = np.percentile(me_binned, percentiles)
```

iii. The agent said equal binning keeps behavior synchronized with neural data and follows the paper's denoising method. It summarized mismatch handling as truncation to the minimum length.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five per-session equal-percentile categories are formed using the 0, 20, 40, 60, 80, and 100 percentiles of binned motion energy. `np.digitize` creates labels 0–4 and clipping guarantees the range.

ii.
```python
percentiles = np.linspace(0, 100, N_ME_BINS + 1)
bin_edges = np.percentile(me_binned, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
me_discrete = np.digitize(me_binned, bin_edges[1:])
me_discrete = np.clip(me_discrete, 0, N_ME_BINS - 1)
```

iii. The task requests five equal-percentile bins selected per session, which the agent explicitly followed. It bins the continuous trace before categorization so class labels are never averaged.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent forces equal raw lengths by truncating both fluorescence arrays and motion energy to their minimum length, bins both by the same factor, and applies identical trial indices. It does not repair dropped video frames at their actual positions.

ii.
```python
common_len = min(n_frames, len(me))
F = F[:, :common_len]
Fneu = Fneu[:, :common_len]
me = me[:common_len]
...
session_neural.append(dff_binned[:, start:end].astype(np.float32))
session_output.append(me_discrete[start:end][np.newaxis, :].astype(np.int64))
```

iii. The agent's final summary says truncation handles frame mismatch. The trajectory contains no evidence that it checked where video frames were dropped; thus its justification addresses equal array lengths, not preservation of framewise timing after a mid-session drop.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and motion-energy arrays are truncated to the shorter length. Short tails that cannot complete a 10-frame bin or 60-second trial are also discarded. There is no interpolation, missing-value imputation, or explicit assertion for unexpected mismatch patterns.

ii.
```python
common_len = min(n_frames, len(me))
F = F[:, :common_len]
Fneu = Fneu[:, :common_len]
me = me[:common_len]
...
n_bins = n_frames // bin_size
...
n_trials = n_bins // bins_per_trial
```

iii. The agent characterized the mismatch as some sessions having fewer motion-energy frames and chose truncation as a simple common-length policy. It did not use the interframe intervals to locate and interpolate dropped frames.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant computation is maximin baseline estimation: Gaussian, minimum, and maximum filters over every neuron and every frame in every session. Loading large arrays and decoder training are also costly, though training is validation rather than conversion.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The trajectory shows conversion runs with a long timeout and repeated decoder training, but the agent did not explicitly profile or document conversion timings. The full-session baseline filters are the clearest computational hotspot from the code.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop mostly packages array views/copies into the required nested-list format; trial data could first be reshaped in bulk, though lists are still required. Session and mouse loops operate on separate files and variable neuron counts, so straightforward vectorization is limited.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end].astype(np.float32))
    session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
    session_output.append(me_discrete[start:end][np.newaxis, :].astype(np.int64))
```

iii. The agent did not discuss vectorization. Unlike the reference implementation, it has no repeated `np.insert` dropped-frame loop; its main remaining loop is low-cost output assembly.

## 6-c. What processing does the code repeat multiple times?

i. Loading, baseline filtering, binning, percentile calculation, and trial packaging are repeated independently for every session. These repetitions are necessary because data and percentile thresholds are session-specific. During development, the agent also ran conversion twice and decoder training twice after correcting baseline division.

ii.
```python
for mouse_idx, mouse in enumerate(MICE):
    ...
    for session_name in sessions:
        ...
        dff = compute_dff(...)
        dff_binned = bin_data(dff, BIN_SIZE)
        me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]
```

iii. The trajectory explains the repeated development runs as diagnosis and verification of extreme neural values. It provides no claim that avoidable processing is repeated within the final conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Baseline filtering is performed on the common-length continuous recording before incomplete bin/trial tails are discarded, so processing on the final unused tail is wasted. It also computes and prints per-session motion-energy distributions, which are diagnostic only. No major scientific intermediate is computed solely to be discarded.

ii.
```python
dff = compute_dff(F, Fneu, fs, ...)
...
n_trials = n_bins // bins_per_trial
...
print(f"ME bins distribution: {[np.sum(me_discrete==i) for i in range(N_ME_BINS)]}")
```

iii. The agent did not identify unnecessary downstream-discarded processing. Its diagnostics were used to verify conversion quality, while incomplete tails are an unavoidable consequence of the requested fixed trial size (though they could be removed earlier).
