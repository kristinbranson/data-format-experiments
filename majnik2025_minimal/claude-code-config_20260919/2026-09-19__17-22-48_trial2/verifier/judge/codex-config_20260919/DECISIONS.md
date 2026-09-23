# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory directly under `/app/data` as a subject, then every subject child matching `*_a` as a session. For each session it loads Suite2p `F.npy`, `Fneu.npy`, `iscell.npy`, and `ops.npy`, plus motion energy and, when needed, camera timestamps. It ultimately includes 6 mice and 41 sessions.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
session_dirs = sorted(glob.glob(os.path.join(DATA_DIR, subj, '*_a')))
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
```

iii. The trajectory shows the agent inspected the directory tree and every session’s array shapes before implementing this. It concluded that all 6 mice and all 41 dated `_a` recording directories contained complete neural and behavioral data.

## 1-b. How are the data split into subjects?

i. Each immediate subdirectory of `/app/data` is treated as one mouse, sorted lexicographically; its enumeration index is stored once per session in `subject_idx`.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
for subj_i, subj in enumerate(subjects):
    ...
    data['subject_idx'].append(subj_i)
```

iii. The agent observed that the six top-level data directories are mouse directories and reported retaining all six.

## 1-c. How are the data split into sessions?

i. A session is each sorted `*_a` directory beneath a mouse. Each daily recording becomes one outer-list session in `neural`, `input`, and `output`.

ii.
```python
session_dirs = sorted(glob.glob(os.path.join(DATA_DIR, subj, '*_a')))
for day_i, session_dir in enumerate(session_dirs):
    ...
    data['neural'].append(neural_trials)
```

iii. Inspection showed that dated `_a` directories are the recording sessions. Sorting provides stable chronological ordering for the supplied ISO-formatted dates.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping 60-second trials. At 30 Hz with 10-frame bins, each trial contains 180 bins (1,800 original frames). Any incomplete tail is excluded before binning.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS / BIN_SECONDS))
FRAMES_PER_TRIAL = BINS_PER_TRIAL * BIN_FRAMES
n_trials = n_frames // FRAMES_PER_TRIAL
n_keep = n_trials * FRAMES_PER_TRIAL
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
```

iii. The task explicitly requests 60-second trials, while the experiment is continuous and has no natural trials. The agent noted that the supplied 20- and 30-minute recordings divide exactly into such trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. All complete 60-second trials are kept; only an incomplete terminal fragment would be dropped.

ii.
```python
n_trials = n_frames // FRAMES_PER_TRIAL
n_keep = n_trials * FRAMES_PER_TRIAL
```

iii. The agent found complete data in all sessions and no paper-defined trial rejection rule for these continuous recordings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p raw ROI fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`. `ops.npy` supplies the checked sampling rate and `iscell.npy` is used as a curation assertion.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
```

iii. The agent traced the paper’s processing to the repository’s `F_processing` implementation and identified these as the relevant Suite2p signals.

## 2-b. How is the `neural` data processed?

i. The agent subtracts 0.7 times neuropil fluorescence, estimates a maximin baseline after Gaussian smoothing (sigma 10 frames; 60-second window), subtracts that baseline, averages non-overlapping groups of 10 frames, and then z-scores each neuron over the retained session.

ii.
```python
Fc = F.astype(np.float64) - NEUCOEFF * Fneu.astype(np.float64)
Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
Flow = minimum_filter1d(Flow, int(WIN_BASELINE * fs))
Flow = maximum_filter1d(Flow, int(WIN_BASELINE * fs))
return Fc - Flow
...
neural = bin_time(dff[:, :n_keep])
neural = (neural - neural.mean(axis=1, keepdims=True)) / \
         (neural.std(axis=1, keepdims=True) + 1e-9)
```

iii. Baseline subtraction and 10-frame averaging follow the paper/code. Z-scoring is an explicit deviation: trajectory experiments indicated that the supplied decoder’s uncentered SVD was dominated by bright ROIs, while session-wise z-scoring improved subset validation balanced accuracy from about 0.24 to 0.34 without changing temporal structure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROIs are removed during conversion. The code verifies that every released ROI has `iscell[:, 0] == 1`; failure aborts conversion.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
assert np.all(iscell[:, 0] == 1)
```

iii. The agent checked all sessions and found that the release already consists entirely of Suite2p-accepted, Track2p-tracked ROIs, so another filter would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is the first imaging frame/session start. Trials are consecutive windows from that point, with offsets documented as 0 to 60 seconds.

ii.
```python
sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
neural_trials.append(neural[:, sl])
...
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. There is no stimulus or behavioral event in this spontaneous recording, making session start the only natural anchor for the requested artificial trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged into each bin, yielding 3 Hz or 333.333 ms resolution. Neural and behavior streams receive identical non-overlapping rebinning.

ii.
```python
BIN_FRAMES = 10
BIN_SECONDS = BIN_FRAMES / FS
return x.reshape(*x.shape[:-1], n // bin_frames, bin_frames).mean(axis=-1)
...
'time_bin_size': 1000.0 * BIN_SECONDS
```

iii. The agent cites the Methods statement that neural and behavioral traces were denoised by averaging 10 consecutive timestamps for decoding analyses.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from the temporal bin index and the fixed/verified 30-Hz frame rate, rather than loaded from a raw timestamp array. Values denote bin centers in seconds from session start.

ii.
```python
fs = float(ops['fs'])
assert fs == FS
t = (np.arange(n_bins) + 0.5) * BIN_SECONDS
```

iii. The imaging rate is consistently 30 Hz. Using bin centers is a natural timestamp for values formed by averaging each 10-frame interval.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The zero-based binned sample index is shifted by one half-bin and multiplied by 10/30 seconds, then converted to float32 when assigned to trials.

ii.
```python
t = (np.arange(n_bins) + 0.5) * BIN_SECONDS
input_trials.append(t[sl][None, :].astype(np.float32))
```

iii. This produces seconds from session start at the center of every averaged time bin and preserves continuity across trial boundaries.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural arrays have one value per common 10-frame bin and are cut with the exact same trial slice.

ii.
```python
neural_trials.append(neural[:, sl])
input_trials.append(t[sl][None, :].astype(np.float32))
```

iii. Common indexing guarantees one-to-one alignment; the generated time vector spans precisely the retained number of neural bins.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from `move_deve/motion_energy_glob.npy`. For sessions whose camera trace is short, `move_deve/tstamps.npy` identifies missing camera-frame positions.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve',
                          'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
```

iii. The trajectory confirms that the agent inspected both timestamp and interframe-interval files, then selected timestamps because their gaps reconstruct exactly the observed deficits.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The initial artificial zero is replaced with the first measurable value. Missing camera samples are linearly interpolated on a reconstructed imaging-frame grid. The retained trace is averaged in 10-frame bins and discretized using session-specific quintiles.

ii.
```python
me[0] = me[1]
full = np.full(n_frames, np.nan)
full[idx] = me
full[missing] = np.interp(np.flatnonzero(missing), idx, me)
...
behav = bin_time(me[:n_keep])
thresholds = np.percentile(behav, np.arange(1, N_QUANTILES) * 20)
```

iii. The first value cannot contain an interframe difference. Interpolation restores frame correspondence, and behavior is averaged before categorization because averaging category labels would be invalid.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Each session’s binned values are split at its 20th, 40th, 60th, and 80th percentiles into integer labels 0–4. Values exactly equal to a threshold go into the lower category.

ii.
```python
thresholds = np.percentile(behav, np.arange(1, N_QUANTILES) * (100 / N_QUANTILES))
labels = np.searchsorted(thresholds, behav, side='left').astype(np.int64)
```

iii. This directly implements the requested five equal-percentile bins selected independently per session; the agent verified near-perfect class balance.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera frame indices are reconstructed from timestamp intervals relative to their median interval. Missing positions are interpolated to make the behavior trace exactly `n_frames` long. Neural and behavior are then truncated, identically binned, and sliced into trials with common indices.

ii.
```python
steps = np.round(np.diff(ts) / np.median(np.diff(ts))).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
assert idx[-1] == n_frames - 1
...
neural = bin_time(dff[:, :n_keep])
behav = bin_time(me[:n_keep])
output_trials.append(labels[sl][None, :])
```

iii. The camera was microscope-triggered, so ordinary samples are frame-for-frame. The agent found nine dropped-frame sessions and verified that timestamp reconstruction exactly accounted for every missing sample; it also reported population-motion cross-correlation peaking at zero lag.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are reconstructed and linearly interpolated; assertions validate the resulting endpoint, unique indices, 30-Hz rate, and accepted-cell flags. The initial motion-energy artifact is replaced, and incomplete trial tails are discarded.

ii.
```python
assert idx[-1] == n_frames - 1
assert len(np.unique(idx)) == len(idx)
full[missing] = np.interp(np.flatnonzero(missing), idx, me)
...
n_keep = (n_frames // FRAMES_PER_TRIAL) * FRAMES_PER_TRIAL
```

iii. The agent chose explicit reconstruction plus fail-fast checks rather than silently accepting length mismatches. It verified the method against every affected session.

## 6-a. What are the most time-consuming steps of the code?

i. The principal conversion cost is full-session baseline estimation for every neuron (Gaussian, minimum, and maximum filters), followed by loading and handling large fluorescence arrays. Decoder training was much slower during validation but is not part of conversion.

ii.
```python
Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. These filters traverse every neuron-frame value and include a 1,800-frame sliding window. The trajectory devoted substantial runtime to full conversion and repeated decoder experiments.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial append loop could be replaced by reshaping each already-binned session into trial axes (with a transpose for neural data). Subject/session iteration must remain because files and neuron counts differ. Missing-frame filling is already vectorized.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    neural_trials.append(neural[:, sl])
    input_trials.append(t[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :])
```

iii. The agent did not explicitly discuss this loop, but its fixed-size contiguous slices make vectorization straightforward. Its timestamp approach avoids the reference implementation’s repeated `np.insert` loop.

## 6-c. What processing does the code repeat multiple times?

i. Loading, baseline filtering, alignment, binning, percentile calculation, and trial assembly repeat once per session. Float32 conversion is performed separately for neural data and each time trial. No large intermediate analysis is redundantly recomputed within a session.

ii.
```python
for subj_i, subj in enumerate(subjects):
    ...
    for day_i, session_dir in enumerate(session_dirs):
        ...
        dff = compute_dff(F, Fneu, fs)
        me = load_motion_energy(session_dir, n_frames)
```

iii. The trajectory gives no separate justification; repetition follows the file-per-session organization and session-specific baselines, z-scores, alignment, and quintile thresholds.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `iscell.npy` and most of `ops.npy` are loaded only for validation, not as decoder features. Detailed `session_info` and percentile thresholds are retained as metadata but unused by the supplied decoder. The mean and standard deviation used for z-scoring are not saved. These are small relative to the core filtering and are useful checks/provenance rather than harmful work.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
assert fs == FS
assert np.all(iscell[:, 0] == 1)
...
'motion_energy_quintile_thresholds': thresholds.tolist(),
```

iii. The agent intentionally inspected `iscell` and `ops` to verify curation and sampling assumptions and added rich metadata for auditability, even though downstream training does not consume those fields.
