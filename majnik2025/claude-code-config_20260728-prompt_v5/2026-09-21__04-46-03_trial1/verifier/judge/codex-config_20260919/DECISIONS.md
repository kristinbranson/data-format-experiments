# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. At import time, the script finds sorted `jm*` subject directories under the fixed `/app/data` path. For each subject it finds sorted date-like subdirectories (names beginning with `2`), then loads each session separately. It loads `F.npy`, `Fneu.npy`, and `ops.npy` for neural processing and `motion_energy_glob.npy` and `tstamps.npy` for behavior. Full mode processes every discovered session; sample mode stops after two sessions globally.

ii.
```python
SUBJECTS = sorted([d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

def get_sessions(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d)) and d[0] == '2'])
    return sessions

F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'),
              allow_pickle=True).item()
me = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
```

iii. The notes say the dataset contains six `jm*` mice and 41 dated sessions in the documented directory hierarchy. The agent chose to use all available data, including 30-minute recordings despite the paper describing 20-minute sessions, because the task asks for all data and the longer data are present.

## 1-b. How are the data split into subjects?

i. Each sorted `jm*` directory is one subject. Its position in `SUBJECTS` is stored once per session in `subject_idx`.

ii.
```python
for subj_i, subject in enumerate(subjects_list):
    sessions = get_sessions(subject)
    ...
    all_subject_idx.append(subj_i)
```

iii. The notes identify the folder names as the six mouse IDs and report session counts of 6–7 per mouse, consistent with the paper and data README.

## 1-c. How are the data split into sessions?

i. A session is each sorted, date-like subdirectory beneath a mouse directory. Each recording is processed independently and appended as one target-format session.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d)) and d[0] == '2'])
...
all_neural.append(neural)
all_input.append(inp)
all_output.append(out)
```

iii. The agent states that every such folder is a daily recording containing matching Suite2p and movement files, and sorting gives deterministic chronological order.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into consecutive, non-overlapping 60-second trials after 10-frame binning. At 30 Hz this is 180 binned samples per trial. Any incomplete tail is silently omitted by integer division.

ii.
```python
BINS_PER_TRIAL = int(TRIAL_DURATION_S / BIN_DURATION_S)  # 180 bins
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    trials.append(data[:, start:end])
```

iii. There are no natural task trials in the spontaneous recording, so the agent followed the explicit instruction to split sessions into 60-second trials. The notes expect 20 or 30 complete trials for the available 20- or 30-minute sessions.

## 1-e. How are trials filtered based on quality controls?

i. No complete 60-second trial is quality-filtered. Only a session tail too short to make a complete trial is dropped.

ii.
```python
n_trials = n_bins // bins_per_trial
```

iii. The notes found no trial-curation rule in the paper because these are continuous spontaneous-behavior recordings rather than experimental trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from Suite2p `F.npy` and `Fneu.npy`; `ops.npy` supplies baseline parameters (`sig_baseline`, `win_baseline`, and `fs`).

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'ops.npy'),
              allow_pickle=True).item()
```

iii. The agent interpreted the paper's “baseline corrected fluorescence traces as our dF/F” as requiring raw and neuropil fluorescence with Suite2p default baseline parameters rather than `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The script subtracts 0.7 times neuropil fluorescence, Gaussian-smooths the result, applies running minimum then running maximum filters to estimate a maximin baseline, floors that baseline at 10, and computes `(Fc - baseline) / baseline_safe`. It then averages every 10 frames and casts each trial to `float32`. This ratio differs from the human reference, which uses Suite2p `dcnv.preprocess`'s baseline-subtracted output without division.

ii.
```python
Fc = F - NEUCOEFF * Fneu
smoothed = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
flow_min = minimum_filter1d(smoothed, size=win, axis=1)
baseline = maximum_filter1d(flow_min, size=win, axis=1)
baseline_safe = np.maximum(baseline, 10.0)
dff = (Fc - baseline) / baseline_safe
...
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The trajectory shows that the agent discovered `dcnv.preprocess` returns baseline-subtracted fluorescence, but then deliberately chose a “true normalized dF/F” interpretation and added division by the baseline. The notes claim this matches Suite2p and report sanity checks on the resulting traces, but that interpretation does not match the supplied human conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered in the conversion; all rows of `F.npy` are retained.

ii.
```python
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The agent checked that the released arrays contain tracked neurons and that all `iscell` values pass the 0.5 threshold, concluding the dataset was already curated and additional filtering would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Trial zero starts at session start, and later trials are consecutive 60-second chunks of the same binned stream. Metadata describes session start as the alignment event and gives offsets 0 to 60 seconds.

ii.
```python
start = t * bins_per_trial
end = start + bins_per_trial
trials.append(data[:, start:end])
...
'temporal_alignment_event': 'Session start (beginning of recording)',
'off_start': 0.0,
'off_end': TRIAL_DURATION_S,
```

iii. The agent reasoned that the dataset is continuous spontaneous behavior with no stimulus event, so the artificial trials should remain indexed from the beginning of the recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Raw 30 Hz data are averaged in non-overlapping groups of 10 frames, producing 333.33 ms bins (3 Hz). Shorter-than-10-frame tails are discarded.

ii.
```python
BIN_SIZE = 10
BIN_DURATION_S = BIN_SIZE / FRAME_RATE
trimmed = data[:, :n_bins * bin_size]
return trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
```

iii. The paper states that neural and behavior traces were denoised by averaging 10 consecutive timestamps, so the agent applied identical binning to both streams before trial splitting and categorization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthetic, not loaded from a raw variable: it is derived from the binned sample index and the fixed 10/30-second bin duration.

ii.
```python
time_input = np.arange(n_bins) * BIN_DURATION_S
```

iii. The agent used the known constant acquisition rate because the requested input is elapsed session time, and the neural bins provide its index directly.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A zero-based sequence of bin indices is multiplied by 1/3 second, split into the same 180-bin trials, reshaped to `(1, 180)`, and cast to `float32`. Time remains continuous across trial boundaries rather than resetting each minute.

ii.
```python
time_input = np.arange(n_bins) * BIN_DURATION_S
input_trials = split_into_trials(time_input, BINS_PER_TRIAL)
input_trials_formatted = [t_in.reshape(1, -1).astype(np.float32)
                          for t_in in input_trials]
```

iii. The notes explicitly verify a sample bin at 33.3333 seconds and state that the requested quantity is seconds from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One time value is created per binned neural column, and both arrays are split with the same trial length and boundaries.

ii.
```python
n_bins = dff_binned.shape[1]
time_input = np.arange(n_bins) * BIN_DURATION_S
neural_trials = split_into_trials(dff_binned, BINS_PER_TRIAL)
input_trials = split_into_trials(time_input, BINS_PER_TRIAL)
```

iii. The common binned index makes time and neural data one-to-one; the agent also reports checking continuity between trials.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is based on `move_deve/motion_energy_glob.npy`. The agent additionally loads `tstamps.npy` to resample sessions whose camera and neural frame counts differ; unlike the human reference, it does not use `interframe_int.npy`.

ii.
```python
me = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ts = np.load(os.path.join(sess_dir, 'move_deve', 'tstamps.npy'))
me_aligned = align_motion_energy(me, ts, n_frames)
```

iii. The data README says missing-frame indices can be found from either timestamps or interframe intervals and interpolated, so the agent selected timestamps and noted that camera acquisition was microscope-triggered.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. When lengths differ, timestamps are multiplied by 1000 and the entire motion trace is interpolated at an assumed 30 Hz neural time grid. The result is averaged in 10-frame bins, discretized per session using percentile edges, split into 60-second trials, reshaped, and cast to `int64`. The global timestamp resampling differs from the reference's local insertion at detected dropped-frame positions.

ii.
```python
cam_times_s = me_timestamps * 1000.0
neural_times_s = np.arange(n_neural_frames) / FRAME_RATE
me_aligned = np.interp(neural_times_s, cam_times_s, me)
...
me_binned = bin_data(me_aligned, BIN_SIZE)
me_discrete, bin_edges = discretize_motion_energy(me_binned)
```

iii. The agent justified interpolation from camera timestamps as a vectorized way to fill missing frames. However, it also observed that camera timestamps advance at about 33.6 ms while it constructs a 33.33 ms target grid; thus this implementation warps the whole trace in mismatched sessions rather than only filling gaps.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Six percentile edges from 0 through 100 are computed independently on each session's binned motion energy. The four internal edges are passed to `np.digitize`, yielding categories 0–4, then labels are clipped to that range.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(me_binned, percentiles)
labels = np.digitize(me_binned, bin_edges[1:-1], right=False)
labels = np.clip(labels, 0, n_bins - 1)
```

iii. This follows the decoder instruction to use five equal-percentile bins selected per session. The agent validated approximately 20% occupancy per class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Equal-length sessions are assumed already frame-aligned and copied unchanged. Unequal-length sessions are resampled by `np.interp` from recorded camera timestamps onto `np.arange(n_neural_frames)/30`; neural and motion are then independently averaged in the same groups of 10 and split at identical boundaries. Because the timestamp clock spans about 1209.67 seconds for a nominal 1200-second/36,000-frame session, this resampling shifts/warps samples throughout affected sessions and does not match the reference's dropped-frame insertion.

ii.
```python
if len(me) == n_neural_frames:
    return me.copy()
cam_times_s = me_timestamps * 1000.0
neural_times_s = np.arange(n_neural_frames) / FRAME_RATE
me_aligned = np.interp(neural_times_s, cam_times_s, me)
```

iii. The agent's stated intent was one-to-one alignment after interpolating missing camera captures, based on the README. The intent is reasonable, but the notes' claim that aligned traces overlay correctly was checked on full-frame sessions, where the function bypasses interpolation, and therefore did not validate the problematic branch.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames are handled through timestamp-based interpolation whenever motion energy is shorter than neural data. The code does not explicitly handle motion arrays longer than neural data, NaNs, missing files, or duplicate percentile edges. Incomplete temporal bins and incomplete final trials are truncated.

ii.
```python
if len(me) == n_neural_frames:
    return me.copy()
...
me_aligned = np.interp(neural_times_s, cam_times_s, me)
...
n_bins = n_frames // bin_size
...
n_trials = n_bins // bins_per_trial
```

iii. The notes identify the known data defect as dropped camera frames and cite the README's permission to interpolate them. They regard discarded tails as acceptable because only complete fixed-duration trials can be used. The chosen interpolation, however, does not reproduce the reference's local repair.

## 6-a. What are the most time-consuming steps of the code?

i. The full-array Gaussian, minimum, and maximum filters over every neuron and time point are the principal computation. Loading hundreds of megabytes and pickling the roughly 395 MB output are also substantial. With `--show-processing`, rendering plots adds optional overhead.

ii.
```python
smoothed = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
flow_min = minimum_filter1d(smoothed, size=win, axis=1)
baseline = maximum_filter1d(flow_min, size=win, axis=1)
...
pickle.dump(data, f, protocol=4)
```

iii. The notes identify maximin baseline processing as the expensive stage and estimate the full conversion at 40–60 seconds. They mention a “per-neuron loop,” although SciPy performs these filters through vectorized compiled routines rather than an explicit Python neuron loop.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. `split_into_trials` builds trials with a Python loop and is called separately for neural, output, and input. Each stream could be reshaped into a trial axis and then converted to the required list. The outer subject/session loops are appropriate because files and session shapes differ; output formatting list comprehensions could also be consolidated but are minor.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    trials.append(data[:, start:end])
```

iii. The agent emphasized that binning and missing-frame interpolation were vectorized. Its notes do not identify the trial-splitting loops, and incorrectly describe the already compiled baseline filters as a per-neuron Python loop.

## 6-c. What processing does the code repeat multiple times?

i. It invokes nearly identical trial-splitting logic three times per session, computes the same start/end indices in each invocation, and then traverses all three trial lists again to cast and reshape them. Session-level logging also scans arrays for output distributions.

ii.
```python
neural_trials = split_into_trials(dff_binned, BINS_PER_TRIAL)
output_trials = split_into_trials(me_discrete, BINS_PER_TRIAL)
input_trials = split_into_trials(time_input, BINS_PER_TRIAL)
...
output_trials_formatted = [o.reshape(1, -1).astype(np.int64) for o in output_trials]
input_trials_formatted = [t_in.reshape(1, -1).astype(np.float32) for t_in in input_trials]
neural_trials_formatted = [n.astype(np.float32) for n in neural_trials]
```

iii. The notes do not discuss repeated processing. The repetition is structurally simple and small relative to baseline filtering, but shared boundaries could have been computed once.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Per-session percentile edges are computed and optionally plotted but not stored in the converted dataset. Diagnostic output distributions, elapsed-time calculations, subject display-name lookup, and extensive printing are not used downstream. Motion energy is promoted to `float64` before ultimately becoming integer labels. With `--show-processing`, raw/intermediate arrays are plotted only for inspection.

ii.
```python
me = np.load(...).astype(np.float64)
me_discrete, bin_edges = discretize_motion_energy(me_binned)
all_outputs = np.concatenate(output_trials)
unique, counts = np.unique(all_outputs, return_counts=True)
elapsed = time.time() - t0
```

iii. The agent added these operations as sanity checks and visualization aids, not as decoder features. Its notes consider the checks evidence of correctness but do not explicitly identify their results as discarded.
