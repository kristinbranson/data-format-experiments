# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `jm*` directory under `data`, sorts its session subdirectories, flattens them into a subject/session list, and processes every entry in full mode. For each session it loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `tstamps.npy`. The resulting per-session trial lists are appended to the dataset.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
...
for subj in subjects:
    for sess in sessions[subj]:
        session_list.append((subj, sess))
...
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The agent states that this directory layout contains six mice and 41 daily sessions and reports validating those totals. It chose actual recording lengths (20 or 30 minutes) rather than truncating sessions to the paper's nominal duration.

## 1-b. How are the data split into subjects?

i. A subject is each sorted directory whose name begins with `jm`. Only subjects actually represented in the chosen session list are saved, and each session receives the index of its subject in `used_subjects`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
...
if subj not in used_subjects:
    used_subjects.append(subj)
subject_idx_list.append(used_subjects.index(subj))
```

iii. The notes identify the `jm*` folders as the six mice and verify 6 subjects with session counts 7, 7, 7, 7, 6, and 7.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject directory is treated as one session. Each becomes one outer element of `neural`, `input`, and `output`.

ii.
```python
sess_list = sorted([s for s in os.listdir(subj_path)
                    if os.path.isdir(os.path.join(subj_path, s))])
sessions[subj] = sess_list
...
all_neural.append(neural_trials)
all_input.append(input_trials)
all_output.append(output_trials)
```

iii. The agent interprets the dated subfolders as daily recordings and says the resulting 41 sessions match the supplied data.

## 1-d. How are the data split into trials?

i. The agent splits each binned continuous session into non-overlapping 120-second blocks of 360 binned samples. Incomplete tails are implicitly discarded by integer division. This conflicts with the evaluated instruction to use 60-second trials.

ii.
```python
TRIAL_DURATION_SEC = 120
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE
...
n_trials = n_timepoints // trial_length
for i in range(n_trials):
    start = i * trial_length
    end = start + trial_length
    trials.append(data[:, start:end].astype(np.float32))
```

iii. The notes justify two-minute blocks because the paper used consecutive two-minute blocks for cross-validation. That rationale overlooks the newer, explicit decoder-task instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-quality filter. Every complete 120-second block is retained; only the final incomplete remainder would be dropped.

ii.
```python
n_trials = n_timepoints // trial_length
for i in range(n_trials):
    ...
```

iii. The agent found no explicit trial-curation rule in the paper and documented “No explicit trial curation mentioned.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from Suite2p `F.npy` and `Fneu.npy` in each session's `suite2p/plane0` directory.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The notes identify these as raw fluorescence and neuropil fluorescence and say this matches the paper and loading notebook.

## 2-b. How is the `neural` data processed?

i. It subtracts 0.7 times neuropil fluorescence, applies Suite2p `preprocess` with a maximin baseline (60-second window, Gaussian sigma 10 frames, 30 Hz), and averages non-overlapping groups of 10 frames. Despite the `dff` name, the documented operation is baseline subtraction, not division by baseline.

ii.
```python
Fc = F - neucoeff * Fneu
device = torch.device('cpu')
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
...
return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
```

iii. The agent says these are the Suite2p/paper parameters and explicitly notes that Suite2p's preprocessing performs baseline subtraction rather than literal division. Ten-frame averaging follows the decoding method's denoising step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filter is applied in conversion; all rows in `F.npy` are retained.

ii.
```python
n_neurons, n_neural_frames = F.shape
...
brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The agent inspected `iscell.npy`, found all entries classified as cells, and concluded the supplied Track2p output was already filtered and contained cells tracked across days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Fixed blocks start at session time zero and proceed contiguously; metadata calls the alignment event `session_start`.

ii.
```python
start = i * trial_length
end = start + trial_length
trials.append(data[:, start:end].astype(np.float32))
...
'temporal_alignment_event': 'session_start',
'off_start': 0.0,
```

iii. The dataset is continuous spontaneous behavior, so the agent uses session start as the only meaningful origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz frames are averaged into each sample, producing 333.33 ms bins (3 Hz). Short tails that do not fill a ten-frame bin are discarded.

ii.
```python
FS = 30.0
BIN_SIZE = 10
TIME_BIN_MS = (BIN_SIZE / FS) * 1000
...
data_trimmed = data[:, :n_bins * bin_size]
return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
```

iii. The agent cites the methods statement that both fluorescence and behavior were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated from the binned sample index, the 10-frame bin size, and the fixed 30 Hz sampling rate; no raw timestamp variable is used.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The agent describes this as elapsed seconds from session start. It chose the center of each bin, whereas the reference uses each bin's left edge.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It adds a five-frame half-bin offset to every binned index, divides by 30 Hz, casts each trial slice to `float32`, and reshapes it to `(1, time)`.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
...
trial_time = time_input[start:end].astype(np.float32)
input_trials_list.append(trial_time.reshape(1, -1))
```

iii. The code comment says each binned time point represents the bin center. The notes verify a range beginning near 0.17 seconds.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is created after neural binning with exactly one value per neural bin, then sliced using the same trial start and end indices. It remains absolute session time across trials, but labels samples by bin centers while neural values are ten-frame averages.

ii.
```python
n_binned = dff_binned.shape[1]
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
...
start = i * TRIAL_FRAMES_BINNED
end = start + TRIAL_FRAMES_BINNED
trial_time = time_input[start:end]
```

iii. The agent initially considered resetting time within trials, then correctly retained session-absolute time because the task asks for time from the beginning of the experiment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output starts from `move_deve/motion_energy_glob.npy`. Camera timestamps in `move_deve/tstamps.npy` are additionally loaded to infer missing camera frames.

ii.
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The notes describe motion energy as a precomputed video measure and timestamps as camera times used for dropped-frame alignment. The reference uses the closely related `interframe_int.npy` instead.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If motion energy is short, timestamp differences are compared with the median interval to map observed camera samples to neural-frame indices; missing positions are linearly interpolated. The aligned signal is averaged over 10 frames and then discretized per session.

ii.
```python
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
...
me_full = np.interp(indices, indices[valid], me_full[valid])
...
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The agent found up to 148 missing camera frames and revised an earlier absolute-timestamp approach after testing showed that interval gaps reproduced the known missing-frame count and improved decoder validation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. For each session independently, the 20th, 40th, 60th, and 80th percentiles of the already aligned and averaged motion-energy trace form thresholds. `np.digitize` assigns integer classes 0 through 4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
bin_edges = np.percentile(me_binned, percentiles)
binned = np.digitize(me_binned, bin_edges)
return binned.astype(np.int64)
```

iii. The task requires five equal-percentile bins selected per session. The agent reports checking that every session was approximately 20% in each class. Although the notes mention normalization during planning, the final code correctly omits it because percentile labels are invariant to monotonic affine normalization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is expanded to the neural raw-frame count using inferred dropped-frame positions and interpolation, then both streams are independently averaged in the same 10-frame groups and sliced with identical trial boundaries.

ii.
```python
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
...
trial_me = me_discrete[start:end].astype(np.int64)
```

iii. The agent treats imaging and video as synchronous 30 Hz streams and uses interpolation solely to restore dropped camera frames. It documents an explicit missing-frame test and neural/output alignment plots.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion-energy streams are reconstructed at missing frame positions by linear interpolation. Exact-length streams are passed through. Binning and trial splitting truncate incomplete tails. The code does not explicitly reject excess motion-energy samples, empty valid sets, or residual mismatches after timestamp mapping.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_full[neural_idx] = me.astype(np.float64)
me_full = np.interp(indices, indices[valid], me_full[valid])
...
n_trials = n_timepoints // trial_length
```

iii. The agent identified missing video frames during exploration, validated the inferred gap count against the length deficit, and reports checking that the converted neural data contain no NaN or infinite values.

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p maximin baseline preprocessing over every neuron and full session is the principal compute step; loading and saving the roughly 395 MB result are the main I/O costs. The code times dF/F and binning separately and forces preprocessing onto CPU.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
print(f"    dF/F computation: {time.time()-t1:.2f}s")
...
device = torch.device('cpu')
dff = preprocess(..., device=device)
```

iii. The notes measured roughly 0.4–0.6 seconds for neural preprocessing per sample session and about 45 seconds for full conversion, identifying this stage as dominant.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-camera-frame cumulative missing-frame loop can be replaced by vectorized interval ratios plus a cumulative sum. The three trial-building passes (one inside `split_into_trials`, then separate input and output loops) could use reshape/slicing or one shared loop. Subject indices could be built from a dictionary instead of repeated list searches.

ii.
```python
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
...
for i in range(len(neural_trials)):
    ... input_trials_list.append(...)
for i in range(len(neural_trials)):
    ... output_trials_list.append(...)
```

iii. The agent did not discuss these loops in its notes. They are small relative to baseline filtering, so vectorization would simplify or modestly speed conversion rather than change its dominant cost.

## 6-c. What processing does the code repeat multiple times?

i. Loading, baseline correction, motion alignment, binning, percentile computation, and trial slicing are repeated independently for every session. Within a session, the same trial boundaries are calculated separately for neural data, time input, and output.

ii.
```python
for sess_i, (subj, sess_name) in enumerate(session_list):
    neural_trials, input_trials, output_trials, n_neurons = process_session(...)
...
start = i * TRIAL_FRAMES_BINNED
end = start + TRIAL_FRAMES_BINNED
```

iii. The agent does not identify repeated work explicitly. Per-session processing is necessary because recordings and percentile thresholds are session-specific; repeated boundary construction is avoidable but cheap.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Core full-mode conversion does little discarded computation beyond temporary full-session arrays and copies/casts. If `--show-processing` is enabled, it creates five-panel diagnostic figures for the first two sessions; those plots and their selected trace slices are not used by the decoder. Session bin edges and continuous binned motion energy are also not saved after labels are formed.

ii.
```python
if show_processing and sess_idx < 2:
    plot_processing(...)
...
fig, axes = plt.subplots(5, 1, figsize=(16, 20))
...
plt.savefig(fname, dpi=100)
```

iii. The agent intentionally added plots for visual verification and reports reviewing them for anomalies. Thus this is optional validation work, not an accidental part of the normal full conversion.
