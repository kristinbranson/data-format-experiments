# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every directory under `data` as a mouse, alphabetically enumerates every subdirectory beneath each mouse as a session, and loads each session's `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. Full mode processes the resulting session list; sample mode keeps only its first two entries.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
for mouse_idx, mouse in enumerate(mice):
    sessions = sorted([d for d in os.listdir(mouse_dir)
                       if os.path.isdir(os.path.join(mouse_dir, d))])
    for sess in sessions:
        all_sessions.append((mouse, mouse_idx, sess_dir, sess))

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The notes say the explored hierarchy contains six mouse folders, 41 daily sessions, suite2p arrays, and motion-energy arrays. The agent treated that observed hierarchy as exhaustive and reported processing all 41 sessions in full mode.

## 1-b. How are the data split into subjects?

i. Each immediate directory under `data` is treated as one subject; names are sorted, stored in `subjects`, and their enumeration index is attached to every session from that directory.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))])
for mouse_idx, mouse in enumerate(mice):
    ...
    all_subject_idx.append(actual_mouse_idx)
```

iii. The notes identify the six directories `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` as mice and verify the session-to-subject mapping.

## 1-c. How are the data split into sessions?

i. Each immediate subdirectory of a mouse directory is a separate session. Sessions are sorted lexically and represented as separate outer-list entries in `neural`, `input`, and `output`.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                   if os.path.isdir(os.path.join(mouse_dir, d))])
for sess in sessions:
    sess_dir = os.path.join(mouse_dir, sess)
    all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

iii. The notes describe each subdirectory as one daily recording and validate 41 sessions, with six or seven sessions per mouse.

## 1-d. How are the data split into trials?

i. The continuous, 10-frame-averaged session is divided into non-overlapping **120-second** blocks. Only complete blocks are emitted; the tail is silently omitted through floor division. This conflicts with the instruction to use 60-second trials.

ii.
```python
trial_duration_sec = 120
effective_fs = fs / bin_size
trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. The agent justified two-minute blocks as “matching paper CV,” citing the paper's five-fold splits on consecutive two-minute blocks. It prioritized that paper detail over the decoder task's explicit 60-second-trial requirement.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-quality filtering. All complete 120-second blocks are retained; incomplete trailing data are dropped by floor division.

ii.
```python
n_trials = n_bins // trial_bins
for t in range(n_trials):
    ...
    neural_trials.append(neural_trial)
```

iii. The notes do not identify trial-level quality criteria. They report successful format validation but do not justify or explicitly document the discarded tail.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from plane 0 suite2p raw fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`; preprocessing parameters are read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. The notes identify these as the inputs used by Track2p's `F_processing` and suite2p's maximin baseline correction.

## 2-b. How is the `neural` data processed?

i. The agent subtracts 0.7 times neuropil fluorescence, estimates a maximin baseline by Gaussian smoothing followed by minimum and maximum filters, subtracts that baseline, and then averages non-overlapping groups of 10 frames. Session-specific suite2p settings are obtained from `ops` with standard defaults.

ii.
```python
Fc = F - neucoeff * Fneu
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
dff = Fc - Flow
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

iii. The notes say this was corrected after detecting an initially wrong filter order, window, and division operation. The final choice was justified by comparison with Track2p and suite2p source: Gaussian, minimum, maximum, then subtraction with a 60-second window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed in conversion; every row of `F.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. During exploration the agent checked `iscell.npy` and found every supplied ROI had `iscell[:,0] == 1` and probability above 0.5, so it considered additional filtering unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The continuous neural series begins at recording-session start and is sliced into consecutive blocks using common indices. Metadata calls session start the alignment event and gives `off_start = 0.0`; there is no event-centered realignment.

ii.
```python
start = t * trial_bins
end = (t + 1) * trial_bins
neural_trial = dff_binned[:, start:end]
...
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
```

iii. The implicit rationale is that these are continuous recordings rather than event-driven trials. The notes validate trial boundaries and session-relative time but do not discuss a separate biological event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz samples are averaged into each bin, producing 3 Hz data and a nominal bin size of 333.33 ms. A short tail of fewer than 10 frames is discarded.

ii.
```python
bin_size = 10
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
...
'time_bin_size': 1000.0 * bin_size / 30.0,
'effective_frame_rate_hz': 3.0,
```

iii. The agent cites the methods statement that neural and behavioral traces were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the post-binning sample index and suite2p frame rate (`ops['fs']`); it is not loaded from behavioral timestamps.

ii.
```python
fs = ops.get('fs', 30.0)
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
```

iii. The notes map “Time index” to the decoder input and describe it as time in seconds at 3 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Integer bin indices are divided by the effective sampling frequency. Each trial receives its slice of this session-continuous vector, reshaped to one row and cast to `float32`.

ii.
```python
time_vec = np.arange(n_bins) / effective_fs
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. The agent's validation notes say the input values and trial boundaries were checked and correct; no additional transformation was considered necessary.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural activity are generated at the same binned resolution and sliced with identical `start:end` indices, so each time value labels the corresponding neural bin.

ii.
```python
neural_trial = dff_binned[:, start:end]
time_trial = time_vec[start:end].reshape(1, -1)
```

iii. The notes report a direct check of input time and trial boundaries. The common index and effective frame rate were intended to ensure alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived solely from each session's precomputed `move_deve/motion_energy_glob.npy`. Unlike the reference, the agent does not load `interframe_int.npy` to locate dropped frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
...
me = data['motion_energy']
```

iii. The notes identify global motion energy from videography as the behavioral target and state only that missing camera frames should be interpolated.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If its length differs from neural data, motion energy is linearly resampled across the whole normalized recording interval. It is then averaged in non-overlapping 10-frame bins. After complete trials are selected, all retained values from every session are concatenated for categorical discretization.

ii.
```python
x_orig = np.linspace(0, 1, len(me))
x_target = np.linspace(0, 1, n_target)
me_interp = np.interp(x_target, x_orig, me.astype(float))
...
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
all_values = np.concatenate(all_values)
bin_edges = np.percentile(all_values, percentiles)
```

iii. The notes say missing camera frames are interpolated, both behavior and neural traces are averaged by 10 frames, and five equal-percentile bins are used. They explicitly record the choice as “globally,” without reconciling it with “selected per session.”

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Percentiles 0, 20, 40, 60, 80, and 100 are calculated from the pooled retained motion-energy samples across all subjects and sessions. The four internal cut points are applied globally with `np.digitize`, yielding integer labels 0–4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
```

iii. The agent wanted globally balanced classes and reports exactly 20% in each bin. Its notes call this “5 equal-percentile bins globally,” despite the instruction requiring thresholds selected per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is forced to the neural frame count by global linear resampling, then both streams are independently averaged in 10-frame bins and sliced with the same trial indices. This equalizes lengths but shifts samples around each true dropped-frame location rather than locally repairing the drop.

ii.
```python
me = interpolate_missing_frames(me, n_frames)
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
neural_trial = dff_binned[:, start:end]
me_trial = me_binned[start:end].reshape(1, -1)
```

iii. The stated goal was to interpolate missing camera frames so behavior and neural data have matching lengths. The notes claim interpolation but provide no dropped-frame-location check or alignment validation against `interframe_int.npy`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Any motion-energy length mismatch is handled by whole-trace linear resampling to the neural length, regardless of mismatch direction or location. Partial 10-frame bins and incomplete 120-second trials are discarded implicitly. There are no assertions for unexpected files, lengths, NaNs, or empty sessions.

ii.
```python
if len(me) == n_target:
    return me
x_orig = np.linspace(0, 1, len(me))
x_target = np.linspace(0, 1, n_target)
return np.interp(x_target, x_orig, me.astype(float))
...
n_bins = n // bin_size
n_trials = n_bins // trial_bins
```

iii. The notes characterize the issue as missing camera frames and choose interpolation. They report format verification without errors, but do not document the loss of tails or justify global resampling instead of drop-local interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The agent's full-session maximin baseline calculation is the dominant conversion work: Gaussian, 1,800-frame minimum, and 1,800-frame maximum filters are applied to every neuron and frame. Loading hundreds of megabytes and optional plotting/recomputation also incur cost.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The notes emphasize exact baseline recomputation and record total processing time, but do not give a stage-by-stage profile. Their discussion of fixing the large 1,800-frame window supports identifying baseline filtering as the expensive stage.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial construction and output discretization use nested Python loops. Sessions cannot generally be stacked because neuron counts differ, but trials within a session could be reshaped in bulk, and `np.digitize` could be applied to each session array rather than trial by trial. Directory/session loops are appropriate for file I/O.

ii.
```python
for t in range(n_trials):
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
...
for session_trials in all_me_trials:
    for trial in session_trials:
        all_values.append(trial.flatten())
```

iii. The agent gives no efficiency justification for these loops. The notes focus on correctness and runtime totals, not vectorization.

## 6-c. What processing does the code repeat multiple times?

i. With `--show-processing`, the first two sessions are loaded again and their baseline-corrected neural traces, binned neural traces, interpolated motion energy, and binned motion energy are recomputed after conversion. Summary reporting also concatenates trial arrays again per session.

ii.
```python
sess_data = load_session_data(sess_dir)
dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_interp = interpolate_missing_frames(me, sess_data['n_frames'])
```

iii. This repetition supports optional diagnostic plots. The notes mention plot creation and visual checks, but do not call out the duplicated computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `ops.npy` is loaded wholesale although only four scalar settings are used. Optional plotting recomputes large arrays used only for PNG diagnostics. Conversion also stores global bin-edge metadata and detailed session summaries, which the decoder does not require, though they are useful provenance rather than harmful processing.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
...
if args.show_processing:
    dff = compute_dff(...)
    dff_binned = bin_data(dff, ...)
    plt.savefig(...)
```

iii. The agent intended the optional work for sanity checking and user-facing diagnostics. Its notes document plots and validation, but do not distinguish these artifacts from downstream decoder requirements.
