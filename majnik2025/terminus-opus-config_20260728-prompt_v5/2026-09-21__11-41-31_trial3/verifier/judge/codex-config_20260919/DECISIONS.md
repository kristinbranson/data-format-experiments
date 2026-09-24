# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All sorted `jm*` directories and every immediate session subdirectory are enumerated. Per session, `F.npy`, `Fneu.npy`, `ops.npy`, `motion_energy_glob.npy`, and `tstamps.npy` are loaded.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The agent identified six mice and 41 recordings and used the directory convention as the complete dataset.

## 1-b. How are the data split into subjects?

i. Each sorted `jm*` directory is a mouse; sessions store its index in that list.

ii. `subj_idx = subjects.index(subj)`

iii. The notes identify each such directory as one mouse.

## 1-c. How are the data split into sessions?

i. Each sorted immediate subdirectory under a mouse is one daily session and one outer-list entry.

ii. `sessions = sorted([d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))])`

iii. Each dated directory contains one Suite2p recording and behavior stream.

## 1-d. How are the data split into trials?

i. After binning, sessions are cut into consecutive nonoverlapping 60-second (180-bin) trials; an incomplete tail is dropped.

ii.
```python
n_trials = n_timepoints // timepoints_per_trial
for t in range(n_trials):
    trials.append(data_2d[:, t*timepoints_per_trial:(t+1)*timepoints_per_trial].copy())
```

iii. The recordings are continuous, so this directly implements the requested artificial trial definition.

## 1-e. How are trials filtered based on quality controls?

i. No trial-quality filter is applied. Incomplete tails are omitted, and a session is skipped if it has fewer than two complete trials.

ii. `if len(neural_trials) < 2: continue`

iii. The latter enforces the target schema; no full-data session was affected.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `F.npy` and `Fneu.npy`, with parameters read from `ops.npy`.

ii. `F, Fneu, ops, me, tstamps = load_session_data(data_dir, subject, session)`

iii. The agent interpreted the paper's baseline-corrected fluorescence as neuropil-corrected fluorescence processed by Suite2p.

## 2-b. How is the `neural` data processed?

i. The code subtracts `0.7*Fneu`, runs Suite2p maximin preprocessing with `ops` settings/defaults, averages groups of 10 frames, and casts trials to float32.

ii.
```python
Fc = F.copy() - NEUCOEFF * Fneu
dff = preprocess(F=Fc.copy(), baseline=ops.get('baseline', 'maximin'),
                 win_baseline=ops.get('win_baseline', 60.0),
                 sig_baseline=ops.get('sig_baseline', 10.0),
                 fs=ops.get('fs', 30.0), prctile_baseline=ops.get('prctile_baseline', 8.0),
                 device=device)
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
```

iii. Inspection of Suite2p led the agent to use baseline subtraction, not division, matching the paper pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. All rows of `F` are retained; `iscell.npy` is not loaded.

ii. `n_neurons, n_neural_frames = F.shape`

iii. The agent verified that all provided ROIs already have `iscell=1` and are tracked cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Consecutive trials are referenced to session start; there is no biological-event alignment.

ii.
```python
'temporal_alignment_event': 'session_start',
'off_start': 0.0, 'off_end': float(TRIAL_DURATION_SEC),
```

iii. The recordings have no stimulus event, making session start the applicable origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are averaged into each 333.33-ms bin (3 Hz) for neural and behavior streams.

ii. `dff_binned = bin_data(dff, BIN_SIZE, axis=1)` and `'time_bin_size': 1000.0 / BINNED_RATE`

iii. This follows the paper's 10-timestamp decoding denoising.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthesized from trial and bin indices and the nominal 3-Hz binned rate, not a raw variable.

ii. `time_in_trial = t * TRIAL_DURATION_SEC + np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE`

iii. The agent used the constant 30-Hz neural clock to express seconds from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Bin positions become seconds, receive the trial offset, are cast to float32, and reshaped to `(1,180)`.

ii. `input_trials.append(time_in_trial.astype(np.float32).reshape(1, -1))`

iii. Checks confirmed 1/3-second increments and continuous session-relative time.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. One generated time coordinate corresponds to each identically indexed neural bin in each trial.

ii. `np.arange(TIMEPOINTS_PER_TRIAL) / BINNED_RATE`

iii. The agent verified identical 180-point trial lengths.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses `motion_energy_glob.npy`, plus `tstamps.npy` when camera and neural lengths differ.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The first is precomputed global video motion energy; timestamps were chosen to repair missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. It is resampled when short, averaged over 10 frames, split into complete trials, then categorized using quintiles computed over all retained trial values in that session.

ii.
```python
me_aligned = align_motion_energy(me, tstamps, n_neural_frames)
me_binned = bin_data(me_aligned.reshape(1, -1), BIN_SIZE, axis=1).flatten()
bin_edges = np.percentile(np.concatenate(me_binned_trials), np.linspace(0, 100, 6))
```

iii. Binning follows the paper; per-session quintiles follow the task.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 0/20/40/60/80/100 percentiles define five categories; interior edges are passed to `np.digitize`, clipped to 0–4, and cast to int64.

ii.
```python
bin_indices = np.digitize(me_trial, bin_edges[1:-1], right=False)
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
```

iii. This directly implements five equal-percentile bins selected per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Equal-length streams are used framewise. Otherwise camera timestamps (kiloseconds converted to seconds) are globally interpolated onto a synthetic exact 30-Hz neural clock; both streams are then identically binned and sliced.

ii.
```python
neural_times = np.arange(n_neural_frames) / FRAME_RATE
cam_times = tstamps * 1000.0
me_aligned = np.interp(neural_times, cam_times, me.astype(np.float64))
```

iii. The agent relied on the README's recommendation to interpolate missing frames, but only validated resulting shapes.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short camera streams are timestamp-interpolated to neural length. Incomplete final bins/trials are truncated; sessions below two trials are skipped. There is no explicit NaN policy.

ii. `return np.interp(neural_times, cam_times, me.astype(np.float64))`

iii. The notes document nine mismatched sessions and cite the README's interpolation suggestion.

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p maximin preprocessing is the principal compute cost; full-array loading is the main I/O cost.

ii. `dff = compute_dff(F, Fneu, ops)`

iii. The agent timed stages and reported about 23 seconds for all 41 sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial slicing, per-trial digitization, time creation, casting, and reshaping could largely use session-wide reshapes/vectorization.

ii. `for t in range(n_trials): ...` and `for me_trial in me_binned_trials: ...`

iii. The agent did not discuss this; these loops are small compared with Suite2p and lists are ultimately required.

## 6-c. What processing does the code repeat multiple times?

i. Each session repeats Torch import/device choice and separate passes for slicing neural/behavior, digitizing output, building time, casting, and reshaping.

ii. `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

iii. The notes give no special justification; most repetition follows the per-session schema.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It makes extra full-array/trial copies, emits extensive diagnostics, and optionally calculates normalized traces and plots that are absent from the pickle.

ii. `Fc = F.copy() - NEUCOEFF * Fneu`, `dff = preprocess(F=Fc.copy(), ...)`, and `if show_processing: plot_processing(...)`

iii. These were used for validation and visualization, not decoder features.
