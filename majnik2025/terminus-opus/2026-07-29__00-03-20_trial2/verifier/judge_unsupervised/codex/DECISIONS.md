# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans the local `data/` directory, sorts all top-level subject folders, then sorts all session folders inside each subject. For each session it calls `load_session_data()`, which loads Suite2p fluorescence (`F.npy`, `Fneu.npy`), Suite2p metadata (`ops.npy`), and behavioral motion energy (`motion_energy_glob.npy`). Trials are not read from disk directly; they are created later by splitting each loaded session into fixed-length blocks.

ii. 
```python
def load_session_data(session_dir):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

```python
mice = sorted([d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))])
...
for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir) 
                      if os.path.isdir(os.path.join(mouse_dir, d))])
    for sess in sessions:
        sess_dir = os.path.join(mouse_dir, sess)
        all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

iii. In `CONVERSION_NOTES.md`, the agent says the data are organized as six mouse folders with 41 session folders, each containing `suite2p/plane0` and `move_deve`, and says this matches `data/README.md`. In trajectory steps 4-6 it explicitly explored the folder hierarchy before implementing the loader.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the top-level directories under `data/`. The script sorts those directory names and stores them in `subjects`; each session later carries the corresponding `mouse_idx`.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))])
```

```python
for mouse_idx, mouse in enumerate(mice):
    ...
    all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

```python
result = {
    ...
    'subjects': unique_mice,
    'subject_idx': subject_idx_arr,
```

iii. `CONVERSION_NOTES.md` Step 2 records six subject IDs (`jm031` ... `jm046`) and maps them to mice A-F using the dataset README. The trajectory shows the agent relied on the directory structure and alphabetical ordering described in `data/README.md`.

## 1-c. How are the data split into sessions?

i. Sessions are defined as the subject subdirectories under each mouse folder. The script sorts those directory names chronologically and treats each directory as one session.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir) 
                  if os.path.isdir(os.path.join(mouse_dir, d))])
for sess in sessions:
    sess_dir = os.path.join(mouse_dir, sess)
    all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

iii. The agent justified this from `data/README.md`, which says each subject folder contains one folder per recording day and that the folder name is the recording date. `CONVERSION_NOTES.md` Step 2 lists the per-mouse session counts found this way.

## 1-d. How are the data split into trials?

i. The script creates synthetic trials by splitting each continuous session into fixed 120-second chunks after 10-frame temporal binning. At 30 Hz with `bin_size=10`, this yields 3 Hz data and `trial_bins = 120 * 3 = 360` bins per trial. Sessions with 20 minutes become 10 trials; sessions with 30 minutes become 15 trials.

ii.
```python
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs

trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins
...
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent says: "Trials: 2-minute blocks (matching paper CV)." The trajectory and notes tie this choice to the methods text that says decoding used "5 fold splits on consecutive 2 minute blocks of the recording."

## 1-e. How are trials filtered based on quality controls?

i. There is no dedicated trial-quality filter. The only effective filtering is structural: after binning, the script keeps only the number of complete fixed-length blocks given by floor division, so any leftover partial tail at the end of a session is discarded.

ii.
```python
trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins
...
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. I did not find an explicit trial-QC rationale in the notes beyond the decision to use fixed-length blocks. The agent's notes focus on missing-frame interpolation and on the fact that decoder validation requires at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, with per-session preprocessing parameters read from `suite2p/plane0/ops.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

```python
dff = compute_dff(F, Fneu, 
                  neucoeff=neucoeff,
                  win_baseline=win_baseline,
                  sig_baseline=sig_baseline,
                  fs=fs)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `F.npy, Fneu.npy -> neural` and says this matches Suite2p baseline correction. In trajectory step 10 the agent explicitly decided to use `F.npy`/`Fneu.npy`, not `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The script applies neuropil correction and Suite2p-style baseline subtraction: `Fc = F - neucoeff * Fneu`, Gaussian smoothing, minimum filter, maximum filter, then subtraction of the estimated baseline. It then averages every 10 consecutive frames and casts each per-trial slice to `float32`.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
dff = Fc - Flow
```

```python
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. The agent's notes say it corrected an earlier mistake and now matches Suite2p exactly: Gaussian -> min -> max -> subtract, with a 60 s window and no division. Trajectory steps 75-78 document this correction and the explicit comparison to Suite2p and Track2p code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neural filtering is performed in `convert_data.py`. The script assumes the saved Suite2p traces already represent the tracked neurons to keep and never loads `iscell.npy` or `spks.npy`. In practice, every ROI in the dataset appears to pass the 0.5 cell threshold according to the agent's exploration notes.

ii.
```python
def load_session_data(session_dir):
    ...
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. `CONVERSION_NOTES.md` Step 1 says: "All ROIs in iscell.npy are classified as cells" and also says the provided data already contain tracked neurons with matched rows across days. Trajectory steps 10-11 show the agent inspected `iscell.npy` and concluded extra filtering was unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script treats the start of the recording session as the alignment event. Neural traces are segmented contiguously from the beginning of each session, and the metadata labels the alignment event as `start of recording session`.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
```

```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. The justification is implicit in the task context: these are spontaneous continuous recordings rather than event-locked behavioral trials. The notes call the input variable "Time in seconds from start of recording."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data are at 333.33 ms resolution because the script averages every 10 frames from 30 Hz data, producing an effective 3 Hz sampling rate.

ii.
```python
bin_size = 10
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

```python
'metadata': {
    'time_bin_size': 1000.0 * bin_size / 30.0,
    'bin_size_frames': bin_size,
    'original_frame_rate_hz': 30.0,
    'effective_frame_rate_hz': 3.0,
```

iii. `CONVERSION_NOTES.md` repeatedly justifies this with the methods text: for decoding, the paper says both neural and behavior traces were "averaged in bins of 10 consecutive timestamps." Trajectory steps 8 and 10 also cite that sentence.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not read from a dedicated timestamp file. It is synthesized from the number of binned neural frames and the imaging frame rate read from `ops.npy`.

ii.
```python
ops = data['ops']
fs = ops.get('fs', 30.0)
...
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
```

iii. The notes describe this as "Time in seconds at 3 Hz" and map it from the time index rather than from `tstamps.npy`. I did not find a deeper justification than using recording frame order plus known frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After neural rebinning, the script creates a session-long time vector in seconds using `np.arange(n_bins) / effective_fs`, then slices that vector into per-trial windows. Because the vector is built before splitting, trial 2 starts where trial 1 left off rather than resetting to zero.

ii.
```python
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says "`Time index -> input[0] -> Time in seconds at 3 Hz`." The trajectory frames this as the decoder input "time elapsed in seconds from start of recording."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is generated at the same post-binning sampling rate as the neural data and is sliced with the same `[start:end]` indices used for `neural_trial`. Each time trace therefore has exactly the same number of time points as its corresponding neural matrix.

ii.
```python
neural_trial = dff_binned[:, start:end].astype(np.float32)
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. The notes justify this indirectly by treating time as the decoder input paired with each neural bin. The verification logs confirm matching time lengths across trials.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. `CONVERSION_NOTES.md` Step 5 maps `motion_energy_glob.npy -> output[0]`. The trajectory also cites the dataset README that describes this file as processed behavioral motion energy from videography.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script optionally resamples motion energy to the neural frame count if lengths differ, bins it by averaging every 10 samples, keeps it as a continuous scalar trace until the final discretization step, and does **not** apply an explicit normalization step before binning into categories.

ii.
```python
def interpolate_missing_frames(me, n_target):
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp
```

```python
me = interpolate_missing_frames(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

iii. The key justification appears in trajectory step 68: the agent decided that equal-percentile discretization itself counted as "normalization" and therefore kept global percentile binning without a separate normalization stage.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The script concatenates all motion-energy values from all sessions and trials, computes global percentile cutpoints for five bins, then assigns each time point to a category with `np.digitize`. It labels the classes `bin0_lowest`, `bin1`, `bin2`, `bin3`, and `bin4_highest`.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)

bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
...
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. In trajectory step 68, the agent explicitly argues for global rather than per-session percentiles so that differences in activity level across sessions and mice are preserved.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first forced to the same total frame count as the neural recording via `interpolate_missing_frames`, then binned with the same 10-frame averaging used for neural data, and finally cut into the same fixed trial windows.

ii.
```python
me = interpolate_missing_frames(me, n_frames)
...
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
me_trial = me_binned[start:end].reshape(1, -1).astype(np.float32)
```

iii. The agent's notes say "Missing camera frames: interpolate" and point to the dataset README note that mismatched camera frames can be interpolated over. The final code simplifies that idea to whole-trace resampling.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles mismatched motion-energy lengths by globally interpolating the entire motion-energy vector to the neural frame count. It does not use `tstamps.npy` or `interframe_int.npy` to identify the exact drop locations, and it does not implement any other missing-data logic.

ii.
```python
def interpolate_missing_frames(me, n_target):
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp
```

```python
me = interpolate_missing_frames(me, n_frames)
```

iii. The notes and trajectory acknowledge that some recordings have missing camera frames and cite the dataset README. The final justification is pragmatic rather than exact: interpolate so that motion-energy length matches neural length.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant expensive step is per-session fluorescence preprocessing: loading large `F.npy`/`Fneu.npy` arrays and running Gaussian, min, and max filters across all neurons and frames. A second expensive path is optional plotting under `--show-processing`, which recomputes dF/F and binned motion energy for visualization.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

```python
for i, (mouse, mouse_idx, sess_dir, sess_name) in enumerate(all_sessions):
    ...
    neural_trials, input_trials, me_trials = process_session(...)
```

iii. The code itself prints per-session and total processing times. The notes also record that the big correction effort centered on matching the Suite2p preprocessing path, which is the heavy numerical part of the script.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious Python loops could be vectorized: the per-trial slicing loop in `process_session()` and the nested loops in `discretize_output()` that flatten every trial just to build one long list and then iterate again to digitize.

ii.
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    ...
    neural_trials.append(neural_trial)
    input_trials.append(time_trial)
    me_trials.append(me_trial)
```

```python
all_values = []
for session_trials in all_me_trials:
    for trial in session_trials:
        all_values.append(trial.flatten())
...
for session_trials in all_me_trials:
    session_output = []
    for trial in session_trials:
        binned = np.digitize(trial.flatten(), bin_edges[1:-1])
```

iii. I did not find an explicit justification in the notes for leaving these loops as Python loops. This is an inference from the code structure.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats dF/F computation when `--show-processing` is enabled, because it recomputes `compute_dff()` for plotting after already computing it in `process_session()`. It also traverses motion-energy data multiple times: once to collect all values for percentile edges, again to digitize them, and again in the summary logging.

ii.
```python
neural_trials, input_trials, me_trials = process_session(...)
```

```python
if args.show_processing:
    ...
    dff = compute_dff(sess_data['F'], sess_data['Fneu'],
                      neucoeff=ops.get('neucoeff', 0.7),
                      win_baseline=ops.get('win_baseline', 60.0),
                      sig_baseline=ops.get('sig_baseline', 10.0),
                      fs=ops.get('fs', 30.0))
```

iii. There is no explicit justification for the repeated work except convenience for diagnostics and plotting.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary processing is the diagnostic/plotting path under `--show-processing`, which recomputes dF/F, motion-energy binning, and visualization-only summaries that are not consumed by the decoder. Even without plotting, the script concatenates outputs and neural data to print distributions and ranges, then discards those temporary arrays.

ii.
```python
if args.show_processing:
    ...
    dff = compute_dff(...)
    dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
    ...
    plt.savefig(f'processing_{mouse}_{sess_name}.png', dpi=100)
```

```python
all_out_vals = np.concatenate([t.flatten() for s in result['output'] for t in s])
...
all_n = np.concatenate([t for t in result['neural'][i]], axis=1)
```

iii. I did not find an explicit written justification beyond sanity checking and documentation. The notes do describe these plots and summaries as validation aids rather than essential decoder inputs.
