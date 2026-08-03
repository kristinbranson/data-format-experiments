# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code scans `/app/data`, keeps every directory whose name starts with `jm` as a subject, then scans every subdirectory of each subject as a session. For each session it loads neural fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, and behavioral motion energy from `move_deve/motion_energy_glob.npy` plus `move_deve/tstamps.npy`. Trials are not loaded directly; they are created later from processed session-level arrays.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. `CONVERSION_NOTES.md` states that the dataset contains 6 `jm*` subjects and 41 daily sessions. The trajectory shows the agent inspecting the directory tree and concluding that each `jm*` folder is one mouse and each subdirectory is one daily recording.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the sorted `jm*` directories under the data root.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. In the trajectory and notes, the AI treated the `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` folder names as mouse IDs.

## 1-c. How are the data split into sessions?

i. Sessions are the sorted subdirectories inside each subject folder. Each session is processed independently and becomes one session entry in the output lists.

ii.
```python
subj_dir = os.path.join(data_dir, subj)
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])

for sess_name in sessions:
    sess_dir = os.path.join(subj_dir, sess_name)
    dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
```

iii. The AI justified this by treating each daily recording directory as one session, matching the folder layout it explored in the trajectory.

## 1-d. How are the data split into trials?

i. The AI does not use the full-resolution recording as trials. It first bins each session into 10-frame averages, then splits the binned session into consecutive 2-minute blocks. Any leftover bins that do not fill a full 2-minute block are dropped.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_S = 120
```

```python
bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
n_bins = dff_binned.shape[1]
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])
```

iii. `CONVERSION_NOTES.md` says this was chosen to match the paper’s decoding analysis phrase about “consecutive 2 minute blocks of the recording,” and the trajectory explicitly states the AI decided to use 2-minute blocks as trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI code applies no explicit trial-quality filter. It does, however, discard tail data implicitly in two places: `bin_data` truncates any frame remainder smaller than 10 frames, and `split_into_trials` truncates any remaining bins smaller than one 2-minute block.

ii.
```python
def bin_data(data, bin_size):
    n = data.shape[-1]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
```

```python
n_bins = dff_binned.shape[1]
n_trials = n_bins // bins_per_trial
```

iii. There is no separate quality-control justification in the notes. The AI’s written justification focuses on fixed binning and fixed 2-minute blocks rather than trial exclusion rules.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p fluorescence arrays `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. `CONVERSION_NOTES.md` says neural data follows the paper’s Suite2p default processing using raw fluorescence plus neuropil fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F manually. It subtracts neuropil (`F - 0.7 * Fneu`), estimates a “maximin” baseline with Gaussian smoothing plus running minimum plus running maximum, divides by that baseline to get dF/F, casts to `float32`, and then averages the result in non-overlapping 10-frame bins.

ii.
```python
Fc = F - neucoeff * Fneu

win = int(win_baseline * fs)
sig = int(sig_baseline * fs)

Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)

Flow = np.maximum(Flow, 1e-6)
dff = (Fc - Flow) / Flow
```

```python
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. In `CONVERSION_NOTES.md`, the AI says it followed the paper’s description of Suite2p default dF/F and also followed the paper’s decoding text about averaging in bins of 10 timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI includes every row from `F.npy`/`Fneu.npy` and does not use `iscell.npy`, ROI scores, or any per-neuron QC mask.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
```

iii. The trajectory shows the AI inspected `iscell.npy` and found all entries were cells, but the final code never uses `iscell.npy`; the effective choice was to keep all available neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to recording onset rather than to any within-session event. Trials are contiguous 2-minute blocks taken from the start of the session, after 10-frame temporal binning. The metadata labels the alignment event as `recording_onset`.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

```python
'metadata': {
    'temporal_alignment_event': 'recording_onset',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
}
```

iii. `CONVERSION_NOTES.md` says the decoder input is “Time in seconds from recording onset,” and the trial-structure section treats the recording as a continuous stream divided into consecutive blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame bins at 30 Hz, so each time point is 333.33 ms. Yes, temporal rebinning is applied to both neural data and motion energy.

ii.
```python
BIN_SIZE = 10
FS = 30
```

```python
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The notes explicitly justify this with the paper quote about “averaging in bins of 10 consecutive timestamps.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not derive decoder input time from a raw timestamp variable. It computes time from the binned frame index and the fixed imaging rate.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. `CONVERSION_NOTES.md` says the input is “Time in seconds from recording onset,” computed as `(bin_index * 10 + 5) / 30`.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a per-bin time axis using bin centers, then slices that vector into the same 2-minute trials used for neural and motion data, reshaping each trial to `(1, n_timepoints)`.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

```python
for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
```

iii. The notes explicitly justify bin-center timing as the decoder input representation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time vector is aligned by construction: the code computes one time axis for the binned session, splits it with the same start/end indices as the neural data, and stores both on the same trial grid.

ii.
```python
neural_trials.append(dff_binned[:, start:end])
time_trials.append(time_bins[start:end])
```

```python
for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
```

iii. The AI’s notes say both neural and behavioral traces are binned identically, and the code uses the same trial boundaries for all three streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy from `move_deve/motion_energy_glob.npy` and uses `move_deve/tstamps.npy` to infer dropped frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. In the notes, the AI says missing camera frames are handled by “detecting gaps using interframe intervals from `tstamps.npy`” and then interpolating.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code interpolates motion energy to the neural frame count using timestamp-derived frame indices, averages the interpolated signal in 10-frame bins, and later discretizes it globally. The notes also claim the motion energy is normalized before discretization, but the code shown in `convert_data.py` does not perform an explicit standard-deviation normalization step.

ii.
```python
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)

frame_indices = np.zeros(len(me), dtype=int)
frame_indices[0] = 0
cum_idx = 0
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx

all_indices = np.arange(n_frames)
me_interp = np.interp(all_indices, frame_indices, me)
```

```python
me_binned = bin_data(me, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` says dropped frames are repaired from `tstamps.npy`, both streams are binned in 10-frame windows, and motion energy is supposed to be normalized and discretized into global quintiles.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes global percentile thresholds across all collected motion-energy trial values and uses `np.digitize` to assign five categories `0` through `4`, labeled `bin_0` through `bin_4`.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
thresholds = np.percentile(all_me_values, percentiles)
```

```python
binned = np.digitize(me_values, thresholds[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes justify this with the task requirement to discretize motion energy into five equal-percentile bins and emphasize that the thresholds are computed globally across sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first interpolates the motion-energy trace to the neural frame count, then bins motion energy and neural traces with the same 10-frame window size, and finally splits both into the same 2-minute trial windows.

ii.
```python
me = interpolate_motion_energy(me, tstamps, n_frames)
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

```python
neural_trials.append(dff_binned[:, start:end])
me_trials.append(me_binned[start:end])
```

iii. `CONVERSION_NOTES.md` says the camera stream can be shorter because of dropped frames and that interpolation is used to place motion energy on the same frame grid as the imaging data before identical binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing behavioral samples by inferring dropped frames from timestamp gaps and interpolating motion energy to the neural frame count. It also silently drops incomplete trailing data because `bin_data` truncates leftover frames smaller than one bin and `split_into_trials` truncates leftover bins smaller than one 2-minute block. There is no explicit assertion that the repaired motion-energy length matches the neural length.

ii.
```python
if len(me) == n_frames:
    return me
...
all_indices = np.arange(n_frames)
me_interp = np.interp(all_indices, frame_indices, me)
```

```python
n_bins = n // bin_size
n_use = n_bins * bin_size
```

```python
n_trials = n_bins // bins_per_trial
```

iii. The notes justify this by saying some sessions have dropped camera frames and that the code maps camera frames to 2p frame indices and linearly interpolates to fill them.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps in the AI code are the session-wise neural preprocessing in `compute_dff` and the full-session motion interpolation/binned conversion. `compute_dff` applies Gaussian smoothing plus min/max filters over every neuron’s full trace, and motion interpolation loops over every timestamp interval before another pass for binning.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
```

iii. The notes emphasize denoising and dropped-frame repair as key processing stages; the code structure shows these are the dominant full-session numerical passes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-interval Python loop in `interpolate_motion_energy`, the repeated per-trial loops that append slices into Python lists, and the separate per-trial loop that applies discretization and reshape operations session by session.

ii.
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
```

```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])
```

```python
for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
    sess_output.append(me_disc.reshape(1, -1))
```

iii. The AI did not give an explicit optimization discussion in the notes; this follows directly from the implemented loops.

## 6-c. What processing does the code repeat multiple times?

i. The AI code makes one pass over all sessions to load/process them and collect motion-energy values, then a second pass over the stored session data to build the final output structure and apply discretization. It also repeats `subjects.index(subj)` for every session and repeatedly loops over trials to reshape time and output arrays.

ii.
```python
all_sessions_data = []
all_me_flat = []
...
for subj in subjects:
    ...
    dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
    ...
    for me_t in me_trials:
        all_me_flat.append(me_t)
```

```python
for sess_data in all_sessions_data:
    subj = sess_data['subject']
    subj_i = subjects.index(subj)
    ...
    for t_trial in sess_data['time_trials']:
        sess_input.append(t_trial.reshape(1, -1))
    ...
    for me_trial in sess_data['me_trials']:
        me_disc = apply_discretization(me_trial, thresholds)
        sess_output.append(me_disc.reshape(1, -1))
```

iii. No explicit justification is given in the notes. The repeated processing appears to come from the AI’s decision to compute global thresholds first and only then package the session/trial outputs.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI code performs auxiliary work that is not part of the final converted dataset used downstream: it runs `print_sanity_checks`, creates a separate sample dataset, stores session names in the intermediate `all_sessions_data` records without writing them into the final pickle, and prints aggregate statistics that are not consumed later.

ii.
```python
all_sessions_data.append({
    'subject': subj,
    'session': sess_name,
    'neural_trials': neural_trials,
    'me_trials': me_trials,
    'time_trials': time_trials,
    'n_neurons': n_neurons,
})
```

```python
print_sanity_checks(data)
...
sample = create_sample(data)
sample_path = os.path.join('/app', 'sample_data.pkl')
with open(sample_path, 'wb') as f:
    pickle.dump(sample, f)
```

iii. There is no explicit justification beyond validation and convenience. These steps support inspection/testing but are not needed by the downstream decoder when using the final full dataset.
