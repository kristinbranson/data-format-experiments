# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories starting with `jm` in the data directory. Sessions are subdirectories within each subject folder, sorted alphabetically. For each session, calcium data is loaded from suite2p output files (`F.npy`, `Fneu.npy` in `suite2p/plane0/`), and motion energy from `motion_energy_glob.npy` in `move_deve/`. Timestamps are loaded from `tstamps.npy` (also in `move_deve/`).

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The agent explored the directory structure to identify all subject and session directories, and found the relevant data files by examining what was available in each session directory.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data directory, sorted alphabetically.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. Each subdirectory contains suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as 120-second (2-minute) non-overlapping blocks of the continuous recording. At 30 Hz with 10-frame bins, this gives 360 bins per trial. The agent chose 120s based on the paper's mention of "consecutive 2 minute blocks" for cross-validation, rather than the 60-second trial duration specified in the task instructions.

ii.
```python
TRIAL_DURATION_S = 120  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
```

iii. The agent cited the paper's description of "consecutive 2 minute blocks" as the basis for trial duration, reasoning that this matches the cross-validation scheme used in the paper's decoding analysis.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials (those with enough bins to fill a full trial) are retained. Remainder bins that don't fill a complete trial are discarded.

ii. N/A (no filtering code)

iii. The agent did not implement any trial filtering. Since the data was already pre-processed by Track2p, the agent assumed all included data was of sufficient quality.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The agent re-implements suite2p's maximin baseline method from scratch: (1) neuropil subtraction with coefficient 0.7, (2) Gaussian smoothing with sigma = 300 frames, (3) running minimum filter with window = 1800 frames, (4) running maximum filter with window = 1800 frames, (5) dF/F computed as `(Fc - Flow) / Flow`. The baseline is clamped to a minimum of 1e-6 to avoid division by zero.

ii.
```python
Fc = F - neucoeff * Fneu

win = int(win_baseline * fs)  # 1800 frames
sig = int(sig_baseline * fs)  # 300 frames

Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)

Flow = np.maximum(Flow, 1e-6)
dff = (Fc - Flow) / Flow
```

iii. The agent reasoned that the paper's mention of "baseline corrected fluorescence traces as our dF/F" confirmed they should compute dF/F values this way. The agent read suite2p parameters from `ops.npy` to determine `neucoeff=0.7, baseline=maximin, win_baseline=60.0, sig_baseline=10.0`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. The agent noted that `iscell.npy` has all values as 1.0 (all ROIs already classified as cells), so no further filtering was needed.

ii. N/A (no filtering code)

iii. The data was already pre-filtered by the Track2p pipeline, so all ROIs in `F.npy` are tracked neurons across sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to recording onset. Since trials are contiguous blocks of the continuous recording with no stimulus events, no event-based alignment is needed. The metadata records `temporal_alignment_event: 'recording_onset'`.

ii.
```python
'temporal_alignment_event': 'recording_onset',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing from 30 Hz to 3 Hz (333.33 ms time bin). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10  # frames per bin
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)

'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33 ms
```

iii. The agent cited the paper's methods: "averaging in bins of 10 consecutive timestamps" for the decoding analysis.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from bin indices and the bin duration, giving seconds from the start of the recording session.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. Since the frame rate is constant at 30 Hz and bins are 10 frames wide, time can be computed directly from bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each bin: `(bin_index * 10 + 5) / 30` seconds. The time runs continuously across trials within a session (i.e., trial 2 starts where trial 1 ended, not reset to zero). Named `time_s` in the output.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS

for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
```

iii. The agent computed time as bin centers (adding half-bin offset) rather than bin left edges.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices as the neural data, so alignment is inherent. Both share the same number of time bins per trial.

ii. N/A (alignment is implicit via shared indexing)

iii. Since both neural data and time are derived from the same bin structure, no explicit alignment is needed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used to detect and interpolate dropped camera frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped frames are detected via timestamps and interpolated using `np.interp` to match the neural data length, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 percentile-based bins using **global** thresholds computed across all sessions.

ii.
```python
# Interpolation
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
# ... builds frame_indices mapping, then:
me_interp = np.interp(all_indices, frame_indices, me)

# Binning
me_binned = bin_data(me, BIN_SIZE)

# Global discretization
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
binned = np.digitize(me_values, thresholds[1:-1])
```

iii. The agent concatenated all motion energy values across all sessions to compute global percentile thresholds, rather than computing thresholds per session. The agent noted this caused skewed per-session distributions but accepted it as "expected behavior for global percentile binning."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using percentile thresholds computed **globally** across all sessions. `np.digitize` assigns values to bins 0-4 based on the 20th, 40th, 60th, and 80th percentile edges.

ii.
```python
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)  # global
# ...
binned = np.digitize(me_values, thresholds[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The agent used global percentiles, producing bins that are equal across the full dataset but potentially skewed within individual sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera and neural data are acquired at 30 Hz but the camera occasionally drops frames. The agent detects gaps using `tstamps.npy` by comparing interframe intervals to the median interval. It maps each camera frame to its corresponding 2p frame index and uses linear interpolation (`np.interp`) to fill in missing frames. After interpolation, both streams are binned together.

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

iii. The agent used `np.interp` for continuous linear interpolation rather than discrete insertion of averaged neighboring values. It computed the number of skipped frames from the ratio of each interframe interval to the median interval.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected and interpolated (see 4-d). Remainder bins at the end of a session that don't fill a complete trial are discarded. No other error handling is present (e.g., no assertions to verify lengths match after interpolation).

ii.
```python
# In bin_data:
n_bins = n // bin_size
n_use = n_bins * bin_size  # truncates remainder

# In interpolate_motion_energy:
if len(me) == n_frames:
    return me
# ... interpolation ...
```

iii. The agent treated remainder truncation as standard practice. The interpolation function returns the original array if lengths already match, and interpolates otherwise. There is no explicit assertion that the interpolated result matches the expected length.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the re-implemented baseline correction (`compute_dff`), which applies Gaussian filtering, minimum filtering, and maximum filtering over the full session length for every neuron. Loading `.npy` files is also I/O bound.

ii. N/A

iii. The baseline correction involves sliding window operations with large windows (1800 frames for min/max filters, 300 frames Gaussian sigma) over full session-length data.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation builds frame indices in a Python loop iterating over all interframe intervals. This could be vectorized using cumulative sum operations on the rounded interval ratios.

ii.
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
```

iii. The number of frames is typically large (~54000), so this loop could benefit from vectorization, though its impact is small relative to the baseline correction.

## 6-c. What processing does the code repeat multiple times?

i. Motion energy is processed in two passes: first collected during the initial session processing loop (where trials are split), then discretized globally in a second pass. The trial splitting is done in the first pass but the discretized output is applied in the second pass during data structure assembly.

ii.
```python
# First pass: process and split
for sess_name in sessions:
    dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
    neural_trials, me_trials, time_trials = split_into_trials(...)
    for me_t in me_trials:
        all_me_flat.append(me_t)

# Second pass: discretize and assemble
for sess_data in all_sessions_data:
    for me_trial in sess_data['me_trials']:
        me_disc = apply_discretization(me_trial, thresholds)
```

iii. The two-pass approach is necessary for global discretization since all ME values must be collected before computing global thresholds.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores additional metadata fields (`me_thresholds`, `imaging_rate_hz`, `bin_size_frames`, `trial_duration_s`, `n_percentile_bins`, `neuropil_coefficient`, `baseline_method`, `win_baseline_s`, `sig_baseline_s`) that are not part of the required data format. It also creates a separate `sample_data.pkl` file. Additionally, it prints extensive sanity check output that is not used downstream.

ii.
```python
'metadata': {
    ...
    'imaging_rate_hz': FS,
    'bin_size_frames': BIN_SIZE,
    'trial_duration_s': TRIAL_DURATION_S,
    'n_percentile_bins': N_BINS_OUTPUT,
    'me_thresholds': thresholds.tolist(),
    ...
}

sample = create_sample(data)
with open(sample_path, 'wb') as f:
    pickle.dump(sample, f)
```

iii. The extra metadata is informational and doesn't affect downstream analysis. The sample dataset creation is a convenience for testing.
