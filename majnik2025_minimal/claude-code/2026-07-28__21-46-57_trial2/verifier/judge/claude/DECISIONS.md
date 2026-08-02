# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory. For each subject, sessions are subdirectories sorted alphabetically. For each session, neural data is loaded from suite2p output files (`F.npy`, `Fneu.npy` in `suite2p/plane0/`), motion energy from `move_deve/motion_energy_glob.npy`, and timestamps from `move_deve/tstamps.npy`. All sessions are processed in a first pass, then discretization and trial splitting are applied.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

# For each subject:
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])

# For each session:
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The AI follows the standard directory structure convention. It uses `tstamps.npy` for dropped frame interpolation rather than `interframe_int.npy` used by the reference. All `jm*` directories are included as subjects.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data directory, sorted alphabetically. Each directory represents one mouse.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. The naming convention `jm*` is consistent across the dataset and matches the paper's subject identifiers.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, sorted alphabetically. Each subdirectory contains one daily recording with its suite2p output and motion energy files.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. Each subdirectory contains a complete recording session's data. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. The AI splits data into 2-minute (120-second) non-overlapping trials, citing the paper's cross-validation scheme ("splits were done on consecutive 2 minute blocks of the recording"). After 10-frame temporal binning, each trial has 360 time bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 120  # 2 minutes per trial
BIN_SIZE = 10  # frames per bin

bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
n_bins = dff_binned.shape[1]
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
```

iii. The AI interpreted the paper's cross-validation block structure as the trial structure. The paper states "splits were done on consecutive 2 minute blocks of the recording" for cross-validation, and the AI used this as the trial duration.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering or quality controls are applied. All complete trials (those filling a full 2-minute block) are included.

ii. N/A (no filtering code)

iii. The AI did not implement any trial-level quality filtering. This matches the reference, which also does not filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), from the `suite2p/plane0/` directory of each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil correction `Fc = F - 0.7 * Fneu`, (2) maximin baseline estimation (Gaussian smooth -> running minimum -> running maximum), and (3) dF/F computation `(Fc - baseline) / baseline`. This is followed by temporal binning in 10-frame windows. Notably, the AI reimplements the maximin baseline method manually using scipy rather than using suite2p's `dcnv.preprocess`, and divides by the baseline (dF/F) rather than just subtracting it.

ii.
```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF, ...):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)   # 1800 frames
    sig = int(sig_baseline * fs)   # 300 frames

    # Maximin baseline
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)

    Flow = np.maximum(Flow, 1e-6)
    dff = (Fc - Flow) / Flow
    return dff.astype(np.float32)

# Then binned:
dff_binned = bin_data(dff, BIN_SIZE)  # 10-frame averaging
```

iii. The AI cites the paper's methods: "We used baseline corrected fluorescence traces as our dF/F using the default Suite2p parameters." The AI interpreted "dF/F" literally as requiring division by baseline. The 10-frame binning comes from the paper: "averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural data quality filtering is applied. All neurons from the suite2p `F.npy` output are included.

ii. N/A (no filtering code)

iii. Suite2p's cell detection already identifies ROIs. No further filtering (e.g., by `iscell`) was applied. This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the recording onset. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed. The temporal alignment event is described as `recording_onset`.

ii.
```python
'temporal_alignment_event': 'recording_onset',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging in 10-frame bins, resulting in a time bin size of 333.33 ms (10 frames / 30 Hz). This is applied to both neural and behavioral data.

ii.
```python
BIN_SIZE = 10  # frames per bin
# ...
'time_bin_size': BIN_SIZE / FS * 1000,  # in ms = 333.33 ms

def bin_data(data, bin_size):
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    return data[:, :n_use].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)
```

iii. The AI cites the paper: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The reference solution does NOT apply this binning and keeps the native 30 Hz resolution.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin indices and the frame rate, representing time from the start of the recording session.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. Since the frame rate is constant at 30 Hz, computing time from bin indices is straightforward. The AI uses the center of each bin as the time value.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each temporal bin: `(bin_index * 10 + 5) / 30` seconds. This gives time elapsed from the start of the recording session. When split into trials, each trial retains its session-relative time values (e.g., trial 2 starts at ~120s, not at 0s).

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
# ...
time_trials.append(time_bins[start:end])  # preserves session-relative time
```

iii. The AI interpreted "time from beginning of experiment" as time from recording onset. Unlike the reference, which resets time to 0 at the start of each trial, the AI preserves session-relative timestamps.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same bin indices as the neural data after temporal binning, so they are inherently aligned. Each time bin center corresponds to the matching neural data bin.

ii.
```python
# Both computed from same binned indices
dff_binned = bin_data(dff, BIN_SIZE)
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. Since both are derived from the same temporal binning, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used for dropped frame interpolation.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video. The reference uses `interframe_int.npy` instead of `tstamps.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) Dropped frames are detected using interframe intervals from `tstamps.npy` and filled via `np.interp` (linear interpolation), (2) temporal binning in 10-frame windows (averaging), (3) discretization into 5 equal-percentile bins computed globally across all sessions. No explicit normalization is applied (the AI argues percentile binning is invariant to normalization).

ii.
```python
# Interpolation
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
# Maps camera frames to 2p frame indices and interpolates
me_interp = np.interp(all_indices, frame_indices, me)

# Binning
me_binned = bin_data(me, BIN_SIZE)

# Global discretization
all_me_concat = np.concatenate(all_me_flat)
thresholds = np.percentile(all_me_concat, percentiles)
binned = np.digitize(me_values, thresholds[1:-1])
```

iii. The AI notes that percentile-based discretization is invariant to normalization, so no separate normalization step is needed. The reference normalizes by standard deviation before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles). Percentile thresholds are computed globally across all sessions (after temporal binning). Values are assigned to bins 0-4 using `np.digitize`.

ii.
```python
def discretize_motion_energy(all_me_values, n_bins=N_BINS_OUTPUT):
    percentiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(all_me_values, percentiles)
    return thresholds

def apply_discretization(me_values, thresholds):
    n_bins = len(thresholds) - 1
    binned = np.digitize(me_values, thresholds[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. Global percentile-based discretization ensures consistent bin definitions across sessions and produces approximately equal class counts.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated to match the number of neural frames (filling dropped camera frames), then both are temporally binned using the same 10-frame windows. After binning, the same indices are used to split both into trials.

ii.
```python
me = interpolate_motion_energy(me, tstamps, n_frames)  # match to neural frame count
me_binned = bin_data(me, BIN_SIZE)  # same binning as neural
# Same trial splitting indices used for both
```

iii. The interpolation step ensures frame-for-frame correspondence before binning. The same binning and trial splitting indices guarantee alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected using interframe intervals from `tstamps.npy`. Missing frames are filled by linear interpolation (`np.interp`). Remainder frames/bins at the end of a session that don't fill a complete trial are discarded.

ii.
```python
def interpolate_motion_energy(me, tstamps, n_frames):
    ifi = np.diff(tstamps)
    median_ifi = np.median(ifi)
    # Build cumulative frame index mapping
    for i in range(len(ifi)):
        n_skipped = int(np.round(ifi[i] / median_ifi))
        cum_idx += n_skipped
        frame_indices[i + 1] = cum_idx
    me_interp = np.interp(all_indices, frame_indices, me)
    return me_interp
```

iii. Linear interpolation is a reasonable approach for filling dropped frames. The reference uses a different method: detecting drops via `interframe_int.npy` with threshold `dt * 1000 > 0.04` and inserting the average of neighboring values.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline estimation for dF/F, which involves Gaussian filtering, running minimum, and running maximum operations on the full session data for every neuron. The AI uses scipy's `gaussian_filter1d`, `minimum_filter1d`, and `maximum_filter1d` on CPU, whereas the reference uses suite2p's GPU-accelerated `dcnv.preprocess`.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. Running sliding window operations over large arrays (hundreds of neurons by tens of thousands of frames) is computationally expensive. The reference benefits from GPU acceleration via suite2p.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation loop iterates frame-by-frame to build the cumulative frame index mapping. This could potentially be vectorized using cumulative sums.

ii.
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
```

iii. The number of frames is typically large (tens of thousands), but the loop is simple and fast in practice. The reference's loop over dropped frames (`np.insert` in a loop) has a similar issue.

## 6-c. What processing does the code repeat multiple times?

i. The code does not obviously repeat processing. The two-pass structure (first pass: load and process, second pass: assemble into format) is clean. However, the motion energy trial data is iterated over twice: once when collecting into `all_me_flat` and again when applying discretization in the assembly pass.

ii.
```python
# First pass - collect
for me_t in me_trials:
    all_me_flat.append(me_t)

# Second pass - discretize
for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
```

iii. This is a minor redundancy; the two-pass approach is needed because global thresholds must be computed before discretization can be applied.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes bin center times for the time input, which is marginally useful. It also stores extensive metadata (thresholds, parameters) that may not be used by the decoder. The temporal binning itself could be considered unnecessary if the decoder can handle native-resolution data.

ii.
```python
'me_thresholds': thresholds.tolist(),
'neuropil_coefficient': NEUCOEFF,
'baseline_method': 'maximin',
'win_baseline_s': WIN_BASELINE,
'sig_baseline_s': SIG_BASELINE,
```

iii. Storing extra metadata is harmless and useful for documentation, but doesn't affect the decoder's operation.
