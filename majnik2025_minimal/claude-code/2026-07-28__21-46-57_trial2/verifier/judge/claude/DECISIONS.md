# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories starting with `jm` in the data directory. Sessions are subdirectories within each subject folder. Neural data is loaded from suite2p output files (`F.npy`, `Fneu.npy`), motion energy from `motion_energy_glob.npy`, and timestamps from `tstamps.npy`. All sessions are processed in a first pass to collect motion energy values for global discretization, then assembled into the final structure.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

# Per session:
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The AI followed the standard directory structure (subject folders containing session subfolders). It uses `tstamps.npy` (rather than `interframe_int.npy`) for dropped frame detection. The two-pass approach (process all sessions, then discretize globally) ensures consistent binning.

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

iii. Each subdirectory contains suite2p output and motion energy files for one recording session. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. The AI defines trials as 120-second (2-minute) non-overlapping segments of the continuous recording, citing the paper's description of "consecutive 2 minute blocks." After 10-frame temporal binning, each trial contains 360 bins (120s * 30Hz / 10). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 120  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
BIN_SIZE = 10
bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])
```

iii. The AI cites the paper: "splits were done on consecutive 2 minute blocks of the recording." CONVERSION_NOTES.md states: "Following the paper's cross-validation approach."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks are included.

ii. N/A

iii. No quality control criteria for trials are mentioned in the paper or by the AI.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), from `plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil subtraction (`Fc = F - 0.7 * Fneu`), (2) a custom re-implementation of suite2p's maximin baseline estimation (gaussian smooth -> running min -> running max), and (3) dF/F computation by dividing by baseline: `(Fc - F0) / F0`. This differs from the reference which uses suite2p's `dcnv.preprocess` directly, which performs baseline *subtraction* (Fc - F0) without dividing by F0. After dF/F, the AI also applies 10-frame temporal binning (averaging).

ii.
```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                win_baseline=WIN_BASELINE, sig_baseline=SIG_BASELINE):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)  # 1800 frames
    sig = int(sig_baseline * fs)  # 300 frames

    # Maximin baseline (Suite2p default)
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)

    Flow = np.maximum(Flow, 1e-6)
    dff = (Fc - Flow) / Flow
    return dff.astype(np.float32)

# Then temporal binning:
dff_binned = bin_data(dff, BIN_SIZE)  # average in 10-frame bins
```

iii. CONVERSION_NOTES.md states: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and documents the maximin method. The AI re-implemented the baseline computation rather than using suite2p's `dcnv.preprocess`, and added division by baseline to produce dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the suite2p `F.npy` output are included. No `iscell.npy` filtering.

ii. N/A (no filtering code present)

iii. The AI notes that Suite2p's cell detection pipeline already identifies ROIs. Neuron counts (221-746 per mouse, mean ~500) are consistent with the paper's reported statistics (526 +/- 190).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to recording onset. Since trials are contiguous segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'recording_onset',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),  # 120.0
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal binning, averaging consecutive frames into bins of 10. This produces a time bin size of 333.33 ms (10 frames / 30 Hz * 1000). Both neural and behavioral data are binned identically.

ii.
```python
BIN_SIZE = 10  # frames per bin (paper: "averaging in bins of 10 consecutive timestamps")

def bin_data(data, bin_size):
    n = data.shape[-1]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    if data.ndim == 1:
        return data[:n_use].reshape(n_bins, bin_size).mean(axis=1)
    else:
        return data[:, :n_use].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)

# Metadata:
'time_bin_size': BIN_SIZE / FS * 1000,  # 333.33 ms
```

iii. CONVERSION_NOTES.md cites the paper: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and known frame rate/bin size.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. Since the frame rate is constant at 30 Hz and binning is deterministic, time can be computed from bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each temporal bin in seconds from session start: `(bin_index * 10 + 5) / 30`. Each trial's time array reflects its position within the session (not reset per trial).

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
# Then split into trials:
time_trials.append(time_bins[start:end])
# Stored as (1, n_timepoints):
sess_input.append(t_trial.reshape(1, -1))
```

iii. The bin center offset (BIN_SIZE/2) provides the temporal midpoint of each bin rather than the start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned with neural data by construction -- both are indexed by the same bin indices after temporal binning.

ii. N/A (alignment is implicit through shared indexing)

iii. Since both neural and time data derive from the same binning scheme, they are inherently aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Frame timestamps from `tstamps.npy` are used to detect and interpolate dropped frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video. The timestamp file is needed to identify dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped frames are detected via timestamp gaps and linearly interpolated using `np.interp`, (2) motion energy is averaged in 10-frame bins (same as neural data), (3) the continuous signal is discretized into 5 equal-percentile bins computed globally across all sessions. Notably, the AI does NOT normalize motion energy by its standard deviation before discretization.

ii.
```python
# Interpolation:
def interpolate_motion_energy(me, tstamps, n_frames):
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
    return me_interp

# Binning:
me_binned = bin_data(me, BIN_SIZE)

# Discretization (no std normalization):
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
binned = np.digitize(me_values, thresholds[1:-1])
```

iii. CONVERSION_NOTES.md documents the interpolation approach and notes up to 116 dropped frames in one session. Global percentile-based discretization ensures balanced class counts.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins using quintile thresholds computed globally across all binned motion energy values from all sessions. Values are assigned to bins 0-4 using `np.digitize`.

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

iii. The task specification requires "five equal-percentile bins." Global discretization ensures consistent bin definitions across sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. Dropped video frames are detected using timestamp gaps (comparing interframe intervals to the median interval) and filled via linear interpolation (`np.interp`). After interpolation, both streams have the same number of frames and are binned identically in 10-frame bins.

ii.
```python
def interpolate_motion_energy(me, tstamps, n_frames):
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
    return me_interp

# Called with expected neural frame count:
me = interpolate_motion_energy(me, tstamps, n_frames)
```

iii. The median-based gap detection is more principled than a fixed threshold. Linear interpolation via `np.interp` fills gaps smoothly. After interpolation, motion energy and neural data share the same frame indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected via timestamp gaps and interpolated (see 4-d). Remainder frames/bins that don't fill a complete trial are discarded. No explicit handling of other data quality issues.

ii.
```python
# Dropped frame interpolation (see 4-d)
me = interpolate_motion_energy(me, tstamps, n_frames)

# Remainder truncation in bin_data:
n_use = n_bins * bin_size  # truncate to multiple of bin_size

# Remainder truncation in trial splitting:
n_trials = n_bins // bins_per_trial  # discard incomplete trials
```

iii. The interpolation ensures motion energy matches neural frame count. Discarding remainder is minor data loss (at most one incomplete trial per session).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the custom maximin baseline computation, which involves gaussian smoothing, running minimum, and running maximum over the full session for every neuron. Unlike the reference which uses suite2p's GPU-accelerated `dcnv.preprocess`, the AI's implementation runs on CPU using scipy functions.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. These sliding window operations over the full session length for every neuron are computationally intensive, especially without GPU acceleration.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation loop iterates frame-by-frame to build the frame index mapping. This could be vectorized using cumulative sums of rounded interval ratios.

ii.
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
```

iii. The number of frames is large (tens of thousands) but the loop is simple, so the performance impact is moderate.

## 6-c. What processing does the code repeat multiple times?

i. N/A -- the code uses a two-pass structure (process all sessions, then discretize) but does not repeat any processing.

ii. N/A

iii. The two-pass approach is necessary for global percentile computation and is not redundant.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporal binning (10-frame averaging) is applied as a pre-processing step. While the paper describes this for decoding, the reference solution keeps data at native resolution, allowing downstream code to apply its own binning if desired. Pre-binning reduces flexibility.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The AI followed the paper's description of 10-frame binning for decoding. However, this binning could be considered part of the decoder pipeline rather than data conversion.
