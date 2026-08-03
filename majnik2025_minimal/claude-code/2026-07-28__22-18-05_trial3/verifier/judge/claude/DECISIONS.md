# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory. Sessions are subdirectories within each subject folder. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/` for behavioral data.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir)
                      if os.path.isdir(os.path.join(mouse_dir, d))])
    for sess_name in sessions:
        F = np.load(os.path.join(s2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
        me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
        ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI identifies the standard directory structure (subject folders containing session subfolders with suite2p output and motion energy files). All `jm*` directories are treated as subjects, all subdirectories as sessions.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data directory, sorted alphabetically.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
subjects = mice
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. The AI splits continuous recordings into 2-minute (120-second) non-overlapping blocks. With 10-frame binning at 30Hz, each trial has 360 time bins. This is based on the paper's statement "splits were done on consecutive 2 minute blocks."

ii.
```python
TRIAL_DURATION_S = 120  # 2-minute blocks
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360 bins per trial

n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The AI cites the paper: "splits were done on consecutive 2 minute blocks of the recording." Remainder bins that don't fill a complete trial are discarded.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials are included.

ii. N/A (no filtering code)

iii. No justification needed as there is no natural quality metric for artificial trial segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), from `plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil subtraction with coefficient 0.7, (2) suite2p's `preprocess` with `maximin` baseline method, and (3) temporal binning by averaging 10 consecutive frames.

ii.
```python
def compute_dff_suite2p(F, Fneu, fs=30.0, neucoeff=0.7,
                        win_baseline=60.0, sig_baseline=10.0):
    Fc = (F - neucoeff * Fneu).astype(np.float32)
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                     device=torch.device('cpu'))
    return dff

# Then binned:
dff_binned = bin_data(dff, BIN_SIZE)  # BIN_SIZE = 10
```

iii. The AI cites the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and "we slightly denoised the dF/F ... by averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the suite2p `F.npy` output are included. The AI notes that all ROIs already have iscell=1.0 because Track2p provides only successfully tracked neurons.

ii. N/A (no filtering code)

iii. From CONVERSION_NOTES: "All ROIs in the dataset are already filtered (iscell[:,0] == 1.0 for all). Track2p provides only successfully tracked neurons across all sessions."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of each imaging session. Since trials are contiguous segments, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Start of imaging session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous and spontaneous.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames, going from 30Hz native rate to 3Hz (333.33ms bins). This is based on the paper's methods describing binning by 10 frames for the decoding analysis.

ii.
```python
BIN_SIZE = 10  # frames per bin
time_bin_size_ms = (BIN_SIZE / FS) * 1000  # ~333.33 ms

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The AI cites the paper: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed synthetically as the bin center time based on the bin index and the binning parameters.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)  # (1, n_timepoints)
```

iii. Since the frame rate is constant and bins are regularly spaced, computing time from bin indices is straightforward.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time is computed as the center of each bin: `(bin_index + 0.5) * (BIN_SIZE / FS)`. This gives time in seconds from the start of the session.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
```

iii. Using bin centers rather than bin edges provides a more accurate representation of the time associated with each binned data point.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time variable is constructed to be frame-aligned with the neural data by construction, since both use the same bin indices. Each time value corresponds to the center of the same bin used for neural data.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
    bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
```

iii. Alignment is guaranteed because the same `start:end` indices are used for both neural data and time computation.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect and interpolate dropped frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped frames are interpolated using a gap detection method based on interframe intervals (median IFI * 1.5 threshold), (2) temporal binning by averaging 10 consecutive frames, (3) discretization into 5 equal-percentile bins. Note: discretization is done per-session, not globally across all sessions. No explicit normalization by standard deviation is applied before discretization.

ii.
```python
# Interpolation
me = interpolate_missing_frames(me_raw, ifi, n_frames)

# Binning
me_binned = bin_data(me, BIN_SIZE)

# Discretization (per session)
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. The AI's CONVERSION_NOTES state: "Discretized into 5 equal-percentile bins (quintiles) per session."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using equal percentiles (quintiles). The bin edges are computed per session using `np.percentile` with `np.linspace(0, 100, 6)`. Values are then assigned to bins 0-4 via `np.digitize`.

ii.
```python
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])  # 0 to n_bins-1
    return bins, edges
```

iii. Equal-percentile binning ensures balanced class counts. The `+ 1e-10` on the last edge ensures the maximum value is included in the last bin.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Missing video frames are detected by comparing interframe intervals to a threshold of `median_ifi * 1.5`. Gap locations are identified, and the number of missing frames per gap is estimated by `round(ifi / median_ifi) - 1`. Interpolated values are inserted at gap positions using linear interpolation between neighboring frames. After interpolation, motion energy is binned with the same bin size as neural data.

ii.
```python
def interpolate_missing_frames(me, ifi, n_neural_frames):
    median_ifi = np.median(ifi)
    threshold = median_ifi * 1.5
    gap_indices = np.where(ifi > threshold)[0]
    missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
    # ... insert interpolated values at gap positions
    return result[:n_neural_frames]
```

iii. The approach uses a relative threshold (1.5x median IFI) rather than a fixed threshold. Linear interpolation fills gaps to match neural frame count.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are detected via interframe intervals and interpolated. If motion energy is longer than neural data, it is truncated. If after interpolation the length still doesn't match, the result is padded with the last value. Remainder frames/bins at the end of sessions that don't fill a complete trial are discarded.

ii.
```python
if n_missing <= 0:
    return me[:n_neural_frames]  # truncate

# After interpolation, pad if needed:
while dst_idx < n_neural_frames:
    result[dst_idx] = result[dst_idx - 1]
    dst_idx += 1
return result[:n_neural_frames]
```

iii. The truncation and padding ensure the motion energy signal always matches the neural data length regardless of edge cases.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` baseline correction, which runs on CPU (unlike the reference which uses GPU). Loading `.npy` files is also I/O bound.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
```

iii. Running on CPU (`torch.device('cpu')`) rather than GPU makes the baseline correction significantly slower.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation uses a loop with index tracking (`for src_idx in range(len(me))`) that iterates over every frame, not just gap positions. This could be vectorized using array operations.

ii.
```python
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    dst_idx += 1
    if gap_pos < len(gap_indices) and src_idx == gap_indices[gap_pos]:
        n_insert = missing_per_gap[gap_pos]
        for k in range(n_insert):
            ...
```

iii. While the loop iterates over all frames (tens of thousands), the actual computation is simple array assignment, so the performance impact is moderate.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any major processing steps. Each session is processed once through the pipeline (load, preprocess, bin, discretize, split into trials).

ii. N/A

iii. The pipeline is straightforward with no redundant processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins data by 10 frames before creating trials, which is not done in the reference solution. If the downstream decoder expects native 30Hz data, this binning would be unnecessary preprocessing. Additionally, the code creates a sample dataset that may not be needed.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE)  # BIN_SIZE = 10
me_binned = bin_data(me, BIN_SIZE)
```

iii. The binning was motivated by the paper's description of their decoding analysis, but the instructions don't explicitly require it and the reference solution does not bin.
