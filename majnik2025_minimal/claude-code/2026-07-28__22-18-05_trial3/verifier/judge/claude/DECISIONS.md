# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory. For each subject, all subdirectories are treated as sessions. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. Data is loaded using `np.load`.

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

iii. The AI follows standard directory conventions and loads all expected data files. The approach is consistent with the reference paper's data organization.

## 1-b. How are the data split into subjects?

i. Subjects are directories starting with `jm` in the data directory, sorted alphabetically. Each directory corresponds to one mouse.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
subjects = mice
```

iii. The `jm*` prefix convention identifies mouse directories. 6 mice are found: jm031, jm032, jm038, jm039, jm040, jm046.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject's folder is treated as a session, sorted alphabetically. Each session directory contains one daily recording.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. Each subdirectory contains the suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. The AI creates artificial trials as 2-minute (120-second) non-overlapping blocks. With 10-frame binning at 30 Hz, each trial contains 360 time bins. Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 120  # 2-minute blocks
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360 bins per trial

n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
```

iii. The AI justifies 2-minute blocks by citing the paper: "splits were done on consecutive 2 minute blocks of the recording." The CONVERSION_NOTES.md documents this decision.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials are included.

ii. N/A - no filtering code exists.

iii. The AI does not mention any trial quality criteria. All trials that fill a complete 2-minute block are retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) suite2p's `preprocess` function with `maximin` baseline correction (60s window, sigma=10), and (3) temporal binning by averaging 10 consecutive frames.

ii.
```python
def compute_dff_suite2p(F, Fneu, fs=30.0, neucoeff=0.7,
                        win_baseline=60.0, sig_baseline=10.0):
    Fc = (F - neucoeff * Fneu).astype(np.float32)
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                     device=torch.device('cpu'))
    return dff

# Then binning:
dff_binned = bin_data(dff, BIN_SIZE)  # BIN_SIZE = 10
```

iii. The AI cites the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and "we slightly denoised the dF/F... by averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. All ROIs in the F.npy file are included. The AI notes that all ROIs in the dataset already have iscell=1.0.

ii. N/A - no filtering code exists.

iii. From CONVERSION_NOTES.md: "All ROIs in the dataset are already filtered (iscell[:,0] == 1.0 for all). Track2p provides only successfully tracked neurons across all sessions."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the imaging session. Since trials are contiguous blocks from the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'Start of imaging session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames. With the 30 Hz imaging rate, this gives an effective resolution of 333.33 ms per time bin (3 Hz).

ii.
```python
BIN_SIZE = 10  # frames per bin
time_bin_size_ms = (BIN_SIZE / FS) * 1000  # ~333.33 ms

def bin_data(data, bin_size):
    if data.ndim == 2:
        n_neurons, n_time = data.shape
        n_bins = n_time // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)

dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The AI cites the paper's decoding section: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI reports time_bin_size as 333.33 ms in metadata.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin indices based on the frame rate and bin size.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)  # (1, n_timepoints)
```

iii. Since the frame rate is constant at 30 Hz and bin size is 10, the time of each bin is computed analytically.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input represents the time of each bin center from the start of the session (not the trial). The bin center is computed as `(bin_index + 0.5) * (BIN_SIZE / FS)`, giving time in seconds. This means time is cumulative across trials within a session.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
```
Where `start` and `end` are session-level bin indices (not trial-level).

iii. The +0.5 offset places the time at the center of each bin rather than the start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is perfectly aligned with the neural data because both use the same bin indices. Each time value corresponds exactly to the center of the corresponding neural activity bin.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
    bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
```

iii. Both neural and time use the same indexing scheme, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect dropped video frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped video frames are detected using interframe intervals and interpolated, (2) motion energy is temporally binned by averaging 10 consecutive frames (same as neural), (3) binned values are discretized into 5 equal-percentile bins per session.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
me_binned = bin_data(me, BIN_SIZE)
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. The AI follows the paper's description of denoising behavioral traces by binning, and applies percentile-based discretization as specified in the task instructions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles). The discretization is done per session, not globally across all sessions. Bin edges are computed using `np.percentile` with linearly spaced percentiles (0, 20, 40, 60, 80, 100), then `np.digitize` assigns each value to a bin (0-4).

ii.
```python
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])
    return bins, edges

# Called per session:
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. The per-session discretization means each session has balanced bin counts independently, but bin edges vary across sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned with neural data in two ways: (1) dropped video frames are interpolated to match the neural frame count, (2) both signals are binned with the same bin size (10 frames), ensuring one-to-one correspondence of time bins.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)  # n_frames from neural
me_binned = bin_data(me, BIN_SIZE)
# Then same trial splitting indices are used:
trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The camera was triggered by the microscope acquisition at the same 30 Hz rate, providing synchronization. Interpolation handles the occasional dropped video frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles two types of data issues: (1) dropped video frames are detected via interframe intervals and interpolated using linear interpolation between neighboring values, and (2) remainder frames/bins at the end of sessions that don't fill complete trials are discarded.

ii.
```python
def interpolate_missing_frames(me, ifi, n_neural_frames):
    median_ifi = np.median(ifi)
    threshold = median_ifi * 1.5
    gap_indices = np.where(ifi > threshold)[0]
    missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
    # ... builds result with interpolated values ...
    return result[:n_neural_frames]
```

iii. The interpolation uses a data-driven threshold (1.5x median interframe interval) to detect gaps, and estimates the number of missing frames per gap from the interval duration.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` baseline correction, which runs on CPU in the AI's implementation (unlike the reference which uses GPU). Loading the `.npy` files is also significant due to file I/O.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
```

iii. The baseline correction involves sliding window operations over the full session length for every neuron. Running on CPU rather than GPU makes this slower.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame interpolation function uses a Python for loop to iterate over each source frame individually, building the result array element by element. This could potentially be vectorized.

ii.
```python
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    dst_idx += 1
    if gap_pos < len(gap_indices) and src_idx == gap_indices[gap_pos]:
        n_insert = missing_per_gap[gap_pos]
        for k in range(n_insert):
            alpha = (k + 1) / (n_insert + 1)
            result[dst_idx] = me[src_idx] * (1 - alpha) + me[src_idx + 1] * alpha
            dst_idx += 1
```

iii. The number of source frames can be large (36000-54000), so iterating one by one in Python is inefficient, though it only runs once per session.

## 6-c. What processing does the code repeat multiple times?

i. The motion energy discretization is done per session rather than globally, meaning `np.percentile` is called separately for each session. This is not exactly repeated processing but a different strategy than computing global percentiles once.

ii.
```python
# Called inside the session loop:
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. Each session gets its own percentile computation, which is redundant if a global discretization is desired.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does not perform any obviously unnecessary processing that is discarded. All computed data (neural, input, output) is stored in the final output. The only discarded data is the remainder frames/bins that don't fill complete trials, which is inherent to the trial-splitting approach.

ii. N/A

iii. The code is relatively streamlined without extraneous processing steps.
