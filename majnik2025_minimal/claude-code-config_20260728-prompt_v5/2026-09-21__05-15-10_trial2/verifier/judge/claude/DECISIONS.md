# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 6 mouse names (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it lists subdirectories sorted alphabetically as sessions. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
DATA_DIR = '/app/data'
# ...
for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = sorted([
        d for d in os.listdir(mouse_dir)
        if os.path.isdir(os.path.join(mouse_dir, d))
    ])
    for sess in sessions:
        session_dir = os.path.join(mouse_dir, sess)
        # ...
        F = np.load(os.path.join(s2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
        me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
        tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The AI explored the data directory structure and identified the 6 mice and their session subdirectories. It hardcoded the mouse names after discovery. It chose `tstamps.npy` for aligning motion energy to neural data instead of `interframe_int.npy`.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hardcoded list `MICE`. Each directory name in the list corresponds to one mouse.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
for mouse_i, mouse in enumerate(MICE):
    # ...
    subject_idx.append(mouse_i)
```

iii. The AI identified the 6 mice from the data directory and listed them explicitly. This achieves the same result as the reference's dynamic discovery approach, but is less flexible.

## 1-c. How are the data split into sessions?

i. Each subdirectory within a mouse's folder is treated as one session, sorted alphabetically. All sessions for all mice are processed.

ii.
```python
sessions = sorted([
    d for d in os.listdir(mouse_dir)
    if os.path.isdir(os.path.join(mouse_dir, d))
])
for sess in sessions:
    session_dir = os.path.join(mouse_dir, sess)
```

iii. The AI identified from exploration that each subdirectory contains one day's recording with suite2p output and motion energy files. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as 60-second non-overlapping segments of the continuous recording. After binning (10 frames per bin at 30 Hz), each trial is 180 bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DUR = 60  # trial duration in seconds
TRIAL_BINS = int(TRIAL_DUR / BIN_DUR)  # 180 bins per 60s trial
# ...
def split_trials(neural, me_binned, trial_bins=TRIAL_BINS):
    n_bins = neural.shape[1]
    n_trials = n_bins // trial_bins
    for t in range(n_trials):
        start = t * trial_bins
        end = start + trial_bins
        neural_trials.append(neural[:, start:end])
        me_trials.append(me_binned[start:end])
    return neural_trials, me_trials
```

iii. The instructions specify "Split sessions into 60-second trials." Since the recording is continuous with no stimulus events, fixed-length segmentation is the appropriate approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second trials are kept; only the incomplete remainder at the end of a session is discarded.

ii. N/A - no filtering code exists.

iii. The AI did not implement trial filtering. There is no mention in the trajectory of considering trial quality controls beyond discarding remainders.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The AI identified these as the standard suite2p output files needed for dF/F computation, consistent with the paper's description.

## 2-b. How is the `neural` data processed?

i. The AI reimplements Suite2p's dF/F pipeline manually: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) maximin baseline estimation using Gaussian smoothing (sigma=10), minimum filter (window=1800 frames), then maximum filter (window=1800 frames), (3) baseline subtraction (`dff = Fc - Flow`). This is followed by temporal binning (averaging every 10 frames).

ii.
```python
def compute_dff(F, Fneu, fs=FS):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    dff = Fc - Flow
    return dff.astype(np.float32)
```

iii. From the trajectory (step 26, 29, 31): The AI researched Suite2p's default parameters and confirmed neucoeff=0.7, the maximin baseline method with sig_baseline=10 and win_baseline=60. The AI chose to reimplement rather than call `dcnv.preprocess` directly, reasoning it would follow "Suite2p default parameters" as stated in the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons from `F.npy` are included.

ii. N/A - no filtering code.

iii. The AI noted (step 23) that `iscell` values are all 1s for the tracked dataset, so no additional filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity, and trials are artificial segments starting from the beginning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing from 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before discretization.

ii.
```python
BIN_SIZE = 10  # frames per bin
BIN_DUR = BIN_SIZE / FS  # 0.3333 seconds
# ...
def bin_data(data, bin_size=BIN_SIZE):
    if data.ndim == 1:
        n = len(data) // bin_size * bin_size
        return data[:n].reshape(-1, bin_size).mean(axis=1)
    else:
        n = data.shape[1] // bin_size * bin_size
        return data[:, :n].reshape(data.shape[0], -1, bin_size).mean(axis=2)
```

iii. The AI cited the paper's methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and the bin duration, giving seconds from the start of that session.

ii.
```python
start_bin = t * TRIAL_BINS
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

iii. Since the frame rate is constant at 30 Hz, computing time from bin indices is equivalent to using actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `(bin_index + 0.5) * bin_duration`, where bin_duration = 10/30 seconds. The +0.5 offset places the time at the center of each bin rather than the left edge.

ii.
```python
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
```

iii. The AI chose to use bin centers. This is a reasonable choice for representing when a binned measurement occurs, though the reference uses bin left edges.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices used to index neural data, so alignment is inherent. Each time point corresponds one-to-one with a neural time bin.

ii. The bin index `start_bin + i` for the i-th time point in a trial directly corresponds to the same bin used for `neural[:, start:end]`.

iii. No additional alignment step is needed since both are derived from the same bin grid.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. The AI also loads `tstamps.npy` for frame alignment.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The AI identified `motion_energy_glob.npy` as the pre-computed global motion energy signal. It chose `tstamps.npy` (instead of `interframe_int.npy`) for handling dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) interpolation to align ME to neural frame count when camera frames were dropped (using `np.interp` with normalized positions), (2) temporal binning by averaging every 10 frames, (3) discretization into 5 equal-percentile bins per session.

ii.
```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned

# Binning:
me_binned = bin_data(me_aligned, BIN_SIZE)

# Discretization (per session):
def discretize_me(me_trials, n_bins=N_BINS_OUTPUT):
    all_me = np.concatenate(me_trials)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(all_me, percentiles)
    for me in me_trials:
        binned = np.digitize(me, boundaries)
        discretized.append(binned.astype(np.int64))
    return discretized
```

iii. The AI reasoned (step 29) that tstamps represent camera frame timing and could be used to interpolate ME to neural frame positions. Percentile-based discretization per session follows the instruction "five equal-percentile bins, selected per session."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins computed per session. The inner percentile boundaries (20th, 40th, 60th, 80th) are computed from the pooled trial data of each session. `np.digitize` maps values to bins 0-4.

ii.
```python
def discretize_me(me_trials, n_bins=N_BINS_OUTPUT):
    all_me = np.concatenate(me_trials)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    boundaries = np.percentile(all_me, percentiles)
    for me in me_trials:
        binned = np.digitize(me, boundaries)  # 0 to n_bins-1
        discretized.append(binned.astype(np.int64))
    return discretized
```

iii. The instructions specify "five equal-percentile bins, selected per session." The AI computes percentile boundaries from all trial data within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When the camera drops frames, the ME array is shorter than the neural data. The AI uses `np.interp` to interpolate ME values from camera frame positions to neural frame positions using evenly spaced normalized positions. After alignment, both streams are binned identically.

ii.
```python
def align_motion_energy(me, n_neural_frames, tstamps):
    if len(me) == n_neural_frames:
        return me
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned
```

iii. The AI investigated (step 29) the tstamps values and determined they could map camera frames to neural frames. However, the actual implementation uses `np.linspace(0, 1, ...)` for both camera and neural positions rather than using the tstamps values directly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are handled by interpolating ME to the neural frame count (see 4-d). Remainder bins at the end of sessions that don't fill a complete 60-second trial are discarded. No other error handling is implemented.

ii.
```python
# ME alignment for dropped frames
me_aligned = align_motion_energy(me, n_neural, tstamps)

# Remainder discarding (implicit in split_trials)
n_trials = n_bins // trial_bins  # integer division discards remainder
```

iii. The AI identified the ME/neural length mismatch issue from the data exploration phase and addressed it with interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F baseline computation (`compute_dff`), which applies Gaussian smoothing, minimum filter, and maximum filter over the full session length for every neuron. Loading `.npy` files is also I/O bound.

ii.
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The sliding window operations over ~36000-54000 frames for hundreds of neurons are computationally expensive. Unlike the reference which uses suite2p's GPU-accelerated `dcnv.preprocess`, the AI's manual implementation runs on CPU only.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially, creating slices. This could be done with array reshaping instead. The discretization loop over trials could also be vectorized.

ii.
```python
for t in range(n_trials):
    start = t * trial_bins
    end = start + trial_bins
    neural_trials.append(neural[:, start:end])
    me_trials.append(me_binned[start:end])
```

iii. However, the loop overhead is negligible since the number of trials per session is small (20-30).

## 6-c. What processing does the code repeat multiple times?

i. No processing is obviously repeated multiple times.

ii. N/A

iii. The code processes each session once in a single pass.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `tstamps.npy` file is loaded but not actually used in the alignment function. The `align_motion_energy` function accepts `tstamps` as a parameter but uses `np.linspace` instead.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
# ...
def align_motion_energy(me, n_neural_frames, tstamps):
    # tstamps is passed but never used!
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned
```

iii. The AI loaded tstamps intending to use it for alignment but ended up using evenly-spaced positions instead. The tstamps data is loaded but wasted.
