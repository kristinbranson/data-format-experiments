# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all mouse directories (jm031-jm046) found in the data directory, then iterates over all session subdirectories within each mouse. For each session, it loads F.npy and Fneu.npy from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/` for behavioral data. All data are loaded using `np.load()`. No filtering of sessions or mice is applied — all available data are included.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir)
                      if os.path.isdir(os.path.join(mouse_dir, d))])
    for sess_name in sessions:
        sess_dir = os.path.join(mouse_dir, sess_name)
        s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
        move_dir = os.path.join(sess_dir, 'move_deve')
        F = np.load(os.path.join(s2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
        me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
        ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI explored the data directory structure, read the data README, and the load_data.ipynb notebook to understand the expected directory layout. The approach follows the patterns shown in the provided notebook (iterating over subject folders then session folders). The AI verified that all 6 mice with 41 total sessions were loaded.

## 1-b. How are the data split into subjects (mice)?

i. Each top-level directory starting with 'jm' in the data directory is treated as a separate subject. Subjects are sorted alphabetically (jm031, jm032, jm038, jm039, jm040, jm046). A subject index is maintained per session using `mouse_idx`.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
subjects = mice
# ...
for mouse_idx, mouse in enumerate(mice):
    # ...
    all_subject_idx.append(mouse_idx)
```

iii. The AI followed the data README which states "For each subject there is a folder corresponding to the subject id." The 6 mice match the paper's description of "6 mice imaged daily for a minimum of 6 consecutive days."

## 1-c. How are the data split into sessions?

i. Each subdirectory within a mouse folder is treated as a separate session. Sessions are sorted alphabetically (which corresponds to chronological order since folder names follow YYYY-MM-DD format). Each session becomes one entry in the output lists (neural, input, output).

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
for sess_name in sessions:
    # ... process session ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(mouse_idx)
```

iii. The AI identified that each session folder corresponds to a recording day. The data README confirms "Each subject folder contains a number of session folders, each corresponding to one recording day."

## 1-d. How are the data split into trials?

i. Each session's binned data is split into consecutive, non-overlapping 2-minute blocks (360 time bins each, since 120s * 30Hz / 10 frames/bin = 360 bins). The number of trials per session depends on the session length: 10 trials for 20-minute sessions (jm031, jm032) and 15 trials for 30-minute sessions (jm038-jm046). Any remaining data at the end that doesn't fill a complete 2-minute block is discarded.

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

iii. The AI referenced the paper's methods: "splits were done on consecutive 2 minute blocks of the recording." The paper uses 5-fold cross-validation on these blocks, and the AI adopted the same 2-minute block structure for trial segmentation.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 2-minute blocks are included. Incomplete blocks at the end of a session are simply not included (truncated). No trials are rejected for quality reasons.

ii.
```python
# No filtering code — all complete trials are kept
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    # ... append trial data without any quality checks ...
    session_neural.append(trial_neural.astype(np.float32))
```

iii. The AI noted in CONVERSION_NOTES.md that "All ROIs in the dataset are already filtered" and the Track2p pipeline provides only tracked neurons. No additional quality filtering of trials is mentioned in the paper's methods for the decoding analysis, so none was applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from two Suite2p output files per session: `F.npy` (raw fluorescence traces, shape n_neurons x n_frames) and `Fneu.npy` (neuropil fluorescence traces, same shape).

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The AI followed the paper's statement: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." Suite2p's dF/F computation requires both the raw fluorescence (F) and neuropil fluorescence (Fneu) as inputs.

## 2-b. How is the `neural` data processed?

i. The neural data undergoes three processing steps:
1. **Neuropil correction**: Fc = F - 0.7 * Fneu (Suite2p default neucoeff=0.7)
2. **Baseline correction**: Suite2p's `preprocess` function with 'maximin' method (Gaussian smoothing with sigma=10 frames, running minimum over 1800 frames, running maximum over 1800 frames, then subtraction from Fc)
3. **Temporal binning**: Averaged in bins of 10 consecutive frames

ii.
```python
def compute_dff_suite2p(F, Fneu, fs=30.0, neucoeff=0.7,
                        win_baseline=60.0, sig_baseline=10.0):
    Fc = (F - neucoeff * Fneu).astype(np.float32)
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                     device=torch.device('cpu'))
    return dff

# Later:
dff = compute_dff_suite2p(F, Fneu, fs=FS)
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The AI initially tried computing dF/F manually but found it produced worse decoder accuracy. After investigating Suite2p's source code, the AI switched to using Suite2p's actual `preprocess` function to ensure exact implementation match. The AI noted in the trajectory reasoning that `preprocess` returns F - baseline (not (F-baseline)/baseline), which is what the paper calls "baseline corrected fluorescence traces."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. The AI found that all ROIs in the dataset already have `iscell[:,0] == 1.0`, meaning all neurons passed Suite2p's cell classifier. Since the data comes from Track2p, it only contains neurons successfully tracked across all sessions.

ii.
```python
# No iscell filtering code is present in convert_data.py
# The AI verified:
# iscell[:5]: [[1.  0.98993323] [1.  0.88940529] ...]
# All iscell values are 1.0
```

iii. The AI verified that iscell values were all 1.0 and noted: "Track2p provides only successfully tracked neurons across all sessions." The paper states "We considered all ROIs above the default threshold of 0.5 as true cells." Since all provided ROIs are already above threshold, no additional filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the start of the imaging session. Each trial begins at a fixed offset from the session start (trial 0 starts at time 0, trial 1 at 120s, etc.). The `temporal_alignment_event` is set to "Start of imaging session" with `off_start` = 0.0.

ii.
```python
data['metadata'] = {
    'temporal_alignment_event': 'Start of imaging session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. Since this is a spontaneous behavior paradigm (no stimulus/trial structure), there is no natural alignment event other than the start of recording. The AI chose to align to session start, which is the only meaningful reference point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 333.33 ms (10 frames at 30 Hz). Temporal rebinning is applied: the original 30 Hz data (33.33 ms per frame) is averaged in bins of 10 consecutive frames, yielding an effective 3 Hz sampling rate.

ii.
```python
BIN_SIZE = 10  # frames per bin
FS = 30.0  # imaging rate in Hz
time_bin_size_ms = (BIN_SIZE / FS) * 1000  # ~333.33 ms

def bin_data(data, bin_size):
    if data.ndim == 1:
        n = len(data)
        n_bins = n // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
```

iii. The paper states: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI directly implemented this 10-frame binning as described.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input variable is not derived from any raw data file. It is computed analytically from the bin indices and the known imaging parameters (30 Hz frame rate, 10-frame bins).

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)  # (1, n_timepoints)
```

iii. The instructions specify the decoder input should be "Time elapsed from the beginning of the experiment." Since the imaging rate is constant at 30 Hz and bins are 10 frames, the time for each bin can be computed directly without needing any raw timing data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center time of each bin in seconds. For each trial, the bin indices (relative to session start) are converted to seconds: `time = (bin_index + 0.5) * (10 / 30)`. The +0.5 offset places the time at the center of each bin rather than the start.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)  # (1, n_timepoints)
```

iii. The AI chose to use bin centers (adding 0.5 to the bin index) which is a standard convention. The time ranges from 0.167s (first bin center) to 1199.83s or 1799.83s (last bin center) depending on session length.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because both are indexed by the same bin positions. Each time bin in the input corresponds to the same time bin in the neural data. The time values are computed from the same bin indices used to slice the neural data.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
    bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
    trial_input = bin_times.reshape(1, -1)
```

iii. Since the time input is synthetically computed from the bin indices (not from an external data source), alignment is trivially exact. The same `start:end` range is used for both neural and input data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (raw motion energy values) and `interframe_int.npy` (interframe intervals for detecting missing frames) in each session's `move_deve/` directory.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The data README describes the `move_deve` folder as containing "processed behavioural data (motion energy extracted from videography of spontaneous behaviour 'motion_energy_glob.npy')." The paper describes motion energy as "pixel-wise difference of consecutive frames... squared all individual pixel-wise values and summed across pixels."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy undergoes three processing steps:
1. **Interpolation**: Missing video frames are interpolated using interframe interval analysis to detect gap locations, then linear interpolation fills in missing values
2. **Temporal binning**: Averaged in bins of 10 consecutive frames (same as neural data)
3. **Discretization**: Binned into 5 equal-percentile bins per session using `np.percentile` and `np.digitize`

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
me_binned = bin_data(me, BIN_SIZE)
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. The AI designed the interpolation to handle cases where video frames were dropped (documented in the data README). The discretization into 5 equal-percentile bins follows the instructions: "Motion energy, normalized and discretized into five equal-percentile bins."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) using quantile-based thresholds computed per session. The percentile edges are [0, 20, 40, 60, 80, 100], and `np.digitize` assigns each value to a bin (0-4).

ii.
```python
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])  # 0 to n_bins-1
    return bins, edges
```

iii. The instructions specify "discretized into five equal-percentile bins." The AI computed percentile bin edges per session (using the full session's binned motion energy). Each bin contains exactly 20% of the data points, which was verified in the validation output.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned with neural data by first interpolating missing video frames to match the neural frame count, then applying identical temporal binning (10 frames per bin). Both signals are then sliced using the same trial boundaries.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)  # match to neural frame count
me_binned = bin_data(me, BIN_SIZE)  # same binning as neural
# Same trial slicing:
trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The paper states the camera was triggered by the microscope acquisition, "allowing for simple synchronisation across the two modalities." The AI exploited this one-to-one correspondence between video frames and neural frames, only needing to interpolate where video frames were dropped.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames (where motion energy length < neural frame count) are handled by interpolation. The AI uses interframe intervals to detect gap locations, then inserts linearly interpolated values. If motion energy is longer than neural data, it is truncated. After interpolation, if the count still doesn't match, the remaining positions are padded with the last known value.

ii.
```python
def interpolate_missing_frames(me, ifi, n_neural_frames):
    if len(me) == n_neural_frames:
        return me
    n_missing = n_neural_frames - len(me)
    if n_missing <= 0:
        return me[:n_neural_frames]
    median_ifi = np.median(ifi)
    threshold = median_ifi * 1.5
    gap_indices = np.where(ifi > threshold)[0]
    missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
    # ... interpolation logic ...
```

iii. The data README states: "In some recordings there might be some missing frames from the camera... they can be interpolated over." The AI verified that for the worst case (jm031 2023-10-22_a), 116 frames were missing, and the interframe interval analysis correctly identified all gap locations.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the Suite2p `preprocess` function call for baseline correction, which performs Gaussian smoothing, running minimum, and running maximum over the full fluorescence traces for all neurons. This involves multiple passes over potentially large arrays (e.g., 746 neurons x 54000 frames). The interpolation of missing frames is also somewhat costly but less significant.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
```

iii. The trajectory shows the full conversion takes noticeable time. The Suite2p preprocess function applies three sequential filtering operations on each neuron's trace (Gaussian smoothing, running min, running max), each requiring computation over the full time series.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interpolate_missing_frames` function uses a Python for-loop to iterate through each source frame and insert interpolated values at gap locations. This could be vectorized using numpy array operations (e.g., `np.interp`). The trial-splitting loop could also potentially be vectorized using array reshaping.

ii.
```python
# This loop iterates frame-by-frame:
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    dst_idx += 1
    if gap_pos < len(gap_indices) and src_idx == gap_indices[gap_pos]:
        n_insert = missing_per_gap[gap_pos]
        for k in range(n_insert):
            alpha = (k + 1) / (n_insert + 1)
            # ...
```

iii. The AI chose a straightforward iterative implementation for clarity. For the typical case (116 missing frames out of 36000), performance is adequate, so vectorization was not critical.

## 6-c. What processing does the code repeat multiple times?

i. The `bin_data` function is called separately for neural data and motion energy in each session, performing essentially the same reshape-and-mean operation. The Suite2p `preprocess` function is called once per session, but its internal operations (Gaussian filter, running min/max) are applied identically each time. The percentile computation for discretization is done per session.

ii.
```python
# Called separately for each data stream per session:
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. These repeated calls are necessary since each session's data has different dimensions and must be processed independently. This is not truly redundant computation — it is the same operation applied to different data.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes time values using bin centers (+0.5 offset) which adds marginal precision that likely doesn't affect decoder performance. The `interpolate_missing_frames` function computes detailed gap analysis even when the number of missing frames is small and simple approaches (like `np.interp`) would suffice. The code also stores float32 precision for neural data when lower precision might be adequate.

Additionally, the trailing frames that don't fill a complete 2-minute trial block are processed through dF/F computation and binning but then discarded:

ii.
```python
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS  # integer division discards remainder
# All bins beyond n_trials * TRIAL_BINS are computed but never used
```

iii. The AI processes the full session through dF/F and binning before splitting into trials, meaning any trailing data that doesn't fill a complete 2-minute block is processed but discarded. This is a minor inefficiency since the computation is dominated by the dF/F baseline correction.
