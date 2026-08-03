# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over a hardcoded list of 6 mice (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it discovers session directories by listing subdirectories starting with '2' (date format). For each session, it loads neural data (`F.npy`, `Fneu.npy`) from `suite2p/plane0/` and behavioral data (`motion_energy_glob.npy`, `interframe_int.npy`) from `move_deve/`. All data is loaded in a single pass, iterating mouse-by-mouse, session-by-session.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions

# In convert_data():
for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    for sess_name in sessions:
        sess_dir = os.path.join(mouse_dir, sess_name)
        suite2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
        move_dir = os.path.join(sess_dir, 'move_deve')
        F = np.load(os.path.join(suite2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
        me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
        ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The agent explored the data directory structure and the `load_data.ipynb` notebook provided with the dataset. The notebook demonstrates this same pattern of iterating over subject directories and loading Suite2p outputs. The agent used a sub-agent to explore the data organization and confirmed the directory layout matches the data README.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by a hardcoded list of 6 mouse IDs. Each mouse corresponds to a top-level directory under `data/`. A `subject_idx` array maps each session to its mouse index. The subjects list and ordering matches the paper's 6 mice (jm031=Mouse A through jm046=Mouse F).

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
subjects = MICE[:]
subject_idx = []
# ... in the loop:
subject_idx.append(mouse_i)
```

iii. The agent identified the 6 mice from the data directory structure and cross-referenced with the paper's description of "6 mice" and the data README which maps subject IDs to alphabetical labels (jm031=Mouse A, etc.).

## 1-c. How are the data split into sessions?

i. Each subdirectory within a mouse folder that starts with '2' (date format YYYY-MM-DD_a) is treated as a session. Sessions are sorted chronologically. Each session becomes one entry in the `neural`, `input`, and `output` lists. This yields 41 total sessions (7 per mouse for 5 mice, 6 for jm040).

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. The agent followed the data organization described in the data README, where each session folder corresponds to one recording day. The sorting ensures chronological order (P7-P14).

## 1-d. How are the data split into trials?

i. Trials are non-overlapping 2-minute blocks of the binned recording. After temporal binning (10 frames averaged), each trial is 360 bins. 20-minute sessions yield 10 trials; 30-minute sessions yield 15 trials. Any remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2-minute trial blocks
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial

def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. The agent cited the paper's methods: "splits were done on consecutive 2 minute blocks of the recording." The 2-minute trial segmentation matches the paper's cross-validation blocking strategy for decoding analyses.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 2-minute blocks are retained. The only "filtering" is that partial blocks at the end of a recording (frames not filling a complete 360-bin trial) are discarded.

ii.
```python
# No trial filtering code exists. split_into_trials simply discards remainder:
n_trials = data.shape[1] // trial_length
```

iii. The agent did not implement trial filtering, and the CONVERSION_NOTES.md does not mention any trial quality criteria. The paper describes spontaneous behavior recordings without task-based trial structure that would require quality filtering. The agent verified that no NaN or Inf values exist in the final data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from two raw Suite2p output files: `F.npy` (raw fluorescence traces, shape n_neurons x n_timepoints) and `Fneu.npy` (neuropil fluorescence, same shape). These are loaded from `suite2p/plane0/` for each session.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
```

iii. The agent chose F.npy and Fneu.npy to compute dF/F, following the paper's description: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The agent noted that `spks.npy` (deconvolved spikes) was also available but chose F/Fneu for dF/F computation as described in the paper.

## 2-b. How is the `neural` data processed?

i. Processing involves three steps: (1) Neuropil correction: Fc = F - 0.7*Fneu; (2) Baseline estimation using Suite2p's "maximin" method: Gaussian smooth (sigma=window/6), then minimum filter, then maximum filter over a 60-second window (1800 frames); (3) dF/F = (Fc - F0) / F0. After dF/F computation, the data is temporally binned by averaging 10 consecutive frames (30Hz to 3Hz). Final data is stored as float32.

ii.
```python
def compute_dff(F, Fneu):
    Fc = F - NEUCOEFF * Fneu  # NEUCOEFF = 0.7
    win = int(WIN_BASELINE_SEC * FS)  # 60s * 30Hz = 1800 frames
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (Fc - F0) / F0_safe
    return dff

def bin_array(data, bin_size):
    n_bins = data.shape[1] // bin_size
    return data[:, :n_bins * bin_size].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)

dff = compute_dff(F, Fneu)
dff_binned = bin_array(dff, BIN_SIZE)
```

iii. The agent initially tried an 8th-percentile baseline filter but it was too slow (killed after 30+ minutes). The agent then switched to Suite2p's default "maximin" baseline method, reasoning that Suite2p's actual default method uses minimum_filter1d and maximum_filter1d which are much faster. The agent cited Suite2p source code for this method. The neuropil coefficient of 0.7 and 60-second window are Suite2p defaults as stated in the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons in the Suite2p output are used. The agent verified that all ROIs in `iscell.npy` are already marked as cells (probability >= 0.5), since Track2p's output only includes successfully tracked cells.

ii.
```python
# No iscell filtering. F.npy is loaded directly without any neuron subsetting:
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
# All neurons are used as-is
```

iii. The agent explored `iscell.npy` and found all values in column 0 are 1.0 (all ROIs marked as cells). The agent reasoned: "The data I'm working with is already filtered to valid cells, so I don't need additional thresholding." The CONVERSION_NOTES state: "No additional neuron filtering was applied since Track2p already curated the population."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. There is no specific event-based alignment since these are spontaneous behavior recordings without task events. Each trial's neural data is simply the corresponding 2-minute block of the continuous recording.

ii.
```python
# Trials are consecutive blocks from session start:
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)

# Metadata reflects this:
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. The agent noted that this dataset involves spontaneous behavior (no stimulus-evoked trials), so the alignment event is simply the start of the recording session. The 2-minute trial blocks are consecutive, non-overlapping segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~333.33 ms per bin. Temporal rebinning is applied: 10 consecutive frames at 30 Hz are averaged to produce each bin (30 Hz to 3 Hz). This yields 360 bins per 2-minute trial.

ii.
```python
BIN_SIZE = 10           # temporal binning factor (10 frames averaged)
FS = 30                 # imaging frame rate (Hz)
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial

dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The agent followed the paper's methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The metadata records `time_bin_size: 333.33` ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is NOT derived from any raw data variable. It is synthetically computed from the trial index, bin index, and known recording parameters (frame rate, bin size, trial duration). No raw timestamps or timing files are used.

ii.
```python
for t_i in range(n_trials):
    start_sec = t_i * TRIAL_DURATION_SEC
    time_bins = np.linspace(
        start_sec + BIN_DURATION_MS / 2000,
        start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
        TRIAL_BINS
    )
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The agent computed time elapsed from session start in seconds, using the center of each time bin. Since the imaging frame rate is constant (30 Hz) and bins are regular (10 frames each), the time can be computed deterministically without needing raw timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each temporal bin, accumulated from session start. For trial `t_i`, the start time is `t_i * 120` seconds. Within each trial, `np.linspace` generates 360 evenly-spaced time values from the center of the first bin to the center of the last bin. The first bin center of trial 0 is at ~0.167s; the last bin center of the last trial is at ~1199.83s (20-min sessions) or ~1799.83s (30-min sessions).

ii.
```python
start_sec = t_i * TRIAL_DURATION_SEC  # t_i * 120
time_bins = np.linspace(
    start_sec + BIN_DURATION_MS / 2000,  # center of first bin (~0.167s for trial 0)
    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
    TRIAL_BINS  # 360
)
input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The agent interpreted "Time elapsed from the beginning of the experiment" as time from the start of each recording session. Time values are continuous across trials within a session (not reset per trial).

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with neural data because it is synthetically constructed to match the exact bin structure. Each time value corresponds one-to-one with a neural data time bin. Both have the same number of timepoints (360 per trial).

ii.
```python
# Same TRIAL_BINS (360) used for both:
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
# Time input also has TRIAL_BINS points per trial
time_bins = np.linspace(..., TRIAL_BINS)
```

iii. Since time is computed rather than measured, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` (global motion energy from videography) in the `move_deve/` subdirectory of each session. Additionally, `interframe_int.npy` (inter-frame intervals) is used for alignment to handle dropped video frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The agent identified these files from the `move_deve/` directory structure and the data README, which describes "motion energy extracted from videography of spontaneous behaviour."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves: (1) Alignment to neural frames by detecting dropped video frames via `interframe_int.npy` (gaps > 1.5x median interval) and inserting NaN at gap locations; (2) Linear interpolation of NaN values; (3) Temporal binning by averaging 10 consecutive frames (matching neural data binning); (4) Discretization into 5 equal-percentile bins using global quintiles across all sessions.

ii.
```python
me_aligned = align_motion_energy(me_raw, ifi, n_frames)
me_binned = bin_array(me_aligned, BIN_SIZE)
# ... later, after collecting all ME values:
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)
binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The agent followed the paper's binning approach ("averaging in bins of 10 consecutive timestamps") and the task specification for discretization ("normalized and discretized into five equal-percentile bins"). The agent did not apply explicit normalization (e.g., z-scoring or min-max) before discretization, treating equal-percentile binning as the normalization step.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories (0-4, labeled Q1-Q5) using global quintile bin edges. Bin edges are computed from all binned motion energy values across all sessions and mice. `np.digitize` with the interior bin edges maps each value to a category (0 through 4). Each bin contains exactly 20% of the global data.

ii.
```python
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)  # [0, 20, 40, 60, 80, 100]
bin_edges = np.percentile(all_me_concat, percentiles)
# Discretize:
binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to N_OUTPUT_BINS-1
session_output.append(binned.reshape(1, -1).astype(np.int64))
```

iii. The agent chose global (across all mice and sessions) quintile binning to ensure balanced classes overall. The verification output confirms exact 20% per bin globally. However, per-session distributions are very uneven (some sessions of jm046 have no data in lower bins).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural frames before binning using the `align_motion_energy` function. This function detects dropped video frames from `interframe_int.npy` (intervals > 1.5x median = dropped frame), inserts NaN at those locations to align with neural frame indices, then linearly interpolates the NaN values. After alignment, both neural and motion energy data have the same number of frames and are binned identically.

ii.
```python
def align_motion_energy(me, interframe_int, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    median_ifi = np.median(interframe_int)
    aligned = np.full(n_neural_frames, np.nan)
    neural_idx = 0
    for i in range(len(me)):
        if neural_idx < n_neural_frames:
            aligned[neural_idx] = me[i]
        neural_idx += 1
        if i < len(interframe_int):
            n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
            neural_idx += n_dropped
    nans = np.isnan(aligned)
    if np.any(nans) and not np.all(nans):
        x = np.arange(n_neural_frames)
        aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
    return aligned
```

iii. The agent noted that 9/41 sessions had missing video frames (1-148 frames missing). The data README states: "In some recordings there might be some missing frames from the camera... can be interpolated over." The agent chose linear interpolation for missing frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames (detected via `interframe_int.npy`) are handled by inserting NaN values at the positions of dropped frames and then linearly interpolating. For neural data, a division-by-zero safeguard is applied during dF/F computation (baseline values < 1e-6 are clamped). The agent verified that no NaN or Inf values remain in the final output.

ii.
```python
# Missing video frames:
aligned = np.full(n_neural_frames, np.nan)
# ... insert known values, then interpolate:
aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])

# Division by zero in dF/F:
F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
dff = (Fc - F0) / F0_safe
```

iii. The CONVERSION_NOTES document: "Linear interpolation for missing frames: Simple and appropriate for small gaps (1-3 frames typical). Larger gaps (116-148 frames in 2 sessions) also interpolated." The agent also verified post-processing: "Trials with NaN: 0, with Inf: 0."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation, which applies Gaussian smoothing, minimum filtering, and maximum filtering over a 1800-frame window for every neuron. The agent's initial approach (8th-percentile filter) was so slow it was killed after 30+ minutes. The replacement maximin method is faster but still involves three sequential 1D filter operations over the full time axis for each neuron. Loading the .npy files (especially F.npy which can be hundreds of neurons x 54000 frames) is also significant I/O.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)  # win=1800
Flow = minimum_filter1d(Flow, size=win, axis=1)
F0 = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The agent explicitly documented that the percentile_filter was "very slow" and switched to the maximin method for performance.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `align_motion_energy` function uses a Python for-loop iterating over each motion energy frame to detect and handle dropped frames. This could potentially be vectorized using cumulative sum operations on the interframe intervals. The trial-splitting loop and per-trial time computation loop are also Python-level loops, though these are minor given the small number of trials.

ii.
```python
# Python loop in align_motion_energy:
for i in range(len(me)):
    if neural_idx < n_neural_frames:
        aligned[neural_idx] = me[i]
    neural_idx += 1
    if i < len(interframe_int):
        n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
        neural_idx += n_dropped

# Python loop for time input:
for t_i in range(n_trials):
    start_sec = t_i * TRIAL_DURATION_SEC
    time_bins = np.linspace(...)
```

iii. The agent did not discuss vectorization opportunities in the CONVERSION_NOTES.

## 6-c. What processing does the code repeat multiple times?

i. The code performs two passes over motion energy data: (1) collecting all binned ME values during the main session loop, and (2) re-iterating over stored ME trial data to apply discretization after computing global bin edges. This two-pass approach is necessary for global percentile computation but involves storing and re-iterating all ME trial data. The `get_sessions()` function is also called twice for each mouse - once during the main loop and once when building session_info metadata.

ii.
```python
# First pass: collect raw ME trials
output_all_raw.append(me_trials)
for mt in me_trials:
    all_me_values.append(mt)

# Second pass: discretize
for session_me_trials in output_all_raw:
    for me_trial in session_me_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])

# get_sessions called again in metadata:
'sessions': get_sessions(os.path.join(data_dir, mouse)),
```

iii. The agent did not discuss this duplication. The two-pass ME processing is a natural consequence of needing global statistics before discretizing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `PRCTILE_BASELINE = 8` parameter is defined but never used in the final code (leftover from the abandoned percentile_filter approach). The code also processes all sessions even in `sample_only` mode, discarding most of the data when saving only 2 sessions. The `bin_array` function discards remainder frames that don't fill complete bins, which is a small amount of data loss. The sample_data.pkl uses global bin edges computed from all data, not just the sample sessions.

ii.
```python
PRCTILE_BASELINE = 8    # baseline percentile (Suite2p default) -- UNUSED

# sample_only still processes everything:
if not sample_only:
    # save full
# Always saves sample regardless
```

iii. The CONVERSION_NOTES mention `baseline_percentile` in the metadata, but this parameter is not actually used in the maximin computation. The agent did not note this inconsistency.
