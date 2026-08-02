# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subject directories (folders starting with "jm") in the data directory, then iterates over sorted session subdirectories within each subject. For each session, it loads `F.npy`, `Fneu.npy` from `suite2p/plane0/` and `motion_energy_glob.npy` from `move_deve/`. There is no single bulk-load step; data is loaded session-by-session in a nested loop over subjects and sessions.

ii.
```python
def get_subjects_and_sessions():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions

# In process_session():
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified from the data README and reference code that the data is organized as subject folders containing session subfolders, each with `suite2p/plane0/` and `move_deve/` subdirectories. This follows the standard Track2p output format described in the data documentation.

## 1-b. How are the data split into subjects?

i. Subjects are identified by directory names starting with "jm" in the data directory. A mapping from internal IDs (jm031, jm032, etc.) to paper names (Mouse_A through Mouse_F) is maintained. The `subject_idx` array maps each session to its subject via an index into the `subjects` list.

ii.
```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}

# In convert_data():
subject_names = list(SUBJECT_MAP.values())
for subj in subjects_to_process:
    subj_idx = subject_names.index(SUBJECT_MAP[subj])
    # ...
    subject_idx_list.append(subj_idx)
```

iii. The AI noted from the data README that "subjects are named in alphabetically increasing order (e.g. jm031 - mouse A, jm032 - mouse B ... jm046 - mouse F)" and created the explicit mapping. All 6 subjects are included.

## 1-c. How are the data split into sessions?

i. Sessions are identified as subdirectories within each subject folder. Each session corresponds to one recording day. Sessions are sorted alphabetically (which is chronological since folder names are date-based, e.g., "2023-10-18_a"). Each session becomes one entry in the `neural`, `input`, and `output` lists.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
# ...
for i, session in enumerate(sessions):
    neural_trials, input_trials, output_trials, n_neurons = process_session(subj, session, ...)
    neural_all.append(neural_trials)
```

iii. The AI documented finding 41 total sessions (7,7,7,7,6,7 per subject), matching the paper's description of "at least 6 consecutive days" per mouse.

## 1-d. How are the data split into trials?

i. The original experiment is continuous spontaneous recording with no discrete trials. The AI splits each session's binned data into consecutive 2-minute blocks (360 time bins each, since 120s * 30Hz / 10 frames/bin = 360). This yields 10 trials per 20-min session and 15 trials per 30-min session. Leftover bins at the end that don't fill a complete 2-minute block are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120   # 2 minutes
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial

# In process_session():
n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The AI justified 2-minute trials by reference to the paper's cross-validation structure: "consecutive 2 minute blocks of the recording" (used for 5-fold nested CV in the paper's decoding analysis). The CONVERSION_NOTES state: "No discrete trials in the original experiment (continuous spontaneous recording) - Split continuous recording into 2-minute blocks (matching paper's CV structure)."

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 2-minute blocks from each session are included. Only incomplete trailing blocks (fewer than 360 bins) are discarded.

ii.
```python
n_trials = n_total_bins // TRIAL_BINS  # integer division discards remainder
```

iii. The AI noted in CONVERSION_NOTES: "No explicit trial curation (continuous spontaneous recording). Missing camera frames should be interpolated." Since this is continuous spontaneous activity (no task events), there are no behavioral criteria to filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from two Suite2p output files per session: `F.npy` (raw fluorescence traces, shape n_neurons x n_frames) and `Fneu.npy` (neuropil fluorescence traces, same shape). These are loaded from `suite2p/plane0/` within each session directory.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The AI identified from reference code exploration that the data is stored in Suite2p format and that F.npy and Fneu.npy are needed for dF/F computation following the paper's methods.

## 2-b. How is the `neural` data processed?

i. Processing follows three steps: (1) Neuropil correction: Fc = F - 0.7 * Fneu, (2) Baseline correction using Suite2p's `preprocess` function with maximin method (win_baseline=60s, sig_baseline=10, prctile_baseline=8, fs=30Hz), which returns baseline-subtracted fluorescence (Fc - F0), (3) Temporal binning by averaging every 10 consecutive frames (~333ms bins).

ii.
```python
def compute_dfof(F, Fneu, fs=FRAME_RATE):
    Fc = F - NEUROPIL_COEFF * Fneu  # Neuropil correction
    dfof = s2p_preprocess(
        Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
        SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
    )
    return dfof.astype(np.float32)

def bin_timeseries(data, bin_size):
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The AI justified this by quoting the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The Suite2p parameters were confirmed from ops.npy. During Critical Review 2, the AI discovered and fixed an issue where it was initially dividing by F0 (creating extreme outliers from near-zero baselines), switching to using Suite2p's baseline-subtracted output directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI determined that the provided data has already been filtered through Track2p's cell matching pipeline, which includes iscell > 0.5 threshold filtering. All iscell values in the provided data are 1.0, confirming pre-filtering. All neurons present in the data files are used.

ii.
```python
# No filtering code - all neurons from F.npy are used directly
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
# ... F is used in its entirety
```

iii. The AI documented: "Suite2p iscell probability > 0.5 (already applied in provided data)" and "Track2p matching: only neurons tracked across ALL days (already applied)." Verified that all iscell values are 1.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the continuous recording session. Since the experiment is continuous spontaneous recording with no discrete events, trials are simply consecutive 2-minute blocks from the beginning of the recording. The first trial starts at time 0 (start of recording), and each subsequent trial follows immediately.

ii.
```python
# Trials are consecutive slices of binned data starting from the beginning
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The AI set `temporal_alignment_event` to "start of continuous recording session" and `off_start` to 0.0 in the metadata, noting there are no discrete stimulus or behavioral events to align to.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~333.33 ms (10 frames at 30 Hz). Yes, temporal rebinning is applied: the raw 30 Hz data (33.33 ms per frame) is binned by averaging groups of 10 consecutive frames, reducing the temporal resolution by a factor of 10.

ii.
```python
BIN_SIZE = 10              # frames per bin
FRAME_RATE = 30.0          # Hz
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # ~333.33 ms

dfof_binned = bin_timeseries(dfof, BIN_SIZE)  # (n_neurons, n_bins)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The AI cited the paper's decoding methods: "averaging in bins of 10 consecutive timestamps" and confirmed the 30 Hz frame rate, yielding ~333 ms time bins.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input "time elapsed from start of experiment" is not derived from any raw data variable. It is computed from the bin indices using the known frame rate and bin size. Each bin's time is calculated as bin_index * (BIN_SIZE / FRAME_RATE) seconds.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The AI's CONVERSION_NOTES state the input mapping as: "time_elapsed = bin_index * (10/30) seconds" - a synthetic variable computed from the bin indices and known acquisition parameters.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the time values are computed as `bin_index * (BIN_SIZE / FRAME_RATE)` where `bin_index` ranges from the global bin start to bin end of that trial. This means time is continuous across trials within a session (e.g., trial 2 starts at 120s, trial 3 at 240s). The time values reflect absolute time from the start of the recording session.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The AI noted the input range spans [0.0, 1199.7] for 20-min sessions and [0.0, 1799.7] for 30-min sessions, which is consistent with the recording durations.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time is perfectly aligned with the neural data because both use the same bin indexing. The time for each bin is computed from the same global bin index used to slice the neural data, ensuring 1:1 temporal correspondence.

ii.
```python
# Same start/end indices used for both neural and input
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The AI verified: "Time inputs are continuous across trial boundaries" and "time input is monotonically increasing within each trial."

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output motion energy is derived from `motion_energy_glob.npy` in each session's `move_deve/` directory. This file contains pre-computed motion energy values (pixel-wise squared difference of consecutive video frames, summed across pixels) at frame-level resolution.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified from the data README and paper that motion energy is pre-computed and stored in the `move_deve` subdirectory, representing "pixel-wise squared difference of consecutive video frames, summed across pixels."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves four steps: (1) Interpolation of motion energy to match neural frame count if there are missing camera frames, (2) Temporal binning by averaging every 10 consecutive frames (same as neural data), (3) Discretization into 5 equal-percentile (quintile) bins per session using `np.percentile` and `np.digitize`, producing integer labels 0-4.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)

def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The AI justified quintile discretization per the instructions ("normalized and discretized into five equal-percentile bins") and per-session computation to handle different motion energy scales across sessions and developmental ages.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories (labels 0-4) using percentile-based thresholds computed per session. The 20th, 40th, 60th, and 80th percentiles of the session's binned motion energy values serve as thresholds. `np.digitize` assigns each value to the appropriate bin, producing exactly 20% of values in each bin (perfect quintiles).

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
thresholds = np.percentile(me_binned, percentiles)
labels = np.digitize(me_binned, thresholds).astype(np.int64)
```

iii. The AI verified: "Output distribution: 20% per bin (perfect quintiles)" across all sessions in both verification outputs.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned with neural data through shared frame-level timing. If the camera has fewer frames than the neural recording, linear interpolation is used to match frame counts. Both signals are then binned with the same 10-frame bins and split into trials using the same bin indices.

ii.
```python
# Interpolation to match neural frames
me = interpolate_motion_energy(me_raw, n_frames)
# Same binning
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
# Same trial splitting indices
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI relied on the paper's description that "microscope trigger initiates camera frame acquisition" providing frame-by-frame synchronization at 30Hz, and the data README's note about interpolating missing camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data quality issue is missing camera frames in some sessions (motion energy has fewer frames than neural data). The AI handles this by linear interpolation of the motion energy to match the neural frame count. No NaN/Inf values were found in any neural data. No other missing data handling is implemented.

ii.
```python
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)
    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp
```

iii. The AI documented missing frames per the data README: "indices of missing frames can be obtained by looking at tstamps.npy or interframe_int.npy and treated as missing values or interpolated over." The AI chose interpolation, noting specific sessions with discrepancies (e.g., jm031 sessions 3,4,5 with 35998/35997/35884 ME frames vs 36000 neural).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation via Suite2p's `preprocess` function, which performs baseline correction with a maximin filter involving sliding window operations over the full time series. The full conversion takes ~25.5s for 41 sessions, with per-session times of ~0.1-1.0s depending on neuron count and recording length.

ii.
```python
# This is the bottleneck - Suite2p baseline correction
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
```

iii. The AI measured timing: sessions with more neurons (685-746) take ~0.9-1.0s while smaller ones (221 neurons) take ~0.1-0.3s, indicating the dF/F computation scales with neuron count.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially, creating individual arrays with append operations. This could potentially be replaced with array reshaping to split all trials at once. However, since the loop is simple slicing, the performance impact is minimal.

ii.
```python
# Current loop-based trial splitting
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    # ... similar for input and output
```

iii. The AI's CONVERSION_NOTES Step 6 status was "NOT STARTED" (content not filled in), but the code itself is already fairly well vectorized. The binning function uses reshape+mean rather than looping, and the Suite2p preprocessing handles neurons in batch.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each session is processed exactly once. The dF/F computation, binning, and discretization are each called once per session. The `get_subjects_and_sessions()` function is called once at the start.

ii.
```python
# Single pass over all sessions:
for subj in subjects_to_process:
    for i, session in enumerate(sessions):
        neural_trials, input_trials, output_trials, n_neurons = process_session(subj, session, ...)
```

iii. The AI designed a single-pass pipeline where each session is loaded, processed, and results appended in one iteration. No redundant re-loading or re-processing occurs.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code truncates any leftover frames that don't fill a complete 10-frame bin (via integer division in `bin_timeseries`), and also discards leftover bins that don't fill a complete 2-minute trial. For a 36000-frame session (3600 bins), this is clean (3600/360=10 trials exactly). For a 54000-frame session (5400 bins), this is also clean (5400/360=15). So in practice, no data is wasted for the standard session lengths. However, the interpolation step processes the full motion energy time series even though some trailing frames may be discarded.

ii.
```python
# Binning truncates to exact multiple of bin_size
n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]

# Trial splitting discards trailing bins
n_trials = n_total_bins // TRIAL_BINS
```

iii. The AI noted in CONVERSION_NOTES that standard sessions divide evenly into trials, so no significant data is wasted. The processing is efficient with no unnecessary intermediate computations.
