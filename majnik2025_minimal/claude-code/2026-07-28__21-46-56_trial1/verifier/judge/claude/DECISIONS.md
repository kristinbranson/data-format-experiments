# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes the list of 6 mice (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it discovers session directories by listing subdirectories starting with '2' (date format). For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve/`. All data is loaded in a single pass iterating over mice and sessions.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions

# Loading per session:
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI hardcoded the mouse list based on inspecting the data directory, rather than auto-discovering directories. Session discovery filters for directories starting with '2' to match the date-format naming convention. The CONVERSION_NOTES.md states the data comes from "6 mice (jm031-jm046) imaged daily in barrel cortex."

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded as `MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`. The outer loop iterates over this list, and a `mouse_i` index tracks which subject each session belongs to.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    subject_idx.append(mouse_i)
```

iii. Hardcoding is functionally equivalent to auto-discovery for this fixed dataset but less robust to changes.

## 1-c. How are the data split into sessions?

i. Sessions correspond to subdirectories within each mouse's folder that start with '2' (date format), sorted alphabetically (chronologically). Each subdirectory is one recording session.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. Filtering for directories starting with '2' matches the date naming convention (e.g., `2023-10-18_a`). This produces the same results as the reference's approach of taking all subdirectories.

## 1-d. How are the data split into trials?

i. Trials are defined as consecutive 2-minute (120-second) non-overlapping blocks. After temporal binning (10 frames averaged), each trial contains 360 bins. Remainder data that doesn't fill a complete trial is discarded.

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

iii. The AI justified 2-minute blocks by referencing the paper's decoding section: "splits were done on consecutive 2 minute blocks of the recording." The CONVERSION_NOTES.md explicitly states "Following the paper: 'splits were done on consecutive 2 minute blocks of the recording'".

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials (those that fill a full 2-minute block) are included.

ii. N/A (no filtering code)

iii. The AI did not mention any trial filtering criteria. Incomplete remainder data at session end is discarded but this is segmentation, not quality filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence), loaded from the `suite2p/plane0/` subdirectory of each session.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
```

iii. These are standard suite2p outputs. The CONVERSION_NOTES.md states: "Source: Suite2p outputs from Track2p's matched-cell pipeline."

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) baseline estimation using a manual reimplementation of Suite2p's "maximin" method (Gaussian smoothing with sigma=window/6, then min filter, then max filter over a 60-second window), (3) dF/F computation (`(Fc - F0) / F0`), and (4) temporal binning by averaging 10 consecutive frames (30 Hz to 3 Hz).

ii.
```python
def compute_dff(F, Fneu):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE_SEC * FS)
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (Fc - F0) / F0_safe
    return dff

# Then binning:
dff_binned = bin_array(dff, BIN_SIZE)
```

iii. The CONVERSION_NOTES.md justifies this by quoting the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The AI interpreted "baseline corrected fluorescence traces as our dF/F" as computing (Fc - F0) / F0, and separately reimplemented the baseline method rather than using suite2p's `dcnv.preprocess`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. All neurons in the F.npy output are included.

ii. N/A (no filtering code)

iii. The CONVERSION_NOTES.md states: "No additional cell filtering: Track2p data already contains only matched, verified cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the recording session. Since the recording is continuous with no stimulus events, trials are consecutive 2-minute segments starting from time 0.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning: 10 consecutive frames are averaged, reducing the rate from 30 Hz to 3 Hz (~333.33 ms per bin). This is applied to both neural and behavioral data.

ii.
```python
BIN_SIZE = 10           # temporal binning factor (10 frames averaged)
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms

def bin_array(data, bin_size):
    if data.ndim == 1:
        n_bins = len(data) // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_bins = data.shape[1] // bin_size
        return data[:, :n_bins * bin_size].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)

dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The CONVERSION_NOTES.md quotes the paper: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the trial index and bin duration parameters.

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

iii. Time is synthesized from known parameters (trial index, trial duration, bin size). No raw timestamp data is used.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time values are computed as bin centers: for each trial, `np.linspace` generates evenly-spaced times from the center of the first bin to the center of the last bin within that trial. The time is in seconds from the start of the session.

ii.
```python
start_sec = t_i * TRIAL_DURATION_SEC
time_bins = np.linspace(
    start_sec + BIN_DURATION_MS / 2000,  # center of first bin
    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,  # center of last bin
    TRIAL_BINS
)
```

iii. Using bin centers rather than bin edges is a reasonable choice that represents the midpoint of each temporal bin.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time values are computed to have one value per temporal bin, matching the neural data after binning. Each time value corresponds to the center of the same bin as the neural data.

ii. (Same code as 3-b; each `time_bins` array has shape `(1, TRIAL_BINS)` matching the binned neural data's time dimension.)

iii. Alignment is inherent since both are indexed by the same bin positions.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Interframe intervals from `interframe_int.npy` are used to detect and handle dropped video frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The CONVERSION_NOTES.md states: "Source: `move_deve/motion_energy_glob.npy` - global motion energy from videography."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) dropped video frames are detected using interframe intervals > 1.5x median interval, (2) NaN values are inserted at dropped frame positions and then linearly interpolated, (3) motion energy is temporally binned (averaged over 10 frames), (4) global percentile bin edges are computed across all sessions, (5) motion energy is discretized into 5 bins using `np.digitize`.

ii.
```python
def align_motion_energy(me, interframe_int, n_neural_frames):
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

me_binned = bin_array(me_aligned, BIN_SIZE)

# Global percentile discretization:
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)
binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The CONVERSION_NOTES.md justifies the 1.5x median threshold for dropped frame detection and notes that "Missing frames interpolated linearly. Affects 9/41 sessions."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins using global quintile boundaries computed across all sessions. `np.digitize` maps continuous values to bin indices 0-4.

ii.
```python
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)
binned = np.digitize(me_trial, bin_edges[1:-1])  # 0 to N_OUTPUT_BINS-1
```

iii. The task instructions specify "normalized and discretized into five equal-percentile bins." The AI computes global percentile edges so each bin contains ~20% of all data.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video camera is triggered by the microscope at 30 Hz, so frames are nominally synchronized. Dropped video frames are detected using interframe intervals exceeding 1.5x the median interval. NaN values are inserted at dropped positions and linearly interpolated. After alignment, both neural and motion energy are binned by 10 frames, maintaining frame-for-frame correspondence.

ii.
```python
me_aligned = align_motion_energy(me_raw, ifi, n_frames)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The AI's approach uses `np.round(interframe_int[i] / median_ifi - 1)` to detect how many frames were dropped at each gap, inserts NaN, and uses `np.interp` for linear interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected and interpolated (see 4-b/4-d). If motion energy length already matches neural frames, no interpolation is needed. Remainder frames/bins that don't fill a complete trial are discarded. The AI also checks for NaN and Inf values in the final neural data.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)

# ...NaN insertion and interpolation...

# Sanity checks:
nan_count = sum(np.any(np.isnan(t)) for s in neural_all for t in s)
inf_count = sum(np.any(np.isinf(t)) for s in neural_all for t in s)
print(f"\nTrials with NaN: {nan_count}, with Inf: {inf_count}")
```

iii. The CONVERSION_NOTES.md documents: "Missing video frames: Detected via interframe_int.npy (gaps > 1.5x median interval). Missing frames interpolated linearly. Affects 9/41 sessions (1-148 frames missing)."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the manual dF/F computation involving Gaussian smoothing, minimum filtering, and maximum filtering across all neurons for the full session length. Loading large .npy files is also I/O intensive.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
F0 = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The AI noted in the trajectory that it had to rewrite the baseline computation because an initial percentile-based approach was too slow, switching to the faster min/max filter approach.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `align_motion_energy` function uses a Python for-loop to iterate over all motion energy frames, inserting NaN at dropped positions. This could be vectorized. The discretization also loops over sessions and trials individually.

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
```

iii. The loop iterates over every motion energy frame (up to 54000 per session), which is inefficient but functionally correct.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat major processing steps. However, it collects motion energy values into `all_me_values` during the first pass (per-trial), then concatenates them again for percentile computation, and then loops over sessions/trials again for discretization - three passes over the motion energy data.

ii.
```python
# First pass: collect per-trial ME
for mt in me_trials:
    all_me_values.append(mt)

# Second pass: compute global percentiles
all_me_concat = np.concatenate(all_me_values)
bin_edges = np.percentile(all_me_concat, percentiles)

# Third pass: discretize
for session_me_trials in output_all_raw:
    for me_trial in session_me_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The multi-pass approach is necessary because global percentile edges must be computed before discretization, but the data collection could be more streamlined.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dF/F (dividing by baseline F0) rather than just using the baseline-corrected fluorescence as the reference does. This additional division step changes the neural data representation. The temporal binning (averaging 10 frames) is also an extra processing step not done by the reference solution. The sample dataset creation is always performed even when not needed.

ii.
```python
# dF/F division (not done in reference):
dff = (Fc - F0) / F0_safe

# Temporal binning (not done in reference):
dff_binned = bin_array(dff, BIN_SIZE)

# Sample always created:
sample_data = { ... }
with open(sample_output_file, 'wb') as f:
    pickle.dump(sample_data, f)
```

iii. These processing steps follow the paper's methods section but differ from the reference solution's approach.
