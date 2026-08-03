# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over a hardcoded list of 6 mouse names (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it lists session subdirectories (filtered to those starting with a digit), then loads `F.npy`, `Fneu.npy`, `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` from `move_deve/`.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions

def process_session(mouse, session, session_dir, ...):
    F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
    Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
    ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified the data directory structure from exploration in Steps 0-2 of the workflow. The hardcoded list of mice ensures all 6 subjects are included. Sessions are identified by directories starting with a digit within each mouse folder. The AI also loads `ops.npy` to extract frame rate metadata, which the reference solution does not do.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of mouse names. Each mouse directory contains multiple session subdirectories.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for mouse_idx, mouse in enumerate(mice_to_process):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = get_sessions(mouse_dir)
```

iii. The AI documented 6 mice in CONVERSION_NOTES.md Step 2, matching the paper's statement of "a full dataset of 6 mice."

## 1-c. How are the data split into sessions?

i. Each session is a dated subdirectory within a mouse's folder (e.g., `2023-10-18_a`). Sessions are sorted alphabetically, which gives chronological order. Each session contains one continuous recording.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. The AI noted 41 total sessions (6-7 per mouse) in its CONVERSION_NOTES.md, consistent with the data.

## 1-d. How are the data split into trials?

i. The AI splits continuous recordings into 2-minute (120-second) non-overlapping blocks. After binning by 10 frames, each trial has 360 timepoints. This choice is based on the paper's methods which mention "consecutive 2 minute blocks" for cross-validation splits.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2-minute blocks
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FRAME_RATE  # 3600
BINNED_PER_TRIAL = FRAMES_PER_TRIAL // BIN_SIZE     # 360

n_trials = n_binned // BINNED_PER_TRIAL
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
```

iii. The AI justified this in its trajectory: "Methods mention 2-minute blocks for CV splits. I'll use 2-minute blocks as trials." The CONVERSION_NOTES.md Step 5 states: "Trial segmentation: Use 2-minute blocks. Paper uses 'consecutive 2 minute blocks' for CV splits."

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete trials from the continuous recording are included. Remainder frames that don't fill a complete trial at the end of a session are discarded (implicitly, by the `n_trials = n_binned // BINNED_PER_TRIAL` truncation).

ii. N/A (no filtering code)

iii. The AI noted in CONVERSION_NOTES.md Step 3: "Trial curation rules: None explicitly mentioned - continuous recordings."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from the `suite2p/plane0/` directory of each session.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The AI identified these as standard suite2p output files and noted this in CONVERSION_NOTES.md Step 1.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) baseline correction using the maximin method (Gaussian smoothing, min filter, max filter, then subtraction), and (3) temporal binning by averaging every 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    dff = Fc - Flow
    return dff.astype(np.float32)

dff_binned = bin_data(dff, BIN_SIZE, axis=-1)  # BIN_SIZE = 10
```

iii. The AI documented the dF/F computation in CONVERSION_NOTES.md Step 1, noting the paper says "Suite2p defaults" and that the code uses `neucoeff=0.7, baseline=maximin`. The binning by 10 frames follows the paper's methods: "averaging in bins of 10 consecutive timestamps." The AI initially implemented division by baseline `(Fc - Flow) / Flow` but later fixed it to subtraction only `Fc - Flow`, as documented in Step 7 bug fix notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI noted that all `iscell` values in the provided data are already 1.0 (pre-filtered by the Track2p pipeline).

ii. N/A (no filtering code)

iii. From CONVERSION_NOTES.md Step 1: "All iscell values are 1.0 (already filtered)." Step 3: "Neuron curation rules: iscell probability > 0.5 (already applied in provided data)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the recording session. Since the recording is continuous and trials are consecutive non-overlapping blocks, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. The AI noted there is no stimulus-driven trial structure, so alignment is simply to the start of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames, resulting in a time bin size of 333.33 ms (from 30 Hz native rate). This follows the paper's methods section.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE  # 333.33 ms

dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

iii. From CONVERSION_NOTES.md Step 3: "Bin size for decoding: 10 frames" quoting "averaging in bins of 10 consecutive timestamps." Step 5: "Time bin size: 333.33 ms (10 frames / 30 Hz)."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin indices and the time bin size.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. Since the frame rate is constant at 30 Hz, time can be computed from indices. The AI computes time from binned indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * bin_duration_in_seconds`. Each bin index is multiplied by 333.33 ms / 1000, giving time in seconds from the start of the session.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
# TIME_BIN_MS = 333.33
```

iii. No additional processing beyond index-to-time conversion.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is inherently aligned because it is computed from the same bin indices used to slice the neural data. Each time bin corresponds exactly to one neural data time bin.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end]
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. Alignment is guaranteed by construction since both neural and time use the same indexing.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified this file during data exploration in Step 2.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI applies four processing steps: (1) alignment of motion energy to neural frame count (padding/interpolation for missing end frames), (2) temporal binning by averaging 10 frames, (3) per-session min-max normalization to [0, 1], and (4) discretization into 5 equal-percentile bins per session.

ii.
```python
me_aligned = align_motion_energy(me, n_frames)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)

# Per-session normalization and discretization:
me_all = np.concatenate(me_values)
me_norm = normalize_motion_energy(me_all)  # min-max to [0,1]
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1])
```

iii. The AI's CONVERSION_NOTES.md Step 5 states: "Motion energy normalization: Per-session min-max normalization before percentile binning." The binning by 10 frames follows the methods text.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses per-session equal-percentile binning with 5 bins. For each session, all motion energy values are min-max normalized to [0, 1], then percentile edges are computed from the session's data, and `np.digitize` assigns bin labels 0-4.

ii.
```python
me_norm = normalize_motion_energy(me_all)  # per-session [0,1]
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. CONVERSION_NOTES.md metadata: "Per-session min-max normalization before percentile binning."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by padding/interpolating missing end frames to match the neural frame count, then bins both streams by the same factor of 10 and segments into the same trial boundaries.

ii.
```python
def align_motion_energy(me, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    elif len(me) < n_neural_frames:
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        if np.any(np.isnan(me_aligned)):
            valid = ~np.isnan(me_aligned)
            indices = np.arange(n_neural_frames)
            me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
        return me_aligned
```

iii. The AI acknowledged motion energy sometimes has fewer frames than neural data due to missing camera frames and chose to pad and interpolate.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion energy frames are handled by padding the end with NaN and then using linear interpolation (`np.interp`) across valid values. Remainder frames at the end of sessions that don't fill a complete trial are discarded (implicit truncation via integer division).

ii.
```python
me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_aligned[:len(me)] = me.astype(np.float64)
if np.any(np.isnan(me_aligned)):
    valid = ~np.isnan(me_aligned)
    indices = np.arange(n_neural_frames)
    me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
```

iii. The AI noted the discrepancy in CONVERSION_NOTES.md Step 4: "Motion energy frames: Sometimes fewer than neural frames. README: missing camera frames. Resolution: Interpolate or truncate to common length."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F baseline correction, which involves Gaussian smoothing, min-filter, and max-filter over the full session length for all neurons. The full conversion takes ~80 seconds for 41 sessions.

ii. N/A

iii. From conversion_full_out.txt, per-session times range from ~0.6s (221 neurons, 36000 frames) to ~3.0s (746 neurons, 54000 frames), with total time of 79.7s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over trials to slice arrays, but this is already efficient (just array slicing). The per-session discretization loop could potentially be vectorized but the number of sessions is small.

ii. N/A

iii. The AI noted in CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: None significant - vectorized operations used throughout."

## 6-c. What processing does the code repeat multiple times?

i. The AI normalizes motion energy per session after already having binned it per session. The percentile computation and digitization happen in a separate pass after all sessions are processed, requiring iteration over session data twice.

ii.
```python
# First pass: process all sessions
for session in sessions:
    neural_trials, input_trials, me_values = process_session(...)
    session_data.append(...)

# Second pass: discretize per session
for mouse_idx, mouse, session, neural_trials, input_trials, me_values in session_data:
    me_all = np.concatenate(me_values)
    me_norm = normalize_motion_energy(me_all)
    ...
```

iii. This two-pass approach is a design choice to allow global percentile computation if desired, though ultimately per-session percentiles were used.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ops.npy` to extract frame rate but uses a hardcoded `FRAME_RATE = 30` as the default anyway (since `ops.get('fs', FRAME_RATE)` returns the same value). The min-max normalization before percentile binning is also redundant since percentile binning is invariant to monotonic transformations.

ii.
```python
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
```

iii. Loading ops.npy is a minor overhead. The min-max normalization before percentile binning doesn't change the bin assignments.
