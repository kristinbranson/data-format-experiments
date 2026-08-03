# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over a hardcoded list of 6 mice (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it lists session subdirectories sorted alphabetically (chronologically). For each session, it loads `F.npy`, `Fneu.npy`, `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` from `move_deve/`. All data is loaded via `np.load`.

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

iii. The AI documented in CONVERSION_NOTES.md Step 2 the full data structure: 6 mice with 6-7 sessions each, each session containing Suite2p neural data and motion energy behavioral data. The AI noted all iscell values are 1.0 (pre-filtered by Track2p).

## 1-b. How are the data split into subjects (mice)?

i. The data is split by iterating over the 6 mouse directories. Each mouse's sessions are processed together, and a `subject_idx` array maps each session to the mouse index.

ii.
```python
for mouse_idx, mouse in enumerate(mice_to_process):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    all_subject_idx.append(mouse_idx)

subjects = mice_to_process
data['subject_idx'] = np.array(all_subject_idx, dtype=np.int64)
```

iii. The AI identified 6 mice from the data directory structure, matching the paper's description of "a full dataset of 6 mice."

## 1-c. How are the data split into sessions?

i. Each subdirectory within a mouse folder (named by date, e.g., `2023-10-18_a`) is treated as a separate session. Sessions are sorted chronologically. Each session's data becomes one entry in the `neural`, `input`, and `output` lists.

ii.
```python
for session in sessions:
    session_dir = os.path.join(mouse_dir, session)
    neural_trials, input_trials, me_values = process_session(
        mouse, session, session_dir, ...)
    session_data.append((mouse_idx, mouse, session, neural_trials, input_trials, me_values))
```

iii. The AI noted 41 total sessions (7 each for 5 mice, 6 for jm040), matching the data structure. This is consistent with the paper's "at least 6 consecutive days."

## 1-d. How are the data split into trials?

i. Continuous recordings are segmented into 2-minute blocks (trials). After binning neural data by 10 frames, each trial is 360 timepoints. The number of complete trials depends on session duration: 10 trials for 20-minute sessions (jm031, jm032) and 15 trials for 30-minute sessions (jm038-jm046).

ii.
```python
TRIAL_DURATION_SEC = 120  # 2-minute blocks
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FRAME_RATE  # 3600
BINNED_PER_TRIAL = FRAMES_PER_TRIAL // BIN_SIZE     # 360

n_trials = n_binned // BINNED_PER_TRIAL

for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    neural_trials.append(neural_trial)
```

iii. The AI justified this based on methods.txt: "splits were done on consecutive 2 minute blocks of the recording." This was originally used for cross-validation folds in the paper's ridge regression decoder.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All complete 2-minute blocks are included. Leftover frames at the end of a session that don't fill a complete trial are discarded.

ii.
```python
n_trials = n_binned // BINNED_PER_TRIAL  # integer division discards remainder
```

iii. The AI noted in CONVERSION_NOTES.md: "Trial curation rules: None explicitly mentioned - continuous recordings." The paper does not describe any trial filtering criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence traces) from each session's `suite2p/plane0/` directory. The `ops.npy` file provides parameters (frame rate, baseline correction settings).

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
```

iii. The AI documented this in CONVERSION_NOTES.md Step 1, noting these are standard Suite2p output files. The load_data.ipynb notebook also loads F.npy and suggests computing dF/F "the way as described in the paper."

## 2-b. How is the `neural` data processed?

i. The processing pipeline is:
1. Neuropil correction: `Fc = F - 0.7 * Fneu`
2. Baseline estimation using maximin method (gaussian filter -> min filter -> max filter)
3. Baseline subtraction: `dff = Fc - Flow` (subtraction only, no division)
4. Temporal binning: average in non-overlapping bins of 10 frames

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)  # 1800 frames
    Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    dff = Fc - Flow
    return dff.astype(np.float32)

dff_binned = bin_data(dff, BIN_SIZE, axis=-1)  # BIN_SIZE=10
```

iii. The AI initially implemented dF/F as `(Fc - Flow) / Flow` (traditional definition) but discovered this caused extreme values. After inspecting the Suite2p source code (`dcnv.py`), the AI found that Suite2p's `baseline_maximin` performs subtraction only, not division. The AI corrected to `Fc - Flow`, documented this bug fix in CONVERSION_NOTES.md Step 7. The AI chose neucoeff=0.7 based on ops.npy and the paper stating "default Suite2p parameters," despite the Track2p GUI code using neucoeff=0.0.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. The AI determined that all neurons in the provided data have `iscell[:,0] == 1.0`, meaning they were already pre-filtered by Track2p's cell detection and tracking pipeline.

ii. No filtering code exists in convert_data.py. The AI loads all neurons from F.npy without any exclusion.

iii. From CONVERSION_NOTES.md: "All iscell values are 1.0 (already filtered)" and "iscell probability > 0.5 (already applied in provided data)." The paper states cells above Suite2p's default threshold of 0.5 are considered true cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Trials are consecutive 2-minute blocks from the beginning of each session recording. There is no alignment to a specific experimental event since this is a spontaneous behavior recording (no stimulus presentation).

ii.
```python
data['metadata'] = {
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The AI noted this is a continuous recording of spontaneous activity, so the natural alignment is to the session start. The 2-minute trial blocks are taken consecutively from the beginning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 333.33 ms (10 frames at 30 Hz). Temporal rebinning is applied: the raw 30 Hz data is averaged in non-overlapping bins of 10 consecutive frames.

ii.
```python
BIN_SIZE = 10           # frames to average
FRAME_RATE = 30         # Hz
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE  # 333.33 ms

def bin_data(data, bin_size, axis=-1):
    n = data.shape[axis]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    data_trunc = data[..., :n_use]
    new_shape = data.shape[:-1] + (n_bins, bin_size)
    return data_trunc.reshape(new_shape).mean(axis=-1)
```

iii. From methods.txt: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data variable. It is computed synthetically from the bin indices, using the bin size and frame rate to convert bin index to elapsed time in seconds.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. The AI computed time elapsed from the start of the recording session based on the bin index. The time is continuous across trials within a session (trial 0 starts at 0s, trial 1 starts at 120s, etc.).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time is computed as `bin_index * time_bin_ms / 1000.0` where `time_bin_ms = 333.33 ms`. No further processing. The time represents seconds elapsed from the start of the recording session.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
# For trial 0: start=0, end=360 -> [0.0, 0.333, 0.667, ..., 119.67]
# For trial 1: start=360, end=720 -> [120.0, 120.333, ..., 239.67]
```

iii. The AI's CONVERSION_NOTES.md Step 5 maps "time index" to "input[0]" with transform "Time elapsed from start of session in seconds."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time is inherently aligned with neural data because it is derived from the same bin indices used for the neural data. Each time bin corresponds exactly to the same temporal window as the corresponding neural data bin.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end]  # same start:end
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0)  # same start:end
```

iii. Since the time input is computed from the same indexing scheme as the neural data, alignment is exact by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` in each session's `move_deve/` directory. This contains the global motion energy computed from videography.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. From CONVERSION_NOTES.md: "Motion energy: pixel-wise difference of consecutive video frames, squared, summed across pixels." This matches the paper's description: "We took each two consecutive frames, computed their pixelwise difference. We then squared all individual pixel-wise values and summed across pixels."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The processing pipeline is:
1. Align motion energy frames to neural frames (interpolation if fewer ME frames)
2. Bin by averaging 10 consecutive frames
3. Normalize per-session to [0, 1] using min-max normalization
4. Discretize into 5 equal-percentile bins per session

ii.
```python
me_aligned = align_motion_energy(me, n_frames)  # interpolate if needed
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)  # bin by 10

# Per-session normalization and discretization:
me_all = np.concatenate(me_values)
me_norm = normalize_motion_energy(me_all)  # min-max to [0,1]
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. The AI documented these steps in CONVERSION_NOTES.md Step 5. The binning matches methods.txt ("averaging in bins of 10 consecutive timestamps"). The normalization and discretization follow the task specification ("normalized and discretized into five equal-percentile bins").

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) per session. Percentile edges are computed from all binned motion energy values within a session, then `np.digitize` assigns each value to a bin (0-4).

ii.
```python
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)  # [0, 20, 40, 60, 80, 100]
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)  # 0..4
```

iii. The task specifies "five equal-percentile bins." The per-session approach ensures each session has roughly uniform bin distributions, confirmed by the output showing exactly 0.2 for each bin in every session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural frames before any processing. If the motion energy has fewer frames than neural data (due to missing camera frames), the missing values at the end are filled by linear interpolation. If longer, it is truncated. Both neural and motion energy data are then binned identically (10-frame bins), ensuring temporal alignment.

ii.
```python
def align_motion_energy(me, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    elif len(me) < n_neural_frames:
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        valid = ~np.isnan(me_aligned)
        indices = np.arange(n_neural_frames)
        me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
        return me_aligned
    else:
        return me[:n_neural_frames].astype(np.float64)
```

iii. The data README notes: "In some recordings there might be some missing frames from the camera" and suggests the frames "can be interpolated over." The AI's approach of placing available frames at the start and interpolating is a reasonable but imperfect solution - ideally, the specific missing frame indices from `tstamps.npy` or `interframe_int.npy` should be used to identify which frames are missing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data quality issue is missing motion energy frames (camera frames can be fewer than neural frames). The AI handles this by linear interpolation to match the neural frame count. No other missing data handling is implemented - the code assumes neural data is complete and well-formed. The `normalize_motion_energy` function has a guard for zero-range data (returns zeros).

ii.
```python
# Missing ME frames: interpolate
me_aligned = align_motion_energy(me, n_frames)

# Zero-range protection in normalization
def normalize_motion_energy(me):
    me_min = np.nanmin(me)
    me_max = np.nanmax(me)
    if me_max - me_min < 1e-10:
        return np.zeros_like(me)
    return (me - me_min) / (me_max - me_min)
```

iii. The AI noted the missing frames issue from the data README and chose to interpolate. However, the interpolation approach assumes missing frames are at the end, rather than using `tstamps.npy` or `interframe_int.npy` to identify the actual positions of missing frames, as suggested by the data README.

## 6-a. What are the most time-consuming steps of the code?

i. The dF/F computation (baseline estimation with gaussian, min, and max filters on the full session) is the most expensive step. For a session with 685 neurons and 54000 frames, it takes ~2.8 seconds. Total conversion for all 41 sessions takes ~80 seconds.

ii.
```python
# Baseline estimation involves three filter operations on full arrays:
Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
```

iii. From the conversion output, per-session processing times range from 0.6s (221 neurons, 36000 frames) to 3.05s (746 neurons, 54000 frames). The AI noted in CONVERSION_NOTES.md: "Code inefficiencies identified: None significant - vectorized operations used throughout."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over trials sequentially to extract slices, which is already efficient since it's just array slicing. The motion energy discretization loop iterates per-session, which is also minimal overhead. No significant vectorization opportunities exist because the main processing (dF/F, binning) is already vectorized.

ii.
```python
# Trial segmentation loop - simple array slicing, hard to improve
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    neural_trials.append(neural_trial)
```

iii. The AI's code is well-vectorized. The `bin_data` function uses reshape+mean which is efficient. The main filtering operations use scipy's optimized C implementations.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each session is processed once. The motion energy is concatenated and then re-indexed for per-trial discretization, but this is necessary for the per-session percentile computation.

ii. No duplicated processing observed.

iii. The code has a clean single-pass architecture: load -> dF/F -> bin -> segment -> discretize -> save.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The per-session min-max normalization of motion energy before percentile binning is unnecessary - computing percentile bins directly on the raw (or binned) motion energy values would produce identical bin assignments, since percentiles are invariant to monotonic transformations.

ii.
```python
# This normalization step is unnecessary since percentile binning follows
me_norm = normalize_motion_energy(me_all)  # min-max to [0,1]
# Percentile edges computed on normalized values produce same bins as on raw values
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1])
```

iii. The AI included normalization as an explicit step ("Per-session min-max normalization before percentile binning") per the task description mentioning "normalized," but since the subsequent discretization uses equal-percentile bins, the normalization is mathematically redundant.
