# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 6 mouse IDs (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it scans the mouse directory for session subdirectories whose names start with a digit. For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` from `move_deve/`. No interframe interval data is loaded.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions

# In process_session:
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified the standard suite2p output structure and motion energy files. It hardcoded the mouse list rather than dynamically discovering them. It also loads `ops.npy` to extract the frame rate, which the reference solution does not do.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of mouse IDs. Each mouse corresponds to a top-level directory in the data folder.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
mice_to_process = MICE[:2] if sample else MICE
```

iii. The AI identified the 6 mice from the data directory and hardcoded them. This produces the same result as the reference's dynamic discovery approach (`d.name.startswith('jm')`).

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each mouse folder, filtered to those starting with a digit character, sorted alphabetically.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. The digit-prefix filter is used to exclude non-session files like `ground_truth.csv`. The reference solution includes all subdirectories.

## 1-d. How are the data split into trials?

i. The AI uses 120-second (2-minute) non-overlapping blocks as trials. After 10-frame temporal binning, each trial is 360 binned timepoints. The paper mentions "consecutive 2 minute blocks" for cross-validation splits.

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

iii. The AI chose 2-minute blocks based on the paper's mention of "5 fold splits ... done on consecutive 2 minute blocks." CONVERSION_NOTES.md states: "Trial segmentation: Use 2-minute blocks (3600 frames at 30Hz). Paper uses 'consecutive 2 minute blocks' for CV splits."

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete trials (those that fill a full 2-minute block) are included for each session.

ii. N/A - no filtering code exists.

iii. No trial quality filtering is mentioned in the paper for this continuous recording paradigm.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The frame rate is read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
```

iii. These are the standard suite2p output files for calcium imaging data.

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) maximin baseline correction reimplemented manually using scipy filters (gaussian_filter, minimum_filter1d, maximum_filter1d), yielding `dff = Fc - Flow`, (3) temporal binning by averaging 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)  # window in frames
    Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    dff = Fc - Flow
    return dff.astype(np.float32)

# Temporal binning:
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)  # BIN_SIZE = 10
```

iii. CONVERSION_NOTES.md states the AI initially used division `(Fc-Flow)/Flow` but fixed it to subtraction-only `Fc-Flow` to match suite2p's actual implementation. The 10-frame binning follows the paper's methods: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI notes that all `iscell` values are already 1.0 in the provided data.

ii. N/A - no filtering code.

iii. CONVERSION_NOTES.md states: "All iscell values are 1.0 (already filtered)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Trials are contiguous segments starting from the beginning of each session.

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event in this continuous recording paradigm. The AI aligns to session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies 10-frame temporal binning, resulting in a time bin size of 333.33 ms (10 frames / 30 Hz). Both neural and behavioral data are binned.

ii.
```python
BIN_SIZE = 10           # frames to average for denoising
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE  # 333.33 ms

dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

iii. The AI followed the paper's methods section: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index after temporal binning.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. Since the frame rate is constant and bins are uniform, time is computed analytically from indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * TIME_BIN_MS / 1000.0`, where `TIME_BIN_MS = 333.33 ms`. The bin indices accumulate across trials within a session, so the time represents elapsed time from the start of the recording session.

ii.
```python
start = t * BINNED_PER_TRIAL  # t is the trial index
end = (t + 1) * BINNED_PER_TRIAL
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. This gives monotonically increasing time across trials within a session.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices used for neural data, so alignment is inherent.

ii. Same indices used for both: `start = t * BINNED_PER_TRIAL`, `end = (t + 1) * BINNED_PER_TRIAL`.

iii. Since both neural and time data use the same bin indices, they are trivially aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory. Unlike the reference, the AI does NOT load `interframe_int.npy`.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified this as the pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) alignment to neural frame count via linear interpolation when ME is shorter (padding with NaN then interpolating), (2) temporal binning by averaging 10 consecutive frames, (3) per-session min-max normalization to [0, 1], (4) discretization into 5 equal-percentile bins per session.

ii.
```python
# Alignment:
def align_motion_energy(me, n_neural_frames):
    if len(me) < n_neural_frames:
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        if np.any(np.isnan(me_aligned)):
            valid = ~np.isnan(me_aligned)
            indices = np.arange(n_neural_frames)
            me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
        return me_aligned

# Binning:
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)

# Normalization:
def normalize_motion_energy(me):
    me_min = np.nanmin(me)
    me_max = np.nanmax(me)
    return (me - me_min) / (me_max - me_min)

# Discretization (per session):
me_norm = normalize_motion_energy(me_all)
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1])
```

iii. CONVERSION_NOTES.md states: "Per-session min-max normalization before percentile binning." The AI chose per-session normalization and percentile computation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using per-session equal-percentile edges. Min-max normalization is applied per session before computing percentiles. `np.digitize` assigns bin labels 0-4.

ii.
```python
me_norm = normalize_motion_energy(me_all)  # per-session min-max to [0,1]
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. Per-session percentiles ensure each session individually has a uniform bin distribution (exactly 20% per bin).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When motion energy has fewer frames than neural data, the AI pads with NaN and uses linear interpolation (`np.interp`) to fill missing values at the end. It does NOT use interframe interval data to identify where frames were dropped.

ii.
```python
def align_motion_energy(me, n_neural_frames):
    if len(me) < n_neural_frames:
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        if np.any(np.isnan(me_aligned)):
            valid = ~np.isnan(me_aligned)
            indices = np.arange(n_neural_frames)
            me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
        return me_aligned
```

iii. The AI noted that motion energy sometimes has slightly fewer frames than neural data but used simple end-padding interpolation rather than using interframe intervals to identify dropped frame locations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion energy frames are handled by padding NaN at the end and interpolating. If ME is longer than neural data, it is truncated. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
# ME shorter than neural:
me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_aligned[:len(me)] = me.astype(np.float64)
me_aligned = np.interp(indices, indices[valid], me_aligned[valid])

# ME longer than neural:
return me[:n_neural_frames].astype(np.float64)
```

iii. The AI handles length mismatches but does not use the available interframe interval data to correctly identify where frames were dropped.

## 6-a. What are the most time-consuming steps of the code?

i. The manual baseline correction using scipy filters (gaussian_filter, minimum_filter1d, maximum_filter1d) on full-session data is the most time-consuming step. The full conversion takes ~80 seconds for 41 sessions.

ii.
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
```

iii. The conversion log shows ~0.6-3s per session depending on neuron count, with the baseline correction being the dominant computation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. No significant loops that could be vectorized. The trial segmentation loop is simple index slicing. The per-session discretization loop could theoretically be combined but is not a bottleneck.

ii. N/A

iii. The AI used vectorized numpy operations throughout.

## 6-c. What processing does the code repeat multiple times?

i. The AI computes min-max normalization of motion energy and then recomputes the normalized values when discretizing per trial (`me_trial_norm = me_norm[offset:offset + n_t]`). This is a minor redundancy.

ii.
```python
me_norm = normalize_motion_energy(me_all)
# Then later per trial:
me_trial_norm = me_norm[offset:offset + n_t]
```

iii. The per-trial slicing from the already-normalized array is not truly repeated computation; it's just indexing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ops.npy` to extract the frame rate but then uses it only to confirm it matches the hardcoded `FRAME_RATE = 30`. The temporal binning step (averaging 10 frames) adds processing that the reference solution does not perform and may not be needed for the decoder.

ii.
```python
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
```

iii. Loading ops.npy is a minor unnecessary I/O operation.
