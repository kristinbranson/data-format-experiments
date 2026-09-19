# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 6 mouse IDs (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each mouse, it scans subdirectories whose names start with a digit to identify sessions. From each session directory, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/`, `ops.npy` for frame rate, and `motion_energy_glob.npy` from `move_deve/`.

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

iii. The AI hardcoded the mouse list based on exploration of the data directory. Session directories are identified by starting with a digit (date-based naming). The AI also loads `ops.npy` to extract the frame rate, which the reference does not.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of 6 mouse IDs. In sample mode, only the first 2 mice are used.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
mice_to_process = MICE[:2] if sample else MICE
```

iii. The AI identified the mice from directory exploration and hardcoded them. This produces the same result as the reference's dynamic directory scanning approach.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each mouse's directory whose names start with a digit, sorted alphabetically. In sample mode, only 2 sessions per mouse are used.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions

if sample:
    sessions = sessions[:2]
```

iii. The filtering for directories starting with a digit ensures non-session directories (e.g., `ground_truth.csv`) are excluded.

## 1-d. How are the data split into trials?

i. The AI splits each session into 2-minute (120-second) non-overlapping trials. After binning by 10 frames, this gives 360 time bins per trial. Remainder bins at the end of a session are discarded.

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

iii. The AI justified this by citing the paper's methods: "splits were done on consecutive 2 minute blocks of the recording." However, the task instructions explicitly state "Split sessions into 60-second trials," which the AI did not follow.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials (those that fill a full trial duration) are kept; only remainder bins at the end of a session are discarded.

ii. N/A (no filtering code)

iii. There are no explicit trial quality criteria mentioned in the paper for continuous recordings, and the AI chose not to apply any.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The AI also loads `ops.npy` to extract the frame rate.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
```

iii. These are the standard suite2p output files for calcium imaging data.

## 2-b. How is the `neural` data processed?

i. The AI reimplements suite2p's baseline correction manually using scipy filters: (1) neuropil subtraction with coefficient 0.7, (2) Gaussian smoothing with sigma=10, (3) minimum filtering with window=1800 frames (60s * 30Hz), (4) maximum filtering with same window, (5) baseline subtraction (Fc - Flow). The AI initially computed dF/F as (Fc - Flow) / Flow but fixed this to subtraction-only after observing extreme values.

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
```

iii. The AI chose to reimplement the baseline correction manually rather than importing suite2p's `dcnv.preprocess`. The CONVERSION_NOTES document that the bug with division was fixed. The manual reimplementation matches the logic described in the paper and the suite2p source code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. The AI notes that all `iscell` values are already 1.0 in the provided data, indicating pre-filtering.

ii. N/A (no filtering code)

iii. The AI documents in CONVERSION_NOTES: "All iscell values are 1.0 (already filtered)" and "iscell threshold 0.5" from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of the continuous recording, no event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event in this free behavior paradigm. The recording is continuous and trials are artificially segmented.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and behavioral data are averaged into non-overlapping bins of 10 frames, going from 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE  # 333.33 ms

dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

iii. The methods text states "averaging in bins of 10 consecutive timestamps." Both streams are binned together to maintain alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index and the bin duration, giving seconds from the start of the session.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. With a constant frame rate of 30 Hz and fixed bin size, time can be computed directly from bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Each time bin's value is computed as `bin_index * TIME_BIN_MS / 1000.0`, where TIME_BIN_MS = 333.33 ms. The bin index is the global index within the session (not reset per trial), so time increases continuously across trials.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. This gives time elapsed from session start in seconds.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time bins are computed from the same bin indices used for neural and output data, so alignment is inherent.

ii. Same code as 3-a/3-b.

iii. Since all three data streams (neural, input, output) are indexed by the same bin indices within each trial, they are automatically aligned.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Unlike the reference, the AI does NOT use `interframe_int.npy` for dropped frame detection.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) the motion energy trace is aligned to the neural frame count by padding with NaN and interpolating if shorter, or truncating if longer; (2) the aligned trace is averaged into 10-frame bins; (3) per-session min-max normalization is applied; (4) the normalized values are discretized into 5 equal-percentile bins using `np.digitize`.

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

# Then after binning:
me_norm = normalize_motion_energy(me_all)  # min-max to [0,1]
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1])
```

iii. The AI normalizes motion energy to [0, 1] before percentile binning, which is unnecessary since percentile-based binning is invariant to monotonic transformations. The AI's method of handling dropped frames (NaN padding + interpolation at the end) differs from the reference's interframe-interval-based approach.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session using `np.digitize` with edges computed from `np.percentile` at [0, 20, 40, 60, 80, 100].

ii.
```python
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. Equal-percentile binning ensures approximately equal frequency in each bin (~20% each), as specified in the instructions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by padding the shorter ME array to match the neural frame count using NaN followed by linear interpolation, or truncating if longer. After alignment, both streams are binned by 10 frames and split into trials using the same indices.

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

iii. This approach assumes dropped frames are at the end of the recording. The reference instead uses interframe intervals to detect where within the recording frames were dropped and interpolates at those specific locations. The AI's approach may introduce misalignment if frames are dropped mid-recording.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (motion energy shorter than neural data) are handled by padding with NaN at the end and linearly interpolating. If motion energy is longer than neural data, it is truncated. Remainder bins at the end of a session that don't fill a complete trial are discarded.

ii.
```python
def align_motion_energy(me, n_neural_frames):
    if len(me) < n_neural_frames:
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        ...
    else:
        return me[:n_neural_frames].astype(np.float64)
```

iii. The AI's approach to missing frames is simpler than the reference's interframe-interval-based detection but may be less accurate if frames are dropped mid-recording rather than at the end.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F baseline correction computation, which involves Gaussian filtering, minimum filtering, and maximum filtering over the full session length for every neuron. The AI reports ~0.8s per session total conversion time.

ii. N/A

iii. The AI reimplements baseline correction using scipy filters on CPU, while the reference uses suite2p's `dcnv.preprocess` which can use GPU acceleration.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over trials to slice arrays, but this is straightforward and cannot be easily vectorized further. The code is already largely vectorized with numpy operations.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end]
    ...
```

iii. The AI's code is reasonably well vectorized. The main loop over trials is for organization (creating list-of-arrays format) rather than computation.

## 6-c. What processing does the code repeat multiple times?

i. The motion energy normalization is conceptually repeated: `normalize_motion_energy` normalizes to [0,1], then percentile edges are computed on the normalized data. Since percentile-based binning is invariant to monotonic scaling, the normalization step is redundant.

ii.
```python
me_norm = normalize_motion_energy(me_all)  # redundant step
edges = np.percentile(me_norm, percentiles)
```

iii. The normalization does not change the outcome but adds unnecessary computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The min-max normalization of motion energy before percentile binning is unnecessary since percentile-based discretization is invariant to monotonic transformations. Loading `ops.npy` to extract frame rate is also unnecessary if the frame rate is known to be constant at 30 Hz (as it is in this dataset).

ii.
```python
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)

me_norm = normalize_motion_energy(me_all)
```

iii. While loading ops.npy is a reasonable defensive practice, the normalization step adds complexity without changing the result.
