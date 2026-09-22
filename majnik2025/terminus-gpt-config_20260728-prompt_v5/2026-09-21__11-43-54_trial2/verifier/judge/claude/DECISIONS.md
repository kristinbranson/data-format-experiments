# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by scanning the data root for directories starting with `jm` (subjects), then iterating over subdirectories (sessions). For each session it loads suite2p outputs (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) from `suite2p/plane0/` and motion/behavior files (`motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`) from `move_deve/`. It only includes sessions where both `suite2p/plane0` and `move_deve` directories exist.

ii.
```python
def discover_sessions(data_root):
    data_root = Path(data_root)
    sessions = []
    for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
        for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
            plane = sess_dir / 'suite2p' / 'plane0'
            move = sess_dir / 'move_deve'
            if plane.exists() and move.exists():
                sessions.append({...})
    return sessions

def load_session_arrays(sess):
    F = np.load(plane / 'F.npy').astype(np.float32)
    Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
    iscell = np.load(plane / 'iscell.npy')
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
    tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
    interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
    return F, Fneu, iscell, ops, motion, tstamps, interframe
```

iii. The AI's CONVERSION_NOTES.md documents that subjects are `jm*` directories and sessions are subdirectories with suite2p and move_deve data. The approach mirrors the data's directory structure.

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted directories starting with `jm` in the data root. A subject-to-index mapping is built from the unique sorted subject names across all discovered sessions.

ii.
```python
for subj_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')]):
    ...

subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI notes that there are 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046), consistent with the paper stating "6 mice."

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject folder that contains both `suite2p/plane0/` and `move_deve/` directories is treated as a session. Sessions are sorted alphabetically.

ii.
```python
for sess_dir in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
    plane = sess_dir / 'suite2p' / 'plane0'
    move = sess_dir / 'move_deve'
    if plane.exists() and move.exists():
        sessions.append({
            'subject': subj_dir.name,
            'session': sess_dir.name,
            'session_dir': sess_dir,
            'plane_dir': plane,
            'move_dir': move,
        })
```

iii. The AI's CONVERSION_NOTES.md confirms 41 total sessions across 6 subjects, with 6-7 sessions per subject.

## 1-d. How are the data split into trials?

i. Trials are created by splitting continuous sessions into non-overlapping 60-second segments. With 30 Hz frame rate and 10-frame binning, each trial has 180 time bins. Sessions with fewer than 2 trials are excluded. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
trial_len_bins = int(round(60.0 / (bin_size / float(ops.get('fs', 30.0)))))

def split_into_trials(neural, inp, out, trial_len_bins):
    n_time = neural.shape[1]
    n_trials = n_time // trial_len_bins
    if n_trials < 2:
        return [], [], []
    keep = n_trials * trial_len_bins
    neural = neural[:, :keep]
    ...
    neural_trials = [neural[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
```

iii. The decoder task instructions specify "Split sessions into 60-second trials" and require at least 2 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. The only filtering is excluding sessions with fewer than 2 trials.

ii.
```python
if n_trials < 2:
    return [], [], []
...
if len(neural_trials) < 2:
    continue
```

iii. The AI notes there is no native trial structure, so no trial quality criteria are applied. The minimum of 2 trials per session is enforced by the format requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from the `suite2p/plane0/` directory. The AI also loads `iscell.npy` and `ops.npy` but does not use `iscell` for filtering.

ii.
```python
F = np.load(plane / 'F.npy').astype(np.float32)
Fneu = np.load(plane / 'Fneu.npy').astype(np.float32)
iscell = np.load(plane / 'iscell.npy')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```

iii. The AI identifies F.npy and Fneu.npy as the standard suite2p output files for raw and neuropil fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`F - 0.7 * Fneu`) and then computes a simple percentile-based dF/F: it takes the 8th percentile of each neuron's corrected trace as the baseline, then computes `(Fc - baseline) / |baseline|`. This is NOT suite2p's `dcnv.preprocess` with `maximin` baseline — it is a simpler approximation.

ii.
```python
def compute_dff(F, Fneu, neuropil_coeff=0.7, baseline_percentile=8.0):
    Fc = F - neuropil_coeff * Fneu
    baseline = np.percentile(Fc, baseline_percentile, axis=1, keepdims=True).astype(np.float32)
    baseline = np.where(np.abs(baseline) < 1e-3, 1e-3, baseline)
    dff = (Fc - baseline) / np.abs(baseline)
    return dff.astype(np.float32)
```

iii. The AI's CONVERSION_NOTES.md states: "Reconstruct an approximate Suite2p-style baseline-corrected fluorescence signal during conversion." The AI acknowledged it was creating an approximation of the dF/F signal described in the paper's methods.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. The AI loads `iscell.npy` but does not use it to filter neurons. All neurons in F.npy are included.

ii.
```python
iscell = np.load(plane / 'iscell.npy')
# iscell is loaded but never used for filtering
neural = compute_dff(F, Fneu, ...)  # uses all rows of F
```

iii. The AI's CONVERSION_NOTES.md states: "Apply no extra iscell filtering: All provided tracked ROIs pass the Suite2p 0.5 threshold; additional filtering would diverge from the provided curated dataset."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments, no event-based alignment is needed. The AI sets `temporal_alignment_event` to `'session start'` with `off_start: 0.0` and `off_end: 60.0`.

ii.
```python
'temporal_alignment_event': 'session start',
'off_start': 0.0,
'off_end': 60.0,
```

iii. There is no stimulus event to align to; the recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy data are rebinned by averaging in non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bins). This matches the paper's description.

ii.
```python
bin_size = 10
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
...
'time_bin_size': 1000.0 * (10.0 / 30.0),  # ms
```

iii. The methods state: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is derived from the `tstamps.npy` file (converted from day-based units to seconds by multiplying differences by 86400) or from the frame rate (`ops['fs']`) as a fallback. It is then binned using 10-frame averaging.

ii.
```python
def get_neural_frame_times_sec(n_frames, ops, tstamps=None):
    fs = float(ops.get('fs', 30.0))
    if tstamps is not None and len(tstamps) >= n_frames:
        dt = np.diff(tstamps[:n_frames])
        if len(dt) > 0:
            dt_sec = float(np.median(dt) * 86400.0)
            if 0.5 / fs < dt_sec < 1.5 / fs:
                t = (tstamps[:n_frames] - tstamps[0]) * 86400.0
                return np.asarray(t, dtype=np.float32)
    return (np.arange(n_frames, dtype=np.float32) / fs).astype(np.float32)
```

iii. The AI investigated the timestamp units and determined they are in day-based units requiring conversion to seconds. If timestamps are unreliable, it falls back to computing time from the frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The timestamps are converted from day-units to seconds (multiply by 86400), referenced to session start (subtract first timestamp), then averaged into 10-frame bins using the same binning function applied to neural and motion data.

ii.
```python
t = (tstamps[:n_frames] - tstamps[0]) * 86400.0
...
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
inp = time_b[None, :].astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md that timestamps needed unit conversion and cross-checked with the frame rate.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is derived from the same frame count as the neural data, so it is inherently aligned. Both are binned with the same 10-frame averaging. The time input represents the mean timestamp of each bin.

ii.
```python
neural_times_sec = get_neural_frame_times_sec(n_frames, ops, tstamps)
...
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
```

iii. By deriving time from the same frame indices as the neural data, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory. Timestamps (`tstamps.npy`) are used for temporal alignment with the neural data.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy').astype(np.float32)
tstamps = np.load(move / 'tstamps.npy').astype(np.float64)
```

iii. The motion energy file contains the pre-computed global motion energy from behavioral video, as described in the paper's methods.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) motion energy is temporally aligned/interpolated onto neural frame times using timestamps, (2) the aligned trace is averaged into 10-frame bins, (3) the binned trace is discretized into 5 equal-percentile bins computed per session.

ii.
```python
motion_aligned = align_motion_to_neural(motion, tstamps, neural_times_sec, ops)
...
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
motion_disc, edges = discretize_into_quantile_bins(motion_b, n_bins=5)
```

iii. The AI's CONVERSION_NOTES.md documents: "session-wise 5-bin discretization" and "10-frame binning" matching the paper's methods.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The binned motion energy is discretized into 5 equal-percentile bins per session using `np.quantile` to compute edges, with edge deduplication to handle ties. `np.digitize` maps values to bin indices 0-4.

ii.
```python
def discretize_into_quantile_bins(values, n_bins=5):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    labels = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    return labels, edges
```

iii. The decoder task specifies "five equal-percentile bins, selected per session." The AI's edge deduplication handles edge cases with many identical values.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses timestamp-based interpolation. It converts timestamps from day-units to seconds, then interpolates the motion energy signal onto neural frame times using `np.interp`. If timestamps are unavailable or unreliable, it falls back to linear interpolation based on array length.

ii.
```python
def align_motion_to_neural(motion, tstamps, neural_times_sec, ops):
    n_motion = len(motion)
    if len(tstamps) >= n_motion:
        mt = (tstamps[:n_motion] - tstamps[0]) * 86400.0
        good = np.isfinite(mt) & np.isfinite(motion)
        mt = mt[good]; mv = motion[good]
        if len(mt) >= 2 and np.all(np.diff(mt) > 0):
            return np.interp(neural_times_sec, mt.astype(np.float32), mv.astype(np.float32)).astype(np.float32)
    ...
```

iii. The AI notes in CONVERSION_NOTES.md that some sessions have behavior/neural length mismatches (1 to 148 frames) and that timestamp-based interpolation handles dropped camera frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where motion energy array is shorter than neural data) are handled by interpolating the motion energy signal onto neural frame times using timestamps. The AI has multiple fallback strategies: timestamp-based interpolation, direct matching if lengths are equal, or linear resampling as a last resort. Remainder frames that don't fill a complete trial are discarded.

ii.
```python
def align_motion_to_neural(motion, tstamps, neural_times_sec, ops):
    # Primary: timestamp-based interpolation
    ...
    # Fallback 1: direct match
    if len(motion) == len(neural_times_sec):
        return motion.astype(np.float32)
    # Fallback 2: linear resampling
    x_old = np.linspace(0, 1, num=len(motion), dtype=np.float32)
    x_new = np.linspace(0, 1, num=len(neural_times_sec), dtype=np.float32)
    return np.interp(x_new, x_old, motion.astype(np.float32)).astype(np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md that 9 sessions have motion/timestamp arrays shorter than neural arrays by 1 to 148 frames, and chose interpolation as the alignment approach.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading numpy files (I/O bound), (2) computing dF/F (percentile computation across all neurons), and (3) the 10-frame binning with reshape operations. The AI does not use suite2p's dcnv.preprocess, so there is no GPU-accelerated baseline correction step.

ii. N/A

iii. The AI's CONVERSION_NOTES.md reports processing time of ~0.17 s/session, suggesting the code is relatively fast.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop creates individual trial arrays using list comprehensions with slicing, which is already reasonably efficient. The edge deduplication loop in `discretize_into_quantile_bins` iterates over bin edges but this is a very small loop (5 iterations).

ii.
```python
neural_trials = [neural[:, i * trial_len_bins:(i + 1) * trial_len_bins].astype(np.float32) for i in range(n_trials)]
```

iii. The AI's code is generally well-vectorized, using numpy operations for neuropil correction, percentile computation, binning, and interpolation.

## 6-c. What processing does the code repeat multiple times?

i. The `bin_time_series` function is called three times separately for neural, motion, and time data, each performing the same reshape-and-mean operation. These could potentially be combined if the arrays were stacked.

ii.
```python
neural_b = bin_time_series(neural, bin_size, axis=1, reducer='mean')
motion_b = bin_time_series(motion_aligned[None, :], bin_size, axis=1, reducer='mean')[0]
time_b = bin_time_series(neural_times_sec[None, :], bin_size, axis=1, reducer='mean')[0]
```

iii. This is a minor inefficiency; the binning operations are fast and the data structures differ in shape.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `iscell.npy` and `interframe_int.npy` for every session but neither is used in the final processing pipeline. `iscell` is loaded but never used for filtering. `interframe_int` is loaded but the alignment is done via timestamps instead.

ii.
```python
iscell = np.load(plane / 'iscell.npy')
interframe = np.load(move / 'interframe_int.npy').astype(np.float64)
# Neither iscell nor interframe is used in process_session()
```

iii. These files are loaded as part of a comprehensive loading function but are not utilized in the processing pipeline.
