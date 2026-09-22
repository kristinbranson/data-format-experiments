# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 6 subjects and iterates over them. For each subject, it discovers session directories by listing sorted subdirectories. For each session, it loads `F.npy`, `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` from `move_deve/`.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
def get_session_dirs(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))])
    return [os.path.join(subj_dir, s) for s in sessions]
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The AI identified the standard directory structure (subject folders containing session subfolders with suite2p output and motion energy files). Subject names are hardcoded rather than discovered dynamically but produce the same result.

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded as a list of 6 mouse IDs. Each subject corresponds to a top-level directory in the data folder.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for subj_i, subject in enumerate(SUBJECTS):
    sess_dirs = get_session_dirs(subject)
```

iii. The AI determined the 6 subject IDs from exploring the data directory structure and hardcoded them for reliability.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject's directory. Each subdirectory contains one daily recording session.

ii.
```python
def get_session_dirs(subject):
    subj_dir = os.path.join(DATA_DIR, subject)
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))])
    return [os.path.join(subj_dir, s) for s in sessions]
```

iii. The AI noted that each subdirectory contains suite2p and motion energy data for one recording session. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. Trials are artificially defined as 60-second non-overlapping segments of the continuous recording. After binning (10 frames at 30 Hz → 3 Hz), each trial is 180 bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
TRIAL_BINS = int(TRIAL_DURATION_SEC * BINNED_FS)  # 180 bins per trial
...
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. Per the task instructions, trials are defined as 60-second segments. Since there is no natural trial structure (spontaneous behavior), fixed-length segmentation is appropriate.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second segments are included.

ii. N/A - no filtering code exists.

iii. The AI noted there is no stimulus-driven trial structure and no explicit trial curation rules in the reference materials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) in the `suite2p/plane0/` subdirectory of each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI reimplements suite2p's baseline correction manually rather than calling `suite2p.extraction.dcnv.preprocess`. Processing steps: (1) neuropil subtraction (`Fc = F - 0.7 * Fneu`), (2) Gaussian smoothing with sigma=10, (3) minimum filter with window=1800 frames (60s), (4) maximum filter with same window, (5) baseline subtraction (`dff = Fc - Flow`). The result is then binned by averaging 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF,
                sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)  # 1800 frames
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff.astype(np.float32)
```

iii. The AI read the reference code's `F_processing` function in `track2p/gui/data_management.py` and reimplemented the maximin baseline method manually. The AI noted that the paper says "using the default Suite2p parameters" and chose neucoeff=0.7 (from ops.npy) rather than 0.0 (from the GUI code). The AI also noted that the code does baseline subtraction only (Fc - Flow), not division ((Fc - Flow)/Flow).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the suite2p output are included.

ii. No filtering code exists.

iii. The AI noted that all `iscell` values are 1.0 (pre-filtered by Track2p), so all neurons in the data are tracked cells across all days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_SEC),
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # ~333.33 ms
...
dff_binned = bin_data(dff, BIN_SIZE, axis=1)
me_binned = bin_data(me_full.reshape(1, -1), BIN_SIZE, axis=1).squeeze()
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index and the bin duration, giving seconds from the start of that session.

ii.
```python
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
...
input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
```

iii. Since the frame rate is constant at 30 Hz and there are no stored timestamps, computing time from bin indices is equivalent.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * BIN_SIZE / FS` for each bin. The time runs continuously across the whole session, so the first trial starts at 0.0s and subsequent trials start at multiples of 60s.

ii.
```python
time_seconds = (np.arange(n_total_bins) * BIN_SIZE / FS).astype(np.float32)
```

iii. No additional processing beyond the index-to-time conversion.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data since both use the same bin indices. Each time bin corresponds exactly to one neural data bin.

ii.
```python
# Same start:end slice used for both:
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
input_trials.append(time_seconds[start:end].reshape(1, -1).astype(np.float32))
```

iii. No alignment step is needed since both are derived from the same binning scheme.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Additionally, `tstamps.npy` (timestamps) and `interframe_int.npy` (interframe intervals) are used to handle missing camera frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The motion energy file contains pre-computed global motion energy. The timestamps and interframe intervals are needed to detect and interpolate dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped camera frames are detected and interpolated using `np.interp` based on the ratio of each interframe interval to the median IFI, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 percentile-based bins per session.

ii.
```python
def interpolate_missing_frames(me, n_neural_frames, tstamps, ifi):
    median_ifi = np.median(ifi)
    neural_indices = np.zeros(len(me), dtype=int)
    neural_idx = 0
    neural_indices[0] = 0
    for i in range(len(ifi)):
        n_skip = max(1, round(ifi[i] / median_ifi))
        neural_idx += n_skip
        if i + 1 < len(me):
            neural_indices[i + 1] = neural_idx
    me_full = np.interp(np.arange(n_neural_frames), neural_indices, me.astype(np.float64))
    return me_full
...
me_discrete, bin_edges = discretize_motion_energy(me_binned, N_BINS_OUTPUT)
```

iii. The AI used median-based gap detection and `np.interp` for interpolation, which maps each camera frame to the corresponding neural frame index and linearly interpolates missing positions. Discretization uses `np.percentile` and `np.digitize` per session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins computed per session. The bin edges are computed with `np.percentile` at [0, 20, 40, 60, 80, 100] and `np.digitize` maps values to bins 0-4.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS_OUTPUT):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
    me_discrete = np.clip(me_discrete, 0, n_bins - 1)
    return me_discrete.astype(np.int64), bin_edges
```

iii. The instructions specify "five equal-percentile bins, selected per session." The AI followed this by computing percentile boundaries within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera and neural data are both acquired at 30 Hz but occasionally camera frames are dropped. The AI detects drops by comparing each interframe interval to the median IFI, builds a mapping from camera frames to neural frame indices, and uses `np.interp` to fill gaps. After interpolation, the ME and neural data are the same length and are sliced identically.

ii.
```python
me_full = interpolate_missing_frames(me, n_frames, tstamps, ifi)
...
# Same indices used for neural and output:
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI's interpolation approach builds a frame-to-frame mapping rather than inserting individual frames at detected drop positions (as the reference does). Both achieve the goal of matching ME length to neural length.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected by comparing interframe intervals to the median and interpolated using `np.interp`. Remainder frames at the end of a session that don't fill a complete 60-second trial are discarded. If ME length already matches neural length, no interpolation is done.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
...
me_full = np.interp(np.arange(n_neural_frames), neural_indices, me.astype(np.float64))
```

iii. The AI handles the known issue of dropped camera frames and the expected issue of remainder frames. No other data quality issues were identified.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation (baseline correction), which takes 0.25-1.27s per session depending on neuron count. This dominates the per-session processing time.

ii. N/A (from conversion output logs:)
```
time: load=0.07s dff=1.27s bin=0.08s total=1.42s
```

iii. The baseline correction involves sliding window operations (gaussian + min + max filters) over the full session length for every neuron.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interpolate_missing_frames` function contains a Python loop over all interframe intervals to build the neural-to-camera frame mapping. This could be vectorized with cumulative sum operations.

ii.
```python
for i in range(len(ifi)):
    n_skip = max(1, round(ifi[i] / median_ifi))
    neural_idx += n_skip
    if i + 1 < len(me):
        neural_indices[i + 1] = neural_idx
```

iii. The number of frames is large (36000-54000), so this loop runs many times, though since it's simple arithmetic the overhead is modest.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once.

ii. N/A

iii. The AI's code processes each session in a single pass through `process_session()`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `tstamps.npy` for every session but only uses it in `interpolate_missing_frames`. When no frames are missing (the common case), the timestamps and IFI data are loaded but unused.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. Loading small .npy files has negligible cost, so this is not a significant inefficiency.
