# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six mice in `MICE`, then iterates through each mouse directory, each date-named session directory, and loads Suite2p fluorescence plus motion-energy files for every session before later splitting the continuous recordings into trials.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    for sess_name in sessions:
        sess_dir = os.path.join(mouse_dir, sess_name)
        suite2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
        move_dir = os.path.join(sess_dir, 'move_deve')

        F = np.load(os.path.join(suite2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
        me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
        ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. `CONVERSION_NOTES.md` says the dataset contains six mice `jm031-jm046`. The trajectory shows the agent explored the dataset structure first, then decided to use the known six mice and per-session Suite2p plus `move_deve` files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the hard-coded `MICE` list rather than discovered dynamically from the filesystem.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
subjects = MICE[:]
...
for mouse_i, mouse in enumerate(MICE):
```

iii. The notes justify this by describing the dataset as exactly six mice. No additional subject-discovery logic is documented in the trajectory.

## 1-c. How are the data split into sessions?

i. For each mouse, sessions are subdirectories whose names start with `'2'` and are sorted lexicographically; each such directory is treated as one recording session.

ii.
```python
def get_sessions(mouse_dir):
    """Get sorted session directories for a mouse."""
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. The trajectory shows the agent inspected the mouse folders and saw date-like session names such as `2023-...`, then used that naming pattern as the session split rule.

## 1-d. How are the data split into trials?

i. The AI treats each continuous session as a sequence of non-overlapping 2-minute trials after 10-frame temporal averaging. Each trial therefore has `360` binned timepoints.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
```

iii. Both `CONVERSION_NOTES.md` and the trajectory justify this from the paper’s decoding text: the agent explicitly cites averaging over 10 timestamps and using consecutive 2-minute blocks.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. Incomplete tails are simply dropped implicitly by floor-division inside the binning and trial-splitting helpers.

ii.
```python
def bin_array(data, bin_size):
    if data.ndim == 1:
        n_bins = len(data) // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_bins = data.shape[1] // bin_size
        return data[:, :n_bins * bin_size].reshape(data.shape[0], n_bins, bin_size).mean(axis=2)

def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. The notes say no additional cell filtering was applied and describe trialing purely as fixed blocks. I did not find a separate trial-QC justification beyond using complete bins and complete trial blocks only.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu)
```

iii. The notes describe these as raw fluorescence and neuropil fluorescence from the matched-cell Suite2p pipeline.

## 2-b. How is the `neural` data processed?

i. The AI computes a dF/F-like signal: neuropil subtraction, a hand-written approximation to Suite2p’s maximin baseline, division by the baseline, then 10-frame averaging.

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

dff_binned = bin_array(dff, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` says the paper used default Suite2p baseline-corrected fluorescence as dF/F, and the trajectory shows the agent first tried a percentile filter, found it too slow, then rewrote the code to a faster maximin-style baseline implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural-quality filtering is applied after loading; all loaded ROIs are kept.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
...
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The notes explicitly justify this as relying on Track2p/Suite2p curation, stating that no additional cell filtering was applied because the dataset already contains matched, verified cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of the recording session, not to any task event. Trials are just consecutive blocks taken from the session timeline.

ii.
```python
start_sec = t_i * TRIAL_DURATION_SEC
...
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
```

iii. The notes justify this by framing the experiment as spontaneous behavior during continuous recording, so the decoder input/output are aligned to session time rather than a stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final data are at 10-frame bins, not the native frame rate. This gives a time bin size of about `333.33 ms` and applies explicit temporal averaging.

ii.
```python
BIN_SIZE = 10
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
'time_bin_size': BIN_DURATION_MS,
```

iii. The notes and trajectory both justify this from the paper’s statement about averaging over 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from a raw file. It is synthesized from trial index, fixed trial duration, frame rate, and bin size.

ii.
```python
start_sec = t_i * TRIAL_DURATION_SEC
time_bins = np.linspace(
    start_sec + BIN_DURATION_MS / 2000,
    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
    TRIAL_BINS
)
```

iii. The notes describe the decoder input as “Time elapsed from start of session,” and the trajectory shows the agent decided to build it from recording time rather than from `tstamps.npy`.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each 2-minute trial, the AI creates a linearly spaced vector of bin-center times in seconds and reshapes it to `(1, timepoints)`.

ii.
```python
input_trials = []
for t_i in range(n_trials):
    start_sec = t_i * TRIAL_DURATION_SEC
    time_bins = np.linspace(
        start_sec + BIN_DURATION_MS / 2000,
        start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
        TRIAL_BINS
    )
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The notes justify the use of bin centers explicitly: “time runs from trial_start to trial_end (centers of bins).”

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is generated on the same per-trial binned grid as the neural data, with one time value per 10-frame neural bin.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
...
time_bins = np.linspace(..., TRIAL_BINS)
input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The notes say the input is trial-wise and time-varying, and the trajectory indicates the agent intentionally matched it to the same 10-frame averaged bins as neural and motion-energy traces.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output motion energy comes from `motion_energy_glob.npy`, with `interframe_int.npy` used to infer dropped video frames for alignment.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
me_aligned = align_motion_energy(me_raw, ifi, n_frames)
```

iii. The notes explicitly identify `motion_energy_glob.npy` as the behavioral source and `interframe_int.npy` as the source for missing-frame detection.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The final code aligns motion energy to neural frame count by inserting gaps based on `interframe_int` and interpolating them, then averages over 10-frame bins. It does not explicitly normalize motion energy in code, even though the notes and top-level docstring say it is normalized.

ii.
```python
def align_motion_energy(me, interframe_int, n_neural_frames):
    median_ifi = np.median(interframe_int)
    aligned = np.full(n_neural_frames, np.nan)
    ...
    if np.any(nans) and not np.all(nans):
        x = np.arange(n_neural_frames)
        aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
    return aligned

me_aligned = align_motion_energy(me_raw, ifi, n_frames)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` justifies three intended steps: interpolation, normalization, and quintile discretization. The trajectory also shows the agent chose 10-frame averaging from the paper. The important discrepancy is that the final code implements interpolation and averaging, but not the claimed normalization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates all binned motion-energy values across sessions, computes global quintile edges, and discretizes each trial into categories `0-4` with `np.digitize`.

ii.
```python
all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)
...
binned = np.digitize(me_trial, bin_edges[1:-1])
session_output.append(binned.reshape(1, -1).astype(np.int64))
```

iii. The notes justify this as using global quintiles so each class contains 20% of the data overall.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by reconstructing dropped video frames from `interframe_int`, interpolating missing positions, then applying the same 10-frame averaging and 2-minute segmentation used for neural data.

ii.
```python
me_aligned = align_motion_energy(me_raw, ifi, n_frames)
...
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
...
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
```

iii. The trajectory shows the agent analyzed frame-count mismatches and decided to infer dropped frames from `interframe_int > 1.5x median`, then interpolate so motion energy could be indexed against neural frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are filled by interpolation after gap detection from `interframe_int`. For neural preprocessing, very small baseline values are clamped to avoid division by zero. Partial bins or partial trials are silently discarded by truncation.

ii.
```python
aligned = np.full(n_neural_frames, np.nan)
...
aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
...
F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
...
return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
```

iii. The notes explicitly justify linear interpolation for missing frames and say it preserves temporal alignment despite some larger gaps. I did not find an explicit justification for silently truncating leftover bins/trials, but that is what the code does.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive part is neural preprocessing, especially the baseline computation inside `compute_dff`, which runs over every neuron and frame for every session. Motion-energy alignment and the later full-dataset concatenation are smaller secondary costs.

ii.
```python
dff = compute_dff(F, Fneu)
...
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
F0 = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The trajectory explicitly shows the agent hit a performance problem with a slower percentile filter, called out dF/F computation as the bottleneck, and rewrote that step to a faster maximin-style baseline.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the per-frame loop in `align_motion_energy`, the per-trial loop that constructs `input_trials`, and the nested loops used to split sessions and discretize outputs.

ii.
```python
for i in range(len(me)):
    if neural_idx < n_neural_frames:
        aligned[neural_idx] = me[i]
    neural_idx += 1
    if i < len(interframe_int):
        n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
        neural_idx += n_dropped

for t_i in range(n_trials):
    ...
    input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. I did not find an explicit optimization justification for keeping these loops. The trajectory only documents optimizing the baseline calculation after the first version was too slow.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats directory scanning via `get_sessions` after already iterating through all sessions, and it keeps multiple passes over motion-energy data: first collecting raw trial arrays for global percentiles and later looping again to discretize them.

ii.
```python
sessions = get_sessions(mouse_dir)
...
for mt in me_trials:
    all_me_values.append(mt)
...
for session_me_trials in output_all_raw:
    session_output = []
    for me_trial in session_me_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])
...
'session_info': {
    mouse: {
        'sessions': get_sessions(os.path.join(data_dir, mouse)),
```

iii. I did not find an explicit justification for these repeated passes; they appear to follow from how the agent staged processing and metadata collection.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores raw motion-energy trials in `output_all_raw` only to replace them later with discretized outputs, collects `all_me_values` solely for percentile calculation, and performs extensive sanity-check printing plus sample-dataset creation that are not used by downstream decoding of the final full dataset.

ii.
```python
output_all_raw = []  # store raw ME values before discretization
all_me_values = []  # collect all binned ME for global percentile computation
...
output_all_raw.append(me_trials)
...
all_me_concat = np.concatenate(all_me_values)
...
sample_data = {
    'neural': data['neural'][:n_sample],
    ...
}
```

iii. The notes justify the sanity checks and sample outputs as validation artifacts, not as part of the decoder-ready representation itself. I found no separate justification for retaining the intermediate raw motion-energy trial lists beyond convenience for the later discretization pass.
