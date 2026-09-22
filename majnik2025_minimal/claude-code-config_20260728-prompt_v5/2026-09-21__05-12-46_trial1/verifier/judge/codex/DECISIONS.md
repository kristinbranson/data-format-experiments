# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six mice in `MICE`, then iterates through each mouse's session directories whose names start with a digit. For each session it loads Suite2p fluorescence files `F.npy` and `Fneu.npy`, loads `ops.npy` for imaging parameters, and loads behavioral motion energy from `move_deve/motion_energy_glob.npy`. Trials are not loaded directly from disk; they are created later by segmenting the continuous session traces into 60-second chunks.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for mouse_idx, mouse in enumerate(MICE):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    ...
    me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. The trajectory shows the AI concluded the dataset was a fixed set of six mice and that the traces were already matched across sessions. In step 27 it notes the data are "already tracked and matched across sessions" and lists the six-mouse structure; in step 62 it summarizes the dataset as "6 mice, 41 sessions total."

## 1-b. How are the data split into subjects?

i. Subjects are the six hard-coded mouse IDs in `MICE`, kept in that explicit order.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
subjects = MICE
...
for mouse_idx, mouse in enumerate(MICE):
```

iii. The AI justified this from the paper/dataset description rather than by scanning the directory tree. Step 62 explicitly summarizes the converted dataset as containing those six mice.

## 1-c. How are the data split into sessions?

i. Within each mouse, sessions are the subdirectories whose names begin with a digit, sorted lexicographically.

ii.
```python
def get_sessions(mouse_dir):
    """Get sorted session directories for a mouse."""
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. There is no extended explicit justification in the trajectory beyond treating each dated folder as one recording day. Step 62 summarizes the result as "41 sessions total (6-7 sessions per mouse)."

## 1-d. How are the data split into trials?

i. The AI treats each recording session as continuous data and splits it into non-overlapping 60-second trials after 10-frame binning. Since the effective rate is 3 Hz, each trial has 180 time bins. Any partial tail shorter than a full 60-second trial is discarded because `n_trials` is floored.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION = 60
...
bin_duration = BIN_SIZE / fs
bins_per_trial = int(TRIAL_DURATION / bin_duration)  # 180 bins per trial
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end].astype(np.float32))
    session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
    session_output.append(me_discrete[start:end][np.newaxis, :].astype(np.int64))
```

iii. Step 62 states the key trial decision directly: "Trials: 60-second non-overlapping segments from continuous recordings."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit per-trial quality-control filter. The only effective exclusion is that incomplete trailing bins at the end of a session are not turned into trials because `n_trials` uses floor division.

ii.
```python
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    ...
```

iii. The trajectory does not describe any trial-level QC beyond the fixed-length trial segmentation. The final summary in step 62 lists only the 60-second segmentation rule, not any additional trial filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p fluorescence files `F.npy` and `Fneu.npy`, and it uses `ops.npy` to obtain parameters such as frame rate and baseline settings.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
fs = ops['fs']
```

iii. Step 27 says the AI needed to "Compute dF/F from F.npy ... using F and Fneu," and step 62 summarizes the final choice as neuropil correction plus maximin baseline subtraction.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil correction, then a manual maximin-style baseline estimate using Gaussian smoothing followed by minimum and maximum filters, and finally subtracts that baseline from the neuropil-corrected trace. After that it averages neural data in non-overlapping 10-frame bins.

ii.
```python
def compute_dff(F, Fneu, fs, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff

...
dff = compute_dff(F, Fneu, fs,
                  neucoeff=ops.get('neucoeff', 0.7),
                  sig_baseline=ops.get('sig_baseline', 10.0),
                  win_baseline=ops.get('win_baseline', 60.0))
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The trajectory shows an explicit course correction. In step 39 the AI first planned a division-based dF/F, but after poor decoder performance and huge values it revised this in step 51, concluding the paper's code used baseline subtraction only. Step 62 reports the final decision as "`Fc = F - 0.7*Fneu` + maximin baseline subtraction (not division)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied in code. All rows of `F.npy`/`Fneu.npy` are used.

ii.
```python
n_neurons, n_frames = F.shape
...
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. Step 27 says the data are "already tracked and matched across sessions" and that "All neurons pass iscell." Step 62 summarizes this as "No neuron filtering needed."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the recording session rather than to an internal behavioral or stimulus event. Trials are contiguous windows within the session.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION),
    ...
}
```

iii. The trajectory does not debate another alignment event. Step 62 describes the input as "Time elapsed from session start" and the trial construction as fixed windows from continuous recordings.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have 333.33 ms time bins. The AI rebins both neural and motion-energy traces by averaging non-overlapping blocks of 10 original 30 Hz frames, yielding an effective 3 Hz sampling rate.

ii.
```python
BIN_SIZE = 10
FS = 30
...
def bin_data(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    data = data[..., :n_bins * bin_size]
    new_shape = data.shape[:-1] + (n_bins, bin_size)
    return data.reshape(new_shape).mean(axis=-1)

...
time_bin_ms = (BIN_SIZE / FS) * 1000  # 333.33 ms
```

iii. Step 27 notes the need to "Bin by 10 frames as specified in the methods," and step 62 repeats that the AI used "10-frame averaging ... (3 Hz effective rate)."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not load a timestamp array for the decoder input. Instead it derives time from the binned sample index, using the session frame rate from `ops.npy` and the fixed 10-frame bin size.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
fs = ops['fs']
...
bin_duration = BIN_SIZE / fs  # seconds per bin
time_vec = np.arange(n_bins) * bin_duration
```

iii. Step 39 says the input feature "will be time elapsed in seconds per bin," and step 62 summarizes the input as "Time elapsed from session start (seconds)."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a uniformly spaced per-bin time vector in seconds starting at 0 for each session after temporal binning. It then slices this vector into the same 60-second trial windows as neural/output data.

ii.
```python
bin_duration = BIN_SIZE / fs  # seconds per bin
time_vec = np.arange(n_bins) * bin_duration
...
session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
```

iii. There is no more elaborate justification in the trajectory. The AI consistently described the decoder input as elapsed seconds per time bin.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time to neural data by constructing `time_vec` with the same number of binned samples as `dff_binned`, then slicing both arrays with the same `[start:end]` indices for each trial.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE)
...
n_bins = dff_binned.shape[1]
time_vec = np.arange(n_bins) * bin_duration
...
session_neural.append(dff_binned[:, start:end].astype(np.float32))
session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
```

iii. This alignment is implicit in the implementation. The trajectory does not discuss a separate alignment procedure beyond using the same binned session timeline for all streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy solely from `move_deve/motion_energy_glob.npy`. It does not use `interframe_int.npy` or `tstamps.npy` in the final implementation.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)
```

iii. Early exploration recognized that timestamp-related files existed and could indicate missing frames, but the final implementation and step 62 summary reduce the decision to "Frame-level data ... truncated to minimum of neural and motion energy frame counts."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI converts motion energy to `float64`, truncates it to the minimum shared length with the neural recording, averages it in 10-frame bins, and then discretizes the binned values within each session into five percentile bins.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy')).astype(np.float64)
common_len = min(n_frames, len(me))
...
me = me[:common_len]
...
me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]
...
percentiles = np.linspace(0, 100, N_ME_BINS + 1)
bin_edges = np.percentile(me_binned, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
me_discrete = np.digitize(me_binned, bin_edges[1:])
me_discrete = np.clip(me_discrete, 0, N_ME_BINS - 1)
```

iii. The final summary in step 62 explicitly states the AI's decision: motion energy is binned like the neural data, discretized per session, and any frame mismatch is handled by truncating to the minimum length.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized separately within each session into five equal-percentile bins. The AI computes session-specific percentile edges from the binned motion-energy trace and assigns category labels with `np.digitize`.

ii.
```python
percentiles = np.linspace(0, 100, N_ME_BINS + 1)
bin_edges = np.percentile(me_binned, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
me_discrete = np.digitize(me_binned, bin_edges[1:])  # values 0 to N_ME_BINS-1
me_discrete = np.clip(me_discrete, 0, N_ME_BINS - 1)
```

iii. Step 27 says the AI needed to "Discretize motion energy into 5 equal-percentile bins per session," and step 62 repeats "5 per-session quintile bins."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by truncating both streams to the shorter of the two frame counts before any downstream processing. After truncation, it bins both with the same 10-frame windows and slices the same trial ranges from both arrays.

ii.
```python
common_len = min(n_frames, len(me))
F = F[:, :common_len]
Fneu = Fneu[:, :common_len]
me = me[:common_len]
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]
...
session_neural.append(dff_binned[:, start:end].astype(np.float32))
session_output.append(me_discrete[start:end][np.newaxis, :].astype(np.int64))
```

iii. Step 62 states the alignment choice directly: "Frame mismatch handling: Truncated to minimum of neural and motion energy frame counts."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles frame-count mismatches by dropping excess neural or motion-energy frames via truncation to `common_len`. It does not interpolate missing behavioral frames. Incomplete trailing segments shorter than a full trial are also dropped implicitly because only full trials are emitted.

ii.
```python
common_len = min(n_frames, len(me))
F = F[:, :common_len]
Fneu = Fneu[:, :common_len]
me = me[:common_len]
...
n_trials = n_bins // bins_per_trial
```

iii. The trajectory shows the AI was aware of possible missing video frames during exploration, but its final implemented decision is the step 62 summary: "Frame mismatch handling: Truncated to minimum of neural and motion energy frame counts."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming processing in the AI's script is the per-session neural preprocessing: loading large arrays, running the Gaussian/minimum/maximum filtering used for baseline estimation, and then trializing all sessions. Decoder training is outside `convert_data.py`, so within the conversion script the neural filtering is the main expensive step.

ii.
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
...
for mouse_idx, mouse in enumerate(MICE):
    ...
    for session_name in sessions:
        ...
        dff = compute_dff(...)
```

iii. The trajectory did not explicitly analyze runtime inside `convert_data.py`, but repeated attention to dF/F computation in steps 39, 48, and 51 indicates that this preprocessing block was the central nontrivial computation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial assembly loop could be vectorized by reshaping session-length arrays directly into `(n_trials, ...)` blocks instead of appending trial slices one by one. The repeated per-session Python loops over mice and sessions also add overhead, but the clearest vectorization opportunity is the trial loop.

ii.
```python
session_neural = []
session_input = []
session_output = []

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    session_neural.append(dff_binned[:, start:end].astype(np.float32))
    session_input.append(time_vec[start:end][np.newaxis, :].astype(np.float32))
    session_output.append(me_discrete[start:end][np.newaxis, :].astype(np.int64))
```

iii. The trajectory does not discuss vectorization. This assessment comes from the structure of the final script.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same load-preprocess-bin-discretize-trialize pipeline independently for every session. It also repeatedly computes percentile edges session-by-session and repeatedly converts sliced trial arrays to new dtypes inside the trial loop.

ii.
```python
for mouse_idx, mouse in enumerate(MICE):
    ...
    for session_name in sessions:
        ...
        dff = compute_dff(...)
        dff_binned = bin_data(dff, BIN_SIZE)
        me_binned = bin_data(me[np.newaxis, :], BIN_SIZE)[0]
        ...
        bin_edges = np.percentile(me_binned, percentiles)
        ...
        for t in range(n_trials):
            ...
            session_neural.append(dff_binned[:, start:end].astype(np.float32))
```

iii. The trajectory does not explicitly call out repeated work. This is inferred from the code structure.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads the full `ops.npy` object even though it only uses a few scalar entries from it, and it computes/records several metadata fields that are not needed by the downstream decoder. It also constructs trial lists through Python slicing and type conversion rather than keeping a more compact array representation, which creates extra intermediate objects that are discarded after pickling.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
...
'metadata': {
    'task_description': ...,
    'time_bin_size': time_bin_ms,
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION),
    'imaging_rate_hz': FS,
    'bin_size_frames': BIN_SIZE,
    'effective_rate_hz': FS / BIN_SIZE,
    'trial_duration_s': TRIAL_DURATION,
    'n_motion_energy_bins': N_ME_BINS,
    'dff_method': ...,
}
```

iii. The trajectory does not contain an explicit justification for these extra operations. This is a code-level assessment of work that is not needed for the core decoder interface.
