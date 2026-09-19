# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six mice in `MICE`, iterates over each mouse, enumerates dated session folders with `get_sessions`, then loads calcium data from `suite2p/plane0/F.npy` and `Fneu.npy` plus behavior from `move_deve/motion_energy_glob.npy` and `interframe_int.npy`. Trials are not loaded directly from disk; sessions are later split into fixed-size blocks in memory.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions

for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    F = np.load(os.path.join(suite2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
    ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. In the trajectory, the agent first listed `/app/data`, found six mouse folders (`jm031` to `jm046`), inspected session contents, and then encoded that dataset structure directly into `MICE`. It justified the load path by noting Suite2p stores neural traces in `F.npy`/`Fneu.npy` and motion data in `move_deve`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the hard-coded mouse IDs in `MICE`. Each entry in `MICE` is treated as one mouse, and `subject_idx` stores the mouse index for each session.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
subjects = MICE[:]
subject_idx = []

for mouse_i, mouse in enumerate(MICE):
    ...
    subject_idx.append(mouse_i)
```

iii. The trajectory shows the agent enumerated the six subject directories and then fixed them in code rather than rediscovering them dynamically each run.

## 1-c. How are the data split into sessions?

i. Within each mouse directory, sessions are the sorted subdirectories whose names begin with `'2'`, i.e. date-stamped recording folders such as `2023-10-18_a`. Non-session files such as `ground_truth.csv` are excluded.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. In the trajectory, the agent listed subject folders and noticed that real sessions are the date-like directories, while some subjects also contain non-directory files such as `ground_truth.csv`. That led to the `'2'` prefix filter.

## 1-d. How are the data split into trials?

i. The agent defines trials as consecutive non-overlapping 2-minute blocks after temporal binning. At 30 Hz with 10-frame bins, each trial has `360` time bins.

ii.
```python
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 360 bins per trial

def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. The trajectory explicitly says the agent believed “the paper splits data into consecutive 2-minute blocks,” and it used that to choose `TRIAL_DURATION_SEC = 120`.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. The code only keeps complete fixed-length trial blocks by integer division; any trailing partial block is silently discarded.

ii.
```python
def split_into_trials(data, trial_length):
    if data.ndim == 1:
        n_trials = len(data) // trial_length
        return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
    else:
        n_trials = data.shape[1] // trial_length
        return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. The trajectory does not describe any trial QC rule beyond fixed-length splitting. The code reflects that: it never checks trial quality metrics.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
```

iii. The trajectory states that Suite2p stores raw fluorescence in `F.npy` and neuropil fluorescence in `Fneu.npy`, so those files were used to reconstruct the neural signal.

## 2-b. How is the `neural` data processed?

i. The agent computes a dF/F-like signal itself. It first applies neuropil subtraction (`F - 0.7 * Fneu`), then estimates a baseline with Gaussian smoothing followed by `minimum_filter1d` and `maximum_filter1d` over a 60 s window, then divides by baseline: `(Fc - F0) / F0`.

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
```

iii. The trajectory shows two stages of justification. First, the agent wanted Suite2p-style baseline correction. Later, after a slow `percentile_filter` attempt, it switched to the “maximin” formulation and explicitly justified that as both faster and closer to Suite2p defaults.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filter is applied in the final script. The code never reads `iscell.npy` and keeps every row of `F.npy` / `Fneu.npy`.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. In the trajectory and later notes, the agent said extra filtering was unnecessary because the upstream pipeline had already curated cells. That rationale is not implemented as an explicit QC step; the effect is simply to retain all neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the start of the recording session, not to a stimulus or behavior event. After binning, the code slices consecutive trial windows from the session-long trace.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)

'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The trajectory treats this dataset as continuous spontaneous behavior without a discrete trial event, so it chose session start as the alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code rebins both neural and behavioral streams by averaging over 10 original frames at 30 Hz, producing 3 Hz data with a bin size of about 333.33 ms.

ii.
```python
BIN_SIZE = 10
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000

dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The trajectory cites the paper’s statement that dF/F and behavior were averaged in bins of 10 consecutive timestamps, and explicitly translates that to about 333 ms per bin.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from a raw file. The time input is synthesized from the trial index, the bin index within the trial, the frame rate, and the bin size.

ii.
```python
start_sec = t_i * TRIAL_DURATION_SEC
time_bins = np.linspace(
    start_sec + BIN_DURATION_MS / 2000,
    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
    TRIAL_BINS
)
```

iii. The trajectory says “time elapsed from the start of the experiment” should be computed from the constant 30 Hz sampling and 10-frame binning rather than loaded from timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the agent computes a 1D vector of evenly spaced bin-center times in seconds, reshapes it to `(1, n_timepoints)`, and casts it to `float32`.

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

iii. The trajectory gives only a brief justification: it wanted time elapsed from session start, with one value per binned neural sample. The bin-center convention is implied by the code rather than defended explicitly.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned one-to-one with the binned neural data. Each trial gets exactly `TRIAL_BINS` time values, matching the binned neural trial length.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
...
input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The trajectory treats time as a derived companion signal for each neural bin, so it is generated using the same trial count and bin count as the neural arrays.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, and `interframe_int.npy` is additionally used to infer dropped video frames during alignment.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The trajectory shows the agent inspected both files, concluded `motion_energy_glob.npy` held the behavioral signal, and used `interframe_int.npy` to repair frame-count mismatches.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code does not recompute motion energy from video. It takes the precomputed motion-energy trace, aligns it to neural frames by inserting missing-frame gaps and interpolating them, then averages the aligned trace into 10-frame bins.

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

me_aligned = align_motion_energy(me_raw, ifi, n_frames)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. In the trajectory, the agent rejected `tstamps.npy` as unusable, concluded that large interframe gaps mark dropped video frames, and justified interpolation as the way to realign behavior with neural data before binning.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent thresholds motion energy with global percentile edges computed across all trials from all sessions, then applies those same edges everywhere to produce five bins.

ii.
```python
all_me_values = []
...
for mt in me_trials:
    all_me_values.append(mt)

all_me_concat = np.concatenate(all_me_values)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)

binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The trajectory explicitly defends this choice after validation: the agent notes that sample sessions have skewed class balance because “bins were computed globally,” and it treats globally balanced 20% bins as desirable.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to the neural data by constructing a neural-frame-length array, advancing one step per observed video frame, adding extra skipped positions when `interframe_int` indicates dropped frames, and linearly interpolating the resulting gaps.

ii.
```python
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
```

iii. The trajectory says the agent found the raw timestamps confusing, then decided a relative-gap rule (`> 1.5x` the median interval) was enough to detect dropped frames and align motion energy frame-for-frame with neural data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The only explicit missing-data handling is for motion-energy frame drops: missing positions are left as `NaN` and filled by linear interpolation. Partial trailing trials are dropped implicitly by fixed-length trial splitting. The code does not add an explicit post-check that the repaired motion-energy trace matches the neural length.

ii.
```python
aligned = np.full(n_neural_frames, np.nan)
...
if np.any(nans) and not np.all(nans):
    x = np.arange(n_neural_frames)
    aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])

def split_into_trials(data, trial_length):
    n_trials = len(data) // trial_length
    return [data[i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. The trajectory explicitly discusses repairing dropped video frames. It does not mention any stronger validation step; instead it assumes the interpolation-based alignment is sufficient.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is neural preprocessing in `compute_dff`, especially baseline estimation over long recordings. Session-wise loading and per-session binning are secondary costs.

ii.
```python
def compute_dff(F, Fneu):
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE_SEC * FS)
    Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    F0 = maximum_filter1d(Flow, size=win, axis=1)
    ...
```

iii. The trajectory explicitly identifies baseline computation as the bottleneck, first for the abandoned `percentile_filter` version and then for the final Suite2p-like baseline path.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidate is the per-frame loop in `align_motion_energy`, which fills the aligned array one frame at a time. Additional list-building loops over trials and motion-energy segments could also be collapsed or preallocated.

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

for mt in me_trials:
    all_me_values.append(mt)
```

iii. The trajectory does not explicitly frame this as a vectorization problem, but it repeatedly discusses runtime costs and optimization, especially around preprocessing and frame handling.

## 6-c. What processing does the code repeat multiple times?

i. The script repeats some work unnecessarily. `sample_only=True` still preprocesses all sessions to compute global motion-energy bin edges, `get_sessions(...)` is called again while constructing metadata, and neural trials are cast to `float32` in a separate full-data pass after assembly.

ii.
```python
for mouse_i, mouse in enumerate(MICE):
    ...
    for sess_name in sessions:
        ...

'session_info': {
    mouse: {
        'sessions': get_sessions(os.path.join(data_dir, mouse)),
        ...
    }
    for mouse in MICE
}

for i in range(len(data['neural'])):
    data['neural'][i] = [t.astype(np.float32) for t in data['neural'][i]]
```

iii. The trajectory explicitly notices one repeated-work problem: in sample-only mode the script still processes the full dataset because global percentile edges are computed from all sessions.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does substantial bookkeeping that is not used by the downstream decoder: global summary statistics, neuron/session consistency checks, verbose printouts, construction of `sample_data`, and intermediate storage of raw motion-energy trials in `output_all_raw` only to discretize them later. It also stores extra metadata such as `session_info` and global `motion_energy_bin_edges` that the decoder does not need.

ii.
```python
output_all_raw = []  # store raw ME values before discretization
all_me_values = []   # collect all binned ME for global percentile computation
...
nan_count = sum(np.any(np.isnan(t)) for s in neural_all for t in s)
inf_count = sum(np.any(np.isinf(t)) for s in neural_all for t in s)
...
sample_data = {
    'neural': data['neural'][:n_sample],
    ...
}
```

iii. The trajectory mentions redundant sample conversion and global processing for sample mode. The rest of the unnecessary bookkeeping is evident from the final code rather than explicitly justified in the trajectory.
