# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories in the data directory (`/app/data`). Sessions are subdirectories within each subject folder. Neural data is loaded from suite2p output files (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) in `suite2p/plane0/`. Motion energy is loaded from `motion_energy_glob.npy` and interframe intervals from `interframe_int.npy` in the `move_deve/` subdirectory. All subjects and sessions are enumerated and processed in sorted order.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

def list_sessions(subject_dir):
    return sorted([f.path for f in os.scandir(subject_dir) if f.is_dir()])

# Loading neural data:
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))

# Loading motion energy:
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(session_dir, 'move_deve', 'interframe_int.npy')).astype(np.float64)
```

iii. From the trajectory (steps 2-3), the agent explored the directory structure and identified the standard convention: subject folders contain session subfolders, each with suite2p output and motion energy files. The agent examined methods.txt, the data README, and the load_data.ipynb notebook to understand the data organization. The agent also loaded ops.npy to read processing parameters, following the track2p code's approach.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all directories in the data directory, sorted alphabetically. This yields 6 subjects: jm031, jm032, jm038, jm039, jm040, jm046.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

iii. The agent identified the naming convention from exploring the data directory (trajectory steps 2-3). The `os.path.isdir` check ensures non-directory entries (like `README.md`, `load_data.ipynb`) are excluded.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording. Non-directory entries (e.g., `ground_truth.csv`) are skipped by the `is_dir()` check.

ii.
```python
def list_sessions(subject_dir):
    return sorted([f.path for f in os.scandir(subject_dir) if f.is_dir()])
```

iii. The agent discovered the session structure from the data README and directory listing (trajectory step 3). The date-based naming convention (YYYY-MM-DD_a) ensures chronological sorting.

## 1-d. How are the data split into trials?

i. There is no natural trial structure in this dataset. Trials are artificially defined as 60-second non-overlapping segments of the continuous recording. After 10-frame binning (30 Hz -> 3 Hz), each trial is 180 bins (`int(round(60.0 * 30 / 10)) = 180`). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))  # 180
ntrials = nbins // bins_per_trial
# ...
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

iii. The agent followed the task instructions specifying "Split sessions into 60-second trials" (trajectory step 16). Sessions of 20 minutes yield 20 trials; sessions of 30 minutes yield 30 trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are included.

ii. N/A

iii. No filtering criteria are mentioned in the paper for trial exclusion, and the task instructions do not specify any trial filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p output files: `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), `ops.npy` (processing parameters), and `iscell.npy` (cell classification), all from `plane0`.

ii.
```python
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
```

iii. The agent identified these as the standard suite2p output files (trajectory step 6). The paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The agent also loaded ops.npy to read the actual processing parameters used, rather than hardcoding them.

## 2-b. How is the `neural` data processed?

i. The AI manually reimplements the `F_processing` function from `track2p/gui/data_management.py` using scipy. Steps: (1) neuropil subtraction with coefficient read from ops (0.7), (2) Gaussian smoothing with sig_baseline from ops (10.0), (3) minimum filter with window = win_baseline * fs (60 * 30 = 1800 frames), (4) maximum filter with same window, (5) subtract this baseline from the neuropil-corrected trace.

ii.
```python
def compute_dff(F, Fneu, ops):
    neucoeff = float(ops.get('neucoeff', 0.7))
    sig_baseline = float(ops.get('sig_baseline', 10.0))
    win_baseline = float(ops.get('win_baseline', 60.0))
    fs = float(ops['fs'])
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow
```

iii. The agent examined the `F_processing` function in `track2p/gui/data_management.py` (trajectory step 9) and replicated the exact same maximin baseline correction algorithm. Parameters are read from the session's `ops.npy` file rather than hardcoded, following the track2p GUI code pattern where ops parameters are passed to `F_processing`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the suite2p cell classifier: only cells where `iscell[:, 0] > 0` are kept. The agent notes that all tracked neurons pass this check (all have iscell=1), since Track2p only provides tracked cells that already passed curation.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
# only tracked cells are provided, and all pass the suite2p classifier
keep = iscell[:, 0] > 0
F, Fneu = F[keep], Fneu[keep]
```

iii. The agent surveyed all sessions (trajectory step 8) and confirmed that all tracked neurons have iscell=1. The filtering step was included for robustness, matching the paper's statement: "We considered all ROIs above the default threshold of 0.5 as true cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. Metadata records `temporal_alignment_event` as a description of the trial structure, with `off_start=0.0` and `off_end=60.0`.

ii.
```python
'metadata': {
    'temporal_alignment_event': (
        'start of each 60 s block of the continuous recording (trials are '
        'consecutive 60 s segments starting at the first imaging frame of the session)'),
    'off_start': 0.0,
    'off_end': TRIAL_SEC,
}
```

iii. There is no stimulus event to align to. The recording is continuous spontaneous activity, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both the neural and the motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing from 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before the motion energy is discretized.

ii.
```python
BIN_FRAMES = 10
# ...
def bin_average(x, binsize):
    x = np.asarray(x)
    n = (x.shape[-1] // binsize) * binsize
    x = x[..., :n]
    new_shape = x.shape[:-1] + (n // binsize, binsize)
    return x.reshape(new_shape).mean(axis=-1)
# ...
dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)
# ...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,  # ms = 333.33
```

iii. The paper's methods section states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent identified this in trajectory step 5.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index, BIN_FRAMES, and the frame rate (fs from ops). Values represent the center of each time bin in seconds from session start.

ii.
```python
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
# then per trial:
input_s.append(t_b[sl][None, :].astype(np.float32))
```

iii. Since the frame rate is constant at 30 Hz and there are no stored timestamps with the neural data, computing time from bin indices is straightforward and equivalent to using timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as the center of each time bin: `(bin_index * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs`. This gives the midpoint of each 10-frame bin in seconds. For a 20-minute session, values range from 0.15 to 1199.85 s. The time runs continuously across trials within a session (not reset per trial).

ii.
```python
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
```

iii. Using bin centers rather than left edges is a more accurate representation of when the data was acquired during each bin.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time variable is computed from the same bin indices used to slice the neural data, so alignment is inherent. Both neural data and time are sliced using the same `sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)`.

ii.
```python
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
    input_s.append(t_b[sl][None, :].astype(np.float32))
```

iii. Since both are derived from the same binned time grid, no separate alignment step is needed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Interframe intervals from `interframe_int.npy` are used to detect and recover dropped camera frames.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(session_dir, 'move_deve', 'interframe_int.npy')).astype(np.float64)
```

iii. The agent identified these files from the data directory listing and the README (trajectory steps 2-3). The motion energy file contains pre-computed global motion energy from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four processing steps: (1) the first motion energy sample is corrected by setting me[0] = me[1] since it is 0 by construction (no preceding frame for differencing); (2) dropped camera frames are recovered by computing actual frame indices from interframe intervals using `np.round(ifi / median)` and interpolated using `np.interp`; (3) the trace is averaged into 10-frame bins; (4) the binned signal is discretized into 5 equal-percentile bins with edges computed within each session.

ii.
```python
# Correct first sample:
me[0] = me[1]

# Recover dropped frames:
med = np.median(ifi)
steps = np.round(ifi / med).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
me = np.interp(np.arange(nframes), idx, me)

# Bin:
me_b = bin_average(me, BIN_FRAMES)

# Discretize:
edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The agent investigated dropped frames by examining sessions with mismatched lengths (trajectory step 13), finding that gaps in interframe intervals at approximately 2x the median interval correspond to missed camera frames. The cumulative sum of rounded frame steps reconstructs the actual frame grid, allowing proper linear interpolation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session. The 20th, 40th, 60th, and 80th percentile boundaries are computed from each session's binned motion energy. `np.digitize` maps values to 5 categories (0-4).

ii.
```python
edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The task specifies "discretized into five equal-percentile bins, selected per session." The agent implemented this with per-session percentile computation. The output values are labeled `['q1_lowest', 'q2', 'q3', 'q4', 'q5_highest']`.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz (camera triggered by microscope). Dropped camera frames make the motion energy shorter. The AI recovers the actual frame positions using interframe intervals: `np.round(ifi / median_ifi)` gives integer frame steps, whose cumulative sum gives the frame indices of each acquired camera frame on the full neural timeline. `np.interp` then interpolates onto the full frame grid. After alignment, both streams are binned together and sliced with the same indices.

ii.
```python
me = load_motion_energy(session_dir, nframes)
# Inside load_motion_energy:
med = np.median(ifi)
steps = np.round(ifi / med).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
assert len(idx) == len(me)
assert idx[-1] == nframes - 1
return np.interp(np.arange(nframes), idx, me)
```

iii. The agent investigated the interframe interval data (trajectory step 13) and confirmed that gaps at ~2x the median interval account for missing frames. The frame-grid reconstruction and linear interpolation produce a motion energy trace with the exact same length as the neural data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three types of data issues are handled: (1) The first motion energy sample is always 0 by construction (no preceding frame for differencing), so it is replaced with the second sample; (2) Dropped video frames are detected and interpolated via the interframe interval approach (see 4-b/4-d), with assertions verifying correct alignment; (3) Remainder frames at the end of a session that don't fill a complete 60-second trial are discarded.

ii.
```python
# First sample correction:
me[0] = me[1]

# Alignment assertion:
assert idx[-1] == nframes - 1, (session_dir, idx[-1], nframes)

# Remainder discarding:
ntrials = nbins // bins_per_trial  # only complete trials
```

iii. The assertion ensures any frame count mismatch is caught rather than producing silently misaligned data.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the scipy-based baseline correction in `compute_dff`, which applies `gaussian_filter`, `minimum_filter1d`, and `maximum_filter1d` over the full session length for every neuron. For sessions with 685 neurons and 54000 frames, this involves substantial computation.

ii.
```python
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. Each filtering operation processes all neurons across the full session timeline. The minimum and maximum filters use a window of 1800 frames (60s * 30Hz), which is a large sliding window.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial segmentation loop iterates over trials, creating slice views and appending to lists. This could theoretically be replaced with array_split or reshape operations, but the loop is simple and the overhead is negligible. There are no performance-critical loops that would benefit significantly from vectorization.

ii.
```python
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
```

iii. The main computation (filtering, binning) is already vectorized via numpy/scipy operations.

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any processing unnecessarily. Each session is processed once in a single pass: load, compute dF/F, align motion energy, bin, discretize, segment into trials. There is no redundant recomputation.

ii. N/A

iii. The single-pass design processes each session's data exactly once.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and processes `iscell.npy` for filtering, even though all tracked neurons have iscell=1, making the filter a no-op. Additionally, the `session_info` list stored in metadata includes detailed per-session information that is not used by the downstream decoder but serves as documentation.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
keep = iscell[:, 0] > 0  # always True for all neurons in this dataset
```

iii. While the iscell filtering is technically unnecessary for this specific dataset, it adds robustness and follows the paper's methodology.
