# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code scans `/app/data` for subject directories, then scans each subject directory for session subdirectories. For each session it loads suite2p fluorescence files (`ops.npy`, `F.npy`, `Fneu.npy`, `iscell.npy`) from `suite2p/plane0` and motion-energy files (`motion_energy_glob.npy`, `interframe_int.npy`) from `move_deve`. Trials are not loaded from disk directly; they are created later by splitting each processed session into fixed 60 s segments.

ii. 
```python
def list_sessions(subject_dir):
    return sorted([f.path for f in os.scandir(subject_dir) if f.is_dir()])

def main():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d))])
    ...
    for si, subject in enumerate(subjects):
        for session_dir in list_sessions(os.path.join(DATA_DIR, subject)):
            s2p = os.path.join(session_dir, 'suite2p', 'plane0')
            ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
            F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
            Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
            iscell = np.load(os.path.join(s2p, 'iscell.npy'))
            ...
            me = load_motion_energy(session_dir, nframes)
```

iii. In the trajectory, the agent said it found 6 mice with multiple sessions and that each session contained suite2p outputs plus `move_deve` files. It then decided to process all sessions by iterating over subject directories and session subdirectories and loading the fluorescence, ops, and motion-energy arrays.

## 1-b. How are the data split into subjects?

i. Subjects are defined as all directories directly under `/app/data`, sorted alphabetically. The code does not explicitly restrict subjects to names beginning with `jm`; it relies on the fact that the only directories present are subject folders.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

iii. In the trajectory, the agent reported that the dataset contained 6 mice and listed only `jm*` subject folders as directories. Its justification was effectively empirical: the directory layout already separated mice, so all directories in `/app/data` could be treated as subjects.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories inside each subject directory, sorted alphabetically.

ii. 
```python
def list_sessions(subject_dir):
    return sorted([f.path for f in os.scandir(subject_dir) if f.is_dir()])

for si, subject in enumerate(subjects):
    for session_dir in list_sessions(os.path.join(DATA_DIR, subject)):
        ...
```

iii. In the trajectory, the agent inspected the dataset layout and concluded that each daily recording lived in its own subdirectory under each mouse, so all such subdirectories should be included as sessions.

## 1-d. How are the data split into trials?

i. The AI treats the recordings as continuous sessions with no native trial structure, then splits each session into consecutive, non-overlapping 60 s trials after temporal binning. The number of bins per trial is `round(60 * fs / 10)`, which is 180 bins at 30 Hz with 10-frame averaging. Any remainder bins at the end of the session are dropped implicitly because `ntrials` is computed with floor division.

ii. 
```python
BIN_FRAMES = 10
TRIAL_SEC = 60.0
...
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
ntrials = nbins // bins_per_trial
neural_s, input_s, output_s = [], [], []
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
    input_s.append(t_b[sl][None, :].astype(np.float32))
    output_s.append(me_cat[sl][None, :].astype(np.int64))
```

iii. In the trajectory, the agent explicitly said there was no natural trial structure and that the task required 60 s trials, so it chose consecutive 60 s blocks of the continuous recording.

## 1-e. How are trials filtered based on quality controls?

i. The AI code does not apply any explicit trial-quality filter. It keeps every full 60 s trial and only excludes incomplete trailing data by flooring `ntrials`.

ii. 
```python
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))
ntrials = nbins // bins_per_trial
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    ...
```

iii. The trajectory does not describe any trial-level quality-control rule beyond using complete 60 s segments. The agent focused its QC logic on motion-energy frame recovery rather than trial rejection.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from the suite2p fluorescence arrays `F.npy` and `Fneu.npy`, with processing parameters read from `ops.npy`. The code also reads `iscell.npy` to decide which ROIs to keep.

ii. 
```python
s2p = os.path.join(session_dir, 'suite2p', 'plane0')
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
```

iii. In the trajectory, the agent inspected the suite2p outputs and concluded that `F.npy` and `Fneu.npy` were the appropriate raw neural traces, with the ops dictionary providing the imaging rate and baseline parameters used in the paper/repository pipeline.

## 2-b. How is the `neural` data processed?

i. The code computes baseline-corrected fluorescence by subtracting neuropil (`F - neucoeff * Fneu`) and then applying a maximin-style baseline estimate using a Gaussian smoothing step followed by minimum and maximum filters. After that, it denoises the trace by averaging non-overlapping 10-frame bins.

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

dff = compute_dff(F, Fneu, ops)
dff_b = bin_average(dff, BIN_FRAMES)
```

iii. In the trajectory, the agent justified this as matching the paper and repository: it identified the Track2p GUI's `F_processing` logic, noted `neucoeff=0.7`, `baseline='maximin'`, `sig_baseline=10`, `win_baseline=60 s`, and said this was the paper-consistent definition of dF/F for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies an `iscell[:, 0] > 0` mask before processing, so only ROIs passing suite2p's cell classifier are retained. In practice, the trajectory states that all Track2p-provided cells already satisfy this criterion, so this is intended as a consistency check rather than an additional exclusion step.

ii. 
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
# only tracked cells are provided, and all pass the suite2p classifier
keep = iscell[:, 0] > 0
F, Fneu = F[keep], Fneu[keep]
```

iii. In the trajectory and top-of-file docstring, the agent said the Track2p dataset already contained only tracked neurons and that all had `iscell==1`, but it still kept the `iscell` mask because the paper described keeping ROIs above the default 0.5 threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not use a stimulus or behavioral event. Instead, it treats the start of each 60 s block as the trial alignment point and describes trials as consecutive 60 s segments beginning at the first imaging frame of the session.

ii. 
```python
'metadata': {
    ...
    'temporal_alignment_event': (
        'start of each 60 s block of the continuous recording (trials are '
        'consecutive 60 s segments starting at the first imaging frame of the session)'),
    'off_start': 0.0,
    'off_end': TRIAL_SEC,
    ...
}
```

iii. In the trajectory, the agent repeatedly justified this by saying the data were continuous spontaneous recordings with no natural event structure, so the required "trials" should be created as consecutive 60 s blocks starting from session onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame non-overlapping averages at a 30 Hz acquisition rate, giving a 3 Hz signal with 333.33 ms bins. The same rebinning is applied to both neural activity and motion energy.

ii. 
```python
BIN_FRAMES = 10
...
dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)
...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. In the trajectory, the agent cited the methods text stating that decoding analyses slightly denoised traces by averaging bins of 10 consecutive frames and concluded that both neural and behavioral signals should be rebinned together.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not loaded from an explicit timestamp array. It is derived from the binned frame index, `BIN_FRAMES`, and the frame rate `fs` from `ops.npy`.

ii. 
```python
fs = float(ops['fs'])
...
nbins = dff_b.shape[1]
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
```

iii. In the trajectory, the agent justified this by saying the recording rate was fixed at 30 Hz and that the decoder input should be "time elapsed from the beginning of the session", so explicit timestamps were unnecessary for the decoder input.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code computes time after temporal binning and uses the center of each 10-frame bin, not the left edge. It builds a full session-length binned time vector in seconds and then slices it per trial.

ii. 
```python
nbins = dff_b.shape[1]
# time (s) at the centre of each bin, from the beginning of the session
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
...
input_s.append(t_b[sl][None, :].astype(np.float32))
```

iii. In the trajectory, the agent described the input simply as time from session start on the binned grid. The "bin centre" choice is stated in code comments rather than explicitly defended in the trajectory.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is generated on the same binned session grid as the rebinned neural trace, then identical trial slices are applied to both, so each time sample corresponds one-to-one with one neural time bin.

ii. 
```python
dff_b = bin_average(dff, BIN_FRAMES)
...
t_b = (np.arange(nbins) * BIN_FRAMES + (BIN_FRAMES - 1) / 2.0) / fs
...
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
input_s.append(t_b[sl][None, :].astype(np.float32))
```

iii. In the trajectory, the agent said the decoder input should be time from session start and that both streams should be binned together before trializing, which is its justification for this alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, and `move_deve/interframe_int.npy` is used to reconstruct dropped video frames on the microscope frame grid.

ii. 
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(session_dir, 'move_deve', 'interframe_int.npy')).astype(np.float64)
```

iii. In the trajectory, the agent inspected `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`, concluded that motion energy was the target behavioral signal, and used inter-frame intervals to recover missing camera samples.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first replaces the first motion-energy sample with the second because the first value is structurally zero. If the motion-energy array is shorter than the neural recording, it reconstructs the observed camera-frame indices from the median inter-frame interval and linearly interpolates motion energy back onto the full neural frame grid. It then averages the repaired signal into 10-frame bins and discretizes the binned motion energy into 5 per-session percentile bins.

ii. 
```python
def load_motion_energy(session_dir, nframes):
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
    ifi = np.load(os.path.join(session_dir, 'move_deve', 'interframe_int.npy')).astype(np.float64)
    # the first sample is 0 by construction (no preceding frame to difference with)
    me[0] = me[1]
    if len(me) == nframes:
        return me
    med = np.median(ifi)
    steps = np.round(ifi / med).astype(int)
    idx = np.concatenate([[0], np.cumsum(steps)])
    assert len(idx) == len(me)
    assert idx[-1] == nframes - 1, (session_dir, idx[-1], nframes)
    return np.interp(np.arange(nframes), idx, me)

me_b = bin_average(me, BIN_FRAMES)
edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. In the trajectory, the agent explicitly reasoned that gaps in `interframe_int.npy` were integer multiples of the normal frame interval, so motion-energy samples could be mapped back to the 2-photon frame grid and interpolated. It also cited the methods text for 10-frame averaging before decoding and chose per-session quintiles to satisfy the decoder task.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The binned motion-energy trace for each session is discretized into 5 categories using per-session equal-percentile thresholds. The code computes the 20th, 40th, 60th, and 80th percentile cutoffs and then uses `np.digitize` to assign category labels `0` through `4`.

ii. 
```python
edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. In the trajectory, the agent justified this directly from the task statement: output motion energy had to be discretized into five equal-percentile bins within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code aligns motion energy to neural data by reconstructing where each observed camera frame falls on the full neural frame grid and interpolating across any dropped camera frames, yielding one motion-energy sample per imaging frame. After that, neural and motion-energy traces are rebinned with the same 10-frame averaging and cut into trials using the same slices.

ii. 
```python
def load_motion_energy(session_dir, nframes):
    ...
    med = np.median(ifi)
    steps = np.round(ifi / med).astype(int)
    idx = np.concatenate([[0], np.cumsum(steps)])
    assert idx[-1] == nframes - 1, (session_dir, idx[-1], nframes)
    return np.interp(np.arange(nframes), idx, me)

dff_b = bin_average(dff, BIN_FRAMES)
me_b = bin_average(me, BIN_FRAMES)
...
sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
output_s.append(me_cat[sl][None, :].astype(np.int64))
```

iii. In the trajectory, the agent justified this by showing that the inter-frame gaps indicated dropped camera frames, and that after reconstructing the frame indices the motion-energy trace could be placed on the same frame grid as the calcium recording before shared binning and trial slicing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing camera frames by reconstructing their positions from `interframe_int.npy` and linearly interpolating motion energy across the missing points. It also replaces the first motion-energy sample, which is structurally zero, with the second sample. Incomplete trailing bins are dropped in `bin_average`, and incomplete trailing trial segments are dropped because `ntrials` uses floor division.

ii. 
```python
me[0] = me[1]
...
steps = np.round(ifi / med).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
assert idx[-1] == nframes - 1, (session_dir, idx[-1], nframes)
return np.interp(np.arange(nframes), idx, me)

def bin_average(x, binsize):
    x = np.asarray(x)
    n = (x.shape[-1] // binsize) * binsize
    x = x[..., :n]
    ...

ntrials = nbins // bins_per_trial
```

iii. In the trajectory, the agent explicitly investigated sessions with missing motion-energy frames, concluded that the gaps were recoverable from the inter-frame intervals, and described this repair as necessary to keep behavior aligned with neural activity.

## 6-a. What are the most time-consuming steps of the code?

i. The likely bottleneck is the per-session neural preprocessing, especially `compute_dff`, because it applies Gaussian smoothing and long-window minimum/maximum filters over the full neuron-by-frame matrix for every session. The large `.npy` loads are also substantial, but the code's dominant computation is the baseline-correction pass across all frames and neurons.

ii. 
```python
F = np.load(os.path.join(s2p, 'F.npy')).astype(np.float64)
Fneu = np.load(os.path.join(s2p, 'Fneu.npy')).astype(np.float64)
...
dff = compute_dff(F, Fneu, ops)

def compute_dff(F, Fneu, ops):
    ...
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow
```

iii. The trajectory does not explicitly discuss runtime bottlenecks, so this is inferred from the implemented operations. The agent did emphasize the dF/F preprocessing pipeline as the main substantive transformation applied session by session.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The motion-energy repair is already vectorized with `np.interp`, so the main remaining Python loop is the per-trial assembly loop that repeatedly appends slices to lists. That loop could be replaced with batched reshaping or list comprehensions over precomputed arrays, but the code does not attempt that optimization.

ii. 
```python
ntrials = nbins // bins_per_trial
neural_s, input_s, output_s = [], [], []
for tr in range(ntrials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural_s.append(np.ascontiguousarray(dff_b[:, sl], dtype=np.float32))
    input_s.append(t_b[sl][None, :].astype(np.float32))
    output_s.append(me_cat[sl][None, :].astype(np.int64))
```

iii. The trajectory does not explicitly discuss vectorization, but the code itself shows that the agent chose a vectorized interpolation strategy for motion-energy repair and left only the trial-packaging loop in Python.

## 6-c. What processing does the code repeat multiple times?

i. There is no obvious redundant repeated processing of the same session data. Each session is loaded once, neural preprocessing is run once, motion energy is repaired once, and percentile binning is computed once for that session.

ii. 
```python
for si, subject in enumerate(subjects):
    for session_dir in list_sessions(os.path.join(DATA_DIR, subject)):
        ...
        dff = compute_dff(F, Fneu, ops)
        me = load_motion_energy(session_dir, nframes)
        dff_b = bin_average(dff, BIN_FRAMES)
        me_b = bin_average(me, BIN_FRAMES)
        edges = np.percentile(me_b, np.linspace(0, 100, N_ME_BINS + 1)[1:-1])
```

iii. The trajectory does not claim any intentional repeated processing. Its plan was a single pass per session followed by trialization and saving.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is no major discarded processing, but there are two small pieces of extra work. First, the code loads `iscell.npy` and applies a keep mask even though the trajectory says all Track2p-provided neurons already satisfy `iscell==1`, so that filtering is effectively a no-op. Second, it builds detailed `session_info` metadata that is not needed by the decoder itself.

ii. 
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
keep = iscell[:, 0] > 0
F, Fneu = F[keep], Fneu[keep]
...
session_info.append({
    'subject': subject,
    'session': os.path.basename(session_dir),
    'n_neurons': int(F.shape[0]),
    'n_frames': int(nframes),
    'n_trials': int(ntrials),
    'fs': fs,
})
```

iii. In the trajectory, the agent explicitly said that all tracked cells already passed the suite2p classifier, which makes the `iscell` masking step redundant for this dataset. The extra metadata reflects the agent's preference for richer documentation rather than a necessity for downstream decoding.
