# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all top-level subject directories under `/app/data`, then loads session directories matching `*_a` inside each subject. For each session it reads suite2p calcium files (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) and later reads motion-energy files from `move_deve` (`motion_energy_glob.npy`, and `tstamps.npy` if dropped frames need repair). Trials are not loaded directly from disk; they are created after the full session is loaded and preprocessed.

ii. 
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))

for subj_i, subj in enumerate(subjects):
    session_dirs = sorted(glob.glob(os.path.join(DATA_DIR, subj, '*_a')))
    for day_i, session_dir in enumerate(session_dirs):
        s2p = os.path.join(session_dir, 'suite2p', 'plane0')
        F = np.load(os.path.join(s2p, 'F.npy'))
        Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
        iscell = np.load(os.path.join(s2p, 'iscell.npy'))
        ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
        ...
        me = load_motion_energy(session_dir, n_frames)
```

iii. In the trajectory, the agent first inspected the README, notebook, and on-disk directory layout (steps 6, 8, 10) and then encoded that structure in the script docstring (step 36), stating that each mouse/day contains suite2p outputs plus `move_deve` motion-energy files.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the sorted top-level directories under `/app/data`. The code does not explicitly restrict them to names starting with `jm`; it relies on the dataset layout.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
```

iii. The trajectory shows the agent relied on the dataset structure discovered from the README and directory listing (steps 6 and 8), which showed six top-level mouse folders.

## 1-c. How are the data split into sessions?

i. Sessions are defined as the sorted subdirectories within each subject that match the pattern `*_a`, i.e. one daily recording folder per session.

ii.
```python
for subj_i, subj in enumerate(subjects):
    session_dirs = sorted(glob.glob(os.path.join(DATA_DIR, subj, '*_a')))
    for day_i, session_dir in enumerate(session_dirs):
        ...
```

iii. The trajectory indicates the agent inspected the subject/session folder names directly (steps 8 and 14) and then encoded the assumption that daily recording folders are the session units.

## 1-d. How are the data split into trials?

i. The dataset is treated as continuous per-session data with no native trial structure. The agent creates artificial trials by cutting each session into consecutive non-overlapping 60 s chunks. It computes the number of full trials from the raw frame count, keeps only complete 60 s spans, bins them, and then slices the binned session into 180-bin trials.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS / BIN_SECONDS))   # 180
FRAMES_PER_TRIAL = BINS_PER_TRIAL * BIN_FRAMES             # 1800

n_trials = n_frames // FRAMES_PER_TRIAL
n_keep = n_trials * FRAMES_PER_TRIAL
...
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    neural_trials.append(neural[:, sl])
    input_trials.append(t[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :])
```

iii. The trajectory shows the agent accepted the instruction-defined trialization rather than searching for a natural event structure. The script docstring written in step 36 states, “Sessions are cut into consecutive 60 s trials.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. The only trial-level curation is to drop any trailing session frames that do not make up a full 60 s trial.

ii.
```python
n_trials = n_frames // FRAMES_PER_TRIAL
n_keep = n_trials * FRAMES_PER_TRIAL

neural = bin_time(dff[:, :n_keep])
behav = bin_time(me[:n_keep])
```

iii. In the trajectory and final script docstring (steps 36 and 65), the agent states that all 6 mice and all 41 sessions are kept and that the released data already has complete neural/behavioral coverage, so it did not introduce extra trial rejection.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from suite2p fluorescence traces `F.npy` and `Fneu.npy`. The code also loads `iscell.npy` and `ops.npy`, but those are used only for validation/assertions, not to construct the neural signal itself.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
...
dff = compute_dff(F, Fneu, fs)
```

iii. The trajectory shows the agent inspected the authors’ GUI processing code (`F_processing`) in step 21 and then described the neural signal in the script docstring (step 36/65) as coming from suite2p fluorescence with neuropil subtraction and maximin baseline correction.

## 2-b. How is the `neural` data processed?

i. The agent subtracts neuropil (`F - 0.7 * Fneu`), applies a manual Suite2p-style maximin baseline computation using a Gaussian filter followed by minimum and maximum filters, averages the traces into non-overlapping 10-frame bins, and then z-scores each neuron within each session. That last z-scoring step is explicitly described by the agent as an extra deviation from the paper/reference pipeline.

ii.
```python
def compute_dff(F, Fneu, fs=FS):
    Fc = F.astype(np.float64) - NEUCOEFF * Fneu.astype(np.float64)
    Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
    win = int(WIN_BASELINE * fs)
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow

...
dff = compute_dff(F, Fneu, fs)
neural = bin_time(dff[:, :n_keep])
neural = (neural - neural.mean(axis=1, keepdims=True)) / \
         (neural.std(axis=1, keepdims=True) + 1e-9)
neural = neural.astype(np.float32)
```

iii. The trajectory shows two distinct justifications. First, after reading the authors’ code in step 21, the agent matched the maximin baseline logic. Second, in steps 44, 46, 48, 54, and 65, the agent experimentally compared variants and concluded that per-neuron z-scoring improved decoder validation accuracy, so it added z-scoring even though the docstring says it is “the only step that is not in the paper.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The agent assumes Track2p output already contains only accepted cells and verifies that assumption by asserting `iscell[:, 0] == 1` for every ROI.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
...
assert np.all(iscell[:, 0] == 1)
```

iii. The trajectory includes an explicit data check in step 19 and the script docstring in steps 36/65 states that all released ROIs are already Track2p-tracked and Suite2p-classifier-accepted, so no further neuron selection is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session, specifically the first imaging frame. The session is then partitioned into consecutive non-overlapping 60 s trials with no additional event-based alignment.

ii.
```python
t = (np.arange(n_bins) + 0.5) * BIN_SECONDS
...
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    neural_trials.append(neural[:, sl])

...
'temporal_alignment_event': (
    'start of the recording session (first imaging frame); each session is cut into '
    'consecutive non-overlapping 60 s trials'),
'off_start': 0.0,
'off_end': TRIAL_SECONDS,
```

iii. The script docstring and metadata written in step 36 make this explicit: the dataset is continuous, so the only alignment event used is session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses non-overlapping 10-frame bins at 30 Hz, so the temporal resolution is 1/3 s = 333.33 ms per bin. Yes, temporal rebinning is applied by averaging each block of 10 consecutive frames for both neural and behavioral streams.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
BIN_SECONDS = BIN_FRAMES / FS

def bin_time(x, bin_frames=BIN_FRAMES):
    x = np.asarray(x)
    n = (x.shape[-1] // bin_frames) * bin_frames
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // bin_frames, bin_frames).mean(axis=-1)

...
'time_bin_size': 1000.0 * BIN_SECONDS,
```

iii. In the trajectory, the agent cites the paper’s decoding procedure in the script docstring (step 36): both streams are “denoised by averaging in bins of 10 consecutive frames.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a dedicated raw timestamp variable. It is synthesized from the imaging frame rate and the binned sample index after trial truncation and rebinning.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
BIN_SECONDS = BIN_FRAMES / FS
...
t = (np.arange(n_bins) + 0.5) * BIN_SECONDS
```

iii. The trajectory does not show a separate justification for this beyond the overall session-start alignment decision. The script uses the known constant frame rate from `ops['fs']` and assumes a uniform time grid.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After dropping incomplete trailing data and binning the session, the agent creates one time value per binned sample. It uses the center of each 10-frame bin, not the left edge, by adding `0.5` before multiplying by the bin duration.

ii.
```python
n_bins = behav.shape[0]
...
# time elapsed since the start of the session, at the centre of each bin
t = (np.arange(n_bins) + 0.5) * BIN_SECONDS
```

iii. The trajectory contains no separate written defense of the midpoint convention. The only explicit rationale is the inline code comment added in step 36 that calls these values “the centre of each bin.”

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned by building it on the exact same binned session grid as the neural data and then slicing trials with the same `sl` indices used for neural and output arrays.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    neural_trials.append(neural[:, sl])
    input_trials.append(t[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :])
```

iii. The trajectory does not contain a separate discussion of this alignment choice; it is implicit in the shared trial slicing used throughout the assembly loop written in step 36.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`. When the behavioral video length does not match the imaging length, the agent also uses `move_deve/tstamps.npy` to reconstruct dropped-frame positions before interpolation.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
...
ts = np.load(os.path.join(session_dir, 'move_deve', 'tstamps.npy'))
```

iii. The trajectory shows the agent inspected the `move_deve` files directly (steps 8, 14, 16, 31) and then justified timestamp use in the `load_motion_energy` docstring written in step 36: camera timestamps identify dropped frames more exactly than relying only on array length.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent first replaces the first motion-energy sample with the second sample to remove the initial artificial zero, then repairs dropped frames by reconstructing the camera-frame index from timestamps and linearly interpolating missing values. After that it averages motion energy into 10-frame bins and discretizes the binned values into 5 within-session percentile bins.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy')).astype(np.float64)
me[0] = me[1]
...
dts = np.diff(ts)
steps = np.round(dts / np.median(dts)).astype(int)
idx = np.concatenate([[0], np.cumsum(steps)])
...
full = np.full(n_frames, np.nan)
full[idx] = me
missing = np.isnan(full)
full[missing] = np.interp(np.flatnonzero(missing), idx, me)
...
behav = bin_time(me[:n_keep])
thresholds = np.percentile(behav, np.arange(1, N_QUANTILES) * (100 / N_QUANTILES))
labels = np.searchsorted(thresholds, behav, side='left').astype(np.int64)
```

iii. The trajectory shows the agent investigated dropped-frame structure numerically (steps 14, 16, 31) and then encoded the justification in the step-36 docstring: the camera is hardware-triggered to the microscope, so timestamps can be used to reconstruct exact missing-frame positions and interpolate them.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The binned motion-energy trace is converted into 5 categories by computing the 20th, 40th, 60th, and 80th percentiles within each session and using those thresholds to assign each time bin to a session-local quintile label.

ii.
```python
N_QUANTILES = 5
...
thresholds = np.percentile(behav, np.arange(1, N_QUANTILES) * (100 / N_QUANTILES))
labels = np.searchsorted(thresholds, behav, side='left').astype(np.int64)
...
'output_values': [[f'quintile_{i + 1}' for i in range(N_QUANTILES)]],
```

iii. The trajectory shows the agent followed the instruction that output should be discretized into five equal-percentile bins and explicitly summarized this choice in the script docstring and final report (steps 36 and 65).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent assumes camera frames and imaging frames are synchronized one-for-one because the behavior camera is microscope-triggered. If frames are missing from the motion-energy trace, it reconstructs their positions from `tstamps.npy`, linearly interpolates the missing values onto the imaging-frame grid, then bins and trial-slices the repaired motion-energy trace with the same temporal structure as the neural data.

ii.
```python
me = load_motion_energy(session_dir, n_frames)
...
if len(me) == n_frames:
    return me
...
idx = np.concatenate([[0], np.cumsum(steps)])
assert idx[-1] == n_frames - 1, (session_dir, idx[-1], n_frames)
...
full[idx] = me
full[missing] = np.interp(np.flatnonzero(missing), idx, me)
...
behav = bin_time(me[:n_keep])
...
output_trials.append(labels[sl][None, :])
```

iii. The trajectory supports this directly: the agent inspected motion-energy lengths and timestamps (steps 14, 16, 31), then wrote the explicit alignment rationale into the `load_motion_energy` docstring and metadata in step 36.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several small defects explicitly: it replaces the first motion-energy sample if it is the known edge artifact, reconstructs dropped camera frames using timestamps and interpolates them, asserts consistency of the reconstructed frame grid, and silently drops any trailing data that do not complete a full 60 s trial.

ii.
```python
me[0] = me[1]
...
assert idx[-1] == n_frames - 1, (session_dir, idx[-1], n_frames)
assert len(np.unique(idx)) == len(idx)
...
full[missing] = np.interp(np.flatnonzero(missing), idx, me)
...
n_trials = n_frames // FRAMES_PER_TRIAL
n_keep = n_trials * FRAMES_PER_TRIAL
```

iii. The trajectory shows the agent explicitly investigated dropped-frame cases (steps 14, 16, 31) before writing the timestamp-based repair logic in step 36. There is no separate discussion of the first-sample replacement beyond the code comment that calls it an edge artifact.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive computation is the per-session neural preprocessing: the maximin-style baseline calculation over the full neuron-by-time fluorescence matrix (`gaussian_filter`, `minimum_filter1d`, `maximum_filter1d`), followed by session-wide binning and z-scoring. The repeated disk loads of large `.npy` arrays are also significant but secondary.

ii.
```python
def compute_dff(F, Fneu, fs=FS):
    Fc = F.astype(np.float64) - NEUCOEFF * Fneu.astype(np.float64)
    Flow = gaussian_filter(Fc, [0.0, SIG_BASELINE])
    win = int(WIN_BASELINE * fs)
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow

...
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
...
neural = (neural - neural.mean(axis=1, keepdims=True)) / \
         (neural.std(axis=1, keepdims=True) + 1e-9)
```

iii. The trajectory does not include an explicit profiling discussion. This assessment follows from the operations the agent chose in the final code and from its focus on `F_processing`/baseline logic in step 21.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Most low-level time-series processing is already vectorized. The remaining loop that could be reduced is the per-trial append loop, which could be replaced with a reshape/split-based assembly step. The outer subject/session loops are structurally necessary for per-session loading and metadata.

ii.
```python
neural_trials, input_trials, output_trials = [], [], []
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    neural_trials.append(neural[:, sl])
    input_trials.append(t[sl][None, :].astype(np.float32))
    output_trials.append(labels[sl][None, :])
```

iii. The trajectory does not show the agent discussing vectorization explicitly. Relative to the intermediate data-inspection work, the final motion-energy repair path is already vectorized via `np.interp` rather than repeated insertion.

## 6-c. What processing does the code repeat multiple times?

i. There is no major avoidable repeated processing beyond the intended per-session pipeline. For every session, the code reloads the same kinds of files, reruns the same preprocessing steps, recomputes per-session motion-energy thresholds, and rebuilds per-session trial lists.

ii.
```python
for subj_i, subj in enumerate(subjects):
    session_dirs = sorted(glob.glob(os.path.join(DATA_DIR, subj, '*_a')))
    for day_i, session_dir in enumerate(session_dirs):
        ...
        dff = compute_dff(F, Fneu, fs)
        me = load_motion_energy(session_dir, n_frames)
        ...
        thresholds = np.percentile(behav, np.arange(1, N_QUANTILES) * (100 / N_QUANTILES))
        ...
        for tr in range(n_trials):
            ...
```

iii. The trajectory does not contain an explicit runtime justification here. The repetition is mostly a direct consequence of processing each session independently.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main extra work that downstream decoder training does not use is loading `iscell.npy` and `ops.npy` only to assert assumptions, and constructing rich `session_info` metadata including per-session thresholds. Those checks/metadata are not consumed by the decoder itself.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
fs = float(ops['fs'])
assert fs == FS
assert np.all(iscell[:, 0] == 1)
...
session_info.append({
    'subject': subj,
    'date': os.path.basename(session_dir).replace('_a', ''),
    'day_index': day_i,
    'n_neurons': int(n_neurons),
    'n_frames': int(n_frames),
    'n_trials': int(n_trials),
    'frame_rate_hz': fs,
    'motion_energy_quintile_thresholds': thresholds.tolist(),
})
```

iii. The trajectory suggests these were added for validation and documentation rather than decoding. Steps 19 and 36/65 show the agent actively checking `iscell`/`ops` assumptions and reporting rich session metadata.
