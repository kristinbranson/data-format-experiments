# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers all subjects by listing top-level directories under `DATA_DIR`, discovers all sessions by listing subdirectories under each subject, then loads each session from Suite2p and movement files. For each session it reads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`, processes them, then splits the processed continuous recording into trials.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

all_sessions = []
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

```python
def load_session(subject_dir, session_name):
    session_dir = os.path.join(subject_dir, session_name)
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))

    return F, Fneu, me, interframe
```

iii. The notes say the dataset has 6 subject folders, each with session folders containing `suite2p` and `move_deve`, and the agent states that the reference notebook directly loads `F.npy` while movement data comes from `motion_energy_glob.npy`. The trajectory shows it intentionally used directory traversal to cover the full dataset.

## 1-b. How are the data split into subjects?

i. Subjects are defined by top-level folders in `DATA_DIR`, sorted alphabetically, and the final `subject_list` is the ordered unique list of subject IDs appearing in `all_sessions`.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

```python
subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))
subject_idx.append(subject_list.index(subj))
```

iii. In `CONVERSION_NOTES.md`, the agent documents the six mice (`jm031` ... `jm046`) and cites the dataset README that each top-level folder is one subject. It chose sorted folder names to preserve deterministic ordering.

## 1-c. How are the data split into sessions?

i. Sessions are defined by each subject's dated subdirectories, sorted within subject. Each `(subject, session)` pair becomes one session in the output lists.

ii. ```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. The notes say each subject folder contains daily recording-day folders like `2023-10-18_a`, and the agent treated each such folder as one session because that matches the data README and paper's day-by-day recordings.

## 1-d. How are the data split into trials?

i. The agent treats each session as one continuous recording, bins it first, then cuts the binned recording into fixed 2-minute consecutive blocks. Trial count is computed with floor division, so only complete blocks are kept.

ii. ```python
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
```

```python
def split_into_trials(neural_binned, me_binned, trial_frames):
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // trial_frames

    neural_trials = []
    me_trials = []

    for t in range(n_trials):
        start = t * trial_frames
        end = start + trial_frames
        neural_trials.append(neural_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. The notes cite the paper statement that splits were done on consecutive 2-minute blocks and combine that with the 10-frame binning to get 360 bins per trial. The agent explicitly documents 20-minute sessions as 10 trials and 30-minute sessions as 15 trials.

## 1-e. How are trials filtered based on quality controls?

i. The script applies no explicit per-trial quality-control filter. It keeps every complete 2-minute block after preprocessing, and only implicitly drops any leftover incomplete block at the session end via floor division.

ii. ```python
n_trials = n_bins // trial_frames
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The notes state "No explicit curation; missing video frames interpolated" and "End-of-session partial bins discarded." The agent justified this by saying the paper/code did not define an additional trial-quality rejection rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw Suite2p fluorescence and neuropil traces: `F.npy` and `Fneu.npy`.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0,
                prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu
```

iii. The notes say the reference notebook loads `F.npy` and the paper says to use baseline-corrected fluorescence traces as dF/F with Suite2p defaults, so the agent chose `F` plus `Fneu` rather than `spks.npy`.

## 2-b. How is the `neural` data processed?

i. The agent computes a Suite2p-style baseline-corrected fluorescence signal by neuropil-correcting `F` with `Fneu`, estimating a maximin baseline with min-filter, max-filter, and Gaussian smoothing, subtracting that baseline, then averaging into 10-frame bins.

ii. ```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0,
                prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu
    win_frames = int(win_baseline * fs)
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
    dff = Fc - F0
    return dff
```

```python
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
```

```python
dff_binned = bin_data(dff, bin_size)
```

iii. The notes and trajectory say the agent initially used the wrong formula and then fixed it after checking Suite2p semantics: `sig_baseline` should stay in frames and the paper's "baseline corrected fluorescence" means subtraction, not division.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply any new neural QC filter. It assumes the provided Suite2p matrices are already filtered to tracked cells present across all days and does not re-check `iscell.npy` in the conversion path.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The notes explicitly say the data are "ALREADY track2p output in suite2p format" and "All iscell values are 1.0 in the provided data (pre-filtered by track2p)." The agent therefore decided no extra cell filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data to the start of each 2-minute recording block. After converting the continuous session to binned time series, it slices consecutive 360-bin chunks beginning at block onset.

ii. ```python
'temporal_alignment_event': 'Start of 2-minute recording block',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

```python
start = t * trial_frames
end = start + trial_frames
neural_trials.append(neural_binned[:, start:end].astype(np.float32))
```

iii. The notes say trial segmentation follows the paper's "consecutive 2 minute blocks" language. The agent therefore treated block start as the alignment event because the dataset has spontaneous behavior rather than discrete task events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 imaging frames per bin at 30 Hz, so the final bin size is 10/30 s = 0.333... s = 333.3 ms. The script explicitly rebins both neural and behavioral streams by averaging consecutive frames.

ii. ```python
FS = 30.0
BIN_SIZE = 10
```

```python
def bin_data(data, bin_size):
    ...
    return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
```

```python
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The notes quote the paper's "averaging in bins of 10 consecutive timestamps" and derive an effective 3 Hz sampling rate. That is the agent's stated reason for rebinning to 333.3 ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a raw timestamp variable. The agent synthesizes it from the number of binned timepoints in each trial plus the constants `BIN_SIZE` and `FS`, effectively creating time from trial start rather than true time from experiment start.

ii. ```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
    session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The notes say "Time elapsed in seconds from start of 2-min trial" and give the range `[0, 119.7s]`. The agent justified this as the decoder input construction, but it did not identify a raw experiment-time variable.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For every trial, the agent creates a 1D ramp `0, 0.333..., 0.666..., ...` using the binned frame duration, reshapes it to `(1, n_timepoints)`, and stores it as float32. It resets this ramp for every trial.

ii. ```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The notes describe this as "Elapsed time in seconds from start of 2-min trial." The justification is purely structural: it matches the per-trial decoder input format and the binned neural length.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The synthetic time vector is aligned by assigning one time value per neural bin within each trial. Each trial gets the same number of time bins as its neural matrix, but the time axis restarts at zero for every trial.

ii. ```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. The notes say the time input was verified against `np.arange * bin_size/fs`. The agent's justification was that identical bin counts guarantee one-to-one alignment with the neural matrices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `output` comes from `move_deve/motion_energy_glob.npy`, with `interframe_int.npy` used to detect dropped video frames before alignment.

ii. ```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes say the processed behavioral signal is already stored in `motion_energy_glob.npy`, and the dataset README says missing frame indices can be recovered from `interframe_int.npy` and interpolated.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent interpolates missing video frames to match the neural frame count, averages motion energy in 10-frame bins, splits the continuous binned stream into 2-minute trials, and later converts the continuous values into category indices.

ii. ```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
me_binned = bin_data(me_interp, bin_size)
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
```

```python
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
```

iii. The notes describe missing-frame interpolation and 10-frame averaging as the main behavioral preprocessing steps. The trajectory shows the agent explicitly inspected `interframe_int.npy` gaps to justify interpolation.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The agent pools all binned motion-energy values from all sessions and trials, computes five equal-percentile bins globally, fixes non-increasing bin edges by adding tiny epsilons, and assigns each time bin to a class 0-4 with `np.digitize`.

ii. ```python
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
```

```python
for i in range(1, len(bin_edges)):
    if bin_edges[i] <= bin_edges[i-1]:
        bin_edges[i] = bin_edges[i-1] + 1e-10
```

```python
binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
session_disc.append(binned.reshape(1, -1).astype(np.int64))
```

iii. The notes say "Global quintile bins across all sessions" and report the expected balanced 20% distribution. The agent chose a global thresholding scheme to match the instruction to discretize motion energy into five equal-percentile bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by first matching the neural frame count through interpolation, then applying the same 10-frame binning and the same 2-minute trial boundaries used for neural data.

ii. ```python
n_neurons, n_frames = F.shape
me_interp = interpolate_missing_frames(me, interframe, n_frames)
```

```python
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
```

iii. The notes state "Neural and video are synchronized at 30 Hz; missing frames interpolated" and describe visual alignment checks on trial 1. The agent used shared binning and shared trial slicing as its alignment strategy.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling path is for missing behavior frames. If motion-energy samples are fewer than neural frames, the script infers dropped-frame locations from `interframe_int.npy` and fills them by linear interpolation; if motion-energy samples are longer than neural frames, it truncates them. Duplicate percentile edges are also repaired by epsilon adjustments.

ii. ```python
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)
```

```python
ratios = interframe_int / median_ifi
missed_counts = np.round(ratios).astype(int) - 1
...
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
```

```python
if bin_edges[i] <= bin_edges[i-1]:
    bin_edges[i] = bin_edges[i-1] + 1e-10
```

iii. The notes say missing camera frames should be interpolated and that this was checked against frame-count mismatches. The agent also documents fixing a dF/F bug during development, but that fix was to its own code rather than to the source data.

## 6-a. What are the most time-consuming steps of the code?

i. The agent identifies dF/F computation as the dominant runtime, especially baseline estimation over long fluorescence traces. Session-by-session loading and full-dataset looping are secondary, while plotting is optional and limited to the first two sessions.

ii. ```python
dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)
```

```python
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
```

iii. In the notes, the runtime table attributes about 1.5 s per session mainly to dF/F plus binning, and the trajectory says fixing the baseline parameters reduced total runtime from about 1420 s to about 60 s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several explicit Python loops could be vectorized or reduced: the loop building `all_sessions`, the loop over sessions, the loop that constructs `me_positions` for missing-frame interpolation, the trial-splitting loop, the nested loops for motion-energy discretization, and the nested loops that build time inputs.

ii. ```python
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
```

```python
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

```python
for session_trials in all_me_trials:
    session_disc = []
    for me in session_trials:
        binned = np.digitize(me, bin_edges[1:-1])
```

iii. The notes do not focus on optimization, but the code structure makes these hotspots clear. The agent accepted the loops because the full run was fast enough after the dF/F fix.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly computes the same per-trial synthetic time vector shape for every session, repeatedly converts continuous session data into trial lists in both neural and motion-energy branches, and repeatedly scans nested trial lists when summarizing or discretizing outputs.

ii. ```python
for session_trials in neural_all:
    session_inputs = []
    for trial in session_trials:
        n_timepoints = trial.shape[1]
        time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

```python
all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
```

```python
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
```

iii. The notes emphasize correctness checks rather than deduplication. The repeated work was left in place because it kept the code straightforward and did not block the full conversion runtime.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes optional plotting of intermediate processing states, computes and prints summary statistics and output distributions used only for inspection, stores extra metadata fields not needed by the decoder, and builds percentile bin edges that are not kept in the saved output except indirectly via labels.

ii. ```python
if show_processing and i < 2:
    _plot_processing(...)
```

```python
print(f"\n=== Conversion Summary ===")
...
print(f"Output distribution: {dict(zip(unique, counts/counts.sum()))}")
```

```python
'metadata': {
    'imaging_rate_hz': FS,
    'bin_size_frames': BIN_SIZE,
    'trial_duration_s': TRIAL_DURATION_SEC,
    'neucoeff': NEUCOEFF,
    'baseline_method': 'maximin',
```

iii. The notes describe processing plots and sanity checks as validation aids. They support human review but are not required by the downstream decoder once `converted_data.pkl` has been written.
