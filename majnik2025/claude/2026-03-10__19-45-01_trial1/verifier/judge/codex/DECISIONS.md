# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script discovers subjects by listing all directories directly under `DATA_DIR`, then discovers sessions by listing all directories under each subject directory. For each session it loads fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, plus motion energy and video timing from `move_deve/motion_energy_glob.npy` and `interframe_int.npy`. Trials are not loaded directly from disk; they are created later by splitting each processed continuous session into fixed-length blocks.

ii. 
```python
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
```

iii. The justification in `CONVERSION_NOTES.md` is that each subject has session directories containing suite2p output and movement data, so the script can enumerate directory structure directly. The notes also state that motion-energy timing information is needed because some sessions have missing video frames.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined as every top-level directory inside `DATA_DIR`, sorted alphabetically. The output `subjects` list is then derived from the ordered session list, preserving first occurrence order.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
```

```python
subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))  # unique, ordered
subject_idx.append(subject_list.index(subj))
```

iii. The notes justify this by treating each top-level folder as one mouse and report six such subjects (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`). The notes do not mention any extra subject-filtering rule beyond directory structure.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories within each subject directory, sorted alphabetically. Each session corresponds to one continuous daily recording.

ii. 
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. `CONVERSION_NOTES.md` treats the per-subject session folders as the natural organization of the dataset and reports 41 total sessions across the six mice.

## 1-d. How are the data split into trials?

i. The AI decided there is no native trial structure and split each continuous session into consecutive non-overlapping 2-minute trials after temporal binning. It uses 10-frame bins at 30 Hz, so each trial has `120 * 30 / 10 = 360` time bins. Any remainder shorter than a full trial is implicitly discarded.

ii. 
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
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

iii. The justification in `CONVERSION_NOTES.md` is explicit: the AI read the paper as using "consecutive 2 minute blocks" and therefore chose 2-minute trial segmentation. The README and final trajectory summary repeat this rationale.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-quality filtering. Trials are kept if they arise from the fixed segmentation. Partial tails at the end of a session are dropped because `n_trials` is computed with floor division.

ii. 
```python
n_bins = neural_binned.shape[1]
n_trials = n_bins // trial_frames
```

```python
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. `CONVERSION_NOTES.md` says "Trials: No explicit curation; missing video frames interpolated." No additional trial exclusion criteria were documented in the notes or trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `plane0`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The notes state that neural activity comes from suite2p output and that the relevant preprocessing parameters (`fs`, `neucoeff`, baseline settings) were read from the suite2p metadata.

## 2-b. How is the `neural` data processed?

i. The AI computes a custom approximation to Suite2p-style baseline-corrected fluorescence. It first performs neuropil subtraction (`F - 0.7 * Fneu`), then estimates a baseline with a hand-written maximin pipeline using `minimum_filter1d`, `maximum_filter1d`, and `gaussian_filter1d`, then subtracts that baseline (`Fc - F0`). Afterward it averages the result into non-overlapping 10-frame bins.

ii. 
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0,
                prctile_baseline=8.0):
    Fc = F - neucoeff * Fneu
    win_frames = int(win_baseline * fs)
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
    dff = Fc - F0
    return dff
```

```python
from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
```

```python
dff_binned = bin_data(dff, bin_size)  # (n_neurons, n_bins)
```

iii. The AI justified this in the notes and trajectory as matching Suite2p defaults: neuropil correction with coefficient 0.7, maximin baseline, and baseline subtraction rather than division. The trajectory explicitly records that it fixed an earlier bug from `(Fc-F0)/F0` to `Fc-F0` and from `sig_baseline * fs` to `sig_baseline = 10` frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied in `convert_data.py`. All rows of `F.npy`/`Fneu.npy` are propagated into the converted dataset.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
```

iii. The notes justify this by claiming the supplied data is already pre-filtered by Track2p/Suite2p and that all `iscell` values are 1.0, so no extra `iscell` or match-matrix filtering is needed in the conversion script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. The AI aligns neural data to the start of each artificial 2-minute recording block rather than to session start. The metadata describes the alignment event as `"Start of 2-minute recording block"`, with `off_start = 0.0` and `off_end = 120`.

ii. 
```python
'metadata': {
    'temporal_alignment_event': 'Start of 2-minute recording block',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
```

```python
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
```

iii. The notes justify this through the same 2-minute block interpretation used for trialization. The README and trajectory summary both present trial starts as the relevant alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame non-overlapping averages at 30 Hz, giving `10 / 30 = 0.333... s` bins, i.e. 333.3 ms. The same rebinning is applied to neural and motion-energy streams before trial splitting.

ii. 
```python
BIN_SIZE = 10  # number of frames per bin
FS = 30.0
```

```python
def bin_data(data, bin_size):
    if data.ndim == 1:
        n_bins = n // bin_size
        return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_bins = n // bin_size
        trimmed = data[..., :n_bins * bin_size]
        new_shape = trimmed.shape[:-1] + (n_bins, bin_size)
        return trimmed.reshape(new_shape).mean(axis=-1)
```

iii. The notes cite the paper statement about "bins of 10 consecutive timestamps" and repeatedly justify 333.3 ms bins as matching the paper's decoding preprocessing.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not read from any raw timestamp variable. It is synthesized from bin indices using `np.arange(n_timepoints) * (BIN_SIZE / FS)` for each trial.

ii. 
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
    session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as "Time from start" based on constant frame rate and fixed bin width. However, the notes explicitly describe it as elapsed time from the start of each 2-minute trial, not from a raw file.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as the left-edge time of each binned sample within a trial. It is re-created independently for every trial, producing a sequence from `0` to about `119.67` seconds for each 2-minute block, then reshaped to `(1, n_timepoints)` and cast to `float32`.

ii. 
```python
n_timepoints = trial.shape[1]
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. `CONVERSION_NOTES.md` explicitly justifies this choice as "Elapsed time in seconds from start of 2-min trial" and the README repeats "Time elapsed from start of 2-minute trial block."

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by creating one time vector per neural trial after neural trialization. Each time vector has exactly the same number of bins as the corresponding neural trial and is attached in the same session/trial structure.

ii. 
```python
for session_trials in neural_all:
    session_inputs = []
    for trial in session_trials:
        n_timepoints = trial.shape[1]
        time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
        session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
    input_all.append(session_inputs)
```

iii. No deeper justification was given beyond the choice to define trial-relative elapsed time. The notes only report that the resulting arrays matched the expected `np.arange * bin_size/fs` pattern in sanity checks.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, and the timing of video frames is taken from `move_deve/interframe_int.npy` so that missing video frames can be handled.

ii. 
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes justify use of `interframe_int.npy` by stating that some sessions have missing video frames and those timing intervals are needed to align motion energy to the neural data.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI first interpolates motion energy to the neural frame count when needed, using a median inter-frame interval to infer how many frames were missed and `np.interp` to resample onto neural-frame positions. It then averages motion energy into 10-frame bins. Discretization is done later.

ii. 
```python
def interpolate_missing_frames(motion_energy, interframe_int, n_neural_frames):
    n_me = len(motion_energy)
    if n_me == n_neural_frames:
        return motion_energy.astype(np.float64)

    n_missing = n_neural_frames - n_me
    if n_missing < 0:
        return motion_energy[:n_neural_frames].astype(np.float64)

    median_ifi = np.median(interframe_int)
    ratios = interframe_int / median_ifi
    missed_counts = np.round(ratios).astype(int) - 1
```

```python
    me_positions = np.zeros(n_me, dtype=int)
    me_positions[0] = 0
    for i in range(1, n_me):
        me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]

    neural_positions = np.arange(n_neural_frames)
    me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
```

```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
me_binned = bin_data(me_interp, bin_size)  # (n_bins,)
```

iii. The notes justify this as necessary because some sessions have missing video frames and the behavioral signal must match neural length before joint binning. The trajectory and README both describe interpolation as the fix for dropped video frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into five global equal-percentile bins computed across all sessions and all trials combined, not separately per session. Duplicate percentile edges are forced to be strictly increasing by adding a tiny epsilon.

ii. 
```python
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
binned = np.digitize(me, bin_edges[1:-1])  # 0 to n_bins-1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as "Global quintile bins across all sessions," mainly to obtain an even overall class balance. The notes also mention sanity-checking that the final output distribution was about 20% per bin globally.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to the neural data by interpolating or truncating the motion-energy stream to the neural frame count, then binning both streams with the same bin size and splitting them with the same trial boundaries.

ii. 
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
```

```python
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The notes justify this by asserting that neural and video streams are synchronized at 30 Hz apart from dropped camera frames, so restoring equal length before common binning and trialization is sufficient.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are handled by interpolation using `interframe_int.npy`. If motion energy is longer than neural data, it is truncated. If the session length is not divisible by the bin size or by the trial length, trailing partial bins or partial trials are silently dropped through integer division and array trimming. Duplicate percentile edges are nudged upward by `1e-10`.

ii. 
```python
if n_me == n_neural_frames:
    return motion_energy.astype(np.float64)

n_missing = n_neural_frames - n_me
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)
```

```python
return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
...
trimmed = data[..., :n_bins * bin_size]
```

```python
n_trials = n_bins // trial_frames
```

```python
if bin_edges[i] <= bin_edges[i-1]:
    bin_edges[i] = bin_edges[i-1] + 1e-10
```

iii. The explicit justification in the notes focuses on missing video frames: the AI believed interpolation was needed to restore alignment. No explicit justification was given for silent truncation or epsilon-adjusted percentile edges; those appear to be implementation choices for robustness.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is neural preprocessing, especially the custom maximin baseline estimation in `compute_dff`, which applies large 1D filters over every neuron's full time series. The notes estimate about 1.5 s per session for "dF/F + binning" and about 60 s total for all 41 sessions. Optional plotting in `--show-processing` mode would add extra overhead, but it is not part of the default path.

ii. 
```python
def compute_dff(...):
    Fc = F - neucoeff * Fneu
    win_frames = int(win_baseline * fs)
    F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
    dff = Fc - F0
```

```python
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
```

iii. `CONVERSION_NOTES.md` explicitly reports runtime dominated by "dF/F + binning" and documents that fixing the baseline code reduced full runtime from about 1420 s to about 60 s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the loop that builds `me_positions` one element at a time in `interpolate_missing_frames`; it could be replaced by a cumulative sum. Other serial loops that are structurally simple include the per-session/per-trial loops used for time-input construction and discretization.

ii. 
```python
me_positions = np.zeros(n_me, dtype=int)
me_positions[0] = 0
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
```

```python
for session_trials in neural_all:
    session_inputs = []
    for trial in session_trials:
        n_timepoints = trial.shape[1]
        time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

```python
for session_trials in all_me_trials:
    session_disc = []
    for me in session_trials:
        binned = np.digitize(me, bin_edges[1:-1])
```

iii. No explicit justification about these loops was recorded in the notes or trajectory. Their current structure appears to reflect implementation simplicity rather than an argued efficiency choice.

## 6-c. What processing does the code repeat multiple times?

i. The code makes multiple full passes over the dataset: one pass to load and preprocess sessions, another to discretize all motion-energy trials, another to synthesize time inputs, and another to concatenate all outputs just to print global distribution statistics. It also repeatedly computes `subject_list.index(subj)` inside the session loop.

ii. 
```python
for i, (subj, sess) in enumerate(all_sessions):
    ...
    dff_binned, me_binned = process_session(F, Fneu, me, interframe)
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
    neural_all.append(neural_trials)
    me_all_raw.append(me_trials)
    subject_idx.append(subject_list.index(subj))
```

```python
output_all, bin_edges, bin_labels = discretize_motion_energy(me_all_raw, N_OUTPUT_BINS)
```

```python
for session_trials in neural_all:
    ...
    input_all.append(session_inputs)
```

```python
all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
```

iii. No explicit justification for these repeated passes was documented. The notes focus on correctness and decoder performance rather than memory or pass-count optimization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only used for diagnostics or metadata and not for the saved dataset: `bin_edges` is returned but not saved, output-distribution statistics are computed only for logging, plotting infrastructure is imported and defined even in default runs, and optional processing plots are generated only for manual inspection. The script also imports `sys` and `uniform_filter1d`, which are unused.

ii. 
```python
output_all, bin_edges, bin_labels = discretize_motion_energy(me_all_raw, N_OUTPUT_BINS)
...
all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
unique, counts = np.unique(all_outputs, return_counts=True)
print(f"Output distribution: {dict(zip(unique, counts/counts.sum()))}")
```

```python
import sys
from scipy.ndimage import uniform_filter1d
...
if show_processing and i < 2:
    _plot_processing(...)
```

iii. The notes justify the diagnostics as sanity checks and visual validation, but they do not argue that these steps are needed for the converted dataset itself. For the unused imports and discarded `bin_edges`, no explicit justification was provided.
