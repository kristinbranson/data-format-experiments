# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script discovers subjects by listing every top-level directory under `data/`, then discovers sessions by listing every subdirectory under each subject directory. For each session it loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. Trials are not loaded directly from disk; they are created later by splitting processed continuous recordings.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset is already organized as subject folders with per-session subdirectories containing suite2p output and movement data. The AI treated all top-level directories as subjects because in the provided data layout the only directories are mouse folders.

## 1-b. How are the data split into subjects (mice)?

i. Each top-level directory in `data/` is treated as one subject, and the list is sorted alphabetically. The script later derives `subject_list` from the ordered session list and records per-session subject indices from that list.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
...
subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))
...
subject_idx.append(subject_list.index(subj))
```

iii. The AI's notes say there are 6 subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and imply each folder is one mouse. Its rationale is based on the observed directory structure rather than an explicit `jm*` filter.

## 1-c. How are the data split into sessions?

i. Each subdirectory under a subject directory is treated as one session and sorted alphabetically. The script flattens all `(subject, session)` pairs into a single ordered session list.

ii. 
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. `CONVERSION_NOTES.md` describes each daily recording subdirectory as one session and reports 41 total sessions. The sorting appears to be for deterministic ordering.

## 1-d. How are the data split into trials?

i. The AI does not use 60-second trials. It first bins neural and motion-energy data by averaging every 10 frames, then splits each session into consecutive non-overlapping 2-minute blocks. Each trial therefore contains 360 binned timepoints (`120 s * 30 Hz / 10`).

ii. 
```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
...
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
...
def split_into_trials(neural_binned, me_binned, trial_frames):
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // trial_frames
    ...
    for t in range(n_trials):
        start = t * trial_frames
        end = start + trial_frames
        neural_trials.append(neural_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. The justification in `CONVERSION_NOTES.md` is a reading of the paper's decoder analysis: "averaging in bins of 10 consecutive timestamps" and "splits were done on consecutive 2 minute blocks." The AI chose to mirror that analysis pipeline instead of the human reference's 60-second segmentation of continuous data.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies no explicit trial-quality filtering. It simply keeps all full 2-minute trials that fit into the binned session length; any trailing partial block is dropped implicitly by floor division.

ii. 
```python
def split_into_trials(neural_binned, me_binned, trial_frames):
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // trial_frames
    ...
    for t in range(n_trials):
        start = t * trial_frames
        end = start + trial_frames
```

iii. The notes say "Trials: No explicit curation; missing video frames interpolated." The AI treated end-of-session leftovers as incomplete blocks rather than failed trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` signal is derived from suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0` for each session.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The AI's notes identify these as the suite2p outputs needed for neuropil correction and baseline-corrected fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI computes a custom baseline-corrected fluorescence trace by subtracting neuropil (`F - 0.7 * Fneu`), estimating a maximin baseline with `minimum_filter1d`, `maximum_filter1d`, and `gaussian_filter1d`, subtracting that baseline (`Fc - F0`), and then averaging every 10 frames. It does not call `suite2p.extraction.dcnv.preprocess`.

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
def _maximin_baseline(Fc, win_frames, sig_frames):
    Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
    Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
    Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
    return Flow
```

```python
dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)
dff_binned = bin_data(dff, bin_size)
```

iii. The AI justified this as matching the paper's statement that "baseline corrected fluorescence traces" with default Suite2p parameters were used, and `CONVERSION_NOTES.md` says it intentionally switched from `(Fc-F0)/F0` to `Fc-F0` and from `sig_baseline * fs` to `sig_baseline = 10` frames. The trajectory shows the AI believed this custom implementation was a faithful stand-in for Suite2p preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural filtering is applied in `convert_data.py`. The AI includes every row in `F.npy` and `Fneu.npy` and assumes the provided data are already pre-filtered and track-matched.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)
```

iii. `CONVERSION_NOTES.md` explicitly says all `iscell` values are `1.0` in the provided data and that "neurons are pre-filtered." The AI therefore decided not to apply an `iscell` threshold or any other ROI/cell exclusion step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of each artificial 2-minute block, not to session start. In metadata it describes the alignment event as `"Start of 2-minute recording block"`, with `off_start = 0.0` and `off_end = 120`.

ii. 
```python
'metadata': {
    'temporal_alignment_event': 'Start of 2-minute recording block',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
    ...
}
```

```python
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
```

iii. The AI's notes justify this with the claim that the paper's decoder used consecutive 2-minute blocks, so those blocks became its implicit alignment events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 333.3 ms bins. The AI rebins both neural and motion-energy data by averaging 10 consecutive 30 Hz frames, reducing the effective rate from 30 Hz to 3 Hz.

ii. 
```python
FS = 30.0
BIN_SIZE = 10
...
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

```python
'time_bin_size': BIN_SIZE / FS * 1000
```

iii. The justification in `CONVERSION_NOTES.md` is the paper phrase "averaging in bins of 10 consecutive timestamps." The AI treated that decoder-analysis description as the required conversion output resolution.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not use a stored raw timestamp variable. It derives input time from the number of binned samples in each trial and the fixed bin duration `BIN_SIZE / FS`, effectively encoding time from the start of each 2-minute block.

ii. 
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
    session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The notes describe the input as "Elapsed time in seconds from start of 2-min trial" and justify it as the decoder input requested by the instructions, using frame rate and bin size instead of any raw clock variable.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The computation is simple synthetic time-axis generation after rebinning. For each trial, the AI creates `0, 1, 2, ...` in units of binned steps and multiplies by `10/30` seconds, so every trial resets to `0.0` rather than continuing from session start.

ii. 
```python
n_timepoints = trial.shape[1]
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The justification in `CONVERSION_NOTES.md` is that this should represent "Time elapsed from start of 2-minute trial block (seconds)." There is no separate trajectory evidence of a more detailed timing computation.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI generates one time vector per trial with exactly the same number of binned timepoints as the binned neural trial. Alignment is therefore by construction on the rebinned 2-minute block axis, not by absolute session frame index.

ii. 
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
    session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The justification is implicit: because `input` is created from each trial's `trial.shape[1]`, it shares the same length and binning as the neural data. The notes frame this as time from start of the 2-minute trial block.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives output motion energy from `move_deve/motion_energy_glob.npy` and uses `move_deve/interframe_int.npy` to repair length mismatches caused by missing video frames.

ii. 
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. `CONVERSION_NOTES.md` says motion energy is already computed in `motion_energy_glob.npy` and that missing video frames must be handled using interframe intervals.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy to neural frame count with `np.interp`, averages it in 10-frame bins, and then later discretizes the binned values across all sessions. It does not standardize each session's motion-energy trace by its standard deviation before discretization.

ii. 
```python
def interpolate_missing_frames(motion_energy, interframe_int, n_neural_frames):
    ...
    median_ifi = np.median(interframe_int)
    ratios = interframe_int / median_ifi
    missed_counts = np.round(ratios).astype(int) - 1
    ...
    me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
    return me_interp
```

```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
me_binned = bin_data(me_interp, bin_size)
```

iii. The notes justify interpolation as repairing missing video frames and justify 10-frame averaging by the paper's decoder analysis. The AI does not give a justification for omitting the reference solution's per-session standard-deviation normalization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After binning, the AI pools all motion-energy values from all trials of all sessions, computes quintile boundaries with `np.percentile`, nudges duplicate edges upward by `1e-10`, and uses `np.digitize` to assign category labels `0` through `4`.

ii. 
```python
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)

for i in range(1, len(bin_edges)):
    if bin_edges[i] <= bin_edges[i-1]:
        bin_edges[i] = bin_edges[i-1] + 1e-10
```

```python
binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
session_disc.append(binned.reshape(1, -1).astype(np.int64))
```

iii. The AI's notes justify this as "global quintile bins across all sessions" to produce balanced decoder classes.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by first interpolating the motion-energy trace to the same frame count as the neural recording, then applying the same 10-frame binning and the same fixed 2-minute trial boundaries used for neural data.

ii. 
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
```

iii. `CONVERSION_NOTES.md` says neural and video were treated as synchronized at 30 Hz, with missing video frames interpolated so the two streams could be compared on the same bins and trial blocks.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI explicitly handles missing video frames by reconstructing missing positions from `interframe_int.npy` and interpolating motion energy onto the neural frame grid. If motion energy is longer than neural data it truncates the tail. It also drops any incomplete trailing bins or partial 2-minute trial at the end of a session by floor division.

ii. 
```python
if n_me == n_neural_frames:
    return motion_energy.astype(np.float64)

n_missing = n_neural_frames - n_me
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)
...
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
```

```python
n_bins = neural_binned.shape[1]
n_trials = n_bins // trial_frames
```

iii. The notes justify this as handling "missing video frames" while keeping neural and behavioral streams aligned. The trajectory also shows the AI considered this a necessary cleanup step for occasional dropped video frames.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies neural preprocessing, especially dF/F computation plus binning, as the dominant runtime cost. The notes estimate roughly `~1.5s` per session and `~60s` for the full dataset.

ii. 
```python
dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)
me_interp = interpolate_missing_frames(me, interframe, n_frames)
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
```

iii. `CONVERSION_NOTES.md` explicitly reports runtime estimates for "dF/F + binning" and describes an earlier dF/F bug fix as dramatically affecting runtime.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code relies on Python loops in several places that could be vectorized: building `me_positions` in missing-frame interpolation, splitting sessions into trials, creating per-trial time inputs, and discretizing motion energy trial-by-trial. The most obvious avoidable loop is the cumulative-position construction in `interpolate_missing_frames`.

ii. 
```python
me_positions = np.zeros(n_me, dtype=int)
me_positions[0] = 0
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

iii. There is no explicit justification in the notes for leaving these loops unvectorized. The decision is implicit in the implementation: the AI preferred straightforward Python loops over more compact vectorized array reshaping/indexing.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats a few lightweight passes over data structures: it linearly searches `subject_list.index(subj)` for every session, walks every trial again to build time inputs after trials were already constructed, and traverses all output trials again to print the class distribution. These are not the dominant costs, but they are repeated processing steps.

ii. 
```python
subject_idx.append(subject_list.index(subj))
```

```python
for session_trials in neural_all:
    session_inputs = []
    for trial in session_trials:
        n_timepoints = trial.shape[1]
        time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
        session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

```python
all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
```

iii. The AI did not document these as deliberate tradeoffs. They appear to come from writing the pipeline in separate, easy-to-read stages rather than optimizing passes over already constructed arrays.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes some optional or unused work that does not matter for downstream decoding: optional plotting for the first two sessions, computing `bin_edges` for printing while only saving labels in `output_values`, calculating `total_neurons` without using it, and importing `sys` and `uniform_filter1d` without using them. These do not change the saved dataset but add nonessential work or clutter.

ii. 
```python
import sys
from scipy.ndimage import uniform_filter1d
```

```python
output_all, bin_edges, bin_labels = discretize_motion_energy(me_all_raw, N_OUTPUT_BINS)
...
total_neurons = sum(neural_all[s][0].shape[0] for s in range(len(neural_all)))
```

```python
if show_processing and i < 2:
    _plot_processing(...)
```

iii. The only explicit justification is for plotting, which the AI treated as a visual sanity check. The rest appear incidental rather than deliberate downstream requirements.
