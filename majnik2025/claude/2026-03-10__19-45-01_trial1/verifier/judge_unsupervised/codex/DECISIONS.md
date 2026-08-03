# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers all subject folders under `data/`, then all session folders within each subject, and processes every `(subject, session)` pair. For each session it only loads four arrays: `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `interframe_int.npy`. Trials are not loaded directly from disk; they are created later by splitting each continuous session.

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

def load_session(subject_dir, session_name):
    session_dir = os.path.join(subject_dir, session_name)
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. In `CONVERSION_NOTES.md`, Step 5 says the mapping is `F.npy, Fneu.npy -> neural` and `motion_energy_glob.npy -> output`, with `interframe_int.npy` used for missing-frame handling. The reference README also describes exactly this folder layout and names `motion_energy_glob.npy` and `interframe_int.npy` as the relevant behavior files.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories inside `data/`. The script sorts those directory names alphabetically and uses them as the subject list.

ii. 
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))  # unique, ordered
subject_idx.append(subject_list.index(subj))
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the agent documented six subject folders (`jm031` ... `jm046`). The dataset README says each subject has its own folder and that the names are alphabetically ordered for cross-reference with the paper.

## 1-c. How are the data split into sessions?

i. Sessions are split by subdirectories within each subject directory. The session folder names are date strings such as `2023-10-18_a`, and the script sorts them before processing.

ii. 
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. In Step 2 the agent recorded the per-subject session counts. The dataset README says each subject folder contains one session folder per recording day and that the folder name is the recording date.

## 1-d. How are the data split into trials?

i. Each session is treated as one continuous recording, then binned, then split into consecutive non-overlapping 2-minute blocks. Trial length is hard-coded as `120 s * 30 Hz / 10 frames = 360` binned samples. Sessions of 20 minutes become 10 trials; sessions of 30 minutes become 15 trials.

ii. 
```python
TRIAL_DURATION_SEC = 120
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)  # 120 * 30 / 10 = 360

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

iii. In Step 3 and Step 5 of `CONVERSION_NOTES.md`, the agent says the paper described “consecutive 2 minute blocks” and explicitly chose 2-minute trials. The agent also noted the dataset contains both 20-minute and 30-minute sessions and decided to use all blocks.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filtering. The only trial-level handling is that motion-energy missing frames are interpolated before trialization, and any incomplete final block is implicitly dropped by integer division.

ii. 
```python
n_trials = n_bins // trial_frames

for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
```

iii. Step 3 says “Trials: No explicit curation; missing video frames interpolated.” Step 10 adds that end-of-session partial bins are discarded. The agent justified this by the README note that missing camera frames may be interpolated.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`. The script does not use `spks.npy`, `stat.npy`, `ops.npy`, or `iscell.npy` to construct the neural signal.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))

dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)
```

iii. In Step 5 the agent mapped `F.npy, Fneu.npy -> neural`. The loading notebook also loads `F.npy` and comments that for proper analysis the user should compute dF/F as described in the paper or alternatively use `spks.npy`; the agent chose the dF/F route.

## 2-b. How is the `neural` data processed?

i. The agent decided to compute Suite2p-style baseline-corrected fluorescence by neuropil-correcting `F` with `Fneu`, estimating a maximin baseline, subtracting that baseline, and then averaging into 10-frame bins. However, the actual baseline routine is only an approximation to Suite2p: it applies min, then max, then Gaussian smoothing, whereas the current Suite2p implementation smooths first, then min, then max.

ii. 
```python
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

```python
dff_binned = bin_data(dff, bin_size)
```

iii. Step 5 and trajectory step 94 justify the decision as “baseline subtraction (not division), matching Suite2p’s `dcnv.preprocess()`,” after fixing an earlier bug. The agent’s notes repeatedly say the goal was to match Suite2p default maximin processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies no additional neuron-level quality filter. It assumes the provided Track2p Suite2p outputs are already curated to include only tracked cells that passed the default `iscell` threshold.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

There is no code that loads `iscell.npy` or masks neurons before building `neural`.

iii. Step 1 and Step 3 of `CONVERSION_NOTES.md` say the data are already pre-filtered and that all `iscell` first-column values are `1.0`. The README says the matrices only include successfully tracked neurons across all days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural trials to the start of each 2-minute recording block. Neural traces are first processed continuously within a session, then the binned session array is cut into block-aligned trial slices.

ii. 
```python
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
```

```python
'temporal_alignment_event': 'Start of 2-minute recording block',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. Step 5 says the key trial decision was “2-minute blocks (360 bins each).” The metadata explicitly names the alignment event as the start of a 2-minute recording block.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 333.3 ms bins, produced by averaging every 10 imaging frames at 30 Hz. This is a temporal rebinning step applied to both neural and motion-energy traces.

ii. 
```python
FS = 30.0
BIN_SIZE = 10

return data[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
return trimmed.reshape(new_shape).mean(axis=-1)
```

```python
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. In Step 3 the agent cites the paper phrase “averaging in bins of 10 consecutive timestamps” and records the effective rate as 3 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from any raw timestamp variable. The script synthesizes time purely from the number of bins in each trial plus the constants `BIN_SIZE` and `FS`. It does not use `tstamps.npy` or session offsets.

ii. 
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
    session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. Step 5 says the input is “Elapsed time in seconds from start of 2-min trial,” not from raw experiment timestamps. This was a deliberate design choice, not a value read from disk.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script creates a one-dimensional ramp for every trial: `0, 0.333..., 0.666..., ...`. There is no normalization, no use of camera or imaging timestamps, and no carry-over of elapsed time across trials within a session.

ii. 
```python
n_timepoints = trial.shape[1]
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. In Step 5 the agent explicitly documents the range as `[0, 119.7s]` for each 2-minute trial and describes it as “Time elapsed from start of 2-min trial.”

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned bin-for-bin with each neural trial after trialization, but it is aligned to trial onset rather than experiment onset. Every trial restarts at zero even though the instructions asked for time from the start of the experiment.

ii. 
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

```python
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
```

iii. The notes and README both describe the input as trial-relative time. The agent’s justification is that trials are 2-minute blocks; it did not discuss preserving cumulative experiment time across blocks.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from `move_deve/motion_energy_glob.npy`. `interframe_int.npy` is used as an auxiliary variable to detect missing video frames and place interpolated values.

ii. 
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
```

iii. Step 5 maps `motion_energy_glob.npy -> output[0]`. The dataset README says `motion_energy_glob.npy` contains processed behavioral motion energy and that missing-frame indices can be recovered from `tstamps.npy` or `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent interpolates over missing video frames to match the neural frame count, averages the result into 10-frame bins, and later discretizes it. There is no explicit extra normalization step before discretization.

ii. 
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
me_binned = bin_data(me_interp, bin_size)
```

```python
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)

median_ifi = np.median(interframe_int)
ratios = interframe_int / median_ifi
missed_counts = np.round(ratios).astype(int) - 1
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
```

iii. Step 5 says “Interpolate missing frames, bin by 10, global quintile discretization.” Step 10 says the agent considered interpolation the appropriate edge-case handling because the README explicitly permits interpolation over missing camera frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized globally across all trials and sessions into five equal-percentile bins. The script computes percentile cutoffs from the concatenated continuous values, forces strictly increasing edges if needed, then uses `np.digitize` to assign categories `0` through `4`.

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
binned = np.digitize(me, bin_edges[1:-1])  # 0 to n_bins-1
binned = np.clip(binned, 0, n_bins - 1)
session_disc.append(binned.reshape(1, -1).astype(np.int64))
```

iii. Step 5 calls this decision “Global quintile bins across all sessions.” This matches the decoder instruction to discretize motion energy into five equal-percentile bins.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is first expanded to the neural frame count by interpolation if needed, then binned with the same 10-frame averaging as the neural data, then sliced into the same 2-minute trial windows.

ii. 
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
me_binned = bin_data(me_interp, bin_size)
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
```

iii. Step 10 says the agent considered neural and video to be synchronized at 30 Hz and used missing-frame interpolation so the motion-energy stream could share frame count, bins, and trial boundaries with neural activity.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion-energy frames are handled by interpolation based on `interframe_int.npy`. If behavior has more frames than neural data, the extra frames are truncated. Incomplete trailing bins or trial blocks are dropped by flooring during binning and trial splitting. The script does not contain analogous repair logic for missing neural data.

ii. 
```python
if n_me == n_neural_frames:
    return motion_energy.astype(np.float64)

n_missing = n_neural_frames - n_me
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)
```

```python
missed_counts = np.round(ratios).astype(int) - 1
me_positions = np.zeros(n_me, dtype=int)
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
```

```python
n_bins = n // bin_size
trimmed = data[..., :n_bins * bin_size]
...
n_trials = n_bins // trial_frames
```

iii. The rationale appears in Step 3, Step 5, and trajectory step 43: the agent inspected `interframe_int.npy`, concluded that enlarged intervals identify dropped video frames, and chose interpolation because the README says those frames may be treated as missing or interpolated over.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session neural preprocessing, especially `compute_dff()` and its maximin baseline estimation across all neurons and all frames. Secondary costs are reading the large `.npy` arrays and the all-data motion-energy discretization pass; plotting is also expensive when `--show-processing` is used.

ii. 
```python
for i, (subj, sess) in enumerate(all_sessions):
    F, Fneu, me, interframe = load_session(subj_dir, sess)
    dff_binned, me_binned = process_session(F, Fneu, me, interframe)
```

```python
def process_session(F, Fneu, me, interframe, bin_size=BIN_SIZE):
    dff = compute_dff(F, Fneu, fs=FS, neucoeff=NEUCOEFF)
    me_interp = interpolate_missing_frames(me, interframe, n_frames)
    dff_binned = bin_data(dff, bin_size)
```

iii. In Step 7 the agent estimated `dF/F + binning` at about 1.5 seconds per session and about 60 seconds total, which shows it identified neural preprocessing as the main runtime cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could be vectorized or hoisted: the loop that reconstructs `me_positions` in missing-frame interpolation, the per-trial loop in `split_into_trials`, the nested loops that create identical time ramps trial by trial, and the nested loops that digitize motion energy one trial at a time.

ii. 
```python
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
for session_trials in neural_all:
    session_inputs = []
    for trial in session_trials:
        n_timepoints = trial.shape[1]
        time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. The agent’s notes focus on correctness, but the code structure shows several small Python loops around array operations that could be replaced with reshapes, `cumsum`, or session-wise broadcasting.

## 6-c. What processing does the code repeat multiple times?

i. It repeatedly computes the same trial-relative time vector for every trial, repeatedly searches `subject_list.index(subj)` for every session, and makes extra full-data passes when summarizing outputs after discretization. The conversion also processes each session independently even when many operations are structurally identical across sessions.

ii. 
```python
subject_idx.append(subject_list.index(subj))
```

```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

```python
all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
unique, counts = np.unique(all_outputs, return_counts=True)
```

iii. This is not called out in the notes, but it follows directly from the implementation: identical 360-bin time vectors are recreated trial by trial, and `subject_list.index` rescans a short list on every session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes plotting outputs and visualization statistics that are not part of the saved dataset, computes `total_neurons` without using it, and builds continuous motion-energy trial lists only to discard them after discretization. It also computes `bin_edges` but only keeps human-readable percentile labels in `output_values`.

ii. 
```python
if show_processing and i < 2:
    _plot_processing(...)
```

```python
total_neurons = sum(neural_all[s][0].shape[0] for s in range(len(neural_all)))
...
all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
```

```python
output_all, bin_edges, bin_labels = discretize_motion_energy(me_all_raw, N_OUTPUT_BINS)
...
'output_values': [bin_labels],
```

iii. The notes say the plots were for “visual inspection” and the output-distribution pass was for sanity checking. Those steps support debugging and validation, but they do not contribute to the saved decoder inputs once conversion is complete.
