# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by listing every directory under `DATA_DIR`, then discovers sessions by listing every subdirectory inside each subject directory. For each session it loads fluorescence, neuropil fluorescence, motion energy, and interframe intervals, then processes the continuous recording and later splits it into trials.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset consists of 6 subject directories and 41 session directories, and that each session contains Suite2p outputs plus movement files. The trajectory shows it inspected the directory tree first and decided to load the four arrays above for each session.

## 1-b. How are the data split into subjects?

i. Subjects are defined as all directories found directly under `DATA_DIR`, sorted lexicographically. A separate `subject_list` is then rebuilt from the subject names appearing in `all_sessions`.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])

subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))
subject_idx.append(subject_list.index(subj))
```

iii. In `CONVERSION_NOTES.md`, the AI states there are 6 mice (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and treats each top-level subject directory as one mouse.

## 1-c. How are the data split into sessions?

i. Each session is defined as a subdirectory inside a subject directory, sorted lexicographically. Each `(subject, session)` pair becomes one element of `all_sessions` and later one session in the output lists.

ii. ```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. The notes say each daily recording is one session and report 41 total sessions across the 6 mice.

## 1-d. How are the data split into trials?

i. The AI treats each session as a continuous recording, bins it by 10 frames first, and then splits the binned time series into consecutive 2-minute blocks. Trial length is therefore `120 s * 30 Hz / 10 = 360` binned time points.

ii. ```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120

trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)

def split_into_trials(neural_binned, me_binned, trial_frames):
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // trial_frames
    for t in range(n_trials):
        start = t * trial_frames
        end = start + trial_frames
        neural_trials.append(neural_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. `CONVERSION_NOTES.md` repeatedly justifies this with phrases from the paper such as “bins of 10 consecutive timestamps” and “splits were done on consecutive 2 minute blocks,” and the trajectory summary at the end reiterates “2-minute trial segmentation.”

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering. Trials are included if they fit into the fixed-length segmentation; any trailing partial trial is dropped implicitly by floor division.

ii. ```python
n_trials = n_bins // trial_frames

for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The notes say “Trials: No explicit curation; missing video frames interpolated.” No additional trial exclusion rule is documented in the notes or trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p fluorescence and neuropil traces: `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The notes map `F.npy` and `Fneu.npy` directly to the target `neural` field and describe them as the source for the calcium signal.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil (`F - 0.7 * Fneu`), estimates a baseline with a hand-written “maximin” routine using `minimum_filter1d`, `maximum_filter1d`, and `gaussian_filter1d`, subtracts that baseline, and then averages the result in 10-frame bins.

ii. ```python
Fc = F - neucoeff * Fneu
F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
dff = Fc - F0

Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)

dff_binned = bin_data(dff, bin_size)
```

iii. In `CONVERSION_NOTES.md` and the trajectory, the AI justifies this as “matching Suite2p’s `dcnv.preprocess()`,” including a later bug fix where it changed from division to subtraction and from `sig_baseline * fs` to `sig_baseline = 10` frames. The code, however, still implements its own approximation rather than calling Suite2p’s preprocessing function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied in `convert_data.py`. Every row of `F.npy` is kept.

ii. ```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff_binned, me_binned = process_session(F, Fneu, me, interframe)
```

iii. The notes justify this by saying the provided data are already pre-filtered by Track2p/Suite2p and that all `iscell` values are 1.0, so no further curation is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each artificial 2-minute recording block, not to any natural behavioral event. The metadata explicitly describe the alignment event as the start of the 2-minute block.

ii. ```python
'metadata': {
    'temporal_alignment_event': 'Start of 2-minute recording block',
    'off_start': 0.0,
    'off_end': TRIAL_DURATION_SEC,
}
```

iii. The notes and final trajectory summary describe the dataset as continuous recordings segmented into consecutive 2-minute blocks, so the AI treated block start as the temporal anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 333.3 ms bins, produced by averaging every 10 original 30 Hz frames. Both neural and motion-energy streams are rebinned this way.

ii. ```python
FS = 30.0
BIN_SIZE = 10

dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)

'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The notes explicitly cite the paper phrase “averaging in bins of 10 consecutive timestamps” and use that as the rationale for 10-frame temporal rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from any raw timestamp variable. It is synthesized from the number of time points in each binned trial using `np.arange` and the binned sample period `BIN_SIZE / FS`.

ii. ```python
n_timepoints = trial.shape[1]
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The notes describe the input as “Elapsed time in seconds from start of 2-min trial,” so the AI intentionally generated time analytically instead of loading it from the data files.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one monotonically increasing vector per trial, starting at 0 for every trial and stepping by 1/3 second after the 10-frame binning.

ii. ```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. `CONVERSION_NOTES.md` says the intended variable is “Time elapsed from start of 2-minute trial block (seconds)” with range `[0.0, 119.7] s`.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI gives every trial a time vector with the same number of binned samples as the neural matrix in that trial, so alignment is one-to-one at the rebinned trial level. The time vector resets at the start of each artificial trial.

ii. ```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
    session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this by treating each 2-minute block as the basic aligned unit and using time elapsed within that block as the decoder input.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy`, with `interframe_int.npy` used to infer dropped video frames and repair alignment to neural frames.

ii. ```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes and trajectory say `motion_energy_glob.npy` already contains the behavioral signal and that interframe intervals are used to handle missing video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy onto neural-frame positions using `np.interp` and inferred missing-frame counts from interframe intervals, then averages motion energy in 10-frame bins. It does not normalize by session standard deviation before discretization.

ii. ```python
median_ifi = np.median(interframe_int)
ratios = interframe_int / median_ifi
missed_counts = np.round(ratios).astype(int) - 1
...
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
...
me_binned = bin_data(me_interp, bin_size)
```

iii. The notes justify interpolation as a way to match dropped video frames to the neural stream. They also justify 10-frame averaging by citing the paper’s “10 consecutive timestamps” language. Although the project instructions required normalized motion energy, the implemented code does not perform the stated normalization step.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. After collecting all binned motion-energy values across all sessions and trials, the AI computes global percentile edges for 5 bins and discretizes with `np.digitize`. It also forces the percentile edges to be strictly increasing by adding `1e-10` when needed.

ii. ```python
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)

for i in range(1, len(bin_edges)):
    if bin_edges[i] <= bin_edges[i-1]:
        bin_edges[i] = bin_edges[i-1] + 1e-10

binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes call this “global quintile bins across all sessions” and justify it as producing balanced output classes.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first interpolates motion energy to the neural frame count, then bins both signals by 10 frames, and finally slices them into the same 2-minute trial windows. The resulting `output` trial has one row and the same number of time bins as the neural trial.

ii. ```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)
...
session_disc.append(binned.reshape(1, -1).astype(np.int64))
```

iii. The notes justify this as preserving neural/behavior alignment after repairing dropped video frames and applying the same temporal binning to both streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames by inferring how many frames were missed from interframe intervals and interpolating motion energy to a full neural-frame grid. If motion energy is longer than neural data, it silently truncates the excess. Partial trailing data that do not fill a full binned trial are dropped implicitly.

ii. ```python
if n_me == n_neural_frames:
    return motion_energy.astype(np.float64)

n_missing = n_neural_frames - n_me
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)

me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
...
n_trials = n_bins // trial_frames
```

iii. The notes emphasize missing-frame interpolation as the main edge-case handler and say end-of-session partial bins are discarded. No explicit justification is given for silent truncation when motion energy is longer than neural data.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive step is the neural preprocessing inside `compute_dff`, especially the repeated 1D min/max/Gaussian filters over all neurons and frames. Optional plotting is also relatively heavy in `--show-processing` mode, but only for up to two sessions.

ii. ```python
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
...
fig, axes = plt.subplots(5, 1, figsize=(16, 20))
```

iii. The notes explicitly identify “dF/F + binning” as the dominant runtime and mention a large speed improvement after changing the baseline parameters.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loops are the construction of `me_positions` in missing-frame interpolation, the trial-splitting loop, the nested loops that build `input_all`, and the nested loops used to discretize each trial separately.

ii. ```python
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]

for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])

for session_trials in neural_all:
    for trial in session_trials:
        time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)

for session_trials in all_me_trials:
    for me in session_trials:
        binned = np.digitize(me, bin_edges[1:-1])
```

iii. The notes do not explicitly discuss these loops. This assessment is inferred from the final code structure.

## 6-c. What processing does the code repeat multiple times?

i. The code makes several separate passes over the same session/trial structure: one pass to process sessions, another to split data into trials, another to build time inputs, another to discretize motion-energy trials, and another to summarize output distributions for printing.

ii. ```python
for i, (subj, sess) in enumerate(all_sessions):
    ...
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned, trial_frames)

for session_trials in neural_all:
    for trial in session_trials:
        ...

for session_trials in all_me_trials:
    for me in session_trials:
        ...

all_outputs = np.concatenate([o for sess in output_all for o in sess], axis=1).flatten()
```

iii. The notes focus on correctness and runtime, not repeated passes. This decision summary is therefore inferred from the implementation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and prints several summary quantities used only for logging, keeps `bin_edges` even though only labels are stored in the saved dataset, computes `total_neurons` without using it, and optionally generates processing plots that are not consumed downstream by the decoder.

ii. ```python
output_all, bin_edges, bin_labels = discretize_motion_energy(me_all_raw, N_OUTPUT_BINS)
...
total_neurons = sum(neural_all[s][0].shape[0] for s in range(len(neural_all)))
...
print(f"Output bins: {bin_labels}")
...
if show_processing and i < 2:
    _plot_processing(...)
```

iii. The notes justify the plots and summaries as sanity checks, but they are not part of the saved decoder dataset and are discarded by downstream analyses.
