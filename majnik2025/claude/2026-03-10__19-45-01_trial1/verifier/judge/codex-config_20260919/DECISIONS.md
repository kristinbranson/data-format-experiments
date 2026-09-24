# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every directory directly under `/app/data` as a subject, discovers every subdirectory under each subject as a session, sorts both lists, and loads four NumPy files for each session: Suite2p `F.npy` and `Fneu.npy`, plus `motion_energy_glob.npy` and `interframe_int.npy`. Full mode processes all 41 discovered sessions; sample mode keeps the first session from two different subjects.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
...
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes say the directory contains six subject folders and 41 daily sessions, and that these files are the Suite2p/behavioral sources needed for conversion. The AI reports that all sessions were included even where recordings were 30 rather than 20 minutes.

## 1-b. How are the data split into subjects?

i. Each top-level directory in `DATA_DIR` is treated as a mouse. Subject names are sorted, and an ordered unique list from selected sessions is used to construct `subjects` and `subject_idx`.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
subject_list = list(dict.fromkeys([s[0] for s in all_sessions]))
...
subject_idx.append(subject_list.index(subj))
```

iii. The notes identify the six directories `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` as the six mice and confirm 6–7 sessions per mouse.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a subject is one session. Each `(subject, session)` pair becomes one entry in the outer session dimension of `neural`, `input`, and `output`.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        all_sessions.append((subj, sess))
```

iii. The AI states that each subject contains daily Suite2p recording directories and validates that the resulting session counts are `7,7,7,7,6,7` (41 total).

## 1-d. How are the data split into trials?

i. The AI splits each continuous session into consecutive, non-overlapping 120-second blocks after 10-frame temporal averaging. Each trial has 360 bins; any incomplete tail is silently omitted by integer division.

ii.
```python
TRIAL_DURATION_SEC = 120
trial_frames = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)
...
n_trials = n_bins // trial_frames
for t in range(n_trials):
    start = t * trial_frames
    end = start + trial_frames
    neural_trials.append(neural_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The AI relied on the paper’s phrase “consecutive 2 minute blocks,” documenting 10 trials for 20-minute recordings and 15 for 30-minute recordings. It did not reconcile this with the task’s explicit instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. All complete 120-second blocks are retained, and only an incomplete end-of-session remainder is discarded.

ii.
```python
n_trials = n_bins // trial_frames
for t in range(n_trials):
    ...
```

iii. The notes state that the paper/reference analysis has no explicit trial curation and that partial end bins are discarded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from Suite2p `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0` in each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
Fc = F - neucoeff * Fneu
```

iii. The notes identify these as the supplied, already tracked Suite2p traces and cite the paper’s use of baseline-corrected fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI subtracts `0.7 * Fneu` from `F`, computes a custom baseline by applying a 60-second rolling minimum, then rolling maximum, then Gaussian smoothing with sigma 10 frames, subtracts that baseline, and averages consecutive groups of 10 frames. Although named `dff`, the result is baseline-subtracted fluorescence, not division by baseline.

ii.
```python
Fc = F - neucoeff * Fneu
F0 = _maximin_baseline(Fc, win_frames, sig_baseline)
dff = Fc - F0
...
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
...
dff_binned = bin_data(dff, bin_size)
```

iii. The AI intended to reproduce Suite2p `dcnv.preprocess` with default `maximin` parameters. Its notes describe fixing an earlier erroneous division and 300-frame sigma, but the final custom filter still does not use Suite2p itself and applies the Gaussian after rather than before the min/max filters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional ROI/cell filtering is performed; every row in the supplied `F.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
...
return dff_binned, me_binned
```

iii. The AI inspected the supplied data, found all `iscell` values equal to 1, and concluded that Track2p had already applied the paper’s `iscell > 0.5` and across-day tracking curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is cut at the start of each artificial 120-second block. Metadata calls this event “Start of 2-minute recording block,” with offsets 0 to 120 seconds.

ii.
```python
start = t * trial_frames
end = start + trial_frames
neural_trials.append(neural_binned[:, start:end].astype(np.float32))
...
'temporal_alignment_event': 'Start of 2-minute recording block',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. The AI followed the paper’s two-minute decoding blocks. There is no biological event in this continuous recording, so it used artificial block onset, but this conflicts with the requested 60-second segmentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz frames are averaged without overlap, yielding 3 Hz data and a nominal bin size of 333.33 ms. A final group shorter than 10 frames is dropped.

ii.
```python
FS = 30.0
BIN_SIZE = 10
...
return trimmed.reshape(new_shape).mean(axis=-1)
...
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The notes cite the methods statement that neural and behavioral traces were denoised by averaging bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the within-trial bin index, `BIN_SIZE`, and `FS`; it is not read from a raw timestamp variable. It measures time from the start of each 120-second trial rather than from the start of the session/experiment.

ii.
```python
n_timepoints = trial.shape[1]
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. The mapping notes explicitly choose “Elapsed time in seconds from start of 2-min trial” with range 0–119.7 seconds, despite the decoder specification requesting elapsed time from the beginning of the session.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, integer bin positions are multiplied by `10/30` seconds, reshaped to `(1, time)`, and cast to `float32`. The sequence is regenerated and reset for every trial.

ii.
```python
time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
session_inputs.append(time_input.reshape(1, -1).astype(np.float32))
```

iii. The AI says it independently verified this `np.arange * bin_size/fs` construction and regarded it as the required elapsed-time input.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Each time array is created from the length of its corresponding neural trial, so columns align one-to-one within a trial. However, because it resets at every block, it does not preserve the neural block’s absolute offset from session start.

ii.
```python
for trial in session_trials:
    n_timepoints = trial.shape[1]
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. The notes report shape/value checks and visual alignment, but do not address the loss of session-relative time.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`; `interframe_int.npy` supplies camera interframe intervals used to locate dropped frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
interframe = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes identify motion energy as already computed and use interframe intervals solely for synchronization repair.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy onto the neural frame grid, averages every 10 frames, splits it into 120-second trials, concatenates retained values from all sessions, computes one set of global percentile edges, and digitizes every trial into five integer classes. Duplicate edges are made strictly increasing by adding `1e-10`.

ii.
```python
me_interp = interpolate_missing_frames(me, interframe, n_frames)
me_binned = bin_data(me_interp, bin_size)
...
all_values = np.concatenate([me for session_trials in all_me_trials for me in session_trials])
bin_edges = np.percentile(all_values, percentiles)
...
binned = np.digitize(me, bin_edges[1:-1])
```

iii. The AI justified interpolation as necessary for frame alignment, 10-frame averaging from the paper, and global quintiles as producing a balanced overall distribution. It did not follow the explicit “selected per session” requirement.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. One global set of 0th, 20th, 40th, 60th, 80th, and 100th percentile boundaries is computed across all retained trials and sessions. Internal boundaries are passed to `np.digitize`, producing labels 0–4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
...
binned = np.digitize(me, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes repeatedly call this “global quintile discretization” and validate a global 20% class distribution. No rationale is given for overriding the instruction that bins be selected separately per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If the lengths differ, median-normalized interframe intervals estimate how many frames were missed between observed video samples. The code builds positions for observed motion-energy values and linearly interpolates at every neural-frame position; excess motion-energy samples are truncated. Neural and interpolated motion energy are then averaged with the same bin boundaries and sliced with the same trial indices.

ii.
```python
median_ifi = np.median(interframe_int)
ratios = interframe_int / median_ifi
missed_counts = np.round(ratios).astype(int) - 1
...
me_interp = np.interp(neural_positions, me_positions,
                      motion_energy.astype(np.float64))
...
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
```

iii. The AI says neural imaging and video are synchronized at 30 Hz and interpolation repairs up to 148 missing video frames. It reports visual neural/behavior alignment checks, but does not assert that inferred missing counts reconcile exactly with the target length.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video frames are filled by linear interpolation based on inferred dropped-frame positions. If motion energy is longer than neural data it is silently truncated; interpolation extrapolates endpoint values if inferred positions do not span the target. Short final 10-frame bins and incomplete 120-second trials are discarded. Duplicate percentile edges are perturbed by `1e-10`.

ii.
```python
if n_missing < 0:
    return motion_energy[:n_neural_frames].astype(np.float64)
...
me_interp = np.interp(neural_positions, me_positions, motion_energy.astype(np.float64))
...
n_trials = n_bins // trial_frames
...
bin_edges[i] = bin_edges[i-1] + 1e-10
```

iii. The AI describes interpolation and tail removal as correct edge-case handling and validated output shapes/ranges. Unlike the reference, it does not assert exact equality after inserting known dropped frames.

## 6-a. What are the most time-consuming steps of the code?

i. The full-session maximin baseline filters dominate conversion; loading full fluorescence arrays is the main I/O cost. Optional plotting and full decoder training are additional costs outside the essential conversion. The AI measured roughly 1.5 seconds per session and about 60 seconds for 41 sessions after fixing the Gaussian sigma.

ii.
```python
Flow = minimum_filter1d(Fc, size=win_frames, axis=1)
Flow = maximum_filter1d(Flow, size=win_frames, axis=1)
Flow = gaussian_filter1d(Flow, sigma=sig_frames, axis=1)
```

iii. The notes say an earlier sigma error made processing take about 1420 seconds and that the corrected filtering reduced it to about 60 seconds total.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop constructing `me_positions` can be replaced by a cumulative sum. Trial slicing and output digitization loops could be reshaped/vectorized, and repeated linear searches for subject indices could use a dictionary, though these are small relative to filtering.

ii.
```python
for i in range(1, n_me):
    me_positions[i] = me_positions[i-1] + 1 + missed_counts[i-1]
...
for t in range(n_trials):
    ...
...
subject_idx.append(subject_list.index(subj))
```

iii. The AI did not discuss vectorization in its notes; it focused runtime investigation on the baseline sigma bug.

## 6-c. What processing does the code repeat multiple times?

i. Each session independently repeats loading, neuropil correction, three baseline filters, missing-frame interpolation, and 10-frame averaging. Identical within-trial time vectors are also regenerated for every trial, and `subject_list.index(subj)` is repeatedly searched for every session.

ii.
```python
for i, (subj, sess) in enumerate(all_sessions):
    F, Fneu, me, interframe = load_session(subj_dir, sess)
    dff_binned, me_binned = process_session(F, Fneu, me, interframe)
...
for trial in session_trials:
    time_input = np.arange(n_timepoints) * (BIN_SIZE / FS)
```

iii. The AI does not identify repeated work as a concern. Session-specific signal processing is necessary, but the identical reset time vectors could be computed once and reused.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Motion-energy values in incomplete end-of-session trials are included when binning but discarded before percentile calculation. Optional `--show-processing` creates five-panel plots for two sessions, which are not used by the saved dataset or decoder. The script also computes detailed summary distributions and labels percentile ranges, used only for diagnostics/documentation.

ii.
```python
dff_binned = bin_data(dff, bin_size)
me_binned = bin_data(me_interp, bin_size)
...
n_trials = n_bins // trial_frames
...
if show_processing and i < 2:
    _plot_processing(...)
```

iii. The AI intentionally added plots and summaries as sanity checks, stating that it visually verified alignment. It does not characterize these diagnostics as unnecessary, although they do not affect downstream data.
