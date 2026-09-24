# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for sorted directories beginning with `jm`, scans every such directory for sorted session subdirectories, and loads every session's `suite2p/plane0/F.npy`, `Fneu.npy`, `move_deve/motion_energy_glob.npy`, and `tstamps.npy`. It preprocesses each session immediately and retains it in `all_sessions_data`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
for subj in subjects:
    sessions = sorted([d for d in os.listdir(subj_dir)
                       if os.path.isdir(os.path.join(subj_dir, d))])
    for sess_name in sessions:
        dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
```

iii. The trajectory says the AI inspected all subject/session directories and array shapes and identified six mice with 6–7 daily sessions. It treated the directory convention as the exhaustive dataset organization.

## 1-b. How are the data split into subjects?

i. A subject is each sorted top-level directory whose name starts with `jm`; sessions are mapped back to subjects through `subjects.index(subj)`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
subj_i = subjects.index(subj)
subject_idx.append(subj_i)
```

iii. The AI concluded from the directory survey that `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` are the six mice.

## 1-c. How are the data split into sessions?

i. Every immediate subdirectory of a subject is one session, sorted by its directory name. One output session is created per daily recording directory.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                   if os.path.isdir(os.path.join(subj_dir, d))])
for sess_name in sessions:
    sess_dir = os.path.join(subj_dir, sess_name)
```

iii. The AI observed 20- or 30-minute recordings in dated subdirectories and interpreted each dated directory as a daily session.

## 1-d. How are the data split into trials?

i. The AI splits each binned continuous session into non-overlapping 120-second blocks (360 bins at 3 Hz) and silently discards any incomplete tail.

ii.
```python
TRIAL_DURATION_S = 120
bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
```

iii. The trajectory says it chose two-minute blocks because the paper used consecutive two-minute blocks for cross-validation. It overlooked the task's explicit instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality-control filtering is performed. Only an incomplete final block is excluded by floor division.

ii.
```python
n_trials = n_bins // bins_per_trial
```

iii. The trajectory gives no trial-quality criterion; it focused on ensuring enough fixed-length blocks for decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy` from plane 0. The available `spks.npy` and `iscell.npy` are inspected in the trajectory but not used.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. The AI identified these as raw fluorescence and neuropil fluorescence and decided to reproduce what it understood as Suite2p default preprocessing.

## 2-b. How is the `neural` data processed?

i. The AI subtracts `0.7 * Fneu`, Gaussian-smooths with sigma 300 frames, applies 1,800-frame running minimum and maximum filters, floors the baseline at `1e-6`, divides the baseline-subtracted trace by that baseline, casts to float32, and then averages non-overlapping groups of 10 frames.

ii.
```python
Fc = F - neucoeff * Fneu
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
Flow = np.maximum(Flow, 1e-6)
dff = (Fc - Flow) / Flow
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The AI reasoned that the paper's “baseline corrected fluorescence” meant dF/F and manually reproduced Suite2p's maximin filters using parameters read from `ops.npy`. It explicitly described this as a “reasonable dF/F calculation,” although the reference uses `dcnv.preprocess`, which subtracts the baseline without this division.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron/ROI filtering is applied; every row in `F.npy` is retained. `iscell.npy` is not consulted by the converter.

ii.
```python
n_neurons, n_frames = F.shape
# no mask is applied before dff = compute_dff(F, Fneu)
```

iii. The trajectory observed that neuron counts were consistent across a mouse's tracked sessions and treated all supplied rows as tracked neurons. It did not justify an additional `iscell` filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Trials are contiguous slices from recording onset, and metadata calls the alignment event `recording_onset`, with offsets 0 to 120 seconds.

ii.
```python
neural_trials.append(dff_binned[:, start:end])
'temporal_alignment_event': 'recording_onset',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION_S),
```

iii. The AI reasoned that spontaneous continuous recordings have no trigger event, so recording/session onset is the appropriate descriptive anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged into each non-overlapping bin, yielding 3 Hz or 333.33 ms per bin. Any tail shorter than 10 frames is truncated.

ii.
```python
BIN_SIZE = 10
dff_binned = bin_data(dff, BIN_SIZE)
'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. The AI cited the paper's decoding method, which denoised both dF/F and behavior by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is synthetic, not read from raw timestamps: it is derived from the binned neural sample index, fixed `BIN_SIZE=10`, and `FS=30`.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The AI chose elapsed seconds from recording onset and relied on the fixed 30-Hz imaging rate rather than the camera timestamp data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code calculates the center of every 10-frame bin, producing `0.1667, 0.5, 0.8333, ...` seconds, then reshapes each trial slice to `(1, n_timepoints)`.

ii.
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
sess_input.append(t_trial.reshape(1, -1))
```

iii. The trajectory explicitly selected actual elapsed session time and bin-center timestamps. This differs from the reference's left-edge convention but is internally coherent.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is generated with exactly one value per binned neural column and sliced using the same `start:end` trial boundaries.

ii.
```python
neural_trials.append(dff_binned[:, start:end])
time_trials.append(time_bins[start:end])
```

iii. The AI intended time to remain continuous from session onset across successive blocks, so common indexing provides alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`; `move_deve/tstamps.npy` is used to infer dropped camera frames.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
me = interpolate_motion_energy(me, tstamps, n_frames)
```

iii. The AI recognized motion energy as already computed from framewise pixel changes and used timestamps to repair length mismatches.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If lengths differ, timestamp gaps are converted to nominal frame indices and linearly interpolated to the neural frame count. The repaired signal is averaged over 10-frame bins, split into trials, then digitized using quintile thresholds computed globally from all retained trials in all sessions.

ii.
```python
frame_indices[i + 1] = cum_idx
me_interp = np.interp(np.arange(n_frames), frame_indices, me)
me_binned = bin_data(me, BIN_SIZE)
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
```

iii. The AI reasoned that dropped camera frames required interpolation, that paper-matched denoising required 10-frame means, and that percentile ranks make explicit normalization unnecessary. It ultimately chose global thresholds for consistent definitions, despite earlier reasoning in the trajectory favoring per-session thresholds.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. One set of 0/20/40/60/80/100 percentile edges is calculated over all binned, complete-trial motion-energy values across the entire dataset. Interior edges are passed to `np.digitize`, producing labels 0–4, then clipped.

ii.
```python
thresholds = np.percentile(all_me_values, np.linspace(0, 100, n_bins + 1))
binned = np.digitize(me_values, thresholds[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The final trajectory and notes say global thresholds provide consistent class definitions and exact global balance. The AI noticed severe per-session imbalance but accepted it. This conflicts with “selected per session” in the task.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Timestamp gaps are used to map observed camera samples onto inferred 30-Hz frame indices; `np.interp` fills all indices through the neural frame count. Neural and motion signals are then separately averaged into identical 10-frame bins and sliced at identical trial boundaries.

ii.
```python
n_skipped = int(np.round(ifi[i] / median_ifi))
cum_idx += n_skipped
frame_indices[i + 1] = cum_idx
me_interp = np.interp(all_indices, frame_indices, me)
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The AI inspected the largest mismatch, found dropped frames spread throughout with roughly doubled intervals, and chose interpolation instead of assuming direct index alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion-energy arrays are expanded by timestamp-based linear interpolation. Equal-length arrays are returned unchanged. Incomplete 10-frame bins and incomplete 120-second trials are truncated. There is no explicit guard for longer motion arrays, malformed gaps, NaNs, missing files, or a final inferred index inconsistent with `n_frames`.

ii.
```python
if len(me) == n_frames:
    return me
me_interp = np.interp(np.arange(n_frames), frame_indices, me)
n_bins = n // bin_size
n_trials = n_bins // bins_per_trial
```

iii. The AI found camera/neural length differences as large as 116–148 frames and regarded interpolation as necessary. It considered the gaps isolated single-frame drops and used linear interpolation as a practical repair.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant computation is per-session neural baseline processing: Gaussian filtering followed by long-window minimum and maximum filters over every neuron and frame. Loading large `.npy` arrays and serializing the full nested dataset are secondary costs.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The trajectory did not benchmark stages, but it deliberately implemented full-session Suite2p-like baseline operations; their size and sliding-window nature make them the evident bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The timestamp-to-frame mapping loop can be replaced by rounded interval ratios plus a cumulative sum. Trial slicing and per-trial discretization/reshaping loops could also be expressed as reshapes and array operations; `subjects.index` could be replaced by a precomputed dictionary.

ii.
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx

for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
```

iii. The trajectory does not discuss vectorization. It characterized dropped frames as sparse, which likely reduced its concern about the explicit mapping loop.

## 6-c. What processing does the code repeat multiple times?

i. It performs the full neural preprocessing and motion interpolation independently for every session. It also applies the same global discretization operation one trial at a time, repeatedly searches `subjects` with `subjects.index`, and traverses the completed data again for sanity statistics and sampling.

ii.
```python
for sess_name in sessions:
    dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
...
subj_i = subjects.index(subj)
for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
```

iii. No explicit justification is recorded. Session-specific preprocessing is necessary, whereas repeated lookup, per-trial application, and post-hoc traversals are implementation conveniences.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The required converted dataset uses the processing results, so little core work is discarded. However, the script always builds an extra six-session sample, computes extensive sanity-check aggregates and console output, and imports `sys` without using it; those do not contribute to `converted_data.pkl`. The `sample_mode` and `n_2p_frames` parameters are also unused.

ii.
```python
def convert_data(data_dir=DATA_DIR, sample_mode=False):
def load_and_process_session(session_dir, n_2p_frames=None):
print_sanity_checks(data)
sample = create_sample(data)
```

iii. The trajectory says the sample, plots, statistics, and decoder runs were created for quick validation and confidence in format/performance, not for downstream use of the required full pickle.
