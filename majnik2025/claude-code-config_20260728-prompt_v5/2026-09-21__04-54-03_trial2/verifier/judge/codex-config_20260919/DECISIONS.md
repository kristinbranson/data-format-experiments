# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every directory directly under `/app/data` as a subject, then every directory under each subject as a session. For each session it loads `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and `move_deve/motion_energy_glob.npy`. Full mode processes all discovered session pairs sequentially; sample mode substitutes the first and last pair.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(os.path.join(DATA_ROOT, d))])
result = []
for subj in subjects:
    subj_dir = os.path.join(DATA_ROOT, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    for sess in sessions:
        result.append((subj, sess))
```
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The notes identify the hierarchy as six mouse directories, each containing dated recordings, with fluorescence and behavior in consistent subdirectories. The agent reports finding all 6 subjects and 41 sessions. Unlike the reference, it did not restrict subject directory names to `jm*`.

## 1-b. How are the data split into subjects?

i. Each top-level directory under `/app/data` is treated as a subject. Subject names are sorted, deduplicated from the selected session list, and mapped to integer indices.

ii.
```python
subjects_list = sorted(set(s[0] for s in all_sessions))
subjects_map = {s: i for i, s in enumerate(subjects_list)}
subject_idx_list.append(subjects_map[subj])
```

iii. The agent concluded that the six top-level `jm...` folders are individual mice and verified 6 subjects in the output. It relied on the observed directory contents rather than enforcing the reference's `jm` naming convention.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory within a subject is one session (one dated daily recording). Each becomes one outer entry in `neural`, `input`, and `output`.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
for sess in sessions:
    result.append((subj, sess))
...
for subj, sess in all_sessions:
    neural_trials, input_trials, output_trials, n_neurons = process_session(subj, sess, ...)
    neural_all.append(neural_trials)
```

iii. The notes describe each dated subdirectory as a separate recording day and explicitly adopt “session = recording day.” Sorting provides deterministic subject/session order.

## 1-d. How are the data split into trials?

i. After 10-frame temporal averaging, each session is divided into contiguous, non-overlapping 60-second trials. At 3 Hz this is 180 samples per trial. Integer division silently drops any incomplete final trial.

ii.
```python
binned_rate = FRAME_RATE / BIN_SIZE  # 3 Hz
timepoints_per_trial = int(TRIAL_DURATION * binned_rate)  # 180
n_trials = n_binned_frames // timepoints_per_trial
for t in range(n_trials):
    start = t * timepoints_per_trial
    end = start + timepoints_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. Continuous spontaneous-behavior recordings have no natural trials, so the agent followed the explicit instruction to split sessions into 60-second trials. The observed 20- and 30-minute recordings divide evenly into 20 or 30 trials.

## 1-e. How are trials filtered based on quality controls?

i. No quality-based trial filtering is performed. Only incomplete tail data are omitted through integer division; every complete 60-second segment is retained.

ii.
```python
n_trials = n_binned_frames // timepoints_per_trial
for t in range(n_trials):
    ...
```

iii. The notes state that no trial curation is described for these continuous recordings. They therefore retain all full trials and plan “no partial trials.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from Suite2p `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence) in `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. The agent rejected `spks.npy` because the paper describes decoding from baseline-corrected fluorescence/dF/F. It used both raw and neuropil traces to reproduce neuropil correction.

## 2-b. How is the `neural` data processed?

i. The agent casts fluorescence to float32, computes neuropil-corrected fluorescence `Fc = F - 0.7*Fneu`, estimates a maximin baseline neuron by neuron (Gaussian filter, 60-second minimum filter, then maximum filter), clamps the baseline below at `1e-6`, and computes conventional normalized dF/F `(Fc-F0)/F0`. It then averages every 10 consecutive frames and casts each trial to float32.

ii.
```python
Fc = F.astype(np.float32) - NEUCOEFF * Fneu.astype(np.float32)
F0 = compute_baseline_maximin(Fc)
F0_safe = np.maximum(F0, 1e-6)
dff = (Fc - F0) / F0_safe
```
```python
smoothed = gaussian_filter1d(trace, sig_baseline)
smoothed = minimum_filter1d(smoothed, win)
smoothed = maximum_filter1d(smoothed, win)
...
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The trajectory shows the agent discovered that Suite2p `dcnv.preprocess` returns baseline subtraction rather than division, but deliberately chose conventional normalized dF/F because of the paper's terminology and cross-neuron comparability. The notes nevertheless claim this matches Suite2p defaults; that claim does not match the reference implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron/ROI filtering is applied; every row of `F.npy` and `Fneu.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
...
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```

iii. The agent inspected `iscell.npy`, found all entries to be cells, and concluded that the released arrays already contain Track2p-tracked neurons that passed the `iscell > 0.5` curation and persist across days.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no biological event alignment. Trials are fixed windows indexed from session start; metadata names session start as the alignment event. Trial 0 begins at the event and later trials retain their session-relative time input.

ii.
```python
start = t * timepoints_per_trial
end = start + timepoints_per_trial
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
```
```python
'temporal_alignment_event': 'Session start (beginning of recording)',
'off_start': 0.0,
'off_end': float(TRIAL_DURATION),
```

iii. The notes explain that recordings are continuous spontaneous behavior with no stimulus event, so the natural common reference is the beginning of the recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30-Hz frames are averaged without overlap, producing 3-Hz data with 333.33-ms bins. A tail shorter than 10 frames is trimmed.

ii.
```python
BIN_SIZE = 10
n_bins = n_frames // bin_size
trimmed = data[..., :n_bins * bin_size]
return trimmed.reshape(*data.shape[:-1], n_bins, bin_size).mean(axis=-1)
```
```python
bin_size_ms = (BIN_SIZE / FRAME_RATE) * 1000
```

iii. This follows the paper's decoding procedure of denoising fluorescence and behavior by averaging 10 consecutive timestamps. Applying the same operation to both streams preserves their nominal alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from a raw timestamp file. It is derived from the integer index of each binned sample and the fixed effective rate `30/10 = 3 Hz`.

ii.
```python
binned_rate = FRAME_RATE / BIN_SIZE
time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
```

iii. The agent judged index-derived time appropriate because acquisition is specified at a constant 30 Hz and the requested variable is elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, global binned indices from `start` through `end-1` are divided by 3 Hz, converted to float32, and reshaped to `(1, 180)`. Thus time does not reset at trial boundaries.

ii.
```python
time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
input_trials.append(time_seconds.reshape(1, -1))
```

iii. The notes specify elapsed seconds from session start, including examples where later trials start at their actual session-relative time.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time uses exactly the same `start:end` binned indices and contains one value for every neural column. Index `j` in the time input therefore denotes the acquisition time of neural column `j` in that trial.

ii.
```python
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
time_seconds = (np.arange(start, end) / binned_rate).astype(np.float32)
input_trials.append(time_seconds.reshape(1, -1))
```

iii. The agent's sanity checks reportedly confirmed that time starts at 0 for trial 0 and at 180 seconds for trial 3, with the same 180 samples as neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is derived only from the precomputed `move_deve/motion_energy_glob.npy`. Although the notes discuss `tstamps.npy` and `interframe_int.npy`, the final script does not load either one.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The agent identifies this file as the paper's global behavioral-video motion-energy trace. It investigated timing files but, unable to reconcile their units, chose neural frame count as the reference and length-based interpolation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. If its length differs from neural data, the complete motion trace is linearly resampled over normalized `[0,1]` coordinates to the neural frame count. It is then averaged over 10-frame bins, split into complete 60-second trials, and categorized using session-level quintiles computed from the concatenated retained trials.

ii.
```python
x_orig = np.linspace(0, 1, len(me))
x_target = np.linspace(0, 1, n_target_frames)
me_interp = np.interp(x_target, x_orig, me)
```
```python
me_binned = bin_data(me, BIN_SIZE)
me_trials.append(me_binned[start:end])
me_discretized, bin_edges = discretize_motion_energy(me_trials, N_BINS)
```

iii. The agent wanted to match neural and video lengths before common 10-frame averaging. It described this as handling dropped camera frames “as suggested by data documentation,” although it did not use drop locations as the reference does.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. All retained binned motion samples in one session are concatenated. The 0th, 20th, 40th, 60th, 80th, and 100th percentiles are computed; outer edges are replaced by infinities, and `np.digitize` assigns integer labels 0–4 using the four internal thresholds.

ii.
```python
all_me = np.concatenate(me_binned_trials)
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_me, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
for me_trial in me_binned_trials:
    binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The explicit decoder requirement calls for five equal-percentile bins selected per session. The agent validated an approximately/exactly 20% distribution for each class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Neural frame count is treated as the target. A shorter or longer motion trace is globally resampled to that count, then neural and motion arrays are averaged separately over identical 10-frame positions and sliced with identical trial bounds.

ii.
```python
me = interpolate_motion_energy(me, n_frames)
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
me_trials.append(me_binned[start:end])
```

iii. The agent reasoned that video was microscope-triggered at 30 Hz and thus nominally synchronous. Because it could not interpret timestamp units, it assumed uniformly distributed correspondence and resampled the full trace. This differs from locating actual dropped frames and inserting only at those positions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Any motion/neural length mismatch is handled by global linear resampling of motion energy, regardless of mismatch direction or cause. Short tails that cannot form a 10-frame bin or a complete 60-second trial are discarded. Neural baseline values are clamped to `1e-6` before division. There is no explicit NaN handling or post-interpolation length assertion.

ii.
```python
if len(me) == n_target_frames:
    return me
x_orig = np.linspace(0, 1, len(me))
x_target = np.linspace(0, 1, n_target_frames)
return np.interp(x_target, x_orig, me)
```
```python
trimmed = data[..., :n_bins * bin_size]
n_trials = n_binned_frames // timepoints_per_trial
F0_safe = np.maximum(F0, 1e-6)
```

iii. Notes identify missing camera frames as the key defect and choose interpolation rather than dropping neural samples. They also specify clean full-trial cuts. The documented claim that missing frames were interpolated is accurate at a high level but omits that the implementation stretches the full time axis.

## 6-a. What are the most time-consuming steps of the code?

i. Baseline estimation/dF/F is the measured dominant computation; the code times dF/F, binning, and total session processing separately. It applies three long one-dimensional filters for every neuron. Full processing reportedly took about 47 seconds (roughly 1.1–1.8 seconds per session); loading and plotting, when requested, add I/O cost.

ii.
```python
t1 = time.time()
dff = compute_dff(F, Fneu)
t_dff = time.time() - t1
...
for i in range(n_neurons):
    smoothed = gaussian_filter1d(trace, sig_baseline)
    smoothed = minimum_filter1d(smoothed, win)
    smoothed = maximum_filter1d(smoothed, win)
```

iii. The notes' timing table calls the full pipeline about 1.1–1.8 seconds/session. The trajectory focused on reproducing Suite2p baseline processing, which is the computationally intensive portion.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron baseline loop is the main vectorization candidate because SciPy's filters can operate along the time axis for the whole `(neurons, time)` array. Trial construction and per-trial digitization could also be reshaped/vectorized, though they are small relative to filtering. The outer session loop is necessary for variable neuron counts and session-local thresholds but could be parallelized.

ii.
```python
for i in range(n_neurons):
    trace = Fc[i].astype(np.float32)
    smoothed = gaussian_filter1d(trace, sig_baseline)
    smoothed = minimum_filter1d(smoothed, win)
    smoothed = maximum_filter1d(smoothed, win)
    Flow[i] = smoothed
```
```python
for t in range(n_trials):
    ...
for me_trial in me_binned_trials:
    binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The agent did not explicitly discuss vectorization in its notes. Its use of array reshaping for 10-frame binning is already vectorized, but its hand-written baseline routine unnecessarily loops over neurons.

## 6-c. What processing does the code repeat multiple times?

i. Every session independently repeats file loading, motion resampling, baseline filtering, binning, trial assembly, and percentile calculation. Within baseline estimation, the same filter sequence is invoked separately for every neuron. With `--show-processing`, arrays already computed for conversion are traversed again for plotting.

ii.
```python
for subj, sess in all_sessions:
    neural_trials, input_trials, output_trials, n_neurons = process_session(...)
```
```python
for i in range(n_neurons):
    ...
```

iii. Session-local processing is required because recordings have different neuron counts and output bins must be selected per session. The agent provides no specific justification for the repeated per-neuron filter calls beyond implementing the baseline calculation straightforwardly.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes percentile endpoints and returns `bin_edges` from discretization, but does not save them in the output dataset; they are used only by optional plots. The `session_idx` plotting argument is passed but never used. In `--show-processing`, it creates six-panel diagnostic figures that are not decoder inputs. It also globally interpolates every motion sample in mismatch sessions rather than only reconstructing missing positions, modifying values unnecessarily.

ii.
```python
return discretized, bin_edges
...
me_discretized, bin_edges = discretize_motion_energy(me_trials, N_BINS)
```
```python
def plot_processing(..., session_idx):
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))
```

iii. Diagnostic plots were intentionally added for visual validation and limited to two sessions in show-processing mode. The notes treat this as a sanity check, not downstream data. They do not recognize the unused argument or discarded thresholds as inefficiencies.
