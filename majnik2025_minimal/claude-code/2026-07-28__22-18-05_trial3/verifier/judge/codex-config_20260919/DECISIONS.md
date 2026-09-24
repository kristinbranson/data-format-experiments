# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every directory in `/app/data` whose name starts with `jm`, sorts those mouse names, then sorts every subdirectory under each mouse as a session. For every session it loads `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `move_deve/motion_energy_glob.npy`, and `move_deve/interframe_int.npy`. It processes every discovered session and later slices each continuous session into complete trials.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
...
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The trajectory says the AI inspected the project structure and actual files, recognized the mouse/session directory convention, and intended to process the full dataset. It later checked that this yielded 6 mice and 41 sessions.

## 1-b. How are the data split into subjects?

i. Each sorted top-level `jm*` directory is treated as one subject. Its position in the sorted list is stored as the session's `subject_idx`.

ii.
```python
subjects = mice
for mouse_idx, mouse in enumerate(mice):
    ...
    all_subject_idx.append(mouse_idx)
```

iii. The AI inferred that each `jm*` folder is one mouse and verified six such folders. It also noted that neuron counts stay constant within each tracked mouse.

## 1-c. How are the data split into sessions?

i. Every sorted subdirectory of a mouse directory is one daily recording session. Each becomes one element of the outer `neural`, `input`, and `output` lists.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
for sess_name in sessions:
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The AI interpreted the data hierarchy as longitudinal daily recordings and confirmed 41 total sessions (6–7 per mouse), consistent with the paper.

## 1-d. How are the data split into trials?

i. The AI makes consecutive, non-overlapping **120-second** trials after 10-frame temporal averaging. Each trial has 360 bins; incomplete trailing data are implicitly discarded by floor division. This conflicts with the task's explicit requirement to split sessions into 60-second trials.

ii.
```python
TRIAL_DURATION_S = 120
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)
...
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The trajectory explicitly says the AI chose two-minute blocks because the paper used “consecutive 2 minute blocks” for cross-validation. It prioritized that paper detail over the decoder task's direct 60-second-trial instruction.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality-control filter is applied. All complete 120-second blocks are retained and only a final incomplete block is omitted.

ii.
```python
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    ...
    session_neural.append(trial_neural.astype(np.float32))
```

iii. The AI found no natural trials or stated trial QC in the source material, so it retained all complete fixed-duration blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from Suite2p `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence), both from `plane0` for each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff_suite2p(F, Fneu, fs=FS)
```

iii. The AI identified these as the available Suite2p fluorescence sources and tied them to the paper's use of baseline-corrected fluorescence.

## 2-b. How is the `neural` data processed?

i. It subtracts 0.7 times neuropil fluorescence, casts to `float32`, and calls Suite2p's `preprocess` with the `maximin` baseline, a 60-second baseline window, sigma 10, and 30 Hz. The resulting baseline-subtracted trace is then averaged over non-overlapping groups of 10 frames.

ii.
```python
Fc = (F - neucoeff * Fneu).astype(np.float32)
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
...
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The trajectory shows several iterations: the AI first considered dividing by a baseline, tested it, found worse decoder behavior, then concluded that Suite2p `preprocess` itself returns the paper's “baseline corrected fluorescence traces” and retained baseline subtraction. It used the library implementation to match Suite2p exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron or trace filtering is performed; every row in `F.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
...
trial_neural = dff_binned[:, start:end]
```

iii. The AI inspected `iscell` and stated that all released ROIs were already marked as cells and tracked by Track2p, so an additional filter was unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external event realignment. Neural bins retain their order from session start and are cut into contiguous blocks. Metadata calls the alignment event “Start of imaging session,” with `off_start` set to 0 even though later trials do not begin at that event.

ii.
```python
trial_neural = dff_binned[:, start:end]
...
'temporal_alignment_event': 'Start of imaging session',
'off_start': 0.0,
'off_end': None,
```

iii. The AI treated elapsed session time as the relevant alignment because this is a continuous spontaneous-behavior recording without a stimulus-aligned natural trial structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz frames are averaged into each bin, giving 3 Hz sampling and a bin size of approximately 333.33 ms. Both neural activity and motion energy are rebinned this way before trial splitting.

ii.
```python
BIN_SIZE = 10
FS = 30.0
time_bin_size_ms = (BIN_SIZE / FS) * 1000
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The AI quoted the paper's decoding method: neural and behavioral traces were denoised by averaging bins of 10 consecutive timestamps. It binned both streams identically to preserve alignment.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is generated rather than loaded: it is derived from each binned sample's global bin index within its session, the 10-frame bin width, and the 30 Hz sampling rate.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The AI reasoned that absolute elapsed time from session start supplies temporal context across the fixed blocks and is preferable to resetting time within every trial.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code adds 0.5 to each integer bin index and multiplies by 10/30 seconds, so values represent bin centers rather than bin left edges. It reshapes to `(1, time)` and casts to `float32`.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
session_input.append(trial_input.astype(np.float32))
```

iii. The trajectory justifies session-global rather than trial-relative time, but does not separately explain the half-bin center convention.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The same `start:end` binned index range used for each neural trial generates exactly one time value for every neural column.

ii.
```python
trial_neural = dff_binned[:, start:end]
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
```

iii. The AI kept time continuous across trials specifically to represent elapsed session time; common indices and the shared bin duration provide direct alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output originates from each session's precomputed `move_deve/motion_energy_glob.npy`. `interframe_int.npy` is additionally used to locate dropped video frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
me = interpolate_missing_frames(me_raw, ifi, n_frames)
```

iii. The AI identified global motion energy as the requested behavioral signal and investigated timestamps/interframe intervals to reconcile shorter video traces with neural frame counts.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing video samples are interpolated first. The aligned continuous signal is averaged in non-overlapping 10-frame bins, then session-specific percentiles are computed and the binned values are assigned to five categories.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
me_binned = bin_data(me, BIN_SIZE)
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. The AI reasoned that interpolation restores framewise correspondence, that both streams should receive the paper's 10-frame denoising, and that binning must occur before categorical discretization. It noted that separate amplitude normalization would not change percentile ranks.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Within each session, percentile edges at 0, 20, 40, 60, 80, and 100 percent are computed. `np.digitize` applies the four internal edges to produce integer classes 0–4; the last edge is nudged upward but is not passed to `digitize`.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(values, percentiles)
edges[-1] = edges[-1] + 1e-10
bins = np.digitize(values, edges[1:-1])
```

iii. The AI followed the instruction to make five equal-percentile bins selected per session and confirmed approximately equal class proportions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If motion energy is short, gaps are detected where interframe intervals exceed 1.5 times their median. The estimated number of absent samples is linearly interpolated between adjacent captured values into a preallocated neural-length array; any residual tail is padded with the last value. If motion energy is longer, it is truncated. After this, neural and motion traces are identically 10-frame averaged and sliced with the same trial indices.

ii.
```python
threshold = median_ifi * 1.5
gap_indices = np.where(ifi > threshold)[0]
missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
...
result[dst_idx] = me[src_idx] * (1 - alpha) + me[src_idx + 1] * alpha
...
trial_neural = dff_binned[:, start:end]
trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The trajectory says dropped camera frames were scattered single-frame gaps and that abnormal interframe intervals offered the best locations for interpolation. The AI viewed synchronized acquisition plus restored missing positions as sufficient for shared indexing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion-energy arrays are gap-interpolated and, if necessary, last-value padded; long arrays are truncated. Partial 10-frame bins and partial 120-second trials are discarded. There is no final assertion that inferred gaps exactly explain the length discrepancy.

ii.
```python
if n_missing <= 0:
    return me[:n_neural_frames]
...
while dst_idx < n_neural_frames:
    result[dst_idx] = result[dst_idx - 1]
    dst_idx += 1
...
n_bins = n // bin_size
trimmed = data[:n_bins * bin_size]
```

iii. The AI investigated the mismatches, concluded they represented dropped video frames, and chose interpolation. It observed that actual missing events were generally isolated single frames. The fallback truncation/padding behavior was not explicitly justified in the trajectory.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant computation is Suite2p maximin baseline preprocessing over every neuron and frame in every session. Loading large NumPy arrays and serializing the roughly 414 MB full dataset are also substantial I/O. Decoder training was time-consuming during development but is not part of the conversion script.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
...
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory repeatedly focused on baseline processing, reran the full conversion after neural-processing changes, and recorded the large output size. The final implementation explicitly forces Suite2p preprocessing onto CPU.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-source-frame and per-inserted-frame loops in missing-frame interpolation could be replaced with a vectorized timestamp/index interpolation. Trial construction could also use reshaping for neural/output blocks and a precomputed full-session time vector, though lists are ultimately required by the target format. The mouse/session loops necessarily handle separate files and differently shaped sessions.

ii.
```python
for src_idx in range(len(me)):
    ...
    for k in range(n_insert):
        ...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The AI did not explicitly discuss vectorization. Its preallocated interpolation array is nevertheless more efficient than repeated `np.insert`; the remaining nested loops were presumably accepted because dropped frames are rare and output must be organized as trial lists.

## 6-c. What processing does the code repeat multiple times?

i. File loading, neural preprocessing, motion correction, binning, percentile calculation, and trial assembly are repeated once per session. It also traverses session outputs again for summary statistics and again to construct the optional two-mouse sample dataset.

ii.
```python
for mouse_idx, mouse in enumerate(mice):
    ...
    for sess_name in sessions:
        ...
for i, mouse in enumerate(subjects):
    sess_mask = np.array(all_subject_idx) == i
...
for i in range(len(all_neural)):
    if all_subject_idx[i] in sample_mice:
        sample_sessions.append(i)
```

iii. The trajectory does not identify repeated work as an optimization concern. Per-session processing is required, while the later passes support validation reporting and the extra sample artifact.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes percentile edge arrays only to return them locally, print summary information, and then discard them from the saved dataset. It performs extensive summary scans and optionally duplicates data into `sample_data.pkl`, neither of which is required for downstream analysis. The detailed `session_info` metadata and output label strings are retained, not discarded.

ii.
```python
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
...
if sample_output_path:
    sample_data = {
        'neural': [all_neural[i] for i in sample_sessions],
        ...
    }
```

iii. The AI used summaries, plots, sample conversion, and decoder runs as validation aids. The trajectory says it deliberately created a two-mouse sample dataset for faster verification, although the original task only required the full converted pickle.
