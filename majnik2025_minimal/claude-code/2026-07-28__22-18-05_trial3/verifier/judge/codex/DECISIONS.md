# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all mouse directories whose names start with `jm`, sorts them, then iterates through all session subdirectories inside each mouse. For each session it loads Suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`, and behavioral arrays `motion_energy_glob.npy` and `interframe_int.npy` from `move_deve`. Trials are not loaded directly; they are created later by segmenting processed session-level arrays.

ii. 
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir)
                      if os.path.isdir(os.path.join(mouse_dir, d))])
    ...
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
    ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes describe the source data as six mice with Suite2p fluorescence traces plus motion energy from videography, and the trajectory says the agent interpreted the `jm*` folders as mice and the subdirectories as daily sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories whose names begin with `jm`, sorted lexicographically. The subject list is used directly as `subjects`, and each session gets the corresponding `mouse_idx`.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
...
subjects = mice
...
all_subject_idx.append(mouse_idx)
```

iii. The notes identify the dataset as six mice (`jm031` through `jm046`), and the trajectory shows the agent accepted the folder naming convention as the mouse split.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory under a mouse directory. Sessions are sorted by directory name and processed one by one. Each processed session becomes one element in `all_neural`, `all_input`, and `all_output`.

ii.
```python
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
...
for sess_name in sessions:
    sess_dir = os.path.join(mouse_dir, sess_name)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The notes call these longitudinal daily recordings and report 41 total sessions, consistent with this session-per-subdirectory interpretation.

## 1-d. How are the data split into trials?

i. The AI does not use the continuous session as-is. It first bins both neural and motion-energy data into 10-frame windows, then splits each binned session into consecutive non-overlapping 2-minute blocks. At 30 Hz and 10-frame bins, each trial has 360 time bins.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_S = 120
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360 bins per trial
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The trajectory explicitly says the agent chose 2-minute blocks because the paper mentioned “consecutive 2 minute blocks” for cross-validation, and the notes repeat that this was used as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement an explicit trial quality-control filter. Trials are created mechanically from each session, with only incomplete trailing data implicitly discarded because `n_trials` uses floor division after binning.

ii.
```python
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The notes describe validation and dataset statistics, but they do not describe any trial-level QC exclusion rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from the Suite2p fluorescence arrays `F.npy` and `Fneu.npy`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The notes explicitly say the neural data comes from Suite2p-processed fluorescence traces `F.npy` and `Fneu.npy`.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`F - 0.7 * Fneu`), then calls Suite2p’s `preprocess` function with a `maximin` baseline configuration on CPU. After that, it averages every 10 consecutive frames. The notes describe this as Suite2p-default “dF/F,” although the code actually uses `preprocess`, which returns baseline-corrected fluorescence rather than an explicit divide-by-baseline step.

ii.
```python
def compute_dff_suite2p(F, Fneu, fs=30.0, neucoeff=0.7,
                        win_baseline=60.0, sig_baseline=10.0):
    Fc = (F - neucoeff * Fneu).astype(np.float32)
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                     device=torch.device('cpu'))
    return dff

...
dff = compute_dff_suite2p(F, Fneu, fs=FS)
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The notes justify this by quoting the paper’s statement about using Suite2p default baseline-corrected traces, and the trajectory shows the agent decided to add 10-frame averaging because of a paper sentence about denoising traces by averaging 10 timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filter is applied in the script. The code never loads or checks `iscell.npy`; it simply processes every row of `F.npy` and `Fneu.npy`. The notes justify this by claiming all ROIs are already true cells and successfully tracked neurons.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))
```

iii. The notes say “No Neuron Filtering” and claim all ROIs already satisfy `iscell[:,0] == 1.0`, so the agent decided not to add any further QC gate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural trials to session start rather than to a behavioral event. Trials are contiguous blocks cut from the start of the session onward, and the metadata labels the alignment event as the start of the imaging session.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]

...
'metadata': {
    'temporal_alignment_event': 'Start of imaging session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The notes say the decoder input is time elapsed from session start, and the trajectory says the agent believed there was no stimulus event, so session start should be the alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI converts the session from 30 Hz to 3 Hz by averaging every 10 frames. The stored `time_bin_size` is therefore about 333.33 ms.

ii.
```python
BIN_SIZE = 10
FS = 30.0
time_bin_size_ms = (BIN_SIZE / FS) * 1000  # ~333.33 ms
...
dff_binned = bin_data(dff, BIN_SIZE)
...
'time_bin_size': time_bin_size_ms,
```

iii. The notes justify this using the paper sentence about averaging “10 consecutive timestamps,” and state that both neural and behavioral data were binned this way.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not read from any raw timestamp file. It is synthesized from bin indices plus the constants `BIN_SIZE` and `FS`.

ii.
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The notes say the decoder input is “Time elapsed from session start (seconds),” and the trajectory indicates the agent chose to compute this directly from frame/bin count.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the AI computes the center time of each 10-frame bin relative to session start. It uses `start:end` in binned coordinates, adds `0.5` to place times at bin centers, multiplies by `BIN_SIZE / FS`, and reshapes to `(1, n_timepoints)`.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    ...
    bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
    trial_input = bin_times.reshape(1, -1)
```

iii. The notes justify the choice only at a high level: time should be elapsed seconds from session start, and the trajectory shows the agent tied this to the 10-frame binning scheme.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The input time series is aligned to the same 2-minute binned trial boundaries used for neural data. Each trial uses the same `start:end` indices as `trial_neural`, so the time vector has one value per neural bin and advances continuously from session start.

ii.
```python
trial_neural = dff_binned[:, start:end]
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The notes say both streams were binned together, and the trajectory shows the agent’s intent was to keep the behavioral and neural data on the same 10-frame grid.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/interframe_int.npy` used to locate dropped video-frame gaps.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes explicitly identify motion energy as the behavioral signal and describe `interframe_int.npy` as the support array for missing-frame handling.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. In code, the AI interpolates missing frames using interframe intervals, bins the result by 10 frames, and discretizes it into percentile bins. In the notes, the AI also says the motion energy is normalized, but the script does not perform an explicit normalization step such as dividing by standard deviation before discretization.

ii.
```python
def interpolate_missing_frames(me, ifi, n_neural_frames):
    median_ifi = np.median(ifi)
    threshold = median_ifi * 1.5
    gap_indices = np.where(ifi > threshold)[0]
    missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
    ...
    return result[:n_neural_frames]

...
me = interpolate_missing_frames(me_raw, ifi, n_frames)
me_binned = bin_data(me, BIN_SIZE)
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. The notes justify these steps by citing the paper’s 10-frame averaging and by saying missing video frames should be interpolated. The trajectory shows the agent focused on synchronizing motion energy to neural frame counts and then turning it into five percentile bins.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes motion energy into five equal-percentile bins separately within each session, not globally across the whole dataset.

ii.
```python
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])
    return bins, edges

...
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. The notes say “Discretized into 5 equal-percentile bins (quintiles) per session,” and the final trajectory summary repeats that the output bins are per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first stretches motion energy to the neural frame count using gap interpolation, then bins both motion energy and neural traces by the same 10-frame rule, and finally slices the same trial boundaries from both arrays. Alignment is therefore by matched binned indices, not by an external event.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
dff = compute_dff_suite2p(F, Fneu, fs=FS)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
trial_neural = dff_binned[:, start:end]
trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The notes justify this as synchronous acquisition plus dropped-frame correction, and the trajectory shows the agent’s main concern was making the motion-energy array the same length as the neural recording before trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames heuristically. If motion energy is shorter than the neural recording, it detects gaps from unusually large interframe intervals, interpolates one or more values inside each gap, and pads with the last value if still short. If motion energy is longer, it truncates it. There is no final assertion that the repaired length exactly matches the expected length.

ii.
```python
if len(me) == n_neural_frames:
    return me

n_missing = n_neural_frames - len(me)
if n_missing <= 0:
    return me[:n_neural_frames]

median_ifi = np.median(ifi)
threshold = median_ifi * 1.5
gap_indices = np.where(ifi > threshold)[0]
missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
...
while dst_idx < n_neural_frames:
    result[dst_idx] = result[dst_idx - 1]
    dst_idx += 1

return result[:n_neural_frames]
```

iii. The notes justify this as “Missing video frames correctly identified via interframe interval analysis,” and the trajectory shows the agent believed missing frames were a known data issue that needed interpolation before alignment.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive part of the script is likely the Suite2p preprocessing of every session’s fluorescence traces, because it runs over all neurons and all timepoints before any trial split. Loading the large `.npy` arrays is the other obvious cost.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
...
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The notes do not provide a timing profile, but they emphasize using Suite2p’s actual preprocessing, and the code structure makes that the dominant computation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization target is the missing-frame interpolation logic, which uses an outer loop over every source frame and an inner loop over inserted frames. The per-trial assembly loop is also explicit and could be replaced with more array reshaping if the trial structure were fixed.

ii.
```python
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    dst_idx += 1
    if gap_pos < len(gap_indices) and src_idx == gap_indices[gap_pos]:
        n_insert = missing_per_gap[gap_pos]
        ...
        for k in range(n_insert):
            alpha = (k + 1) / (n_insert + 1)
            if dst_idx < n_neural_frames:
                result[dst_idx] = me[src_idx] * (1 - alpha) + me[src_idx + 1] * alpha
                dst_idx += 1

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    ...
    session_neural.append(trial_neural.astype(np.float32))
```

iii. There is no explicit written justification for keeping these loops; this is apparent from code inspection.

## 6-c. What processing does the code repeat multiple times?

i. The script repeats the same percentile-threshold computation separately for every session because discretization is done inside the session loop. It also rebuilds a fresh time vector for every trial even though the within-trial structure is fixed.

ii.
```python
for sess_name in sessions:
    ...
    me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
    ...
    for t in range(n_trials):
        ...
        bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
```

iii. The notes justify the per-session percentile choice directly, but they do not discuss the repeated computation cost.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The most obvious discarded computation is the percentile edge array `me_edges`, which is returned by `discretize_percentile_bins` and stored in a local variable but never used afterward. The summary-statistics pass at the end of the script is also only for logging, not for the saved dataset.

ii.
```python
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
...
total_neurons = []
for i, mouse in enumerate(subjects):
    sess_mask = np.array(all_subject_idx) == i
    sess_indices = np.where(sess_mask)[0]
    if len(sess_indices) > 0:
        nn = all_neural[sess_indices[0]][0].shape[0]
        total_neurons.append(nn)
```

iii. The notes discuss these summary statistics as validation, but they are not used by the downstream decoder input file itself.
