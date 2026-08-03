# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all data by scanning `/app/data` for mouse folders whose names start with `jm`, then scanning each mouse folder for session subdirectories. For each session it loads neural arrays from `suite2p/plane0/F.npy` and `Fneu.npy`, and behavioral arrays from `move_deve/motion_energy_glob.npy` and `interframe_int.npy`. It then converts each continuous session into multiple trial blocks.

ii. 
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir)
                      if os.path.isdir(os.path.join(mouse_dir, d))])

    for sess_name in sessions:
        sess_dir = os.path.join(mouse_dir, sess_name)
        s2p_dir = os.path.join(sess_dir, 'suite2p', 'plane0')
        move_dir = os.path.join(sess_dir, 'move_deve')

        F = np.load(os.path.join(s2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
        me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
        ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The justification in `CONVERSION_NOTES.md` is that the released dataset is organized as six mouse folders, each containing daily session folders with `suite2p` neural outputs and `move_deve` motion-energy outputs. The trajectory shows the agent inspected that folder structure and adopted it directly.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level folder: each `jm*` directory becomes one subject. The subject list is the sorted mouse-folder list, and each converted session stores its corresponding mouse index in `subject_idx`.

ii. 
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

subjects = mice

for mouse_idx, mouse in enumerate(mice):
    ...
    all_subject_idx.append(mouse_idx)
```

iii. The data README says each subject has its own folder and that `jm031` through `jm046` correspond to mice A through F. The trajectory shows the agent relied on that README note when mapping subjects.

## 1-c. How are the data split into sessions?

i. Sessions are split by directory: each session is one dated subdirectory inside a mouse folder. The agent sorts those subdirectories chronologically and treats each one as a separate session in the output lists.

ii. 
```python
mouse_dir = os.path.join(data_dir, mouse)
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])

for sess_name in sessions:
    sess_dir = os.path.join(mouse_dir, sess_name)
```

iii. The README says each session subfolder corresponds to one recording day and the names are date strings. The trajectory shows the agent inspected real examples like `2023-10-18_a` and decided chronological sort was the correct session order.

## 1-d. How are the data split into trials?

i. The raw recordings are continuous, so the agent creates artificial trials by splitting each session into consecutive, non-overlapping 2-minute blocks after temporal binning. Each block contains 360 binned time steps.

ii. 
```python
BIN_SIZE = 10
FS = 30.0
TRIAL_DURATION_S = 120
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360 bins per trial

n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. In trajectory step 23 the agent justified this by citing the paper’s decoder cross-validation procedure: “splits were done on consecutive 2 minute blocks,” and concluded that each 2-minute block could be treated as a trial to satisfy the target format.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-level quality-control filter. Trials are kept as long as they fit into the consecutive 2-minute blocking scheme. Any incomplete trailing data would be dropped implicitly by integer division during binning or trial counting. Missing camera frames are handled before trialization by motion-energy interpolation.

ii. 
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)

n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
```

iii. The agent’s notes explicitly say “No Neuron Filtering,” but say nothing about trial exclusion. The trajectory indicates the agent believed the release was already curated enough for decoder use and only repaired missing video frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `F.npy` and `Fneu.npy` in each session’s `suite2p/plane0` directory. The agent does not use `spks.npy`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff_suite2p(F, Fneu, fs=FS)
```

iii. In `CONVERSION_NOTES.md` the agent says the paper used baseline-corrected fluorescence traces and therefore chose the Suite2p fluorescence arrays rather than deconvolved spikes. The trajectory shows the agent considered `spks.npy` but decided the paper pointed to fluorescence-based traces.

## 2-b. How is the `neural` data processed?

i. The agent applies Suite2p-style neuropil subtraction and baseline correction. It computes `Fc = F - 0.7 * Fneu`, then passes `Fc` into `suite2p.extraction.dcnv.preprocess(..., 'maximin', ...)`, which returns baseline-subtracted traces. Those traces are then averaged in bins of 10 frames.

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

iii. The notes justify this as matching the paper statement that “baseline corrected fluorescence traces” with default Suite2p parameters were used. The trajectory shows the agent originally considered dividing by baseline, then changed course after inspecting Suite2p’s `preprocess` docstring and deciding to use baseline-subtracted output directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural filtering is applied inside `convert_data.py`. The agent assumes the released Track2p data already contain only tracked neurons that passed Suite2p’s cell criterion, and it ignores `iscell.npy` during conversion.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))
```

iii. `CONVERSION_NOTES.md` says “All ROIs in the dataset are already filtered” and cites the paper’s `iscell > 0.5` rule. The trajectory records that the agent inspected `iscell.npy`, saw that all rows had `iscell[:,0] == 1`, and concluded no extra filtering was necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of the imaging session. The agent does not align to any within-session behavioral event; instead it slices consecutive binned data blocks and marks the metadata event as session start.

ii. 
```python
trial_neural = dff_binned[:, start:end]
...
'metadata': {
    ...
    'temporal_alignment_event': 'Start of imaging session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The justification is implicit in the task framing and explicit in the notes: the decoder input is “Time elapsed from session start,” and the source data are spontaneous continuous recordings rather than trial-based event-locked experiments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at a nominal imaging rate of 30 Hz, for a time bin size of about 333.33 ms. Yes, temporal rebinning is applied by averaging consecutive groups of 10 frames.

ii. 
```python
BIN_SIZE = 10
FS = 30.0
time_bin_size_ms = (BIN_SIZE / FS) * 1000

def bin_data(data, bin_size):
    ...
    return trimmed.reshape(..., bin_size).mean(axis=...)
```

iii. The justification appears in both step 23 of the trajectory and `CONVERSION_NOTES.md`: the agent cites the paper sentence saying the dF/F and behavior traces were denoised by averaging “10 consecutive timestamps.”

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not derived from a dedicated raw timestamp variable. It is synthesized from trial-bin indices plus the assumed constant imaging rate `FS = 30.0` and bin size `BIN_SIZE = 10`.

ii. 
```python
FS = 30.0
BIN_SIZE = 10
...
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. In trajectory step 23 the agent debated whether to use timestamps, noted confusion about `tstamps.npy`, and chose a simpler absolute-time-from-session-start construction based on the nominal frame rate instead.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each 2-minute block, the agent computes the center time of every 10-frame bin using a uniform grid. It adds `0.5` bin to move from the left bin edge to the bin center, multiplies by `10/30` seconds, and reshapes the result to shape `(1, n_timepoints)`.

ii. 
```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)  # (1, n_timepoints)
```

iii. The trajectory explains the rationale: absolute time from session start “makes more sense for the decoder” than within-trial relative time because it preserves temporal context over the whole recording.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The input time series is aligned by construction to the same binned trial windows as the neural data. For each artificial trial, the code uses the same `start:end` indices for `trial_neural` and for the corresponding time vector.

ii. 
```python
trial_neural = dff_binned[:, start:end]
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The justification in the trajectory is that once all streams are rebinned to the same 10-frame grid, index-based slicing is enough to keep them synchronized.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `move_deve/motion_energy_glob.npy`, with `interframe_int.npy` used to locate missing camera frames for interpolation. The code does not derive motion energy from raw videos.

ii. 
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
...
me = interpolate_missing_frames(me_raw, ifi, n_frames)
```

iii. The README says `motion_energy_glob.npy` already contains processed behavioral data and that missing-frame indices can be obtained from `tstamps.npy` or `interframe_int.npy`. The agent followed that released-data interface.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code interpolates dropped camera frames when needed, then averages motion energy in 10-frame bins. After binning it discretizes the values into five percentile bins. There is no separate explicit normalization step in the implementation, even though the notes describe the output as “normalized and discretized.”

ii. 
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
me_binned = bin_data(me, BIN_SIZE)
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. In trajectory step 23 the agent reasoned that percentile binning is rank-based, so a prior normalization step would not change category assignments. That is the main justification for omitting separate normalization in code.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Thresholding is done per session, not globally. The agent computes the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of the session’s binned motion-energy values, slightly bumps the top edge to avoid ties at the maximum, and uses `np.digitize` to assign class labels `0` through `4`.

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

iii. The notes justify this as producing five equal-percentile bins per session. The trajectory also shows the agent wanted exactly balanced categories and treated percentile binning as the natural way to get them.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by first interpolating it to the neural frame count, then binning it with the same 10-frame binning used for neural traces, and finally slicing the same `start:end` trial windows.

ii. 
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The justification comes from the data README, which explicitly allows missing camera frames to be interpolated over, and from the paper/trajectory note that videography was triggered by the microscope at 30 Hz.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling policy is for missing camera frames. If motion energy is shorter than the neural recording, the code uses `interframe_int.npy` to infer gap locations and fills missing samples by linear interpolation. If the motion-energy array is too long it truncates it; if interpolation still leaves the tail short it pads with the last value. Any remainder that does not fill a complete 10-frame bin or a full 2-minute trial is silently discarded by floor division.

ii. 
```python
if len(me) == n_neural_frames:
    return me

n_missing = n_neural_frames - len(me)
if n_missing <= 0:
    return me[:n_neural_frames]

gap_indices = np.where(ifi > threshold)[0]
...
while dst_idx < n_neural_frames:
    result[dst_idx] = result[dst_idx - 1]
    dst_idx += 1

n_bins = n_time // bin_size
trimmed = data[:, :n_bins * bin_size]

n_trials = n_total_bins // TRIAL_BINS
```

iii. The README explicitly says missing video frames can be treated as missing or interpolated over. The trajectory shows the agent chose interpolation because it wanted frame counts to match before binning and trialization.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive steps are session-scale array loading, Suite2p preprocessing of the full fluorescence matrices, and the Python-loop interpolation path for motion energy when frames are missing. The sample-dataset creation and summary printing are small by comparison.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff_suite2p(F, Fneu, fs=FS)
...
for src_idx in range(len(me)):
    ...
    for k in range(n_insert):
        ...
```

iii. The agent did not document a performance analysis, but the code structure makes these the dominant costs because they touch the largest arrays and, in the interpolation function, do so in Python loops.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The interpolation loop is the clearest vectorization target: it iterates over every motion-energy frame and every inserted missing sample in Python. The loop that builds trial lists one trial at a time could also be reduced by reshaping or pre-splitting already binned arrays.

ii. 
```python
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    ...
    for k in range(n_insert):
        ...

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
    ...
```

iii. There is no explicit justification in the notes. This is an implementation consequence of the agent choosing straightforward imperative code over a more vectorized formulation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the same load-process-bin-split pipeline independently for every session. It also repeatedly converts `all_subject_idx` to a NumPy array when printing summary statistics, and it rebuilds a second sample dataset after already building the full dataset in memory.

ii. 
```python
for mouse_idx, mouse in enumerate(mice):
    ...
    for sess_name in sessions:
        ...
        me = interpolate_missing_frames(me_raw, ifi, n_frames)
        dff = compute_dff_suite2p(F, Fneu, fs=FS)
        dff_binned = bin_data(dff, BIN_SIZE)
        me_binned = bin_data(me, BIN_SIZE)
        ...

print(f"Sessions per subject: {[np.sum(np.array(all_subject_idx) == i) for i in range(len(subjects))]}")

for i in range(len(all_neural)):
    if all_subject_idx[i] in sample_mice:
        sample_sessions.append(i)
```

iii. The trajectory and notes do not justify these repetitions; they arise from a direct, per-session implementation and from producing both full and sample outputs in one run.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and returns `me_edges` from percentile binning but never uses the edges afterward. It also loads `interframe_int.npy` for every session even when the motion-energy length already matches, and it spends time building summary statistics and a sample dataset that are not part of the main converted dataset used by downstream decoding.

ii. 
```python
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)

ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))

print(f"Mean neurons per mouse: {np.mean(total_neurons):.0f} (± {np.std(total_neurons):.0f} std)")

if sample_output_path:
    ...
    with open(sample_output_path, 'wb') as f:
        pickle.dump(sample_data, f)
```

iii. The agent’s notes justify the sample dataset and validation outputs as deliverables, but they are still extra work relative to the single `converted_data.pkl` object used for downstream analysis.
