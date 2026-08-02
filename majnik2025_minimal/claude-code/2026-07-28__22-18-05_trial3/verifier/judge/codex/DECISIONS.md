# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all subject directories in `/app/data` whose names start with `jm`, then enumerates every subdirectory inside each subject as a session. For each session it loads neural fluorescence from `suite2p/plane0/F.npy` and `Fneu.npy`, and behavioral motion energy from `move_deve/motion_energy_glob.npy` plus `interframe_int.npy`. Trialization happens later, after neural and motion-energy preprocessing.

ii. <Code snippets>

```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir)
                      if os.path.isdir(os.path.join(mouse_dir, d))])
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset consists of 6 `jm*` mice with longitudinal sessions, with Suite2p fluorescence traces and session-level motion-energy files. The trajectory also shows the AI explicitly deciding to use all mice and all sessions found under that folder structure.

## 1-b. How are the data split into subjects?

i. Subjects are identified purely from directory names: each top-level directory in the data folder starting with `jm` is treated as one mouse, and the list is sorted.

ii. <Code snippets>

```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
subjects = mice
```

iii. The AI's notes state that the dataset contains 6 mice (`jm031`-`jm046`) and treats those folders as the subject definition.

## 1-c. How are the data split into sessions?

i. Each subdirectory inside a mouse directory is treated as one session, and sessions are processed in sorted order.

ii. <Code snippets>

```python
for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir)
                      if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. The justification in `CONVERSION_NOTES.md` is that the recordings are longitudinal daily sessions. The trajectory describes them as daily recordings nested under each mouse.

## 1-d. How are the data split into trials?

i. The AI defines artificial trials as consecutive non-overlapping 2-minute blocks after first averaging the data into 10-frame bins. With `FS = 30 Hz` and `BIN_SIZE = 10`, each trial contains `TRIAL_BINS = 360` time bins.

ii. <Code snippets>

```python
BIN_SIZE = 10
FS = 30.0
TRIAL_DURATION_S = 120
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)
```

```python
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
    trial_input = bin_times.reshape(1, -1)
    trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The AI justified this in both `CONVERSION_NOTES.md` and the trajectory by citing the paper's decoding-analysis language about "consecutive 2 minute blocks" and "averaging in bins of 10 consecutive timestamps".

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement an explicit trial-quality filter. It only keeps whole 2-minute trial blocks; any leftover bins at the end of a session are implicitly discarded because `n_trials` is computed with floor division.

ii. <Code snippets>

```python
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. No explicit quality-control justification for trial filtering was found. The only stated rationale was the choice to use 2-minute consecutive blocks as the trial definition.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p fluorescence arrays `F.npy` and `Fneu.npy`.

ii. <Code snippets>

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The notes identify these as the neural fluorescence and neuropil traces supplied by Suite2p.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`F - 0.7 * Fneu`), then runs Suite2p's `preprocess(..., 'maximin', ...)` baseline-correction routine, then averages the resulting signal in 10-frame bins.

ii. <Code snippets>

```python
Fc = (F - neucoeff * Fneu).astype(np.float32)
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
```

```python
dff = compute_dff_suite2p(F, Fneu, fs=FS)
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The AI's notes say it followed "Suite2p defaults" for baseline-corrected fluorescence and cite the paper's statement about averaging neural traces in bins of 10 timestamps for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI performs no neural-quality filtering in code. All rows of `F.npy` are included for every session, and every neuron is assigned to a single brain region index of zero.

ii. <Code snippets>

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
```

```python
all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))
```

iii. `CONVERSION_NOTES.md` says there is "No Neuron Filtering" because the dataset is already filtered/tracked and, according to the AI, all ROIs are effectively accepted already.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to the start of the imaging session. It does not use any event timestamps; instead, each trial is a contiguous block taken from the session-wide binned time series.

ii. <Code snippets>

```python
'metadata': {
    'temporal_alignment_event': 'Start of imaging session',
    'off_start': 0.0,
    'off_end': None,
```

```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
```

iii. The notes explicitly describe the decoder input as time elapsed from session start and the trajectory says there is no separate stimulus event, only continuous recording segmented into blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, giving a time bin size of `333.33 ms`. Yes: the AI rebins both neural and motion-energy data by averaging every 10 consecutive frames.

ii. <Code snippets>

```python
BIN_SIZE = 10
FS = 30.0
time_bin_size_ms = (BIN_SIZE / FS) * 1000
```

```python
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The AI repeatedly justified this using the paper's decoding-analysis sentence about "averaging in bins of 10 consecutive timestamps", including in `CONVERSION_NOTES.md` and the trajectory.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not derive the input time from a stored raw timestamp variable. It constructs time from bin indices, using the known frame rate and bin size.

ii. <Code snippets>

```python
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The trajectory states that the decoder input should be time elapsed from the beginning of the experiment, and the AI used synthetic time derived from the imaging cadence rather than any raw time file.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one time value per 10-frame bin, using the center of each bin: `(bin index + 0.5) * BIN_SIZE / FS`. Time is absolute from session start, not reset to zero within each trial.

ii. <Code snippets>

```python
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The notes describe the decoder input as "Time elapsed from session start (seconds), time-varying". The trajectory explicitly mentions using time of each bin center from session start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns the time input and neural data by slicing both with the same binned trial boundaries (`start:end` over 10-frame bins). Each trial therefore has one time vector with the same number of columns as the corresponding neural matrix.

ii. <Code snippets>

```python
trial_neural = dff_binned[:, start:end]
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The code structure itself is the justification: the same bin indices are used to slice both modalities. No separate justification beyond that was found.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy from `move_deve/motion_energy_glob.npy` and uses `move_deve/interframe_int.npy` to detect timing gaps that imply dropped video frames.

ii. <Code snippets>

```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. `CONVERSION_NOTES.md` describes `motion_energy_glob.npy` as precomputed motion energy from videography and says interframe intervals are used to handle missing video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. In code, the AI (1) interpolates missing video frames to match neural frame count, (2) averages motion energy in 10-frame bins, and (3) discretizes the binned values into 5 percentile bins. Despite the notes claiming normalization by standard deviation, the code does not perform any explicit normalization such as `me / me.std()`.

ii. <Code snippets>

```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
me_binned = bin_data(me, BIN_SIZE)
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

```python
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])
    return bins, edges
```

iii. The AI's written justification was that the paper binned behavior traces in 10-frame windows and that motion energy should be normalized and discretized into 5 equal-percentile bins. The trajectory and notes both say this was intended to match the decoding analysis.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes 5 equal-percentile thresholds separately for each session's binned motion-energy trace, then uses `np.digitize` to assign class labels `0` through `4`.

ii. <Code snippets>

```python
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

```python
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    bins = np.digitize(values, edges[1:-1])
    return bins, edges
```

iii. `CONVERSION_NOTES.md` says the output is discretized into "5 equal-percentile bins (quintiles) per session" so that each session has balanced class counts.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first expands motion energy to the neural frame count using interframe-interval gap detection and interpolation, then bins both modalities by the same 10-frame averaging rule, then slices them into trials with identical `start:end` bin ranges.

ii. <Code snippets>

```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
    trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The notes justify this by saying missing frames were interpolated and both neural and behavioral traces were binned identically to stay synchronized.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles short motion-energy traces by detecting gaps from large `interframe_int` values and inserting linearly interpolated samples. If motion energy is longer than neural data it truncates it; if interpolation still leaves the series short it pads with the final value. It does not implement an assertion that the repaired trace exactly reflects the observed gaps.

ii. <Code snippets>

```python
if len(me) == n_neural_frames:
    return me

n_missing = n_neural_frames - len(me)
if n_missing <= 0:
    return me[:n_neural_frames]
```

```python
threshold = median_ifi * 1.5
gap_indices = np.where(ifi > threshold)[0]
missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
```

```python
while dst_idx < n_neural_frames:
    result[dst_idx] = result[dst_idx - 1]
    dst_idx += 1
```

iii. The AI's justification in the trajectory and notes was that dropped camera frames must be interpolated to align video-derived motion energy with neural data. No explicit defense of the truncate-or-pad fallbacks was found.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work appears to be session-wise Suite2p preprocessing of all neurons (`preprocess`) and the repeated loading of large `.npy` arrays. The nested interpolation loop over missing frames is also a relatively slow pure-Python path compared with vectorized NumPy.

ii. <Code snippets>

```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
```

```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The trajectory focuses heavily on the Suite2p preprocessing step when debugging decoder performance, which is the clearest evidence of what the AI considered the major computational step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped-frame interpolation logic uses nested Python loops over source frames and over inserted frames, and the trialization logic loops over every trial to slice arrays into lists. Both could be vectorized or preallocated more aggressively.

ii. <Code snippets>

```python
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    dst_idx += 1
    if gap_pos < len(gap_indices) and src_idx == gap_indices[gap_pos]:
        n_insert = missing_per_gap[gap_pos]
        if src_idx + 1 < len(me):
            for k in range(n_insert):
                alpha = (k + 1) / (n_insert + 1)
                if dst_idx < n_neural_frames:
                    result[dst_idx] = me[src_idx] * (1 - alpha) + me[src_idx + 1] * alpha
                    dst_idx += 1
```

```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    ...
    session_neural.append(trial_neural.astype(np.float32))
```

iii. No explicit efficiency justification was found. This is inferred directly from the implementation.

## 6-c. What processing does the code repeat multiple times?

i. The code recomputes motion-energy percentile thresholds separately for every session instead of computing one global threshold set. It also repeatedly converts `all_subject_idx` to a NumPy array inside the summary loop.

ii. <Code snippets>

```python
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

```python
for i, mouse in enumerate(subjects):
    sess_mask = np.array(all_subject_idx) == i
    sess_indices = np.where(sess_mask)[0]
```

iii. The notes explicitly justify per-session percentile binning as a way to force balanced bins within each session. No explicit justification was given for the repeated summary conversions.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `me_edges` but never uses or saves it. It also creates extensive printed summary statistics and, when requested, a separate sample dataset, neither of which is needed by downstream decoding from the full converted dataset itself.

ii. <Code snippets>

```python
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

```python
print(f"Subjects: {subjects}")
print(f"Total sessions: {len(all_neural)}")
...
if sample_output_path:
    sample_data = {
        'neural': [all_neural[i] for i in sample_sessions],
        ...
    }
```

iii. No explicit justification was found. These are observable from the code structure.
