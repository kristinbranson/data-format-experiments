# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads subject folders by scanning the data directory for names starting with `jm`, then scans each subject directory for session subdirectories. For each session it loads calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy`, and motion energy data from `move_deve/motion_energy_glob.npy` plus `interframe_int.npy` for dropped-frame handling. Trials are not loaded directly from disk; they are created later by splitting each session into fixed-length blocks.

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
...
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. In the trajectory, the AI justified this by inspecting `/app/data` and concluding that all mice are stored in `jm*` folders and each session folder contains `suite2p` and `move_deve` subdirectories. It explicitly noted that `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `interframe_int.npy` were the relevant raw inputs.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the top-level directories in the data folder whose names start with `jm`, sorted lexicographically. The resulting subject list is reused directly as `subjects`.

ii.
```python
mice = sorted([d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
...
subjects = mice
```

iii. The trajectory shows the AI examined `/app/data` and adopted the existing naming convention, treating each `jm*` directory as one mouse.

## 1-c. How are the data split into sessions?

i. Each session is a subdirectory inside a mouse directory, again sorted lexicographically. Each such directory becomes one session in the converted dataset.

ii.
```python
mouse_dir = os.path.join(data_dir, mouse)
sessions = sorted([d for d in os.listdir(mouse_dir)
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. The AI justified this from direct inspection of the data layout: each dated subdirectory under a mouse contains one imaging session with matching neural and motion files.

## 1-d. How are the data split into trials?

i. The AI does not use 60-second trials. It splits each session into consecutive non-overlapping 2-minute blocks, after 10-frame temporal binning. With `FS=30 Hz` and `BIN_SIZE=10`, each trial contains `120 * 30 / 10 = 360` bins. Partial trailing data are implicitly dropped because `n_trials = n_total_bins // TRIAL_BINS`.

ii.
```python
TRIAL_DURATION_S = 120  # 2-minute blocks
TRIAL_BINS = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360 bins per trial
...
n_total_bins = dff_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. In the trajectory, the AI explicitly justified this by citing the paper’s cross-validation language about “consecutive 2 minute blocks” and decided to reuse those blocks as trials, even though the task instruction asked for 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Every fixed-length trial produced by the session split is kept.

ii.
```python
n_trials = n_total_bins // TRIAL_BINS
...
for t in range(n_trials):
    ...
    session_neural.append(trial_neural.astype(np.float32))
    session_input.append(trial_input.astype(np.float32))
    session_output.append(trial_output.astype(np.int64))
```

iii. The trajectory does not show any separate trial QC decision beyond handling missing motion-energy frames before binning.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the Suite2p fluorescence arrays `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff_suite2p(F, Fneu, fs=FS)
```

iii. The AI justified this from the paper and Suite2p outputs, stating that the paper used baseline-corrected fluorescence traces derived from raw fluorescence and neuropil fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction with coefficient `0.7`, then calls Suite2p’s `preprocess` function with the `maximin` baseline method, `win_baseline=60.0`, `sig_baseline=10.0`, and `fs=30`. The returned baseline-subtracted trace is then averaged into non-overlapping 10-frame bins.

ii.
```python
def compute_dff_suite2p(F, Fneu, fs=30.0, neucoeff=0.7,
                        win_baseline=60.0, sig_baseline=10.0):
    Fc = (F - neucoeff * Fneu).astype(np.float32)
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                     device=torch.device('cpu'))
    return dff
...
dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The trajectory shows an extended justification process. The AI first tried its own dF/F implementations, checked Suite2p source code, then concluded that the paper’s “baseline corrected fluorescence traces as our dF/F” meant using Suite2p’s baseline-subtracted output directly. It switched to `preprocess` after decoder performance improved.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural QC filter is applied in the conversion script. All rows present in `F.npy` are kept; `iscell.npy` is not used.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=int))
```

iii. In the trajectory, the AI noted that inspected `iscell` entries were already all marked as true cells and inferred that the released Track2p dataset already contains tracked neurons, so it did not add more neuron filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to session start rather than to any stimulus or behavioral event. Each trial is just a contiguous block of the session timeline. The metadata describes the alignment event as the start of the imaging session.

ii.
```python
trial_neural = dff_binned[:, start:end]
...
'metadata': {
    ...
    'temporal_alignment_event': 'Start of imaging session',
    'off_start': 0.0,
    'off_end': None,
```

iii. The trajectory justifies this implicitly: the AI repeatedly described the recording as continuous spontaneous behavior data with no natural event to align to, so it used session start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and motion-energy streams by averaging non-overlapping windows of 10 raw frames. At 30 Hz this yields 3 Hz data, or `333.33 ms` per bin.

ii.
```python
BIN_SIZE = 10
FS = 30.0
time_bin_size_ms = (BIN_SIZE / FS) * 1000
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. The trajectory explicitly cites the paper’s statement about averaging “10 consecutive timestamps” and uses that as the reason for the 10-frame rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a raw timestamp file. It is synthesized from the binned time index, the imaging rate `FS`, and the bin size `BIN_SIZE`.

ii.
```python
FS = 30.0
BIN_SIZE = 10
...
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. In the trajectory, the AI reasoned that constant-rate imaging makes time recoverable from frame/bin index, so an explicit timestamp array is unnecessary for the decoder input.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the AI computes one scalar time per bin using the bin centers, not the left edges. The value is `(global_binned_index + 0.5) * (10 / 30)` seconds, so time runs continuously through the session and is sliced into the current trial.

ii.
```python
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS
...
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The only explicit justification in the code is the comment “Time of each bin center from session start.” The trajectory did not separately defend the half-bin center choice.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time one-to-one with the binned neural samples by using the same `start:end` indices used to slice each neural trial. The time vector has one entry per neural time bin in the same trial, but it represents bin centers.

ii.
```python
trial_neural = dff_binned[:, start:end]
...
bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
trial_input = bin_times.reshape(1, -1)
```

iii. The trajectory does not contain a separate alignment argument beyond the general decision to use session time and fixed trial blocks.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `move_deve/motion_energy_glob.npy`. The script also uses `move_deve/interframe_int.npy` to detect dropped video frames before alignment.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(float)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The trajectory shows the AI inspected the session contents, identified `motion_energy_glob.npy` as the behavioral signal, and used `interframe_int.npy` after finding sessions where motion-energy length was shorter than the neural frame count.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The final code interpolates missing motion-energy frames to match neural length, averages the repaired trace into 10-frame bins, and then discretizes the binned values. The top-level docstring and trajectory mention “normalizing” motion energy, but no separate normalization step exists in the final script.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
...
me_binned = bin_data(me, BIN_SIZE)
...
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
```

iii. In the trajectory, the AI justified interpolation after studying `tstamps` and `interframe_int` and concluding that missing video frames should be repaired before alignment. It also justified 10-frame averaging from the paper’s “10 consecutive timestamps” language, and described percentile-based normalization/discretization as matching the decoder requirement.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes each session independently into 5 equal-percentile bins using `np.percentile` followed by `np.digitize`. It nudges the final edge slightly upward to avoid a non-increasing last boundary when there are ties.

ii.
```python
def discretize_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] = edges[-1] + 1e-10
    bins = np.digitize(values, edges[1:-1])
    return bins, edges
```

iii. The trajectory repeatedly says the decoder output should be “5 equal-percentile bins” and treats session-wise percentile binning as the intended discretization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first repairs motion energy to the neural frame count, then bins motion energy and neural data with the same 10-frame averaging, and finally slices both with identical `start:end` trial indices. This makes the output one sample per neural time bin.

ii.
```python
me = interpolate_missing_frames(me_raw, ifi, n_frames)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
...
trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. The trajectory shows the AI examining mismatched session lengths, inferring that dropped camera frames caused the mismatch, and deciding that interpolation plus common binning would restore framewise alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling logic is for missing motion-energy frames. If the motion trace is shorter than the neural trace, the AI detects large interframe gaps, interpolates missing frames, and if necessary pads the remainder with the last available value. If the motion trace is longer, it truncates it. There is no assertion that the repaired signal matches the expected length for a specific number of inserted gaps.

ii.
```python
if len(me) == n_neural_frames:
    return me

n_missing = n_neural_frames - len(me)
if n_missing <= 0:
    return me[:n_neural_frames]
...
gap_indices = np.where(ifi > threshold)[0]
missing_per_gap = np.round(ifi[gap_indices] / median_ifi).astype(int) - 1
...
while dst_idx < n_neural_frames:
    result[dst_idx] = result[dst_idx - 1]
    dst_idx += 1
```

iii. The trajectory shows the AI inspecting sessions with missing motion frames, observing that large interframe gaps exactly matched the missing-frame counts in one test case, and deciding interpolation was the right repair strategy.

## 6-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are Suite2p preprocessing on full fluorescence matrices and the per-frame interpolation logic for motion energy, with file I/O also nontrivial because every session loads large `.npy` arrays.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs,
                 device=torch.device('cpu'))
...
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    ...
    for k in range(n_insert):
        ...
```

iii. The trajectory spent most debugging effort on matching Suite2p preprocessing, which indicates the AI viewed that neural preprocessing as the dominant substantive computation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested loops in `interpolate_missing_frames` are the clearest vectorization target. The code copies one source frame at a time and inserts missing samples with an inner Python loop. Trial assembly is also loop-based, but the interpolation loop is the most avoidable Python overhead.

ii.
```python
for src_idx in range(len(me)):
    result[dst_idx] = me[src_idx]
    dst_idx += 1
    ...
    for k in range(n_insert):
        alpha = (k + 1) / (n_insert + 1)
        if dst_idx < n_neural_frames:
            result[dst_idx] = me[src_idx] * (1 - alpha) + me[src_idx + 1] * alpha
            dst_idx += 1
```

iii. The trajectory does not discuss vectorization directly, but this follows from the final implementation choices.

## 6-c. What processing does the code repeat multiple times?

i. The final script does not contain much avoidable repeated processing. It applies the same load/preprocess/bin/split steps once per session, which is necessary. The only mild repetition is that similar trial slicing logic is repeated for neural, input, and output arrays inside the same trial loop.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    trial_neural = dff_binned[:, start:end]
    bin_times = (np.arange(start, end) + 0.5) * (BIN_SIZE / FS)
    trial_output = me_discrete[start:end].reshape(1, -1)
```

iii. There is no explicit trajectory discussion of repeated processing in the final script.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes `me_edges` from percentile binning but never stores or uses them in the output dataset. It also builds and optionally saves a `sample_data.pkl` subset and prints summary statistics, none of which are needed for the final converted dataset used by the decoder. The richer metadata fields are also beyond the strict minimum required by the reference solution.

ii.
```python
me_discrete, me_edges = discretize_percentile_bins(me_binned, N_OUTPUT_BINS)
...
if sample_output_path:
    sample_data = {
        'neural': [all_neural[i] for i in sample_sessions],
        ...
    }
...
print("SUMMARY")
```

iii. The trajectory shows these extras were added for verification and convenience after the main conversion was already working, not because downstream analysis required them.
