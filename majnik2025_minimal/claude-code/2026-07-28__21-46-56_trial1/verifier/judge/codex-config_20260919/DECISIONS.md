# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes six mice, iterates over their date-prefixed session directories, and loads `F.npy`, `Fneu.npy`, `motion_energy_glob.npy`, and `interframe_int.npy` for every session. It processes the complete arrays before splitting them into trials.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = get_sessions(mouse_dir)
    for sess_name in sessions:
        F = np.load(os.path.join(suite2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
        me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
        ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The trajectory says the agent inspected all mouse/session dimensions and concluded that the dataset contains six known mice and 41 daily recordings. It chose the observed array lengths rather than assuming every recording had the paper's stated 20-minute duration.

## 1-b. How are the data split into subjects?

i. Subjects are the six hard-coded mouse IDs. Each daily recording receives the index of its mouse in this list.

ii.
```python
subjects = MICE[:]
for mouse_i, mouse in enumerate(MICE):
    ...
    subject_idx.append(mouse_i)
```

iii. The agent found these six mouse directories while exploring the data and noted that this agrees with the paper's six-mouse dataset.

## 1-c. How are the data split into sessions?

i. A session is each sorted subdirectory whose first character is `2` within a mouse directory; each becomes one element of the outer session lists.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0] == '2']
    sessions.sort()
    return sessions
```

iii. The agent observed 6–7 date-named daily session directories per mouse and treated each as a distinct recording session.

## 1-d. How are the data split into trials?

i. The agent splits each session into consecutive, non-overlapping 120-second blocks, or 360 samples after 10-frame averaging. Incomplete tails are silently omitted by integer division/slicing.

ii.
```python
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FS / BIN_SIZE)
n_trials = n_bins // TRIAL_BINS
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
```

iii. The trajectory explicitly says the agent followed the paper's use of consecutive two-minute cross-validation blocks. It overlooked the task's direct instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-quality filter is applied. Every complete 120-second block is retained; only an incomplete session tail is dropped.

ii.
```python
n_trials = data.shape[1] // trial_length
return [data[:, i*trial_length:(i+1)*trial_length] for i in range(n_trials)]
```

iii. The trajectory does not identify any trial-level quality criterion in the paper or data, so the agent retained all complete blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(suite2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. The agent interpreted the paper's “baseline corrected fluorescence traces” as requiring raw fluorescence plus neuropil fluorescence and Suite2p-style correction.

## 2-b. How is the `neural` data processed?

i. The agent subtracts `0.7 * Fneu`, smooths with a Gaussian whose sigma is `win/6`, applies 60-second minimum and maximum filters, divides the baseline-subtracted trace by the estimated baseline, averages every 10 frames, and finally casts trials to `float32`.

ii.
```python
Fc = F - NEUCOEFF * Fneu
win = int(WIN_BASELINE_SEC * FS)
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
F0 = maximum_filter1d(Flow, size=win, axis=1)
dff = (Fc - F0) / F0_safe
dff_binned = bin_array(dff, BIN_SIZE)
```

iii. The trajectory states that this was intended to reproduce Suite2p's default maximin baseline and conventional dF/F. The agent chose an in-house SciPy approximation after reading that the paper used default Suite2p processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed; all rows in `F.npy` are kept.

ii.
```python
n_neurons, n_frames = F.shape
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The trajectory says Track2p had already matched/curated the cells and that all available ROIs were valid, so further `iscell` filtering was considered unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external event alignment. Trials are contiguous blocks anchored to recording/session start, and metadata names the start of the recording session as the alignment event.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. The agent treated this as a continuous spontaneous-behavior recording without stimulus events, making session start the natural temporal origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten consecutive 30 Hz frames are averaged into each sample, producing 3 Hz data and 333.33 ms bins. The same operation is applied to neural activity and motion energy.

ii.
```python
BIN_SIZE = 10
FS = 30
BIN_DURATION_MS = (BIN_SIZE / FS) * 1000
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The trajectory quotes the paper's decoding method of denoising dF/F and behavior by averaging 10 consecutive timestamps and deliberately applies it before categorizing behavior.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from trial number, the fixed 120-second trial duration, and the 333.33 ms bin duration; no raw timestamp array is used.

ii.
```python
start_sec = t_i * TRIAL_DURATION_SEC
time_bins = np.linspace(
    start_sec + BIN_DURATION_MS / 2000,
    start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
    TRIAL_BINS
)
```

iii. The agent found the stored timestamps to have unclear units and relied on the confirmed constant 30 Hz Suite2p frame rate instead.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. It creates evenly spaced bin-center times in seconds, continuous across trials within a session, reshapes them to `(1, time)`, and casts to `float32`.

ii.
```python
time_bins = np.linspace(start_sec + BIN_DURATION_MS / 2000,
                        start_sec + TRIAL_DURATION_SEC - BIN_DURATION_MS / 2000,
                        TRIAL_BINS)
input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The agent chose bin centers (for example 0.17, 0.50, …) as the representative time of each averaged sample.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Every neural trial and time trial has the same 360 positions; time position `j` is the center of the same 10-frame group used for neural position `j`.

ii.
```python
neural_trials = split_into_trials(dff_binned, TRIAL_BINS)
input_trials.append(time_bins.reshape(1, -1).astype(np.float32))
```

iii. The agent generated time directly from the known bin and trial indices to ensure one-to-one positional alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output comes from the precomputed global motion-energy trace `motion_energy_glob.npy`; `interframe_int.npy` is additionally used to locate dropped camera frames.

ii.
```python
me_raw = np.load(os.path.join(move_dir, 'motion_energy_glob.npy')).astype(np.float64)
ifi = np.load(os.path.join(move_dir, 'interframe_int.npy'))
```

iii. The agent identified the global motion-energy file as the paper's behavioral measure and the interframe intervals as necessary alignment information.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing positions are inserted into a neural-length array and linearly interpolated, the aligned signal is averaged in 10-frame bins, complete 120-second trials are retained, and a global set of quintile boundaries is applied to every session.

ii.
```python
me_aligned = align_motion_energy(me_raw, ifi, n_frames)
me_binned = bin_array(me_aligned, BIN_SIZE)
me_trials = split_into_trials(me_binned, TRIAL_BINS)
all_me_concat = np.concatenate(all_me_values)
bin_edges = np.percentile(all_me_concat, np.linspace(0, 100, 6))
```

iii. The agent reasoned that interpolation preserves frame alignment and that global percentiles guarantee exactly balanced classes over the full dataset. It acknowledged considering per-session normalization but ultimately selected global boundaries.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Quintile edges (0, 20, 40, 60, 80, 100 percentiles) are calculated once across all retained binned samples from all mice and sessions. The four internal edges are passed to `np.digitize`, yielding integer categories 0–4.

ii.
```python
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
bin_edges = np.percentile(all_me_concat, percentiles)
binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The trajectory says the global choice was made to ensure the five classes each occupy 20% of the complete dataset; validation confirmed global balance, despite skew within individual sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When lengths differ, interframe intervals relative to their median determine how many positions to skip in a neural-length array. Observed motion values are placed sequentially and all remaining NaNs are linearly interpolated. Neural and motion arrays are then binned and split with identical boundaries.

ii.
```python
n_dropped = int(np.round(interframe_int[i] / median_ifi - 1))
neural_idx += n_dropped
...
aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
dff_binned = bin_array(dff, BIN_SIZE)
me_binned = bin_array(me_aligned, BIN_SIZE)
```

iii. The agent tested the interval-gap method on a mismatched session and found the inferred number of dropped frames equaled the neural/video length difference, supporting framewise interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Shorter motion-energy streams are expanded to the neural length and missing values are linearly interpolated, including leading/trailing gaps through `np.interp`. Near-zero fluorescence baselines are replaced by `1e-6` before division. Partial temporal bins and partial trials are discarded. The code reports neural NaN/Inf counts but does not fail on them.

ii.
```python
aligned = np.full(n_neural_frames, np.nan)
aligned[nans] = np.interp(x[nans], x[~nans], aligned[~nans])
F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
n_trials = len(data) // trial_length
```

iii. The trajectory calls linear interpolation appropriate for typically small dropped-frame gaps, while acknowledging that two larger gaps may introduce artifacts. It also favored preserving common stream lengths and verified the result contained no NaN/Inf trials.

## 6-a. What are the most time-consuming steps of the code?

i. Full-session fluorescence baseline processing (Gaussian, minimum, and maximum filters for every neuron) is the dominant computation. Re-running it over all 41 sessions made even the nominal sample-only mode slow; decoder training is also lengthy but is outside conversion itself.

ii.
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=win / 6, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
F0 = maximum_filter1d(Flow, size=win, axis=1)
```

iii. The trajectory explicitly observed that dF/F processing made `--sample-only` take essentially as long as full conversion because all sessions were still processed.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-frame dropped-frame placement loop could be replaced with vectorized cumulative indices. Trial slicing, time construction, output digitization, float conversion, sanity checks, and nested concatenation also use Python loops/list comprehensions that could be batched or reshaped.

ii.
```python
for i in range(len(me)):
    aligned[neural_idx] = me[i]
    ...
for session_me_trials in output_all_raw:
    for me_trial in session_me_trials:
        binned = np.digitize(me_trial, bin_edges[1:-1])
```

iii. The trajectory did not provide a formal vectorization analysis, although it recognized preprocessing as costly and later described the script as optimized. The listed opportunities follow directly from the final code.

## 6-c. What processing does the code repeat multiple times?

i. Session directories are scanned during conversion and again while building metadata. Motion trials are first stored, then concatenated for percentiles, traversed again for digitization, concatenated again for distribution checks, and traversed again for sample creation/float conversion. Full neural preprocessing is repeated even in sample-only mode.

ii.
```python
sessions = get_sessions(mouse_dir)
...
'sessions': get_sessions(os.path.join(data_dir, mouse)),
...
all_me_concat = np.concatenate(all_me_values)
for session_me_trials in output_all_raw:
```

iii. The trajectory specifically noticed the redundant full-data preprocessing in `--sample-only`, killed that rerun, and reused output from the already completed full conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes extensive console diagnostics, neuron-count summaries, global distribution checks, and a separate two-session sample dataset; these do not enter the requested full dataset or downstream decoder. It also retains raw motion trials until after global thresholds are computed and computes all sessions even when `sample_only=True`.

ii.
```python
all_binned = np.concatenate([np.concatenate([t.flatten() for t in s]) for s in output_all])
mean_neurons = np.mean(all_neuron_counts)
std_neurons = np.std(all_neuron_counts, ddof=1)
sample_data = {'neural': data['neural'][:n_sample], ...}
```

iii. The agent used these operations for sanity checking and convenience. The trajectory confirms that sample-only's full preprocessing was recognized as redundant only after it was launched.
