# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six mouse IDs, walks each mouse directory, selects sorted subdirectories whose names begin with a digit, and loads `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy` for every selected recording. Full mode processes all such sessions; sample mode processes the first two mice and first two sessions per mouse.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions

F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The notes report that exploration found six mice and 41 daily recordings in this directory layout. The agent considered the supplied data already tracked and cell-filtered and chose to use every available recording, including the 30-minute recordings that exceed the paper's stated 20 minutes.

## 1-b. How are the data split into subjects?

i. Each hard-coded mouse directory is one subject. Its position in `MICE` becomes the integer `subject_idx` for all of that mouse's sessions.

ii.
```python
for mouse_idx, mouse in enumerate(mice_to_process):
    ...
    all_subject_idx.append(mouse_idx)

'subjects': mice_to_process,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The agent states that the six `jm*` folders are the six mice in the paper and verified the resulting seven, seven, seven, seven, six, and seven sessions per subject.

## 1-c. How are the data split into sessions?

i. One dated, digit-prefixed subdirectory under a mouse is treated as one session (one daily recording), with session names sorted lexicographically.

ii.
```python
sessions = get_sessions(mouse_dir)
for session in sessions:
    session_dir = os.path.join(mouse_dir, session)
    neural_trials, input_trials, me_values = process_session(...)
```

iii. The notes identify each dated folder as a daily recording and report 41 total sessions. Sorting makes their order deterministic.

## 1-d. How are the data split into trials?

i. The agent divides each continuous session into non-overlapping 120-second blocks after 10-frame averaging. Each trial therefore has 360 binned samples. Any incomplete block at the end is silently omitted.

ii.
```python
TRIAL_DURATION_SEC = 120
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FRAME_RATE
BINNED_PER_TRIAL = FRAMES_PER_TRIAL // BIN_SIZE
...
n_trials = n_binned // BINNED_PER_TRIAL
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
```

iii. The agent used the paper's “consecutive 2 minute blocks” used for decoder cross-validation and expected 10 trials for 20-minute recordings and 15 for 30-minute recordings. It overlooked the conversion instruction that explicitly required 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. Only complete 120-second blocks are retained; a shorter tail is dropped by floor division.

ii.
```python
n_trials = n_binned // BINNED_PER_TRIAL
```

iii. The notes found no trial-curation rule in the paper because the recordings are continuous. They did not describe incomplete-tail removal as a quality filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy`; the sampling rate is read from `ops.npy` with a 30 Hz fallback.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
```

iii. The agent identified raw and neuropil fluorescence as the relevant Suite2p streams and used the acquisition metadata for the baseline window.

## 2-b. How is the `neural` data processed?

i. The code subtracts 0.7 times neuropil fluorescence, estimates a maximin baseline after Gaussian smoothing, subtracts that baseline, converts to `float32`, and averages non-overlapping groups of 10 frames. Although names and an old docstring call this dF/F, the implemented operation is baseline-corrected fluorescence without division.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
dff = Fc - Flow
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
```

iii. The notes chose `neucoeff=0.7`, `maximin`, Gaussian sigma 10 frames, and a 60-second baseline window as the Suite2p defaults described by the paper. The trajectory records that an initial baseline-division implementation produced implausible values and was changed to subtraction to match Suite2p's actual preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filter is applied; every row in `F.npy` is retained.

ii.
```python
n_neurons, n_frames = F.shape
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. Dataset exploration found all provided `iscell[:, 0]` values equal to 1 and concluded that Track2p's supplied fluorescence arrays had already been filtered at the paper's 0.5 threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no discrete experimental event. Trials are contiguous blocks indexed from session start, and metadata describes the alignment event as the start of the recording session.

ii.
```python
start = t * BINNED_PER_TRIAL
end = (t + 1) * BINNED_PER_TRIAL
neural_trial = dff_binned[:, start:end]
...
'temporal_alignment_event': 'start of recording session',
'off_start': 0.0,
```

iii. The agent treated the continuous recording's start as the only meaningful alignment event. Absolute session time is preserved across successive blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten successive 30 Hz frames are averaged into each sample, giving 333.333 ms bins (3 Hz) for both neural and motion-energy data.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

iii. The agent cites the methods statement that both dF/F and behavior traces were denoised by averaging 10 consecutive timestamps and bins both streams before trial segmentation and categorization.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the binned sample index and fixed bin duration, rather than loaded from a raw timestamp file.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. The agent interpreted “beginning of the experiment” as the beginning of each recording session. Its notes map a time index directly to elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The absolute binned index is multiplied by 10/30 seconds and cast to `float32`; the resulting vector is reshaped to `(1, time)`.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. The agent explicitly checked that time does not reset between trials: for example, the last 120-second trial of a 20-minute session spans approximately 1080.0–1199.7 seconds.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The same `start:end` binned indices used to slice neural activity generate the time vector, so every time value corresponds one-to-one with a neural column.

ii.
```python
neural_trial = dff_binned[:, start:end]
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. The agent's shape checks required neural, input, and output to have identical time dimensions and verified continuous elapsed-time ranges.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived only from each session's precomputed `move_deve/motion_energy_glob.npy`. The code does not load `interframe_int.npy` or timestamps to locate dropped frames.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The notes identify this file as global pixel-wise squared frame differences and acknowledge that it can be shorter than neural data due to missing camera frames. They planned to “interpolate or truncate to common length.”

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The trace is forced to the neural length, averaged in 10-frame bins, restricted to complete trials, concatenated within a session, min-max normalized, and assigned to five within-session percentile bins.

ii.
```python
me_aligned = align_motion_energy(me, n_frames)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
me_all = np.concatenate(me_values)
me_norm = normalize_motion_energy(me_all)
edges = np.percentile(me_norm, np.linspace(0, 100, N_OUTPUT_BINS + 1))
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. The agent chose per-session scaling because motion-energy scale varies by recording and per-session percentiles produce equal-frequency categories. It applied denoising before categorization. Its alignment rationale was only that shorter video arrays should be interpolated to neural length, without using known drop locations.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five percentile categories are computed separately per session from all retained, binned samples. Thresholds are the 20th, 40th, 60th, and 80th percentiles after min-max normalization, and `np.digitize` returns integer labels 0–4.

ii.
```python
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. The agent reasoned that per-session percentiles compensate for session-specific scale and verified approximately 20% occupancy per class. Min-max normalization does not change percentile membership because it is monotonic.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Equal-length streams are used directly. A shorter motion trace is copied at the start of a neural-length array and all remaining positions at the end are filled by `np.interp`, which in practice repeats the final valid value. A longer trace is truncated. Both streams are then independently averaged by 10 and sliced with identical trial indices.

ii.
```python
me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
me_aligned[:len(me)] = me.astype(np.float64)
valid = ~np.isnan(me_aligned)
indices = np.arange(n_neural_frames)
me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

iii. The agent knew camera frames could be missing but assumed the deficit could be repaired at the end. It did not use `interframe_int.npy` to insert values at the actual internal drop positions, unlike the reference solution.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion-energy arrays are end-padded by interpolation/extrapolation, long arrays are truncated, constant motion traces normalize to zeros, and incomplete final trial blocks are dropped. The code has no assertion that corrected alignment reflects actual dropped-frame locations.

ii.
```python
if len(me) < n_neural_frames:
    me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_aligned[:len(me)] = me.astype(np.float64)
    me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
else:
    return me[:n_neural_frames].astype(np.float64)
...
if me_max - me_min < 1e-10:
    return np.zeros_like(me)
```

iii. The notes recognized missing camera frames and selected interpolation or truncation as a pragmatic repair. Validation focused on matching shapes and legal values, so it did not expose internal temporal shifts caused by dropped frames.

## 6-a. What are the most time-consuming steps of the code?

i. Full-session maximin baseline estimation is the main compute-heavy step; loading large fluorescence arrays and serializing the roughly 414 MB result are also substantial. Optional plotting and decoder training are outside the core conversion.

ii.
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
```

iii. The notes estimated about 0.8 seconds per session and about 33 seconds total, but claimed there were no significant inefficiencies. The baseline applies long sliding filters across every neuron and frame.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent reported no significant candidates. Nevertheless, the trial construction loops could be replaced with reshape/slicing operations, and session output trials could be digitized as one vector and split afterward. These loops are small relative to filtering and are also used to construct the required nested lists.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
...
for me_trial in me_values:
    bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` states “Code inefficiencies identified: None significant - vectorized operations used throughout.” The expensive baseline and 10-frame binning are indeed vectorized, though that statement overlooks the remaining trial loops.

## 6-c. What processing does the code repeat multiple times?

i. Necessary loading, baseline correction, alignment, and binning repeat once per session. Percentile logic is written directly in `convert_data` even though an unused `discretize_to_bins` helper implements the same operation. Plotting, when requested, concatenates trial arrays again for display.

ii.
```python
def discretize_to_bins(values, n_bins=5):
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    return np.digitize(values, edges[1:-1]).astype(np.int64)
...
edges = np.percentile(me_norm, percentiles)
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. The agent did not identify repeated processing. The duplicate helper is unused, so it does not add runtime, while per-session repetition is required by the data organization and per-session thresholds.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Per-session min-max normalization before percentile thresholding is mathematically redundant because monotonic affine scaling leaves percentile assignments unchanged. Loading the entire `ops.npy` dictionary only to retrieve `fs` is minor overhead. Optional plots recompute concatenated display vectors but are not part of the saved data.

ii.
```python
me_norm = normalize_motion_energy(me_all)
edges = np.percentile(me_norm, percentiles)
...
ops = np.load(..., allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
```

iii. The agent said there was no significant unnecessary processing and justified normalization as handling session-specific scales. Equal-percentile binning already handles those scales, so normalization affects neither the saved class labels nor decoder inputs.
