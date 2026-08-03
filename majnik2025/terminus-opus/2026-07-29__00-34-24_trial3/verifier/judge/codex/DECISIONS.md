# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs in `MICE`, iterates through those mouse folders, discovers sessions by listing subdirectories whose names start with a digit, and for each session loads `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. Trials are not loaded from disk; they are created later by splitting each processed session into fixed-duration blocks.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions

for mouse_idx, mouse in enumerate(mice_to_process):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
    Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
    ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset contains 6 mice and that each session is one recording day. In the trajectory it says the recording has no explicit trial structure and therefore trials must be created from continuous recordings.

## 1-b. How are the data split into subjects?

i. Subjects are the six hard-coded mouse IDs in `MICE`, in that fixed order.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
mice_to_process = MICE[:2] if sample else MICE
...
subjects = mice_to_process
```

iii. The notes document exactly six mice, `jm031` through `jm046`, so the AI chose to encode that list directly instead of discovering subjects dynamically from the filesystem.

## 1-c. How are the data split into sessions?

i. Each session is a subdirectory within a mouse directory whose name begins with a digit. Sessions are sorted lexicographically and each session becomes one output session.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. In the notes, the AI states that “Session = recording day” and documents that the session folders are date-like names such as `2023-10-18_a`, which explains the digit-prefix filter.

## 1-d. How are the data split into trials?

i. The AI treats the continuous recording as pseudo-trials. It first averages every 10 frames, then splits each session into consecutive 2-minute blocks. Each trial therefore contains `360` binned time points (`120 s * 30 Hz / 10`).

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FRAME_RATE
BINNED_PER_TRIAL = FRAMES_PER_TRIAL // BIN_SIZE
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
n_trials = n_binned // BINNED_PER_TRIAL
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
```

iii. The notes and trajectory justify this with the methods text saying decoding used “averaging in bins of 10 consecutive timestamps” and “consecutive 2 minute blocks of the recording.” The AI explicitly decided to reuse those 2-minute blocks as trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filter. The only exclusions are implicit truncations: leftover frames that do not fill a full 10-frame bin are dropped by `bin_data`, and leftover binned time points that do not fill a full 2-minute block are omitted because `n_trials` uses floor division.

ii.
```python
def bin_data(data, bin_size, axis=-1):
    n = data.shape[axis]
    n_bins = n // bin_size
    n_use = n_bins * bin_size
    ...
    return data_trunc.reshape(new_shape).mean(axis=-1)
...
n_trials = n_binned // BINNED_PER_TRIAL
for t in range(n_trials):
    ...
```

iii. The notes say there are no explicit trial curation rules because the recordings are continuous. The AI does not give any additional trial QC rationale beyond needing fixed-size blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from suite2p fluorescence traces `F.npy` and `Fneu.npy`. The script also reads `ops.npy` to get the frame rate used in processing.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
...
fs = ops.get('fs', FRAME_RATE)
```

iii. The notes say the paper used baseline-corrected fluorescence traces with default Suite2p parameters, and the trajectory records the AI reading `ops.npy` to recover `fs=30`, `neucoeff=0.7`, and the baseline settings.

## 2-b. How is the `neural` data processed?

i. The AI manually reimplements a Suite2p-like baseline correction: neuropil subtraction (`F - 0.7 * Fneu`), Gaussian smoothing, min filter, max filter, then subtraction of that baseline. Afterward it averages the resulting trace in non-overlapping 10-frame bins.

ii.
```python
def compute_dff(F, Fneu, fs=30.0, neucoeff=0.7, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win, axis=1)
    Flow = maximum_filter1d(Flow, win, axis=1)
    dff = Fc - Flow
    return dff.astype(np.float32)
...
dff = compute_dff(F, Fneu, fs=fs)
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
```

iii. The notes justify the baseline part by citing the paper’s “default Suite2p parameters,” and the trajectory shows the AI first considered dividing by baseline, then changed to subtraction-only after inspecting Suite2p’s implementation. The extra 10-frame averaging is justified in the notes as decoding-time denoising from the methods text.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any additional neuron filtering in `convert_data.py`. It keeps every row of `F.npy` and assumes the provided data were already filtered and tracked.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. The notes repeatedly say all `iscell` flags are already `1.0`, the dataset already contains only tracked neurons, and no extra filtering is necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the recording session. There is no stimulus or behavioral alignment event; trials are consecutive blocks measured from session start.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
...
'metadata': {
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. In the notes the AI states there is no natural trial structure and that the input should be “Time elapsed from start of session,” so it uses recording start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 333.33 ms bins because the AI averages every 10 original 30 Hz frames. Temporal rebinning is therefore applied to both neural and behavioral streams.

ii.
```python
BIN_SIZE = 10
FRAME_RATE = 30
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
...
'metadata': {
    'time_bin_size': TIME_BIN_MS,
    'bin_size_frames': BIN_SIZE,
    'frame_rate_hz': FRAME_RATE,
}
```

iii. The notes explicitly justify this by quoting the methods text about averaging 10 consecutive timestamps for decoding.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a stored raw variable. The AI synthesizes time from bin indices and the assumed frame rate.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. The notes map the decoder input to “Time elapsed from start of session in seconds,” and the trajectory says the task specification itself motivated using elapsed time rather than any separate timestamp file.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI uses the left-edge index of each 10-frame bin, multiplies by the 333.33 ms bin size, and stores that as seconds elapsed from session start. It therefore inherits the 10-frame temporal averaging scheme.

ii.
```python
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE
...
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. The notes justify this as matching the required decoder input (“Time elapsed from the beginning of the experiment”) after the chosen 10-frame denoising step.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The input time series is created with the same binned trial boundaries as the neural data, so each trial has one time value per neural time bin.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
    input_trials.append(time_bins.reshape(1, -1))
```

iii. The notes say the decoder input should be time-varying and aligned to the same trialization as the neural data; the AI’s chosen pseudo-trial structure therefore determines the alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy only from `move_deve/motion_energy_glob.npy`. It does not use `interframe_int.npy` or `tstamps.npy` in the final script.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
...
me_aligned = align_motion_energy(me, n_frames)
```

iii. The notes identify `motion_energy_glob.npy` as the behavioral signal and mention that missing frames sometimes occur, but the final code simplifies alignment to a length-based interpolation step rather than using the timing files.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI length-matches motion energy to the neural recording with simple interpolation or truncation, averages it in 10-frame bins, applies per-session min-max normalization, and later discretizes each session separately into percentile bins.

ii.
```python
def align_motion_energy(me, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    elif len(me) < n_neural_frames:
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        valid = ~np.isnan(me_aligned)
        indices = np.arange(n_neural_frames)
        me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
        return me_aligned
    else:
        return me[:n_neural_frames].astype(np.float64)

me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
me_norm = normalize_motion_energy(me_all)
```

iii. The notes justify the 10-frame averaging by the methods text and justify normalization/discretization by the task’s request for “normalized and discretized into five equal-percentile bins.” The trajectory shows the AI explicitly choosing per-session normalization and session-wise percentile edges.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. For each session separately, the AI computes percentiles on the min-max normalized motion-energy values and uses those percentile cutoffs to assign five categories with `np.digitize`.

ii.
```python
me_norm = normalize_motion_energy(me_all)
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
...
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. The notes say this choice was made because the task asked for equal-percentile bins and the AI believed “per-session percentiles makes more sense” after per-session normalization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is done by forcing the motion-energy array to the same length as the neural recording, then binning both streams identically and splitting them with the same trial boundaries.

ii.
```python
me_aligned = align_motion_energy(me, n_frames)
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    output_trials.append(bin_labels.reshape(1, -1))
```

iii. The notes mention that motion energy can have fewer frames than neural data and says it should be interpolated to match. The final script operationalizes that as simple length-based interpolation/truncation rather than dropped-frame insertion at recorded indices.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles short motion-energy vectors by padding to neural length and linearly interpolating across the missing positions; longer vectors are truncated. Incomplete frame tails are silently dropped during 10-frame averaging, and incomplete trailing 2-minute blocks are omitted.

ii.
```python
elif len(me) < n_neural_frames:
    me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_aligned[:len(me)] = me.astype(np.float64)
    valid = ~np.isnan(me_aligned)
    indices = np.arange(n_neural_frames)
    me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
    return me_aligned
else:
    return me[:n_neural_frames].astype(np.float64)
...
n_bins = n // bin_size
n_use = n_bins * bin_size
...
n_trials = n_binned // BINNED_PER_TRIAL
```

iii. The notes say motion-energy files can have missing camera frames and that interpolation is a sensible fix. The trajectory shows the AI noticing `interframe_int.npy` but the final implementation falls back to a simpler length-only strategy.

## 6-a. What are the most time-consuming steps of the code?

i. The AI does not name a single bottleneck explicitly, but the code structure and notes indicate the expensive work is per-session preprocessing of full fluorescence matrices in `compute_dff`, plus session-level I/O. It also instruments timing at the session and whole-run level.

ii.
```python
def process_session(...):
    t0 = time.time()
    ...
    dff = compute_dff(F, Fneu, fs=fs)
    ...
    t1 = time.time()
    print(f'  {mouse}/{session}: {n_neurons} neurons, {n_frames} frames, '
          f'{n_binned} binned frames, {n_trials} trials, {t1-t0:.2f}s')
...
t_start = time.time()
...
t_total = time.time() - t_start
print(f'  Total time: {t_total:.1f}s')
```

iii. In `CONVERSION_NOTES.md` Step 6 the AI says there were no significant inefficiencies and records runtime estimates for the full conversion, implying that preprocessing and data loading were the main measured costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s documented position is that there are no significant vectorization opportunities left. The remaining explicit loops are over sessions and pseudo-trials when assembling list-of-trial outputs and per-session discretized outputs.

ii.
```python
for mouse_idx, mouse in enumerate(mice_to_process):
    ...
    for session in sessions:
        ...

for t in range(n_trials):
    ...
    neural_trials.append(neural_trial)
    input_trials.append(time_bins.reshape(1, -1))
    me_binned_values.append(me_binned[start:end])
...
for me_trial in me_values:
    ...
    output_trials.append(bin_labels.reshape(1, -1))
```

iii. The notes explicitly say “Code inefficiencies identified: None significant - vectorized operations used throughout.”

## 6-c. What processing does the code repeat multiple times?

i. The AI does not identify any repeated processing as a concern. The only repeated work is the intended per-session preprocessing and then a second pass that discretizes motion energy and assembles the final dataset.

ii.
```python
session_data = []
...
for session in sessions:
    neural_trials, input_trials, me_values = process_session(...)
    session_data.append((mouse_idx, mouse, session, neural_trials, input_trials, me_values))

for mouse_idx, mouse, session, neural_trials, input_trials, me_values in session_data:
    me_all = np.concatenate(me_values)
    ...
```

iii. The notes say there are no significant inefficiencies and describe the script as a straightforward vectorized pipeline, so the AI did not call out any repeated computation as problematic.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does not acknowledge any unnecessary discarded processing. In practice, the script still loads `ops.npy` only to read `fs`, defines an unused helper `discretize_to_bins`, and includes optional plotting paths, but it does not discuss these as wasteful.

ii.
```python
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
...
def discretize_to_bins(values, n_bins=5):
    ...
    return bin_labels.astype(np.int64)
...
if show_processing:
    plot_processing(data, mice_to_process)
```

iii. `CONVERSION_NOTES.md` Step 6 says there were no significant inefficiencies, so the AI’s stated view is that nothing important is computed and then thrown away.
