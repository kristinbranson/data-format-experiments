# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs, enumerates each subject directory, then enumerates all child directories whose names start with a digit as sessions. For each session it loads only `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. Trials are not native in the raw data; they are created later by splitting each session into consecutive 2-minute blocks after 10-frame binning.

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
    for session in sessions:
        session_dir = os.path.join(mouse_dir, session)
        neural_trials, input_trials, me_values = process_session(...)
```

```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by stating that the dataset contains 6 mouse folders, each session folder corresponds to one recording day, and the provided Suite2p outputs already contain tracked neurons for each day. The trajectory shows it verified the folder layout and then treated each mouse/day directory as the basic unit to load.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level mouse folders. The AI uses the hard-coded mouse list as the authoritative subject list and stores a single `subject_idx` per session.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
subjects = mice_to_process
...
all_subject_idx.append(mouse_idx)
...
'subjects': subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The notes say the data README maps `jm031`-`jm046` to mice A-F and that there are 6 subjects total. The trajectory shows the AI accepted that folder organization directly and did not try to infer subjects from metadata files.

## 1-c. How are the data split into sessions?

i. Each date-named recording-day directory inside a mouse folder is treated as one session. Sessions are sorted lexicographically.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. The notes and trajectory both say the session subfolders correspond to daily recordings, and the data README says the folder name is a recording date that identifies one recording day. That is the AI's basis for equating one folder with one session.

## 1-d. How are the data split into trials?

i. The raw recordings are continuous, so the AI creates pseudo-trials. It first averages every 10 frames, then splits each binned session into consecutive 2-minute blocks. A 20-minute session becomes 10 trials; a 30-minute session becomes 15 trials.

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
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    neural_trials.append(neural_trial)
```

iii. The AI explicitly justified this in the trajectory and notes: there are no natural trials in spontaneous behavior, but the methods mention decoding on consecutive 2-minute blocks, so it reused those blocks as the target-format trial unit.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only implicit filtering is structural: after binning, the script keeps only complete non-overlapping 2-minute blocks and silently drops any leftover partial block or incomplete tail.

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
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
```

iii. The notes say "Trial curation rules: None explicitly mentioned - continuous recordings." The trajectory shows the AI concluded there was no trial-level QC in the source materials and therefore did not add any beyond dropping incomplete terminal fragments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence traces `F.npy` and neuropil traces `Fneu.npy`, with the sampling rate read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
...
fs = ops.get('fs', FRAME_RATE)
...
dff = compute_dff(F, Fneu, fs=fs)
```

iii. The notes state that `load_data.ipynb` identifies `F.npy` as raw fluorescence, and that the paper asks for baseline-corrected fluorescence traces using Suite2p defaults. The trajectory shows the AI considered `spks.npy` but rejected it because the notebook suggested computing fluorescence-derived traces as described in the paper.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil correction `Fc = F - 0.7 * Fneu`, estimates a maximin baseline with a Gaussian smooth followed by rolling min and max filters, subtracts that baseline, then averages the result in non-overlapping bins of 10 frames.

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
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
```

iii. The justification comes from two places. In `CONVERSION_NOTES.md`, the AI says the paper uses "baseline corrected fluorescence traces" with Suite2p defaults and 10-frame averaging. In the trajectory, it first implemented division by baseline, then debugged that choice and switched to subtraction after inspecting Suite2p's `preprocess` and `baseline_maximin` behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script performs no additional neuron filtering itself. It assumes the provided Suite2p/Track2p outputs are already curated to tracked cells, and it uses every row of `F.npy`/`Fneu.npy`.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
...
n_neurons = neural_trials[0].shape[0]
all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The notes say all `iscell[:,0] == 1.0` in the provided data and that the dataset already contains only successfully tracked neurons present across all days. The trajectory shows the AI inspected `iscell.npy`, confirmed that no extra filtering was needed, and then omitted `iscell` from the final processing path.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the recording session. Trials are consecutive blocks measured from that session start; there is no event-based within-session realignment.

ii.
```python
'metadata': {
    ...
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. The trajectory says the source recordings are spontaneous and have no natural trial event, while the task-defined decoder input is "time elapsed from the beginning of the experiment." That led the AI to use recording start as the alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10-frame bins at 30 Hz, so the time bin size is `1000 * 10 / 30 = 333.33 ms`. Both neural and motion-energy streams are rebinned by averaging over these non-overlapping 10-frame windows.

ii.
```python
BIN_SIZE = 10
FRAME_RATE = 30
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

iii. The notes cite the methods text: "denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI therefore uses the same 10-frame temporal averaging for both streams.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a timestamp array in the raw files. Instead, it is synthesized from the binned-frame index plus constants for frame rate and bin size.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. The trajectory shows the AI noticed that `tstamps.npy` exists and is in kiloseconds, but chose not to use it. Its stated rationale was that the task asks for time from the beginning of the experiment and that a uniform time base derived from the neural frame grid is the simplest way to align input to neural data.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each pseudo-trial, the script creates a 1D sequence of absolute session times in seconds. The first trial covers approximately `0.0` to `119.7` s, later trials continue the running session clock rather than resetting to zero.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
    input_trials.append(time_bins.reshape(1, -1))
```

iii. In the trajectory, the AI explicitly re-checked the task wording and concluded that the input should be absolute elapsed time from recording start, not time-from-trial-start.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is generated on exactly the same binned grid used for the neural data. Each trial's `start:end` range is shared between neural slices and the time vector.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
...
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    ...
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. The AI's justification in the trajectory is that generating the input directly from the neural bin index guarantees 1:1 alignment with the neural time axis, regardless of the behavior-camera timestamps.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output comes from the precomputed behavior signal `move_deve/motion_energy_glob.npy`.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The notes say the methods describe motion energy as squared frame-to-frame video differences summed over pixels, and the data README says that processed variable is already stored in `motion_energy_glob.npy`. The AI therefore uses the provided processed behavior file directly.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI does not recompute motion energy from video. Instead it length-aligns the provided `motion_energy_glob.npy` to the neural frame count, averages it in 10-frame bins, concatenates all binned values within a session, min-max normalizes them to `[0, 1]`, and later discretizes them.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
me_aligned = align_motion_energy(me, n_frames)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
me_all = np.concatenate(me_values)
me_norm = normalize_motion_energy(me_all)
```

```python
def normalize_motion_energy(me):
    me_min = np.nanmin(me)
    me_max = np.nanmax(me)
    if me_max - me_min < 1e-10:
        return np.zeros_like(me)
    return (me - me_min) / (me_max - me_min)
```

iii. `CONVERSION_NOTES.md` says the paper already defines motion energy upstream, while the task adds a normalization-and-discretization requirement. The trajectory also shows the AI explicitly chose per-session normalization because each session has its own motion-energy scale.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses five equal-percentile bins computed separately within each session after per-session min-max normalization. It computes percentile edges on the concatenated session-wide normalized motion-energy vector and applies `np.digitize` to each trial.

ii.
```python
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
...
for me_trial in me_values:
    n_t = len(me_trial)
    me_trial_norm = me_norm[offset:offset + n_t]
    bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
    output_trials.append(bin_labels.reshape(1, -1))
    offset += n_t
```

iii. The trajectory contains the explicit decision point: global percentiles versus per-session percentiles. The AI justified per-session binning by saying the paper normalizes behavior per session and that per-session percentiles better preserve equal-frequency class labels when sessions differ in scale.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is done only by matching array lengths, not by using the timestamp files. If motion energy is shorter than the neural trace, the script copies it into the start of a longer array, fills the remainder with NaNs, and linearly interpolates. If the lengths already match, it assumes sample `i` of motion energy aligns to neural frame `i`.

ii.
```python
def align_motion_energy(me, n_neural_frames):
    if len(me) == n_neural_frames:
        return me.astype(np.float64)
    elif len(me) < n_neural_frames:
        me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
        me_aligned[:len(me)] = me.astype(np.float64)
        if np.any(np.isnan(me_aligned)):
            valid = ~np.isnan(me_aligned)
            indices = np.arange(n_neural_frames)
            me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
        return me_aligned
    else:
        return me[:n_neural_frames].astype(np.float64)
```

iii. The notes cite the data README saying missing-camera-frame indices can be found from `tstamps.npy` or `interframe_int.npy` and can be treated as missing or interpolated over. The AI decided not to use those arrays and instead use a simpler length-based interpolation scheme, as reflected in the trajectory and the final code.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or mismatched behavior frames with a simple heuristic: if motion energy is shorter than the neural trace, pad the end with NaNs and linearly interpolate; if longer, truncate; if equal length, keep as-is. It also silently truncates any incomplete 10-frame bin or incomplete 2-minute block.

ii.
```python
if len(me) == n_neural_frames:
    return me.astype(np.float64)
elif len(me) < n_neural_frames:
    me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_aligned[:len(me)] = me.astype(np.float64)
    ...
    me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
    return me_aligned
else:
    return me[:n_neural_frames].astype(np.float64)
```

```python
n_bins = n // bin_size
n_use = n_bins * bin_size
...
n_trials = n_binned // BINNED_PER_TRIAL
```

iii. The notes say missing camera frames can be interpolated or treated as missing, and the trajectory shows the AI chose interpolation as the simpler path. It did not implement timestamp-based gap localization and did not document any special handling for NaNs within neural traces because it found none.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant work is per-session full-array preprocessing: loading large `.npy` matrices, computing the maximin baseline for every neuron over the entire session, binning the full neural and motion-energy traces, and then doing a second pass over session motion-energy values for normalization/discretization.

ii.
```python
F = np.load(...)
Fneu = np.load(...)
...
dff = compute_dff(F, Fneu, fs=fs)
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

```python
for mouse_idx, mouse, session, neural_trials, input_trials, me_values in session_data:
    me_all = np.concatenate(me_values)
    me_norm = normalize_motion_energy(me_all)
    edges = np.percentile(me_norm, percentiles)
```

iii. The conversion logs in `conversion_full_out.txt` show runtime scaling strongly with neuron count and session length, which is consistent with the AI's notes that the full-session dF/F computation is the main cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-building loop in `process_session` and the output-discretization loop in `convert_data` could have been reshaped/vectorized instead of repeatedly slicing and appending Python lists. Session discovery is not a bottleneck, but these repeated per-trial loops are avoidable.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    neural_trials.append(neural_trial)
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
    input_trials.append(time_bins.reshape(1, -1))
    me_binned_values.append(me_binned[start:end])
```

```python
for me_trial in me_values:
    n_t = len(me_trial)
    me_trial_norm = me_norm[offset:offset + n_t]
    bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
    output_trials.append(bin_labels.reshape(1, -1))
    offset += n_t
```

iii. The AI claimed in the notes that there were no significant inefficiencies, but the code itself shows several Python-level loops over trials that could be replaced by reshaping to `(n_trials, 360)` once the session has already been binned.

## 6-c. What processing does the code repeat multiple times?

i. The script makes two passes over session motion energy: first it slices and stores per-trial binned values in `process_session`, then it concatenates those values back together in `convert_data`, normalizes them, and re-slices them to create outputs. It also recomputes percentile edges separately for every session.

ii.
```python
me_binned_values = []
...
me_binned_values.append(me_binned[start:end])
...
return neural_trials, input_trials, me_binned_values
```

```python
me_all = np.concatenate(me_values)
me_norm = normalize_motion_energy(me_all)
...
for me_trial in me_values:
    ...
    me_trial_norm = me_norm[offset:offset + n_t]
```

iii. This follows from the AI's design choice to postpone discretization until it had all motion-energy values from a session, so it could compute session-specific percentile thresholds.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the entire `ops.npy` dict but uses only `fs`; it defines `discretize_to_bins` but never calls it; it prints large per-session summaries; and, when `--show-processing` is used, it generates visualization PNGs that are not consumed by decoder training. It also stores intermediate per-trial motion-energy slices only to concatenate them again.

ii.
```python
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
```

```python
def discretize_to_bins(values, n_bins=5):
    ...
    return bin_labels.astype(np.int64)
```

```python
if show_processing:
    plot_processing(data, mice_to_process)
```

iii. The trajectory and notes emphasize validation, debugging, and visualization. Those steps were useful for the AI while building the file, but they do not contribute to the final `converted_data.pkl` consumed downstream.
