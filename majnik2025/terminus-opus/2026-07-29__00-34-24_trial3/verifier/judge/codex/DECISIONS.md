# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six mice in a `MICE` list, then iterates over each mouse's session directories, loading per-session neural and behavioral files. Within each session it loads `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`, processes them, and later aggregates the session outputs into the final dataset.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for mouse_idx, mouse in enumerate(mice_to_process):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = get_sessions(mouse_dir)
    ...
    for session in sessions:
        session_dir = os.path.join(mouse_dir, session)
        neural_trials, input_trials, me_values = process_session(
            mouse, session, session_dir, show_processing=show_processing
        )

F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. In `CONVERSION_NOTES.md` Step 2 and the trajectory, the AI justified this by noting there are six mice (`jm031` to `jm046`), each with daily recording folders containing Suite2p outputs and motion-energy files, and that each session directory should be treated as one recording session.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded mouse IDs in `MICE`, with `subject_idx` assigned from the loop index over that list.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for mouse_idx, mouse in enumerate(mice_to_process):
    ...
    all_subject_idx.append(mouse_idx)
...
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array(all_subject_idx, dtype=np.int64),
}
```

iii. The notes say the dataset contains exactly six mice and repeatedly refer to those IDs explicitly, so the AI chose to encode that discovered set directly rather than rescan the directory tree dynamically.

## 1-c. How are the data split into sessions?

i. Sessions are the sorted subdirectories inside each mouse directory whose names start with a digit, and each such directory is treated as one recording day.

ii.
```python
def get_sessions(mouse_dir):
    """Get sorted list of session directories for a mouse."""
    sessions = [d for d in os.listdir(mouse_dir)
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. In the notes and trajectory, the AI described the subject folders as containing date-named session folders and explicitly decided that “each session = one recording day for one mouse.”

## 1-d. How are the data split into trials?

i. The AI does not use the requested 60-second trials. It bins each session by 10 frames first and then segments the continuous recording into non-overlapping 2-minute blocks (`TRIAL_DURATION_SEC = 120`), producing 360 binned time points per trial.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2-minute blocks
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FRAME_RATE  # 3600
BINNED_PER_TRIAL = FRAMES_PER_TRIAL // BIN_SIZE     # 360
...
n_trials = n_binned // BINNED_PER_TRIAL
...
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
```

iii. The AI justified this in `CONVERSION_NOTES.md` Step 5 and trajectory steps 30-31 by pointing to the paper’s decoder-validation procedure using consecutive 2-minute blocks, and chose to reuse those blocks as pseudo-trials.

## 1-e. How are trials filtered based on quality controls?

i. No explicit trial-quality filtering is applied. Trials are created by fixed segmentation, and only complete 2-minute blocks are kept because `n_trials` is computed by floor division.

ii.
```python
n_trials = n_binned // BINNED_PER_TRIAL
...
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
```

iii. The notes say there were no explicit trial curation rules because the recordings are continuous rather than event-structured, so the AI treated all complete blocks as valid.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from raw fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`; `ops.npy` is also loaded to obtain the frame rate.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
...
fs = ops.get('fs', FRAME_RATE)
```

iii. The AI’s notes and trajectory say the paper uses Suite2p-default baseline-corrected fluorescence traces, so it identified `F` and `Fneu` as the relevant raw neural sources and consulted `ops.npy` for the Suite2p parameters.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`F - 0.7 * Fneu`), estimates a baseline with a Gaussian filter followed by minimum and maximum filters (its manual approximation to Suite2p `maximin`), subtracts that baseline, casts to `float32`, and then averages in non-overlapping 10-frame bins.

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

iii. The AI initially tried division by baseline, then corrected itself in trajectory steps 37 and 43 after checking Suite2p and concluding that the paper’s “baseline corrected fluorescence traces” correspond to subtraction-only baseline correction with Suite2p-style defaults.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied in the conversion script. The AI assumes the provided arrays are already Track2p/Suite2p-filtered and uses all rows in `F.npy`.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
...
all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. In `CONVERSION_NOTES.md` Steps 1-3 and trajectory steps 9-10, the AI noted that all `iscell[:,0]` values were 1.0 and concluded that the data had already been filtered to tracked cells, so no extra QC filter was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the start of the recording session, not to a stimulus or behavioral event. The session is treated as continuous, and each trial inherits its position within that session timeline.

ii.
```python
data = {
    ...
    'metadata': {
        ...
        'temporal_alignment_event': 'start of recording session',
        'off_start': 0.0,
        'off_end': None,
    }
}
```

iii. The notes say there is no natural event structure in the dataset, and trajectory step 61 explicitly re-checks that “time elapsed from the beginning of the experiment” means session-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have 333.33 ms bins. Both neural and motion-energy streams are averaged into non-overlapping bins of 10 frames at 30 Hz before trial segmentation.

ii.
```python
BIN_SIZE = 10
FRAME_RATE = 30
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
'time_bin_size': TIME_BIN_MS,
```

iii. The AI repeatedly cites the methods text line that decoding used averages over 10 consecutive timestamps, and uses that as the justification for applying identical 10-frame denoising bins to both streams.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from a raw timestamp file. It is derived from the binned sample index, `TIME_BIN_MS`, and the implicit session start.

ii.
```python
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE
...
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. The AI’s notes explain that the task requires time elapsed from experiment/session start, and trajectory step 61 confirms that it intentionally uses absolute session time rather than trial-relative time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the AI generates a contiguous vector of binned time indices in seconds, using the left edge of each 10-frame bin. No raw timestamp correction or interpolation is used.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
    input_trials.append(time_bins.reshape(1, -1))
```

iii. The AI justified this as the simplest representation of “time elapsed from the beginning of the experiment,” given the constant 30 Hz acquisition and uniform 10-frame bins.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Time is aligned to neural data by using the same binned indices and the same trial start/end slices as the binned neural array.

ii.
```python
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
...
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. The AI did not separately discuss this in the notes, but the code and the trajectory’s “time elapsed from session start” discussion imply alignment by shared bin index after identical segmentation.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy solely from `move_deve/motion_energy_glob.npy`. It does not use `interframe_int.npy` or `tstamps.npy` in the final script.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
...
me_aligned = align_motion_energy(me, n_frames)
```

iii. Early exploration in the trajectory recognized `interframe_int.npy` and `tstamps.npy`, but the implemented script settles on a simpler length-matching approach using only the motion-energy array itself.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI length-aligns motion energy to the neural recording, min-max normalizes it within session, bins it by 10 frames, and then discretizes it per session using percentile thresholds.

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
...
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
me_norm = normalize_motion_energy(me_all)
edges = np.percentile(me_norm, percentiles)
```

iii. In the notes and trajectory step 30, the AI justified session-wise normalization and percentile binning by reasoning that motion-energy scale is session-specific and that equal-frequency bins should therefore be defined within each session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes five equal-percentile bins separately within each session and uses `np.digitize` to assign labels `0` through `4`.

ii.
```python
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
...
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
output_trials.append(bin_labels.reshape(1, -1))
```

iii. The AI explicitly debated global versus per-session thresholds in the code comments and notes, and chose per-session equal-percentile discretization because the task called for session-local motion-energy categorization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is done by forcing the motion-energy vector to the same length as the neural recording. If it is shorter, the AI pads the tail with `NaN` and applies `np.interp`; if longer, it truncates. It then bins and slices the resulting vector with the same trial boundaries as the neural data.

ii.
```python
def align_motion_energy(me, n_neural_frames):
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
...
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
me_binned_values.append(me_binned[start:end])
```

iii. The AI’s notes recognized that video frames could be missing relative to neural frames; the implemented justification was that interpolation or truncation was sufficient to enforce frame-count alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles shorter motion-energy traces by interpolating them up to the neural frame count, handles longer traces by truncation, and silently drops incomplete trailing time after trial segmentation because of floor division.

ii.
```python
elif len(me) < n_neural_frames:
    me_aligned = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_aligned[:len(me)] = me.astype(np.float64)
    ...
    me_aligned = np.interp(indices, indices[valid], me_aligned[valid])
    return me_aligned
else:
    return me[:n_neural_frames].astype(np.float64)
...
n_trials = n_binned // BINNED_PER_TRIAL
```

iii. The notes and trajectory mention missing camera frames and describe interpolation as the chosen remedy. The script itself does not add stricter checks such as asserting exact alignment after frame-drop repair.

## 6-a. What are the most time-consuming steps of the code?

i. The code’s most expensive work is the per-session neural preprocessing in `compute_dff` and the associated full-session I/O. The baseline-estimation filters operate over every neuron and every frame before any downsampling.

ii.
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
...
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI’s notes do not provide a separate timing breakdown, but its runtime discussion and debugging centered on the baseline-correction stage, which is the obvious dominant computation in the script.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code is mostly vectorized already, but the explicit Python loops over trials and over session-local motion-energy chunks could be reduced by reshaping whole sessions into trial blocks in one operation rather than slicing one trial at a time.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    ...
    me_binned_values.append(me_binned[start:end])
...
for me_trial in me_values:
    n_t = len(me_trial)
    me_trial_norm = me_norm[offset:offset + n_t]
    bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 6 says no major inefficiencies were identified because the heavy array operations are vectorized; the remaining loops are largely for packaging data into the nested session/trial output structure.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated scientific processing beyond applying the same per-session pipeline to every session. Within a session, the script separately loops once to collect motion-energy trial chunks and again to convert those chunks into discrete output trials.

ii.
```python
for t in range(n_trials):
    ...
    me_binned_values.append(me_binned[start:end])
...
for mouse_idx, mouse, session, neural_trials, input_trials, me_values in session_data:
    me_all = np.concatenate(me_values)
    ...
    for me_trial in me_values:
        ...
        output_trials.append(bin_labels.reshape(1, -1))
```

iii. The AI did not explicitly call this out as a concern; the notes characterize the implementation as largely straightforward and vectorized, with repeated steps arising mainly from the nested session/trial format.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary processing is per-session min-max normalization of motion energy before percentile binning: because percentile binning is invariant to monotonic affine rescaling, the normalized values themselves are never used downstream. The script also loads `ops.npy` mainly to recover `fs`, even though the frame rate is fixed at 30 Hz in the dataset and constants.

ii.
```python
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
...
def normalize_motion_energy(me):
    me_min = np.nanmin(me)
    me_max = np.nanmax(me)
    if me_max - me_min < 1e-10:
        return np.zeros_like(me)
    return (me - me_min) / (me_max - me_min)
...
me_norm = normalize_motion_energy(me_all)
edges = np.percentile(me_norm, percentiles)
```

iii. The AI justified normalization in the notes as standardizing session-specific motion-energy scale, but because the next step is within-session percentile binning, that normalization does not change the categorical output that is ultimately saved.
