# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six mouse IDs in `MICE`, iterates over them, lists session subdirectories for each mouse, and then loads session-level arrays from `suite2p/plane0` and `move_deve`. Trial structure is not loaded from disk; it is created later by segmenting each session after preprocessing.

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
```

```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The notes say the dataset has six mice and that each session is one recording day, so the AI chose to process those six mice explicitly and treat each date-stamped folder as one session. The notes also state that trials must be created because the recordings are continuous rather than trial-structured.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the hard-coded entries in `MICE`, in that fixed order.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
mice_to_process = MICE[:2] if sample else MICE
...
subjects = mice_to_process
```

iii. In `CONVERSION_NOTES.md`, the AI documented that the dataset contains exactly six mice (`jm031` to `jm046`) and used that known set directly.

## 1-c. How are the data split into sessions?

i. For each mouse, sessions are the subdirectories whose names start with a digit, sorted lexicographically. Each such directory is treated as one session.

ii.
```python
def get_sessions(mouse_dir):
    sessions = [d for d in os.listdir(mouse_dir) 
                if os.path.isdir(os.path.join(mouse_dir, d)) and d[0].isdigit()]
    sessions.sort()
    return sessions
```

iii. The notes say “Session = recording day,” and the data exploration step documented date-named folders under each mouse, which is the basis for this split.

## 1-d. How are the data split into trials?

i. Trials are artificial 2-minute blocks, created after 10-frame temporal binning. Each trial therefore contains 360 binned timepoints.

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

iii. The AI justified this in the notes by citing the methods text: decoding used “consecutive 2 minute blocks of the recording,” and neural and behavioral traces were averaged in bins of 10 timestamps.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The only effective filtering is that incomplete final trial fragments are dropped because the code uses integer division when computing the number of trials.

ii.
```python
n_binned = dff_binned.shape[1]
n_trials = n_binned // BINNED_PER_TRIAL
...
for t in range(n_trials):
    ...
```

iii. The notes state that there were no explicit trial curation rules because the recordings are continuous; the AI only enforced fixed-length block construction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` and `Fneu.npy`. The AI also reads `ops.npy` to obtain the frame rate.

ii.
```python
F = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'Fneu.npy'))
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
```

iii. The notes identify the neural mapping as `F.npy, Fneu.npy -> neural`, with Suite2p-style processing and a frame rate taken from Suite2p metadata.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`F - 0.7 * Fneu`), computes a maximin-style baseline with Gaussian smoothing and min/max filters, subtracts that baseline, and then averages the resulting traces in non-overlapping 10-frame bins.

ii.
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
dff = Fc - Flow
```

```python
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
```

iii. The notes say the AI intended to follow “Suite2p defaults” (`neucoeff=0.7`, `baseline=maximin`) from the paper and then apply the paper’s decoder-time denoising by averaging 10 consecutive timestamps. The notes also record that an earlier `(Fc-Flow)/Flow` version was corrected to subtraction only.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied during conversion. All rows in `F.npy` are carried through.

ii.
```python
n_neurons, n_frames = F.shape
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
neural_trials.append(neural_trial)
```

iii. The notes explicitly state that all `iscell[:, 0]` values were already `1.0`, so the AI treated the provided data as pre-filtered and did not apply further curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the start of the recording session. Each trial is a contiguous block taken from the session-long neural trace, and the metadata names the alignment event as the start of the recording session.

ii.
```python
neural_trial = dff_binned[:, start:end].astype(np.float32)
...
'metadata': {
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The trajectory shows the AI explicitly reconsidered whether absolute session time was the right interpretation and concluded it matched the instruction “Time elapsed from the beginning of the experiment.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has a 333.33 ms time bin size. Yes: the AI rebins both neural and motion-energy traces by averaging every 10 frames from the native 30 Hz data.

ii.
```python
BIN_SIZE = 10
FRAME_RATE = 30
TIME_BIN_MS = 1000.0 * BIN_SIZE / FRAME_RATE
...
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
```

iii. The notes justify this with the methods statement that “for all decoding analysis” both dF/F and behavior were averaged in bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not derived from a stored raw time variable. The AI computes it from the binned sample index and the effective bin duration.

ii.
```python
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. The notes map the decoder input to “time index -> Time elapsed from start of session in seconds,” and the trajectory says this was chosen to satisfy the task wording.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI converts the binned frame index within a session to seconds by multiplying by `TIME_BIN_MS / 1000`. Because `start` and `end` are session-global binned indices, each trial contains absolute session time rather than trial-relative time.

ii.
```python
start = t * BINNED_PER_TRIAL
end = (t + 1) * BINNED_PER_TRIAL
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
```

iii. The trajectory explicitly notes that the time input should represent “absolute time from session start, not from trial start,” and then concludes this matches “Time elapsed from the beginning of the experiment.”

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. It is aligned one-to-one with the neural bins because both are created from the same `start:end` slice on the same binned session timeline.

ii.
```python
neural_trial = dff_binned[:, start:end].astype(np.float32)
...
time_bins = (np.arange(start, end) * TIME_BIN_MS / 1000.0).astype(np.float32)
input_trials.append(time_bins.reshape(1, -1))
```

iii. The notes describe the input as a per-trial time-varying signal and use the same 10-frame binning and 2-minute trial segmentation as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy only from `move_deve/motion_energy_glob.npy`. It does not use `interframe_int.npy` or other timestamp files during conversion.

ii.
```python
me = np.load(os.path.join(session_dir, 'move_deve', 'motion_energy_glob.npy'))
...
me_aligned = align_motion_energy(me, n_frames)
```

iii. The notes identify `motion_energy_glob.npy` as the source variable and separately mention that motion-energy recordings can have missing camera frames, which the AI planned to handle by interpolation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns motion energy to the neural frame count by interpolating to the target length, averages it in non-overlapping 10-frame bins, min-max normalizes each session to `[0, 1]`, and then discretizes it session by session.

ii.
```python
me_aligned = align_motion_energy(me, n_frames)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
me_all = np.concatenate(me_values)
me_norm = normalize_motion_energy(me_all)
edges = np.percentile(me_norm, percentiles)
```

```python
def normalize_motion_energy(me):
    me_min = np.nanmin(me)
    me_max = np.nanmax(me)
    if me_max - me_min < 1e-10:
        return np.zeros_like(me)
    return (me - me_min) / (me_max - me_min)
```

iii. The notes cite decoder-time 10-frame averaging from the methods. For normalization and per-session discretization, the notes and in-code comments say the paper’s motion-energy scale is session-specific and that per-session percentiles “make more sense.”

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Within each session, the AI computes the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of the min-max-normalized motion energy, then uses `np.digitize` to map each time bin to one of five classes `0..4`.

ii.
```python
percentiles = np.linspace(0, 100, N_OUTPUT_BINS + 1)
edges = np.percentile(me_norm, percentiles)
...
bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
output_trials.append(bin_labels.reshape(1, -1))
```

iii. The code comment at the top of the second pass says the AI considered global percentiles but chose per-session percentiles because it believed that was more sensible after per-session normalization.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first forces the raw motion-energy trace to have the same number of frames as the neural recording, then bins both with the same 10-frame window, and finally slices both with the same trial boundaries.

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
```

```python
dff_binned = bin_data(dff, BIN_SIZE, axis=-1)
me_binned = bin_data(me_aligned, BIN_SIZE, axis=0)
...
me_binned_values.append(me_binned[start:end])
```

iii. The notes say motion energy sometimes has missing camera frames and that these should be interpolated to match neural frame count. The trajectory describes this as an alignment step needed before binning and trialization.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main error-handling is for motion-energy length mismatches: if motion energy is shorter than the neural trace, the AI pads to the neural length with `NaN` and interpolates; if it is longer, it truncates. Any leftover frames that do not fill a complete 2-minute binned trial are silently discarded by integer division.

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
```

```python
n_bins = n // bin_size
n_use = n_bins * bin_size
data_trunc = data[..., :n_use]
...
n_trials = n_binned // BINNED_PER_TRIAL
```

iii. The notes explicitly call out “Missing ME frames: Interpolate to match neural frame count” as the chosen policy.

## 6-a. What are the most time-consuming steps of the code?

i. The AI’s notes claim there are no significant inefficiencies and that operations are vectorized throughout, but in the code the heaviest work is the per-session baseline computation for every neuron (`gaussian_filter` + `minimum_filter1d` + `maximum_filter1d`), plus repeated loading and binning of large arrays.

ii.
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., sig_baseline])
Flow = minimum_filter1d(Flow, win, axis=1)
Flow = maximum_filter1d(Flow, win, axis=1)
...
F = np.load(...)
Fneu = np.load(...)
me = np.load(...)
```

iii. The only explicit justification in the notes is: “Code inefficiencies identified: None significant - vectorized operations used throughout.”

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python loops that could be reduced: trial-by-trial construction of `neural_trials`, `input_trials`, and `me_binned_values`; session-by-session concatenation and rediscretization of motion energy; and repeated concatenations for summaries/plots.

ii.
```python
for t in range(n_trials):
    start = t * BINNED_PER_TRIAL
    end = (t + 1) * BINNED_PER_TRIAL
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    input_trials.append(time_bins.reshape(1, -1))
    me_binned_values.append(me_binned[start:end])
```

```python
for mouse_idx, mouse, session, neural_trials, input_trials, me_values in session_data:
    me_all = np.concatenate(me_values)
    ...
    for me_trial in me_values:
        ...
        output_trials.append(bin_labels.reshape(1, -1))
```

iii. The notes again justify the implementation only by saying the code is already sufficiently vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats several data-assembly steps: it first stores per-trial motion-energy slices in `me_binned_values`, then concatenates them back into one session vector, then slices them into trials again after discretization. It also re-concatenates trial data later for output distributions and plotting.

ii.
```python
me_binned_values.append(me_binned[start:end])
...
me_all = np.concatenate(me_values)
...
for me_trial in me_values:
    n_t = len(me_trial)
    me_trial_norm = me_norm[offset:offset + n_t]
    bin_labels = np.digitize(me_trial_norm, edges[1:-1]).astype(np.int64)
    output_trials.append(bin_labels.reshape(1, -1))
```

```python
counts = np.bincount(all_output.astype(int), minlength=N_OUTPUT_BINS)
...
all_neural = np.concatenate([t[0, :] for t in neural_sess])
all_time = np.concatenate([t[0, :] for t in input_sess])
all_output = np.concatenate([t[0, :] for t in output_sess])
```

iii. The notes do not explicitly discuss this repeated processing; the only general justification given is that the code is “vectorized throughout.”

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `ops.npy` only to recover `fs` even though the workflow otherwise assumes a fixed 30 Hz rate; it defines an unused `discretize_to_bins` helper; it keeps an unused `fig_data` argument; and the optional plotting and extensive summary printing are not used in downstream decoder analyses.

ii.
```python
ops = np.load(os.path.join(session_dir, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
fs = ops.get('fs', FRAME_RATE)
```

```python
def discretize_to_bins(values, n_bins=5):
    ...

def process_session(mouse, session, session_dir, show_processing=False, fig_data=None):
    ...
```

```python
if show_processing:
    plot_processing(data, mice_to_process)
```

iii. The notes do not defend these extra steps directly; they only state that the script should support `--show-processing` and that no major inefficiencies were identified.
