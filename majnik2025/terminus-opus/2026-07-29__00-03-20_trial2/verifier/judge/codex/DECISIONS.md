# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans every directory directly under `data/` as a subject, scans every subdirectory under each subject as a session, and then loads session-level arrays from `suite2p/plane0` and `move_deve`. It loads `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. Trials are not loaded directly from disk; they are created later by splitting continuous session data.

ii. 
```python
def load_session_data(session_dir):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

```python
mice = sorted([d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))])

for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir) 
                      if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. In `CONVERSION_NOTES.md`, the AI justified this from the observed directory structure: six top-level mouse folders, each containing daily session folders with Suite2p outputs and motion-energy files. The notes do not justify excluding or including subjects via a stricter name pattern.

## 1-b. How are the data split into subjects?

i. Subjects are defined as all directories immediately under `data/`, sorted alphabetically. The code does not restrict subjects to names matching `jm*`; it assumes every top-level directory is a mouse.

ii. 
```python
mice = sorted([d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))])
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI described the dataset as six mouse subject directories and treated those directories as the subject split.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories inside each mouse directory, again sorted alphabetically.

ii. 
```python
for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir) 
                      if os.path.isdir(os.path.join(mouse_dir, d))])
    for sess in sessions:
        sess_dir = os.path.join(mouse_dir, sess)
        all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

iii. The AI’s notes describe each mouse as having 6-7 daily sessions and explicitly list session subdirectories under each subject.

## 1-d. How are the data split into trials?

i. The AI treats each session as a continuous recording, bins it to 3 Hz, and then splits it into non-overlapping 120-second trials. Trials are therefore artificial 2-minute blocks rather than natural task trials.

ii. 
```python
def process_session(session_dir, bin_size=10, trial_duration_sec=120):
    ...
    effective_fs = fs / bin_size
    trial_bins = int(trial_duration_sec * effective_fs)
    n_trials = n_bins // trial_bins
    ...
    for t in range(n_trials):
        start = t * trial_bins
        end = (t + 1) * trial_bins
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly justified this with “Trials: 2-minute blocks (matching paper CV).” The trajectory makes the same argument: because the paper mentions 5-fold cross-validation over consecutive 2-minute blocks, the AI treated those blocks as trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-quality filtering. It simply keeps all complete 2-minute trial blocks and implicitly drops any leftover bins that do not fill a full trial.

ii. 
```python
trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins

for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    ...
```

iii. No explicit trial filtering rationale appears in the notes. The only implicit rule is to keep full fixed-length blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural data is derived from raw fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`. The code also reads `ops.npy` to get preprocessing parameters such as `fs`, `neucoeff`, `win_baseline`, and `sig_baseline`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. `CONVERSION_NOTES.md` Step 5 maps `F.npy, Fneu.npy` to the target `neural` field and Step 1 states that Suite2p baseline correction should be applied to those traces.

## 2-b. How is the `neural` data processed?

i. The AI performs manual Suite2p-style preprocessing: neuropil subtraction (`F - neucoeff * Fneu`), Gaussian smoothing, min filter, max filter, baseline subtraction, then non-overlapping averaging in 10-frame bins. This produces baseline-corrected fluorescence, not deconvolved spikes.

ii. 
```python
Fc = F - neucoeff * Fneu
win = int(win_baseline * fs)
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
dff = Fc - Flow
```

```python
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

iii. In `CONVERSION_NOTES.md` Steps 1, 5, 6, and 10, the AI repeatedly justifies this as matching Suite2p’s `maximin` baseline method: Gaussian smoothing, min filter, max filter, and subtraction rather than division.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply any additional neuron filtering in the conversion code. It uses every row in `F.npy`/`Fneu.npy` for each session.

ii. 
```python
n_neurons, n_frames = F.shape
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 1, the AI states that all ROIs in `iscell.npy` were classified as cells and that the provided data already contains tracked neurons across days, so it decided not to add extra filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of each recording session. After session-wide binning, each trial is a contiguous slice from the binned session trace. The metadata names the alignment event as the start of the recording session.

ii. 
```python
time_vec = np.arange(n_bins) / effective_fs
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
```

```python
'metadata': {
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The justification is implicit in the notes: the recording is treated as continuous, and the input is defined as elapsed time from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins both neural and behavioral data into non-overlapping bins of 10 original frames at 30 Hz, giving an effective sampling rate of 3 Hz and a time bin size of 333.33 ms.

ii. 
```python
bin_size = 10
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

```python
'time_bin_size': 1000.0 * bin_size / 30.0,
'effective_frame_rate_hz': 3.0,
```

iii. In `CONVERSION_NOTES.md` Steps 1, 3, and 5, the AI cites the paper’s decoding analysis description that traces were averaged in bins of 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not read from a dedicated timestamp array. It is derived from the binned frame index together with `fs` from `ops.npy` and the fixed bin size.

ii. 
```python
ops = data['ops']
fs = ops.get('fs', 30.0)
...
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
```

iii. The AI’s notes describe the decoder input as “Time in seconds at 3 Hz” and treat constant-rate frame indexing as sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes a session-long time vector in seconds after temporal binning, then slices that vector into per-trial segments and stores each as shape `(1, n_timepoints)`.

ii. 
```python
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
input_trials.append(time_trial)
```

iii. The justification is implicit: once the effective frame rate is 3 Hz, the bin index uniquely determines elapsed seconds from session start.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time and neural data by generating `time_vec` on the same binned session axis as `dff_binned` and then slicing both arrays with the same `start:end` trial boundaries.

ii. 
```python
time_vec = np.arange(n_bins) / effective_fs
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. No separate justification is given beyond the continuous-session representation used throughout the script.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In the final code, motion-energy output is derived only from `motion_energy_glob.npy`. The code does not load `interframe_int.npy` or `tstamps.npy` when producing the output variable.

ii. 
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
...
return {
    ...
    'motion_energy': me,
```

iii. The notes recognize that missing camera frames exist, but the final implementation simplifies this to generic interpolation to target length rather than using frame-interval metadata.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI optionally rescales motion energy to match the neural frame count using linear interpolation over normalized sample positions, bins the result into 10-frame averages, and later discretizes it. It does not detect dropped frames from `interframe_int.npy`.

ii. 
```python
def interpolate_missing_frames(me, n_target):
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp
```

```python
me = interpolate_missing_frames(me, n_frames)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

iii. The trajectory shows the AI knew missing camera frames existed, but the final code adopts a simpler “interpolate to the target length” rule rather than reproducing the reference dropped-frame repair procedure.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates motion-energy values across all sessions and all trials, computes one global set of percentile edges, and discretizes every session using those shared edges into five bins.

ii. 
```python
all_values = []
for session_trials in all_me_trials:
    for trial in session_trials:
        all_values.append(trial.flatten())
all_values = np.concatenate(all_values)

percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)
```

```python
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly states “ME discretization: 5 equal-percentile bins globally.” In the trajectory it argues that global binning preserves cross-session activity differences, and then keeps that design.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by forcing the motion-energy trace to the same frame count as the neural trace via interpolation, binning both streams the same way, and then slicing them with identical trial boundaries.

ii. 
```python
n_neurons, n_frames = F.shape
...
me = interpolate_missing_frames(me, n_frames)
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
me_trial = me_binned[start:end].reshape(1, -1).astype(np.float32)
```

iii. The AI’s notes justify alignment in terms of synchronous 30 Hz acquisition and occasional missing camera frames, but the implemented method is generic interpolation rather than dropped-frame insertion guided by the frame-interval file.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The only explicit error-handling path is for motion-energy length mismatch: if the motion-energy array length differs from the neural frame count, the AI linearly interpolates the full trace to the target length. The code also silently drops incomplete trailing bins in `bin_data` and silently drops any leftover binned samples that do not fill a full trial.

ii. 
```python
def interpolate_missing_frames(me, n_target):
    if len(me) == n_target:
        return me
    x_orig = np.linspace(0, 1, len(me))
    x_target = np.linspace(0, 1, n_target)
    me_interp = np.interp(x_target, x_orig, me.astype(float))
    return me_interp
```

```python
n_bins = n // bin_size
...
data_trimmed = data[tuple(slices)]
...
n_trials = n_bins // trial_bins
```

iii. The notes mention “Missing camera frames: interpolate,” but do not add additional safeguards such as assertions against unresolved mismatch.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is the per-session fluorescence preprocessing: the Gaussian, minimum-filter, and maximum-filter passes over large `(neurons, frames)` arrays. Secondary costs are loading the large NumPy arrays and, if enabled, recomputing the same processing for plotting.

ii. 
```python
Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
```

```python
for i, (mouse, mouse_idx, sess_dir, sess_name) in enumerate(all_sessions):
    neural_trials, input_trials, me_trials = process_session(...)
```

iii. `CONVERSION_NOTES.md` Step 11 reports session processing times and treats the Suite2p-style baseline computation as the central preprocessing step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities in this code are the Python loops that split session data into trials and the nested loops used to flatten motion-energy trials before discretization. Those loops build lists trial-by-trial even though they operate on regular block boundaries.

ii. 
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
    me_trial = me_binned[start:end].reshape(1, -1).astype(np.float32)
```

```python
all_values = []
for session_trials in all_me_trials:
    for trial in session_trials:
        all_values.append(trial.flatten())
all_values = np.concatenate(all_values)
```

iii. The AI does not explicitly discuss these vectorization opportunities in the notes; this is inferred from the final code structure.

## 6-c. What processing does the code repeat multiple times?

i. If `--show-processing` is enabled, the code reloads session data and recomputes baseline correction and binning for plotted sessions after already having processed those same sessions for the dataset. It also traverses all output trials again to compute global bin edges and summary counts.

ii. 
```python
neural_trials, input_trials, me_trials = process_session(
    sess_dir, bin_size=bin_size, trial_duration_sec=trial_duration_sec
)
```

```python
if args.show_processing:
    ...
    sess_data = load_session_data(sess_dir)
    ...
    dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)
    dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

iii. There is no explicit justification in the notes for this recomputation; it appears to be a convenience choice for optional diagnostics.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the default conversion path, there is not much obviously unnecessary scientific processing, but the code does load the full `ops.npy` dict to use only a few scalar parameters and builds extra metadata such as `session_info` and `me_bin_edges` that the downstream decoder does not require. The optional plotting branch also performs additional processing purely for visualization.

ii. 
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
...
'session_info': session_info,
'me_bin_edges': bin_edges.tolist(),
```

```python
if args.show_processing:
    ...
    plt.savefig(f'processing_{mouse}_{sess_name}.png', dpi=100)
```

iii. The notes justify plotting and extensive metadata as documentation and sanity-check support rather than as inputs to downstream analysis.
