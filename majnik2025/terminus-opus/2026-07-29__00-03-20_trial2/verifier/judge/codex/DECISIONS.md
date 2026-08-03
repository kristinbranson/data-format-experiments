# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script treats every directory under `data/` as a mouse, every subdirectory under each mouse as a session, and then loads session-level arrays from Suite2p and behavioral files. Specifically, it loads `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`, then processes each session into trial lists.

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

all_sessions = []
for mouse_idx, mouse in enumerate(mice):
    mouse_dir = os.path.join(data_dir, mouse)
    sessions = sorted([d for d in os.listdir(mouse_dir) 
                      if os.path.isdir(os.path.join(mouse_dir, d))])
    for sess in sessions:
        sess_dir = os.path.join(mouse_dir, sess)
        all_sessions.append((mouse, mouse_idx, sess_dir, sess))
```

iii. In `CONVERSION_NOTES.md`, the AI documented the dataset as six mouse folders with 6-7 session folders each and mapped `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy` into the converted dataset. In the trajectory, it explicitly concluded that the data are organized as mouse directories containing daily recording directories and then implemented loading around that structure.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the sorted directory names under `data/`. The script uses these directory names directly as mouse IDs in `subjects`.

ii. 
```python
mice = sorted([d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))])
...
if args.sample:
    unique_mice = list(dict.fromkeys(s[0] for s in all_sessions))
else:
    unique_mice = mice
...
'subjects': unique_mice,
```

iii. The notes list six subject folders (`jm031` through `jm046`) and treat each as one mouse. The trajectory repeatedly refers to “6 mice” discovered from the top-level directory listing and uses those folder names as the subject identities.

## 1-c. How are the data split into sessions?

i. Sessions are defined as the sorted subdirectories within each mouse directory. Each one becomes one entry in `all_sessions` and ultimately one session in `neural`, `input`, and `output`.

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

iii. The notes describe each mouse as having 6-7 sessions and list one session folder per daily recording. The trajectory also states that each subject folder contains daily session folders and that those are the natural session units to convert.

## 1-d. How are the data split into trials?

i. The AI decided that sessions should be chopped into artificial 2-minute trials after temporal binning. It first averages every 10 imaging frames, then defines each trial as `trial_duration_sec = 120`, giving 360 time bins per trial at 3 Hz.

ii. 
```python
bin_size = 10
trial_duration_sec = 120
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
effective_fs = fs / bin_size
trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins
...
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. In `CONVERSION_NOTES.md`, the AI’s “Key Decisions” section states: “Trials: 2-minute blocks (matching paper CV).” In the trajectory, it justified this by pointing to the paper’s 5-fold cross-validation on consecutive 2-minute blocks and chose to reuse those blocks as the trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The script does not apply explicit trial-quality filtering. It keeps every full 2-minute block. The only filtering-like behavior is structural truncation: `bin_data()` drops leftover frames that do not fill a 10-frame bin, and `n_trials = n_bins // trial_bins` drops leftover binned timepoints that do not fill a full 2-minute trial.

ii. 
```python
def bin_data(data, bin_size=10, axis=-1):
    n = data.shape[axis]
    n_bins = n // bin_size
    ...
    slices[axis] = slice(0, n_bins * bin_size)
    data_trimmed = data[tuple(slices)]
```

```python
trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins
...
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. The notes and trajectory focus on fixed blocking rather than quality-based trial exclusion, and they do not describe any behavioral or neural trial QC. The notes only discuss interpolation of missing camera frames and fixed 2-minute trial construction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from raw fluorescence `F.npy` and neuropil fluorescence `Fneu.npy`. It also reads `ops.npy` to obtain processing parameters such as frame rate and baseline settings.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. The notes map `F.npy` and `Fneu.npy` to the target `neural` field and summarize Suite2p defaults such as `neucoeff = 0.7`, `fs = 30`, `win_baseline = 60`, and `sig_baseline = 10`, which the script pulls from `ops.npy` when available.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction and then a manual Suite2p-style maximin baseline correction: Gaussian smoothing, minimum filter, maximum filter, and subtraction of the baseline estimate. After that, the processed trace is averaged in non-overlapping bins of 10 frames before trial splitting.

ii. 
```python
def compute_dff(F, Fneu, neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0, fs=30.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter1d(Fc, sigma=sig_baseline, axis=1)
    Flow = minimum_filter1d(Flow, size=win, axis=1)
    Flow = maximum_filter1d(Flow, size=win, axis=1)
    dff = Fc - Flow
    return dff
```

```python
dff = compute_dff(F, Fneu, 
                  neucoeff=neucoeff,
                  win_baseline=win_baseline,
                  sig_baseline=sig_baseline,
                  fs=fs)
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

iii. The notes explicitly say the script was corrected to “match Suite2p exactly” and list the final order as Gaussian smoothing, then min, then max, then subtraction. The trajectory shows the AI revising earlier attempts until it decided the Suite2p/Track2p processing was subtraction-only baseline correction with a 60 s window and then 10-frame averaging for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neural quality-control filter is applied in the script. It uses all rows of `F.npy`/`Fneu.npy` for each session and never loads or thresholds `iscell.npy`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
```

```python
all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. In `CONVERSION_NOTES.md`, the AI wrote that all ROIs in `iscell.npy` were already classified as cells and therefore no additional filtering was needed. The trajectory also records an explicit check of `iscell` followed by the conclusion that all ROIs passed the default threshold and should all be kept.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to recording-session start, not to any experimental event within the recording. The script constructs a session-relative time axis and uses the same session-relative block boundaries for every trial.

ii. 
```python
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
...
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    neural_trial = dff_binned[:, start:end].astype(np.float32)
```

```python
'metadata': {
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
```

iii. The notes map the decoder input to “Time in seconds at 3 Hz” and treat the recording as a continuous session rather than an event-locked trial task. The trajectory says there is no task event to align to, so the AI chose session start as the alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are at 3 Hz, with one bin every 10 original imaging frames. Both neural and motion-energy streams are rebinned by averaging consecutive groups of 10 frames.

ii. 
```python
bin_size = 10
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

```python
'metadata': {
    'time_bin_size': 1000.0 * bin_size / 30.0,
    'bin_size_frames': bin_size,
    'original_frame_rate_hz': 30.0,
    'effective_frame_rate_hz': 3.0,
```

iii. The notes cite the paper’s statement that decoding analyses averaged “10 consecutive timestamps,” and the AI used that as justification for rebinned 3 Hz data. The trajectory repeatedly refers to “bin by 10 frames (30 Hz -> 3 Hz)” as a deliberate design choice.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time is not loaded from a timestamp file. It is synthesized from the binned sample index together with the session frame rate from `ops.npy` and the chosen `bin_size`.

ii. 
```python
ops = data['ops']
fs = ops.get('fs', 30.0)
...
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
```

iii. The notes’ variable mapping says “Time index -> input[0] | Time in seconds at 3 Hz,” which matches the script’s synthetic time-axis construction rather than any direct use of raw timestamp arrays such as `tstamps.npy`.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The script computes a dense session-relative time vector after 10-frame binning, then slices that vector into 2-minute trial segments. The resulting input is a `(1, time)` array per trial in seconds.

ii. 
```python
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
input_trials.append(time_trial)
```

iii. The notes justify this through the same 10-frame averaging decision used for decoding, and the trajectory links the time input to the AI’s chosen 3 Hz representation and 2-minute block structure.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by construction: it uses the same rebinned sampling rate and the same `start:end` indices used to slice the neural data into trial blocks.

ii. 
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    
    neural_trial = dff_binned[:, start:end].astype(np.float32)
    time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. In the trajectory and notes, the AI treats the time signal as simply “time elapsed from start of recording” at the same 3 Hz grid as the rebinned neural data. The notes’ validation section also says it checked that trial boundaries for the time input were correct.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In the script itself, output motion energy is derived from `move_deve/motion_energy_glob.npy`. The script does not load `interframe_int.npy` or `tstamps.npy`.

ii. 
```python
move_dir = os.path.join(session_dir, 'move_deve')
...
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The notes identify `motion_energy_glob.npy` as the behavioral source variable. They also mention missing camera frames in the dataset exploration, but the implementation ultimately uses only the motion-energy vector plus target frame count.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the motion-energy trace if its length does not match the neural frame count, averages it in 10-frame bins, and later discretizes all binned values globally. The script does not normalize each session’s motion-energy trace before thresholding.

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

iii. The notes state that behavior traces should be binned by 10 and that missing camera frames should be interpolated. In the trajectory, the AI explicitly debated whether to add per-session normalization, then decided to keep the existing global percentile approach rather than z-scoring motion energy first.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is converted into five global equal-percentile bins. The script concatenates all binned motion-energy values from all sessions/trials, computes percentile edges, then digitizes each trial into labels `0` through `4`.

ii. 
```python
def discretize_output(all_me_trials, n_bins=5):
    all_values = []
    for session_trials in all_me_trials:
        for trial in session_trials:
            all_values.append(trial.flatten())
    all_values = np.concatenate(all_values)
    
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_values, percentiles)
    
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
```

```python
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` lists “ME discretization: 5 equal-percentile bins globally” as a key decision. In the trajectory, the AI considered per-session normalization and per-session balancing but ultimately kept global percentile binning because it wanted to preserve differences between less-active and more-active sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by first resampling the motion-energy vector to the neural frame count, then applying the same 10-frame averaging and the same 2-minute trial boundaries used for neural data.

ii. 
```python
me = interpolate_missing_frames(me, n_frames)
...
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    me_trial = me_binned[start:end].reshape(1, -1).astype(np.float32)
```

iii. The notes and trajectory both identify missing camera frames as the main alignment issue and justify interpolation so motion energy can be put on the same sample grid as neural data. The rest of the alignment follows directly from the AI’s choice to use the same rebinned session grid for every stream.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles motion-energy length mismatches by global linear interpolation to the neural frame count. It also silently discards leftover frames that do not fit complete 10-frame bins or complete 2-minute trials. It does not include an explicit assertion or a dropped-frame-specific repair step.

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
slices[axis] = slice(0, n_bins * bin_size)
...
n_trials = n_bins // trial_bins
```

iii. The notes repeatedly mention “Missing camera frames: interpolate.” The trajectory also records that the AI observed ME/neural length mismatches and chose interpolation as the repair mechanism. No stronger handling or explicit QC assertion appears in the final script.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session neural preprocessing over full fluorescence matrices: loading `F`/`Fneu`, computing Suite2p-style baseline-corrected traces, then binning them. Optional plotting also recomputes processing for displayed sessions.

ii. 
```python
for i, (mouse, mouse_idx, sess_dir, sess_name) in enumerate(all_sessions):
    ...
    neural_trials, input_trials, me_trials = process_session(
        sess_dir, bin_size=bin_size, trial_duration_sec=trial_duration_sec
    )
```

```python
dff = compute_dff(F, Fneu, 
                  neucoeff=neucoeff,
                  win_baseline=win_baseline,
                  sig_baseline=sig_baseline,
                  fs=fs)
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

iii. The notes emphasize the baseline-correction pipeline and report total processing times for sample and full conversions. The trajectory also describes full conversion time primarily in terms of processing each session’s neural data, which implies that `compute_dff()` is the main bottleneck.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorizable loops are the Python loops that split sessions into trial lists and the nested loops that flatten and discretize motion-energy trials. The optional plotting loops over neurons are also serial.

ii. 
```python
for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
    ...
    neural_trials.append(neural_trial)
    input_trials.append(time_trial)
    me_trials.append(me_trial)
```

```python
all_values = []
for session_trials in all_me_trials:
    for trial in session_trials:
        all_values.append(trial.flatten())
...
for session_trials in all_me_trials:
    session_output = []
    for trial in session_trials:
        binned = np.digitize(trial.flatten(), bin_edges[1:-1])
```

iii. The AI does not explicitly discuss vectorization in its notes, so this is mostly an inference from the implementation. The trajectory shows it focused on correctness and validation rather than on optimizing these loops.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats baseline correction and binning for the same sessions when `--show-processing` is enabled, because it recomputes `compute_dff()` and `bin_data()` for plotting after already processing those sessions for the output dataset. It also first stores continuous motion-energy trials in `all_me_raw` and then loops over them again for discretization.

ii. 
```python
all_me_raw.append(me_trials)
...
all_output, bin_edges = discretize_output(all_me_raw, n_bins=n_output_bins)
```

```python
if args.show_processing:
    ...
    dff = compute_dff(sess_data['F'], sess_data['Fneu'],
                      neucoeff=ops.get('neucoeff', 0.7),
                      win_baseline=ops.get('win_baseline', 60.0),
                      sig_baseline=ops.get('sig_baseline', 10.0),
                      fs=ops.get('fs', 30.0))
    ...
    dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
```

iii. There is no explicit justification in the notes beyond the desire to generate processing visualizations and validation artifacts. The trajectory shows repeated reruns and verification-oriented plotting, which is consistent with this duplicated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script keeps the continuous binned motion-energy trials in `all_me_raw` only long enough to convert them into categorical outputs, after which the continuous values are not preserved in the saved dataset. With `--show-processing`, it also recomputes processed neural and motion-energy traces solely for plots that are not part of the final pickle.

ii. 
```python
all_me_raw = []
...
all_me_raw.append(me_trials)
...
all_output, bin_edges = discretize_output(all_me_raw, n_bins=n_output_bins)
...
'output': all_output,
```

```python
if args.show_processing:
    ...
    dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)
    ...
    me_interp = interpolate_missing_frames(me, sess_data['n_frames'])
    me_binned_plot = bin_data(me_interp.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

iii. The notes justify these steps only indirectly through validation and documentation. The trajectory shows the AI wanted plots and rich metadata, but those extra computations are not needed by the downstream decoder once the categorical `output` arrays have been produced.
