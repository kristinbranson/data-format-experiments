# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for all subject directories, scans each subject directory for session subdirectories, and loads each session from Suite2p plus behavior files. Specifically, it loads `F.npy`, `Fneu.npy`, `ops.npy`, and `motion_energy_glob.npy`. Trials are not loaded from disk; they are created later by splitting each continuous session in `process_session()`.

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
...
sessions = sorted([d for d in os.listdir(mouse_dir) 
                  if os.path.isdir(os.path.join(mouse_dir, d))])
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset consists of 6 mouse subjects with 41 session directories, each containing Suite2p outputs plus `move_deve` behavior files. In the trajectory, it repeatedly states that the provided data already contains tracked neurons and that each subject/session folder should be processed as one continuous recording.

## 1-b. How are the data split into subjects?

i. Subjects are split by taking every directory directly under `data/`, sorting them alphabetically, and treating each as one mouse.

ii. 
```python
mice = sorted([d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))])
```

iii. In `CONVERSION_NOTES.md`, the AI documents 6 subject directories (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and treats each as one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are split as sorted subdirectories inside each mouse directory. Each session subdirectory is treated as one recording session.

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

iii. The notes describe each subdirectory as one daily recording and emphasize deterministic ordering by sorting the folder names.

## 1-d. How are the data split into trials?

i. The AI assumes the sessions are continuous recordings with no natural trial markers. It bins the full session by 10 frames first, then splits the binned time series into contiguous 120-second blocks. At 30 Hz with binning to 3 Hz, each trial has 360 time bins.

ii. 
```python
bin_size = 10
trial_duration_sec = 120
...
effective_fs = fs / bin_size
trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins

for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. In both `CONVERSION_NOTES.md` and the trajectory, the AI explicitly justifies this as matching the paper's cross-validation over consecutive 2-minute blocks, and says those blocks should serve as the decoder "trials."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply explicit trial quality-control filtering. The only exclusion is implicit: incomplete trailing data are dropped when the number of bins does not divide evenly into 120-second trial blocks.

ii. 
```python
trial_bins = int(trial_duration_sec * effective_fs)
n_trials = n_bins // trial_bins

for t in range(n_trials):
    start = t * trial_bins
    end = (t + 1) * trial_bins
```

iii. In the notes, the AI says the data are continuous recordings and that the paper does not define trial-level curation rules, so it simply splits them into fixed blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from Suite2p fluorescence files `F.npy` and `Fneu.npy`, and uses `ops.npy` to read preprocessing parameters such as frame rate and neuropil coefficient.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. The notes say the paper used baseline-corrected fluorescence traces from Suite2p, so the AI chose raw fluorescence plus neuropil traces, with parameters taken from the Suite2p `ops` file.

## 2-b. How is the `neural` data processed?

i. The AI manually implements a Suite2p-style baseline correction: neuropil subtraction (`F - neucoeff * Fneu`), Gaussian smoothing, minimum filter, maximum filter, and baseline subtraction. It then averages every 10 consecutive frames.

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

iii. `CONVERSION_NOTES.md` says the AI initially implemented the baseline incorrectly, then revised it to "match Suite2p exactly" after reading Suite2p and Track2p code. The notes also justify 10-frame averaging as matching the paper's denoising for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no extra neural quality filter in the conversion script. It does not use `iscell.npy` or any ROI mask at runtime, so all rows in `F.npy` are included.

ii. 
```python
return {
    'F': F,
    'Fneu': Fneu,
    'motion_energy': me,
    'ops': ops,
    'n_neurons': n_neurons,
    'n_frames': n_frames,
}
```

iii. In the notes, the AI says it inspected `iscell.npy`, found all ROIs already marked as cells in the provided data, and therefore concluded no further filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the recording session. It does not use any event marker from the raw data. Each trial is just a contiguous block from that session-level timeline.

ii. 
```python
time_vec = np.arange(n_bins) / effective_fs
...
neural_trial = dff_binned[:, start:end].astype(np.float32)
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

```python
'metadata': {
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
```

iii. The notes argue that the experiment is spontaneous behavior with continuous recording rather than event-triggered trials, so session start is the relevant alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI converts the data to 3 Hz by averaging every 10 original imaging frames. The final time bin size is `1000 * 10 / 30 = 333.33 ms`.

ii. 
```python
bin_size = 10
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
...
'time_bin_size': 1000.0 * bin_size / 30.0,
'effective_frame_rate_hz': 3.0,
```

iii. In the notes and trajectory, the AI repeatedly cites the methods text saying the authors "averaged in bins of 10 consecutive timestamps" for decoding, and uses that as the reason to downsample both neural and behavior traces.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The AI does not load a time variable from disk. It derives input time from the index of each binned frame together with the frame rate from `ops.npy`.

ii. 
```python
fs = ops.get('fs', 30.0)
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
```

iii. The notes say time should be "elapsed in seconds from start of recording" and that no explicit timestamp stream is needed because the frame rate is fixed.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI first rebins the session to 3 Hz, then constructs a single session-level time vector in seconds using `np.arange(n_bins) / effective_fs`, and finally slices it into per-trial arrays.

ii. 
```python
effective_fs = fs / bin_size
time_vec = np.arange(n_bins) / effective_fs
...
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. The trajectory says the input should be "time elapsed in seconds from start of recording" at the same 3 Hz resolution as the binned neural data.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time to neural data by using the same binned session length and the same `start:end` trial slices for both arrays. Each time point corresponds to one neural bin.

ii. 
```python
neural_trial = dff_binned[:, start:end].astype(np.float32)
time_trial = time_vec[start:end].reshape(1, -1).astype(np.float32)
```

iii. The notes describe time as the decoder input paired with the neural matrix after the same 10-frame averaging and the same 2-minute trial boundaries.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI derives motion energy only from `move_deve/motion_energy_glob.npy`.

ii. 
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The notes identify `motion_energy_glob.npy` as the behavioral variable from videography and treat it as the source signal to decode.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the motion-energy trace to the neural frame count whenever lengths differ, then averages every 10 frames, and only later discretizes the resulting values into 5 percentile bins. It does not normalize motion energy by session standard deviation before discretization.

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

iii. In the notes and trajectory, the AI says the paper mentions missing camera frames that should be interpolated, and also says both dF/F and behavior should be binned by 10 frames for decoding.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI pools all per-trial motion-energy values across sessions, computes 5 equal-percentile bins, sets the outer edges to `-inf` and `inf`, and digitizes every value into integers `0` through `4`.

ii. 
```python
all_values = np.concatenate(all_values)
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(all_values, percentiles)

bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
...
binned = np.digitize(trial.flatten(), bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1).astype(np.int64)
```

iii. The notes say the decoder output must be categorical with 5 equal-percentile bins, and the AI uses global percentile thresholds to produce balanced classes overall.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first resamples motion energy to the neural frame count, then rebins both streams by 10 frames, and finally slices both by the same trial boundaries. That gives one motion-energy category per neural time bin.

ii. 
```python
me = interpolate_missing_frames(me, n_frames)
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
...
me_trial = me_binned[start:end].reshape(1, -1).astype(np.float32)
```

iii. The notes say the video and imaging streams are synchronous except for occasional dropped camera frames, so interpolation plus shared binning and shared trial boundaries should realign them.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing motion-energy frames by linear interpolation to the neural frame count. It handles non-divisible trailing data by truncation during binning and again when computing the number of full 120-second trials. It does not add assertions around these repairs.

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

iii. The notes repeatedly mention that some sessions have missing camera frames and say those should be interpolated. They also describe the recordings as continuous blocks, so the AI accepts trimming leftover tail segments.

## 6-a. What are the most time-consuming steps of the code?

i. The code's main cost is per-session neural preprocessing on large arrays: `compute_dff()` runs Gaussian, minimum, and maximum filters over every neuron trace. Session-wide binning and the final global motion-energy discretization are also substantial. If `--show-processing` is enabled, the script recomputes preprocessing for plotted sessions.

ii. 
```python
dff = compute_dff(F, Fneu, 
                  neucoeff=neucoeff,
                  win_baseline=win_baseline,
                  sig_baseline=sig_baseline,
                  fs=fs)
...
dff_binned = bin_data(dff, bin_size=bin_size, axis=1)
me_binned = bin_data(me.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

```python
all_output, bin_edges = discretize_output(all_me_raw, n_bins=n_output_bins)
```

iii. `CONVERSION_NOTES.md` focuses most of its validation effort on the baseline-correction path and reports per-session processing times, which shows the AI regarded this as the expensive step.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several Python loops in place that could be vectorized: the per-trial slicing loop in `process_session()`, the nested loops over all sessions and trials in `discretize_output()`, and the summary loops that repeatedly concatenate data.

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
for session_trials in all_me_trials:
    for trial in session_trials:
        all_values.append(trial.flatten())
...
for session_trials in all_me_trials:
    session_output = []
    for trial in session_trials:
        binned = np.digitize(trial.flatten(), bin_edges[1:-1])
```

iii. The AI does not explicitly justify these loops in the notes; they appear to be straightforward implementation choices rather than deliberate optimization decisions.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats preprocessing work in the optional plotting path: it reloads sessions and recomputes `compute_dff()`, interpolation, and binning for sessions already processed in the main conversion path. It also repeatedly flattens and concatenates outputs for discretization and for summary printing.

ii. 
```python
for i, (mouse, mouse_idx, sess_dir, sess_name) in enumerate(all_sessions):
    ...
    neural_trials, input_trials, me_trials = process_session(...)
```

```python
if args.show_processing:
    ...
    sess_data = load_session_data(sess_dir)
    ...
    dff = compute_dff(sess_data['F'], sess_data['Fneu'], ...)
    ...
    me_interp = interpolate_missing_frames(me, sess_data['n_frames'])
    me_binned_plot = bin_data(me_interp.reshape(1, -1), bin_size=bin_size, axis=1).flatten()
```

iii. The notes present the plotting and validation checks as sanity checks, so this repeated computation seems to have been accepted as a convenience for inspection rather than avoided.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the normal conversion path, the code computes and prints extensive summaries that are not used downstream, and it retains continuous motion-energy trials in `all_me_raw` only as an intermediate before categorization. In the optional `--show-processing` path, it performs extra recomputation and plotting purely for diagnostics.

ii. 
```python
all_me_raw.append(me_trials)
...
all_output, bin_edges = discretize_output(all_me_raw, n_bins=n_output_bins)
```

```python
print(f"\n=== Data Summary ===")
...
all_out_vals = np.concatenate([t.flatten() for s in result['output'] for t in s])
...
if args.show_processing:
    ...
    plt.savefig(f'processing_{mouse}_{sess_name}.png', dpi=100)
```

iii. The notes explicitly frame these steps as validation and sanity-check machinery rather than part of the decoder-facing representation itself.
