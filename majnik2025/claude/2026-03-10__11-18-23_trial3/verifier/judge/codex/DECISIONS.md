# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by scanning the local `data` directory for folders whose names start with `jm`, then discovers sessions as all subdirectories inside each subject folder. In full mode it iterates every discovered subject/session pair. For each session it loads calcium traces from `suite2p/plane0/F.npy` and `Fneu.npy`, and motion energy from `move_deve/motion_energy_glob.npy`. Trials are not loaded from disk directly; they are created later by splitting each processed session into fixed-size blocks.

ii.
```python
def get_subjects_and_sessions():
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
    all_sessions = {}
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted([d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d))])
        all_sessions[subj] = sessions
    return subjects, all_sessions
```

```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

```python
for subj in subjects_to_process:
    sessions = sessions_to_process[subj]
    for i, session in enumerate(sessions):
        neural_trials, input_trials, output_trials, n_neurons = process_session(
            subj, session, show_processing=(show_processing and session_count < 2),
            ax_list=fig_ax
        )
```

iii. In `CONVERSION_NOTES.md`, the AI justified this from the observed directory structure: 6 `jm*` subject folders, date-labeled session folders inside each subject, and a standard per-session layout with `suite2p/plane0` and `move_deve`. The notes also state that the provided data is already Track2p-matched, so each session can be processed independently from those files.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories whose names start with `jm`, sorted lexicographically. In the output dictionary, the AI renames those subjects to `Mouse_A` through `Mouse_F` using a hard-coded map, and records per-session membership in `subject_idx`.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
```

```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}
```

```python
subject_names = list(SUBJECT_MAP.values()) if not sample_mode else [SUBJECT_MAP[subjects_to_process[0]]]
subj_idx = subject_names.index(SUBJECT_MAP[subj])
subject_idx_list.append(subj_idx)
```

iii. The notes say each `jm*` directory corresponds to one mouse. The rename appears to be a presentation choice rather than a data-driven one; the notes describe the subjects as `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`, then later refer to them as Mouse A-F.

## 1-c. How are the data split into sessions?

i. Sessions are split as all subdirectories under each subject directory, again sorted lexicographically. In full mode the AI keeps every session; in sample mode it keeps only the first two sessions of the first subject.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    all_sessions[subj] = sessions
```

```python
if sample_mode:
    subjects_to_process = [subjects[0]]
    sessions_to_process = {subjects[0]: all_sessions[subjects[0]][:2]}
else:
    subjects_to_process = subjects
    sessions_to_process = all_sessions
```

iii. `CONVERSION_NOTES.md` describes the session folders as daily recordings with date-like names and lists the counts per mouse. The AI treated each of those subdirectories as one session because each contains a full set of Suite2p and motion-energy files.

## 1-d. How are the data split into trials?

i. The AI decided there is no native trial structure and therefore creates artificial trials as consecutive 2-minute blocks after first averaging every 10 frames. At 30 Hz, 10-frame bins give 3 Hz time bins, and each 2-minute trial contains 360 bins. Any final partial block is dropped implicitly by floor division.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial
```

```python
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
```

```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI explicitly justified this in `CONVERSION_NOTES.md` and the trajectory by citing the paper’s decoder methods: “splits were done on consecutive 2 minute blocks of the recording” and the recording was “continuous spontaneous recording” with no discrete trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies no explicit quality-control filtering at the trial level. The only effective filtering is structural: `bin_timeseries` truncates leftover frames that do not fill a complete 10-frame bin, and `n_trials = n_total_bins // TRIAL_BINS` drops any final partial 2-minute block.

ii.
```python
def bin_timeseries(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

```python
n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
```

iii. In `CONVERSION_NOTES.md`, the AI wrote “No explicit trial curation (continuous spontaneous recording)” and separately noted that missing camera frames should be interpolated. There is no further justification or code for rejecting noisy or low-quality trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from raw Suite2p fluorescence traces `F.npy` and neuropil traces `Fneu.npy` from `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

```python
Fc = F - NEUROPIL_COEFF * Fneu
```

iii. The notes say Track2p itself does not compute dF/F, so the AI concluded it needed to compute the neural signal post hoc from the stored Suite2p fluorescence arrays.

## 2-b. How is the `neural` data processed?

i. The AI performs neuropil subtraction (`F - 0.7 * Fneu`), then applies Suite2p’s `preprocess` function with `maximin` baseline parameters to produce baseline-corrected traces. After that, it averages every 10 frames before trialization.

ii.
```python
Fc = F - NEUROPIL_COEFF * Fneu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)

return dfof.astype(np.float32)
```

```python
dfof_binned = bin_timeseries(dfof, BIN_SIZE)  # (n_neurons, n_bins)
```

iii. The justification appears in both the notes and trajectory: the paper says the authors used “baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters),” and for decoding they “slightly denoised the dF/F … by averaging in bins of 10 consecutive timestamps.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is performed in the final script. The AI uses every row present in `F.npy`/`Fneu.npy` and never loads or applies `iscell.npy`, any ROI mask, or any activity-based exclusion rule.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_frames = F.shape
```

iii. The notes justify this by saying the provided data is already Track2p-matched and pre-filtered: “All iscell values are 1” and “F.npy contains only matched cells,” so no extra quality-control pass is necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the continuous recording session, not to an experimental stimulus/event. Each trial is just a consecutive block cut from that session-long axis.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'start of continuous recording session',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The notes say the recording is “continuous spontaneous recording” with “no discrete trials,” and the trajectory cites the paper’s description of frame-by-frame camera/microscope synchronization during an uninterrupted session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data from 30 Hz frames to 10-frame bins, giving one sample every 333.33 ms. This rebinning is applied to both neural and motion-energy time series before trial splitting.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # ~333.33 ms
```

```python
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The AI justified this by quoting the paper’s decoding section: “averaging in bins of 10 consecutive timestamps,” and repeated that rationale in both `CONVERSION_NOTES.md` and the script docstring.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time signal is not read from a timestamp file. It is synthesized from the bin index within the session, using the fixed frame rate and bin size.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. In the notes, the AI explicitly mapped the input to “Time index” and described the transform as `time_elapsed = bin_index * (10/30) seconds`, justified by the fixed 30 Hz acquisition rate and the requested decoder input “Time elapsed from the beginning of the experiment.”

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each artificial trial, the AI computes a monotonically increasing vector of binned time points, starting from the session-global bin indices `start:end`, multiplies by seconds per bin, reshapes to `(1, T)`, and casts to `float32`.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes say this input should be “time from start,” and the later sanity checks mention that “time inputs are continuous across trial boundaries,” which is why the AI uses global session bin indices rather than resetting time within each trial.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI aligns time to neural data by generating the time vector from the exact same binned `start:end` indices used to slice neural and output arrays for each trial.

ii.
```python
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS

neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI’s justification is implicit in the code and explicit in the notes: the three streams share a common 10-frame binned session axis, so using the same slice boundaries preserves alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In the final script, motion-energy output is derived only from `move_deve/motion_energy_glob.npy`. The script does not load `interframe_int.npy` or `tstamps.npy`, even though the notes mention those files when discussing missing frames.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The notes justify the general choice by saying `motion_energy_glob.npy` contains the precomputed global movement signal from the behavior video, and that the README allows missing camera frames to be interpolated. The final code operationalizes only the first part of that plan.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The final script linearly resamples motion energy to the neural frame count whenever lengths differ, then averages it in 10-frame bins. After binning, it immediately discretizes the values into 5 categories. The final code does not explicitly normalize the motion-energy trace by session standard deviation, despite the notes claiming that normalization.

ii.
```python
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)

    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp
```

```python
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)
```

iii. The trajectory and notes justify this with two ideas: the README says missing frames can be interpolated, and the paper says decoding used 10-frame averaging. The notes also state that motion energy should be “normalized per session,” but that stated normalization is absent from the final code.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds motion energy into 5 categories by computing the 20th, 40th, 60th, and 80th percentiles of each session’s own binned motion-energy trace, then using `np.digitize` to assign labels 0-4. The output labels are named `quintile_1` through `quintile_5`.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

```python
'output_values': [
    [f'quintile_{i+1}' for i in range(N_OUTPUT_BINS)]
]
```

iii. `CONVERSION_NOTES.md` explicitly says “Discretization per session: Quintile bins computed per session to handle different motion energy scales,” and later treats the resulting exact 20%/bin class balance as a sanity check.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by first resampling the motion-energy array to the neural frame count, then applying the same 10-frame binning and the same trial slice boundaries used for the neural signal.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
...
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
...
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes cite the paper’s description of camera/microscope synchronization at 30 Hz and the dataset README’s suggestion that missing camera frames can be interpolated. The final code therefore treats any length mismatch as a resampling problem rather than inserting specifically identified missing frames.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles minor data problems in a permissive way. If motion energy is shorter than the neural trace, it resamples the entire motion-energy sequence to the target length with `np.interp`. If the number of frames is not divisible by 10, `bin_timeseries` truncates the remainder. If the number of bins is not divisible by the 2-minute trial length, the final partial block is dropped. The final script does not assert exact alignment after repair and does not special-case NaNs or corrupted sessions.

ii.
```python
if len(me) == n_frames:
    return me.astype(np.float64)

x_orig = np.linspace(0, 1, len(me))
x_new = np.linspace(0, 1, n_frames)
me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
return me_interp
```

```python
n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]
```

```python
n_trials = n_total_bins // TRIAL_BINS
```

iii. The notes refer to the data README statement that missing motion-energy frames can be treated as missing or interpolated, and the AI chose interpolation. There is no further written justification for truncating tail bins/blocks beyond using fixed-size trials.

## 6-a. What are the most time-consuming steps of the code?

i. The AI appears to treat Suite2p baseline correction as the dominant compute step, with all other session processing relatively lightweight. The code measures per-session elapsed time around `process_session`, and the notes report roughly 0.5-0.7 s per session for full processing. Optional plotting is an additional cost when enabled.

ii.
```python
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
```

```python
t_sess = time.time()
...
neural_trials, input_trials, output_trials, n_neurons = process_session(...)
...
elapsed = time.time() - t_sess
print(f"... time: {elapsed:.1f}s")
```

iii. `CONVERSION_NOTES.md` identifies full processing runtime per session and repeatedly emphasizes the Suite2p baseline-correction step. Earlier trajectory steps also inspected Suite2p’s baseline code while debugging dF/F.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The final script already vectorizes binning with `reshape(...).mean(...)` and uses `np.interp` for motion-energy length matching, so the main remaining Python loops are the outer subject/session loops and the inner per-trial append loop. Those trial slices could be packed into one higher-rank array before converting to the required list-of-trials structure, but the AI did not attempt that.

ii.
```python
for subj in subjects_to_process:
    ...
    for i, session in enumerate(sessions):
        ...
        for t in range(n_trials):
            start = t * TRIAL_BINS
            end = (t + 1) * TRIAL_BINS
            neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
            input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
            output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. There is no explicit optimization discussion for these loops in the notes. The design seems driven by the required nested list format rather than by maximum vectorization.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats the full per-session pipeline independently for every session: loading arrays, computing baseline-corrected neural traces, resampling motion energy, binning, discretizing, and slicing into trials. It also repeats some ancillary work, such as importing plotting libraries in multiple places and computing percentile thresholds separately for each session.

ii.
```python
for subj in subjects_to_process:
    ...
    for i, session in enumerate(sessions):
        neural_trials, input_trials, output_trials, n_neurons = process_session(...)
```

```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The notes explicitly frame several steps as “per-session independently,” especially dF/F processing and motion-energy discretization, so the repetition is intentional rather than accidental.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary work is optional plotting for visual verification. When `--show-processing` is enabled, the script keeps raw intermediate arrays long enough to render multi-panel figures for the first two sessions, then discards them. The script also carries a few unused imports (`sys`, `warnings`) and writes metadata fields that the downstream decoder does not need.

ii.
```python
import sys
import warnings
```

```python
if show_processing and session_count < 2:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))
    fig_ax = (fig, axes)
```

```python
if show_processing and ax_list is not None:
    plot_processing(ax_list, subj, session, F, Fneu, dfof, me, me_binned,
                   me_discrete, dfof_binned, n_frames)
```

iii. The notes explicitly describe those plots as “Processing plots … for visual verification,” so this extra work was intentional validation overhead rather than part of the dataset representation used downstream.
