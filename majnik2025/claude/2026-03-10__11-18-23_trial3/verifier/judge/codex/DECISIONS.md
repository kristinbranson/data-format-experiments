# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories whose names start with `jm`, then scans each subject directory for all session subdirectories. For each session it loads `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, and `move_deve/motion_energy_glob.npy`. It does not load `interframe_int.npy` in the conversion script, even though its notes discuss missing video frames.

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

iii. In `CONVERSION_NOTES.md`, the AI justified this by documenting the dataset layout as six `jm*` mouse folders, each with session subdirectories containing `suite2p/plane0` outputs and `move_deve` motion-energy files. It also noted that motion-energy missing frames exist and cited the README’s suggestion to interpolate them.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories beginning with `jm`, sorted alphabetically. Internally it processes subjects under those raw IDs, but in the final output it renames them to `Mouse_A` through `Mouse_F` using a hard-coded mapping.

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
...
subject_names = list(SUBJECT_MAP.values()) if not sample_mode else [SUBJECT_MAP[subjects_to_process[0]]]
```

iii. The notes say the raw data contains six subjects `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, and also annotate them as “Mouse A” through “Mouse F.” That appears to be the justification for the renaming.

## 1-c. How are the data split into sessions?

i. Each session is defined as one subdirectory within a subject folder. The subdirectory names are sorted and each one is treated as a separate recording session.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    all_sessions[subj] = sessions
```

iii. The notes describe each `jm*` folder as containing date-labeled session directories, and the code comment calls them “sessions.” No additional session grouping rule was used.

## 1-d. How are the data split into trials?

i. The AI assumes there are no native trials and creates artificial trials by dividing each continuous session into non-overlapping 2-minute blocks after 10-frame temporal binning. With 30 Hz imaging and 10-frame bins, each trial is 360 bins long.

ii.
```python
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)
```

```python
n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The justification is explicit in `CONVERSION_NOTES.md`: it says trial size should be “2-minute blocks” because the paper’s decoder cross-validation used “consecutive 2 minute blocks of the recording,” and the final trajectory summary repeats “2-minute trial segmentation (360 bins per trial).”

## 1-e. How are trials filtered based on quality controls?

i. The AI applies no explicit trial-quality filtering. It keeps every complete 2-minute block that fits within a session and silently drops any leftover tail data that does not fill a full block because trial count is computed with floor division.

ii.
```python
n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS
...
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

```python
n_frames = data.shape[-1]
n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]
```

iii. The notes say there is “No explicit trial curation (continuous spontaneous recording)” and frame tails are effectively discarded by the binning/trial logic. No other trial QC rule was justified.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0`.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The notes identify these as the matched Suite2p outputs already provided per session, and say dF/F must be computed post hoc from them with Suite2p preprocessing.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil contamination with coefficient 0.7, then applies Suite2p’s `preprocess` with `maximin` baseline correction parameters. It treats that baseline-corrected output as the final neural signal, then averages it into non-overlapping 10-frame bins.

ii.
```python
Fc = F - NEUROPIL_COEFF * Fneu
...
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
```

```python
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
```

iii. The notes justify this as matching the paper’s “baseline corrected fluorescence traces as our dF/F” and the Suite2p defaults recovered from `ops.npy`. The trajectory summary also says an earlier version wrongly divided by baseline and was fixed to use the baseline-subtracted output directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied in the conversion script. All rows of `F.npy` and `Fneu.npy` are used, based on the assumption that the supplied Track2p-formatted data has already been filtered to matched cells and effectively has `iscell = 1` throughout.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_frames = F.shape
...
brain_region_idx_all.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The notes explicitly say “All iscell values are 1 (pre-filtered through Track2p matching)” and “All neurons included: Data is already filtered to tracked neurons only.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the start of the continuous recording session. There is no event-centered realignment; trials are contiguous blocks cut from the session timeline.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'start of continuous recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The notes justify this by treating the experiment as “continuous spontaneous recording” with “no discrete trials,” so the recording start becomes the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses non-overlapping bins of 10 frames at 30 Hz, so the time bin is about 333.33 ms. Both neural data and motion energy are rebinned this way.

ii.
```python
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000
```

```python
def bin_timeseries(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The notes cite the paper’s statement that decoding analyses used “bins of 10 consecutive timestamps,” and repeatedly describe the target resolution as 333 ms.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is not loaded from a raw file. It is synthesized from the binned time index within each session, using the bin number and the fixed bin duration.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes describe this mapping as “Time index -> time_elapsed = bin_index * (10/30) seconds,” justified by the fixed 30 Hz frame rate and 10-frame bins.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the AI creates a regularly spaced vector in seconds by multiplying the binned sample indices by `BIN_SIZE / FRAME_RATE`. Because `start` and `end` are absolute bin offsets within the full session, time continues across trial boundaries rather than resetting to zero at each trial.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as “Decoder input: time from start,” and the critical-review section says a spot-check of trial 3 gave `[360.00, 479.67]s`, which the AI used as evidence that the construction was correct.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. Input time is aligned by using the exact same per-trial start and end bin indices that are used to slice neural activity. Each time sample therefore corresponds one-to-one with the neural bin at the same position in the same trial.

ii.
```python
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes justify alignment by describing the input as derived from the same binned session timeline, and they report “time inputs are continuous across trial boundaries” as a temporal-alignment check.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In the script, motion energy is derived directly from `move_deve/motion_energy_glob.npy`. The AI’s notes also discuss `interframe_int.npy` as a way to identify missing frames, but that file is not actually used by the conversion code.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
me = interpolate_motion_energy(me_raw, n_frames)
```

iii. The notes justify the source variable by describing `motion_energy_glob.npy` as precomputed whole-video motion energy and acknowledging that camera-frame drop information exists in `interframe_int.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The implemented pipeline is: load `motion_energy_glob.npy`, linearly resample the whole trace to the neural frame count if lengths differ, average into 10-frame bins, and discretize per session into five percentile bins. Separately, the notes and top-of-file docstring state that motion energy is “normalized per session,” but the script contains no normalization step.

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
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)
```

```python
- Motion energy normalized per session and discretized into 5 equal-percentile bins
```

iii. The notes justify interpolation by citing the data README’s suggestion that missing camera frames “be treated as missing values or interpolated over.” They justify per-session discretization as a way to handle different motion-energy scales across sessions. No explicit justification is given for the claimed normalization, and no normalization code is present.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI converts binned motion energy into five session-specific equal-percentile categories. It computes the 20th, 40th, 60th, and 80th percentiles and uses `np.digitize` to assign labels 0 through 4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
thresholds = np.percentile(me_binned, percentiles)
labels = np.digitize(me_binned, thresholds).astype(np.int64)
```

iii. The notes justify this as “quintile bins computed per session to handle different motion energy scales,” and the trajectory summary highlights “perfect 20% per quintile bin” as a sanity check.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by forcing the full motion-energy trace to the same length as the neural frame count with global linear interpolation, then binning it with the same 10-frame averaging operation and slicing trials with the same `start:end` indices as the neural data.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
...
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes say synchronization is frame-by-frame at 30 Hz via the microscope trigger, and missing camera frames should be interpolated. The script operationalizes that as whole-trace interpolation rather than explicit dropped-frame insertion.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles motion-energy length mismatches by linearly interpolating the whole motion-energy trace to the neural frame count. It also drops incomplete tails produced by non-multiple-of-10 frame counts and by session lengths that do not divide evenly into the chosen trial size. There are no explicit assertions or detailed checks around these cases in the conversion script.

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
n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]
...
n_trials = n_total_bins // TRIAL_BINS
```

iii. The notes explicitly identify missing motion-energy frames as a dataset issue and cite the README’s recommendation to interpolate over them. No stronger justification is given for the choice to globally resample the entire trace rather than insert only the missing frames.

## 6-a. What are the most time-consuming steps of the code?

i. The dominant cost is the per-session Suite2p baseline-correction call on the neural fluorescence traces. Session-by-session plotting under `--show-processing` is an additional optional cost.

ii.
```python
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
```

iii. The notes’ runtime estimates are framed around “full processing” per session, and the neural preprocessing step is the only substantial numerical operation over every neuron and frame. No other explicit bottleneck analysis is recorded.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loop that could have been vectorized is the per-trial assembly loop in `process_session`, which repeatedly slices and appends trial arrays. The script does use vectorized whole-trace interpolation (`np.interp`) for motion energy, so it avoids the repeated `np.insert` pattern seen in the human reference.

ii.
```python
neural_trials = []
input_trials = []
output_trials = []

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. There is no explicit justification in the notes for leaving this loop as-is. The only indirect efficiency justification is that the notes report short runtime estimates for the full dataset.

## 6-c. What processing does the code repeat multiple times?

i. There is no major repeated scientific processing of the same data within the script. Each session is processed once. The closest repeated work is per-trial construction of small time arrays and optional generation of plotting figures for up to two sessions.

ii.
```python
for subj in subjects_to_process:
    ...
    for i, session in enumerate(sessions):
        ...
        neural_trials, input_trials, output_trials, n_neurons = process_session(...)
```

```python
for t in range(n_trials):
    ...
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
```

iii. No explicit justification was recorded because the AI did not describe any repeated-processing concern in its notes. The code structure suggests it considered the runtime acceptable.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion path does not contain much obviously discarded scientific processing. The main optional extra work is `--show-processing`, which creates diagnostic plots not consumed by downstream analyses. The script also stores additional metadata fields beyond the required schema.

ii.
```python
if show_processing and ax_list is not None:
    plot_processing(ax_list, subj, session, F, Fneu, dfof, me, me_binned,
                   me_discrete, dfof_binned, n_frames)
```

```python
'metadata': {
    'task_description': ...,
    'time_bin_size': TIME_BIN_MS,
    'temporal_alignment_event': 'start of continuous recording session',
    'off_start': 0.0,
    'off_end': None,
    'frame_rate': FRAME_RATE,
    'bin_size_frames': BIN_SIZE,
    'trial_duration_s': TRIAL_DURATION_SEC,
    ...
}
```

iii. The notes justify the plotting path as visual verification of every processing step. No other unnecessary computation is explicitly acknowledged.
