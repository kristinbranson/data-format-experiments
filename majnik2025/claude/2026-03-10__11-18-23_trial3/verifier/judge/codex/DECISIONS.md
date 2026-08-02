# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subject folders by scanning `data/` for directories whose names start with `jm`, then scans each subject directory for session subdirectories. For each session it loads `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy`. It does not load `interframe_int.npy` in the conversion script, even though its notes discuss missing-frame handling.

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

F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by saying the dataset is organized as `jm*` subject folders with session subfolders, and that each session contains Suite2p outputs plus motion-energy files. It also argued that the data were already preprocessed into matched Suite2p format.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted `jm*` directories. In the saved output, however, the AI remaps them to aliases `Mouse_A` through `Mouse_F` rather than keeping the raw subject ids.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}

subject_names = list(SUBJECT_MAP.values()) if not sample_mode else [SUBJECT_MAP[subjects_to_process[0]]]
```

iii. The notes state that each `jm*` directory corresponds to one mouse and list the six subjects explicitly. The alias mapping appears to come from the AI’s own documentation choice rather than from the reference code.

## 1-c. How are the data split into sessions?

i. Each session is one subdirectory inside a subject directory, sorted lexicographically. The AI treats each such directory as one session in the final dataset.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted([d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d))])
    all_sessions[subj] = sessions
```

iii. The notes justify this by describing each date-labeled subdirectory as one daily recording session.

## 1-d. How are the data split into trials?

i. The AI treats the experiment as continuous rather than trial-based. It first bins the continuous recording into 10-frame bins, then splits each session into consecutive non-overlapping 2-minute blocks (`360` bins per block). Any leftover bins at the end are implicitly dropped because `n_trials = n_total_bins // TRIAL_BINS`.

ii.
```python
BIN_SIZE = 10
TRIAL_DURATION_SEC = 120
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says there are no native trials and that it chose 2-minute blocks because the paper’s decoder cross-validation used “consecutive 2 minute blocks of the recording.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-quality filtering. The only effective trial loss is structural: trailing data that do not fill a complete 2-minute block after binning are omitted.

ii.
```python
n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    ...
```

iii. The notes say there is “No explicit trial curation (continuous spontaneous recording)” and frame mismatches should be handled by interpolation instead.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence arrays `F.npy` and `Fneu.npy` from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The notes and docstring both justify this as the Suite2p fluorescence representation used by the paper.

## 2-b. How is the `neural` data processed?

i. The AI subtracts neuropil (`F - 0.7 * Fneu`), applies Suite2p baseline preprocessing (`maximin`), then averages the result in non-overlapping 10-frame bins before trialization.

ii.
```python
Fc = F - NEUROPIL_COEFF * Fneu
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
```

iii. The AI’s stated justification was that the paper used “baseline corrected fluorescence traces” and “bins of 10 consecutive timestamps,” so it treated 10-frame averaging as part of the neural conversion itself.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra neuron-level filtering is applied in the conversion script. The AI assumes the provided data are already filtered to tracked cells and that all `iscell` values are effectively valid.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
n_neurons, n_frames = F.shape
```

iii. `CONVERSION_NOTES.md` says “All iscell values are 1 (pre-filtered through Track2p matching)” and therefore “No additional filtering needed.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to the start of the continuous recording session. Within each session, trials are consecutive 2-minute chunks after 10-frame binning.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))

'metadata': {
    'temporal_alignment_event': 'start of continuous recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The notes justify this by saying there is no stimulus event, only continuous spontaneous recording, so session start is the alignment point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI converts the data to 10-frame bins at 30 Hz, giving `333.33 ms` bins. Yes, it rebins both neural activity and motion energy by averaging over each 10-frame chunk.

ii.
```python
FRAME_RATE = 30.0
BIN_SIZE = 10
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000

def bin_timeseries(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The AI repeatedly justified this choice from the paper’s decoder analysis language about averaging in bins of 10 timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from a raw file. The AI computes it from the bin indices within each session, using the known frame rate and 10-frame bin width.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as “time_elapsed = bin_index * (10/30) seconds” because there are no explicit timestamps needed for the decoder input.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI creates a regularly spaced time vector in seconds for each 2-minute trial, using session-relative bin indices after temporal rebinning. The time value at each point corresponds to the start of a 10-frame bin.

ii.
```python
start = t * TRIAL_BINS
end = (t + 1) * TRIAL_BINS
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The rationale given in the notes is that time from the beginning of the experiment is the required decoder input and can be computed directly from sample index and frame rate.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is generated on the same per-trial index grid as the binned neural data. Each trial uses identical `start:end` boundaries for both arrays.

ii.
```python
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The AI’s notes say time is meant to be monotonically increasing and aligned to the same bins as the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. In code, the AI derives motion energy only from `move_deve/motion_energy_glob.npy`. It does not use `interframe_int.npy`, although its notes mention missing camera frames.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
me = interpolate_motion_energy(me_raw, n_frames)
```

iii. The notes justify using `motion_energy_glob.npy` as the precomputed motion-energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the entire motion-energy trace to the neural frame count if lengths differ, averages it in 10-frame bins, and then discretizes the binned signal. Its notes claim an additional per-session normalization step, but that normalization is not present in the script.

ii.
```python
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)
    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp

me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)
```

iii. The notes justify interpolation from the data README’s suggestion that missing frames may be interpolated, and justify 10-frame averaging from the paper’s decoder preprocessing. They also state “Normalize motion energy per session,” but the script never executes such a normalization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes 20th/40th/60th/80th percentile thresholds separately within each session’s binned motion-energy trace, then uses `np.digitize` to assign labels `0..4`.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The notes explicitly defend “Discretization per session” so that quintiles adapt to session-specific motion-energy scales.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI first forces the motion-energy trace to have the same total length as the neural recording via full-trace linear interpolation, then bins it with the same 10-frame averaging and slices it with the same `start:end` trial boundaries as the neural data.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The notes justify this by saying the microscope trigger synchronizes neural and video streams and missing camera frames can be interpolated away.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing motion-energy frames by linearly stretching the entire motion-energy trace to the neural length. It does not identify individual dropped frames. It also drops incomplete trailing data implicitly when the total bin count is not divisible by the chosen 2-minute trial length.

ii.
```python
if len(me) == n_frames:
    return me.astype(np.float64)

x_orig = np.linspace(0, 1, len(me))
x_new = np.linspace(0, 1, n_frames)
me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
return me_interp

n_trials = n_total_bins // TRIAL_BINS
```

iii. The justification in the notes comes from the data README phrase that missing camera frames may be “treated as missing values or interpolated over.” The script chooses whole-trace interpolation as the concrete implementation.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive substantive processing step is Suite2p baseline preprocessing in `compute_dfof`, which runs on the full neuron-by-frame matrix for every session. Plot generation in `--show-processing` mode is also extra overhead, but optional.

ii.
```python
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
```

iii. The notes’ runtime table and sanity checks focus on per-session processing time and treat the neural preprocessing step as the core signal-processing cost.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest remaining pure-Python loop is the per-trial assembly loop, which could be replaced by reshaping into `(n_trials, ...)` blocks before converting to the required list structure. Session iteration is unavoidable at file-loading granularity. The motion-energy interpolation itself is already vectorized via `np.interp`.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
    output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The AI did not explicitly call this out in its notes. Its stated efficiency story instead emphasized using vectorized 10-frame binning and simple interpolation.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats percentile computation separately for every session because discretization happens inside `process_session`. It also repeatedly constructs trial time vectors and repeatedly imports/sets up plotting helpers when `--show-processing` is used.

ii.
```python
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)

for t in range(n_trials):
    ...
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)

if show_processing:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
```

iii. The notes explicitly endorse per-session discretization, so this repetition is partly intentional rather than accidental.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It bins and processes the full continuous recording even though any tail shorter than a full 2-minute block is then discarded. In `--show-processing` mode it also computes and saves diagnostic plots that are not used by downstream decoding. More importantly, because the final format stores lists of trials, the continuous-session intermediate arrays are only temporary staging products.

ii.
```python
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
n_trials = n_total_bins // TRIAL_BINS

if show_processing and fig_ax is not None:
    fig.savefig(f'processing_{subj}_{session}.png', dpi=100)
```

iii. The AI’s notes justify the plots as visual verification and the continuous preprocessing as necessary to create trialized decoder input, but those intermediates are not themselves consumed downstream.
