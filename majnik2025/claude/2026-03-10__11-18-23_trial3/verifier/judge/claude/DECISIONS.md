# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by listing directories starting with `jm` in the data directory, then discovers sessions as subdirectories within each subject. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` from `move_deve/` for behavioral data. This matches the reference approach.

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

# Per session:
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI documented in CONVERSION_NOTES.md that directories matching `jm*` are subjects and subdirectories are sessions, following the standard convention of the dataset. The AI correctly identified all 6 subjects and 41 sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted directories starting with `jm` in the data directory. The AI additionally maps these to paper names (Mouse_A through Mouse_F) via a `SUBJECT_MAP` dictionary, using the paper names as subject identifiers in the output.

ii.
```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}

subject_names = list(SUBJECT_MAP.values())
```

iii. The AI noted that the naming convention is consistent across the dataset. The mapping to Mouse_A-F follows the paper's naming convention.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject folder. Each subdirectory contains one daily recording session. The AI processes all sessions for each subject (or a limited subset in sample mode).

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The AI documented in CONVERSION_NOTES.md that session directories are date-labeled (e.g., `2023-10-18_a`), and sorting ensures deterministic chronological order.

## 1-d. How are the data split into trials?

i. The AI splits continuous recordings into 2-minute non-overlapping segments (120s). After 10-frame temporal binning, each trial has 360 time bins (120s * 30Hz / 10 frames = 360 bins). The reference solution uses 60-second trials with no binning (1800 frames per trial).

ii.
```python
TRIAL_DURATION_SEC = 120   # 2 minutes
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial

n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The AI justified 2-minute trials by citing the paper's cross-validation structure: "consecutive 2 minute blocks of the recording." The paper uses 2-minute blocks for CV folds, but this was the trial segmentation choice the AI made.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Remainder bins/frames at the end of a session that don't fill a complete trial are discarded, but no quality-based filtering is performed.

ii. N/A (no filtering code)

iii. The AI noted in CONVERSION_NOTES.md that this is a continuous spontaneous recording with no explicit trial curation needed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. This matches the reference.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The AI identified these as standard suite2p output files, consistent with the paper's methodology.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (Fc = F - 0.7 * Fneu), then suite2p's baseline correction using the `maximin` method. This matches the reference. However, the AI additionally applies temporal binning by averaging 10 consecutive frames, which the reference does not do.

ii.
```python
Fc = F - NEUROPIL_COEFF * Fneu  # 0.7
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)

# Binning (not in reference):
dfof_binned = bin_timeseries(dfof, BIN_SIZE)  # BIN_SIZE = 10 frames

def bin_timeseries(data, bin_size):
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The AI justified the binning by citing the paper: "averaging in bins of 10 consecutive timestamps" for decoding. The neuropil and baseline correction parameters were confirmed via `ops.npy`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. All neurons in the `F.npy` file are included. The AI noted that the data is already filtered through Track2p matching (all `iscell` values are 1). This matches the reference.

ii. N/A (no filtering code)

iii. The AI documented that Track2p already performed cell matching and suite2p's cell detection pipeline identified ROIs, so no additional filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since the recording is continuous and trials are artificial segments, no event-based alignment is applied. This matches the reference approach.

ii.
```python
'temporal_alignment_event': 'start of continuous recording session',
'off_start': 0.0,
'off_end': None,
```

iii. The AI noted there is no stimulus event to align to, as the recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames, resulting in a time bin size of ~333.33ms (10/30Hz * 1000). The reference keeps the native 30Hz resolution (~33.33ms bins) with no rebinning.

ii.
```python
BIN_SIZE = 10  # frames per bin
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # ~333.33 ms

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The AI cited the paper: "averaging in bins of 10 consecutive timestamps" for the decoding analysis. The paper does mention this binning for their ridge regression decoder.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin indices and the known frame rate/bin size. This is similar to the reference approach (which uses frame indices divided by frame rate).

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The AI noted that the frame rate is constant at 30 Hz, so computing time from indices is equivalent to using stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as bin_index * (BIN_SIZE / FRAME_RATE) = bin_index * (10/30) seconds. This gives time elapsed from the start of the session in seconds. In the reference, time is computed as frame_index / FS = frame_index / 30 seconds. Both represent absolute time from session start, but at different temporal resolutions.

ii.
```python
# AI code:
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
# where start and end are bin indices across the full session

# Reference code:
t = ((s + np.arange(trial_frames)) / FS).astype(np.float32)
# where s is the frame offset from session start
```

iii. No explicit justification provided; this is a straightforward computation.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time input is computed from the same bin indices used to slice the neural data, ensuring perfect alignment. Each bin index maps to a specific time. The reference similarly uses the same frame indices.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. No explicit justification needed; the alignment is inherent in using the same indices.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory. The reference also uses `interframe_int.npy` for dropped frame detection, but the AI does not use this file.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified motion energy as a pre-computed global motion energy signal from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI's processing pipeline: (1) interpolate motion energy to match neural frame count using `np.interp`, (2) bin by 10 frames (averaging), (3) discretize into 5 equal-percentile bins per session. The reference instead: (1) detects dropped frames via interframe intervals, (2) inserts interpolated values at specific positions, (3) normalizes by standard deviation per session, (4) discretizes into 5 percentile bins globally across all sessions.

ii.
```python
# Interpolation:
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)
    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp

# Binning:
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

# Per-session discretization:
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The AI documented the interpolation approach and per-session discretization. The per-session approach was justified as handling different motion energy scales across sessions.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes motion energy into 5 bins using per-session percentile boundaries (quintiles). The thresholds at the 20th, 40th, 60th, and 80th percentiles are computed for each session independently using `np.digitize`. The reference computes percentile boundaries globally across all sessions after normalization.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. Per-session discretization yields exactly 20% per bin per session. The AI verified this produces balanced class counts.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses linear interpolation (`np.interp`) to resample the motion energy signal to match the neural frame count, then bins both signals by 10 frames. The reference uses dropped-frame detection via `interframe_int.npy` to insert interpolated values at specific missing frame positions.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
# Then both neural and ME are binned by same bin_size:
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The AI documented that video and neural data are acquired synchronously at 30Hz, with occasional dropped video frames causing length mismatches that need interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames by resampling the entire motion energy trace to match the neural frame count using linear interpolation. Sessions where ME already matches the neural frame count are left unchanged. Remainder bins at the end of a session that don't fill a complete trial are discarded.

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

iii. The AI noted in CONVERSION_NOTES.md that some sessions have fewer ME frames than neural frames, citing the data README about missing frames.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `s2p_preprocess` baseline correction (runs on GPU when available). The AI included timing per session and reported ~0.1-0.6s per session.

ii. N/A (timing reported in output logs)

iii. The AI documented in CONVERSION_NOTES.md that the full conversion takes approximately 25 seconds for all 41 sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials sequentially, slicing arrays. This could potentially be vectorized using reshape operations. However, the loop body is simple array slicing and the number of iterations is small (10-15 trials per session).

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

iii. No explicit discussion of vectorization opportunities in the AI's notes.

## 6-c. What processing does the code repeat multiple times?

i. The AI's code does not repeat any major processing steps. Each session is processed once in a single pass. The discretization is done per-session within the `process_session` function, so percentile computation happens once per session.

ii. N/A

iii. No explicit discussion in the AI's notes.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies 10-frame temporal binning that reduces the time resolution by 10x, producing data at ~3Hz rather than the native 30Hz. While the paper mentions binning for their ridge regression decoder, the instructions do not require this preprocessing step, and it discards temporal information. The downstream decoder (`train_decoder.py`) works on whatever temporal resolution is provided.

ii.
```python
BIN_SIZE = 10
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The AI justified binning by citing the paper's methods for their specific decoder, but this is a preprocessing choice embedded in the conversion rather than left to downstream analysis.
