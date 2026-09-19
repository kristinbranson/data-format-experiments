# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects by listing directories starting with `jm` in the data directory, then lists subdirectories within each subject folder as sessions. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` from `move_deve/` for behavioral data. All subjects and sessions are processed in a single pass.

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

iii. The AI follows the standard suite2p directory structure and the data organization documented in the data README. All `jm*` directories are treated as subjects. The AI correctly identifies that each session directory contains suite2p outputs and motion energy files.

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted directories starting with `jm` in the data directory. The AI additionally maps folder names to paper names (e.g., `jm031` -> `Mouse_A`) via a hardcoded `SUBJECT_MAP` dictionary and uses the paper names in the final data structure.

ii.
```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}

subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])

subject_names = list(SUBJECT_MAP.values())  # ['Mouse_A', 'Mouse_B', ...]
```

iii. The AI noted the correspondence between folder names and paper names and chose to use the paper names for clarity in the output. The reference uses the raw folder names (`jm031`, etc.).

## 1-c. How are the data split into sessions?

i. Each subdirectory within a subject's folder is treated as a separate session, sorted alphabetically (which corresponds to chronological order since directories are date-named). All sessions for all subjects are included.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. This matches the data organization where each dated subdirectory contains one day's recording.

## 1-d. How are the data split into trials?

i. The AI splits each continuous recording into non-overlapping **2-minute (120-second)** trials of 360 bins each (120s * 30Hz / 10 frames/bin). Remainder bins that don't fill a complete trial are discarded. The instructions explicitly state "Split sessions into 60-second trials", but the AI chose 120-second trials based on the paper's mention of "consecutive 2 minute blocks" for cross-validation.

ii.
```python
TRIAL_DURATION_SEC = 120   # 2 minutes (paper: "consecutive 2 minute blocks")
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial

# In process_session:
n_trials = n_total_bins // TRIAL_BINS
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The AI justified the 2-minute trial duration by citing the paper's cross-validation structure which uses "consecutive 2 minute blocks". However, the task instructions explicitly say "Split sessions into 60-second trials." The reference uses 60-second trials (180 bins per trial).

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete trials (that fill a full trial duration) are included. Partial trials at the end of sessions are discarded.

ii. No filtering code; only the trial-splitting loop with integer division implicitly discards remainders.

iii. The continuous spontaneous recording has no natural trial structure, so there are no quality-based reasons to exclude trials. This matches the reference approach.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/` for each session.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. These are the standard suite2p output files. This matches the reference approach.

## 2-b. How is the `neural` data processed?

i. The AI applies neuropil subtraction (`Fc = F - 0.7 * Fneu`), then uses suite2p's `preprocess` function with the `maximin` baseline method. The result is then binned by averaging 10 consecutive frames.

ii.
```python
def compute_dfof(F, Fneu, fs=FRAME_RATE):
    Fc = F - NEUROPIL_COEFF * Fneu  # NEUROPIL_COEFF = 0.7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dfof = s2p_preprocess(
        Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
        SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
    )
    return dfof.astype(np.float32)

# Then binned:
dfof_binned = bin_timeseries(dfof, BIN_SIZE)  # BIN_SIZE = 10
```

iii. The AI correctly follows the paper's method: "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The preprocessing parameters (win_baseline=60s, sig_baseline=10, prctile_baseline=8, baseline='maximin') match the ops.npy defaults and the reference code. One minor difference: the AI passes `Fc.copy().astype(np.float32)` while the reference passes `Fc` directly (the reference does not explicitly copy or cast to float32 before calling preprocess).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in `F.npy` are included. The AI notes that iscell values are all 1.0 (pre-filtered by Track2p), so no filtering is needed.

ii. No filtering code is present.

iii. The AI correctly identifies that the data is already filtered through Track2p matching. This matches the reference approach, which also does not filter neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since there is no stimulus event, trials are contiguous segments starting from the beginning of the continuous recording.

ii.
```python
'temporal_alignment_event': 'start of continuous recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus or event to align to. The recording is continuous spontaneous activity. This matches the reference approach (which uses `temporal_alignment_event: 'session_start'` and `off_start: None`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting from 30 Hz to 3 Hz (333.33 ms time bins). This rebinning is applied before discretization of motion energy.

ii.
```python
BIN_SIZE = 10  # frames per bin
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # ~333.33 ms

def bin_timeseries(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." This matches the reference approach exactly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the bin indices and bin duration, not from any raw data variable. It represents seconds elapsed from the start of the recording session.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. Since the frame rate is constant at 30 Hz and bins are 10 frames each, computing time from bin indices is equivalent to using timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Each bin index is multiplied by the bin duration (10/30 = 1/3 second) to get time in seconds from session start. The time is computed per trial but runs continuously across trials within a session (i.e., trial 2 starts at 120s for 2-minute trials, not at 0s).

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
# For trial 0: start=0, end=360 -> time from 0.0 to 119.67s
# For trial 1: start=360, end=720 -> time from 120.0 to 239.67s
```

iii. The time input is straightforward arithmetic. The reference uses the same approach but with 60-second trial boundaries.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is derived from the same bin indices used for neural data, so they are inherently aligned. Each time bin corresponds exactly to the same temporal bin of neural activity.

ii.
```python
# Same start/end indices used for both:
neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. Since both are derived from the same binning structure, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. This is the pre-computed global motion energy signal from the behavioral video. The reference also loads interframe interval data (`interframe_int.npy`) for dropped frame detection, while the AI uses `np.interp` instead.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) interpolation of motion energy to match neural frame count (using `np.interp` linear interpolation), (2) temporal binning by averaging 10 consecutive frames, (3) discretization into 5 equal-percentile bins computed per session.

ii.
```python
# Step 1: Interpolation
def interpolate_motion_energy(me, n_frames):
    if len(me) == n_frames:
        return me.astype(np.float64)
    x_orig = np.linspace(0, 1, len(me))
    x_new = np.linspace(0, 1, n_frames)
    me_interp = np.interp(x_new, x_orig, me.astype(np.float64))
    return me_interp

# Step 2: Binning
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()

# Step 3: Discretization
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The AI correctly bins and discretizes motion energy per session into 5 quintile bins. The interpolation method differs from the reference (see 4-d).

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins per session. Percentile boundaries at [20, 40, 60, 80] are computed, and `np.digitize` assigns each value to bins 0-4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
thresholds = np.percentile(me_binned, percentiles)
labels = np.digitize(me_binned, thresholds).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins, selected per session." The AI follows this correctly. The reference uses a slightly different approach: `np.linspace(0, 100, n_levels + 1)` to get `[0, 20, 40, 60, 80, 100]`, then uses `np.digitize(me, bin_edges[1:-1])` which is functionally equivalent.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When the motion energy array is shorter than the neural data (due to dropped camera frames), the AI uses `np.interp` for global linear interpolation to stretch the motion energy signal to match the neural frame count. The reference instead uses `interframe_int.npy` to identify specific dropped frames and inserts interpolated values at those positions.

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

iii. The AI's approach is a simpler global interpolation that doesn't use the interframe interval data to identify where frames were dropped. The reference uses `interframe_int.npy` to detect specific dropped frames (`dt * 1000 > 0.04`) and inserts interpolated values at those positions. Both approaches produce similar results when only a few frames are missing, but the reference approach is more precise in placing interpolated values at the correct temporal positions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (motion energy shorter than neural data) are handled by linear interpolation to match lengths. Remainder frames at the end of sessions that don't fill a complete trial are discarded. No NaN/Inf checking is performed in the code.

ii.
```python
# Interpolation for missing frames:
me_interp = np.interp(x_new, x_orig, me.astype(np.float64))

# Discarding remainders (implicit via integer division):
n_trials = n_total_bins // TRIAL_BINS
```

iii. The interpolation ensures motion energy and neural data have matching lengths. Discarding remainder frames is a minor data loss (at most one trial duration minus one bin per session).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` baseline correction (`s2p_preprocess`), which runs the `maximin` baseline estimation for every neuron across the full session length. The AI reports ~0.5-0.7s per session, with the full conversion completing in ~25s.

ii. N/A (timing is reported via `time.time()` calls around session processing)

iii. The baseline correction involves sliding window operations over the full session for every neuron. GPU acceleration is used when available.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-splitting loop iterates over trials to slice arrays, which could be done with a single reshape operation. However, since the number of trials is small, this has negligible performance impact.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    ...
```

iii. The loop-based approach is clear and the overhead is minimal since the number of trials per session is small (10-15).

## 6-c. What processing does the code repeat multiple times?

i. No significant redundant processing is identified. Each session is processed once in a single pass.

ii. N/A

iii. The code processes each session sequentially without repeating operations.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No significant unnecessary processing is identified. The code computes only what is needed for the final output.

ii. N/A

iii. The processing pipeline is straightforward and minimal.
