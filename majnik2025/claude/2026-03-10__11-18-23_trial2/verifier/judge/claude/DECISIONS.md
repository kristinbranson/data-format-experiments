# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects by hardcoding a list of 6 subject names (`SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each subject, sessions are discovered as sorted subdirectories. For each session, `F.npy` and `Fneu.npy` are loaded from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions

# In process_session:
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. The AI hardcodes subject names rather than dynamically discovering them. The loading of suite2p files and motion energy is consistent with the data directory structure. The AI uses `tstamps.npy` for ME interpolation rather than `interframe_int.npy`.

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded as `SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`. Each subject directory is iterated over to find sessions.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

subjects_to_process = SUBJECTS
if sample:
    subjects_to_process = SUBJECTS[:1]

for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
```

iii. The AI identified 6 mice from the data directory and hardcoded their names. This produces the same result as the reference's dynamic discovery but is less flexible.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, sorted alphabetically, excluding hidden directories. Each session contains one recording day.

ii.
```python
def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions
```

iii. Each subdirectory represents a daily recording session. Sorting ensures deterministic ordering by date.

## 1-d. How are the data split into trials?

i. The AI splits each session into 2-minute (120-second) non-overlapping blocks after binning by 10 frames. Each trial has 360 time bins (120s * 30Hz / 10 frames). Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 120.0  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
BIN_SIZE = 10

def split_into_trials(dff_binned, me_binned):
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
```

iii. The AI chose 2-minute trials based on the paper's statement about "consecutive 2 minute blocks" for cross-validation splits. The reference uses 60-second trials instead.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials are included.

ii. N/A (no filtering code exists)

iii. The paper does not describe trial-level quality filtering for this continuous recording paradigm.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil subtraction (`F - 0.7 * Fneu`), (2) baseline correction via suite2p's `preprocess` function with `maximin` method, (3) dF/F computation by dividing the baseline-subtracted signal by the baseline (`F_subtracted / baseline`), and (4) binning by averaging 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, device=None):
    F_corr = F - NEUCOEFF * Fneu
    F_corr_copy = F_corr.copy()
    F_subtracted = preprocess(
        F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
        prctile_baseline=PRCTILE_BASELINE, device=device
    )
    baseline = F_corr - F_subtracted
    baseline_safe = np.clip(baseline, 1e-6, None)
    dff = F_subtracted / baseline_safe
    return dff.astype(np.float32)

# Then binning:
dff_binned = bin_traces(dff, BIN_SIZE)
```

iii. The AI interprets "baseline corrected fluorescence traces as our dF/F" from the paper as requiring division by baseline (standard dF/F formula). The reference code uses the output of `dcnv.preprocess` directly (baseline-subtracted, not divided by baseline).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the suite2p output are included, since the data is pre-filtered to tracked cells (iscell all = 1).

ii. N/A (no filtering code)

iii. The AI noted that iscell.npy has all values = 1.0, meaning all cells are pre-filtered by Track2p. No additional filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of continuous recording, no event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to in this spontaneous behavior paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI bins data by averaging 10 consecutive frames, resulting in a time bin size of 333.33 ms (10 frames / 30 Hz). Both neural and motion energy data are binned this way.

ii.
```python
BIN_SIZE = 10  # number of frames per bin (paper: "bins of 10 consecutive timestamps")

def bin_traces(data, bin_size):
    if data.ndim == 1:
        n_bins = n_frames // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_features, n_frames = data.shape
        n_bins = n_frames // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)

time_bin_ms = BIN_SIZE / FS * 1000  # 333.33 ms
```

iii. The AI based this on the paper's statement: "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The reference code does not bin and keeps native 30 Hz resolution.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from a raw data variable. It is computed from the bin index within each trial: `bin_index * (BIN_SIZE / FS)` seconds.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI computes time from frame/bin indices since the frame rate is constant at 30 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time as `bin_index * (10/30)` seconds, representing time from the **start of each trial** (resets to 0 at each trial boundary). This ranges from 0 to ~119.67 seconds per trial.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin = 0.333s
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI interprets "time elapsed from the beginning of the experiment" as time within each trial block, resetting to 0 at each trial boundary.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed directly from the bin indices that define the neural data, so alignment is inherent. Each time bin corresponds to exactly one neural data time bin.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. Since time is derived from the same indexing as the neural data, there is no alignment issue.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps from `tstamps.npy` are used for frame interpolation.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. The motion energy file contains pre-computed global motion energy from behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) interpolation to match neural frame count using linear interpolation via `np.interp`, (2) binning by averaging 10 consecutive frames, (3) discretization into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
# Interpolation:
me_indices = np.linspace(0, n_neural_frames - 1, n_me)
me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))

# Binning:
me_binned = bin_traces(me_interp, BIN_SIZE)

# Discretization:
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
labels = np.digitize(me_values, bin_edges[1:-1])
```

iii. The AI uses linear stretching interpolation rather than detecting specific dropped frames. It does not normalize motion energy by standard deviation before discretization. Global percentile bins ensure approximately equal class counts.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins using global percentile edges computed across all sessions' binned ME values.

ii.
```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges

def discretize_me(me_values, bin_edges):
    n_bins = len(bin_edges) - 1
    labels = np.digitize(me_values, bin_edges[1:-1])
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
```

iii. Equal-percentile binning ensures balanced class counts. The `np.clip` ensures labels stay in [0, 4].

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses linear interpolation (`np.interp` with `np.linspace`) to stretch the ME signal to match the neural frame count. After interpolation, both are binned by 10 frames and split into trials using the same indices.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    if n_me < n_neural_frames:
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp

# Both use same binning and trial splitting
dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)
```

iii. The AI uses a linear stretching approach for interpolation rather than detecting specific dropped frames via interframe intervals. The reference uses `interframe_int.npy` to identify dropped frames and inserts interpolated values at those specific locations.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing ME frames are handled via linear interpolation to match neural frame count. Partial trials at the end of sessions (bins that don't fill a complete 2-minute trial) are discarded. If ME has more frames than neural, it is truncated.

ii.
```python
if n_me < n_neural_frames:
    me_indices = np.linspace(0, n_neural_frames - 1, n_me)
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
else:
    return me[:n_neural_frames].astype(np.float64)
```

iii. The interpolation approach ensures the ME signal always matches the neural data length, though it uses a different method (linear stretching) than the reference (specific dropped-frame insertion).

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` baseline correction with GPU acceleration, followed by the dF/F computation which requires an additional copy and division operation.

ii.
```python
F_corr_copy = F_corr.copy()
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
```

iii. The baseline correction involves sliding window operations over the full session for every neuron. The AI additionally computes the baseline explicitly and divides by it, adding extra computation compared to the reference.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop and the ME discretization loop iterate over trials sequentially, but these are simple slicing operations that are already efficient. The binning is vectorized using reshape+mean.

ii.
```python
# Binning is vectorized:
return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)
```

iii. The code is generally well-vectorized. The trial splitting loop performs simple array slicing which is inherently sequential.

## 6-c. What processing does the code repeat multiple times?

i. In `plot_processing`, the code re-loads `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy` from disk for visualization, duplicating the initial loading.

ii.
```python
def plot_processing(session_dir, ...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. This redundant loading only occurs when `--show-processing` is enabled and is limited to 2 sessions, so the performance impact is minimal.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation (dividing baseline-subtracted signal by baseline) is unnecessary extra processing since the downstream decoder uses the neural data as-is. Additionally, the 10-frame binning reduces temporal resolution, discarding information that the reference retains at native 30 Hz.

ii.
```python
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
```

iii. The reference code uses the baseline-subtracted output of `dcnv.preprocess` directly without computing dF/F. The binning step further reduces information content.
