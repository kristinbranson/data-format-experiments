# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 6 subject names (`SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`), then iterates through each subject's directory to find session subdirectories. For each session, it loads `F.npy`, `Fneu.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. Data is loaded using `np.load`.

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

iii. From CONVERSION_NOTES.md: The AI identified the data structure through exploration of the `data/` directory, finding 6 mouse subjects with session subdirectories each containing suite2p output and motion energy files. The subject list is hardcoded based on this exploration.

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded as an ordered list of 6 mouse IDs. Each subject corresponds to a directory in the data folder.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
```

iii. The AI identified 6 subjects from directory exploration and hardcoded them. The reference solution dynamically discovers subjects by scanning for directories starting with `jm`.

## 1-c. How are the data split into sessions?

i. Sessions are identified as subdirectories within each subject's folder, sorted alphabetically. Hidden directories (starting with `.`) are excluded.

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

iii. From CONVERSION_NOTES.md: Each session directory contains one day's recording. The AI found 41 total sessions across 6 subjects (7,7,7,7,6,7).

## 1-d. How are the data split into trials?

i. The AI splits each session into 2-minute (120-second) non-overlapping blocks. After binning by 10 frames, each trial is 360 time bins. Any remainder bins that don't fill a complete trial are discarded.

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

iii. From CONVERSION_NOTES.md Step 5: "Each session split into 2-minute blocks (matching paper's CV structure)" and "Paper uses 'consecutive 2 minute blocks' for CV". The AI interpreted the paper's cross-validation fold structure as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete trials (those filling a full 2-minute block) are included.

ii. N/A - no filtering code exists.

iii. From CONVERSION_NOTES.md: "No explicit trial curation mentioned - continuous recording."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. From CONVERSION_NOTES.md Step 1: "Loads F.npy (raw fluorescence) from suite2p dir" - standard suite2p output files.

## 2-b. How is the `neural` data processed?

i. Three processing steps: (1) neuropil subtraction (`F - 0.7 * Fneu`), (2) baseline correction using suite2p's `preprocess` function with maximin method, then (3) actual dF/F computation: `(F_corr - baseline) / baseline`. Additionally, the neural data is temporally binned by averaging in 10-frame bins.

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

# Then binned:
dff_binned = bin_traces(dff, BIN_SIZE)
```

iii. From CONVERSION_NOTES.md Step 3: "dF/F computation: 'baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)'" and Step 5: "dF/F method: Use Suite2p's preprocess (baseline_maximin) with default params from ops.npy". The AI interpreted the paper's phrase "baseline corrected fluorescence traces as our dF/F" as meaning actual dF/F = (F - F0) / F0.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the pre-filtered suite2p output are included.

ii. N/A - no filtering code exists.

iii. From CONVERSION_NOTES.md Step 4: "iscell: all iscell[:,0]=1 | 0.5 default | Provided data already pre-filtered to tracked cells; no additional filtering needed."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the recording session. Since recordings are continuous with no stimulus events, trials are contiguous segments starting from the beginning of the session.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. From CONVERSION_NOTES.md Step 5: "Temporal alignment: ME is already synced to neural (camera triggered by microscope)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging in bins of 10 consecutive frames. The resulting time bin size is 10/30 = 333.33 ms. Both neural and motion energy data are binned this way.

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

# Metadata:
time_bin_ms = BIN_SIZE / FS * 1000  # 333.33 ms
```

iii. From CONVERSION_NOTES.md Step 3: "Binning for decoding: 10 frames - 'averaging in bins of 10 consecutive timestamps'". The AI followed the paper's description of binning for its decoding analysis.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed synthetically from the bin index within each trial.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI computes time as `bin_index * (10/30)` seconds, giving within-trial time from 0 to ~119.7 seconds.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `np.arange(n_timebins) * (BIN_SIZE / FS)`, giving seconds elapsed within each trial. This resets to 0 at the start of each trial rather than tracking time from the start of the experiment.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI interpreted "time elapsed from the beginning of the experiment" as within-trial time. CONVERSION_NOTES.md Step 5 states: "Time in seconds from start of trial."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time bins are generated to have exactly the same number of time points as the binned neural data (360 bins per trial), so alignment is inherent.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. Since time is synthetically generated to match neural trial length, no alignment issues arise.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory. Timestamps from `tstamps.npy` are used for frame mismatch interpolation.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. From CONVERSION_NOTES.md Step 2: "motion_energy_glob.npy - Motion energy (n_frames,), uint64" and "tstamps.npy - Timestamps (n_frames,), float64".

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) Interpolation of motion energy to match neural frame count using `np.interp` with linearly spaced indices, (2) temporal binning by averaging in 10-frame bins, (3) discretization into 5 equal-percentile bins computed globally across all sessions. Notably, no normalization by standard deviation is applied before discretization.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    if n_me < n_neural_frames:
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp

# Binning:
me_binned = bin_traces(me_interp, BIN_SIZE)

# Discretization:
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
labels = np.digitize(me_values, bin_edges[1:-1])
```

iii. From CONVERSION_NOTES.md Step 5: "Interpolate missing frames, bin by 10, normalize, discretize into 5 equal-percentile bins | Global percentile bins across all data." Note: the notes mention normalization, but the code does not actually normalize by std.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 bins using equal-percentile edges computed globally across all sessions. `np.digitize` with internal bin edges maps values to integer labels 0-4, clipped to valid range.

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

iii. From CONVERSION_NOTES.md Step 5: "Motion energy normalization: Compute percentile bins globally across all sessions."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When ME has fewer frames than neural data, the AI uses `np.interp` with linearly spaced indices to interpolate ME to match the neural frame count. When ME has more frames, it truncates. Both signals are then binned by 10 and split into trials identically.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    if n_me < n_neural_frames:
        neural_indices = np.arange(n_neural_frames)
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
    else:
        return me[:n_neural_frames].astype(np.float64)
```

iii. From CONVERSION_NOTES.md Step 4: "Camera sync: Video at 30Hz triggered by microscope acquisition -> 1:1 frame correspondence" and Step 5: "Missing ME frames: Interpolate to match neural frame count using timestamps."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing ME frames (where ME has fewer frames than neural data) are handled by linear interpolation using `np.interp`. Partial trials at the end of sessions (remainder bins that don't fill a full trial) are discarded. Partial bins at the end before binning are also discarded.

ii.
```python
# Frame mismatch:
me_indices = np.linspace(0, n_neural_frames - 1, n_me)
me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))

# Partial bins:
trimmed = data[:n_bins * bin_size]

# Partial trials:
n_trials = n_bins // bins_per_trial
```

iii. From CONVERSION_NOTES.md Step 10: "Missing ME frames: handled via interpolation (affects 7 sessions)" and "Partial bins at end of sessions: discarded."

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess()` call for baseline correction, which runs on GPU. The AI reports ~1.25 seconds per session. Loading `.npy` files is also I/O-bound.

ii.
```python
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
```

iii. From CONVERSION_NOTES.md Step 7: "Sample: 2.5s for 2 sessions (~1.25s/session)" and "Estimated full: ~50s for 41 sessions."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials one at a time to slice arrays. This is simple indexing and not a performance concern. The binning is already vectorized using reshape+mean.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. No specific vectorization opportunities were identified or discussed in CONVERSION_NOTES.md. The binning operation is already efficiently vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The AI processes data in two passes: first pass loads/preprocesses all sessions, second pass splits into trials and discretizes. This is structurally necessary for global percentile computation. No truly redundant processing is performed, though the `compute_dff` function computes `preprocess` and then recovers the baseline by subtraction (`baseline = F_corr - F_subtracted`), effectively processing the data through preprocess once but using the result to derive two quantities.

ii.
```python
# compute_dff calls preprocess once, then derives baseline:
F_subtracted = preprocess(F_corr_copy, ...)
baseline = F_corr - F_subtracted
dff = F_subtracted / baseline_safe
```

iii. No discussion of repeated processing in CONVERSION_NOTES.md.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes actual dF/F by dividing by baseline, which is extra processing compared to just using the baseline-subtracted output of `preprocess`. The AI also generates descriptive output_value labels (e.g., "ME<123") which are cosmetic. The `tstamps` loading is partially unnecessary since the interpolation doesn't actually use tstamps values for index computation (it uses `np.linspace` instead).

ii.
```python
# dF/F division (extra step beyond what reference does):
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe

# tstamps loaded but not meaningfully used:
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
# ... but interpolation uses np.linspace, not tstamps
```

iii. No discussion of unnecessary processing in CONVERSION_NOTES.md.
