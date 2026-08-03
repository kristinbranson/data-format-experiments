# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over 6 subject directories (`jm031`-`jm046`) in the `data/` folder. For each subject, it discovers session subdirectories (sorted alphabetically/chronologically). For each session, it loads `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. All data is loaded via `np.load()`.

ii.
```python
def get_sessions(subject_dir):
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions

def process_session(session_dir, device=None):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    me_dir = os.path.join(session_dir, 'move_deve')
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. The AI followed the data organization described in the data README (`load_data.ipynb` and `data/README.md`), which shows loading F.npy from `suite2p/plane0/` for each session. The AI noted that data is organized as `data/<subject>/<session>/suite2p/plane0/` and `data/<subject>/<session>/move_deve/`.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses a hardcoded list of 6 subject IDs (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`), iterating over them in order. Each subject maps to a subdirectory under `data/`. A `subj_idx` counter tracks which subject each session belongs to.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    ...
    subject_idx_list.append(subj_idx)
```

iii. From CONVERSION_NOTES.md: "For cross-referencing with the paper the subjects are named in alphabetically increasing order (e.g. jm031 - mouse A, jm032 - mouse B ... jm046 - mouse F)." The AI identified all 6 mice from the data directory structure.

## 1-c. How are the data split into sessions?

i. Sessions are the subdirectories within each subject folder (e.g., `2023-10-18_a`). Each session corresponds to one recording day. The AI discovers them by listing the subject directory and sorting alphabetically. Each session becomes one entry in the `neural`, `input`, and `output` lists.

ii.
```python
sessions = get_sessions(subject_dir)  # sorted subdirs of subject_dir
for sess_dir in sessions:
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(sess_dir, device=device)
    session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
```

iii. The AI noted in CONVERSION_NOTES.md that there are 41 total sessions (7+7+7+7+6+7), matching the data structure where each subject has 6-7 daily recording sessions.

## 1-d. How are the data split into trials?

i. Each session's continuous recording is split into consecutive 2-minute blocks. With 30 Hz imaging and 10-frame bins, each trial has 360 time bins. Sessions with 36,000 frames (20 min) yield 10 trials; sessions with 54,000 frames (30 min) yield 15 trials. Any remainder frames at the end are discarded.

ii.
```python
def split_into_trials(dff_binned, me_binned):
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
    return neural_trials, me_trials
```

iii. The AI cited the paper: "splits were done on consecutive 2 minute blocks of the recording" and "5 fold splits...consecutive 2 minute blocks". This trial splitting matches the paper's cross-validation structure.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 2-minute blocks are included. Partial blocks at the end of sessions (remainder frames that don't fill a complete 2-minute trial) are discarded.

ii. The splitting code uses integer division (`n_bins // bins_per_trial`) to determine how many complete trials fit, implicitly discarding partial blocks. No explicit quality checks on trials are performed.

iii. From CONVERSION_NOTES.md: "Trial curation: No explicit trial curation mentioned - continuous recording." The paper describes continuous spontaneous behavior recordings without trial-based exclusion criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence traces, shape `n_neurons x n_frames`) and `Fneu.npy` (neuropil fluorescence traces, same shape) loaded from the Suite2p output directory for each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The AI noted that the reference notebook `load_data.ipynb` loads `F.npy` for fluorescence traces, and the paper describes using "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)", which requires both F and Fneu.

## 2-b. How is the `neural` data processed?

i. The processing pipeline is:
1. Neuropil subtraction: `F_corr = F - 0.7 * Fneu`
2. Baseline correction using Suite2p's `preprocess()` function with maximin filter (win=60s, sig=10 frames, prctile=8)
3. dF/F computation: `(F_corr - baseline) / baseline`
4. Temporal binning: averaged in non-overlapping bins of 10 frames

ii.
```python
def compute_dff(F, Fneu, device=None):
    F_corr = F - NEUCOEFF * Fneu  # NEUCOEFF = 0.7
    F_corr_copy = F_corr.copy()
    F_subtracted = preprocess(
        F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
        prctile_baseline=PRCTILE_BASELINE, device=device
    )
    baseline = F_corr - F_subtracted
    baseline_safe = np.clip(baseline, 1e-6, None)
    dff = F_subtracted / baseline_safe
    return dff.astype(np.float32)

def bin_traces(data, bin_size):
    # For 2D: reshape (n_features, n_bins*bin_size) -> mean over bin_size
    n_features, n_frames = data.shape
    n_bins = n_frames // bin_size
    trimmed = data[:, :n_bins * bin_size]
    return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)
```

iii. The AI cited the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and "slightly denoised the dF/F... by averaging in bins of 10 consecutive timestamps". Suite2p parameters were read from `ops.npy`: fs=30, neucoeff=0.7, baseline='maximin', win_baseline=60, sig_baseline=10.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied. The AI determined that the provided data already contains only neurons tracked across all days by Track2p, and that all `iscell.npy` values are 1.0, meaning no further cell classification filtering is needed.

ii. No filtering code exists in the script. All neurons in F.npy are used directly.

iii. From CONVERSION_NOTES.md: "Data already contains only neurons tracked across all days (Track2p output)" and "iscell.npy has all values = 1.0 (all cells marked as cells since they're pre-filtered)". The data README confirms: "the data only includes traces for the cells present across all days."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the recording session. Each trial begins at a fixed offset from the session start (trial_index * 2 minutes). There is no event-based alignment since the experiment involves continuous spontaneous behavior without discrete stimulus events.

ii.
```python
# In split_into_trials:
for t in range(n_trials):
    start = t * bins_per_trial  # 0, 360, 720, ...
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])

# In metadata:
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. The AI noted that this is continuous recording of spontaneous behavior, with no discrete stimuli or trial events to align to. The temporal alignment event is the start of the recording session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The original data is at 30 Hz (33.33 ms per frame). Temporal rebinning is applied by averaging every 10 consecutive frames, resulting in a bin size of 333.33 ms (10/30 seconds). This matches the paper's description.

ii.
```python
FS = 30.0           # imaging frame rate (Hz)
BIN_SIZE = 10        # number of frames per bin
# Time bin: BIN_SIZE / FS = 10/30 = 0.3333 seconds = 333.33 ms

dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)
```

iii. The AI cited the paper: "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The bin size of 333.33 ms is stored in `metadata['time_bin_size']`.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not derived from any raw data variable. It is synthetically generated as `np.arange(n_timebins) * time_bin_s`, producing an evenly spaced time series from 0 to ~119.7 seconds for each trial.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin = 0.3333
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI decided that since the time input is a synthetic variable representing elapsed time, it can be computed from the bin index and the known sampling parameters (30 Hz, 10-frame bins).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (10 / 30)` seconds for each time bin within a trial. Each trial's time resets to 0 at the trial start, so the range is [0, 119.667] seconds per trial. This represents time from the start of each *trial*, not time from the start of the *experiment*.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # 10/30 = 0.3333 s
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI interpreted "time elapsed from the beginning of the experiment" as time within each trial rather than cumulative time across the entire session.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is generated to match the number of time bins in the neural data for each trial (360 bins per trial). Since both are constructed from the same binning scheme, they are inherently aligned.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)  # matches neural time dimension
    input_trials.append(time_input)
```

iii. The AI ensured alignment by generating the time vector to have exactly the same number of time bins as the neural data for each trial.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` located in the `move_deve/` subdirectory of each session. This file contains pixel-wise motion energy extracted from videography of spontaneous behavior.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The data README describes this as "processed behavioural data (motion energy extracted from videography of spontaneous behaviour)." The paper describes motion energy as "pixel-wise difference of consecutive video frames, squared and summed across pixels."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The processing pipeline is:
1. Load raw motion energy from `motion_energy_glob.npy`
2. Interpolate missing frames (when ME has fewer frames than neural data) using linear interpolation
3. Average in non-overlapping bins of 10 frames (same as neural data)
4. Compute global percentile bin edges across all sessions
5. Discretize into 5 equal-percentile bins using `np.digitize()`

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
me_binned = bin_traces(me_interp, BIN_SIZE)
# ... later ...
all_me_concat = np.concatenate(all_me_binned_values)
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
me_disc = discretize_me(me_t, bin_edges)
```

iii. The AI followed the paper's description of binning behavior traces identically to neural data, and the instruction's requirement to normalize and discretize into five equal-percentile bins.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Global percentile bin edges are computed across all binned motion energy values from all sessions. The motion energy is then discretized into 5 bins (0-4) using `np.digitize()`. Equal-percentile bins ensure ~20% of data in each bin globally.

ii.
```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)  # [0, 20, 40, 60, 80, 100]
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges

def discretize_me(me_values, bin_edges):
    n_bins = len(bin_edges) - 1
    labels = np.digitize(me_values, bin_edges[1:-1])  # 0 to n_bins-1
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
```

iii. The instructions specify "normalized and discretized into five equal-percentile bins." The AI computed global percentile edges at [0, 20, 40, 60, 80, 100] percentiles, resulting in exactly 20% of all data points in each bin.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion energy is first interpolated to match the number of neural frames (handling missing camera frames), then binned identically to neural data (10-frame non-overlapping bins), then split into trials at the same boundaries. This ensures frame-by-frame correspondence.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)  # match neural frame count
me_binned = bin_traces(me_interp, BIN_SIZE)  # same binning as neural
neural_trials, me_trials = split_into_trials(dff_binned, me_binned)  # same trial split
```

iii. The AI noted that "ME is already synced to neural (camera triggered by microscope)" based on the paper stating the videography is triggered by the microscope at 30 Hz, providing 1:1 frame correspondence when all frames are present.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data quality issue is missing motion energy frames (where ME has fewer frames than neural data). The AI handles this by interpolating ME values to match the neural frame count. The interpolation uses `np.linspace` to evenly space the available ME indices across the neural frame range and applies linear interpolation. The `tstamps.npy` file is loaded but not actually used to determine which specific frames are missing.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    if n_me < n_neural_frames:
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
    else:
        return me[:n_neural_frames].astype(np.float64)
```

iii. The data README says "The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over." The AI chose interpolation. However, it uses evenly-spaced indices rather than the actual `tstamps.npy` timestamps to identify missing frames.

## 6-a. What are the most time-consuming steps of the code?

i. The dF/F computation (Suite2p's `preprocess()` baseline correction) is the most computationally expensive step per session, taking ~0.3-1.8 seconds depending on session size and neuron count. The full conversion of 41 sessions takes ~41.5 seconds total.

ii. From `conversion_full_out.txt`, session processing times range from 0.2s (jm031, 221 neurons, 36000 frames) to 1.8s (jm039, 746 neurons, 54000 frames).

iii. The AI documented timing in CONVERSION_NOTES.md: "Processing time: ~1s per session" and estimated full conversion at ~50s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over each trial to extract slices, but this is a simple slicing operation that is already efficient. The discretization loop per trial could theoretically be vectorized by discretizing the entire session at once before splitting into trials. The main per-subject/per-session loops are inherently sequential (different files to load).

ii.
```python
# Trial splitting loop (could be replaced with np.split or array_split)
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])

# Per-trial discretization loop (could be done once per session)
for neural_t, me_t in zip(neural_trials, me_trials):
    me_disc = discretize_me(me_t, bin_edges)
```

iii. The AI mentioned writing efficient code with "vectorized binning (reshape + mean)" but the trial-level loops remain.

## 6-c. What processing does the code repeat multiple times?

i. The code performs a two-pass approach: first pass processes all sessions to collect binned ME values for global percentile computation, then second pass re-iterates over session data to split into trials and discretize. However, the session data from the first pass is cached in `session_data`, so raw files are only loaded once. The `plot_processing` function re-loads raw F, Fneu, and ME data from disk for visualization, duplicating I/O already done in `process_session`.

ii.
```python
# First pass
for subj_idx, subject in enumerate(subjects_to_process):
    for sess_dir in sessions:
        dff_binned, me_binned, n_neurons, n_frames_raw = process_session(sess_dir, device=device)
        session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))

# Second pass
for subj_idx, sess_dir, dff_binned, me_binned, n_neurons in session_data:
    neural_trials, me_trials = split_into_trials(dff_binned, me_binned)

# In plot_processing (re-loads raw data):
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The two-pass design is necessary because global ME percentile bins can only be computed after processing all sessions. The re-loading in `plot_processing` is a minor inefficiency only triggered with `--show-processing`.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `tstamps.npy` file is loaded for every session but is not actually used in the interpolation logic. The code passes it to `interpolate_motion_energy` but the function uses `np.linspace` for evenly-spaced indices rather than the actual timestamps. Additionally, any frames beyond the last complete 10-frame bin are processed through dF/F but then discarded during binning (trimmed to `n_bins * bin_size`), and similarly any time bins beyond the last complete 2-minute trial are binned but then discarded during trial splitting.

ii.
```python
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))  # loaded but not used
me_interp = interpolate_motion_energy(me, n_frames, tstamps)  # tstamps ignored inside

# In bin_traces: trimmed = data[:, :n_bins * bin_size]  # discards remainder
# In split_into_trials: n_trials = n_bins // bins_per_trial  # discards remainder bins
```

iii. The AI noted loading tstamps but the interpolation logic doesn't use them to identify specific missing frames - it just evenly spaces the available ME frames.
