# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over a hardcoded list of 6 subject IDs (`SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each subject, it finds session subdirectories, then loads `F.npy` and `Fneu.npy` from the `suite2p/plane0/` subdirectory and `motion_energy_glob.npy` and `tstamps.npy` from the `move_deve/` subdirectory. Processing happens in two passes: first all sessions are preprocessed and binned, then motion energy is globally discretized and trials are split.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = get_sessions(subject_dir)
    for sess_dir in sessions:
        dff_binned, me_binned, n_neurons, n_frames_raw = process_session(sess_dir, device=device)

# In process_session:
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. The AI identified the standard suite2p output structure and motion energy files from the `move_deve` directory. The subject list is hardcoded rather than dynamically discovered.

## 1-b. How are the data split into subjects?

i. Subjects are defined by a hardcoded list of 6 subject IDs. Each subject corresponds to a directory in the data folder.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
subjects_to_process = SUBJECTS
```

iii. The AI noted 6 mice from the paper and data exploration and hardcoded the list. This produces the same result as the reference's dynamic discovery approach.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject's folder, sorted alphabetically. Each subdirectory contains one daily recording session.

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

iii. Each subdirectory contains suite2p output and motion energy files for one recording session.

## 1-d. How are the data split into trials?

i. The AI splits sessions into **2-minute (120-second) non-overlapping blocks**, yielding 360 time bins per trial. This was based on the paper's description of using "consecutive 2 minute blocks" for cross-validation splits. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 120.0  # 2 minutes per trial (paper: "consecutive 2 minute blocks")
...
bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
n_trials = n_bins // bins_per_trial
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. The AI justified 2-minute trials by citing the paper's cross-validation structure: "5 fold splits...consecutive 2 minute blocks." The CONVERSION_NOTES document this under "Trial Structure" in Step 5.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete trials are included.

ii. N/A (no filtering code)

iii. No explicit trial curation is mentioned in the paper for this continuous recording paradigm.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from the `suite2p/plane0/` directory.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. These are standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI applies three steps: (1) neuropil subtraction (`F_corr = F - 0.7 * Fneu`), (2) baseline correction using suite2p's `preprocess` with maximin method, and (3) **division by baseline** to compute true dF/F = `(F_corr - baseline) / baseline`. The baseline is recovered by subtracting the preprocess output from the neuropil-corrected signal.

ii.
```python
F_corr = F - NEUCOEFF * Fneu
F_corr_copy = F_corr.copy()
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
```

iii. The AI cited the paper: "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and interpreted this as requiring division by baseline to produce a proper dF/F ratio.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI notes the data is already pre-filtered to tracked neurons (all `iscell` values are 1.0).

ii. N/A (no filtering code)

iii. The AI documented that "Data already contains only neurons tracked across all days (Track2p output)" and that iscell filtering was unnecessary since all values are already 1.0.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of the continuous recording, no event-based alignment is applied.

ii.
```python
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to in this continuous recording paradigm.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting 30 Hz to 3 Hz (333.33 ms per bin). This matches the paper's description.

ii.
```python
BIN_SIZE = 10  # number of frames per bin (paper: "bins of 10 consecutive timestamps")
...
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
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index within each trial, giving seconds from the start of that trial.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI noted this as "Time elapsed from the beginning of the experiment" but implemented it as time from the start of each trial (resetting to 0 at each trial boundary).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (BIN_SIZE / FS)` for each trial, starting from 0 at the beginning of each trial. For 2-minute trials with 360 bins, time ranges from 0.0 to ~119.67 seconds.

ii.
```python
time_input = make_time_input(n_timebins)
# Where make_time_input returns: np.arange(n_timebins) * (BIN_SIZE / FS)
```

iii. The AI computed time using frame indices and the known frame rate, treating each trial independently.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is generated with the same number of time bins as the neural data for each trial, so alignment is implicit. However, the time resets to 0 at each trial boundary, representing time from trial start rather than session start.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. Since all data streams are binned identically and sliced at the same trial boundaries, temporal alignment is inherent.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Timestamps (`tstamps.npy`) are loaded for frame mismatch interpolation.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) missing frames are handled by linear interpolation using `np.linspace` to stretch the motion energy to match neural frame count, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 percentile-based bins computed **globally** across all sessions.

ii.
```python
# Interpolation:
me_indices = np.linspace(0, n_neural_frames - 1, n_me)
me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))

# Binning:
me_binned = bin_traces(me_interp, BIN_SIZE)

# Global percentile discretization:
all_me_concat = np.concatenate(all_me_binned_values)
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
...
labels = np.digitize(me_values, bin_edges[1:-1])
```

iii. The AI used global percentile bins (across all sessions) for discretization, noting that the task says "five equal-percentile bins." The CONVERSION_NOTES state "Compute percentile bins globally across all sessions."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories using `np.digitize` with percentile-based bin edges computed globally across all sessions. The edges are at the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of the concatenated motion energy from all sessions. Labels range from 0 to 4.

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

iii. The AI chose global percentile bins to ensure the overall distribution is 20% per bin.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. When the motion energy array is shorter than the neural data (due to dropped frames), the AI uses `np.interp` with linearly spaced indices to stretch/interpolate the motion energy to match the neural frame count. After interpolation, both are binned identically and sliced at the same trial boundaries.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    if n_me < n_neural_frames:
        neural_indices = np.arange(n_neural_frames)
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
```

iii. The AI noted that camera is triggered by microscope for 1:1 correspondence, and that missing frames need interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Two types of data issues are handled: (1) Missing motion energy frames (when ME array is shorter than neural data) are filled by linear interpolation using `np.linspace`, and (2) remainder bins at the end of sessions that don't fill a complete trial are discarded via integer division.

ii.
```python
# Missing ME frames:
if n_me < n_neural_frames:
    me_indices = np.linspace(0, n_neural_frames - 1, n_me)
    me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))

# Extra ME frames:
if n_me > n_neural_frames:
    return me[:n_neural_frames].astype(np.float64)
```

iii. The AI handles both fewer and more ME frames than neural frames, with interpolation and truncation respectively.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` baseline correction call, which runs on GPU when available. The full conversion takes about 41.5 seconds for all 41 sessions (~1s/session).

ii. N/A

iii. The baseline correction involves sliding window operations over the full session length for every neuron. Additionally, the AI adds an extra computation step to recover the baseline (`baseline = F_corr - F_subtracted`) and then divide, making the dF/F computation slightly more expensive than the reference.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials one at a time, appending to lists. However, this is a simple slicing operation and is already efficient. The per-trial discretization could theoretically be done on the full session array before splitting, but the AI does split first and then discretize per trial, which adds minimal overhead.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])
```

iii. These loops are simple and efficient enough that vectorization would provide negligible benefit.

## 6-c. What processing does the code repeat multiple times?

i. When `--show-processing` is enabled, the code reloads raw `F.npy`, `Fneu.npy`, and `motion_energy_glob.npy` from disk for plotting purposes, duplicating the initial load. In normal mode, no significant processing is repeated.

ii.
```python
def plot_processing(session_dir, ...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. This repeated loading only happens in visualization mode and doesn't affect the main conversion performance.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `baseline = F_corr - F_subtracted` to then compute `dff = F_subtracted / baseline_safe`. This involves creating extra arrays (baseline, baseline_safe) and performing an additional division operation. The reference code simply uses the output of `dcnv.preprocess` directly without this extra step.

ii.
```python
F_subtracted = preprocess(F_corr_copy, ...)
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
```

iii. The baseline recovery and division step transforms the data differently than the reference but is not "unnecessary" per se -- it's a different interpretation of how to compute dF/F. However, the extra computation does add overhead.
