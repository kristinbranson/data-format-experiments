# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over a hardcoded list of 6 subject IDs (`SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each subject, it lists session subdirectories, then for each session loads `F.npy`, `Fneu.npy` (neural), `motion_energy_glob.npy`, and `tstamps.npy` (behavioral) from the suite2p and move_deve subdirectories. Data is processed per-session and accumulated into lists. A two-pass approach is used: first pass loads and processes all sessions (computing dF/F, binning), second pass splits into trials and discretizes motion energy using globally-computed percentile bin edges.

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

# In convert_data():
for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    sessions = get_sessions(subject_dir)
    for sess_dir in sessions:
        dff_binned, me_binned, n_neurons, n_frames_raw = process_session(sess_dir, device=device)

# In process_session():
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. The AI documented the data structure in CONVERSION_NOTES.md Step 2, noting the hierarchical organization (subject/session/suite2p and move_deve dirs). The loading approach directly mirrors the `load_data.ipynb` notebook provided with the dataset, which loads F.npy from the suite2p directory. The AI also loads Fneu.npy for neuropil correction as described in the paper.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined as a hardcoded list of 6 mouse IDs in alphabetical order. Each subject maps to a directory in the data folder. The subject index is tracked during iteration and stored in `subject_idx_list`, which becomes `subject_idx` in the output data structure.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    ...
    subject_idx_list.append(subj_idx)

data = {
    'subjects': SUBJECTS if not sample else SUBJECTS[:1],
    'subject_idx': np.array(subject_idx_list, dtype=np.int64),
    ...
}
```

iii. The AI identified all 6 subjects from the data directory, cross-referenced with the paper ("full dataset of 6 mice"), and noted the naming convention (jm031=Mouse A through jm046=Mouse F) from the data README.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, sorted alphabetically (which gives chronological order since folders are named YYYY-MM-DD_a). Each session is processed independently. The result is one entry per session in the neural/input/output lists. There are 41 sessions total (7 each for 5 mice, 6 for jm040).

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

iii. The AI documented 41 total sessions (7,7,7,7,6,7 per subject), consistent with the data README and paper ("imaged daily for a minimum of 6 consecutive days").

## 1-d. How are the data split into trials?

i. Each session's continuous recording is split into consecutive 2-minute blocks. At 30 Hz imaging with 10-frame binning, each trial is 360 time bins (2 min * 60 s/min * 30 Hz / 10 frames/bin). Sessions with 36000 frames (20 min) yield 10 trials; sessions with 54000 frames (30 min) yield 15 trials. Any remainder frames that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_S = 120.0  # 2 minutes per trial

def split_into_trials(dff_binned, me_binned):
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial
    neural_trials = []
    me_trials = []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
    return neural_trials, me_trials
```

iii. The AI based the 2-minute trial structure on the paper's description: "5 fold splits...consecutive 2 minute blocks of the recording." This is the CV structure described in the paper, and the AI adopted it as the trial definition for the decoder format.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 2-minute blocks from all sessions are included. Partial blocks at the end of sessions (less than 360 bins) are discarded as a consequence of integer division in the trial splitting logic.

ii.
```python
n_trials = n_bins // bins_per_trial  # integer division discards partial blocks
```

iii. The AI noted in CONVERSION_NOTES.md Step 3: "No explicit trial curation mentioned - continuous recording." Since the paper does not describe any trial exclusion criteria and the data is from continuous spontaneous behavior recordings (no task structure with success/failure), no filtering was deemed necessary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from two raw Suite2p output arrays: `F.npy` (raw fluorescence traces, shape n_neurons x n_frames) and `Fneu.npy` (neuropil fluorescence traces, same shape). Both are loaded from `suite2p/plane0/` for each session.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. The AI followed the paper's description and the load_data.ipynb notebook. The paper describes computing dF/F from "baseline corrected fluorescence traces" using "default Suite2p parameters," which requires both F and Fneu for neuropil subtraction.

## 2-b. How is the `neural` data processed?

i. Processing involves three steps: (1) Neuropil subtraction: F_corr = F - 0.7 * Fneu. (2) Baseline correction using Suite2p's `preprocess` function with maximin filter (win_baseline=60s, sig_baseline=10 frames, prctile_baseline=8). (3) dF/F = (F_corr - baseline) / baseline. After dF/F, traces are binned by averaging in non-overlapping windows of 10 consecutive frames.

ii.
```python
def compute_dff(F, Fneu, device=None):
    F_corr = F - NEUCOEFF * Fneu  # 0.7 neuropil coefficient
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

iii. The AI cited the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." Parameters were extracted from Suite2p's ops.npy (neucoeff=0.7, baseline='maximin', win_baseline=60, sig_baseline=10).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. The AI determined that the provided data already contains only neurons tracked across all days (Track2p output in Suite2p format), and that iscell.npy has all values set to 1.0 for all neurons. Therefore no additional filtering was needed.

ii.
No filtering code exists in convert_data.py. All neurons from F.npy are used directly.

iii. The AI noted in CONVERSION_NOTES.md: "Data already contains only neurons tracked across all days (Track2p output)" and verified that "iscell.npy has all values = 1.0 (all cells marked as cells since they're pre-filtered)." The data README confirms: "the data only includes traces for the cells present across all days."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to the start of the recording session. Since the data is a continuous recording with no discrete events, each trial begins at its chronological position within the session. Trial 0 starts at frame 0, trial 1 at frame 3600 (360 bins * 10 frames/bin), etc. The metadata records `temporal_alignment_event: 'Start of recording session'` and `off_start: 0.0`.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))

data['metadata'] = {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The AI noted that the recordings are continuous spontaneous behavior sessions without discrete stimulus events. The paper's "consecutive 2 minute blocks" are used as trial boundaries. The alignment event is the start of the recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 333.33 ms (10 frames at 30 Hz). Rebinning is applied: the original 30 Hz data (33.33 ms per frame) is averaged in non-overlapping bins of 10 consecutive frames, matching the paper's description. Both neural and behavioral data are binned identically.

ii.
```python
FS = 30.0           # imaging frame rate (Hz)
BIN_SIZE = 10        # number of frames per bin
time_bin_ms = BIN_SIZE / FS * 1000  # = 333.33 ms
```

iii. The paper states "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input time variable is not derived from any raw data variable. It is synthetically computed from the bin index within each trial, using the known bin duration (10 frames / 30 Hz = 0.333 s per bin).

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI interpreted "time elapsed from the beginning of the experiment" as time within each trial. Since the temporal structure is regular (fixed bin size), the time input is deterministically computed from bin indices.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `bin_index * (10 / 30)` seconds, yielding values from 0.0 to 119.67 seconds for each 360-bin trial. This creates an evenly-spaced time vector within each trial. The computation resets to 0 at the start of each trial rather than accumulating across the session.

ii.
```python
def make_time_input(n_timebins):
    time_bin_s = BIN_SIZE / FS  # 0.3333... seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
```

iii. The AI's CONVERSION_NOTES.md Step 5 mapping states: "Time in seconds from start of trial | (bin_index * 10 / 30) seconds within trial."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input has the same number of time bins (360) as the neural data for each trial, since it is synthetically generated with shape (1, n_timebins) to match the neural array shape (n_neurons, n_timebins). Both share the same temporal grid. However, the time resets to 0 for each trial rather than reflecting cumulative time from the start of the recording session.

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)  # always starts at 0
    input_trials.append(time_input)
```

iii. The AI ensured dimensional consistency by deriving the time bins count from the neural trial shape. The alignment is trivially correct since both share the same binning scheme.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` in each session's `move_deve/` directory. This contains the global motion energy computed from video recordings (pixel-wise differences of consecutive frames, squared and summed across pixels).

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. The AI documented in CONVERSION_NOTES.md: "Motion energy: Pixel-wise difference of consecutive video frames, squared and summed across pixels."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing involves: (1) Loading raw motion energy. (2) Interpolating to match neural frame count when there are missing camera frames (linear interpolation using evenly-spaced indices). (3) Binning by averaging in 10-frame windows (same as neural). (4) Computing global percentile bin edges across ALL sessions. (5) Discretizing into 5 equal-percentile bins using `np.digitize`.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)
me_binned = bin_traces(me_interp, BIN_SIZE)

# Global percentile computation:
all_me_concat = np.concatenate(all_me_binned_values)
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)

def discretize_me(me_values, bin_edges):
    n_bins = len(bin_edges) - 1
    labels = np.digitize(me_values, bin_edges[1:-1])  # 0 to n_bins-1
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
```

iii. The AI followed the paper's description of binning ("averaging in bins of 10 consecutive timestamps") and the instruction to discretize into "five equal-percentile bins." The AI chose to compute percentile bins globally across all sessions rather than per-session.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 categories (bins 0-4) using equal-percentile edges computed globally across all binned ME values from all sessions. The percentiles used are [0, 20, 40, 60, 80, 100]. `np.digitize` with the inner bin edges assigns values to bins 0 through 4, with clipping to ensure values stay within range.

ii.
```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)  # [0, 20, 40, 60, 80, 100]
    bin_edges = np.percentile(all_me_values, percentiles)
    return bin_edges

def discretize_me(me_values, bin_edges):
    n_bins = len(bin_edges) - 1
    labels = np.digitize(me_values, bin_edges[1:-1])
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
```

iii. The instructions specify "normalized and discretized into five equal-percentile bins." The AI's global approach yields exactly 20% of all data in each bin globally. However, per-session distributions can be very skewed (e.g., jm046 sessions have output ranges of [2,4] or [3,4] with 0% in lower bins).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data through frame-level correspondence (camera triggered by microscope at 30 Hz), with interpolation for sessions with missing camera frames. After interpolation, both signals have the same number of frames, are binned identically (10-frame averages), and split into trials at the same boundaries. The output ME is discretized after binning but before trial splitting.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)  # match to neural frames
me_binned = bin_traces(me_interp, BIN_SIZE)  # same binning as neural
# Trial splitting uses same indices for both neural and ME
neural_trials, me_trials = split_into_trials(dff_binned, me_binned)
```

iii. The AI noted: "ME is already synced to neural (camera triggered by microscope)" from the paper/data README, and handles the frame mismatch edge case through interpolation.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The main data quality issue is missing camera frames (motion energy has fewer frames than neural data in 9 out of 41 sessions, with mismatches ranging from 1 to 148 frames). The AI handles this by linearly interpolating ME to match the neural frame count using evenly-spaced indices (np.linspace). The tstamps are loaded but not actually used for determining which specific frames are missing. For partial trials at session ends, the remainder is silently discarded. Baseline values near zero are clipped to 1e-6 to avoid division by zero in dF/F.

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

# In compute_dff:
baseline_safe = np.clip(baseline, 1e-6, None)
```

iii. The AI documented the ME frame mismatch issue in CONVERSION_NOTES.md and noted 7 affected sessions (actual count is 9). The data README explicitly states these frames "can be interpolated over," which the AI followed.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the dF/F computation via Suite2p's `preprocess` function, which performs GPU-accelerated maximin baseline correction for each session. Per the conversion output, each session takes 0.2-1.8 seconds, with the full conversion completing in ~41.5 seconds for 41 sessions. Larger neuron counts (e.g., jm039 with 746 neurons) take longer.

ii.
```python
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
```

iii. The AI noted in CONVERSION_NOTES.md Step 6: "Processing time: ~1s per session" and estimated full conversion at ~50s, which was close to the actual 41.5s.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials using a for-loop with array slicing, but this could be done with a single reshape operation (reshape the binned arrays into (n_trials, bins_per_trial, ...) and then create list views). The ME discretization loop iterates per-trial but could be done on the full session-level binned ME array before splitting into trials. The main session loop is inherently sequential due to per-session file I/O.

ii.
```python
# Current loop-based trial splitting:
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
    me_trials.append(me_binned[start:end])

# Could be vectorized as:
# neural_reshaped = dff_binned[:, :n_trials*bins_per_trial].reshape(n_neurons, n_trials, bins_per_trial)
```

iii. The AI did not explicitly discuss vectorization opportunities in CONVERSION_NOTES.md, but noted "Vectorized binning (reshape + mean)" was implemented for the binning step.

## 6-c. What processing does the code repeat multiple times?

i. The `plot_processing` function reloads F.npy, Fneu.npy, and motion_energy_glob.npy from disk for sessions that are plotted, even though this data was already loaded in `process_session`. This duplicates file I/O for up to 2 sessions when `--show-processing` is enabled. Additionally, the output value labels are generated in a loop that could be computed once.

ii.
```python
# In plot_processing() - reloads data already loaded in process_session():
def plot_processing(session_dir, dff_binned, me_binned, ...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))      # already loaded
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))  # already loaded
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))  # already loaded
```

iii. The AI did not explicitly discuss this redundancy.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `tstamps.npy` file is loaded for every session but never meaningfully used -- the interpolation function receives it as a parameter but uses `np.linspace` instead of the actual timestamps. The remainder frames at the end of sessions (after the last complete 2-minute trial) are processed through dF/F and binning but then discarded during trial splitting. For 30-minute sessions, there are no remainder frames (5400 bins / 360 = 15 exactly), but for 20-minute sessions there are also no remainders (3600 / 360 = 10 exactly), so this is not actually wasteful for this dataset.

ii.
```python
# tstamps loaded but not used:
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
# ... passed to interpolate_motion_energy but linspace is used instead:
me_indices = np.linspace(0, n_neural_frames - 1, n_me)  # tstamps ignored
```

iii. The AI documented loading tstamps as part of the data structure but did not discuss the fact that it is unused in the actual interpolation logic.
