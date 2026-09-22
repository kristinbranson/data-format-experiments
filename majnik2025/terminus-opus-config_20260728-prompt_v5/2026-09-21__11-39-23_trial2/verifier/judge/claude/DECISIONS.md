# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 6 subject names (`SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`). For each subject, it lists sorted session subdirectories. For each session, it loads `F.npy`, `Fneu.npy`, and `ops.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` from `move_deve/`. Unlike the reference, it does not load `interframe_int.npy`.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def load_session_data(subject_dir, session_name):
    session_dir = os.path.join(subject_dir, session_name)
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    return F, Fneu, ops, me
```

iii. The AI documented that files follow a standard convention with subject folders containing session subfolders. It noted in CONVERSION_NOTES.md that the data directory contains 6 subjects (jm031-jm046). The subject list is hardcoded rather than discovered dynamically.

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded as a list of 6 names. Each subject's sessions are iterated over, and a `subjects_seen` list is built in order. Subject index is tracked per session via `subjects_seen.index(subject)`.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

subjects_seen = []
for sess_idx, (subject, session_name) in enumerate(all_sessions):
    if subject not in subjects_seen:
        subjects_seen.append(subject)
    subj_idx = subjects_seen.index(subject)
```

iii. The AI noted in CONVERSION_NOTES.md that subjects correspond to the 6 `jm*` directories. The hardcoded list ensures a known order.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a sorted subdirectory within a subject's folder. All subdirectories are included.

ii.
```python
def get_sessions(subject_dir):
    sessions = sorted([d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d))])
    return sessions
```

iii. The AI documented 41 total sessions across 6 subjects (7,7,7,7,6,7 per subject), consistent with the reference.

## 1-d. How are the data split into trials?

i. Trials are defined as 60-second non-overlapping segments of the continuous recording. After binning (10 frames per bin at 30 Hz), each trial is 180 bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION = 60.0  # seconds per trial
BINS_PER_TRIAL = int(TRIAL_DURATION / TIME_PER_BIN)  # 180

def split_into_trials(data, bins_per_trial):
    n_total_bins = data.shape[-1]
    n_trials = n_total_bins // bins_per_trial
    trials = []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        trials.append(data[..., start:end])
    return trials
```

iii. The instructions specify "Split sessions into 60-second trials." The AI followed this directly. Since there's no natural trial structure, fixed-length segmentation is used.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are included.

ii. N/A (no filtering code)

iii. The AI noted in CONVERSION_NOTES.md that there are no explicit trial curation rules mentioned since this is spontaneous activity with no task trials. This matches the reference which also does not filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The AI also loads `ops.npy` to read preprocessing parameters.

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
```

iii. The AI documented these as standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - 0.7 * Fneu`), followed by suite2p's `preprocess` function which performs baseline estimation and correction using the `maximin` method. The AI reads parameters from `ops.npy` (with hardcoded defaults as fallback), then bins the result into 10-frame averages.

ii.
```python
def compute_dff(F, Fneu, ops):
    neucoeff = ops.get('neucoeff', NEUCOEFF)
    baseline = ops.get('baseline', 'maximin')
    win_baseline = ops.get('win_baseline', 60.0)
    sig_baseline = ops.get('sig_baseline', 10.0)
    fs = ops.get('fs', FS)
    Fc = F - neucoeff * Fneu
    Fc = Fc.astype(np.float32)
    device = torch.device('cpu')
    dFF = s2p_preprocess(Fc, baseline, win_baseline, sig_baseline, fs, device=device)
    return dFF
```

iii. The AI documented that neuropil subtraction with coefficient 0.7 is the suite2p default, and the maximin baseline method is suite2p's standard preprocessing. It noted the paper says "baseline corrected fluorescence traces as our dF/F".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in the `F.npy` output are included. The AI verified that all `iscell` values are 1.0 in the data.

ii. N/A (no filtering code)

iii. The AI documented in CONVERSION_NOTES.md that "All ROIs in provided data have iscell=1 (all pass the default 0.5 threshold)" and "Track2p matching: only neurons tracked across all days are included - in our data, this is already done."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the session start. Since trials are contiguous 60-second segments, no event-based alignment is needed. The metadata sets `temporal_alignment_event: 'session_start'`.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The AI noted there is no stimulus event to align to, as the recording is continuous spontaneous activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting from 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_SIZE = 10          # frames to average per bin
TIME_PER_BIN = BIN_SIZE / FS  # seconds per bin (0.3333s)

def bin_data(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    binned = truncated.reshape(new_shape).mean(axis=-1)
    return binned
```

iii. The AI cited the Methods: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the bin index and the bin duration. It is not derived from any raw data variable (like `tstamps.npy`). The value represents the center of each bin.

ii.
```python
time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN  # seconds
```

iii. The AI documented that since the frame rate is constant at 30 Hz, computing time from bin indices is equivalent. It chose to use the center of each bin.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The time input is computed as `(bin_index + 0.5) * time_per_bin`, giving the center time of each bin in seconds from session start. This is a simple linear computation with no additional processing.

ii.
```python
time_axis = (np.arange(n_bins) + 0.5) * TIME_PER_BIN
# For each trial:
trial_time = time_axis[start_bin:end_bin]
input_trials.append(trial_time.reshape(1, -1))
```

iii. The AI noted in CONVERSION_NOTES.md (Step 5): "Time input: Center of each bin (offset by 0.5 x time_per_bin). This represents the midpoint of each temporal bin."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same bin indices used for the neural data, so they are inherently aligned. Each bin's time corresponds to the same time period as the neural bin at the same index.

ii.
```python
# Both use the same bin indexing:
start_bin = t * BINS_PER_TRIAL
end_bin = start_bin + BINS_PER_TRIAL
trial_time = time_axis[start_bin:end_bin]        # input
neural_trial = dFF_binned[:, start_bin:end_bin]   # neural (via split_into_trials)
```

iii. The alignment is implicit since both streams share the same time axis.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Unlike the reference, the AI does NOT use `interframe_int.npy` for dropped frame detection.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
```

iii. The AI documented that the motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) ME is aligned to neural frame count by padding with last value or truncating (NOT by interpolating dropped frames), (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 equal-percentile bins computed within each session.

ii.
```python
def align_me_to_neural(me, n_neural_frames):
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    elif n_me < n_neural_frames:
        padded = np.zeros(n_neural_frames, dtype=np.float64)
        padded[:n_me] = me.astype(np.float64)
        padded[n_me:] = me[-1]  # repeat last value
        return padded
    else:
        return me[:n_neural_frames].astype(np.float64)

def discretize_me(me_binned, n_bins=N_ME_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
    me_discrete = np.clip(me_discrete, 0, n_bins - 1)
    return me_discrete, bin_edges
```

iii. The AI's CONVERSION_NOTES.md states: "ME length mismatches: handled by padding with last value (up to 116 frames = 0.3% of session)."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins using `np.percentile` and `np.digitize`. Bin edges are computed per session. Values are clipped to [0, 4].

ii.
```python
def discretize_me(me_binned, n_bins=N_ME_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(me_binned, percentiles)
    me_discrete = np.digitize(me_binned, bin_edges[1:-1], right=False)
    me_discrete = np.clip(me_discrete, 0, n_bins - 1)
    return me_discrete, bin_edges
```

iii. The instructions specify "Motion energy, discretized into five equal-percentile bins, selected per session."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns ME to neural data by padding (repeat last value) if ME is shorter, or truncating if ME is longer. This differs from the reference, which uses `interframe_int.npy` to detect dropped camera frames and interpolates at the correct positions.

ii.
```python
def align_me_to_neural(me, n_neural_frames):
    n_me = len(me)
    if n_me < n_neural_frames:
        padded = np.zeros(n_neural_frames, dtype=np.float64)
        padded[:n_me] = me.astype(np.float64)
        padded[n_me:] = me[-1]
        return padded
    else:
        return me[:n_neural_frames].astype(np.float64)
```

iii. The AI noted in CONVERSION_NOTES.md: "When ME is shorter than F, pad with last ME value. This handles minor camera trigger drops." This is a simpler approach that does not use the dropped frame information available in the data.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. ME length mismatches are handled by padding with the last value or truncating. Remainder bins at the end of a session that don't fill a complete 60-second trial are discarded. The AI does not use `interframe_int.npy` to detect where dropped frames occurred.

ii.
```python
# ME alignment
def align_me_to_neural(me, n_neural_frames):
    if n_me < n_neural_frames:
        padded = np.zeros(n_neural_frames, dtype=np.float64)
        padded[:n_me] = me.astype(np.float64)
        padded[n_me:] = me[-1]
        return padded

# Trial remainder handling
n_trials = n_total_bins // bins_per_trial  # discards remainder
```

iii. The AI documented in CONVERSION_NOTES.md that ME length mismatches affect up to 116 frames (0.3% of session). It chose padding over interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `preprocess` baseline correction. The AI reports per-session times in the output (load, dff, bin times). The AI noted full conversion takes about 31.7 seconds for 41 sessions.

ii.
```python
t1 = time.time()
dFF = compute_dff(F, Fneu, ops)
t_dff = time.time() - t1
```

iii. The AI documented timing: load 0.0-0.2s, dF/F 0.2-0.8s, bin+discretize <0.1s per session. Total ~31.7s for all 41 sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials one by one, but this is a simple slicing operation that is already efficient. The code is generally well-vectorized.

ii.
```python
for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    trials.append(data[..., start:end])
```

iii. This loop is simple slicing and does not involve computation that would benefit from vectorization. The actual computational heavy-lifting (binning, dF/F) is already vectorized.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session is processed once in a single pass.

ii. N/A

iii. The code follows a straightforward single-pass pipeline: load -> preprocess -> bin -> discretize -> split trials.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ops.npy` for each session to read preprocessing parameters, but the values are always the same as the hardcoded defaults. This adds minimal overhead. The remainder bins at the end of sessions are computed (via binning) but then discarded when splitting into trials.

ii.
```python
ops = np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True).item()
# Uses ops.get() with fallback defaults that always match the ops values
```

iii. Loading ops.npy is a minor overhead that adds robustness by using session-specific parameters if they differ.
