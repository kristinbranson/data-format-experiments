# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers subjects as directories starting with `jm` in the data directory, then iterates over sorted subdirectories within each subject folder as sessions. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and `motion_energy_glob.npy` from `move_deve/` for behavioral data. All sessions are processed in a single loop, and data is accumulated into lists before assembling the final dictionary.

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

# In process_session:
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The AI identified the standard suite2p directory structure and the motion energy file location. The approach of scanning for `jm*` directories matches the data organization. The AI documented this in CONVERSION_NOTES.md Steps 0-2.

## 1-b. How are the data split into subjects?

i. Subjects are identified as sorted directories starting with `jm` in the data directory. However, the AI maps these to paper names ("Mouse_A" through "Mouse_F") using a hardcoded `SUBJECT_MAP` dictionary, and stores these mapped names in the `subjects` field rather than the raw folder names.

ii.
```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}

subject_names = list(SUBJECT_MAP.values()) if not sample_mode else [SUBJECT_MAP[subjects_to_process[0]]]
```

iii. The AI created this mapping based on the paper's naming conventions. CONVERSION_NOTES.md Step 2 lists the correspondence between jm identifiers and Mouse labels.

## 1-c. How are the data split into sessions?

i. Sessions are sorted subdirectories within each subject folder. Each subdirectory contains one daily recording and becomes one session in the output data structure.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The AI correctly identified each subdirectory as a daily recording session. This matches the data organization described in the paper and data README.

## 1-d. How are the data split into trials?

i. The AI splits the continuous recording into **2-minute (120-second) non-overlapping trials**. After 10-frame binning, this yields 360 bins per trial. Any remaining bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120   # 2 minutes (paper: "consecutive 2 minute blocks")
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial

# In process_session:
n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
```

iii. The AI chose 2-minute trials based on the paper's cross-validation structure, which uses "consecutive 2 minute blocks of the recording". CONVERSION_NOTES.md Step 5 documents this decision: "Split continuous recording into 2-minute blocks (matching paper's CV structure)".

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete trials (full 2-minute blocks) are included.

ii. N/A (no filtering code)

iii. The AI noted in CONVERSION_NOTES.md Step 3 that there is "No explicit trial curation (continuous spontaneous recording)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/` in each session directory.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. These are the standard suite2p output files. The AI documented this in CONVERSION_NOTES.md Step 1.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) neuropil subtraction with coefficient 0.7 (`Fc = F - 0.7 * Fneu`), (2) baseline correction using suite2p's `preprocess` function with the `maximin` method, and (3) **temporal binning by averaging 10 consecutive frames** (~333 ms bins).

ii.
```python
def compute_dfof(F, Fneu, fs=FRAME_RATE):
    Fc = F - NEUROPIL_COEFF * Fneu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dfof = s2p_preprocess(
        Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
        SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
    )
    return dfof.astype(np.float32)

# Temporal binning:
dfof_binned = bin_timeseries(dfof, BIN_SIZE)  # BIN_SIZE = 10

def bin_timeseries(data, bin_size):
    n_frames = data.shape[-1]
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. CONVERSION_NOTES.md Step 3 states: "Temporal binning: Average dF/F and behavior in bins of 10 consecutive timestamps for decoding", citing the paper's description. CONVERSION_NOTES.md Step 5 confirms: "Binning = 10 frames: Matches paper's decoding preprocessing".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural filtering is applied. The AI notes that the data is already pre-filtered through Track2p matching (all `iscell` values are 1).

ii. N/A (no filtering code)

iii. CONVERSION_NOTES.md Step 1: "All iscell values are 1 (pre-filtered through Track2p matching)." Step 4: "Data is pre-filtered (Track2p output in suite2p format). No additional filtering needed."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the continuous recording session. Since there are no stimulus events, no event-based alignment is applied. The neural data is simply split into contiguous blocks.

ii.
```python
'temporal_alignment_event': 'start of continuous recording session',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES.md Step 5: "No discrete trials in the original experiment (continuous spontaneous recording)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning by averaging 10 consecutive frames, resulting in a time bin size of ~333.33 ms (10 frames / 30 Hz). The final `time_bin_size` in metadata is set to 333.33 ms.

ii.
```python
BIN_SIZE = 10              # frames per bin
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # ~333.33 ms

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. CONVERSION_NOTES.md Step 3: "Decoding bin size: 10 frames = 333ms" citing the paper's statement about "averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index multiplied by the bin duration.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. Since the frame rate is constant and there are no explicit timestamps in the data, computing time from indices is equivalent.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Time is computed as `bin_index * (10 / 30)` seconds for each bin within a trial. The bin index is absolute within the session (not reset per trial), so time is cumulative from session start.

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
# start and end are absolute bin indices within the session
```

iii. No special processing beyond multiplication by the bin duration.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same bin indices used to slice the neural data, so it is inherently aligned.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. Using the same indices ensures perfect alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. This is the pre-computed global motion energy signal from behavioral video. The AI does NOT load or use `interframe_int.npy` for dropped frame detection (unlike the reference).

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) linear interpolation to match neural frame count when lengths differ, (2) temporal binning by averaging 10 consecutive frames, (3) discretization into 5 equal-percentile bins **per session**.

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

# Discretization (per session):
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. CONVERSION_NOTES.md Step 5: "Normalize motion energy per session" and "Discretize motion energy into 5 equal-percentile bins (per session)". The AI does NOT apply std normalization before discretization.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins **per session** using `np.digitize`. The percentile boundaries are [20, 40, 60, 80].

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The AI chose per-session discretization. This yields exactly 20% per bin within each session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. When the motion energy array is shorter than the neural data, the AI uses `np.interp` for linear interpolation to match the neural frame count. After interpolation and binning, the same bin indices are used to slice both neural and motion energy data.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
# ...
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
# ...
output_trials.append(me_discrete[start:end].reshape(1, -1))
```

iii. The AI uses generic linear interpolation rather than the reference approach of detecting specific dropped frames via `interframe_int.npy`.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (motion energy shorter than neural data) are handled by linear interpolation via `np.interp`. Remainder bins that don't fill a complete trial are discarded. No other error handling is present.

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

iii. The AI noted in CONVERSION_NOTES.md Step 2 that some sessions have fewer motion energy frames, and Step 4 resolved this with interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is `s2p_preprocess` (suite2p baseline correction), which performs sliding window operations across all neurons. This is GPU-accelerated when available.

ii.
```python
dfof = s2p_preprocess(
    Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
    SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
)
```

iii. CONVERSION_NOTES.md Step 7 shows ~0.5-1.0s per session, with total ~25s for 41 sessions. The baseline correction dominates.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials one at a time, but each iteration is just an array slice, so the overhead is minimal. No significant vectorization opportunities exist in the AI's code since the binning is already vectorized.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The AI's code is reasonably efficient; the main loop is I/O bound.

## 6-c. What processing does the code repeat multiple times?

i. The code does not appear to repeat any processing multiple times. Each session is processed once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI applies temporal binning (averaging 10 frames), which is an additional processing step. Additionally, the per-session motion energy discretization computes quintiles that may not align with the global distribution.

ii.
```python
dfof_binned = bin_timeseries(dfof, BIN_SIZE)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
```

iii. The binning reduces temporal resolution from 30 Hz to 3 Hz. Whether this is "unnecessary" depends on the decoder architecture, but the reference solution does not bin.
