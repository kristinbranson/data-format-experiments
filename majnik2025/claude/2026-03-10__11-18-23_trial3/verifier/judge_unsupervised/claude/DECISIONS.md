# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all subject directories (directories starting with 'jm' under `data/`), then for each subject iterates over all session subdirectories sorted alphabetically. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` and `motion_energy_glob.npy` from `move_deve/`. No filtering of subjects or sessions is applied — all discovered directories are processed.

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

iii. The AI followed the data README which describes the folder structure. The `load_data.ipynb` notebook in the data directory uses the same approach: scanning subject directories and loading `F.npy` from `suite2p/plane0/`. The AI noted this is consistent with the Track2p output format described in the paper.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by top-level directories under `data/` starting with 'jm'. Each subject directory contains session subdirectories. The AI maps internal IDs (jm031-jm046) to paper names (Mouse_A through Mouse_F) using a hardcoded mapping. 6 subjects total.

ii.
```python
SUBJECT_MAP = {
    'jm031': 'Mouse_A', 'jm032': 'Mouse_B', 'jm038': 'Mouse_C',
    'jm039': 'Mouse_D', 'jm040': 'Mouse_E', 'jm046': 'Mouse_F'
}

subjects = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('jm')])
```

iii. The data README states "For cross-referencing with the paper the subjects are named in alphabetically increasing order (e.g. jm031 - mouse A, jm032 - mouse B ... jm046 - mouse F)." The AI's mapping follows this exactly.

## 1-c. How are the data split into sessions?

i. Sessions correspond to subdirectories within each subject folder (one per recording day). Sessions are sorted alphabetically (which corresponds to chronological order since folders are named by date). The total is 41 sessions: 7 each for 5 mice plus 6 for jm040.

ii.
```python
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The data README states "Each subject folder contains a number of session folders, each corresponding to one recording day. The name of the folder corresponds to the recording date in the YYYY-MM-DD format." The AI's approach matches this description.

## 1-d. How are the data split into trials?

i. The original data is continuous (spontaneous recordings, no discrete trials). The AI splits each session into consecutive non-overlapping 2-minute blocks (360 time bins each after 10-frame binning). This yields 10 trials per 20-min session (jm031, jm032) and 15 trials per 30-min session (jm038-jm046). Leftover bins at the end that don't fill a complete 2-min block are discarded.

ii.
```python
TRIAL_DURATION_SEC = 120   # 2 minutes
TRIAL_BINS = int(TRIAL_DURATION_SEC * FRAME_RATE / BIN_SIZE)  # 360 bins per trial

n_total_bins = dfof_binned.shape[1]
n_trials = n_total_bins // TRIAL_BINS

for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The paper describes "consecutive 2 minute blocks of the recording" in the context of cross-validation. The AI adopted this as the trial definition, reasoning that it matches the paper's CV block structure. The CONVERSION_NOTES document this decision: "Trial size = 2 minutes (360 bins): Matches paper's CV block structure."

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 2-minute blocks are included. Incomplete blocks at the end of sessions are silently discarded. No quality-based filtering is performed on trials.

ii.
```python
n_trials = n_total_bins // TRIAL_BINS  # integer division discards remainder
for t in range(n_trials):
    # All trials are appended, no filtering
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
```

iii. The AI noted in CONVERSION_NOTES: "No explicit trial curation (continuous spontaneous recording)." This is consistent with the paper which describes continuous spontaneous activity recordings without trial-based quality filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `F.npy` (raw fluorescence traces) and `Fneu.npy` (neuropil fluorescence traces), both from the `suite2p/plane0/` directory of each session.

ii.
```python
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. The paper states they used "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." Suite2p computes dF/F from the raw fluorescence (F) and neuropil (Fneu) traces. The data README notes the `load_data.ipynb` loads F.npy and suggests computing "dF/F the way as described in the paper."

## 2-b. How is the `neural` data processed?

i. Processing has three steps: (1) Neuropil correction: Fc = F - 0.7 * Fneu; (2) Baseline correction using Suite2p's `preprocess` function with the maximin method (parameters: win_baseline=60s, sig_baseline=10 frames, prctile_baseline=8, fs=30Hz); (3) Temporal binning by averaging 10 consecutive frames (~333ms bins).

ii.
```python
def compute_dfof(F, Fneu, fs=FRAME_RATE):
    Fc = F - NEUROPIL_COEFF * Fneu  # NEUROPIL_COEFF = 0.7
    dfof = s2p_preprocess(
        Fc.copy().astype(np.float32), BASELINE_METHOD, WIN_BASELINE,
        SIG_BASELINE, fs, PRCTILE_BASELINE, device=device
    )
    return dfof.astype(np.float32)

def bin_timeseries(data, bin_size):
    n_bins = n_frames // bin_size
    truncated = data[..., :n_bins * bin_size]
    new_shape = truncated.shape[:-1] + (n_bins, bin_size)
    return truncated.reshape(new_shape).mean(axis=-1)
```

iii. The paper states: "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The AI confirmed Suite2p parameters from ops.npy (baseline="maximin", win_baseline=60, sig_baseline=10, fs=30, neucoeff=0.7). The 10-frame binning matches the paper's "averaging in bins of 10 consecutive timestamps."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI determined that the provided data is already filtered through Track2p matching — only neurons tracked across all days are included, and all `iscell` values are already 1. All neurons from F.npy are used directly.

ii.
```python
# No filtering code — all neurons from F.npy are used
F = np.load(os.path.join(sess_dir, 'suite2p', 'plane0', 'F.npy'))
n_neurons, n_frames = F.shape
# n_neurons used directly, no subsetting
```

iii. CONVERSION_NOTES: "All iscell values are 1 (pre-filtered through Track2p matching)" and "Data is pre-filtered by Track2p (all iscell=1)." The data README confirms: "the data only includes traces for the cells present across all days." The paper describes iscell threshold of 0.5 which was already applied by Track2p.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the continuous recording session. Each trial starts at a fixed offset from the session start (trial t starts at t * 360 bins = t * 120 seconds). There is no explicit event-based alignment — the recording start serves as the alignment event.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS  # TRIAL_BINS = 360
    end = (t + 1) * TRIAL_BINS
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))

# metadata:
'temporal_alignment_event': 'start of continuous recording session',
'off_start': 0.0,
'off_end': None,
```

iii. This is a continuous spontaneous activity experiment with no discrete trial events (no stimulus onsets, go cues, etc.). The AI correctly identified that alignment to the recording start is the only meaningful option. The paper describes spontaneous behavior, not a trial-based task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw data is at 30 Hz (one frame every ~33.3ms). The AI applies temporal rebinning by averaging every 10 consecutive frames, resulting in a time bin size of ~333.33 ms (3 Hz effective rate). This matches the paper's description.

ii.
```python
BIN_SIZE = 10              # frames per bin
FRAME_RATE = 30.0          # Hz
TIME_BIN_MS = BIN_SIZE / FRAME_RATE * 1000  # ~333.33 ms

dfof_binned = bin_timeseries(dfof, BIN_SIZE)
```

iii. The paper states: "averaging in bins of 10 consecutive timestamps" for decoding analysis. The AI's bin size exactly matches.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The input is not derived from any raw data variable. It is computed from the bin indices of each trial within the session. Each bin's time is calculated as `bin_index * (BIN_SIZE / FRAME_RATE)` seconds, where `bin_index` is the global bin index within the session (not reset per trial).

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The instructions specify "Time elapsed from the beginning of the experiment" as the decoder input. Since there's no explicit timestamp variable in the raw data, the AI computed it from the known frame rate and bin structure. This is a reasonable approach since the microscope operates at a fixed 30 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each trial, the time input is computed as: `time_sec = bin_index * (10 / 30)` seconds, where `bin_index` ranges from the trial's start bin to end bin within the full session. The first trial starts at time 0.0s, and the last bin of the last trial reaches up to 1199.7s (20-min sessions) or 1799.7s (30-min sessions).

ii.
```python
time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
# For trial 0: np.arange(0, 360) * (10/30) = [0.0, 0.333, ..., 119.67]
# For trial 1: np.arange(360, 720) * (10/30) = [120.0, 120.333, ..., 239.67]
input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. The AI noted this represents "time elapsed from start of recording in seconds" and confirmed in verification that the input range is [0.0, 1199.7] for 20-min sessions and [0.0, 1799.7] for 30-min sessions, which is consistent with the session durations.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is inherently aligned with the neural data because it is computed from the same bin indices used to extract neural trial data. The time for bin `i` in the session corresponds exactly to neural data at bin `i`.

ii.
```python
for t in range(n_trials):
    start = t * TRIAL_BINS
    end = (t + 1) * TRIAL_BINS
    # Neural and time use the same start/end indices
    neural_trials.append(dfof_binned[:, start:end].astype(np.float32))
    time_sec = np.arange(start, end) * (BIN_SIZE / FRAME_RATE)
    input_trials.append(time_sec.reshape(1, -1).astype(np.float32))
```

iii. Since both neural data and time input are derived from the same bin indexing scheme, they are perfectly aligned by construction. No separate alignment step is needed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The output is derived from `motion_energy_glob.npy` in the `move_deve/` subdirectory of each session. This is the pre-computed global motion energy from videography.

ii.
```python
me_raw = np.load(os.path.join(sess_dir, 'move_deve', 'motion_energy_glob.npy'))
```

iii. The paper describes motion energy as "pixel-wise squared difference of consecutive video frames, summed across pixels." The data README describes `move_deve/` as containing "processed behavioural data (motion energy extracted from videography of spontaneous behaviour 'motion_energy_glob.npy')."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Interpolate motion energy to match neural frame count (handles missing camera frames); (2) Bin by averaging 10 consecutive frames (matching neural binning); (3) Discretize into 5 equal-percentile (quintile) bins per session.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)
```

iii. The instructions specify "Motion energy, normalized and discretized into five equal-percentile bins." The paper uses binning of 10 frames for decoding. The AI applied per-session quintile discretization to produce exactly 20% in each bin.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized per session into 5 equal-percentile bins using `np.percentile` to find the 20th, 40th, 60th, and 80th percentile boundaries, then `np.digitize` to assign bin labels 0-4.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    thresholds = np.percentile(me_binned, percentiles)
    labels = np.digitize(me_binned, thresholds).astype(np.int64)
    return labels
```

iii. The instructions say "normalized and discretized into five equal-percentile bins." The AI computes quintile boundaries per session, which produces exactly 20% of data points in each bin (confirmed in verification output). This matches the instruction requirement.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The raw motion energy is first interpolated to match the neural frame count (handling missing camera frames), then binned using the same 10-frame binning as neural data, and finally split into trials using the same bin indices. This ensures frame-by-frame alignment.

ii.
```python
me = interpolate_motion_energy(me_raw, n_frames)  # match neural frame count
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()  # same binning
# Same trial splitting as neural:
output_trials.append(me_discrete[start:end].reshape(1, -1).astype(np.int64))
```

iii. The data README explains the microscope trigger initiates camera acquisition, providing frame-by-frame synchronization. When camera frames are missing, interpolation restores alignment. The AI's approach of interpolating first, then applying identical binning and trial splitting, maintains this alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera frames (where motion energy has fewer frames than neural data) are handled by linear interpolation to match the neural frame count. The AI documented specific sessions with missing frames (e.g., jm031 sessions 3-5 have 35998, 35997, 35884 ME frames vs 36000 neural). No other missing data handling is implemented — there are no NaN checks or outlier removal.

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

iii. The data README states: "indices of missing frames can be obtained by looking at tstamps.npy or interframe_int.npy and treated as missing values or interpolated over." The AI chose interpolation, which is one of the two suggested approaches. The CONVERSION_NOTES document all sessions with frame count mismatches.

## 6-a. What are the most time-consuming steps of the code?

i. The dF/F computation via Suite2p's `preprocess` function is the most time-consuming step, as it involves baseline estimation using a sliding window method with Gaussian smoothing. The full conversion runs in ~25.5 seconds for 41 sessions. Per-session time ranges from 0.1s (small sessions, jm031) to ~1.0s (large sessions, jm039).

ii. No explicit profiling code, but timing per session is printed:
```python
t_sess = time.time()
# ... all processing ...
elapsed = time.time() - t_sess
print(f"  {n_neurons} neurons, {len(neural_trials)} trials, ... time: {elapsed:.1f}s")
```

iii. From CONVERSION_NOTES: "Full processing ~0.5s (36k frames), ~0.7s (54k frames)" per session, with estimated total ~25s. The actual total was 25.5s, confirming the estimate was accurate.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial splitting loop iterates over trials to slice arrays and build lists. This could theoretically be replaced with a single reshape operation for the neural data (reshape to (n_neurons, n_trials, TRIAL_BINS) then split along axis 1), though the current approach is already efficient since it uses array slicing.

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

iii. The code is already reasonably efficient. The binning operation is vectorized via reshape+mean. The trial splitting loop is simple array slicing which is fast. The total conversion time of 25.5s for 41 sessions suggests no major efficiency issues.

## 6-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed exactly once. The dF/F computation, binning, and discretization are all done once per session. The only minor redundancy is that `np.arange(start, end)` is computed inside the trial loop when it could theoretically be precomputed, but this is negligible.

ii. The main processing pipeline runs once per session with no repeated steps:
```python
dfof = compute_dfof(F, Fneu)          # once
dfof_binned = bin_timeseries(dfof, BIN_SIZE)  # once
me_binned = bin_timeseries(me.reshape(1, -1), BIN_SIZE).squeeze()  # once
me_discrete = discretize_motion_energy(me_binned, N_OUTPUT_BINS)  # once
```

iii. The code is efficiently structured with each major computation performed once per session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes the entire continuous recording but discards leftover bins at the end that don't fill a complete 2-minute trial. With 10-frame binning, 36000 frames → 3600 bins → exactly 10 trials (no waste), and 54000 frames → 5400 bins → exactly 15 trials (no waste). However, when sessions have slightly fewer frames due to missing camera frames, the bin_timeseries function already truncates to the nearest multiple of bin_size, potentially discarding a few frames.

ii.
```python
# In bin_timeseries:
n_bins = n_frames // bin_size
truncated = data[..., :n_bins * bin_size]  # truncates remainder

# In trial splitting:
n_trials = n_total_bins // TRIAL_BINS  # discards remainder bins
```

iii. The data loss is minimal — at most a few frames at the end of sessions with non-integer multiples of the bin size. The AI documented this and confirmed the frame counts (36000, 54000) are exact multiples of the bin size (10), so no neural data is lost from binning.
