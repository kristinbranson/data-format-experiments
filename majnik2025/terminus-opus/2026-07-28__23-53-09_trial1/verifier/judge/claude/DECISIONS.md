# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories starting with `jm` in the data directory, sessions as subdirectories within each subject folder, and loads calcium data from suite2p output files (`F.npy`, `Fneu.npy`) and motion energy from `motion_energy_glob.npy`. Timestamps are loaded from `tstamps.npy` for frame alignment.

ii.
```python
def get_subjects_and_sessions(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
    sessions = {}
    for subj in subjects:
        subj_path = os.path.join(data_dir, subj)
        sess_list = sorted([s for s in os.listdir(subj_path)
                           if os.path.isdir(os.path.join(subj_path, s))])
        sessions[subj] = sess_list
    return subjects, sessions

# In process_session:
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The AI follows the standard directory structure convention. All `jm*` directories are included as subjects, all subdirectories within each subject as sessions. This matches the reference approach.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data directory, sorted alphabetically. This is the same as the reference.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. Each `jm*` directory represents one mouse. The naming convention is consistent across the dataset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
sess_list = sorted([s for s in os.listdir(subj_path)
                   if os.path.isdir(os.path.join(subj_path, s))])
```

iii. Each subdirectory contains suite2p output and motion energy files for one recording session. Sorting ensures deterministic order.

## 1-d. How are the data split into trials?

i. The AI splits sessions into **120-second (2-minute)** non-overlapping segments, yielding 360 binned timepoints per trial (120s x 30Hz / 10 = 360). The instructions explicitly state "Split sessions into 60-second trials", and the reference uses 60-second trials (180 binned timepoints per trial). The AI chose 2-minute blocks because the paper describes cross-validation splits of 2-minute blocks.

ii.
```python
TRIAL_DURATION_SEC = 120  # 2 minutes per trial
TRIAL_FRAMES_RAW = int(TRIAL_DURATION_SEC * FS)  # 3600 raw frames per trial
TRIAL_FRAMES_BINNED = TRIAL_FRAMES_RAW // BIN_SIZE  # 360 binned frames per trial
```

iii. The AI's CONVERSION_NOTES.md states: "Split each session into 2-minute blocks (as used for CV in paper)". The AI followed the paper's CV block structure rather than the explicit instruction of 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Remainder frames that don't fill a complete trial at the end of a session are discarded. This matches the reference.

ii. Trial splitting discards remainder via integer division:
```python
n_trials = n_timepoints // trial_length
```

iii. No trial quality filtering criteria are described in the paper or reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. This matches the reference.

ii.
```python
F = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(sess_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - 0.7 * Fneu`), followed by suite2p's `dcnv.preprocess` with `maximin` baseline method. The AI does not pass `prctile_baseline=8.0` or `batch_size` parameters to the preprocess function, relying on defaults instead. The AI also forces CPU computation (`device = torch.device('cpu')`) instead of auto-detecting GPU.

ii.
```python
def compute_dff(F, Fneu, neucoeff=NEUCOEFF, win_baseline=WIN_BASELINE,
                sig_baseline=SIG_BASELINE, fs=FS):
    Fc = F - neucoeff * Fneu
    device = torch.device('cpu')
    dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
    return dff
```

iii. The AI states this follows suite2p's standard preprocessing pipeline as described in the paper methods. The missing parameters (`prctile_baseline`, `batch_size`) may use suite2p's defaults, which could differ from the reference's explicit values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons in `F.npy` are included since they are already pre-filtered by Track2p (all `iscell` values are 1). This matches the reference.

ii. N/A (no filtering code)

iii. Suite2p's cell detection plus Track2p's cell tracking already identifies valid ROIs. No further filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous segments of continuous recording, no event-based alignment is needed. This matches the reference.

ii.
```python
'temporal_alignment_event': 'session_start',
'off_start': 0.0,
'off_end': None,
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, taking 30 Hz to 3 Hz (333.33 ms time bin). This matches the reference.

ii.
```python
BIN_SIZE = 10  # number of frames to average
TIME_BIN_MS = (BIN_SIZE / FS) * 1000  # 333.33 ms

def bin_data(data, bin_size=BIN_SIZE):
    if data.ndim == 2:
        n_neurons, n_frames = data.shape
        n_bins = n_frames // bin_size
        data_trimmed = data[:, :n_bins * bin_size]
        return data_trimmed.reshape(n_neurons, n_bins, bin_size).mean(axis=2)
    elif data.ndim == 1:
        n_frames = len(data)
        n_bins = n_frames // bin_size
        data_trimmed = data[:n_bins * bin_size]
        return data_trimmed.reshape(n_bins, bin_size).mean(axis=1)
```

iii. The Methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is computed from the time bin index and the bin duration, giving seconds from the start of the session. This matches the reference approach of computing time from indices rather than from any raw data variable.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. Since the frame rate is constant at 30 Hz, computing time from bin indices is equivalent to using stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes time as the **center** of each bin: `(bin_index * BIN_SIZE + BIN_SIZE/2) / FS`. The reference uses the **left edge** of each bin: `bin_index * BIN_SIZE / FS`. This is a minor difference.

ii.
```python
# AI's code (center of bin):
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS

# Reference code (left edge of bin):
t = ((s + np.arange(trial_frames)) * BIN_FRAMES / FS).astype(np.float32)
```

iii. The AI uses the center of the time bin as a more accurate representation of the averaged time interval. The reference uses the left edge.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices as the neural data, so alignment is inherent. This matches the reference approach.

ii.
```python
time_input = (np.arange(n_binned) * BIN_SIZE + BIN_SIZE / 2) / FS
# Later sliced to match neural trials:
for i in range(len(neural_trials)):
    start = i * TRIAL_FRAMES_BINNED
    end = start + TRIAL_FRAMES_BINNED
    trial_time = time_input[start:end].astype(np.float32)
    input_trials_list.append(trial_time.reshape(1, -1))
```

iii. Time is derived from the same indices used for neural data segmentation, so alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. The AI also uses `tstamps.npy` (camera timestamps) for dropped frame detection, whereas the reference uses `interframe_int.npy` (inter-frame intervals).

ii.
```python
me = np.load(os.path.join(sess_path, 'move_deve', 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(sess_path, 'move_deve', 'tstamps.npy'))
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three processing steps: (1) dropped frames are detected via timestamp-based analysis (IFI > 1.5x median) and interpolated using `np.interp`, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 percentile-based bins computed within each session. This is similar to the reference but uses a different dropped-frame detection method and interpolation approach.

ii.
```python
# Dropped frame detection and interpolation:
def align_motion_energy(me, n_neural_frames, tstamps):
    ifi = np.diff(tstamps)
    median_ifi = np.median(ifi)
    neural_idx = np.zeros(len(me), dtype=int)
    neural_idx[0] = 0
    for i in range(1, len(me)):
        n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
        neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
    neural_idx = np.clip(neural_idx, 0, n_neural_frames - 1)
    me_full = np.full(n_neural_frames, np.nan, dtype=np.float64)
    me_full[neural_idx] = me.astype(np.float64)
    valid = ~np.isnan(me_full)
    if not valid.all():
        indices = np.arange(n_neural_frames)
        me_full = np.interp(indices, indices[valid], me_full[valid])
    return me_full

# Discretization:
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. The AI uses timestamps to reconstruct which neural frames correspond to which camera frames. The reference uses inter-frame intervals with a fixed threshold. Both aim to handle missing camera frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins per session using `np.percentile` and `np.digitize`. This matches the reference approach.

ii.
```python
def discretize_motion_energy(me_binned, n_bins=N_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    bin_edges = np.percentile(me_binned, percentiles)
    binned = np.digitize(me_binned, bin_edges)
    return binned.astype(np.int64)
```

iii. Equal-percentile binning ensures approximately equal numbers of samples in each bin. Bin edges are computed per session to account for different motion levels across days/mice.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy to neural data by mapping camera frames to neural frames using timestamp-based inter-frame interval analysis. When the motion energy array is shorter than neural data (dropped frames), it detects gaps > 1.5x median IFI, maps ME values to the correct neural frame indices, and interpolates the gaps. After alignment, both streams are binned together by averaging 10 frames.

ii.
```python
me_aligned = align_motion_energy(me, n_neural_frames, tstamps)
# ... then both are binned:
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
```

iii. The AI's approach uses `tstamps.npy` and computes IFI from timestamps, while the reference uses `interframe_int.npy` directly with a threshold of `dt * 1000 > 0.04`. Both methods aim to achieve the same result of aligning the motion energy to neural data length.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected and interpolated (see 4-b/4-d). Remainder frames at the end of a session that don't fill a complete trial are discarded via integer division. This is similar to the reference approach.

ii.
```python
# Dropped frames handled in align_motion_energy()
# Remainder frames discarded:
n_trials = n_timepoints // trial_length
```

iii. The interpolation ensures the motion energy signal matches the neural data length. Discarding remainder frames is minor data loss.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the suite2p `dcnv.preprocess` baseline correction (~0.4-0.6s per session). The AI forces CPU computation, whereas the reference auto-detects GPU availability.

ii.
```python
device = torch.device('cpu')
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
```

iii. The baseline correction involves sliding window operations over the full session length for every neuron. The AI's CPU-only approach may be slower than the reference's GPU-enabled approach.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The dropped frame detection loop in `align_motion_energy` iterates frame by frame to build the mapping. The trial splitting loop in `split_into_trials` also iterates per trial. These could potentially be vectorized.

ii.
```python
# Frame-by-frame loop in align_motion_energy:
for i in range(1, len(me)):
    n_dropped = max(0, round(ifi[i-1] / median_ifi) - 1)
    neural_idx[i] = neural_idx[i-1] + 1 + n_dropped
```

iii. The number of dropped frames is typically very small, so the performance impact of these loops is negligible.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each session is processed once in `process_session`.

ii. N/A

iii. The code processes each session in a single pass.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI copies the fluorescence array before preprocessing (`Fc.copy()`), which is unnecessary memory usage. The remainder frames beyond the last full trial are computed but then discarded.

ii.
```python
dff = preprocess(Fc.copy(), 'maximin', win_baseline, sig_baseline, fs, device=device)
```

iii. The copy is a safety measure to avoid modifying the input array, but since `Fc` is not reused, it's unnecessary.
