# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as all subdirectories in the data directory (sorted alphabetically). Sessions are subdirectories within each subject folder. For each session, calcium data is loaded from suite2p output files (`F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`), and behavior data from `motion_energy_glob.npy` and `tstamps.npy` in the `move_deve` subdirectory.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
# ...
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
# ...
me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
```

iii. The agent explored the data directory structure, the paper's methods, and the Track2p reference code. It identified suite2p plane0 output and move_deve behavioral files as the core data sources. It also loads `ops.npy` to get the actual frame rate and `iscell.npy` to verify all ROIs pass quality checks. It uses `tstamps.npy` (camera timestamps) rather than `interframe_int.npy` for dropped frame recovery.

## 1-b. How are the data split into subjects?

i. Subjects correspond to all subdirectories in the data directory, sorted alphabetically. Unlike the reference which filters for directories starting with `jm`, the AI includes all directories.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
```

iii. The agent found that all subdirectories in the data folder correspond to mouse subjects. Since the dataset only contains subject directories, no prefix filtering was deemed necessary.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording.

ii.
```python
sessions = sorted(d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d)))
```

iii. The agent found session directories named by date (e.g., `2023-10-22_a`), each containing a suite2p folder and behavioral data for one recording session. Sorting ensures chronological order.

## 1-d. How are the data split into trials?

i. Trials are defined as 60-second non-overlapping segments of the continuous recording. At 30 Hz with 10-frame bins, this gives 180 bins per trial. Remainder bins that don't fill a complete trial are discarded. An assertion verifies at least 2 trials per session.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * NOMINAL_FS / FRAMES_PER_BIN))  # 180
# ...
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))
```

iii. Per instruction, trials are 60-second segments. The agent verified this yields 20 trials for 20-min sessions (jm031, jm032) and 30 trials for 30-min sessions (others).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are included; only the remainder frames at the end are discarded. An assertion ensures at least 2 trials per session.

ii.
```python
assert n_trials >= 2
```

iii. There is no stimulus-driven trial structure, so no trial-quality criteria apply.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The frame rate `fs` is read from `ops.npy`.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
fs = float(ops['fs'])
```

iii. These are the standard suite2p output files for fluorescence traces. The agent also loads `ops.npy` to get the actual imaging frame rate rather than assuming 30 Hz.

## 2-b. How is the `neural` data processed?

i. The agent reimplements the Track2p `F_processing` function: **no neuropil subtraction** (`neucoeff=0`), followed by maximin baseline correction using Gaussian smoothing (sigma=10 frames), minimum filter (60s window), then maximum filter (60s window), and finally `dF = Fc - Flow`. This is implemented using scipy's `gaussian_filter`, `minimum_filter1d`, and `maximum_filter1d` rather than calling suite2p's `dcnv.preprocess`.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow
# ...
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
```

iii. The agent found the Track2p code's `F_processing` function in `track2p/gui/data_management.py` which defaults to `neucoeff=0`. The agent chose to follow the Track2p code implementation over the potentially ambiguous paper statement about "default Suite2p parameters." The agent reasoned this was the actual processing used in the paper's analyses since it's the Track2p GUI's default.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The agent verifies via assertion that all ROIs in `iscell.npy` are marked as cells (all have probability 1), confirming the Track2p output already contains only tracked, quality-filtered neurons.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in Track2p output'
```

iii. The agent verified empirically that `iscell[:,0].sum()` equals the total ROI count for every session, confirming the released data is already curated. No further filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The metadata provides `off_start=0.0` and `off_end` equal to the trial duration in seconds.

ii.
```python
'temporal_alignment_event':
    'start of the 60 s trial block; each session is cut into '
    'consecutive non-overlapping 1800-frame (60 s) blocks starting '
    'at imaging onset',
'off_start': 0.0,
'off_end': float(np.mean(bin_durations) * BINS_PER_TRIAL),
```

iii. There is no stimulus event to align to. The recording is continuous and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing from ~30 Hz to ~3 Hz (~336 ms time bins). The exact bin duration is computed from the actual median inter-frame interval across sessions rather than assuming exactly 1/30 s.

ii.
```python
FRAMES_PER_BIN = 10
# ...
dff_b = bin_mean(dff, FRAMES_PER_BIN)
me_b = bin_mean(me, FRAMES_PER_BIN)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
# ...
time_bin_size_ms = float(np.mean(bin_durations) * 1000.0)
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent also bins the timestamps to get bin-center times.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is derived from `tstamps.npy` — the camera timestamps (in units of 1000 seconds, converted by multiplying by 1000). These are interpolated to the imaging frame grid (filling dropped frames), then averaged in 10-frame bins to get bin-center times.

ii.
```python
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
# ...
tstamps = np.full(n_frames, np.nan)
tstamps[idx] = ts_cam
tstamps = interp_nan(tstamps)
# ...
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
```

iii. The agent used actual camera timestamps rather than computing time from bin indices, which accounts for any drift in the actual frame rate.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Camera timestamps (in 1000s units) are converted to seconds, placed on the imaging frame grid using `camera_frame_index`, NaN-interpolated for missing frames, then averaged in 10-frame bins. The resulting bin-center times are sliced per trial.

ii.
```python
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
tstamps = np.full(n_frames, np.nan)
tstamps[idx] = ts_cam
tstamps = interp_nan(tstamps)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
# ...
ninp.append(t[sl][None, :].astype(np.float32))
```

iii. Using actual timestamps rather than computed indices preserves any small deviations in frame timing.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The timestamps are placed onto the same imaging frame grid as the neural data, then binned identically (10-frame averages), so they share the same time axis. The same trial slicing is applied.

ii.
```python
dff_b = bin_mean(dff, FRAMES_PER_BIN)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
# both sliced with the same trial slice:
ntr.append(np.ascontiguousarray(dff[:, sl]))
ninp.append(t[sl][None, :].astype(np.float32))
```

iii. Since both neural data and timestamps live on the same imaging frame grid and are binned together, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Camera timestamps from `tstamps.npy` are used to map camera frames to imaging frames (handling dropped frames).

ii.
```python
me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
```

iii. The motion energy file contains a pre-computed global motion energy signal. The timestamps are needed to align camera frames with imaging frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) Camera frames are mapped to imaging frames using `camera_frame_index` based on timestamps; dropped frames and frame 0 are NaN-filled and linearly interpolated. (2) The trace is averaged into 10-frame bins. (3) The binned signal is discretized into 5 equal-percentile (quintile) bins per session.

ii.
```python
idx = camera_frame_index(ts_cam, n_frames)
me = np.full(n_frames, np.nan)
me[idx] = me_cam
me[0] = np.nan  # frame 0 ME is 0 by construction
me = interp_nan(me)
# ...
me_b = bin_mean(me, FRAMES_PER_BIN)
# ...
def discretize_quintiles(x, nbins=N_OUTPUT_BINS):
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)
```

iii. The agent reconstructed the imaging-frame index from camera timestamps, detecting dropped frames as gaps that are exact multiples of the frame interval. Frame 0's motion energy is treated as missing since there's no preceding frame to compute a difference against.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) per session. Bin edges are computed using `np.percentile` with edges at 20th, 40th, 60th, and 80th percentiles, and `np.digitize` assigns each sample to a bin (0-4).

ii.
```python
def discretize_quintiles(x, nbins=N_OUTPUT_BINS):
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)
```

iii. The instructions specify "five equal-percentile bins, selected per session." The agent computes quintile edges within each session independently.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera was hardware-triggered by the microscope, so camera frame i corresponds to imaging frame i. Dropped camera frames create gaps that are recovered using the camera timestamps to reconstruct the true imaging-frame index. Missing values are linearly interpolated. After alignment, both neural and motion energy are on the same frame grid and are binned identically.

ii.
```python
def camera_frame_index(tstamps, n_frames):
    ncam = len(tstamps)
    if ncam == n_frames:
        return np.arange(n_frames)
    ifi = np.diff(tstamps)
    dt = np.median(ifi)
    n_missing = np.round(ifi / dt).astype(int) - 1
    idx = np.concatenate([[0], np.cumsum(1 + n_missing)])
    assert idx[-1] == n_frames - 1
    assert n_missing.sum() == n_frames - ncam
    return idx
# ...
me = np.full(n_frames, np.nan)
me[idx] = me_cam
me = interp_nan(me)
```

iii. The agent systematically investigated dropped frames across all sessions, verified the frame recovery algorithm, and confirmed all 41 sessions align correctly.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected from timestamp gaps and linearly interpolated. Frame 0's motion energy is treated as missing (since there's no preceding frame). Assertions verify frame count consistency. Remainder frames at session end that don't fill a complete trial are discarded.

ii.
```python
me[0] = np.nan  # frame 0 ME is 0 by construction
me = interp_nan(me)
# ...
assert idx[-1] == n_frames - 1, 'could not reconstruct dropped camera frames'
assert n_missing.sum() == n_frames - ncam
```

iii. The agent investigated timestamp files to detect missing data and implemented robust recovery using the camera timestamps rather than relying on heuristic thresholds.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the `f_processing` baseline correction, which applies Gaussian filtering, minimum filtering, and maximum filtering across the full session length for every neuron using scipy. Loading `.npy` files is I/O bound but fast. The agent noted the code runs on CPU (scipy operations, no GPU acceleration).

ii. N/A

iii. The scipy-based implementation lacks the GPU acceleration that suite2p's `dcnv.preprocess` provides.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial slicing loop iterates over trials sequentially using Python slicing, but this is inherently sequential due to the list-of-arrays output format. The `camera_frame_index` computation is already vectorized.

ii. N/A

iii. The loop overhead is minimal since the main work (array slicing) is done by NumPy.

## 6-c. What processing does the code repeat multiple times?

i. The quintile edge computation is done twice for sessions where `session_info` metadata is populated: once in `discretize_quintiles` and once in the session_info dictionary construction.

ii.
```python
cls = discretize_quintiles(me)  # computes percentile edges
# ...
'motion_energy_quintile_edges': [
    float(v) for v in np.percentile(
        me, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1])],
```

iii. This duplication is minor and does not significantly affect performance.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `iscell.npy` and `ops.npy` solely for assertion/verification purposes — the actual cell filtering is not needed since all ROIs are already cells. The session_info metadata includes detailed per-session information that may not be used by the decoder.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
assert np.all(iscell[:, 0] == 1)
```

iii. These are verification steps rather than data processing, so they add minimal overhead.
