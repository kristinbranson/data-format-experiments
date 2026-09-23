# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies subjects as directories matching `jm*` in the data directory, and sessions as subdirectories matching `20*` within each subject folder. For each session, it loads `spks.npy` (deconvolved calcium activity) and `iscell.npy` from `suite2p/plane0/`, and `motion_energy_glob.npy` and `tstamps.npy` from `move_deve/`. It selectively downloaded data from the Zenodo archive using byte-range HTTP requests rather than downloading the full ~15 GB archive.

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir() and any(p.glob('20*')))
# ...
session_dirs = sorted(p for p in (ROOT / subject).glob('20*') if p.is_dir())
# ...
spks = np.load(p2 / 'spks.npy')
iscell = np.load(p2 / 'iscell.npy')
motion_raw = np.load(move / 'motion_energy_glob.npy')
timestamps = np.load(move / 'tstamps.npy')
```

iii. The agent chose `spks.npy` because the supplied `load_data.ipynb` notebook comment suggests it as an "analysis-ready alternative to deriving dF/F from raw F." The agent reasoned this avoids downloading ~7 GB of raw fluorescence data while retaining Suite2p's processing. It used `tstamps.npy` (camera timestamps) instead of `interframe_int.npy` to detect and interpolate dropped frames.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories matching `jm*` in the data root, sorted alphabetically, with the additional constraint that each must contain at least one subdirectory matching `20*`.

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir() and any(p.glob('20*')))
```

iii. The `jm*` prefix identifies mouse subject folders. The `20*` check ensures only subjects with actual session data are included.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder matching the `20*` glob pattern (date-named folders), sorted alphabetically (which yields chronological order).

ii.
```python
session_dirs = sorted(p for p in (ROOT / subject).glob('20*') if p.is_dir())
```

iii. Session directories are date-named (e.g., `2023-10-22`), so alphabetical sorting produces chronological order. The `20*` filter ensures only date-format directories are selected.

## 1-d. How are the data split into trials?

i. Trials are non-overlapping 60-second blocks of the continuous recording. After 10-frame temporal averaging (30 Hz to 3 Hz), each trial is 180 time bins. Incomplete final blocks are discarded.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SECONDS / BIN_SECONDS))  # 180
n_trials = n_bins // BINS_PER_TRIAL
n_use = n_trials * BINS_PER_TRIAL
# ...
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
```

iii. The task instructions specify "Split sessions into 60-second trials." Since there is no natural trial structure, the agent uses fixed-length non-overlapping segmentation.

## 1-e. How are trials filtered based on quality controls?

i. The AI raises an error if a session has fewer than 2 complete trials, but otherwise does no trial-level filtering.

ii.
```python
if n_trials < 2:
    raise ValueError(f'fewer than two complete trials in {session_dir}')
```

iii. The instructions state "There needs to be at least two trials within each session in order to evaluate the decoder performance." The AI implements this as a hard check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spks.npy` (Suite2p's deconvolved calcium activity) and filtered by `iscell.npy`. This differs from the reference which uses `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence).

ii.
```python
spks = np.load(p2 / 'spks.npy')
iscell = np.load(p2 / 'iscell.npy')
```

iii. The agent cited the `load_data.ipynb` notebook which says "for more proper analysis compute dF/F the way as described in the paper (or alternatively use spks.npy)." The agent interpreted this as endorsing `spks.npy` as a valid alternative, avoiding the need to download ~7 GB of raw fluorescence data and implement dF/F computation.

## 2-b. How is the `neural` data processed?

i. The AI's processing consists only of: (1) iscell filtering (which is a no-op in practice), (2) casting to float32, and (3) 10-frame temporal averaging. No neuropil subtraction or baseline correction is performed because `spks.npy` is already a fully processed signal from Suite2p.

ii.
```python
keep = (iscell[:, 0] == 1) & (iscell[:, 1] >= 0.5)
if not np.all(keep):
    spks = spks[keep]
activity_binned = mean_blocks(np.asarray(spks, dtype=np.float32))
```

iii. Since the AI used `spks.npy` (deconvolved activity), the typical dF/F processing pipeline (neuropil subtraction with coefficient 0.7, maximin baseline correction) was not needed. The paper's 10-frame averaging was applied identically.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `iscell.npy`, requiring both `iscell[:, 0] == 1` (classified as cell) and `iscell[:, 1] >= 0.5` (cell probability >= 0.5). In practice, all neurons in the dataset already pass this threshold, so no neurons are actually removed.

ii.
```python
keep = (iscell[:, 0] == 1) & (iscell[:, 1] >= 0.5)
if not np.all(keep):
    spks = spks[keep]
```

iii. The paper states "We considered all ROIs above the default threshold of 0.5 as true cells." The agent applied this criterion as a safety measure, noting all neurons already pass since the Track2p pipeline outputs only successfully tracked cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second blocks of continuous recording, no event-based alignment is needed.

ii.
```python
'temporal_alignment_event': 'start of each consecutive 60-second session block',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and behavioral traces are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bins). This is applied before motion energy discretization.

ii.
```python
AVERAGE_FRAMES = 10
BIN_SECONDS = AVERAGE_FRAMES / FS  # 10/30 = 0.3333 s
# ...
activity_binned = mean_blocks(np.asarray(spks, dtype=np.float32))
motion_binned = mean_blocks(motion).astype(np.float32)
```

iii. The paper states "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent applies this identically.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index, using the center of each 10-frame averaging window.

ii.
```python
elapsed = (np.arange(n_use, dtype=np.float32) * AVERAGE_FRAMES +
           (AVERAGE_FRAMES - 1) / 2) / FS
```

iii. Since the frame rate is constant at 30 Hz, time can be computed directly from bin indices. The agent uses the midpoint of each bin's temporal window rather than the left edge.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each bin index `i`, the elapsed time is computed as `(i * 10 + 4.5) / 30`, representing the center of the 10-frame averaging window in seconds. This gives values starting at 0.15 s for the first bin.

ii.
```python
elapsed = (np.arange(n_use, dtype=np.float32) * AVERAGE_FRAMES +
           (AVERAGE_FRAMES - 1) / 2) / FS
```

iii. The center-of-bin approach is a standard convention for averaged/binned time series.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is computed from the same bin indices as the neural data, so alignment is inherent. Each time bin's elapsed time corresponds directly to the same bin in the neural and output arrays.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * BINS_PER_TRIAL, (tr + 1) * BINS_PER_TRIAL)
    ns.append(np.ascontiguousarray(activity_binned[:, sl], dtype=np.float32))
    ins.append(np.ascontiguousarray(elapsed[sl][None, :], dtype=np.float32))
```

iii. Since all data streams share the same binning and indexing scheme, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Camera timestamps (`tstamps.npy`) are used to detect and interpolate dropped frames.

ii.
```python
motion_raw = np.load(move / 'motion_energy_glob.npy')
timestamps = np.load(move / 'tstamps.npy')
motion = restore_motion(motion_raw, timestamps, spks.shape[1])
```

iii. The motion energy file contains pre-computed global motion energy from the behavioral video. Timestamps are needed to identify dropped camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) dropped frames are detected via camera timestamps and linearly interpolated using `np.interp`, (2) the trace is averaged into 10-frame bins, (3) the binned signal is discretized into 5 per-session quintile bins.

ii.
```python
# Interpolation
intervals = np.diff(timestamps)
dt = np.median(intervals)
steps = np.maximum(1, np.rint(intervals / dt).astype(np.int64))
x = np.concatenate(([0], np.cumsum(steps)))
return np.interp(np.arange(n_frames, dtype=np.float64), x, motion)

# Binning
motion_binned = mean_blocks(motion).astype(np.float32)

# Discretization
edges = np.percentile(x, [20, 40, 60, 80])
return np.digitize(x, edges, right=False).astype(np.int64), edges
```

iii. The timestamp-based interpolation is more robust than simple threshold detection, using median inter-frame intervals to identify gaps and `np.interp` for smooth filling. Quintile binning follows the task specification.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins per session. The 20th, 40th, 60th, and 80th percentiles are computed per session after temporal averaging, and `np.digitize` assigns labels 0-4.

ii.
```python
def quintile_labels(x):
    edges = np.percentile(x, [20, 40, 60, 80])
    return np.digitize(x, edges, right=False).astype(np.int64), edges
```

iii. The task specification says "Motion energy, discretized into five equal-percentile bins, selected per session." The implementation produces exactly balanced quintile classes per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. Dropped camera frames are detected using timestamp gaps and filled via linear interpolation, ensuring frame-for-frame alignment. After interpolation, both streams have identical length and are indexed identically.

ii.
```python
motion = restore_motion(motion_raw, timestamps, spks.shape[1])
# ...
if x[-1] != n_frames - 1:
    raise ValueError(f'timestamp gaps imply {x[-1] + 1} frames, expected {n_frames}')
return np.interp(np.arange(n_frames, dtype=np.float64), x, motion)
```

iii. The interpolation ensures the motion energy array matches the neural frame count exactly. The validation check confirms consistency.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are detected via camera timestamps and interpolated using `np.interp`. A validation check ensures the interpolated motion energy length matches the neural data length. Remainder frames at the end of sessions that don't fill a complete 60-second trial are discarded. If a session has fewer than 2 complete trials, an error is raised.

ii.
```python
if x[-1] != n_frames - 1:
    raise ValueError(f'timestamp gaps imply {x[-1] + 1} frames, expected {n_frames}')
# ...
if n_trials < 2:
    raise ValueError(f'fewer than two complete trials in {session_dir}')
```

iii. The agent prioritized data repair over data exclusion: no sessions or trials are discarded due to camera frame losses. Only truly incomplete trial blocks at session boundaries are removed.

## 6-a. What are the most time-consuming steps of the code?

i. Since the AI uses `spks.npy` (pre-computed), the most time-consuming steps are: loading the `.npy` files from disk, the `restore_motion` interpolation, and the 10-frame temporal averaging (`mean_blocks`). There is no GPU-based baseline correction step as in the reference.

ii. N/A

iii. By using pre-processed `spks.npy`, the AI avoids the computationally expensive `dcnv.preprocess` baseline correction that the reference solution performs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is already well-vectorized. The `restore_motion` function uses `np.interp` rather than a per-frame loop. The trial slicing loop is inherently sequential but lightweight.

ii. N/A

iii. The AI's approach to dropped frame interpolation (using `np.interp`) is more efficient than the reference's per-frame `np.insert` loop.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each session is processed once in a single pass.

ii. N/A

iii. The code processes each session sequentially without redundant computation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `iscell` filtering is technically unnecessary since all neurons in the dataset already pass the threshold. The per-session metadata (session_info with quintile edges, frame counts, etc.) is more detailed than required.

ii.
```python
keep = (iscell[:, 0] == 1) & (iscell[:, 1] >= 0.5)
if not np.all(keep):
    spks = spks[keep]
```

iii. The iscell check is a safety measure that has no practical effect on this dataset. The extra metadata, while not required, aids reproducibility.
