# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories starting with `jm` in the data directory. Sessions are subdirectories within each subject folder. For each session, calcium data is loaded from suite2p output files (`F.npy`, `Fneu.npy`, `ops.npy`) in `suite2p/plane0/`, and motion energy from `motion_energy_glob.npy` and timestamps from `tstamps.npy` in `move_deve/`. The agent also loads `ops.npy` to extract per-session processing parameters (fs, neucoeff, baseline settings).

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir())
...
sessions = sorted(p for p in (ROOT / subject).iterdir() if p.is_dir())
...
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
...
motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(md / 'tstamps.npy').astype(np.float64)
```

iii. The agent examined the directory structure, data README, and the paper's notebook (`load_data.ipynb`) to understand the data organization. It systematically loaded all `.npy` files in each session to inspect shapes, dtypes, and value ranges before writing the conversion code. It chose to load `ops.npy` to extract Suite2p parameters rather than hardcoding them, and uses `mmap_mode='r'` for memory efficiency.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories matching the glob `jm*` in the data root, sorted alphabetically. All 6 mice (jm031, jm032, jm038, jm039, jm040, jm046) are included.

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir())
```

iii. The agent identified the naming convention from the data directory structure. Each `jm*` directory represents one mouse.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a subdirectory within a subject's folder, sorted alphabetically. Each subdirectory contains one daily recording. All available sessions are included without filtering.

ii.
```python
sessions = sorted(p for p in (ROOT / subject).iterdir() if p.is_dir())
```

iii. The agent confirmed that each subdirectory contains suite2p output and motion energy files for one recording session, with session dates encoded in directory names.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as 60-second non-overlapping segments of the continuous recording. At 30 Hz with 10-frame binning, each trial is 180 time bins. Incomplete trailing segments are discarded.

ii.
```python
trial_bins = int(round(TRIAL_SECONDS * out_fs))  # 60 * 3 = 180
ntrials = dff.shape[1] // trial_bins
...
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
```

iii. Per the task instructions, "Split sessions into 60-second trials." The agent used fixed-length segmentation since there is no stimulus-driven trial structure.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are included. The only filtering is that incomplete trailing segments (less than 60 seconds) are discarded.

ii. N/A (no filtering code)

iii. The agent did not mention any trial-level quality filtering in its reasoning. The paper does not describe trial exclusion criteria for decoding analysis.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (Suite2p processing parameters), all from `suite2p/plane0/`.

ii.
```python
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
```

iii. The agent read the paper's methods and the notebook, which states to "compute dF/F the way as described in the paper." It chose F.npy and Fneu.npy over spks.npy (deconvolved spikes), following the paper's stated use of "baseline corrected fluorescence traces as our dF/F."

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`F - 0.7 * Fneu`), followed by a manual reimplementation of Suite2p's maximin baseline correction: Gaussian smoothing (sigma=10 frames), minimum filter (window = 60s * fs), maximum filter (same window). The result is then converted to dF/F by dividing `(F_corrected - baseline) / baseline`.

ii.
```python
def maximin_dff(F, Fneu, ops):
    x = np.asarray(F, dtype=np.float32) - np.float32(ops.get('neucoeff', .7)) * np.asarray(Fneu, dtype=np.float32)
    smooth = gaussian_filter1d(x, float(ops.get('sig_baseline', 10.0)), axis=1, mode='reflect')
    win = int(round(float(ops.get('win_baseline', 60.0)) * float(ops['fs'])))
    base = minimum_filter1d(smooth, size=win, axis=1, mode='reflect')
    base = maximum_filter1d(base, size=win, axis=1, mode='reflect')
    eps = np.finfo(np.float32).eps
    denom = np.where(np.abs(base) > eps, base, np.where(base < 0, -eps, eps))
    x = (x - base) / denom
    return x.astype(np.float32, copy=False)
```

iii. The agent inspected Suite2p's `dcnv.preprocess` source code and the paper's methods to understand the baseline correction algorithm. It reimplemented it manually using scipy rather than calling the suite2p library function. The agent applied dF/F normalization (dividing by baseline), interpreting the paper's statement about "baseline corrected fluorescence traces as our dF/F."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. All neurons in the supplied data are included. The agent noted that the data already contains only Track2p-curated neurons (detected at Suite2p probability > 0.5 and tracked across all days).

ii. N/A (no filtering code)

iii. The agent verified that all `iscell` values in the data are above 0.5, confirming the data is already curated. It explicitly chose not to apply additional filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. The temporal alignment event is described as "start of each consecutive 60-second recording block."

ii.
```python
'temporal_alignment_event': 'start of each consecutive 60-second recording block',
'off_start': 0.0, 'off_end': 60.0,
```

iii. There is no stimulus event to align to. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
AVG_FRAMES = 10
...
usable = (F.shape[1] // AVG_FRAMES) * AVG_FRAMES
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)
...
'time_bin_size': 1000.0 * AVG_FRAMES / 30.0,  # 333.33 ms
```

iii. The paper's Methods state: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The agent followed this directly.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the time bin index and the bin duration (10 frames / 30 Hz = 1/3 second per bin), giving seconds from the start of the session.

ii.
```python
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
...
ins.append(elapsed[None, a:b].copy())
```

iii. Since the frame rate is constant at 30 Hz and no raw timestamp data is needed, computing time from bin indices is straightforward and equivalent.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The elapsed time is computed as `bin_index / output_sampling_rate`, where the output sampling rate is 3 Hz (30 Hz / 10 frames). This gives a continuous time vector across the entire session, from which each trial's portion is extracted.

ii.
```python
out_fs = fs / AVG_FRAMES  # 3.0 Hz
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
```

iii. No complex processing needed; simple index-to-time conversion.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is computed from the same bin indices as the neural data, so they are inherently aligned. Each trial's time slice corresponds exactly to the same bin indices used for neural and output data.

ii.
```python
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
    ns.append(np.ascontiguousarray(dff[:, a:b], ...))
    ins.append(elapsed[None, a:b].copy())
    outs.append(labels[None, a:b].copy())
```

iii. All three data streams (neural, input, output) use the same indexing into the binned time axis, ensuring perfect alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory of each session. Timestamps from `tstamps.npy` are used to detect and interpolate dropped camera frames.

ii.
```python
motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(md / 'tstamps.npy').astype(np.float64)
```

iii. The motion energy file contains a pre-computed global motion energy signal from the behavioral video. The agent chose `tstamps.npy` (rather than `interframe_int.npy`) for gap detection, using timestamp-based slot computation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) dropped camera frames are detected via timestamp gaps and interpolated using `np.interp` to match the neural frame count, (2) the trace is averaged into 10-frame bins along with neural data, (3) the binned signal is discretized into 5 equal-percentile bins computed per session.

ii.
```python
# Interpolation of dropped frames
dt = np.median(np.diff(stamps))
slots = np.rint((stamps - stamps[0]) / dt).astype(np.int64)
...
return np.interp(np.arange(nframes), unique, motion[idx])

# Binning
motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)

# Discretization
labels = quintile_labels(motion)
```

iii. The agent examined the data README which describes missing camera frames and suggests interpolation. It used timestamp-based slot computation for more robust gap detection.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile (quintile) bins per session. The bin edges are at the 20th, 40th, 60th, and 80th percentiles, computed independently for each session. Labels are 0-4.

ii.
```python
def quintile_labels(x):
    edges = np.quantile(x, [.2, .4, .6, .8])
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The task specifies "discretized into five equal-percentile bins, selected per session." The agent verified that the output produces balanced bins (equal counts per level within each session).

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The video and neural data are acquired synchronously at 30 Hz. Dropped camera frames cause the motion energy array to be shorter than the neural data. The agent detects gaps using timestamp spacing (median inter-frame interval) and uses `np.interp` to linearly interpolate the motion energy onto the full imaging frame index. After interpolation, both streams have the same length and are indexed identically.

ii.
```python
def aligned_motion(session, nframes):
    ...
    dt = np.median(np.diff(stamps))
    slots = np.rint((stamps - stamps[0]) / dt).astype(np.int64)
    if slots[-1] not in (nframes - 1, nframes):
        slots = np.rint((stamps - stamps[0]) * (nframes - 1) / (stamps[-1] - stamps[0])).astype(np.int64)
    slots = np.clip(slots, 0, nframes - 1)
    unique, idx = np.unique(slots, return_index=True)
    return np.interp(np.arange(nframes), unique, motion[idx])
```

iii. The agent examined the data README which describes missing camera frames and suggests interpolation. It chose a timestamp-based approach rather than using `interframe_int.npy`, computing expected frame slots from timestamp spacing.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are detected via timestamp gaps and interpolated (see 4-d). A ValueError is raised if motion energy and timestamp lengths don't match. Frames that are dropped during binning (not filling a complete 10-frame bin) are silently discarded. Trailing data that doesn't fill a complete 60-second trial is discarded. The agent also validates that `fs == 30` and `F.shape == Fneu.shape`.

ii.
```python
if len(motion) != len(stamps):
    raise ValueError(f'motion/timestamp mismatch in {session}')
if fs != 30 or F.shape != Fneu.shape:
    raise ValueError(f'unexpected imaging data in {session}: {F.shape}, fs={fs}')
```

iii. The agent performs basic integrity checks but does not attempt to handle more complex data issues. Missing frame interpolation ensures the motion energy signal matches the neural data length.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the `maximin_dff` baseline correction, which applies Gaussian smoothing, minimum filter, and maximum filter over large windows (1800 frames) for every neuron. The agent reimplemented this in scipy rather than using suite2p's GPU-accelerated version, making it potentially slower on GPU-equipped systems but functional on CPU-only systems.

ii. N/A

iii. The baseline correction involves sliding window operations over the full session length for every neuron. The scipy implementation is CPU-only.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial assembly loop iterates over trials to slice and append arrays, but this is unavoidable given the list-of-arrays output structure. The `aligned_motion` function uses `np.interp` which is already vectorized. No major vectorization opportunities remain.

ii. N/A

iii. The code is generally well-vectorized, using numpy reshape/mean for binning and vectorized operations for baseline correction.

## 6-c. What processing does the code repeat multiple times?

i. The motion energy file for each session is loaded twice: once in the main loop during `aligned_motion`, and once again in the `session_info` metadata construction to get the original frame count.

ii.
```python
# In aligned_motion:
motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
# In session_info construction:
'motion_frames': int(np.load(session/'move_deve'/'motion_energy_glob.npy', mmap_mode='r').shape[0]),
```

iii. The second load is only for metadata and uses mmap_mode='r', so the performance impact is minimal.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed `session_info` metadata (source frames, motion frames, n_neurons, n_trials, suite2p_fs_hz) which is not used by the decoder but provides useful provenance information. The `motion_processing` and `neural_measure` metadata fields are also informational only.

ii.
```python
session_info.append({
    'subject': subject, 'session': session.name,
    'source_frames': int(F.shape[1]), 'motion_frames': int(np.load(...).shape[0]),
    'n_neurons': int(F.shape[0]), 'n_trials': ntrials,
    'suite2p_fs_hz': fs
})
```

iii. This metadata adds minimal overhead and is good practice for data provenance, even though it's not used during decoding.
