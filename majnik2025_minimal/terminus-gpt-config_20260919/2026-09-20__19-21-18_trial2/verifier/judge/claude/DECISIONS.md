# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are identified as directories starting with `jm` in the data root. Sessions are sorted subdirectories within each subject folder. For each session, calcium data is loaded from `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy`, ops from `suite2p/plane0/ops.npy`, and motion energy from `move_deve/motion_energy_glob.npy` plus timestamps from `move_deve/tstamps.npy`.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
...
sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
...
F = np.load(p / 'F.npy').astype(np.float32, copy=False)
Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(p / 'tstamps.npy').astype(np.float64)
```

iii. The agent inspected the data directory structure, the data README, and the loading notebook to confirm the naming conventions. It identified all six subjects and 41 total sessions following the standard directory layout.

## 1-b. How are the data split into subjects?

i. Subjects correspond to directories starting with `jm` in the data root, sorted alphabetically.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
```

iii. The agent followed the README's description that each `jm*` directory represents one mouse.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a sorted subdirectory within a subject's folder. Each subdirectory contains one daily recording.

ii.
```python
sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
```

iii. The agent confirmed from the README and data inspection that each subdirectory contains one day's recording with suite2p outputs and behavioral data.

## 1-d. How are the data split into trials?

i. There is no natural trial structure. Trials are defined as consecutive, non-overlapping 60-second segments of the continuous recording (60s * 3 Hz = 180 bins per trial). Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
TRIAL_SECONDS = 60
TRIAL_SAMPLES = int(TRIAL_SECONDS * BIN_FS)  # 180
...
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
neural.append([... for x in np.split(activity, ntrials, axis=1)])
```

iii. The agent followed the instruction to "split sessions into 60-second trials" and used the binned frame rate (3 Hz) to compute 180 samples per trial.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second trials are included. Only the remainder at the end of a session (less than 60 seconds) is discarded.

ii. N/A

iii. The agent found no instructions or paper methods requiring trial-level filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence), `Fneu.npy` (neuropil fluorescence), and `ops.npy` (suite2p options containing processing parameters), all from `suite2p/plane0/`.

ii.
```python
F = np.load(p / 'F.npy').astype(np.float32, copy=False)
Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
```

iii. The agent identified from the paper methods and notebook that raw fluorescence traces require neuropil correction and baseline processing as described in the paper.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (F - 0.7 * Fneu), then a maximin baseline correction is performed: Gaussian smoothing (sigma=10 frames), followed by rolling minimum filter (window = 60s * 30Hz = 1800 frames), then rolling maximum filter (same window), and the resulting baseline is subtracted. The agent reimplemented this using scipy.ndimage functions rather than calling suite2p's `dcnv.preprocess`.

ii.
```python
neucoeff = float(ops.get('neucoeff', 0.7))
corrected = F - neucoeff * Fneu
smooth = gaussian_filter1d(corrected, float(ops.get('sig_baseline', 10.0)),
                           axis=1, mode='reflect')
window = int(float(ops.get('win_baseline', 60.0)) * float(ops.get('fs', FS)))
base = minimum_filter1d(smooth, size=window, axis=1, mode='reflect')
base = maximum_filter1d(base, size=window, axis=1, mode='reflect')
corrected -= base
```

iii. The agent read the Suite2p `dcnv.preprocess` and `baseline_maximin` source code to confirm the processing steps, then reimplemented them using scipy to avoid the slow suite2p import. The parameters are read from each session's ops.npy file.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons in the suite2p output are included. The agent noted that Track2p exports already contain only cells tracked across all days and their iscell scores exceed the paper's default 0.5 threshold.

ii. N/A

iii. The agent verified from the data README and iscell.npy values that all provided neurons already pass the paper's quality threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, no event-based alignment is needed. Elapsed time from session start is preserved across trials.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
...
'temporal_alignment_event': 'session start; consecutive 60-second trial segmentation',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
```

iii. The agent recognized that with no stimulus events, alignment to session start is the natural choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing from 30 Hz to 3 Hz (333.33 ms time bins). This matches the paper's decoding analysis.

ii.
```python
AVERAGE_FRAMES = 10
BIN_FS = FS / AVERAGE_FRAMES  # 3.0 Hz

def average_ten(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1] // AVERAGE_FRAMES
    x = x[..., :n * AVERAGE_FRAMES]
    return x.reshape(*x.shape[:-1], n, AVERAGE_FRAMES).mean(axis=-1)
...
activity = average_ten(activity).astype(np.float32)
motion = average_ten(motion)
```

iii. The agent found the paper's methods section stating "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index and the binned sampling rate (3 Hz), giving seconds from the start of the session.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
```

iii. Since the imaging frame rate is constant at 30 Hz and no raw timestamp variable is needed, computing time from bin indices is equivalent to using actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The elapsed time is computed as `bin_index / BIN_FS` where BIN_FS = 3.0 Hz. This gives time in seconds from session start. The time is continuous across trials within a session (not reset per trial).

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
```

iii. The agent chose to keep time relative to session start rather than resetting per trial, as stated in the code's docstring.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The elapsed time array is computed to have the same number of bins as the neural data after 10-frame averaging, so they are inherently aligned by construction.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
```

iii. N/A - alignment is trivial since both share the same time axis.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` and `tstamps.npy` in the `move_deve` subdirectory of each session.

ii.
```python
motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(p / 'tstamps.npy').astype(np.float64)
```

iii. The agent identified these files from the data README and directory inspection.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps: (1) motion energy is interpolated onto the regular 30 Hz imaging grid using camera timestamps, (2) the interpolated trace is averaged in 10-frame bins (matching neural data), (3) the binned signal is discretized into 5 quintiles using per-session percentile boundaries.

ii.
```python
# Interpolation
positive_dt = np.diff(stamps)
step = np.median(positive_dt[positive_dt > 0])
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])

# Averaging
motion = average_ten(motion)

# Discretization
def quintiles(x: np.ndarray) -> np.ndarray:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The agent noted that camera frames occasionally dropped, making timestamp-based interpolation the appropriate alignment method.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) per session. Four boundaries at the 20th, 40th, 60th, and 80th percentiles are computed within each session, and `np.searchsorted` assigns values to classes 0-4.

ii.
```python
def quintiles(x: np.ndarray) -> np.ndarray:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The instruction specifies "five equal-percentile bins, selected per session." The agent computed quintile boundaries per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera timestamps identify the exact position of each motion energy sample including dropped frames. The agent uses `np.interp` to interpolate the motion energy signal onto a regular 30 Hz grid matching the imaging frames. The median timestamp interval defines the expected frame step, and a target grid of `nframes` points (matching neural data length) is constructed.

ii.
```python
positive_dt = np.diff(stamps)
step = np.median(positive_dt[positive_dt > 0])
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
```

iii. The agent analyzed timestamp gap patterns across sessions and determined that timestamp-based interpolation is more robust than the interframe interval approach, correctly handling dropped frames regardless of their number or distribution.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped camera frames are handled by interpolating the motion energy signal onto the regular imaging grid using timestamps. If the motion energy array is shorter than the neural data, the interpolation naturally fills the gaps. If timestamps indicate dropped frames even when arrays are the same length (as in some jm046 sessions), the interpolation still correctly handles the irregular sampling.

ii.
```python
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
```

iii. The agent inspected all sessions and found various cases: some with fewer motion samples than neural frames, some with equal counts but timestamp gaps. The interpolation approach handles all cases uniformly.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the maximin baseline correction, which involves Gaussian smoothing, minimum filtering, and maximum filtering over a 1800-frame window for every neuron across the full session length. The agent's scipy-based implementation runs on CPU, which is slower than Suite2p's GPU-accelerated version.

ii. N/A

iii. The baseline correction involves three passes of large-window filters over the full neuron x time matrix.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code is largely vectorized. The per-session loop is necessary for loading different data files. Within each session, operations are vectorized using numpy/scipy array operations. The `quintiles` function is applied per-session as required by the task specification.

ii. N/A

iii. N/A

## 6-c. What processing does the code repeat multiple times?

i. The code does not repeat any significant processing. Each step is performed once in a single pass through all sessions.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed `session_info` metadata for each session (subject name, session name, neuron count, trial count, duration, imaging rate), which is informational rather than functionally necessary for the decoder.

ii.
```python
session_info.append({
    'subject': subject, 'session': session.name,
    'n_neurons': int(activity.shape[0]), 'n_trials': ntrials,
    'duration_seconds': float(usable / BIN_FS),
    'source_imaging_rate_hz': FS,
})
```

iii. This extra metadata is good practice for documentation but not used by the decoder.
