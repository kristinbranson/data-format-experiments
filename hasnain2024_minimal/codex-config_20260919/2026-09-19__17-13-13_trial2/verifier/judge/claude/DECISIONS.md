# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder, matching the Figure 8 analysis pipeline. Sessions are hard-coded in a `SESSIONS` tuple with animal name, date, and probe number. Each session's main data file (`data_structure_*.mat`) is opened with `h5py` (v7.3 HDF5 format), while motion energy files (`motionEnergy_*.mat`) are read with `scipy.io.loadmat`. The AI does NOT use the `RandomizedDelay_Ephys_Behavior` folder.

ii.
```python
SESSIONS = (
    ("JEB6", "2021-04-18", 2),
    ("JEB7", "2021-04-29", 1),
    ("JEB7", "2021-04-30", 1),
    ("EKH1", "2021-08-07", 2),
    ("EKH3", "2021-08-11", 2),
    ("JGR2", "2021-11-16", 1),
    ("JGR2", "2021-11-17", 1),
    ("JGR3", "2021-11-18", 1),
    ("JEB19", "2023-04-21", 1),
    ("JEB19", "2023-04-20", 1),
    ("JEB19", "2023-04-19", 1),
    ("JEB19", "2023-04-18", 1),
)
```

```python
with h5py.File(data_path, "r") as handle:
    behavior = _load_behavior(handle)
    neural_all, qualities = _load_neural(handle, probe, behavior, edges)
```

iii. The AI reasoned that only the `Ephys_Behavior` sessions contain the two-context (DR + WC) paradigm needed for the "behavioral context" decoder output, and that the randomized-delay sessions are DR-only. It matched the Figure 8 pipeline sessions exactly.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the first element of each session tuple (animal name). The `subjects` list preserves insertion order using `dict.fromkeys`.

ii.
```python
subjects = list(dict.fromkeys(session[0] for session in SESSIONS))
```

```python
"subject_idx": np.asarray(
    [subjects.index(animal) for animal, _, _ in SESSIONS], dtype=np.int64
),
```

iii. The animal name is explicit in the session definition tuple. This is straightforward.

## 1-c. How are the data split into sessions?

i. One session is one entry in the `SESSIONS` tuple, corresponding to one `.mat` file on disk. Each session becomes one element in the `neural`, `input`, and `output` lists. Only the `Ephys_Behavior` folder is searched, resulting in 12 sessions total.

ii.
```python
for session_index, (animal, date, probe) in enumerate(SESSIONS, start=1):
    stem = f"{animal}_{date}"
    data_path = data_dir / f"data_structure_{stem}.mat"
```

iii. The AI chose sessions from the Figure 8 pipeline scripts in the reference code.

## 1-d. How are the data split into trials?

i. Trials are defined by the behavior structure in the MATLAB file. The number of trials is determined by the length of the `go_cue` array. Each trial has associated behavioral flags (hit, miss, no, early, autowater, stim). After filtering, remaining trial indices are used to slice neural and behavioral data.

ii.
```python
result["go_cue"] = _array(bp["ev/goCue"]).astype(np.float64)
result["ntrials"] = np.asarray([len(result["go_cue"])], dtype=np.int64)
...
trial_ids = np.flatnonzero(keep)
for trial in trial_ids:
    neural_trials.append(neural_all[trial].copy())
```

iii. The Bpod table defines trials directly, one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: early-lick trials (`early`) and photostimulation trials (`stim`) are excluded. Ignore trials are deliberately retained. There is no additional filter for trials that extend past the recording end.

ii.
```python
keep = ~behavior["early"] & ~behavior["stim"]
...
trial_ids = np.flatnonzero(keep)
```

iii. The AI followed the paper's exclusion of early-lick and photostim trials. Ignore trials are retained because the task explicitly requires an "ignore" outcome class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike-sorted clusters from `obj.clu{probe}`, specifically the `trial`, `trialtm`, and `quality` fields. The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
probe_group = _deref(handle, handle["obj/clu"], probe - 1)
spike_trial = _array(_deref(handle, probe_group["trial"], unit)).astype(np.int64)
spike_time = _array(_deref(handle, probe_group["trialtm"], unit)).astype(np.float64)
aligned = spike_time - go_cue[spike_trial - 1]
```

iii. These are the standard spike-sorted cluster fields from the data structure.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 10 ms bins spanning -3.0 to 2.5 s from the go cue (550 bins). Counts are divided by the bin width (0.01 s) to get firing rates in Hz. A **causal** 15-bin Gaussian kernel is applied for smoothing, matching `mySmooth.m`. The first half of the Gaussian is zeroed out to make it causal. Boundary handling prepends a copy of the first N bins before convolution.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 0.01
SMOOTH_BINS = 15

counts = np.zeros((ntrials, len(edges) - 1), dtype=np.float64)
np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
rates = _causal_smooth(counts / DT)
```

```python
def _causal_smooth(rates: np.ndarray) -> np.ndarray:
    kernel = gaussian(SMOOTH_BINS, std=SMOOTH_BINS / 5.0)
    kernel[: SMOOTH_BINS // 2] = 0.0  # causal operation in mySmooth.m
    kernel /= kernel.sum()
    padded = np.concatenate((rates[:, :SMOOTH_BINS], rates), axis=1)
    smoothed = signal.convolve(padded, kernel[None, :], mode="same")
    return smoothed[:, SMOOTH_BINS:]
```

iii. The AI read `mySmooth.m` from the code repository and replicated its causal Gaussian smoothing exactly, including the boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels `garbage`, `gabrga`, `noisy`, or `real?` are excluded (case-sensitive matching). Second, units whose condition-averaged mean firing rate is at or below 1 Hz are dropped. The condition averaging follows `removeLowFRClusters.m`, which averages condition PSTHs equally (not weighted by trial count).

ii.
```python
exclusions = {"garbage", "gabrga", "noisy", "real?"}
...
if quality in exclusions:
    continue
...
condition_means = [rates[mask].mean(axis=0) for mask in conditions if np.any(mask)]
mean_rate = float(np.mean(np.stack(condition_means)))
if mean_rate > LOW_FR_HZ:
    neurons.append(rates.astype(np.float32))
```

iii. The AI matched `findClusters.m` for quality exclusion and `removeLowFRClusters.m` for the firing rate threshold, reproducing the paper's 522 ALM units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial-1]` from each spike time. The aligned spikes are then binned into the time grid.

ii.
```python
aligned = spike_time - go_cue[spike_trial - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. This is the standard alignment used by the reference code's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (100 Hz), spanning -3.0 to 2.5 s from the go cue, yielding 550 bins per trial. No rebinning is applied -- spikes are counted directly into these 10 ms bins.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 0.01
edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
```

iii. The AI matched `params.dt = 1/100` from `getDefaultParams.m` and `params.tmin = -3` / `params.tmax = 2.5` from the Figure 8 scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is not derived from raw data variables. It is the bin center times of the neural data grid, computed from the bin edges.

ii.
```python
time = (edges[:-1] + DT / 2).astype(np.float32)
```

iii. The time axis is defined by the binning parameters and represents each bin's center relative to the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The bin center times are computed as `edges[:-1] + DT/2`. No further processing is needed since this is a synthetic variable.

ii.
```python
time = (edges[:-1] + DT / 2).astype(np.float32)
```

iii. N/A -- this is a defined grid, not processed from raw data.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the bin center times of the same grid used for the neural data, so alignment is inherent.

ii.
```python
input_trials.append(time[None, :].copy())
```

iii. N/A -- the input IS the neural time grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no`. Lick direction is inferred from the combination of instructed side and outcome.

ii.
```python
lick = np.zeros(len(keep), dtype=np.int8)  # left
lick[(behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])] = 1
lick[behavior["no"]] = 2
```

iii. The AI followed `getPrevChoice.m` which defines right choice as R-hit or L-miss.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Right lick (1) = R-hit or L-miss. Left lick (0) = all other response trials (L-hit or R-miss). No lick (2) = ignore trials (`no`). This is a per-trial value broadcast across all time bins.

ii.
```python
lick = np.zeros(len(keep), dtype=np.int8)  # left
lick[(behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])] = 1
lick[behavior["no"]] = 2
```

iii. Follows the `getPrevChoice.m` logic from the repository.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater trials are WC context.

ii.
```python
context = behavior["autowater"].astype(np.int8)  # 0 DR, 1 WC
```

iii. This field directly indicates the behavioral context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct cast of the boolean autowater field to int8: DR = 0 (not autowater), WC = 1 (autowater).

ii.
```python
context = behavior["autowater"].astype(np.int8)  # 0 DR, 1 WC
```

iii. Note: The AI codes DR as 0 and WC as 1, which is the reverse of the reference (WC=0, DR=1). The instructions say "WC, DR" which could be interpreted either way.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
outcome = np.zeros(len(keep), dtype=np.int8)  # incorrect
outcome[behavior["hit"]] = 1
outcome[behavior["no"]] = 2
```

iii. Three mutually exclusive outcome flags.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Incorrect (0) = miss, correct (1) = hit, ignore (2) = no response. Per-trial value broadcast across time bins.

ii.
```python
outcome = np.zeros(len(keep), dtype=np.int8)  # incorrect
outcome[behavior["hit"]] = 1
outcome[behavior["no"]] = 2
```

iii. Direct mapping of behavioral flags to the three outcome categories specified in the instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking from `obj.traj`, specifically the side camera (camera 0), feature `"tongue"`. The x, y coordinates and their timestamps (`frameTimes`) are used. The video offset is computed from `obj.sglx` bitcode fields.

ii.
```python
tongue_speed, tongue_visible = _load_velocity(
    handle, behavior, time, camera=0, feature="tongue"
)
```

iii. The AI uses only the side camera for tongue, unlike the reference which uses both side and bottom cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. DLC x, y coordinates are interpolated from camera frame times onto the neural time grid using linear interpolation. For tongue, NaN positions are NOT nearest-filled (gaps remain NaN). Velocity is computed as `np.gradient` of x and y, and speed is the Euclidean norm. No drift subtraction is applied for tongue. The speed is discretized at the per-session 50th percentile.

ii.
```python
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
ypos = _interp_trace(aligned_frames, tracking[:, 1, feature_index], target_time)
visible[trial] = np.isfinite(xpos) & np.isfinite(ypos)

if "tongue" not in feature:
    xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)
xvel, yvel = np.gradient(xpos), np.gradient(ypos)
speed[trial] = np.hypot(xvel, yvel)
```

iii. The AI followed `findVelocity.m` which does not apply nearest-fill or drift subtraction for tongue features.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold on visible, non-excluded trial samples. Below threshold = 0, at or above threshold = 1, not visible = 2.

ii.
```python
tongue_class, tongue_threshold = _median_discretize(tongue_speed, tongue_visible, keep)
```

```python
def _median_discretize(values, available, keep):
    usable = keep[:, None] & available & np.isfinite(values)
    threshold = float(np.percentile(values[usable], 50))
    output = np.full(values.shape, 2, dtype=np.int8)
    output[usable & (values < threshold)] = 0
    output[usable & (values >= threshold)] = 1
    return output, threshold
```

iii. Matches the instruction's 50th percentile threshold specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are corrected by the video offset (computed from bitcode synchronization) and the trial's go cue time. DLC coordinates are then linearly interpolated onto the neural time grid.

ii.
```python
aligned_frames = frames - offset - behavior["go_cue"][trial]
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
```

iii. The video offset is computed similarly to `findVideoOffset.m`, using median instead of mode.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from `obj.traj`, bottom camera (camera 1), feature `"top_paw"`. Same x, y coordinates and frame times as tongue.

ii.
```python
paw_speed, paw_visible = _load_velocity(
    handle, behavior, time, camera=1, feature="top_paw"
)
```

iii. The AI uses `top_paw` from the bottom view, matching the reference's choice.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. DLC x, y coordinates are interpolated onto the neural time grid. For paw (non-tongue), nearest-fill is applied before differentiation, and baseline drift subtraction is applied after differentiation. Speed is the Euclidean norm of the corrected velocity components.

ii.
```python
if "tongue" not in feature:
    xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)
xvel, yvel = np.gradient(xpos), np.gradient(ypos)
if "tongue" not in feature:
    if np.any(np.isfinite(xvel)):
        xvel -= np.nanmedian(np.diff(xpos))
    if np.any(np.isfinite(yvel)):
        yvel -= np.nanmedian(np.diff(ypos))
speed[trial] = np.hypot(xvel, yvel)
```

iii. The AI replicated `findPosition.m` (nearest-fill) and `findVelocity.m` (drift subtraction) for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile, below = 0, at or above = 1, not visible = 2.

ii.
```python
paw_class, paw_threshold = _median_discretize(paw_speed, paw_visible, keep)
```

iii. Same discretization as tongue velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: video offset correction, go cue subtraction, linear interpolation onto neural time grid.

ii.
```python
aligned_frames = frames - offset - behavior["go_cue"][trial]
```

iii. Same alignment pipeline as all video-derived features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file `motionEnergy_*.mat` read with `scipy.io.loadmat`. The data is in `me['data']` as per-trial arrays.

ii.
```python
mat = io.loadmat(path, simplify_cells=True, variable_names=["me"])
raw = mat["me"]["data"]
raw_trials = list(raw) if isinstance(raw, np.ndarray) and raw.dtype == object else [raw]
```

iii. The AI handled the nested structure using `simplify_cells=True`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are linearly interpolated from camera frame times onto the neural time grid. No additional smoothing or processing is applied beyond the interpolation and discretization.

ii.
```python
values[trial] = _interp_trace(aligned_frames, trace, target_time)
```

iii. Motion energy is already a scalar per frame, so only alignment and interpolation are needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw: per-session 50th percentile, below = 0, at or above = 1, no video = 2.

ii.
```python
motion_class, motion_threshold = _median_discretize(motion_energy, motion_available, keep)
```

iii. Same discretization scheme as the other movement variables.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side camera frame times are corrected by the video offset and go cue, then motion energy values are linearly interpolated onto the neural time grid.

ii.
```python
aligned_frames = frames - offset - behavior["go_cue"][trial]
...
values[trial] = _interp_trace(aligned_frames, trace, target_time)
```

iii. Same clock correction as all video-derived streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with non-finite `NdroppedFrames` are skipped entirely for video features, leaving NaN in speed/motion energy arrays. (2) DLC coordinates that are NaN (below likelihood threshold) produce NaN velocity, which becomes "not visible" class 2. (3) For paw, nearest-fill is applied to interpolate over missing positions. (4) For tongue, gaps remain NaN. (5) Velocity traces with entirely NaN values have guards for `np.nanmedian`.

ii.
```python
dropped = _array(_deref(handle, group["NdroppedFrames"], trial))
if np.size(dropped) and not np.all(np.isfinite(dropped)):
    continue
```

iii. The AI distinguishes between tongue (no fill) and paw (nearest-fill) following the reference code's conventions.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files with `h5py` and computing neural firing rates (spike counting and smoothing for each unit) are the most time-consuming steps. The per-trial velocity computation with interpolation is also significant.

ii.
```python
with h5py.File(data_path, "r") as handle:
    behavior = _load_behavior(handle)
    neural_all, qualities = _load_neural(handle, probe, behavior, edges)
```

iii. File I/O and the per-unit spike binning loop dominate runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `_load_neural` iterates over each unit to compute spike counts using `np.add.at`, which could potentially be vectorized with `histogram2d` across all trials at once. The per-trial velocity loops in `_load_velocity` and `_load_motion_energy` iterate over each trial individually.

ii.
```python
for unit in range(probe_group["quality"].shape[0]):
    ...
    np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
```

```python
for trial in range(ntrials):
    ...
    speed[trial] = np.hypot(xvel, yvel)
```

iii. The per-trial loops exist because each trial's video data has different frame counts. The per-unit loop could potentially be restructured but the `np.add.at` approach is already efficient per unit.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is recomputed for each call to `_load_velocity` and `_load_motion_energy` (3 times per session total -- once for tongue, once for paw, once for motion energy). Similarly, `NdroppedFrames` and `frameTimes` are loaded repeatedly for each feature/trial combination.

ii.
```python
offset = _video_offset(handle, behavior)  # called in _load_velocity and _load_motion_energy
```

iii. The video offset is a session constant and could be computed once and passed in.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `_figure8_conditions` which generates 7 condition masks used only for the firing rate threshold computation. These condition masks include detailed trial-type breakdowns (hit/miss x WC/DR x early) that are not used elsewhere. The `dropped` frames check (`NdroppedFrames`) is done per-trial per-feature, reading and dereferencing the HDF5 dataset repeatedly.

ii.
```python
def _figure8_conditions(behavior):
    hit, miss, no = behavior["hit"], behavior["miss"], behavior["no"]
    stim, wc, early = behavior["stim"], behavior["autowater"], behavior["early"]
    return [
        hit | miss | no,
        hit & ~stim & ~wc,
        hit & ~stim & wc,
        miss & ~stim & ~wc,
        miss & ~stim & wc,
        hit & ~stim & ~wc & ~early,
        hit & ~stim & wc & ~early,
    ]
```

iii. The condition averaging replicates the exact logic of `removeLowFRClusters.m` but is more complex than a simple mean rate check would be.
