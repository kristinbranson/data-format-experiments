# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat` in one of two folders (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`), with a companion `motionEnergy_<anm>_<date>.mat`. The 44 sessions and their probe assignments are hard-coded in two dictionaries (`ALM_PROBES` for 25 fixed-delay and `RANDOMIZED_DELAY_PROBES` for 19 randomized-delay), transcribed from the authors' `load<ANM>_ALMVideo.m` scripts. HDF5 (v7.3) files are read with `h5py`; legacy (v5) files are read with `scipy.io.loadmat`. Two completely separate code paths exist for each format.

ii.
```python
ALM_PROBES = {
    "EKH1_2021-08-07": (2,),
    ...
}
RANDOMIZED_DELAY_PROBES = {
    "JEB11_2022-05-10": (1,),
    ...
}
SESSION_GROUPS = (
    ("Ephys_Behavior", "fixed_delay", ALM_PROBES),
    ("RandomizedDelay_Ephys_Behavior", "randomized_delay", RANDOMIZED_DELAY_PROBES),
)
# loading:
if h5py.is_hdf5(data_path):
    with h5py.File(data_path, "r") as matfile:
        ...
else:
    obj = loadmat(data_path, squeeze_me=True, struct_as_record=False, ...)["obj"]
```

iii. The agent noted (step 8): "I'm narrowing the conversion to those recording sessions" and (step 26): "the paper's 19 randomized-delay ALM sessions. The repository explicitly selects 19 of the 22 files (three are commented out and lack matching motion-energy data)."

## 1-b. How are the data split into subjects?

i. The subject is extracted from the session key as the part before the underscore (e.g., `JEB19` from `JEB19_2023-04-19`). Unique subjects are collected preserving insertion order via `dict.fromkeys`.

ii.
```python
subject, date = key.split("_", 1)
...
subjects = list(dict.fromkeys(session_subjects))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent followed the same convention as the authors' file naming.

## 1-c. How are the data split into sessions?

i. One session is one entry in `ALM_PROBES` or `RANDOMIZED_DELAY_PROBES`, corresponding to one file on disk. Both fixed-delay and randomized-delay sessions are processed through the same pipeline. Each becomes one element in `neural`, `input`, and `output`. The result is 44 sessions total.

ii.
```python
for session_idx, (data_path, motion_path, probes, task_variant) in enumerate(sessions, start=1):
    key = _session_key(data_path)
    ...
```

iii. The agent noted (step 26): "I'm including those 19 alongside the 25 fixed-delay sessions."

## 1-d. How are the data split into trials?

i. Each trial is one entry in the Bpod table (`obj.bp`). The number of trials is read from `bp.Ntrials`. Per-trial fields (hit, miss, early, etc.) are read as vectors of that length. Spike data carries trial indices directly (`clu.trial`), so no trial boundaries need to be reconstructed.

ii.
```python
ntrials = int(_vector(matfile["obj/bp/Ntrials"])[0])
...
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
```

iii. The Bpod table defines trials directly.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) early-lick trials (`bp.early`) are excluded, (2) photostimulation trials (`bp.stim.enable`) are excluded, (3) trials without electrophysiology coverage (`trials.bp.haveEphys`) are excluded, and (4) trials where all retained neural units have zero activity across the full time window are excluded. The `haveEphys` mask is aligned to the Bpod trial count by trimming trailing excess or padding with False.

ii.
```python
early = _vector(matfile["obj/bp/early"]).astype(bool)
stimulated = _vector(matfile["obj/bp/stim/enable"]).astype(bool)
have_ephys = _fit_trial_mask(
    _vector(matfile["obj/trials/bp/haveEphys"]), early.size)
candidate_trials = np.flatnonzero(~early & ~stimulated & have_ephys)
...
neural_present = np.any(neural_all != 0, axis=(1, 2))
selected = candidate_trials[neural_present[candidate_trials]]
```

iii. The agent noted (step 76): "two recordings continued behavioral trials after electrophysiology stopped, producing all-zero neural trials. The source exposes `haveEphys` specifically for this boundary." And (step 104): "the reliable boundary is the neural data: after curation, those trials contain no spikes in any retained unit across the full 5 s window."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters. Each cluster carries `trial` (1-based trial index for each spike) and `trialtm` (spike time relative to trial start). The go cue times `bp.ev.goCue` provide the alignment. The `quality` label determines which clusters to keep.

ii.
```python
spike_trials = _deref_vector(matfile, cluster_group["trial"], cluster_idx).astype(np.int64) - 1
spike_times = _deref_vector(matfile, cluster_group["trialtm"], cluster_idx).astype(np.float64)
go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
aligned = spike_times[valid_trial] - go_cue[spike_trials]
```

iii. The agent examined the MATLAB fields to confirm the spike data structure.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, then assigned to 5 ms bins spanning -2.5 to +2.5 s using floor-based bin assignment. Spike counts are accumulated via `np.add.at`. The counts are then smoothed with a **causal** Gaussian kernel of length 15 bins (matching `mySmooth.m` which zeros the first `floor(N/2)` weights) and divided by the bin width (DT) to convert to firing rates (Hz). The kernel is normalized to sum to 1.

ii.
```python
def _gaussian_kernel(length: int) -> np.ndarray:
    n = np.arange(length, dtype=np.float64) - (length - 1.0) / 2.0
    kernel = np.exp(-0.5 * (2.5 * n / (length / 2.0)) ** 2)
    kernel[: length // 2] = 0.0  # causal
    kernel /= kernel.sum()
    return kernel.astype(np.float32)

def _smooth_counts(counts: np.ndarray) -> np.ndarray:
    padded = np.concatenate((counts[:, :SMOOTH_BINS], counts), axis=1)
    smoothed = convolve(padded, SMOOTH_KERNEL[None, :], mode="same", method="direct")
    return smoothed[:, SMOOTH_BINS:] / np.float32(DT)
```

iii. The agent noted (step 12): "spikes aligned to each trial's `goCue`, binned at 5 ms from −2.5 to +2.5 s, converted to Hz, then Gaussian-smoothed with the paper's 15-bin kernel."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Clusters whose `quality` label (lowercased, stripped) matches one of `{"garbage", "gabrga", "noisy", "real?"}` are excluded. (2) Units whose mean firing rate over the window is <= 1 Hz are excluded.

ii.
```python
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
if quality.lower() in REJECTED_QUALITIES:
    continue
...
mean_fr = neural.mean(axis=(0, 2), dtype=np.float64)
keep = mean_fr > LOW_FR_HZ
```

iii. The agent noted (step 12): "The published inclusion rule is all curated units above 1 Hz."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time (`trialtm`) is subtracted by its trial's go cue time (`bp.ev.goCue`), giving spike times in seconds from go cue onset. Spikes outside the [-2.5, 2.5) window are discarded before binning.

ii.
```python
aligned = spike_times[valid_trial] - go_cue[spike_trials]
valid_time = (aligned >= TMIN) & (aligned < TMAX)
spike_trials = spike_trials[valid_time]
aligned = aligned[valid_time]
time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. This follows `alignSpikes.m` from the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spikes are binned into 1000 non-overlapping 5 ms bins spanning -2.5 to +2.5 s from the go cue. This matches the reference's `params.dt = 1/200`. No rebinning is applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT + TMIN + DT / 2.0)
```

iii. The agent confirmed 5 ms bins match the paper.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This variable is constructed from the time bin centers of the 5 ms grid, not from any raw data variable. It is the same 1000-element array for every trial.

ii.
```python
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT + TMIN + DT / 2.0)
...
input_trials = [TIME[None, :].copy() for _ in range(selected.size)]
```

iii. The time from go cue is defined by the binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The input is the bin centers of the time grid, defined from -2.4975 to 2.4975 s in 5 ms steps.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the neural binning grid itself. Spikes are aligned to the go cue and binned into the same edges, so bin k corresponds to the same time interval in both neural and input.

ii.
```python
EDGES = TMIN + np.arange(TIME.size + 1, dtype=np.float64) * DT
```

iii. The same grid is used for all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, and `bp.R` (right target). The `bp.no` (ignore) field is also read but used for confirmation.

ii.
```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
right_target = _vector(matfile["obj/bp/R"]).astype(bool)
```

iii. The agent noted (step 20): "lick direction is the actual choice (target side on correct trials, opposite side on incorrect trials, and `none` on ignores)."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port (right if `R`, left otherwise). A miss means it licked the opposite port. Ignore trials get `none` (code 2). Left=0, right=1, none=2.

ii.
```python
if ignore[source_trial]:
    lick_direction, outcome = 2, 2  # none, ignore
elif hit[source_trial]:
    lick_direction = 1 if right_target[source_trial] else 0
    outcome = 1  # correct
elif miss[source_trial]:
    lick_direction = 0 if right_target[source_trial] else 1
    outcome = 0  # incorrect
```

iii. Follows the same logic as the reference.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`. Autowater trials are the water-cued (WC) context; everything else is delayed-response (DR).

ii.
```python
water_cued = _vector(matfile["obj/bp/autowater"]).astype(bool)
...
context = 0 if water_cued[source_trial] else 1  # WC, DR
```

iii. The agent followed the paper's definition of the two contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabeling: autowater=True maps to WC (0), autowater=False maps to DR (1).

ii. Same as 5-a code.

iii. Matches the prompt's WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, and `bp.no` (ignore).

ii.
```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
```

iii. The three flags are mutually exclusive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabeling: miss=incorrect (0), hit=correct (1), ignore=ignore (2).

ii.
```python
if ignore[source_trial]:
    lick_direction, outcome = 2, 2
elif hit[source_trial]:
    ...
    outcome = 1  # correct
elif miss[source_trial]:
    ...
    outcome = 0  # incorrect
```

iii. Matches the prompt's coding.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically only the **side camera** (index 0) and the `"tongue"` feature. The `ts` array provides x, y coordinates per frame, and `frameTimes` provides the timing. The `NdroppedFrames` field is used to check for valid tracking. `bp.ev.goCue` and `sglx.bitcode.bitstart` / `sglx.fs` / `bp.ev.bitStart` are used for video-to-behavior clock alignment.

ii.
```python
trajectory_side = matfile[np.asarray(matfile["obj/traj"])[0, 0]]
...
tx, ty, tongue_vis = _trajectory_xy(
    matfile, trajectory_side, source_trial, "tongue", TIME,
    video_shift, go_cue[source_trial])
tongue_speed[out_trial] = _speed(tx, ty, tongue=True)
```

iii. The agent used only the side camera view for the tongue.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The tongue x,y positions are **interpolated** onto the 5 ms time grid using linear interpolation (`np.interp`). The speed is computed as `np.hypot(np.gradient(x), np.gradient(y))`, where NaN gradients are replaced with 0. Visibility is determined by whether the interpolated x,y are finite. No Gaussian smoothing of position is applied. The speed is then thresholded at the session's 50th percentile of visible values into two classes, with a third class (2) for not-visible bins.

ii.
```python
def _speed(x, y, tongue):
    xy = np.column_stack((x, y))
    ...
    if tongue:
        xvel = np.gradient(x)
        yvel = np.gradient(y)
        xvel[~np.isfinite(xvel)] = 0.0
        yvel[~np.isfinite(yvel)] = 0.0
    ...
    return np.hypot(xvel, yvel)

# thresholding:
tongue_values = tongue_speed[tongue_visible & np.isfinite(tongue_speed)]
thresholds["tongue_velocity_median"] = float(np.percentile(tongue_values, 50))
output[out_trial, 3, valid] = (
    tongue_speed[out_trial, valid] >= thresholds["tongue_velocity_median"])
```

iii. The agent chose to interpolate directly to the target time grid rather than computing velocity at frame resolution and then binning.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Split at the session's 50th percentile of visible, finite values. Values >= threshold get class 1 (above), < threshold get class 0 (below), and not-visible bins get class 2.

ii. See 7-b code snippets.

iii. Follows the prompt's discretization instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video-to-behavior clock offset is computed as `mode(bitcode.bitstart/fs) - mode(bp.ev.bitStart)`. Frame times are converted to go-cue-relative time by subtracting this offset and the trial's go cue. The tongue positions are then linearly **interpolated** onto the same 5 ms TIME grid used for neural data.

ii.
```python
video_shift = (_mode(matfile["obj/sglx/bitcode/bitstart"]) / fs
               - _mode(matfile["obj/bp/ev/bitStart"]))
...
source_time = frame_times - video_shift - go_cue
x = _interp(source_time, x_raw, aligned_time)
```

iii. Uses `findVideoOffset.m` logic.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking from the **bottom camera** (index 1), using **both** `"top_paw"` and `"bottom_paw"` features. Their speeds are averaged where both are visible.

ii.
```python
trajectory_bottom = matfile[np.asarray(matfile["obj/traj"])[1, 0]]
for feature in ("top_paw", "bottom_paw"):
    px, py, paw_vis = _trajectory_xy(
        matfile, trajectory_bottom, source_trial, feature, TIME,
        video_shift, go_cue[source_trial])
    paw_speeds.append(_speed(px, py, tongue=False))
    paw_masks.append(paw_vis)
paw_stack = np.stack(paw_speeds)
...
paw_speed[out_trial] = np.divide(summed, count, ...)
```

iii. The agent noted: "Both paws were tracked in the bottom view. Average their speed where available; this gives one requested paw-velocity stream without privileging either paw."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw x,y positions are linearly **interpolated** onto the 5 ms grid. Then nearest-fill is applied to the interpolated positions (filling NaN gaps with the nearest finite value). Speed is computed as `np.hypot(np.gradient(x_filled) - baseline, np.gradient(y_filled) - baseline)`, where `baseline` is the median of `np.diff` of the position. The two paws' speeds are averaged where both are visible. The result is thresholded at the session 50th percentile.

ii.
```python
def _speed(x, y, tongue):
    ...
    if not tongue:
        x_filled = _nearest_fill(x)
        y_filled = _nearest_fill(y)
        xvel = np.gradient(x_filled) - baseline[0]
        yvel = np.gradient(y_filled) - baseline[0]
    return np.hypot(xvel, yvel)
```

iii. The agent attempted to match `findVelocity.m`, including its baseline subtraction and nearest-fill behavior.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Split at the session's 50th percentile of visible, finite values. Same 3-class scheme as tongue velocity.

ii.
```python
thresholds["paw_velocity_median"] = float(np.percentile(paw_values, 50))
output[out_trial, 4, valid] = (
    paw_speed[out_trial, valid] >= thresholds["paw_velocity_median"])
```

iii. Follows the prompt.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same video offset correction as the tongue. Positions are linearly interpolated onto the 5 ms TIME grid.

ii. Same as tongue alignment (7-d).

iii. Same offset and grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The separate `motionEnergy_<anm>_<date>.mat` file. It holds one trace per trial with one value per camera frame. The double-struct wrapping is handled (matching `loadMotionEnergy.m`).

ii.
```python
def _motion_trials(motion_path, ntrials):
    me_obj = loadmat(motion_path, squeeze_me=True, struct_as_record=False,
                     variable_names=["me"])["me"]
    motion_data = me_obj.data
    if hasattr(motion_data, "data"):
        motion_data = motion_data.data
    trials = np.asarray(motion_data, dtype=object).reshape(-1)
```

iii. The agent noted (step 38): "its motion-energy field has an extra MATLAB struct wrapper, which the authors' loader explicitly unwraps."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-frame motion energy values are linearly **interpolated** onto the 5 ms time grid. NaN edges are filled with nearest-value fill (matching `loadMotionEnergy.m`). Then thresholded at the session 50th percentile.

ii.
```python
interp_motion = _interp(source_time, raw_motion, TIME)
if np.isfinite(interp_motion).any():
    interp_motion = _nearest_fill(interp_motion)
    motion[out_trial] = interp_motion
    motion_video[out_trial] = True
```

iii. The agent used interpolation and nearest-fill to match the MATLAB loader.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Split at the session 50th percentile of values where video is available. Below threshold=0, at/above=1, no video=2.

ii.
```python
motion_values = motion[motion_video & np.isfinite(motion)]
thresholds["motion_energy_median"] = float(np.percentile(motion_values, 50))
output[out_trial, 5, valid] = (
    motion[out_trial, valid] >= thresholds["motion_energy_median"])
```

iii. Follows the prompt.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The side camera's frame times are corrected by the video offset and go cue, then the motion energy values are linearly interpolated onto the 5 ms TIME grid.

ii.
```python
frame_obj = matfile[np.asarray(trajectory_side["frameTimes"])[source_trial, 0]]
frame_times = np.asarray(frame_obj).reshape(-1, order="F")
source_time = frame_times - video_shift - go_cue[source_trial]
interp_motion = _interp(source_time, raw_motion, TIME)
```

iii. Same alignment as other video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with `NdroppedFrames` empty or NaN result in all-NaN positions, which become "not visible" class. (2) `haveVid` flag is checked; trials without video get all class-2 outputs for movement variables. (3) Trailing bookkeeping entries in `haveEphys` beyond the Bpod trial count are trimmed via `_fit_trial_mask`. (4) For paw positions, nearest-fill interpolation is applied to fill NaN gaps. (5) For tongue, NaN gradients are replaced with 0. (6) Trials where all neural units are zero are excluded.

ii.
```python
def _fit_trial_mask(values, ntrials):
    values = np.asarray(values).reshape(-1).astype(bool)
    if values.size >= ntrials:
        return values[:ntrials]
    return np.pad(values, (0, ntrials - values.size), constant_values=False)
```

iii. The agent noted (step 87): "One randomized file has a one-element bookkeeping overhang" and (step 93): "the compatibility rule trims only trailing bookkeeping excess."

## 11-a. What are the most time-consuming steps of the code?

i. Loading the large MATLAB files dominates runtime. The HDF5 files (v7.3) require field-by-field reading, and the legacy files require full deserialization. The agent reported 44 sessions completed successfully with decoder training also running.

ii.
```python
if h5py.is_hdf5(data_path):
    with h5py.File(data_path, "r") as matfile:
        ...
else:
    obj = loadmat(data_path, squeeze_me=True, ...)
```

iii. File I/O is inherently the bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cluster loop in `_load_neural` (and `_load_neural_v5`) processes each cluster sequentially: reading spike data, aligning, binning, and smoothing. This loop could potentially be vectorized if all cluster data were concatenated first. The per-trial loop in `_load_behavior` for video features could also be partially vectorized.

ii.
```python
for cluster_idx in range(cluster_group["trial"].shape[0]):
    ...
    counts = np.zeros((ntrials, TIME.size), dtype=np.float32)
    np.add.at(counts, (spike_trials, time_idx), 1.0)
    units.append(_smooth_counts(counts))
```

iii. Each cluster is binned independently over all trials, creating a full (ntrials, n_bins) array per cluster. The reference instead uses a single `histogram2d` per cluster (which is similar) but the AI's approach also allocates a full trial array per cluster even before filtering.

## 11-c. What processing does the code repeat multiple times?

i. The go cue vector `bp.ev.goCue` is read once per cluster inside `_load_neural` rather than being pre-loaded. The video offset is computed once per session. Frame times are read multiple times for different features within the same trial.

ii.
```python
# Inside the cluster loop:
go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
```

iii. The repeated go cue reads are redundant but likely not a major performance issue due to HDF5 caching.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `NdroppedFrames` for each trial and feature, which is only used as a validity check. The `quality_counts` dictionary is computed and stored in metadata but not used in the conversion itself. The `_fit_trial_mask` and baseline subtraction computations add overhead. The code computes both paws' velocities and averages them, which is unnecessary processing if only one paw is needed. The `_nearest_fill` function for paw positions fills in gaps that are then used for velocity computation, but these filled values may not be physically meaningful.

ii.
```python
quality_counts[quality or "unlabelled"] = quality_counts.get(quality or "unlabelled", 0) + 1
```

iii. The metadata tracking is useful for auditability but not required for the conversion itself.
