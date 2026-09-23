# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session Figure 8 subset in `SESSIONS` and loads one `data_structure_<animal>_<date>.mat` plus one `motionEnergy_<animal>_<date>.mat` per session from a single `data_dir`. It reads the data structure with `h5py` only, then loads motion energy separately with `scipy.io.loadmat`.

ii.
```python
SESSIONS = (
    ("JEB6", "2021-04-18", 2),
    ...
    ("JEB19", "2023-04-18", 1),
)

for session_index, (animal, date, probe) in enumerate(SESSIONS, start=1):
    stem = f"{animal}_{date}"
    data_path = data_dir / f"data_structure_{stem}.mat"
    motion_path = data_dir / f"motionEnergy_{stem}.mat"
    ...
    with h5py.File(data_path, "r") as handle:
        behavior = _load_behavior(handle)
```

iii. In the trajectory, the AI says it "narrow[ed] to those sessions" because only the two-context electrophysiology sessions jointly provided neural activity, WC/DR labels, kinematics, and motion energy, and then decided that the 12-session Figure 8 pipeline was the right target because it reproduced the paper's reported 522 ALM units exactly.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the first element of each hard-coded session tuple. The final `subjects` list preserves first appearance order, and `subject_idx` is built by looking up each session's animal in that list.

ii.
```python
subjects = list(dict.fromkeys(session[0] for session in SESSIONS))
...
"subject_idx": np.asarray(
    [subjects.index(animal) for animal, _, _ in SESSIONS], dtype=np.int64
),
```

iii. The trajectory does not justify this separately. The implied justification is the same Figure 8 session table used throughout the script.

## 1-c. How are the data split into sessions?

i. Each `(animal, date, probe)` tuple in `SESSIONS` is treated as one session. The loop over `SESSIONS` produces one entry per session in `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
for session_index, (animal, date, probe) in enumerate(SESSIONS, start=1):
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
    region_indices.append(np.zeros(neural_all.shape[1], dtype=np.int64))
```

iii. The trajectory explicitly says the AI chose the repository's "12 two-context sessions" from the Figure 8 pipeline and used that as the session definition.

## 1-d. How are the data split into trials?

i. The AI treats the behavioral arrays as the trial axis, using `len(go_cue)` as `ntrials`. Trial-wise neural, video, and motion-energy arrays are all indexed by that shared trial number, and the final exported trials are `np.flatnonzero(keep)`.

ii.
```python
result["go_cue"] = _array(bp["ev/goCue"]).astype(np.float64)
result["ntrials"] = np.asarray([len(result["go_cue"])], dtype=np.int64)
...
ntrials = int(behavior["ntrials"][0])
...
trial_ids = np.flatnonzero(keep)
for trial in trial_ids:
    neural_trials.append(neural_all[trial].copy())
```

iii. The trajectory does not contain a separate trial-splitting justification. The code implies that the go-cue indexed behavioral table is the canonical trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by dropping early-lick and photostimulation trials. Ignore trials are explicitly kept, and there is no additional filter for trials that extend beyond the usable recording.

ii.
```python
keep = ~behavior["early"] & ~behavior["stim"]
...
trial_ids = np.flatnonzero(keep)
```

iii. The trajectory explicitly says the AI would "exclude photostimulation and early-lick trials as the analyses do, while retaining ignore trials because the requested outcome explicitly requires that class."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the selected probe inside `obj/clu`, specifically each unit's `quality`, `trial`, and `trialtm`, together with `obj/bp/ev/goCue` for alignment.

ii.
```python
probe_group = _deref(handle, handle["obj/clu"], probe - 1)
...
quality = _string(_deref(handle, probe_group["quality"], unit))
spike_trial = _array(_deref(handle, probe_group["trial"], unit)).astype(np.int64)
spike_time = _array(_deref(handle, probe_group["trialtm"], unit)).astype(np.float64)
aligned = spike_time - go_cue[spike_trial - 1]
```

iii. The trajectory's main justification is that the AI was following the Figure 8 ALM pipeline and probe assignments.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, counted into 10 ms bins from `-3.0` to `2.5` s, converted to rates by dividing by `DT`, and then smoothed with a causal 15-bin Gaussian kernel. The exported per-trial neural matrices are those smoothed rates.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 0.01
SMOOTH_BINS = 15
...
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
rates = _causal_smooth(counts / DT)
...
return np.stack(neurons, axis=1), qualities
```

iii. The trajectory explicitly justifies these choices by saying the AI followed the Figure 8 pipeline's `[-3.0, 2.5)` s window, 10 ms bins, and causal 15-bin Gaussian smoothing because that reproduced the reported 522 ALM units.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI drops units whose `quality` string exactly matches one of `garbage`, `gabrga`, `noisy`, or `real?`, then keeps only units whose condition-averaged smoothed firing rate exceeds 1 Hz. It does not lower-case quality labels and does not exclude `poor`.

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

iii. The trajectory says the AI used the Figure 8 "quality exclusions, and >1 Hz filter" because those settings reproduced the paper's 522-unit count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike is aligned by subtracting that trial's go-cue time from `trialtm`, then assigning the result to the fixed bin grid.

ii.
```python
aligned = spike_time - go_cue[spike_trial - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The trajectory says the AI chose go-cue alignment as part of the Figure 8 pipeline and the task requirements.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported neural data uses 10 ms bins (`DT = 0.01`) over a `-3.0` to `2.5` s window, giving 550 time bins. There is no later rebinning after the initial spike counting; all streams are put directly onto this grid.

ii.
```python
DT = 0.01
...
edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
time = (edges[:-1] + DT / 2).astype(np.float32)
```

iii. The trajectory explicitly identifies "10 ms bins" and the `[-3.0, 2.5)` s Figure 8 window as the chosen temporal resolution.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time axis is derived from the AI's chosen fixed bin grid around the go cue, not from an explicit raw time series. The only raw data dependency is the decision to align trials to `bp.ev.goCue`.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
time = (edges[:-1] + DT / 2).astype(np.float32)
...
input_trials.append(time[None, :].copy())
```

iii. The trajectory justifies this only indirectly: the AI decided to use the Figure 8 go-cue-aligned binning grid as the common time base.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI creates a single vector of bin centers from `TMIN`, `TMAX`, and `DT`, then copies that same `1 x T` array into every trial. There is no additional per-trial processing.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
time = (edges[:-1] + DT / 2).astype(np.float32)
...
input_trials.append(time[None, :].copy())
```

iii. No separate justification appears in the trajectory beyond the general Figure 8 time-grid choice.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same bin-center vector used as the target time grid for neural and video-aligned processing, so it is aligned by construction.

ii.
```python
time = (edges[:-1] + DT / 2).astype(np.float32)
...
aligned = spike_time - go_cue[spike_trial - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
input_trials.append(time[None, :].copy())
```

iii. The trajectory's justification is the same general claim that one go-cue-aligned Figure 8 grid should be shared across streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from behavioral trial flags: `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
lick[(behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])] = 1
lick[behavior["no"]] = 2
```

iii. The trajectory explicitly says, "Actual lick choice will be derived from target side plus hit/miss, not merely copied from the instructed side."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI encodes left as the default, sets right on `R-hit` or `L-miss` trials, and sets `none` on ignore trials (`no`). That makes `L-hit` and `R-miss` left by default.

ii.
```python
lick = np.zeros(len(keep), dtype=np.int8)  # left
lick[(behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])] = 1
lick[behavior["no"]] = 2
```

iii. The trajectory justifies this as reconstructing actual lick choice from target side plus outcome, rather than treating instructed side as observed behavior.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived solely from `autowater`.

ii.
```python
context = behavior["autowater"].astype(np.int8)
```

iii. The trajectory does not justify this separately. The code implies a direct use of the session's autowater flag for DR/WC labeling.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly casts `autowater` to integers, so `False -> 0` and `True -> 1`. In the exported metadata that is interpreted as `0 = DR`, `1 = WC`.

ii.
```python
context = behavior["autowater"].astype(np.int8)  # 0 DR, 1 WC
...
"output_values": [
    ["left", "right", "none"],
    ["DR", "WC"],
```

iii. No separate justification appears in the trajectory.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit` and `no`, with miss trials represented implicitly by the default incorrect label.

ii.
```python
outcome = np.zeros(len(keep), dtype=np.int8)  # incorrect
outcome[behavior["hit"]] = 1
outcome[behavior["no"]] = 2
```

iii. The trajectory does not justify this separately. It follows the same behavioral decoding logic used for lick direction and ignore-trial retention.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Trials default to `incorrect`, hit trials become `correct`, and `no` trials become `ignore`.

ii.
```python
outcome = np.zeros(len(keep), dtype=np.int8)  # incorrect
outcome[behavior["hit"]] = 1
outcome[behavior["no"]] = 2
```

iii. The trajectory explicitly says ignore trials were retained because the requested decoder output required that class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side camera only (`camera=0`, feature `"tongue"`), using `obj/traj` frame times and tracked coordinates, plus the session's video/behavior offset and the go cue.

ii.
```python
tongue_speed, tongue_visible = _load_velocity(
    handle, behavior, time, camera=0, feature="tongue"
)
...
frames = _array(_deref(handle, group["frameTimes"], trial)).astype(np.float64).reshape(-1)
tracking = _array(_deref(handle, group["ts"], trial)).astype(np.float64)
aligned_frames = frames - offset - behavior["go_cue"][trial]
```

iii. The only explicit justification is the code comment calling side-view tongue a "repository-standard" feature. The trajectory does not discuss the missing second tongue view separately.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI linearly interpolates tongue `x` and `y` onto the fixed 10 ms grid, marks visibility where both interpolated coordinates are finite, computes `np.gradient` of the interpolated coordinates, and takes the Euclidean norm as speed. It does not smooth, does not use likelihood explicitly, and does not combine the bottom-camera tongue view.

ii.
```python
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
ypos = _interp_trace(aligned_frames, tracking[:, 1, feature_index], target_time)
visible[trial] = np.isfinite(xpos) & np.isfinite(ypos)
...
xvel, yvel = np.gradient(xpos), np.gradient(ypos)
speed[trial] = np.hypot(xvel, yvel)
```

iii. The trajectory does not justify this processing in detail. Its general rationale was to follow what it understood as the Figure 8 pipeline while preserving decoder-valid category structure.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI pools all visible tongue-speed samples from kept trials within a session, takes the 50th percentile, and labels each bin as `0` below threshold, `1` at or above threshold, and `2` when unavailable.

ii.
```python
usable = keep[:, None] & available & np.isfinite(values)
threshold = float(np.percentile(values[usable], 50))
output = np.full(values.shape, 2, dtype=np.int8)
output[usable & (values < threshold)] = 0
output[usable & (values >= threshold)] = 1
```

iii. The trajectory later notes that "each per-session median splits available samples evenly," which is the only explicit justification for this thresholding behavior.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI estimates one session-wide video offset from bitcode timing, subtracts that offset and the trial's go cue from the camera frame times, and then interpolates tongue coordinates directly onto the same fixed time vector used by the neural data.

ii.
```python
return float(np.median(video_bit_start) / sampling_rate - np.median(bit_start))
...
aligned_frames = frames - offset - behavior["go_cue"][trial]
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
ypos = _interp_trace(aligned_frames, tracking[:, 1, feature_index], target_time)
```

iii. In the trajectory, the AI says it checked the repository's "video-offset logic" before deciding on the common time window, and then used one shared go-cue-aligned grid across streams.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera only (`camera=1`, feature `"top_paw"`), again using `obj/traj` frame times and tracked coordinates together with the video offset and go cue.

ii.
```python
paw_speed, paw_visible = _load_velocity(
    handle, behavior, time, camera=1, feature="top_paw"
)
```

iii. The code comment says this uses a repository-standard paw feature; the trajectory does not provide a separate justification.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw coordinates are linearly interpolated onto the 10 ms grid, then missing coordinate samples are filled by nearest-neighbor interpolation before differentiation. The AI computes `np.gradient` on the filled traces, subtracts each coordinate's median finite difference as a drift correction, and takes the Euclidean norm as speed.

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

iii. The trajectory does not justify this in detail. One later update mentions "silencing drift estimation on wholly missing video trials," which matches the special handling for missing paw/video traces.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same `_median_discretize` rule as tongue velocity: pooled per-session median split across kept, visible bins, with unavailable bins mapped to class `2`.

ii.
```python
paw_class, paw_threshold = _median_discretize(paw_speed, paw_visible, keep)
```

iii. The trajectory provides no separate paw-threshold justification beyond the general per-session median split noted after validation.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frames are aligned by subtracting the session-wide video offset and trial go cue, then interpolated onto the same fixed `time` vector used for the neural data.

ii.
```python
offset = _video_offset(handle, behavior)
...
aligned_frames = frames - offset - behavior["go_cue"][trial]
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
```

iii. The trajectory's only explicit rationale is that it checked the repository's video-offset logic and used a shared time base across modalities.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically `mat["me"]["data"]`, and its timing is taken from the side-camera frame times plus the session video offset and trial go cue.

ii.
```python
mat = io.loadmat(path, simplify_cells=True, variable_names=["me"])
raw = mat["me"]["data"]
...
group = _video_group(handle, 0)
...
aligned_frames = frames - offset - behavior["go_cue"][trial]
trace = np.asarray(raw_trials[trial], dtype=np.float64).reshape(-1)
```

iii. The trajectory says the chosen session family was one that jointly provided neural activity, context labels, kinematics, and motion energy. It does not discuss the motion-energy file layout separately.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the AI linearly interpolates the per-frame motion-energy trace onto the fixed 10 ms target grid. It does not smooth or otherwise transform the values before median-thresholding them later.

ii.
```python
values[trial] = _interp_trace(aligned_frames, trace, target_time)
available[trial] = np.isfinite(values[trial])
```

iii. No separate processing justification appears in the trajectory.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with the same `_median_discretize` helper: per-session 50th percentile over kept, available bins, with class `2` used for unavailable bins.

ii.
```python
motion_class, motion_threshold = _median_discretize(motion_energy, motion_available, keep)
...
"output_values": [
    ...
    ["< session median", ">= session median", "no video"],
],
```

iii. The trajectory later says the per-session median split gave internally consistent category distributions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session-wide video offset and the trial's go cue from side-camera frame times, then interpolating onto the same `time` vector used by the neural data.

ii.
```python
offset = _video_offset(handle, behavior)
...
aligned_frames = frames - offset - behavior["go_cue"][trial]
values[trial] = _interp_trace(aligned_frames, trace, target_time)
```

iii. The trajectory gives only the general video-offset and shared-grid rationale.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable video-derived samples are generally turned into `NaN` during interpolation and then mapped to category `2` by `_median_discretize`. Entire trials are skipped for a given camera stream if `NdroppedFrames` is non-finite, and paw coordinates are additionally imputed with nearest-neighbor filling before velocity computation.

ii.
```python
if n < 2:
    return np.full(target_time.shape, np.nan, dtype=np.float64)
...
if np.size(dropped) and not np.all(np.isfinite(dropped)):
    continue
...
if "tongue" not in feature:
    xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)
...
output = np.full(values.shape, 2, dtype=np.int8)
```

iii. The only explicit trajectory justification is the note about "silencing drift estimation on wholly missing video trials." Otherwise the trajectory focuses on passing validation and obtaining reasonable visibility distributions rather than on a principled missing-data policy.

## 11-a. What are the most time-consuming steps of the code?

i. The main expensive steps are the per-unit neural loop in `_load_neural` and the per-trial interpolation/gradient loops in `_load_velocity` and `_load_motion_energy`, plus repeated HDF5 dereferencing. The AI code does substantial numerical work after loading rather than only reading files.

ii.
```python
for unit in range(probe_group["quality"].shape[0]):
    ...
    np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
    rates = _causal_smooth(counts / DT)

for trial in range(ntrials):
    ...
    xpos = _interp_trace(...)
    ypos = _interp_trace(...)
```

iii. The trajectory does not discuss runtime hotspots explicitly.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code keeps explicit Python loops over units in `_load_neural`, over trials in `_load_velocity`, over trials in `_load_motion_energy`, and over kept trial IDs when assembling the output. Some of that is difficult because the source arrays are ragged, but these are still the obvious loop bottlenecks.

ii.
```python
for unit in range(probe_group["quality"].shape[0]):
    ...

for trial in range(ntrials):
    ...

for trial in trial_ids:
    neural_trials.append(neural_all[trial].copy())
```

iii. The trajectory does not contain an explicit efficiency justification.

## 11-c. What processing does the code repeat multiple times?

i. The biggest repeated computation is `_video_offset`, which is recomputed once inside each `_load_velocity` call and again inside `_load_motion_energy` for the same session. The code also copies the identical `time[None, :]` input array for every kept trial and repeatedly searches `subjects.index(animal)` when assembling `subject_idx`.

ii.
```python
offset = _video_offset(handle, behavior)
...
tongue_speed, tongue_visible = _load_velocity(...)
paw_speed, paw_visible = _load_velocity(...)
motion_energy, motion_available = _load_motion_energy(...)
...
input_trials.append(time[None, :].copy())
...
[subjects.index(animal) for animal, _, _ in SESSIONS]
```

iii. The trajectory does not acknowledge this repeated work.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several metadata-only summaries that are not used by downstream decoding, including `qualities` counts, per-session median thresholds, and the `session_info` block. It also duplicates the full `time` vector into every trial even though it is identical across all trials and already described in metadata.

ii.
```python
neural_all, qualities = _load_neural(handle, probe, behavior, edges)
...
session_info.append(
    {
        ...
        "unit_quality_counts": {
            quality: qualities.count(quality) for quality in sorted(set(qualities))
        },
        "tongue_velocity_median": tongue_threshold,
        "paw_velocity_median": paw_threshold,
        "motion_energy_median": motion_threshold,
    }
)
...
input_trials.append(time[None, :].copy())
```

iii. The trajectory does not justify these extra summaries; it only reports them as part of validation and dataset description.
