# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes two session-to-probe maps (`ALM_PROBES` and `RANDOMIZED_DELAY_PROBES`), then iterates the two ephys folders in `SESSION_GROUPS`. Within each folder it glob-discovers `data_structure_*.mat`, matches those files to the hard-coded keys, and expects a paired `motionEnergy_<session>.mat`. HDF5 MAT files are opened with `h5py`; legacy MAT files are opened with `scipy.io.loadmat`.

ii. ```python
SESSION_GROUPS = (
    ("Ephys_Behavior", "fixed_delay", ALM_PROBES),
    ("RandomizedDelay_Ephys_Behavior", "randomized_delay", RANDOMIZED_DELAY_PROBES),
)
...
available = {_session_key(path): path for path in directory.glob("data_structure_*.mat")}
...
if h5py.is_hdf5(data_path):
    with h5py.File(data_path, "r") as matfile:
        ...
else:
    obj = loadmat(data_path, squeeze_me=True, struct_as_record=False,
                  variable_names=["obj"])["obj"]
```

iii. In the trajectory, the agent said it was narrowing the conversion to the 25 fixed-delay and 19 randomized-delay recording sessions selected by the repository loaders, and excluding optogenetic-only directories because the decoder requires neural activity (steps 8 and 26).

## 1-b. How are the data split into subjects?

i. Each session key is split at the first underscore; the prefix is treated as the mouse id. Those subject strings are accumulated per session, then deduplicated in first-seen order when building `subjects`, with `subject_idx` pointing back into that list.

ii. ```python
key = _session_key(data_path)
subject, date = key.split("_", 1)
...
session_subjects.append(subject)
...
subjects = list(dict.fromkeys(session_subjects))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int64),
```

iii. The trajectory does not give a separate subject-parsing justification; it treats the repository session names as the authoritative identifiers for sessions and animals.

## 1-c. How are the data split into sessions?

i. One selected MATLAB file is treated as one session. The code keeps the fixed-delay and randomized-delay folders separate only for discovery/metadata; after loading, each selected file becomes one session entry in `neural`, `input`, and `output`.

ii. ```python
for directory_name, task_variant, probe_map in SESSION_GROUPS:
    directory = data_root / directory_name
    ...
    for key in sorted(probe_map):
        sessions.append((available[key], directory / f"motionEnergy_{key}.mat",
                         probe_map[key], task_variant))
...
for session_idx, (data_path, motion_path, probes, task_variant) in enumerate(sessions, start=1):
```

iii. The trajectory explicitly says the code includes the 25 fixed-delay and 19 randomized-delay ALM recording sessions selected by the paper’s loading scripts (step 26).

## 1-d. How are the data split into trials?

i. Trials are indexed by the Bpod trial axis. The code uses trial-length vectors from `obj.bp` and trial masks such as `early`, `stim.enable`, and `haveEphys` to build `candidate_trials` as integer trial indices. Later arrays (`neural_all`, `output_all`) are subset by those trial indices.

ii. ```python
early = _vector(matfile["obj/bp/early"]).astype(bool)
stimulated = _vector(matfile["obj/bp/stim/enable"]).astype(bool)
have_ephys = _fit_trial_mask(
    _vector(matfile["obj/trials/bp/haveEphys"]), early.size)
candidate_trials = np.flatnonzero(~early & ~stimulated & have_ephys)
...
selected = candidate_trials[neural_present[candidate_trials]]
```

iii. The trajectory says the agent was checking the exact MATLAB fields and trial-selection rules rather than inferring trial boundaries from filenames or reconstructed events (step 8).

## 1-e. How are trials filtered based on quality controls?

i. The code excludes early-lick trials and photostimulation trials first. It then keeps only trials where `haveEphys` is true after trimming/padding that bookkeeping vector to the Bpod trial count. Finally, it removes any remaining candidate trial whose retained neural population is all zeros across the full 5 s window.

ii. ```python
candidate_trials = np.flatnonzero(~early & ~stimulated & have_ephys)
...
neural_all, neural_info = _load_neural(matfile, probes)
neural_present = np.any(neural_all != 0, axis=(1, 2))
selected = candidate_trials[neural_present[candidate_trials]]
```

iii. The trajectory says the agent intentionally removed early-lick and photostimulation trials (step 20). Later, after verification, it said two sessions contained behavioral trials after ephys ended; it first tried `haveEphys`, found that flag wrong for the trailing tail, and then switched to excluding trials with all-zero retained neural activity as “direct evidence” of being outside electrophysiology coverage (steps 76, 87, and 104).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from cluster-level spike metadata in `obj.clu`: per-spike trial ids (`trial`), per-spike times within trial (`trialtm`), and cluster `quality`. Alignment also uses `obj.bp.ev.goCue`.

ii. ```python
cluster_group = matfile[np.asarray(matfile["obj/clu"])[probe - 1, 0]]
...
quality_obj = matfile[np.asarray(cluster_group["quality"])[cluster_idx, 0]]
...
spike_trials = _deref_vector(matfile, cluster_group["trial"], cluster_idx).astype(np.int64) - 1
spike_times = _deref_vector(matfile, cluster_group["trialtm"], cluster_idx).astype(np.float64)
go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
```

iii. The trajectory says the repository confirmed go-cue alignment, session-specific probe selection, and curated-unit inclusion from the published loaders (step 12).

## 2-b. How is the `neural` data processed?

i. For each retained cluster, the code aligns spikes to go cue, bins them into 5 ms bins from -2.5 to 2.5 s, and then applies a 15-bin causal Gaussian-like smoothing kernel implemented with direct convolution. The smoothed counts are converted to spikes/s by dividing by `DT`. No z-scoring or baseline subtraction is applied.

ii. ```python
time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((ntrials, TIME.size), dtype=np.float32)
np.add.at(counts, (spike_trials, time_idx), 1.0)
units.append(_smooth_counts(counts))
...
def _smooth_counts(counts: np.ndarray) -> np.ndarray:
    padded = np.concatenate((counts[:, :SMOOTH_BINS], counts), axis=1)
    smoothed = convolve(padded, SMOOTH_KERNEL[None, :], mode="same", method="direct")
    return smoothed[:, SMOOTH_BINS:] / np.float32(DT)
```

iii. The trajectory says the agent believed the repository’s preprocessing used go-cue-aligned 5 ms spike-rate bins and the paper’s 15-bin causal smoother, and that it was preserving that convention rather than inventing a new filter (steps 12 and 20).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code drops clusters whose manual quality label lower-cases to one of `garbage`, `gabrga`, `noisy`, or `real?`. It then keeps only units whose mean smoothed firing rate over all trials and bins is greater than 1 Hz. It does not drop `poor` units.

ii. ```python
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
if quality.lower() in REJECTED_QUALITIES:
    continue
...
mean_fr = neural.mean(axis=(0, 2), dtype=np.float64)
keep = mean_fr > LOW_FR_HZ
return neural[:, keep, :], {
```

iii. The trajectory explicitly justified the 1 Hz threshold and manual curation generally (step 12), but it did not separately justify omitting `poor` from the rejection list.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each spike’s trial-specific go-cue time from `trialtm`, yielding seconds from go cue before binning.

ii. ```python
aligned = spike_times[valid_trial] - go_cue[spike_trials]
```

iii. The trajectory explicitly says the repository confirmed go-cue alignment for the neural preprocessing (step 12).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 5 ms bins (`DT = 1/200`) over a fixed 5 s window from -2.5 to 2.5 s. There is no later temporal rebinning; the same 1000-bin grid is used throughout.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
EDGES = TMIN + np.arange(TIME.size + 1, dtype=np.float64) * DT
```

iii. The trajectory says the agent resolved the neural preprocessing to 5 ms bins from -2.5 to 2.5 s (step 12).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The decoder input is not read from a dedicated raw-data variable. It is the synthetic time axis defined by `TMIN`, `TMAX`, and `DT`, interpreted as time relative to each trial’s go cue.

ii. ```python
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
...
input_trials = [TIME[None, :].copy() for _ in range(selected.size)]
```

iii. The trajectory’s justification is that everything should be aligned to go cue on the same 5 ms grid used for the neural data (step 12).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No per-trial computation is done beyond copying the precomputed `TIME` vector into shape `(1, 1000)` for each trial.

ii. ```python
input_trials = [TIME[None, :].copy() for _ in range(selected.size)]
```

iii. The trajectory does not give a separate justification beyond using the common go-cue-centered bin grid.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `TIME` grid that the neural counts are binned against, so the time input and neural activity share one 1000-bin axis.

ii. ```python
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
...
time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The trajectory says the agent wanted all streams on the same go-cue-aligned 5 ms grid (steps 12 and 20).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from trial-level behavioral flags: `hit`, `miss`, `no` (ignore), and the instructed right-side target flag `R`.

ii. ```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
right_target = _vector(matfile["obj/bp/R"]).astype(bool)
```

iii. The trajectory says the agent resolved lick direction as actual choice: target side on correct trials, opposite side on incorrect trials, and `none` on ignores (step 20).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code labels ignore trials as class 2 (`none`), hit trials as the instructed side, and miss trials as the opposite side.

ii. ```python
if ignore[source_trial]:
    lick_direction, outcome = 2, 2  # none, ignore
elif hit[source_trial]:
    lick_direction = 1 if right_target[source_trial] else 0
    outcome = 1  # correct
elif miss[source_trial]:
    lick_direction = 0 if right_target[source_trial] else 1
    outcome = 0  # incorrect
```

iii. The trajectory states exactly this actual-choice interpretation and says ignore trials were retained because ignore is a requested decoder outcome (step 20).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `obj.bp.autowater`, stored as `water_cued`.

ii. ```python
water_cued = _vector(matfile["obj/bp/autowater"]).astype(bool)
```

iii. The trajectory says context came from the repository’s `autowater` flag (step 20).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code relabels `autowater=True` to WC (`0`) and `False` to DR (`1`).

ii. ```python
context = 0 if water_cued[source_trial] else 1  # WC, DR
```

iii. The trajectory’s justification is simply that `autowater` carries the context identity for these tasks (step 20).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit`, `miss`, and `no` trial flags.

ii. ```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
```

iii. The trajectory says ignore trials were intentionally retained because ignore is one of the requested decoder outputs (step 20).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit trials are labeled correct (`1`), miss trials incorrect (`0`), and ignore trials ignore (`2`).

ii. ```python
if ignore[source_trial]:
    lick_direction, outcome = 2, 2
elif hit[source_trial]:
    ...
    outcome = 1  # correct
elif miss[source_trial]:
    ...
    outcome = 0  # incorrect
```

iii. The trajectory explicitly says ignores were retained as a separate class rather than being dropped (step 20).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived only from the side-camera DeepLabCut tongue feature (`"tongue"`), using that camera’s `featNames`, `ts`, and `frameTimes`, plus the shared video-to-behavior shift from `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.bitStart`, and the per-trial `goCue`.

ii. ```python
trajectory_side = matfile[np.asarray(matfile["obj/traj"])[0, 0]]
...
tx, ty, tongue_vis = _trajectory_xy(
    matfile, trajectory_side, source_trial, "tongue", TIME,
    video_shift, go_cue[source_trial])
```

iii. The trajectory only says the agent was “resolving video visibility and timing fields” and wanted visibility classes to be meaningful (steps 12 and 20). It does not explicitly justify using only the side-camera tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code linearly interpolates tongue x/y coordinates from the side camera onto the 5 ms neural time grid, marks bins as visible when interpolated x and y are finite, computes velocity as the Euclidean norm of `np.gradient(x)` and `np.gradient(y)`, and leaves invisible bins as NaN. It does not use likelihood thresholds, per-run smoothing, or two-camera normalization/averaging.

ii. ```python
x = _interp(source_time, x_raw, aligned_time)
y = _interp(source_time, y_raw, aligned_time)
visible = np.isfinite(x) & np.isfinite(y)
...
if tongue:
    xvel = np.gradient(x)
    yvel = np.gradient(y)
    xvel[~np.isfinite(xvel)] = 0.0
    yvel[~np.isfinite(yvel)] = 0.0
return np.hypot(xvel, yvel)
```

iii. The trajectory says the agent wanted “not visible/no video” to remain meaningful before nearest-value filling (step 20), but it does not provide an explicit justification for this simplified tongue-velocity pipeline.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code pools all finite visible tongue-speed samples from the retained trials in one session, takes their 50th percentile, sets visible bins above or equal to that threshold to class 1, visible bins below it to class 0, and leaves invisible bins at class 2.

ii. ```python
tongue_values = tongue_speed[tongue_visible & np.isfinite(tongue_speed)]
...
"tongue_velocity_median": float(np.percentile(tongue_values, 50)),
...
output[out_trial, 3:6, :] = 2
valid = tongue_visible[out_trial] & np.isfinite(tongue_speed[out_trial])
output[out_trial, 3, valid] = (
    tongue_speed[out_trial, valid] >= thresholds["tongue_velocity_median"])
```

iii. The trajectory justification is indirect: the agent said it would keep explicit visibility classes meaningful while still using the requested median split (step 20).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted from video time into behavior time by subtracting `video_shift` and then the trial’s `goCue`, and tongue position is then interpolated directly onto the 5 ms `TIME` grid shared with neural data.

ii. ```python
video_shift = (_mode(matfile["obj/sglx/bitcode/bitstart"]) / fs
               - _mode(matfile["obj/bp/ev/bitStart"]))
...
source_time = frame_times - video_shift - go_cue
x = _interp(source_time, x_raw, aligned_time)
y = _interp(source_time, y_raw, aligned_time)
```

iii. The trajectory says the agent was matching the repository’s video timing conventions and wanted all streams on a common go-cue-centered grid (steps 12 and 20).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera DeepLabCut features, `"top_paw"` and `"bottom_paw"`, using their tracked coordinates and the same video timing fields used for tongue alignment.

ii. ```python
trajectory_bottom = matfile[np.asarray(matfile["obj/traj"])[1, 0]]
...
for feature in ("top_paw", "bottom_paw"):
    px, py, paw_vis = _trajectory_xy(
        matfile, trajectory_bottom, source_trial, feature, TIME,
        video_shift, go_cue[source_trial])
```

iii. The only explicit justification appears in the code comments, not the trajectory: the script says averaging the two tracked paws gives one requested paw-velocity stream “without privileging either paw.” The trajectory itself does not separately justify this choice.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw track is interpolated onto the 5 ms grid. Missing x/y samples are nearest-filled before differentiation. Velocity is computed from gradients after subtracting a median baseline term, then the two paw speeds are averaged where visible.

ii. ```python
x_filled = _nearest_fill(x)
y_filled = _nearest_fill(y)
xvel = np.gradient(x_filled) - baseline[0]
yvel = np.gradient(y_filled) - baseline[0]
...
paw_stack = np.stack(paw_speeds)
mask_stack = np.stack(paw_masks)
count = mask_stack.sum(axis=0)
summed = np.where(mask_stack, paw_stack, 0.0).sum(axis=0)
paw_speed[out_trial] = np.divide(
    summed, count, out=np.full(TIME.shape, np.nan), where=count > 0)
```

iii. The trajectory does not give a direct paw-processing justification. The nearest explanation is the step-20 statement that visibility classes should be preserved before nearest-value filling.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code uses the session median of all finite visible paw-speed samples across retained trials, then labels visible bins below median as 0, above or equal to median as 1, and invisible bins as 2.

ii. ```python
paw_values = paw_speed[paw_visible & np.isfinite(paw_speed)]
...
"paw_velocity_median": float(np.percentile(paw_values, 50)),
...
valid = paw_visible[out_trial] & np.isfinite(paw_speed[out_trial])
output[out_trial, 4, valid] = (
    paw_speed[out_trial, valid] >= thresholds["paw_velocity_median"])
```

iii. The trajectory gives only the general justification that the requested discretization is a per-session median split while preserving a visibility class (step 20).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted into behavior time by subtracting `video_shift` and the per-trial `goCue`, then the paw tracks are interpolated onto the same 5 ms `TIME` grid used by neural data.

ii. ```python
source_time = frame_times - video_shift - go_cue
x = _interp(source_time, ts[:, 0, feature_idx], aligned_time)
y = _interp(source_time, ts[:, 1, feature_idx], aligned_time)
```

iii. The trajectory says the agent was matching the repository’s video timing conventions and common go-cue-aligned grid, but it does not separately discuss paw alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the standalone `motionEnergy_<session>.mat` sidecar file, using its `me.data` field (with one extra wrapper layer optionally unwrapped). Those per-trial framewise traces are paired with side-camera `frameTimes`.

ii. ```python
me_obj = loadmat(motion_path, squeeze_me=True, struct_as_record=False,
                 variable_names=["me"])["me"]
motion_data = me_obj.data
if hasattr(motion_data, "data"):
    motion_data = motion_data.data
trials = np.asarray(motion_data, dtype=object).reshape(-1)
...
frame_obj = matfile[np.asarray(trajectory_side["frameTimes"])[source_trial, 0]]
frame_times = np.asarray(frame_obj).reshape(-1, order="F")
raw_motion = np.asarray(motion_trials[source_trial], dtype=np.float64).reshape(-1)
```

iii. The trajectory explicitly notes that one JEB15 format variant had the extra MATLAB struct wrapper and that the agent was adding the same unwrapping rule used by the repository loader (step 38).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code linearly interpolates the raw framewise motion-energy trace onto the 5 ms neural time grid. If any finite samples remain, it nearest-fills out-of-bounds NaNs and uses the filled interpolated trace directly.

ii. ```python
source_time = frame_times - video_shift - go_cue[source_trial]
interp_motion = _interp(source_time, raw_motion, TIME)
if np.isfinite(interp_motion).any():
    interp_motion = _nearest_fill(interp_motion)
    motion[out_trial] = interp_motion
    motion_video[out_trial] = True
```

iii. The trajectory does not directly justify the interpolation-and-fill choice. The closest related justification is step 20, where the agent said visibility classes would be derived before nearest-value filling.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code pools all finite samples from trials where motion video is available, takes the session median, labels bins above or equal to it as 1 and below as 0, and leaves bins at class 2 if no motion video is available.

ii. ```python
motion_values = motion[motion_video & np.isfinite(motion)]
...
"motion_energy_median": float(np.percentile(motion_values, 50)),
...
valid = motion_video[out_trial] & np.isfinite(motion[out_trial])
output[out_trial, 5, valid] = (
    motion[out_trial, valid] >= thresholds["motion_energy_median"])
```

iii. The trajectory gives the same general justification as for the other movement outputs: use the requested median split while preserving an explicit missing/no-video class (step 20).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame times are converted to seconds from go cue using the same `video_shift` and per-trial `goCue`, then interpolated onto the 5 ms `TIME` grid shared with neural data.

ii. ```python
source_time = frame_times - video_shift - go_cue[source_trial]
interp_motion = _interp(source_time, raw_motion, TIME)
```

iii. The trajectory justification is again that video-derived streams should be placed onto the same aligned grid as neural data (steps 12 and 20).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code explicitly trims or pads bookkeeping masks like `haveEphys` to the Bpod trial count. Missing or malformed video arrays return all-NaN position traces and all-false visibility masks. For paw coordinates and motion energy, however, missing samples are nearest-filled before downstream use. If a trial lacks video (`haveVid` false), the movement outputs stay at their default class 2. Motion-energy wrapper-format inconsistencies are unwrapped once.

ii. ```python
def _fit_trial_mask(values: np.ndarray, ntrials: int) -> np.ndarray:
    ...
    if values.size >= ntrials:
        return values[:ntrials]
    return np.pad(values, (0, ntrials - values.size), constant_values=False)
...
if dropped.size == 0 or not np.isfinite(dropped[0]):
    nan = np.full(aligned_time.shape, np.nan, dtype=np.float64)
    return nan, nan.copy(), np.zeros(aligned_time.shape, dtype=bool)
...
values[missing] = values[nearest]
...
if not have_video[source_trial]:
    continue
```

iii. The trajectory explicitly justifies three of these choices: trimming the overlong `haveEphys` bookkeeping vector (step 87), excluding trailing no-ephys trials after discovering `haveEphys` itself was wrong on those tails (step 104), and preserving explicit “not visible/no video” classes rather than letting nearest filling erase them (step 20).

## 11-a. What are the most time-consuming steps of the code?

i. The code is likely dominated by per-session file loading, per-cluster spike binning/smoothing, and per-trial video interpolation/velocity computation. There is a top-level loop over 44 sessions, an inner loop over every retained cluster, and another loop over every retained trial for behavior/video outputs.

ii. ```python
for session_idx, (data_path, motion_path, probes, task_variant) in enumerate(sessions, start=1):
    ...
    neural_all, neural_info = _load_neural(matfile, probes)
    ...
    output_all, _, thresholds = _load_behavior(matfile, motion_path, selected)
```

iii. The trajectory does not explicitly discuss runtime hotspots. The closest evidence is that the agent repeatedly reported long full-dataset passes and validation runs across all 44 sessions (steps 34, 47, 98, and 135).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several Python loops in place: per-probe/per-cluster neural loading, per-trial movement extraction, per-feature paw processing, and the final per-trial output assignment loop. Some of those loops could likely be reduced or partially vectorized, especially repeated per-trial interpolation and output construction.

ii. ```python
for probe in probes:
    ...
    for cluster_idx in range(cluster_group["trial"].shape[0]):
        ...
for out_trial, source_trial in enumerate(selected):
    ...
    for feature in ("top_paw", "bottom_paw"):
        ...
```

iii. The trajectory does not present a justification for keeping these loops; it focuses on correctness and format verification instead of optimization.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several small computations: `go_cue` is re-read inside the cluster loop in `_load_neural`; feature-name lookup is redone for every trial/feature; very similar HDF5 and v5 behavior loaders duplicate the same processing; and `_nearest_fill`/`_interp` are rerun separately for different movement streams.

ii. ```python
for cluster_idx in range(cluster_group["trial"].shape[0]):
    ...
    go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
...
feature_idx = _feature_index(matfile, trajectory_group, trial, feature)
...
def _load_behavior(...):
    ...
def _load_behavior_v5(...):
    ...
```

iii. The trajectory does not explicitly justify these repeated computations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extra metadata that downstream decoder training does not need, including detailed `session_info`, quality-count summaries, and movement thresholds. It also returns `water_cued[selected]` from `_finalize_behavior_outputs` even though the caller discards that value. More broadly, it computes full interpolated x/y traces and intermediate speed arrays that are only used to derive the final discretized categories.

ii. ```python
return output, water_cued[selected], thresholds
...
output_all, _, thresholds = _load_behavior(matfile, motion_path, selected)
...
session_info.append({
    ...
    **neural_info,
    **thresholds,
})
```

iii. The trajectory does not explicitly justify this extra bookkeeping. Its focus was on getting a valid converted dataset and successful decoder training rather than minimizing discarded intermediate work.
