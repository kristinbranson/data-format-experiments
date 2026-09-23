# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the full archived dataset. It hard-codes a 12-session “two-context” cohort from `/app/data/Ephys_Behavior`, opens each `data_structure_<session>.mat` with `h5py`, and separately loads the paired `motionEnergy_<session>.mat` with `scipy.io.loadmat`. Within each selected session it reads behavioral flags, spikes, trajectories, and motion energy directly from the MATLAB object graph rather than globbing both ephys directories or using a generic MAT loader for the main session file.

ii. 
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

CONTEXT_SESSIONS = [
    ("JEB6", "2021-04-18", 2),
    ...
    ("JEB19", "2023-04-18", 1),
]
```

```python
with h5py.File(data_path, "r") as handle:
    ...
    motion_data = load_motion_energy(motion_path)
```

```python
def load_motion_energy(path: Path) -> list[np.ndarray]:
    motion = loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    data = np.atleast_1d(motion.data).ravel()
    return [np.asarray(item, dtype=np.float64).ravel() for item in data]
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as using “the exact two-context session/probe list used by the paper’s Figure 8 scripts,” restricting conversion to the context cohort instead of the full archive.

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects by the animal string in the hard-coded `(animal, date, probe)` tuples. The dataset-level `subjects` list is built in first-appearance order, and `subject_idx` stores the per-session index into that list.

ii. 
```python
for session_index, (animal, date, probe_number) in enumerate(session_specs):
    ...
    if animal not in subject_lookup:
        subject_lookup[animal] = len(subjects)
        subjects.append(animal)
    subject_idx.append(subject_lookup[animal])
```

iii. The notes explicitly say the AI preserved “native subject IDs” from loader names / filenames and did not merge IDs to force the paper’s stated mouse count.

## 1-c. How are the data split into sessions?

i. One session is one hard-coded `(animal, date, probe)` entry from `CONTEXT_SESSIONS`. Each entry maps to one `data_structure_<animal>_<date>.mat` file and one `motionEnergy_<animal>_<date>.mat` file in `Ephys_Behavior`, and each processed session becomes one element of `neural`, `input`, and `output`.

ii. 
```python
def process_session(animal: str, date: str, probe_number: int) -> SessionResult:
    session_id = f"{animal}_{date}"
    data_path = DATA_DIR / f"data_structure_{session_id}.mat"
    motion_path = DATA_DIR / f"motionEnergy_{session_id}.mat"
```

```python
for session_index, (animal, date, probe_number) in enumerate(session_specs):
    result = process_session(animal, date, probe_number)
    neural.append(result.neural)
    inputs.append(result.inputs)
    outputs.append(result.outputs)
```

iii. The AI’s notes say the target cohort is specifically the paper’s 12-session context-decoding subset, not every ephys session on disk.

## 1-d. How are the data split into trials?

i. Trials are indexed by the raw trial axis of each selected session: `bp/Ntrials` defines the expected length, `goCue` is read as one value per trial, and all boolean trial flags are required to have the same length. Retained trials are then selected by a boolean mask and carried forward as raw 0-based trial indices.

ii. 
```python
n_trials_original = int(np.asarray(handle["obj/bp/Ntrials"][()]).squeeze())
go_cue = np.asarray(handle["obj/bp/ev/goCue"][()]).ravel().astype(np.float64)
flags = trial_flags(handle)
if not all(array.size == n_trials_original for array in flags.values()):
    raise ValueError(f"{session_id}: trial flag length mismatch")
```

```python
use = (
    np.isfinite(go_cue)
    & flags["have_ephys"]
    & ~flags["early"]
    & ~flags["stim"]
)
retained_trials = np.flatnonzero(use)
```

iii. The notes describe this as preserving native trial indexing from the raw files and then masking retained trials after curation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they have a finite go cue, `haveEphys=True`, are not early-lick trials, and are not stimulation trials. Ignore trials are retained. The AI does not implement the human reference’s extra cutoff for trials that continue after recording ended.

ii. 
```python
use = (
    np.isfinite(go_cue)
    & flags["have_ephys"]
    & ~flags["early"]
    & ~flags["stim"]
)
retained_trials = np.flatnonzero(use)
```

iii. `CONVERSION_NOTES.md` says the AI intentionally matched paper-like exclusions for `early` and `stim`, kept ignore trials because the decoder requires them, and used `haveEphys` plus finite go cue as additional validity checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the author-selected ALM probe inside `obj/clu`, specifically each cluster’s `trialtm`, `trial`, and `quality` fields, plus the per-trial `obj/bp/ev/goCue` values for alignment.

ii. 
```python
probe_cells = referenced_objects(handle, handle["obj/clu"], preserve_empty=True)
cluster_group = probe_cells[probe_number - 1]
selected_indices, quality_labels = select_quality_indices(handle, cluster_group)
trial_time_objects = referenced_objects(handle, cluster_group["trialtm"])
trial_number_objects = referenced_objects(handle, cluster_group["trial"])
```

```python
go_cue = np.asarray(handle["obj/bp/ev/goCue"][()]).ravel().astype(np.float64)
```

iii. The notes repeatedly cite the paper loaders and `findClusters` / `alignSpikes` as the basis for using only the author-selected ALM probe and aligning spikes to `goCue`.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes in 5 ms bins on a padded `[-3.0, 2.5)` window, divides by `DT` to convert counts to Hz, applies a custom 15-sample causal Gaussian via `lfilter`, then crops the smoothed result down to the output window `[-2.5, 2.5)`. It does not z-score or baseline-subtract.

ii. 
```python
DT = 0.005
FILTER_TMIN = -3.0
OUTPUT_TMIN = -2.5
TMAX = 2.5
SMOOTH_SAMPLES = 15
```

```python
aligned = trial_time - go_cue[trial_number]
bin_index = np.floor((aligned - FILTER_TMIN) / DT).astype(np.int64)
np.add.at(counts[output_index], (trial_number[valid_bin], bin_index[valid_bin]), 1.0)
rates = reference_smooth(counts / np.float32(DT))
rates = rates[keep][:, :, OUTPUT_MASK]
```

iii. The notes justify this as a direct port of `getSeq.m` and `mySmooth.m`, with extra pre-window padding so the causal filter can warm up before the final `[-2.5, 2.5)` decoder window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, the AI keeps only clusters whose exact case-sensitive quality label is not one of `garbage`, `gabrga`, `noisy`, or `real?`. Second, it computes a condition-averaged firing-rate statistic across seven context-analysis masks and retains only units whose mean is strictly greater than 1 Hz. It also aborts a session if fewer than 10 neurons survive.

ii. 
```python
if label in {"garbage", "gabrga", "noisy", "real?"}:
    continue
```

```python
condition_means = np.zeros((n_quality, len(reference_conditions(flags))), dtype=np.float64)
for condition_index, mask in enumerate(reference_conditions(flags)):
    ...
mean_fr = condition_means.mean(axis=1)
keep = mean_fr > LOW_FR_HZ
```

```python
if rates.shape[0] < 10:
    raise ValueError(f"{session_id}: only {rates.shape[0]} retained neurons")
```

iii. The AI’s notes say this matches `findClusters(..., {'all'})`, the figure-script 1 Hz cutoff, and the paper’s session-level “at least 10 units” rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike is aligned by subtracting that trial’s `goCue` time from the spike’s within-trial time. No interpolation is used for spikes.

ii. 
```python
trial_time = np.asarray(trial_time_objects[cluster_index][()]).ravel().astype(np.float64)
trial_number = np.asarray(trial_number_objects[cluster_index][()]).ravel().astype(np.int64) - 1
aligned = trial_time - go_cue[trial_number]
```

iii. The notes explicitly identify `alignSpikes.m` and say all streams are aligned to native `bp.ev.goCue`, which corresponds to water-drop onset on WC trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final converted neural data are on 5 ms bins with 1,000 output timepoints spanning `[-2.4975, 2.4975]` s. No later rebinning is applied, but the AI does smooth on a larger `[-3.0, 2.5)` grid before cropping to the final 1,000-bin window.

ii. 
```python
DT = 0.005
FILTER_EDGES = np.arange(FILTER_TMIN, TMAX + DT / 2, DT, dtype=np.float64)
FILTER_TIME = (FILTER_EDGES[:-1] + DT / 2).astype(np.float32)
OUTPUT_MASK = (FILTER_TIME >= OUTPUT_TMIN) & (FILTER_TIME < TMAX)
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
N_TIME = int(OUTPUT_TIME.size)
```

iii. The notes say 5 ms matches the paper and figure scripts, while the padded window exists only to support the causal smoother.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw per-trial variable. The AI derives it from the fixed aligned bin-center vector implied by `DT`, `OUTPUT_TMIN`, and `TMAX`, i.e. the common time axis relative to each trial’s go cue.

ii. 
```python
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
...
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
```

iii. The notes say the only decoder input requested was time from the alignment event, so the AI used the shared neural/video output grid itself.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the 5 ms output bin centers once, reshapes them to `(1, N_TIME)`, and reuses that same array for every trial in a session.

ii. 
```python
FILTER_TIME = (FILTER_EDGES[:-1] + DT / 2).astype(np.float32)
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
...
input_trials.append(input_template)
```

iii. `CONVERSION_NOTES.md` says the time input is intentionally minimal and reused by reference to avoid unnecessary pickle bloat.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input vector is exactly the same `OUTPUT_TIME` axis used for the cropped neural rates and the aligned behavioral streams, so the input and neural data share the same 1,000 bins.

ii. 
```python
rates = rates[keep][:, :, OUTPUT_MASK]
...
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
```

iii. The notes describe `OUTPUT_TIME` as the common decoder axis for spikes, video-derived outputs, and the time input.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction directly from the per-trial behavioral flags `obj.bp.no`, `obj.bp.L`, and `obj.bp.R`. It does not use `hit`/`miss` to infer the actual chosen side on error trials.

ii. 
```python
for name in ("L", "R", "hit", "miss", "no", "early", "autowater"):
    flags[name] = np.asarray(handle[f"obj/bp/{name}"][()]).ravel().astype(bool)
```

```python
if flags["no"][source_trial_index]:
    lick_direction = 2
elif flags["L"][source_trial_index]:
    lick_direction = 0
elif flags["R"][source_trial_index]:
    lick_direction = 1
```

iii. The notes say “raw L/R/no flags” were used, with `no` taking precedence so ignore trials become the requested `none` class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It is a direct categorical relabeling: `no -> 2`, otherwise `L -> 0`, `R -> 1`, and the chosen class is repeated across all time bins in the trial.

ii. 
```python
output = np.empty((6, N_TIME), dtype=np.int8)
output[0].fill(lick_direction)
```

iii. The notes justify repeating all static labels over time so all outputs share the same `(6, time)` layout.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived solely from the per-trial `obj.bp.autowater` flag.

ii. 
```python
flags[name] = np.asarray(handle[f"obj/bp/{name}"][()]).ravel().astype(bool)
```

```python
context = 0 if flags["autowater"][source_trial_index] else 1
```

iii. The notes say `autowater` is the direct raw indicator of the WC context, with all other trials treated as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI maps `autowater=True` to WC class `0` and `False` to DR class `1`, then repeats that code across time for the whole trial.

ii. 
```python
context = 0 if flags["autowater"][source_trial_index] else 1
...
output[1].fill(context)
```

iii. The notes explicitly state “WC=0 when true, DR=1 when false.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.miss`, `obj.bp.hit`, and implicitly `obj.bp.no` for the remaining ignore trials. The AI also checks that `hit + miss + no == 1` on every retained trial.

ii. 
```python
outcome_count = (
    flags["hit"].astype(np.int8)
    + flags["miss"].astype(np.int8)
    + flags["no"].astype(np.int8)
)
if not np.all(outcome_count[retained_trials] == 1):
    raise ValueError(...)
```

```python
if flags["miss"][source_trial_index]:
    outcome = 0
elif flags["hit"][source_trial_index]:
    outcome = 1
else:
    outcome = 2
```

iii. The notes say the decoder requires an explicit ignore class, so the AI kept trials where neither `hit` nor `miss` is true and labeled them `ignore`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `miss -> incorrect (0)`, `hit -> correct (1)`, and everything else in retained trials to `ignore (2)`, then repeats that code across time.

ii. 
```python
output[2].fill(outcome)
```

iii. The notes describe this as a direct decoder-specific relabeling of the trial outcome flags.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived only from the side-camera DeepLabCut trajectory feature named `"tongue"` in `obj.traj`, plus that camera’s frame times, the session video offset, and per-trial `goCue`. The AI does not use the bottom-camera tongue feature.

ii. 
```python
tx, ty, tongue_raw = aligned_feature_position(
    handle,
    side_group,
    int(source_trial_index),
    "tongue",
    go_cue[source_trial_index],
    offset,
)
```

```python
names = feature_names(handle, group, trial_index)
feature_index = names.index(feature)
frames = frame_times_for_trial(handle, group, trial_index, ts.shape[0])
aligned_frames = frames - video_offset - go_cue
```

iii. The notes say the AI intentionally used the “side-view central tongue” as the canonical tongue marker.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The side-camera tongue x/y coordinates are linearly interpolated directly onto the 5 ms `OUTPUT_TIME` grid. Velocity is then computed as the Euclidean norm of `np.gradient(x)` and `np.gradient(y)` on that grid, with non-finite gradient samples set to zero. No Gaussian smoothing, no likelihood thresholding, and no two-camera averaging are implemented here.

ii. 
```python
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
y = interpolate_with_nan(aligned_frames, y_raw, OUTPUT_TIME)
```

```python
if tongue:
    x_velocity = np.gradient(x)
    y_velocity = np.gradient(y)
    x_velocity[~np.isfinite(x_velocity)] = 0.0
    y_velocity[~np.isfinite(y_velocity)] = 0.0
return np.hypot(x_velocity, y_velocity), visibility
```

iii. The notes frame this as porting the paper’s x/y-derivative velocity definition while preserving missingness before discretization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After per-trial tongue-speed arrays are built for a session, the AI pools all finite visible samples from that session, computes the 50th percentile, assigns class `0` below the threshold, class `1` at or above it, and class `2` for missing / not visible bins.

ii. 
```python
def discretize_session(values: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, float]:
    valid = visibility & np.isfinite(values)
    ...
    threshold = float(np.percentile(values[valid], 50))
    output = np.full(values.shape, 2, dtype=np.int8)
    output[valid & (values < threshold)] = 0
    output[valid & (values >= threshold)] = 1
```

iii. The notes explicitly say the decoder task overrides the paper’s thresholding and requires a per-session median split with a separate missing-data class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video offset from SpikeGLX bitcode vs. Bpod bit start, subtracts that offset and the trial’s `goCue` from frame times, and interpolates the tongue coordinates onto the same `OUTPUT_TIME` grid used for neural data.

ii. 
```python
def video_offset_seconds(handle: h5py.File) -> float:
    bit_start_neural = np.asarray(handle["obj/sglx/bitcode/bitstart"][()]).ravel()
    sampling_rate = float(np.asarray(handle["obj/sglx/fs"][()]).squeeze())
    bit_start_behavior = np.asarray(handle["obj/bp/ev/bitStart"][()]).ravel()
    return matlab_mode(bit_start_neural) / sampling_rate - matlab_mode(bit_start_behavior)
```

```python
aligned_frames = frames - video_offset - go_cue
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
y = interpolate_with_nan(aligned_frames, y_raw, OUTPUT_TIME)
```

iii. The notes cite `findVideoOffset.m` and say all video-derived streams share the same aligned 5 ms output axis as spikes.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut feature `"top_paw"` in `obj.traj`, along with that view’s frame times, the session video offset, and the trial’s `goCue`.

ii. 
```python
px, py, paw_raw = aligned_feature_position(
    handle,
    bottom_group,
    int(source_trial_index),
    "top_paw",
    go_cue[source_trial_index],
    offset,
)
```

iii. The notes state the AI intentionally used the bottom-view `top_paw` marker as the canonical paw signal.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates bottom-view `top_paw` x/y coordinates onto `OUTPUT_TIME`, nearest-fills missing positions, computes `np.gradient` on the filled signals, subtracts a session-trial baseline derivative, nearest-fills the gradients again, and takes Euclidean speed. There is no Gaussian smoothing and no explicit use of the DLC likelihood channel.

ii. 
```python
x_filled = fill_nearest(x)
y_filled = fill_nearest(y)
base_derivative = np.nanmedian(np.diff(np.column_stack((x_filled, y_filled)), axis=0), axis=0)
x_velocity = np.gradient(x_filled) - base_derivative[0]
y_velocity = np.gradient(y_filled) - base_derivative[0]
x_velocity = fill_nearest(x_velocity)
y_velocity = fill_nearest(y_velocity)
return np.hypot(x_velocity, y_velocity), visibility
```

iii. The notes describe this as following the reference velocity code with missingness retained separately for class `2`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is discretized exactly like tongue speed: per session, using the 50th percentile of finite visible samples, with classes `0` below median, `1` at/above median, and `2` for not visible.

ii. 
```python
paw_class, paw_threshold = discretize_session(paw_speed, paw_visible)
```

iii. The notes say all three dynamic decoder outputs use the requested per-session median thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. As with tongue velocity, the AI aligns paw trajectories by subtracting the session video offset and trial go cue from frame times, then interpolates positions onto the shared 5 ms `OUTPUT_TIME` grid.

ii. 
```python
aligned_frames = frames - video_offset - go_cue
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
y = interpolate_with_nan(aligned_frames, y_raw, OUTPUT_TIME)
```

iii. The notes treat all video-derived signals as living on the same go-cue-centered neural time grid after offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the session’s paired `motionEnergy_<session>.mat` file, using `me.data` as one vector per trial, and uses side-camera `frameTimes` from `obj.traj` for temporal alignment.

ii. 
```python
def load_motion_energy(path: Path) -> list[np.ndarray]:
    motion = loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    data = np.atleast_1d(motion.data).ravel()
    return [np.asarray(item, dtype=np.float64).ravel() for item in data]
```

```python
group = trajectory_group(handle, 0)
frame_object = referenced_at(handle, group["frameTimes"], trial_index)
```

iii. The notes say the AI intentionally used the paired motion-energy file rather than trying to reconstruct motion energy from raw video.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The raw per-frame motion-energy trace is aligned to the side-camera timeline, linearly interpolated onto `OUTPUT_TIME`, then nearest-filled across any NaNs that remain after interpolation. No further smoothing or differentiation is applied.

ii. 
```python
aligned_frames = frames - video_offset - go_cue
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
interpolated = fill_nearest(interpolated)
```

iii. The notes justify this as preserving the reference motion-energy stream while replacing the paper’s manual threshold with the decoder’s per-session median split.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the same session-wise median rule: class `0` below the 50th percentile of finite samples, class `1` at/above it, and class `2` when the aligned motion-energy sample is missing / no video.

ii. 
```python
motion_class, motion_threshold = discretize_session(motion_energy, motion_visible)
```

iii. The notes explicitly state that the supplied `moveThresh` is ignored because the decoder task requires a 50th-percentile threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session video offset and trial go cue from the side-camera frame times and interpolating the per-frame motion-energy values onto the same 5 ms `OUTPUT_TIME` grid as the neural data.

ii. 
```python
me, motion_raw = aligned_motion_energy(
    handle,
    motion_data,
    int(source_trial_index),
    go_cue[source_trial_index],
    offset,
)
```

```python
aligned_frames = frames - video_offset - go_cue
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
```

iii. The notes say motion energy shares the same video-offset correction and go-cue-centered output axis as the tongue and paw streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly preserves missing data as class `2`, but it also imputes in some cases. If a trajectory or motion-energy stream is absent or too short, it returns all-NaN arrays that later become class `2`. If `frameTimes` are missing or malformed, it synthesizes evenly spaced 400 Hz timestamps. For paw and motion energy, it nearest-fills missing aligned samples before speed / discretization; for tongue it leaves NaNs in place and uses the visibility mask to mark missing bins.

ii. 
```python
def frame_times_for_trial(...):
    ...
    return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
```

```python
if feature not in names or not trial_video_is_valid(handle, group, trial_index):
    missing = np.full(N_TIME, np.nan, dtype=np.float64)
    return missing.copy(), missing.copy(), {}
```

```python
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
interpolated = fill_nearest(interpolated)
```

iii. The notes justify class `2` as the explicit representation for missing visibility / no video, and justify synthetic frame times plus nearest filling as reference-style handling for malformed or gappy video streams.

## 11-a. What are the most time-consuming steps of the code?

i. The AI designed the script under the assumption that direct session loading plus neural/video processing per session would be the expensive part, especially spike binning/smoothing and per-trial video alignment. The notes report about 35 seconds total for 12 sessions after optimization.

ii. 
```python
counts = np.zeros((n_quality, n_trials, n_filter_time), dtype=np.float32)
for output_index, cluster_index in enumerate(selected_indices):
    ...
    np.add.at(counts[output_index], (trial_number[valid_bin], bin_index[valid_bin]), 1.0)
rates = reference_smooth(counts / np.float32(DT))
```

```python
for output_trial_index, source_trial_index in enumerate(retained_trials):
    tx, ty, ... = aligned_feature_position(...)
    ...
    me, motion_raw = aligned_motion_energy(...)
```

iii. `CONVERSION_NOTES.md` says the expensive stages were selective HDF5 reads, indexed spike binning, FIR smoothing, and per-trial alignment of DLC/motion-energy streams.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorized some operations already (shared spike tensor, one FIR call, shared input array), but it still leaves trial-by-trial loops over all retained video/motion trials and cluster-by-cluster spike accumulation. Those loops could potentially be reduced further, especially repeated per-trial feature decoding and interpolation.

ii. 
```python
for output_index, cluster_index in enumerate(selected_indices):
    ...
```

```python
for output_trial_index, source_trial_index in enumerate(retained_trials):
    tx, ty, tongue_raw = aligned_feature_position(...)
    ...
    me, motion_raw = aligned_motion_energy(...)
```

iii. The notes argue that the biggest worthwhile vectorizations were already implemented and that heterogeneous per-trial video records limited how far vectorization could go cleanly.

## 11-c. What processing does the code repeat multiple times?

i. The code recomputes several small pieces per trial: it re-reads feature names for each trial in `aligned_feature_position`, re-fetches the side trajectory group on every `aligned_motion_energy` call, and interpolates separate x/y streams independently for tongue and paw on every retained trial. It also appends the same `input_template` object repeatedly rather than constructing distinct arrays.

ii. 
```python
def aligned_feature_position(...):
    names = feature_names(handle, group, trial_index)
    ...
    x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
    y = interpolate_with_nan(aligned_frames, y_raw, OUTPUT_TIME)
```

```python
def aligned_motion_energy(...):
    group = trajectory_group(handle, 0)
    ...
```

```python
input_trials.append(input_template)
```

iii. The notes say the AI intentionally avoided larger repeated work such as recomputing video offsets or rebuilding the time axis, but the code still repeats some per-trial metadata and interpolation work.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores substantial diagnostics that are not used in the saved dataset, including sample rates, retained trial flags, raw aligned tongue/paw/motion traces for one diagnostic trial, thresholds, and quality-label audit info. During conversion it also builds visibility masks and raw continuous movement arrays only to collapse them later into categorical outputs.

ii. 
```python
diagnostic_trial = {
    "source_trial_index": int(source_trial_index),
    "tongue_raw": tongue_raw,
    "paw_raw": paw_raw,
    "motion_raw": motion_raw,
    "tongue_speed": tspeed.astype(np.float32),
    ...
}
```

```python
diagnostics = {
    **neural_info,
    "session_id": session_id,
    "sample_rates": rates[: min(20, rates.shape[0]), retained_trials[0], :].copy(),
    "retained_trials": retained_trials,
    "flags": {name: value[retained_trials].copy() for name, value in flags.items()},
    ...
}
```

iii. The notes present this extra processing as audit / sanity-check support rather than as part of the downstream decoder representation.
