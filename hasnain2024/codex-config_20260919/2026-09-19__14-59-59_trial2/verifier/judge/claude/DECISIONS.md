# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 "context sessions" from the `Ephys_Behavior` folder, not all available sessions. These 12 sessions are hard-coded in the `CONTEXT_SESSIONS` list, taken from the paper's Figure 8 context-analysis loaders. Each session is opened as an HDF5 file using `h5py`. Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`. The AI does not load any randomized-delay sessions.

ii. Session list:
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

CONTEXT_SESSIONS = [
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
]
```

Loading a session:
```python
with h5py.File(data_path, "r") as handle:
    ...
```

iii. The AI reasoned that only the 12 two-context sessions are relevant because the decoder requires WC/DR context labels, and randomized-delay sessions have no WC context. The AI also excluded the remaining fixed-delay sessions (JEB13, JEB14, JEB15) that are not part of the Figure 8 context analysis.

## 1-b. How are the data split into subjects?

i. The animal name is taken from the first element of the session tuple (e.g., "JEB6"). Subjects are accumulated in order of first appearance. The AI identifies 7 subjects from the 12 context sessions.

ii.
```python
if animal not in subject_lookup:
    subject_lookup[animal] = len(subjects)
    subjects.append(animal)
subject_idx.append(subject_lookup[animal])
```

iii. The AI noted that the paper reports 6 mice for the context cohort but 7 native IDs appear in the loader files. The AI preserved the 7 native IDs rather than merging any.

## 1-c. How are the data split into sessions?

i. One session is one entry in `CONTEXT_SESSIONS`, identified by (animal, date, probe_number). Each becomes one element of the neural/input/output lists. This yields 12 sessions total, all from the fixed-delay `Ephys_Behavior` folder.

ii.
```python
sessions = CONTEXT_SESSIONS[:2] if arguments.sample else CONTEXT_SESSIONS
...
for session_index, (animal, date, probe_number) in enumerate(session_specs):
    result = process_session(animal, date, probe_number)
```

iii. The AI argued these are the exact sessions used by the paper's Figure 8 context analysis.

## 1-d. How are the data split into trials?

i. Trials are defined by the number `bp.Ntrials`. Each trial has one go cue event and corresponding per-trial flags. Trial flags are loaded for all trials up to `Ntrials`.

ii.
```python
n_trials_original = int(np.asarray(handle["obj/bp/Ntrials"][()]).squeeze())
go_cue = np.asarray(handle["obj/bp/ev/goCue"][()]).ravel().astype(np.float64)
```

iii. Standard trial definition from the Bpod behavioral table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained if they have a finite go cue, `haveEphys` is true, are not early-lick, and are not photostimulation trials. This is implemented as a boolean mask.

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

iii. The AI follows the paper's exclusion of early-lick and photostimulation trials. Additionally requiring finite goCue and haveEphys is a defensive check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the author-designated ALM probe. Each cluster provides `trialtm` (spike time relative to trial start), `trial` (trial number, 1-based), and `quality` (manual curation label). `bp.ev.goCue` provides the alignment event.

ii.
```python
trial_time = np.asarray(trial_time_objects[cluster_index][()]).ravel().astype(np.float64)
trial_number = np.asarray(trial_number_objects[cluster_index][()]).ravel().astype(np.int64) - 1
aligned = trial_time - go_cue[trial_number]
```

iii. Same source variables as the reference code's `alignSpikes.m` and `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtraction, then counted into 5 ms bins spanning -3.0 to +2.5 s (a wider window for smoothing). Counts are converted to Hz by dividing by the bin width. A causal Gaussian kernel is applied: a 15-sample `gausswin` with the leading half zeroed (matching `mySmooth.m`), implemented as an FIR filter with reflected prefix padding. The output is then cropped to the -2.5 to +2.5 s window (1000 bins).

ii.
```python
def causal_gaussian_kernel(length: int = SMOOTH_SAMPLES) -> np.ndarray:
    kernel = gaussian(length, std=(length - 1) / (2 * 2.5)).astype(np.float32)
    kernel[: length // 2] = 0
    kernel /= kernel.sum()
    return kernel

CAUSAL_KERNEL = causal_gaussian_kernel()
CAUSAL_TAPS = CAUSAL_KERNEL[SMOOTH_SAMPLES // 2 :]

def reference_smooth(rates: np.ndarray) -> np.ndarray:
    prefix = rates[..., :SMOOTH_SAMPLES]
    padded = np.concatenate((prefix, rates), axis=-1)
    filtered = lfilter(CAUSAL_TAPS, np.array([1.0], dtype=np.float32), padded, axis=-1)
    return filtered[..., SMOOTH_SAMPLES:].astype(np.float32, copy=False)
```

iii. The AI explicitly aimed to replicate `mySmooth.m` with its causal kernel and reflected-prefix behavior. Bins from -3.0 to -2.5 s serve as a smoothing buffer to avoid edge artifacts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, the quality label is checked case-sensitively, excluding `garbage`, `gabrga`, `noisy`, and `real?` (matching `findClusters.m` with `quality={'all'}`). Then, the mean firing rate is computed as the condition-averaged rate across seven reference conditions (matching `removeLowFRClusters.m`), and units with rate <= 1 Hz are removed.

ii.
```python
def select_quality_indices(handle, cluster_group):
    ...
    if label in {"garbage", "gabrga", "noisy", "real?"}:
        continue
    ...

# condition-averaged firing rate filtering
condition_means = np.zeros((n_quality, len(reference_conditions(flags))), dtype=np.float64)
for condition_index, mask in enumerate(reference_conditions(flags)):
    trial_indices = np.flatnonzero(mask)
    if trial_indices.size:
        condition_means[:, condition_index] = rates[:, trial_indices, :].mean(axis=(1, 2), dtype=np.float64)
mean_fr = condition_means.mean(axis=1)
keep = mean_fr > LOW_FR_HZ
```

iii. The AI matched `findClusters.m`'s exact case-sensitive exclusion set and `removeLowFRClusters.m`'s condition-averaging logic. The AI does not exclude `poor` quality labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time (`trialtm`) is aligned by subtracting the go cue time for its trial: `aligned = trialtm - goCue[trial]`. This is the same as `alignSpikes.m`.

ii.
```python
aligned = trial_time - go_cue[trial_number]
bin_index = np.floor((aligned - FILTER_TMIN) / DT).astype(np.int64)
```

iii. Standard go-cue alignment as in the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 5 ms bins, 1000 bins spanning -2.5 to +2.5 s around the go cue. The internal processing uses bins from -3.0 to +2.5 s (1100 bins) for smoothing padding, then crops to the output window. No rebinning is applied.

ii.
```python
DT = 0.005
FILTER_TMIN = -3.0
OUTPUT_TMIN = -2.5
TMAX = 2.5
FILTER_EDGES = np.arange(FILTER_TMIN, TMAX + DT / 2, DT, dtype=np.float64)
OUTPUT_MASK = (FILTER_TIME >= OUTPUT_TMIN) & (FILTER_TIME < TMAX)
N_TIME = int(OUTPUT_TIME.size)  # 1000
```

iii. 5 ms matches the paper's `dt=1/200` and the reference `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is synthetically constructed from the bin centers of the output time grid. No raw data variable is used.

ii.
```python
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
```

iii. The input represents time relative to the alignment event, defined by the binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The bin centers are computed from the bin edges and used directly.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the same bin-center vector as the neural data's time axis, so they are inherently aligned.

ii.
```python
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
```

iii. Both share the same 1000-bin grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses three per-trial flags from `obj.bp`: `L` (left instruction), `R` (right instruction), and `no` (no response/ignore). The AI does NOT use `hit` or `miss` to derive lick direction.

ii.
```python
if flags["no"][source_trial_index]:
    lick_direction = 2
elif flags["L"][source_trial_index]:
    lick_direction = 0
elif flags["R"][source_trial_index]:
    lick_direction = 1
```

iii. The AI's CONVERSION_NOTES Step 5 describes the mapping as: "bp.L, bp.R, bp.no -> lick direction: Code left=0, right=1, none=2; no overrides instructed L/R to none."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI directly maps the `L` flag to left (0), `R` flag to right (1), and `no` flag to no lick (2). This effectively assigns the **instructed side** as the lick direction for hit and miss trials, rather than deriving the **actual lick direction** from the combination of instructed side and outcome. For miss trials this is incorrect: a miss on a left-instruction trial means the animal licked right, but the AI labels it as left.

ii.
```python
if flags["no"][source_trial_index]:
    lick_direction = 2
elif flags["L"][source_trial_index]:
    lick_direction = 0
elif flags["R"][source_trial_index]:
    lick_direction = 1
```

iii. The AI did not explicitly discuss the distinction between instructed side and actual lick direction. The code treats L/R as the lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial flag `bp.autowater`. When true, the trial is water-cued (WC); otherwise delayed-response (DR).

ii.
```python
context = 0 if flags["autowater"][source_trial_index] else 1
```

iii. Matches the paper's context definition.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True -> WC (0), autowater=False -> DR (1). The value is repeated across all time bins.

ii.
```python
context = 0 if flags["autowater"][source_trial_index] else 1
output[1].fill(context)
```

iii. Straightforward relabeling.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags: `bp.miss` and `bp.hit`. Trials that are neither hit nor miss are classified as ignore.

ii.
```python
if flags["miss"][source_trial_index]:
    outcome = 0
elif flags["hit"][source_trial_index]:
    outcome = 1
else:
    outcome = 2
```

iii. The three outcome states are mutually exclusive, verified by an explicit check.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: miss -> incorrect (0), hit -> correct (1), otherwise -> ignore (2). Repeated across all time bins.

ii.
```python
output[2].fill(outcome)
```

iii. Matches the standard outcome interpretation.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking from `obj.traj` for the **side camera only** (view index 0), using the feature named `tongue`. Frame times `frameTimes` and the video offset from `sglx.bitcode` are used for alignment. The AI does NOT use the bottom camera's tongue tracking (`top_tongue`).

ii.
```python
tx, ty, tongue_raw = aligned_feature_position(
    handle,
    side_group,  # trajectory_group(handle, 0) - side camera
    int(source_trial_index),
    "tongue",
    go_cue[source_trial_index],
    offset,
)
```

iii. The AI's CONVERSION_NOTES Step 5 describes using "Side-camera DLC feature tongue x/y and frame times."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Frame times are aligned by subtracting the video offset and go cue. (2) DLC x/y positions are linearly interpolated onto the 5 ms output time grid. (3) Speed is computed as `np.hypot(np.gradient(x), np.gradient(y))` on the interpolated grid, with NaN gradients set to 0. (4) Visibility is defined as bins where at least one of x or y is finite after interpolation. (5) Session-wide 50th percentile of visible values is the threshold: below -> 0, at or above -> 1, not visible -> 2.

ii.
```python
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
y = interpolate_with_nan(aligned_frames, y_raw, OUTPUT_TIME)
...
# In feature_speed, tongue=True branch:
x_velocity = np.gradient(x)
y_velocity = np.gradient(y)
x_velocity[~np.isfinite(x_velocity)] = 0.0
y_velocity[~np.isfinite(y_velocity)] = 0.0
return np.hypot(x_velocity, y_velocity), visibility
```

iii. The AI followed the reference code's `findVelocity.m` approach for tongue (no nearest-fill, just gradient with NaN->0). Interpolation onto the output grid replaces the reference's frame-rate processing + binning approach.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile of all visible (finite) tongue speed values across retained trials in the session is the threshold. Values below -> 0, at or above -> 1, not visible -> 2.

ii.
```python
def discretize_session(values, visibility):
    valid = visibility & np.isfinite(values)
    threshold = float(np.percentile(values[valid], 50))
    output = np.full(values.shape, 2, dtype=np.int8)
    output[valid & (values < threshold)] = 0
    output[valid & (values >= threshold)] = 1
    return output, threshold
```

iii. Matches the decoder task specification's per-session 50th percentile threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset (computed from bitcode synchronization) and the trial's go cue. DLC positions are then interpolated from these aligned frame times onto the output 5 ms grid (same grid as neural data).

ii.
```python
aligned_frames = frames - video_offset - go_cue
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
```

iii. Video offset computation matches `findVideoOffset.m`. Interpolation onto the neural time grid ensures alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from `obj.traj` for the **bottom camera** (view index 1), using the feature `top_paw`.

ii.
```python
px, py, paw_raw = aligned_feature_position(
    handle,
    bottom_group,  # trajectory_group(handle, 1) - bottom camera
    int(source_trial_index),
    "top_paw",
    go_cue[source_trial_index],
    offset,
)
```

iii. Same `top_paw` feature as used in the reference code.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same alignment and interpolation as tongue. However, for paw (`tongue=False`), the AI applies nearest-fill to gaps in x/y before computing the gradient. A baseline derivative correction is subtracted. Speed is `np.hypot` of the corrected derivatives.

ii.
```python
# In feature_speed, tongue=False branch:
x_filled = fill_nearest(x)
y_filled = fill_nearest(y)
base_derivative = np.nanmedian(np.diff(np.column_stack((x_filled, y_filled)), axis=0), axis=0)
x_velocity = np.gradient(x_filled) - base_derivative[0]
y_velocity = np.gradient(y_filled) - base_derivative[0]
```

iii. The AI replicated `findVelocity.m`'s paw-specific nearest-fill and baseline correction behavior. Note: the code subtracts `base_derivative[0]` from both x and y velocities, which appears to be a bug in the reference code faithfully reproduced.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 50th percentile of visible values per session -> 0 (below), 1 (at/above), 2 (not visible).

ii.
```python
paw_class, paw_threshold = discretize_session(paw_speed, paw_visible)
```

iii. Same discretization approach for all movement variables.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then positions interpolated onto the output 5 ms grid.

ii. Same `aligned_feature_position` function as tongue.

iii. Same alignment pipeline.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_*.mat` files, loaded via `scipy.io.loadmat`. The per-trial motion energy vector has one value per camera frame.

ii.
```python
def load_motion_energy(path: Path) -> list[np.ndarray]:
    motion = loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    data = np.atleast_1d(motion.data).ravel()
    return [np.asarray(item, dtype=np.float64).ravel() for item in data]
```

iii. The AI loads the per-trial data from the standalone motion energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy values are aligned using the side camera's frame times (corrected by video offset and go cue), interpolated onto the output 5 ms grid, and then nearest-filled to handle edge NaNs. The result is discretized at the session 50th percentile.

ii.
```python
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
interpolated = fill_nearest(interpolated)
```

iii. The AI followed `loadMotionEnergy.m`'s approach of nearest-filling edge values after alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other movement variables: 50th percentile of finite values per session. Below -> 0, at/above -> 1, no video -> 2.

ii.
```python
motion_class, motion_threshold = discretize_session(motion_energy, motion_visible)
```

iii. Same discretization as tongue/paw.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Uses side camera frame times corrected by the video offset and trial go cue, then interpolated onto the output time grid.

ii.
```python
def aligned_motion_energy(...):
    ...
    aligned_frames = frames - video_offset - go_cue
    interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
```

iii. Same offset calculation as the other video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms: (1) Trials with no valid video (`trial_video_is_valid` returns False) get all-NaN positions, which become class 2. (2) If frame times are absent or all-NaN, synthetic frame times at 400 Hz are used. (3) DLC positions are interpolated; NaNs propagate through interpolation where no valid frames exist, leading to class 2. (4) For paw, nearest-fill handles gaps. (5) For tongue, NaN gradients are set to 0 while the visibility mask preserves class 2. (6) A validation check ensures hit/miss/no are mutually exclusive for retained trials.

ii.
```python
def frame_times_for_trial(handle, group, trial_index, n_frames):
    ...
    if times.size == n_frames and np.any(np.isfinite(times)):
        return times
    return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
```

iii. The AI checked for edge cases and handled them defensively.

## 11-a. What are the most time-consuming steps of the code?

i. Loading and processing each session's HDF5 file, including dereferencing objects and extracting neural/video data. The full conversion of 12 sessions takes about 34 seconds.

ii. N/A (timing is implicit in the session processing loop)

iii. The AI noted the conversion is well under the 15-minute threshold.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial video processing loop (`for output_trial_index, source_trial_index in enumerate(retained_trials)`) iterates over all trials to process tongue, paw, and motion energy individually. The per-cluster spike binning loop iterates over quality-selected clusters. These loops could potentially be vectorized, but the variable-length nature of per-trial video data makes this difficult.

ii.
```python
for output_trial_index, source_trial_index in enumerate(retained_trials):
    tx, ty, tongue_raw = aligned_feature_position(...)
    ...
```

iii. The AI used indexed accumulation for spike binning and vectorized FIR filtering, which are the main optimizations.

## 11-c. What processing does the code repeat multiple times?

i. Feature names are re-read for every trial and feature via `feature_names(handle, group, trial_index)`. The reference conditions are computed twice in `load_and_process_neural` (once for condition means, once for the initial call). Frame times for the side camera are loaded separately for motion energy and for tongue tracking.

ii.
```python
names = feature_names(handle, group, trial_index)  # called per trial per feature
```

iii. These are relatively minor redundancies given the overall performance is fast.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive diagnostic information per session (sample rates, quality labels, per-trial raw video data, class distributions, etc.) that is not part of the final pickle output. The `diagnostics` dict is only used for optional processing plots. The code also computes the wider -3.0 to +2.5 s spike bins that are cropped to -2.5 to +2.5 s, though this serves the purpose of avoiding edge artifacts in causal smoothing.

ii.
```python
diagnostics = {
    **neural_info,
    "session_id": session_id,
    "sample_rates": rates[: min(20, rates.shape[0]), retained_trials[0], :].copy(),
    ...
}
```

iii. Diagnostic data is useful for validation but does not enter the final output.
