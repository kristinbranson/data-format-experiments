# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a 12-session two-context cohort from `/app/data/Ephys_Behavior` only. For each session it opens one `data_structure_<animal>_<date>.mat` file with `h5py` and one `motionEnergy_<animal>_<date>.mat` file with `scipy.io.loadmat`, then reads only selected HDF5 fields rather than materializing the full MATLAB object.

ii.
```python
DATA_DIR = APP / "data" / "Ephys_Behavior"
SESSIONS = [
    ("JEB6", "2021-04-18", (2,)),
    ...
    ("JEB19", "2023-04-18", (1,)),
]
```
```python
data_path = DATA_DIR / f"data_structure_{session_id}.mat"
motion_path = DATA_DIR / f"motionEnergy_{session_id}.mat"
with h5py.File(data_path, "r") as f:
    ...
motion_trials = load_motion_file(motion_path, n_trials)
```

iii. In `CONVERSION_NOTES.md`, the agent says it intentionally restricted the cohort to the 12 sessions used by the paper’s two-context scripts because it believed only that subset had genuine alternating WC/DR context, and it says direct HDF5 field access was chosen as a speed and memory optimization over loading whole MATLAB objects.

## 1-b. How are the data split into subjects?

i. Subjects are the animal IDs from the hard-coded `(animal, date, probes)` tuples. A session’s subject is the `animal` field, and `subjects` is the first-seen unique-animal list across sessions.

ii.
```python
subjects = []
...
for index, (animal, date, probes) in enumerate(selected):
    if animal not in subjects:
        subjects.append(animal)
```
```python
subject_idx = np.asarray([subjects.index(animal) for animal, _, _ in selected], dtype=np.int64)
```

iii. The notes justify this as preserving the explicit session IDs from the selected scripts and using the filename/session tuple as the authoritative subject identifier.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSIONS` defines one session, with one animal-date pair and one or more selected probes. Each processed session becomes one element in `neural`, `input`, `output`, and `session_info`.

ii.
```python
def process_session(animal: str, date: str, probes: tuple[int, ...], show_plot: bool):
    session_id = f"{animal}_{date}"
    data_path = DATA_DIR / f"data_structure_{session_id}.mat"
```
```python
selected = SESSIONS[:2] if args.sample else SESSIONS
...
for index, (animal, date, probes) in enumerate(selected):
    n, i, o, info = process_session(animal, date, probes, show)
    neural.append(n)
    inputs.append(i)
    outputs.append(o)
```

iii. The justification in the notes is that the two-context analysis scripts, rather than directory globbing, define the intended cohort.

## 1-d. How are the data split into trials?

i. Trials are native Bpod trials indexed `0..n_trials-1`, where `n_trials` comes from `obj/bp/Ntrials`. Trial-wise Bpod flags, neural spike trial IDs, trajectories, and motion-energy traces are all indexed by this trial number, and retained trials are stored by their original indices.

ii.
```python
n_trials = int(np.asarray(f["obj/bp/Ntrials"])[0, 0])
fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
bp = {name: np.asarray(f[f"obj/bp/{name}"]).ravel().astype(bool) for name in fields}
go = np.asarray(f["obj/bp/ev/goCue"]).ravel().astype(np.float64)
```
```python
retained = np.flatnonzero(~bp["early"] & ~bp["stim"] & np.isfinite(go)
                          & (outcome_sum == 1) & (side_sum == 1))
```

iii. The notes describe Bpod as the source of trial structure and emphasize keeping original source trial indices in metadata for traceability.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are not early-lick trials, not stimulation trials, have finite go-cue time, have exactly one of `hit/miss/no`, and exactly one of `R/L`. Ignore trials are kept if they pass those checks. The code does not add a separate “trial extends past end of neural recording” filter.

ii.
```python
outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
side_sum = bp["R"].astype(int) + bp["L"].astype(int)
retained = np.flatnonzero(~bp["early"] & ~bp["stim"] & np.isfinite(go)
                          & (outcome_sum == 1) & (side_sum == 1))
if retained.size < 2:
    raise ValueError(f"{session_id}: fewer than two retained trials")
```

iii. The notes justify excluding early and stimulation trials from the reference task logic, while keeping valid ignore trials because the decoder task explicitly requires an `ignore` outcome class. The extra one-hot consistency checks were added as sanity checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from selected probes in `obj/clu`, specifically cluster `quality`, per-spike `trial`, and per-spike `trialtm`, together with per-trial `obj/bp/ev/goCue` to align spikes to the event.

ii.
```python
for probe_index, probe_ref in enumerate(f["obj/clu"][:, 0], start=1):
    ...
    quality = h5_string(f, group["quality"][cluster_index, 0]).strip()
    trials = h5_vector(f, group["trial"][cluster_index, 0], np.float64)
    trial_times = h5_vector(f, group["trialtm"][cluster_index, 0], np.float64)
```
```python
aligned = trial_times[valid_trial] - go[tr0]
```

iii. The notes explicitly map `obj.clu{probe}.trialtm`, `obj.clu{probe}.trial`, and `bp.ev.goCue` to the final neural tensor.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, counted into 10 ms bins over `[-2.5, 2.5)`, divided by `DT` to convert counts to spikes/s, then smoothed with a 15-sample causal Gaussian kernel intended to match the paper’s `mySmooth`. Surviving units are stacked as trial × neuron × time, then emitted per trial as neuron × time arrays.

ii.
```python
DT = 0.010
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
EDGES = TMIN + np.arange(N_TIME + 1, dtype=np.float64) * DT
SMOOTH_N = 15
```
```python
counts = bin_one_unit(trials, trial_times, go, n_trials)
rates = smooth_reference(counts / DT).astype(np.float32)
...
neural = np.stack(all_rates, axis=1)
```

iii. The notes say this choice was based on the paper’s two-context choice/context pipeline rather than the other reference implementation, and that the agent deliberately matched the 10 ms, `[-2.5,2.5)`, causal-smoothed setup it found there.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered in two stages: the code drops clusters whose lower-cased quality label is in `{"garbage", "gabrga", "noisy", "real?"}`, then keeps only units with mean smoothed firing rate strictly greater than 1 Hz.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
if quality.lower() in BAD_QUALITIES:
    continue
...
mean_rate = float(rates.mean())
if mean_rate > 1.0:
    all_rates.append(rates)
```

iii. In the notes, the agent argues that the paper’s “quality={'all'}” logic excludes only explicit junk labels, and that the strict `>1 Hz` cutoff should match the paper’s population-analysis inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the go-cue time of its own trial before binning, so time zero is the per-trial go cue.

ii.
```python
tr0 = trials[valid_trial].astype(np.int64) - 1
aligned = trial_times[valid_trial] - go[tr0]
bins = np.searchsorted(EDGES, aligned, side="right") - 1
```

iii. The notes say this is meant to reproduce `alignSpikes` with go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural, input, and time-varying output streams all use a common 10 ms grid with 500 bins over `[-2.5,2.5)`. Neural spikes are binned directly onto that grid; camera-derived streams are interpolated onto that same 10 ms grid.

ii.
```python
DT = 0.010
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size
EDGES = TMIN + np.arange(N_TIME + 1, dtype=np.float64) * DT
```

iii. The agent justifies 10 ms as matching the main choice/context and kinematics code path it chose to follow.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a dedicated raw variable. The input is the fixed session-independent `TIME` vector representing bin centers relative to the chosen go-cue alignment window.

ii.
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
...
time_input = TIME.astype(np.float32)[None, :]
```

iii. The notes describe this as “bin centers” matching the aligned neural time base; the go cue determines the alignment convention, but the stored input is the shared relative-time grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code just casts the `TIME` bin centers to `float32`, adds a singleton input dimension, and copies the same 1 × `N_TIME` array to every retained trial.

ii.
```python
time_input = TIME.astype(np.float32)[None, :]
...
input_trials.append(time_input.copy())
```

iii. The justification in the notes is that the decoder input was intentionally restricted to time only, and the correct representation is the same aligned time axis on every trial.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same time grid used for neural binning: neural spikes are counted with `EDGES`, and the input stores the corresponding `TIME` centers.

ii.
```python
EDGES = TMIN + np.arange(N_TIME + 1, dtype=np.float64) * DT
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
```
```python
bins = np.searchsorted(EDGES, aligned, side="right") - 1
...
time_input = TIME.astype(np.float32)[None, :]
```

iii. The notes state that input and neural streams share one common aligned time base by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from per-trial Bpod flags `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
right, left = bool(bp["R"][trial]), bool(bp["L"][trial])
hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])
```

iii. The notes justify this as reconstructing the realized lick side from instructed side plus behavioral outcome, with `no` used to preserve the requested no-lick class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code emits class `2` for no-response trials, class `1` for right hits or left misses, and class `0` for left hits or right misses. The per-trial label is then broadcast across all time bins.

ii.
```python
if no:
    lick = 2
elif hit:
    lick = 1 if right else 0
else:
    lick = 0 if right else 1
...
output[0] = lick
```

iii. The notes say this preserves all three requested lick-direction classes while keeping a uniform time-by-output tensor.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag in `obj/bp`.

ii.
```python
context = 0 if bool(bp["autowater"][trial]) else 1
```

iii. The notes say the agent interpreted `autowater` as the WC indicator and non-`autowater` trials as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater=True` to WC (`0`) and `False` to DR (`1`), then broadcasts the per-trial label across the full time axis.

ii.
```python
context = 0 if bool(bp["autowater"][trial]) else 1
...
output[1] = context
```

iii. The notes say this matches the requested categorical coding for context.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial Bpod flags `hit`, `miss`, and `no`.

ii.
```python
hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])
outcome = 1 if hit else (0 if miss else 2)
```

iii. The notes justify this as preserving incorrect, correct, and ignore outcomes explicitly for the decoder task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `miss -> 0` (incorrect), `hit -> 1` (correct), and everything else/`no -> 2` (ignore), then repeats that label across all bins in the trial.

ii.
```python
outcome = 1 if hit else (0 if miss else 2)
...
output[2] = outcome
```

iii. The notes say retaining ignore trials was a task-driven exception to the reference analyses that often drop them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera trajectory only: the side-view `tongue` feature in `obj/traj`, using its x/y coordinates and frame times, plus `bp.ev.goCue`, `bp.ev.bitStart`, and `sglx.bitcode.bitstart`/`sglx.fs` for clock correction.

ii.
```python
side = get_traj_group(f, 0)
side_names = traj_feature_names(f, side) if side is not None else []
if "tongue" not in side_names:
    raise ValueError(...)
tongue_ix = side_names.index("tongue")
```
```python
aligned_ft = side_ft - vidshift - go[trial]
x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
y = interp_matlab(aligned_ft, side_ts[:, 1, tongue_ix], target_absolute)
tongue[trial], _ = velocity_from_xy(x, y, tongue=True)
```

iii. The notes say the agent considered side-camera tongue tracking sufficient after alignment and missing-value handling, and emphasized preserving “not visible” periods.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the code clock-corrects side-camera frame times, linearly interpolates tongue x and y onto the 10 ms `TIME` grid, computes speed from finite differences of the interpolated coordinates, preserves NaNs where the tongue is not visible, and later discretizes the resulting values by the session median over retained trials.

ii.
```python
def velocity_from_xy(x: np.ndarray, y: np.ndarray, tongue: bool) -> tuple[np.ndarray, np.ndarray]:
    visible = np.isfinite(x) & np.isfinite(y)
    if tongue:
        xf, yf = x.copy(), y.copy()
    ...
    xv = np.gradient(xf)
    yv = np.gradient(yf)
    if tongue:
        xv[~np.isfinite(xv)] = 0
        yv[~np.isfinite(yv)] = 0
    speed = np.hypot(xv, yv)
    speed[~visible] = np.nan
```

iii. The notes say this was intended as a reference-style position-to-velocity transform after clock correction, with the task-required session-median discretization layered on top.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code computes a per-session median over finite tongue-speed samples from retained trials only, then labels values below the median as `0`, values at or above the median as `1`, and missing samples as `2`.

ii.
```python
def discretize_session(values: np.ndarray, retained: np.ndarray) -> tuple[np.ndarray, float]:
    subset = values[retained]
    finite = np.isfinite(subset)
    threshold = float(np.nanpercentile(subset, 50))
    labels = np.full(subset.shape, 2, dtype=np.int64)
    labels[finite & (subset < threshold)] = 0
    labels[finite & (subset >= threshold)] = 1
```

iii. The notes explicitly justify this as matching the prompt’s `<50th` versus `>=50th` rule, with class `2` reserved for not-visible bins.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code computes a session-wide video shift from SpikeGLX bitcode and Bpod bitStart modes, subtracts that shift and the trial’s go cue from side-camera frame times, interpolates the corrected trajectory onto the same `TIME` grid used by the neural data, and uses that aligned series to compute tongue speed.

ii.
```python
vidshift = matlab_mode(sglx_start) / fs - matlab_mode(bit_start)
...
aligned_ft = side_ft - vidshift - go[trial]
x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
y = interp_matlab(aligned_ft, side_ts[:, 1, tongue_ix], target_absolute)
```

iii. The notes cite `findVideoOffset` as the basis for the clock correction and say all streams were forced onto one common aligned grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera `top_paw` and `bottom_paw` trajectory features, using their x/y coordinates and frame times, plus the same go-cue and video-offset variables used for tongue alignment.

ii.
```python
bottom = get_traj_group(f, 1)
bottom_names = traj_feature_names(f, bottom) if bottom is not None else []
paw_indices = [bottom_names.index(x) for x in ("top_paw", "bottom_paw") if x in bottom_names]
```

iii. The notes justify averaging both bottom-view paw markers as a way to avoid arbitrarily choosing one feature when both are present.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code clock-corrects bottom-camera frame times, interpolates each available paw marker onto the 10 ms aligned grid, nearest-fills missing positions before differentiation, computes speed from gradients with a baseline correction, averages speeds across available paw markers, preserves missing bins as NaN, and later discretizes by the session median.

ii.
```python
if tongue:
    xf, yf = x.copy(), y.copy()
else:
    xf, yf = fill_nearest(x), fill_nearest(y)
...
baseline_x = np.nanmedian(np.diff(np.column_stack((xf, yf)), axis=0), axis=0)[0]
xv -= baseline_x
yv -= baseline_x
speed = np.hypot(xv, yv)
speed[~visible] = np.nan
```
```python
for feature_ix in paw_indices:
    ...
    speed, _ = velocity_from_xy(x, y, tongue=False)
    paw_speeds.append(speed)
...
paw[trial] = np.divide(summed, count, out=np.full(N_TIME, np.nan), where=count > 0)
```

iii. The notes say this was intended to reproduce the paper’s non-tongue nearest-fill behavior while still restoring a “not visible” class from the original missingness mask.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Like tongue velocity, paw velocity is split by a per-session median over finite retained-trial samples: `< median -> 0`, `>= median -> 1`, and missing -> `2`.

ii.
```python
paw_labels, paw_threshold = discretize_session(paw, retained)
...
output[4] = paw_labels[pos]
```

iii. The notes explicitly tie this to the prompt’s required session-median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned by subtracting the same session video shift and the trial’s go cue from bottom-camera frame times, then interpolating onto the common `TIME` grid shared with neural data.

ii.
```python
aligned_ft = bottom_ft - vidshift - go[trial]
x = interp_matlab(aligned_ft, bottom_ts[:, 0, feature_ix], target_absolute)
y = interp_matlab(aligned_ft, bottom_ts[:, 1, feature_ix], target_absolute)
```

iii. The notes say paw, tongue, and motion were all aligned onto the same corrected go-cue-centered time base.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the standalone `motionEnergy_<session>.mat` file, specifically the unwrapped `me["data"]` trial traces.

ii.
```python
def load_motion_file(path: Path, n_trials: int) -> list[np.ndarray | None]:
    loaded = scipy_io.loadmat(path, simplify_cells=True)["me"]
    raw = loaded["data"]
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
```

iii. The notes justify using the standalone file because it exists consistently for the chosen sessions and matches the reference loader pattern.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code unwraps the per-trial motion-energy traces, aligns them to the same corrected time base using side-camera frame times, linearly interpolates them onto the 10 ms `TIME` grid, and later discretizes them with a per-session median over retained trials.

ii.
```python
if me is not None and side_ft is not None:
    aligned_ft = side_ft - vidshift - go[trial]
    motion[trial] = interp_matlab(aligned_ft, me, target_absolute)
elif me is not None:
    fallback = np.arange(1, me.size + 1, dtype=np.float64) / 400.0
    motion[trial] = interp_matlab(fallback - 0.5 - go[trial], me, target_absolute)
```

iii. The notes say the intended behavior was to mimic reference clock correction and then apply the task-mandated median split instead of the paper’s manual movement threshold.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded the same way as tongue and paw velocity: per-session median over finite retained-trial values, with classes `0` for below, `1` for at/above, and `2` for missing/no video.

ii.
```python
motion_labels, motion_threshold = discretize_session(motion, retained)
...
output[5] = motion_labels[pos]
```

iii. The notes explicitly say the prompt’s per-session 50th percentile replaced the raw file’s stored `moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by using side-camera frame times, subtracting the session video shift and trial go cue, and interpolating the result onto the common `TIME` grid. If side-camera frame times are unavailable, the code falls back to a synthetic 400 Hz frame-time grid.

ii.
```python
if me is not None and side_ft is not None:
    aligned_ft = side_ft - vidshift - go[trial]
    motion[trial] = interp_matlab(aligned_ft, me, target_absolute)
elif me is not None:
    fallback = np.arange(1, me.size + 1, dtype=np.float64) / 400.0
    motion[trial] = interp_matlab(fallback - 0.5 - go[trial], me, target_absolute)
```

iii. The notes cite reference motion-energy alignment and say the fallback was added to avoid losing trials with missing frame-time metadata.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code keeps sessions/trials when possible and encodes missing behavior as class `2`. Missing or invalid trajectory/frame-time entries return `None`, which propagates to all-NaN continuous streams and then to class `2` after discretization. Paw coordinates are nearest-filled before differentiation but the original visibility mask is reapplied so unobserved bins still become NaN/class `2`. Motion traces missing from the file become `None`; missing side-camera frame times trigger a synthetic 400 Hz fallback for motion energy.

ii.
```python
if traj is None or trial >= traj["ts"].shape[0]:
    return None, None
...
if frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    return None, None
```
```python
if not good.size:
    return values
...
speed[~visible] = np.nan
```
```python
elif me is not None:
    fallback = np.arange(1, me.size + 1, dtype=np.float64) / 400.0
    motion[trial] = interp_matlab(fallback - 0.5 - go[trial], me, target_absolute)
```

iii. The notes justify this as preserving requested “not visible”/“no video” classes instead of dropping trials, while using fills only where the agent believed the reference kinematics code did so.

## 11-a. What are the most time-consuming steps of the code?

i. The code is organized so that per-session neural loading/binning/smoothing and video-stream loading/interpolation dominate runtime. The session loop calls both `load_neural` and `load_behavior_streams`, and the notes emphasize direct HDF5 access and vectorization as the main runtime optimizations.

ii.
```python
with h5py.File(data_path, "r") as f:
    ...
    neural_all, unit_info, neural_stats = load_neural(f, probes, go, n_trials)
    ...
    tongue, paw, motion, behavior_info = load_behavior_streams(
        f, motion_path, go, n_trials, vidshift
    )
```

iii. In Step 6 and Step 7 of the notes, the agent says loading/session processing were the expensive parts and that HDF5 field access, vectorized spike binning, and shared per-session video processing were introduced to reduce cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes spike binning within a unit, but still loops over clusters, trials, and paw features. The clearest remaining non-vectorized work is the per-trial behavior loop in `load_behavior_streams` and the per-cluster loop in `load_neural`.

ii.
```python
for cluster_index in range(group["quality"].shape[0]):
    ...
    counts = bin_one_unit(trials, trial_times, go, n_trials)
```
```python
for trial in range(n_trials):
    ...
    for feature_ix in paw_indices:
        ...
```

iii. The notes defend the current structure by pointing out variable-length camera sequences and by emphasizing that major neural operations were already vectorized.

## 11-c. What processing does the code repeat multiple times?

i. The code intentionally computes some things once per session and then reuses them: one `TIME` grid, one session video shift, one pass over each session’s neural data, and one median threshold per behavior stream. It still repeats per-trial interpolation and velocity calculations for every trial and feature.

ii.
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
```
```python
vidshift = matlab_mode(sglx_start) / fs - matlab_mode(bit_start)
...
tongue_labels, tongue_threshold = discretize_session(tongue, retained)
paw_labels, paw_threshold = discretize_session(paw, retained)
motion_labels, motion_threshold = discretize_session(motion, retained)
```

iii. The notes present this as deliberate reuse of session-wide quantities rather than accidental recomputation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code carries some extra work that is not needed by the final decoder dataset: optional matplotlib processing plots, detailed `session_info`/`units` bookkeeping, and continuous tongue/paw/motion arrays that are immediately discretized and then discarded from the saved dataset.

ii.
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
...
if show_plot:
    processing_plot(...)
```
```python
session_info = {
    ...
    "units": unit_info,
    **neural_stats,
    **behavior_info,
}
```

iii. The notes justify these extras as validation and audit support rather than as necessary downstream inputs.
