# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 25 fixed-delay and 19 randomized-delay sessions and their selected ALM probes, locates each named `data_structure_*.mat` and companion `motionEnergy_*.mat`, and supports both HDF5/v7.3 and conventional v5 MATLAB files. It processes all 44 selected sessions sequentially.

ii.
```python
SESSION_GROUPS = (
    ("Ephys_Behavior", "fixed_delay", ALM_PROBES),
    ("RandomizedDelay_Ephys_Behavior", "randomized_delay", RANDOMIZED_DELAY_PROBES),
)
available = {_session_key(path): path for path in directory.glob("data_structure_*.mat")}
if h5py.is_hdf5(data_path):
    with h5py.File(data_path, "r") as matfile:
        ...
else:
    obj = loadmat(data_path, squeeze_me=True, struct_as_record=False,
                  variable_names=["obj"])["obj"]
```

iii. The trajectory says the repository's `load*ALMVideo.m` files define the published cohort and probe choices, including two probes for three JEB15 sessions; deliberately omitted randomized-delay files lack released motion-energy companions. Both MATLAB formats were found and exercised.

## 1-b. How are the data split into subjects?

i. The subject is parsed from the portion of each session key before the first underscore. Unique subjects retain first-session occurrence order, and `subject_idx` maps every session back to that list.

ii.
```python
subject, date = key.split("_", 1)
session_subjects.append(subject)
subjects = list(dict.fromkeys(session_subjects))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent relied on the stable `<animal>_<date>` filenames, which it had already used to identify sessions; the trajectory does not give a separate explicit justification for preserving first-occurrence order.

## 1-c. How are the data split into sessions?

i. Each named `data_structure_<animal>_<date>.mat` is one session and becomes one element in each top-level session list. Fixed- and randomized-delay folders are combined but recorded as task variants.

ii.
```python
for directory_name, task_variant, probe_map in SESSION_GROUPS:
    ...
    sessions.append((available[key], directory / f"motionEnergy_{key}.mat",
                     probe_map[key], task_variant))
neural_sessions.append(neural_trials)
```

iii. The trajectory treats the authors' session loaders as the authoritative session definition and reports 44 sessions (25 fixed, 19 randomized).

## 1-d. How are the data split into trials?

i. `obj.bp.Ntrials` defines the trial count. Per-trial behavior arrays and trajectory entries are indexed by the resulting zero-based source trial, while spike `cluster.trial` values are converted from one-based to zero-based indices.

ii.
```python
ntrials = int(_vector(matfile["obj/bp/Ntrials"])[0])
spike_trials = _deref_vector(matfile, cluster_group["trial"], cluster_idx).astype(np.int64) - 1
for out_trial, source_trial in enumerate(selected):
```

iii. The trajectory inspected Bpod and acquisition trial-number mappings and concluded existing bookkeeping entries map one-to-one from trial 1.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes early-lick, photostimulation, and `haveEphys == false` trials. It additionally removes candidate trials whose retained neural population is entirely zero throughout the five-second window. Hit, miss, and ignore trials remain.

ii.
```python
candidate_trials = np.flatnonzero(~early & ~stimulated & have_ephys)
neural_present = np.any(neural_all != 0, axis=(1, 2))
selected = candidate_trials[neural_present[candidate_trials]]
```

iii. Early licks and photostimulation were excluded to follow the paper. The trajectory found erroneous true `haveEphys` flags in recording tails and argued that zero activity across all active units for five seconds is effectively impossible, so all-zero population trials provide direct evidence that ephys had ended. Ignore trials were retained because the requested output includes an ignore class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probes in `obj.clu`: each cluster's spike trial (`trial`), within-trial spike time (`trialtm`), and manual `quality`; `obj.bp.ev.goCue` supplies alignment times.

ii.
```python
spike_trials = _deref_vector(matfile, cluster_group["trial"], cluster_idx).astype(np.int64) - 1
spike_times = _deref_vector(matfile, cluster_group["trialtm"], cluster_idx).astype(np.float64)
go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
```

iii. The trajectory identified these as the fields used by the repository's spike-alignment code and used the session-specific published probe selections.

## 2-b. How is the `neural` data processed?

i. For each retained cluster, spikes are aligned and accumulated into trial-by-time count matrices. Counts are passed through a 15-bin causal Gaussian kernel modeled on `mySmooth.m`, then divided by 5 ms to produce Hz. Selected probes are concatenated; there is no z-scoring or baseline normalization.

ii.
```python
np.add.at(counts, (spike_trials, time_idx), 1.0)
padded = np.concatenate((counts[:, :SMOOTH_BINS], counts), axis=1)
smoothed = convolve(padded, SMOOTH_KERNEL[None, :], mode="same", method="direct")
return smoothed[:, SMOOTH_BINS:] / np.float32(DT)
```

iii. The agent states this preserves the repository's 15-bin causal Gaussian smoother exactly, including its unusual prepending behavior, and follows the paper's conversion to firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It removes clusters whose lower-cased quality is `garbage`, `gabrga`, `noisy`, or `real?`, then retains units whose mean smoothed firing rate over all trials and the go-cue window is strictly above 1 Hz. Unlabelled and `poor` units are retained.

ii.
```python
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
if quality.lower() in REJECTED_QUALITIES:
    continue
mean_fr = neural.mean(axis=(0, 2), dtype=np.float64)
keep = mean_fr > LOW_FR_HZ
```

iii. The trajectory cites `findClusters.m` for the four rejected labels and the paper's published inclusion criterion of firing rate above 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time before bin assignment.

ii.
```python
aligned = spike_times[valid_trial] - go_cue[spike_trials]
time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The trajectory says this matches `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data has 1,000 non-overlapping 5 ms bins from -2.5 to +2.5 seconds. Raw spikes are binned once at this resolution; no later temporal rebinning is performed, though neural rates are smoothed across bins.

ii.
```python
DT = 1.0 / 200.0
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
EDGES = TMIN + np.arange(TIME.size + 1, dtype=np.float64) * DT
```

iii. The agent cites repository parameters `dt=1/200`, `tmin=-2.5`, and `tmax=2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the converter-defined common bin grid relative to the raw per-trial go cue used for alignment, rather than copied from a raw vector.

ii.
```python
TIME = np.arange(...) * DT + TMIN + DT / 2.0
input_trials = [TIME[None, :].copy() for _ in range(selected.size)]
```

iii. The trajectory identifies go cue as the required alignment event and uses the paper's time window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It constructs bin centers at `-2.4975, -2.4925, ..., 2.4975` seconds and copies the same one-row continuous series to every retained trial.

ii.
```python
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
```

iii. The code comment says the requested decoder input is continuous and time-varying; no further transformation is needed.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. `TIME` is the center of the same `EDGES` used to bin go-cue-aligned spikes, so each input sample corresponds to its neural bin.

ii.
```python
EDGES = TMIN + np.arange(TIME.size + 1, dtype=np.float64) * DT
time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The shared grid was deliberately built once for all streams to guarantee alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `obj.bp.hit`, `miss`, `no`, and the instructed right-side flag `R`.

ii.
```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
right_target = _vector(matfile["obj/bp/R"]).astype(bool)
```

iii. The trajectory confirmed the behavioral flags' values and treated actual lick direction as derived from target side plus correctness.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Ignore maps to none (2); a hit maps to the target side; a miss maps to the opposite side. Codes are left 0, right 1, none 2 and are repeated over time.

ii.
```python
if ignore[source_trial]:
    lick_direction, outcome = 2, 2
elif hit[source_trial]:
    lick_direction = 1 if right_target[source_trial] else 0
elif miss[source_trial]:
    lick_direction = 0 if right_target[source_trial] else 1
```

iii. The agent reasoned that hit means the instructed port, miss the opposite port, and ignore no lick, satisfying the requested third class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from the per-trial `obj.bp.autowater` flag.

ii.
```python
water_cued = _vector(matfile["obj/bp/autowater"]).astype(bool)
```

iii. The agent identified autowater as the direct marker of water-cued trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. True autowater becomes WC (0), otherwise DR (1), repeated across the time axis.

ii.
```python
context = 0 if water_cued[source_trial] else 1
output[out_trial, 0:3, :] = np.asarray(
    (lick_direction, context, outcome), dtype=np.int8)[:, None]
```

iii. This is a direct relabeling into the prompt's two named contexts.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, `obj.bp.miss`, and `obj.bp.no`.

ii.
```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
```

iii. The agent checked these mutually exclusive behavioral outcome flags and retained ignore because it is explicitly requested.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss becomes incorrect (0), hit becomes correct (1), and `no` becomes ignore (2); any trial with none of those flags raises an error. The class is constant over time.

ii.
```python
if ignore[source_trial]: outcome = 2
elif hit[source_trial]: outcome = 1
elif miss[source_trial]: outcome = 0
else: raise ValueError(...)
```

iii. This directly implements the requested categorical values and uses the error to expose malformed behavior records.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-view `tongue` feature's x/y coordinates from `obj.traj`, its `frameTimes` and dropped-frame marker, plus video/behavior bitcode fields and the go cue. It does not use the bottom-view `top_tongue` or tracking likelihood explicitly.

ii.
```python
tx, ty, tongue_vis = _trajectory_xy(
    matfile, trajectory_side, source_trial, "tongue", TIME,
    video_shift, go_cue[source_trial])
```

iii. The trajectory focused on repository video fields and visibility/timing, but gives no explicit justification for using only the side tongue; the final implementation presents it as the requested tongue stream.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-view x/y are linearly interpolated directly onto the 5 ms grid. Euclidean speed is computed as the magnitude of `np.gradient(x)` and `np.gradient(y)`; non-finite component gradients are set to zero. There is no coordinate smoothing, division by elapsed time, view scaling, or combination of two views.

ii.
```python
x = _interp(source_time, x_raw, aligned_time)
y = _interp(source_time, y_raw, aligned_time)
xvel = np.gradient(x); yvel = np.gradient(y)
xvel[~np.isfinite(xvel)] = 0.0
yvel[~np.isfinite(yvel)] = 0.0
return np.hypot(xvel, yvel)
```

iii. The docstring claims this matches `findVelocity.m`; the trajectory says it aimed to use the authors' interpolation conventions, but it does not separately justify omitting the second view or time normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The median is computed per session across finite samples marked visible in retained trials. Visible values below it are 0 and values at or above it are 1; all other bins remain 2 (`not_visible`).

ii.
```python
tongue_values = tongue_speed[tongue_visible & np.isfinite(tongue_speed)]
thresholds["tongue_velocity_median"] = float(np.percentile(tongue_values, 50))
output[out_trial, 3, valid] = tongue_speed[out_trial, valid] >= thresholds[...]
```

iii. The agent explicitly follows the requested per-session 50th-percentile threshold and recomputed medians after final trial filtering.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-clock shift is estimated from the modes of acquisition and behavioral bitcode starts. Side-camera frame times are shifted by that value and the trial go cue, then x/y are interpolated at the neural bin centers.

ii.
```python
video_shift = (_mode(matfile["obj/sglx/bitcode/bitstart"]) / fs
               - _mode(matfile["obj/bp/ev/bitStart"]))
source_time = frame_times - video_shift - go_cue
x = _interp(source_time, x_raw, aligned_time)
```

iii. The agent cites `findVideoOffset.m` and uses one session-constant offset to put camera samples and spikes on the same go-cue-relative grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both `top_paw` and `bottom_paw` x/y trajectories from the bottom camera, along with bottom-camera frame times, dropped-frame markers, bitcodes, and go cues.

ii.
```python
for feature in ("top_paw", "bottom_paw"):
    px, py, paw_vis = _trajectory_xy(
        matfile, trajectory_bottom, source_trial, feature, TIME,
        video_shift, go_cue[source_trial])
```

iii. The code comment says averaging both tracked paws avoids privileging either paw; this is the agent's explicit rationale.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's x/y is interpolated to bin centers. Missing coordinates are nearest-filled before gradients; a median frame-difference baseline is computed, and its x component is subtracted from both x and y gradients. Euclidean speeds are then averaged wherever either paw is marked visible.

ii.
```python
x_filled = _nearest_fill(x); y_filled = _nearest_fill(y)
xvel = np.gradient(x_filled) - baseline[0]
yvel = np.gradient(y_filled) - baseline[0]
...
paw_speed[out_trial] = np.divide(summed, count, ..., where=count > 0)
```

iii. The agent states that shared baseline subtraction reproduces a repository peculiarity and that averaging the two paws yields a single stream without privileging one.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A per-session median is taken across finite, visible retained samples. Below median is 0, at/above is 1, and unavailable bins retain 2 (`not_visible`).

ii.
```python
paw_values = paw_speed[paw_visible & np.isfinite(paw_speed)]
thresholds["paw_velocity_median"] = float(np.percentile(paw_values, 50))
```

iii. This directly follows the prompt's per-session 50th-percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera times receive the same session bitcode correction and per-trial go-cue subtraction as tongue data; coordinates are then interpolated onto neural bin centers.

ii.
```python
source_time = frame_times - video_shift - go_cue
x = _interp(source_time, x_raw, aligned_time)
```

iii. The agent intended every stream to share the same 5 ms time axis and correctly uses the camera that supplies the paw features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads the standalone session companion `motionEnergy_<key>.mat`, unwraps one possible nested `data` structure, and pairs each per-frame trace with side-camera frame times plus bitcode/go-cue timing.

ii.
```python
me_obj = loadmat(motion_path, ..., variable_names=["me"])["me"]
motion_data = me_obj.data
if hasattr(motion_data, "data"):
    motion_data = motion_data.data
```

iii. The agent cites `loadMotionEnergy.m` for the nested-wrapper case and requires exactly one trace per Bpod trial.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already-computed trace is linearly interpolated at 5 ms neural bin centers. If any interpolated values exist, out-of-range NaNs are nearest-filled; there is no extra smoothing or spatial computation.

ii.
```python
interp_motion = _interp(source_time, raw_motion, TIME)
if np.isfinite(interp_motion).any():
    interp_motion = _nearest_fill(interp_motion)
```

iii. The agent says the spatial motion-energy computation is already upstream and cites `loadMotionEnergy.m` as filling edge NaNs with the nearest value.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Its per-session median is computed across finite samples on retained trials with video. Below median is 0, at/above is 1, and bins without a usable trace remain 2 (`no_video`).

ii.
```python
motion_values = motion[motion_video & np.isfinite(motion)]
thresholds["motion_energy_median"] = float(np.percentile(motion_values, 50))
```

iii. This is the requested per-session 50th-percentile split and explicit no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session bitcode shift and trial go cue; the motion trace is interpolated onto neural bin centers.

ii.
```python
source_time = frame_times - video_shift - go_cue[source_trial]
interp_motion = _interp(source_time, raw_motion, TIME)
```

iii. The agent reasoned that motion energy has one value per side-camera frame, so the side-camera clock is its correct time base.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Trial bookkeeping vectors are trimmed if long and false-padded if short. Missing/invalid trajectory records return NaNs and false visibility, while absent video leaves movement outputs at category 2. Coordinate gaps for paws and edge gaps in motion energy are nearest-filled. A frame/trace length mismatch leaves motion energy unavailable. All-zero neural tail trials are dropped. Unexpected missing features, outcome flags, or trial-count mismatches raise errors.

ii.
```python
if values.size >= ntrials: return values[:ntrials]
return np.pad(values, (0, ntrials - values.size), constant_values=False)
...
if dropped.size == 0 or not np.isfinite(dropped[0]):
    return nan, nan.copy(), np.zeros(..., dtype=bool)
```

iii. The trajectory investigated a one-entry acquisition overhang and confirmed it was trailing before trimming. It used explicit missing classes where possible, but regarded zero-neural tails as invalid ephys coverage and used nearest filling to reproduce claimed repository behavior.

## 11-a. What are the most time-consuming steps of the code?

i. The implementation repeatedly dereferences MATLAB/HDF5 cluster and trajectory objects, constructs full trial-by-time arrays for every unit, interpolates three movement streams per selected trial, and serializes a multi-gigabyte result. The later decoder training was even more expensive but is not part of conversion.

ii.
```python
for cluster_idx in range(cluster_group["trial"].shape[0]):
    ... units.append(_smooth_counts(counts))
for out_trial, source_trial in enumerate(selected):
    ... _trajectory_xy(...)
```

iii. The trajectory observed that the full dataset has 13.76 million labeled timepoints and described conversion passes as lengthy; it did not provide a formal conversion profile.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Cluster loops, selected-trial loops, two-paw feature loops, and final conversion from dense session arrays to per-trial lists remain. Some are difficult to vectorize because HDF5 references and frame counts are ragged. Spike accumulation within a unit and convolution across all its trials are already vectorized.

ii.
```python
for cluster_idx in range(cluster_group["trial"].shape[0]): ...
for out_trial, source_trial in enumerate(selected): ...
for feature in ("top_paw", "bottom_paw"): ...
np.add.at(counts, (spike_trials, time_idx), 1.0)
```

iii. The agent favored exact handling of variable MATLAB layouts and ragged video trials; the trajectory does not explicitly discuss further loop vectorization.

## 11-c. What processing does the code repeat multiple times?

i. HDF5 and v5 paths duplicate most neural and behavioral logic. Go-cue vectors are re-read inside each neural cluster in the HDF5 path. Each trajectory feature independently decodes metadata and interpolates frames, so bottom-camera time work is repeated for both paws. Full conversion was also rerun several times during debugging, though that is not runtime behavior of the final script.

ii.
```python
for cluster_idx ...:
    go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
...
def _load_behavior(...): ...
def _load_behavior_v5(...): ...
```

iii. Separate loaders were retained to support materially different MATLAB representations. The trajectory prioritized exercising both formats and recomputing session medians after trial-selection fixes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes detailed quality-count dictionaries, multiple diagnostics, retained-trial water-cued values, and visibility/speed arrays used only transiently. `_load_behavior` returns `water_cued[selected]`, but the caller discards it. It also computes both paws even though the reference downstream definition uses only one.

ii.
```python
return _finalize_behavior_outputs(...)
# returns output, water_cued[selected], thresholds
output_all, _, thresholds = _load_behavior(...)
```

iii. Diagnostics are kept for audit metadata; the returned water-cued vector has no stated justification and is redundant because context is already in `output`.
