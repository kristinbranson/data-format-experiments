# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 Figure 8 two-context sessions and one ALM probe per session, all from `Ephys_Behavior`. It directly traverses the MATLAB v7.3/HDF5 datasets and separately loads each paired motion-energy file with SciPy. It therefore does not load the randomized-delay sessions or the other author-listed fixed-delay sessions.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")
CONTEXT_SESSIONS = [("JEB6", "2021-04-18", 2), ...]
with h5py.File(data_path, "r") as handle:
    ...
motion = loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
```

iii. The notes justify this as the exact cohort used by the paper's Figure 8 context analysis (12 sessions), chosen because it contains substantial WC and DR blocks and matches the reported context cohort. The agent explicitly rejected randomized-delay and other sessions as outside that context analysis.

## 1-b. How are the data split into subjects?

i. The subject is the animal string in each hard-coded `(animal, date, probe)` tuple. Unique subjects are accumulated in first-session order and each session gets the corresponding integer index. This produces seven native subject IDs.

ii.
```python
if animal not in subject_lookup:
    subject_lookup[animal] = len(subjects)
    subjects.append(animal)
subject_idx.append(subject_lookup[animal])
```

iii. The notes treat filename/native IDs as authoritative and decline to merge IDs merely to force agreement with the paper's statement of six mice.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date pair is one session and one element of the top-level session lists. Only the 12 selected context sessions are processed.

ii.
```python
for session_index, (animal, date, probe_number) in enumerate(session_specs):
    result = process_session(animal, date, probe_number)
    neural.append(result.neural)
```

iii. The agent says the Figure 8 loaders define the relevant two-context cohort and selected these sessions rather than discovering every usable ephys session.

## 1-d. How are the data split into trials?

i. Trial arrays are indexed by the raw Bpod trial index. `goCue.size` defines the trial count; retained indices select the corresponding neural, behavioral, trajectory, and motion-energy entries. Each retained trial becomes one matrix in each session list.

ii.
```python
n_trials_original = go_cue.size
retained_trials = np.flatnonzero(use)
for output_trial_index, source_trial_index in enumerate(retained_trials):
    neural_trials.append(rates[:, source_trial_index, :])
```

iii. The notes describe the raw streams as explicitly trial-indexed, so no trial-boundary reconstruction is needed.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials with a finite go cue and `haveEphys`, and removes early-lick and stimulation trials. Ignore trials are deliberately retained. It does not apply the reference's last-spiking-trial cutoff.

ii.
```python
use = (np.isfinite(go_cue) & flags["have_ephys"]
       & ~flags["early"] & ~flags["stim"])
retained_trials = np.flatnonzero(use)
```

iii. Early/stimulation removal is attributed to paper curation; ignores are retained because the requested decoder has explicit ignore/none classes. `haveEphys` is treated as the recording-availability guard.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probe's `obj.clu` cluster `trialtm`, `trial`, and `quality` fields, together with `obj.bp.ev.goCue` for alignment.

ii.
```python
trial_time_objects = referenced_objects(handle, cluster_group["trialtm"])
trial_number_objects = referenced_objects(handle, cluster_group["trial"])
aligned = trial_time - go_cue[trial_number]
```

iii. The notes state these are sorted extracellular ALM spikes and that `trialtm` is already on the behavioral trial clock.

## 2-b. How is the `neural` data processed?

i. Spikes are accumulated into 5 ms bins beginning at -3 s, converted to Hz, smoothed with a 15-sample causal Gaussian FIR, and cropped to [-2.5, 2.5). No normalization or baseline subtraction is applied.

ii.
```python
np.add.at(counts[output_index], (trial_number[valid_bin], bin_index[valid_bin]), 1.0)
rates = reference_smooth(counts / np.float32(DT))
rates = rates[keep][:, :, OUTPUT_MASK]
```

iii. The agent says this ports `getSeq`/`mySmooth`, uses a longer prefix to avoid crop-edge artifacts, and matches the paper's causal 15-sample Gaussian processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. On one author-designated ALM probe, exact case-sensitive labels `garbage`, `gabrga`, `noisy`, and `real?` are excluded. A unit is then retained only if its average across seven reference condition means is strictly above 1 Hz. Sessions must retain at least ten units.

ii.
```python
if label in {"garbage", "gabrga", "noisy", "real?"}:
    continue
mean_fr = condition_means.mean(axis=1)
keep = mean_fr > LOW_FR_HZ
```

iii. The agent intended to reproduce `findClusters(..., {'all'})`, Figure 8 condition averaging, and the paper's strict >1 Hz inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before binning.

ii.
```python
aligned = trial_time - go_cue[trial_number]
```

iii. This is described as a direct port of `alignSpikes`; both quantities are on the behavioral clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved data have 1000 bins of 5 ms spanning [-2.5, 2.5). Raw spikes are binned directly at 5 ms; video streams are interpolated to those bin centers.

ii.
```python
DT = 0.005
OUTPUT_TMIN, TMAX = -2.5, 2.5
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
```

iii. The agent cites the single-trial Methods and default reference parameters for 5 ms resolution and the five-second window.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read as a raw variable; it is the fixed center-time vector of the bins defined relative to `bp.ev.goCue`.

ii.
```python
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
```

iii. The agent treats the aligned bin grid itself as the requested continuous decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin edges are converted to centers, cropped to the output window, cast to float32, reshaped to `(1, 1000)`, and shared by all trials.

ii.
```python
FILTER_TIME = (FILTER_EDGES[:-1] + DT / 2).astype(np.float32)
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
```

iii. Sharing one immutable array is documented as a pickle/memory optimization.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact saved neural bins, so corresponding columns describe the same time relative to the cue.

ii.
```python
bin_index = np.floor((aligned - FILTER_TMIN) / DT).astype(np.int64)
input_template = OUTPUT_TIME.reshape(1, -1)
```

iii. The common grid is intended to guarantee column-wise alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The code uses `obj.bp.no`, `obj.bp.L`, and `obj.bp.R`; it loads hit/miss flags but does not use them to determine direction.

ii.
```python
if flags["no"][source_trial_index]: lick_direction = 2
elif flags["L"][source_trial_index]: lick_direction = 0
elif flags["R"][source_trial_index]: lick_direction = 1
```

iii. The notes planned a direct L/R mapping with `no` overriding it. This assumes the L/R flag is the actual lick rather than the instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `no` becomes none (2), otherwise L becomes left (0) and R becomes right (1), and the code is repeated across all time bins. Incorrect trials are not flipped to the opposite of the instructed side.

ii.
```python
output[0].fill(lick_direction)
```

iii. The agent justified this as a direct task-field relabeling and did not document the hit/miss-dependent derivation used by the reference.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from the per-trial `obj.bp.autowater` flag.

ii.
```python
context = 0 if flags["autowater"][source_trial_index] else 1
```

iii. The notes identify autowater as WC and its absence as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is relabeled to WC=0 and DR=1 and repeated over time.

ii.
```python
output[1].fill(context)
```

iii. This directly follows the requested class order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `obj.bp.miss` and `obj.bp.hit`; anything else is treated as ignore. `no` is also loaded and used to validate that exactly one outcome flag is present.

ii.
```python
if flags["miss"][source_trial_index]: outcome = 0
elif flags["hit"][source_trial_index]: outcome = 1
else: outcome = 2
```

iii. The notes describe the three raw flags as mutually exclusive and exhaustive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss becomes incorrect (0), hit correct (1), and neither becomes ignore (2); the code is repeated over time.

ii.
```python
output[2].fill(outcome)
```

iii. This matches the requested output ordering and retains ignore trials intentionally.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-camera `obj.traj` feature `tongue`: x/y coordinates, feature names, frame times and validity metadata. It also uses synchronization bitcodes and go-cue times. The bottom-camera `top_tongue` track is not used.

ii.
```python
tx, ty, tongue_raw = aligned_feature_position(
    handle, side_group, ..., "tongue", go_cue[source_trial_index], offset)
```

iii. The agent calls the side-view central tongue marker canonical and chose one marker rather than combining anatomically distinct landmarks/views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y positions are linearly interpolated onto the 5 ms output centers. `np.gradient` is applied to each coordinate, nonfinite derivatives are set to zero, and Euclidean magnitude is taken. No likelihood threshold, position smoothing, real-time derivative scaling, second view, or cross-view normalization is used.

ii.
```python
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
x_velocity = np.gradient(x); y_velocity = np.gradient(y)
x_velocity[~np.isfinite(x_velocity)] = 0.0
return np.hypot(x_velocity, y_velocity), visibility
```

iii. The notes say positive scaling to pixels/s is irrelevant to median classes and describe this as matching the reference derivative/interpolation path.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One median is computed per session over values marked visible and finite. Values below it are 0, values at/above it are 1, and missing/not-visible bins are 2.

ii.
```python
threshold = float(np.percentile(values[valid], 50))
output[valid & (values < threshold)] = 0
output[valid & (values >= threshold)] = 1
```

iii. The agent follows the explicit per-session 50th-percentile requirement and excludes missing samples from threshold estimation.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset is computed from neural and behavioral bitcode modes. Frame time is transformed as `frameTimes - offset - goCue`, then position is interpolated directly to neural bin centers.

ii.
```python
aligned_frames = frames - video_offset - go_cue
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
```

iii. The notes cite `findVideoOffset` and the reference go-cue alignment procedure.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `obj.traj` feature `top_paw`, its x/y coordinates and frame times, plus synchronization fields and go cues.

ii.
```python
px, py, paw_raw = aligned_feature_position(
    handle, bottom_group, ..., "top_paw", go_cue[source_trial_index], offset)
```

iii. The notes select `top_paw` because it is the reliable paw marker used in paper plotting code.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Interpolated x/y gaps are nearest-filled, coordinate gradients are computed, and a median derivative baseline is subtracted. The code subtracts the x baseline from both x and y gradients, then takes Euclidean magnitude. It does not apply the reference's likelihood-run filtering or Gaussian position smoothing.

ii.
```python
base_derivative = np.nanmedian(np.diff(np.column_stack((x_filled, y_filled)), axis=0), axis=0)
x_velocity = np.gradient(x_filled) - base_derivative[0]
y_velocity = np.gradient(y_filled) - base_derivative[0]
```

iii. The agent states it deliberately preserves the supplied `findVelocity.m` behavior, including the apparent x-baseline subtraction for y.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session median over finite, originally visible values defines class 0 below and class 1 at/above; class 2 represents bins marked not visible.

ii.
```python
paw_class, paw_threshold = discretize_session(paw_speed, paw_visible)
```

iii. This is justified by the explicit decoder threshold and preserving pre-fill missingness.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times receive the same bitcode offset and per-trial go-cue subtraction and are interpolated to the neural bin centers.

ii.
```python
aligned_frames = frames - video_offset - go_cue
```

iii. A common output grid is used for all streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is loaded from the paired standalone `motionEnergy_<animal>_<date>.mat` file, with side-camera frame times and synchronization/go-cue fields used for timing.

ii.
```python
motion = loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
data = np.atleast_1d(motion.data).ravel()
```

iii. The notes prefer the paired file and recognize motion energy as already reduced to one value per frame.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Values are linearly interpolated to 5 ms centers and all resulting gaps/edges are nearest-filled. There is no additional spatial processing or smoothing.

ii.
```python
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
interpolated = fill_nearest(interpolated)
```

iii. The agent says the signal was already computed upstream and follows the author's nearest-fill behavior for valid video trials.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Its per-session finite-sample median gives class 0 below and class 1 at/above; class 2 is reserved for no usable video.

ii.
```python
motion_class, motion_threshold = discretize_session(motion_energy, motion_visible)
```

iii. The requested median is said to supersede the supplied manual `moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are offset-corrected, have the trial go cue subtracted, and are interpolated to the common 5 ms centers.

ii.
```python
aligned_frames = frames - video_offset - go_cue
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
```

iii. Motion energy is assumed to follow the side-camera clock, consistent with the reference loader.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing frame times fall back to a synthetic 400 Hz clock. Invalid video trials or absent features become all class 2. Tongue nonfinite gradients are converted to zero but retain a separate visibility mask; paw coordinates and motion energy are nearest-filled while pre-fill visibility is retained for paw classes. Invalid go cues/ephys, early, and stimulated trials are dropped. Unexpected structures raise errors.

ii.
```python
return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
output = np.full(values.shape, 2, dtype=np.int8)
interpolated = fill_nearest(interpolated)
```

iii. The notes argue that class 2 should preserve missing video, while reference-style filling can supply numeric values within otherwise valid streams.

## 11-a. What are the most time-consuming steps of the code?

i. The agent identifies file loading and per-session neural/video processing as the expensive work, especially spike accumulation/smoothing and per-trial trajectory interpolation. It records per-session timings but does not provide a component benchmark.

ii.
```python
started = time.perf_counter()
...
"processing_seconds": float(elapsed)
```

iii. The notes emphasize direct selective HDF5 reads to avoid materializing large unused MATLAB fields.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. It retains a loop over clusters for ragged spike arrays and a loop over retained trials for ragged video streams/output assembly. Spike bin accumulation, smoothing, and session-wide discretization are vectorized.

ii.
```python
for output_index, cluster_index in enumerate(selected_indices): ...
for output_trial_index, source_trial_index in enumerate(retained_trials): ...
np.add.at(counts[output_index], ...)
```

iii. The notes say ragged per-trial camera data limit useful vectorization and specifically optimized the rectangular neural operations.

## 11-c. What processing does the code repeat multiple times?

i. `feature_names` and trajectory dereferencing are repeated for each trial/feature, and output assembly loops over retained trials a second time after the video loop. `trajectory_group(handle, 0)` is also rediscovered inside every motion-energy alignment call. Session offsets, thresholds, and the time template are computed once.

ii.
```python
names = feature_names(handle, group, trial_index)
group = trajectory_group(handle, 0)
for output_trial_index, source_trial_index in enumerate(retained_trials):
```

iii. The agent's notes claim major shared quantities are reused, but do not call out these smaller repeated dereferences/loops.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes extensive diagnostics (quality labels, mean-rate arrays, sample rates, raw sample-trial tracks, masks, class arrays, counts, and timing) even when plots are disabled; most are discarded with `SessionResult`. It also bins/smooths all original trials before selecting retained trials and computes seven condition means solely for neuron filtering.

ii.
```python
diagnostics = {**neural_info, "sample_rates": ..., "diagnostic_trial": diagnostic_trial, ...}
del result
gc.collect()
```

iii. These intermediates support sanity checks and optional plots, but the saved decoder dataset only retains compact audit metadata, not the diagnostics themselves.
