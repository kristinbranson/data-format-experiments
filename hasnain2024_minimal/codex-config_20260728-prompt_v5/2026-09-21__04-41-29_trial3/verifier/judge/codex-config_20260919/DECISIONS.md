# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Hard-codes 12 fixed-delay `Ephys_Behavior` sessions and one probe per session, opens each v7.3 data file with `h5py`, and separately opens its motion-energy v5 file with `loadmat`. It does not load the remaining fixed-delay or any randomized-delay sessions.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")
SESSIONS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    ...
]
for session_info in SESSIONS:
    session_data, session_meta = convert_session(session_info)
```

iii. The trajectory says the agent believed the first 12 ALM-video sessions were the published DR+WC cohort and later sessions were DR-only; it therefore intentionally restricted the conversion to those 12. It also chose direct HDF5 access after finding the files were MATLAB v7.3.

## 1-b. How are the data split into subjects?

i. Uses the hard-coded `subject` string in each session record, builds subjects in first-seen order, and records each session's integer index. The result has 7 subjects.

ii.
```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The agent treated the selected cohort's subject labels as authoritative and built the required index mapping directly from them.

## 1-c. How are the data split into sessions?

i. Treats each hard-coded subject/date record as one session and appends one element to every session-level list. All files come from only `Ephys_Behavior`; output has 12 sessions.

ii.
```python
data_path = DATA_DIR / f"data_structure_{subject}_{date}.mat"
all_neural.append(session_data["neural"])
```

iii. The trajectory explicitly identifies 12 selected sessions as the intended cohort, rather than discovering all author-included sessions across both directories.

## 1-d. How are the data split into trials?

i. Uses array position as the original zero-based trial number. It constructs `keep_trials` from Bpod-length boolean fields and uses those indices consistently for spikes, behavior, trajectories, and motion energy.

ii.
```python
keep_mask = (~stim_enable) & (~early) & (~autolearn)
keep_trials = np.flatnonzero(keep_mask)
for out_trial_idx, trial_idx in enumerate(keep_trials):
```

iii. The agent inspected the raw Bpod and HDF5 layouts and concluded the per-trial arrays and cluster trial identifiers provide trial membership directly.

## 1-e. How are trials filtered based on quality controls?

i. Drops photostimulation, early-lick, and `autolearn` trials. It does not truncate fields to `Ntrials` and does not drop post-recording trials with no ephys.

ii.
```python
autolearn = as_bool_1d(bp["autolearn"]) if "autolearn" in bp.keys() else np.zeros_like(early)
keep_mask = (~stim_enable) & (~early) & (~autolearn)
```

iii. The final trajectory describes control/non-early trials plus `~autolearn`; the extra filter was intended to retain standard task trials. No justification was given for omitting the recording-end check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Uses the selected `obj/clu` probe's `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue`. Each spike's trial and within-trial time determine its aligned bin.

ii.
```python
qualities = clu_group["quality"]
trials = clu_group["trial"]
trial_times = clu_group["trialtm"]
aligned_times = clu_trial_times - go_cue[clu_trials]
```

iii. The trajectory traced the authors' spike-alignment code and identified these fields as the spike-sorted neural source.

## 2-b. How is the `neural` data processed?

i. Counts spikes into 5-ms bins, converts counts to Hz, applies a custom one-sided/causal 15-sample Gaussian-like convolution with reflected leading padding, and stores float32 rates. It uses only the configured probe.

ii.
```python
counts = np.zeros((keep_trials.size, NT))
np.add.at(counts, (mapped_trials, bin_idx), 1.0)
rates = smooth_causal_reflect(counts / DT).astype(np.float32)
```

iii. The agent aimed to reproduce the MATLAB causal smoothing style and stated this explicitly in its metadata and final response.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Drops exact-case labels `garbage`, `gabrga`, `noisy`, and `real?`, then retains units with mean smoothed rate above 1 Hz. It does not drop `poor` or normalize label case, and requires at least 10 retained units.

ii.
```python
if quality in {"garbage", "gabrga", "noisy", "real?"}:
    continue
if mean_fr > LOW_FR_HZ:
    kept_unit_data.append(rates)
if len(kept_unit_data) < 10:
    raise RuntimeError(...)
```

iii. The trajectory cites the paper's `FR > 1 Hz` criterion and 'all curated qualities,' while retaining the reference code's main garbage/noisy exclusions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Subtracts the go-cue time of the spike's own trial from `trialtm`, then bins the result on the common relative-time grid.

ii.
```python
clu_trials = ...astype(np.int64) - 1
aligned_times = clu_trial_times - go_cue[clu_trials]
```

iii. The agent found that spike times and go cue share the behavioral trial clock, so only subtraction was needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Uses 5-ms bins and no coarser rebinning. The stored interval is -3.0 to +2.5 s (1100 bins), rather than the reference's -2.5 to +2.5 s (1000 bins).

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1.0 / 200.0
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT)
```

iii. Early trajectory notes identified the paper's 5-ms grid and -2.5/+2.5 window, but the implemented lower bound was later changed to -3.0 without a recorded scientific justification.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Derives the input solely from the module-level bin-center grid, not a raw per-trial variable; zero is defined by `bp.ev.goCue` used for alignment.

ii.
```python
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) ...]
```

iii. The agent used elapsed time as the sole requested decoder input and reused one common aligned grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Computes centers of uniformly spaced 5-ms edges and copies the same 1×1100 float32 row for every trial.

ii.
```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
```

iii. This directly represents continuous seconds from go cue at each neural bin.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Uses exactly the same `TIME_BINS` grid and length as neural binning, so each input sample corresponds to the same neural column.

ii.
```python
bin_idx = np.floor((aligned_times - TMIN) / DT).astype(np.int32)
input_trials = [TIME_BINS[np.newaxis, :] ...]
```

iii. The common module-level grid was chosen to guarantee alignment across streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Uses `bp.R`, `bp.L`, `bp.hit`, and `bp.miss` (and reads `bp.no`, though it is unused).

ii.
```python
r = as_bool_1d(bp["R"]); l = as_bool_1d(bp["L"])
hit = as_bool_1d(bp["hit"]); miss = as_bool_1d(bp["miss"])
no = as_bool_1d(bp["no"])
```

iii. The trajectory explicitly investigated whether R/L encoded instruction or response before combining them with hit/miss.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Sets correct trials to the instructed side, errors to the opposite side, and all other trials to `none`; repeats the per-trial category across time.

ii.
```python
right_choice = (r & hit) | (l & miss)
left_choice = (l & hit) | (r & miss)
lick_direction = np.full(keep_trials.size, 2)
```

iii. The agent resolved the field ambiguity using the authors' decoding labels and encoded the actual lick direction, including no-response trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derives behavioral context from `bp.autowater`.

ii.
```python
autowater = as_bool_1d(bp["autowater"])
```

iii. The agent interpreted autowater trials as water-cued (WC) and the rest as delayed-response (DR), consistent with the task.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Maps autowater true to WC=0 and false to DR=1, then broadcasts the label across time.

ii.
```python
context = np.where(autowater[keep_trials], 0, 1).astype(np.int64)
```

iii. This is the selected cohort's context definition and matches the requested categorical output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derives outcome from `bp.hit` and `bp.miss`; `bp.no` is loaded but not used explicitly.

ii.
```python
hit = as_bool_1d(bp["hit"])
miss = as_bool_1d(bp["miss"])
no = as_bool_1d(bp["no"])
```

iii. The trajectory identified hit/miss/no as the raw outcome fields.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Defaults to ignore=2, overwrites misses with incorrect=0 and hits with correct=1, then broadcasts across time.

ii.
```python
outcome = np.full(keep_trials.size, 2, dtype=np.int64)
outcome[miss[keep_trials]] = 0
outcome[hit[keep_trials]] = 1
```

iii. Defaulting non-hit/non-miss trials to ignore makes explicit use of the mutually exclusive task outcome convention.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Uses `obj.traj` side and bottom views: feature names, x/y coordinates, and frame times. It requests several tongue landmarks per view, and also uses go cue and bitcode fields for timing. Likelihood is not read directly.

ii.
```python
["tongue", "left_tongue", "right_tongue"]
["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]
```

iii. The agent sought to combine both camera views and preserve missing visibility; it assumed invalid tracking was already represented by nonfinite x/y.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Interpolates each landmark's contiguous finite x/y runs onto the 5-ms grid, nearest-fills within the interpolated array before `np.gradient`, computes pixel displacement per bin (not per second), averages visible landmarks and then raw side/bottom speeds. It does not apply the reference's 5-ms Gaussian position smoothing or per-view 90th-percentile normalization.

ii.
```python
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
vel = np.gradient(filled_xy, axis=0)
speed = np.sqrt((vel**2).sum(axis=1))
tongue_speed[...]= average_over_visible(tongue_speeds, tongue_vis)
```

iii. The trajectory says trajectories were linearly interpolated to the neural grid and combined across views; no justification was given for omitting smoothing, time-scaled derivatives, or view normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Pools visible tongue-speed bins within each session, uses their 50th percentile, maps below to 0, at/above to 1, and invisible to 2.

ii.
```python
tongue_threshold = float(np.nanpercentile(tongue_speed[tongue_visible], 50))
tongue_disc[tongue_visible & (tongue_speed < tongue_threshold)] = 0
```

iii. The prompt explicitly requests a per-session median split and a `not visible` class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Computes a session video-clock shift as median recording bitstart/fs minus median behavioral bitStart, subtracts it and the trial go cue from frame times, then interpolates onto neural bin centers.

ii.
```python
return float(np.nanmedian(bitcode_start / fs) - np.nanmedian(bit_start))
frame_times = frame_times - video_shift - align_time
```

iii. The agent followed the paper's shared-bitcode clock correction, choosing medians rather than the reference solution's modes.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Uses bottom-view trajectory x/y and frame times for both `top_paw` and `bottom_paw`, plus clock-alignment fields. Likelihood is not explicitly read.

ii.
```python
["top_paw", "bottom_paw"]
```

iii. The agent chose to aggregate available paw landmarks, apparently to maximize coverage.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Uses the same interpolation/nearest-fill/gradient pipeline as tongue and averages the two paw landmarks. It omits likelihood thresholding, Gaussian position smoothing, derivatives with real time, and the reference's choice to use only reliable `top_paw`.

ii.
```python
paw_speed_trial, paw_visible_trial, _ = get_feature_positions(...,
    ["top_paw", "bottom_paw"], ...)
```

iii. The trajectory describes linear interpolation with visibility preserved but gives no paper-based justification for combining two distinct paws.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Uses the session median of visible paw-speed bins; below=0, at/above=1, invisible=2.

ii.
```python
paw_threshold = float(np.nanpercentile(paw_speed[paw_visible], 50))
```

iii. This implements the prompt's per-session median categories.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Subtracts the same median bitcode-based session shift and go cue, then interpolates bottom-camera tracking onto `TIME_BINS`.

ii.
```python
frame_times = frame_times - video_shift - align_time
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
```

iii. The shared shift/grid were intended to align video and neural streams exactly.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loads `motionEnergy_<subject>_<date>.mat` via `loadmat(...)["me"]["data"]` and pairs each trial trace with side-camera frame times.

ii.
```python
me_raw = loadmat(me_path, simplify_cells=True)["me"]["data"]
me_trial = np.asarray(me_raw[trial_idx], dtype=np.float64).reshape(-1)
```

iii. The agent used the standalone motion-energy file as the precomputed per-frame source.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Linearly interpolates motion energy to bin centers and nearest-fills extrapolated edge gaps. If frame times mismatch or are mostly NaN, it fabricates fallback 400-Hz times from frame count. The reference instead averages frames within bins without filling.

ii.
```python
aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
fallback_times = np.arange(1, side_ts.shape[0] + 1) / 400.0
fallback_times = fallback_times - 0.5 - align_time
```

iii. The trajectory added the fallback specifically to eliminate a residual `no_video` class, citing a perceived MATLAB fallback; it favored complete output over retaining missingness.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Uses the session median over available interpolated energy; below=0, at/above=1, unavailable=2.

ii.
```python
me_threshold = float(np.nanpercentile(motion_energy[motion_available], 50))
```

iii. This follows the requested per-session threshold, though nearest filling and fallback timing make nearly all bins available.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Uses side-camera frame times corrected by median video shift and trial go cue, with the synthetic fallback when needed, then interpolates to neural centers.

ii.
```python
frame_times = frame_times - video_shift - align_time
aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
```

iii. The trajectory traced a remaining all-NaN frame-time trial and added fallback timing to force alignment rather than mark it missing.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Represents nonfinite tongue/paw bins as category 2, but interpolates within visible runs and nearest-fills positions before differentiation. Motion-energy gaps are edge-filled; invalid/mismatched timestamps may be replaced by synthetic 400-Hz timestamps. It raises on fewer than two trials or ten units.

ii.
```python
return fill_nearest_1d(out)
if frame_times.size != me_trial.size or np.isfinite(frame_times).sum() < 2:
    fallback_times = ...
```

iii. The agent repeatedly patched missing-data cases to remove warnings and a tiny `no_video` class, aiming for decoder-complete streams. This conflicts with the reference rationale that missing video should remain explicitly missing.

## 11-a. What are the most time-consuming steps of the code?

i. The code performs expensive per-session HDF5 reads twice (neural/labels, then video), recursively dereferences many per-trial cells, and loops over units, trials, features, and time rows for convolution. The trajectory also found decoder training/PCA expensive, but that is outside conversion.

ii.
```python
with h5py.File(data_path, "r") as h5file: ...
with h5py.File(data_path, "r") as h5file: ...
for clu_idx in range(qualities.shape[0]):
```

iii. The agent did not explicitly profile conversion. Its progress notes identify full data conversion and later session-wise PCA/training as long-running stages.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Unit convolution is looped row-by-row, video processing loops through every trial and feature, interpolation loops dimensions, and output assembly loops trials. Some can be batched/vectorized; ragged HDF5 video data limits full vectorization.

ii.
```python
for i in range(padded.shape[0]):
    out[i] = np.convolve(...)
for out_trial_idx, trial_idx in enumerate(keep_trials):
for feature_name in feature_names:
```

iii. No explicit trajectory justification addresses vectorization; the implementation favors straightforward handling of ragged MATLAB cells.

## 11-c. What processing does the code repeat multiple times?

i. Reopens every session data file in `compute_video_outputs`, rereads go cue and trajectory references, repeatedly decodes each trial's feature names/arrays for tongue and paw calls, and creates an identical time input per trial.

ii.
```python
with h5py.File(data_path, "r") as h5file:  # convert_session
...
with h5py.File(data_path, "r") as h5file:  # compute_video_outputs
```

iii. The agent separated neural/behavior and video conversion for implementation clarity; the trajectory does not justify the duplicate I/O.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Reads `bp.L` and `bp.no` without using their values independently (`L` is redundant given R; `no` is entirely unused), returns an all-ones third array from `get_feature_positions` that callers discard, computes/nearest-fills coordinate values beyond visibility for gradients, and repeats static per-trial labels at every time bin because the decoder format allows time-varying outputs.

ii.
```python
no = as_bool_1d(bp["no"])
return mean_speed, any_visible, np.ones(TIME_BINS.shape, dtype=bool)
side_tongue_speed, side_tongue_visible, _ = ...
```

iii. The trajectory focused on validator compatibility and time-varying output matrices; it did not identify these discarded intermediates as optimization targets.
