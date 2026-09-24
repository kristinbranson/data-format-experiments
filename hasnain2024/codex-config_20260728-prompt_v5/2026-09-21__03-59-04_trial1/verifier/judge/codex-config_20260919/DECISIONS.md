# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the authors’ uncommented `load*_ALMVideo.m` manifests, locates the corresponding session file in either ephys cohort, and loads every curated session through separate MATLAB v7.3/HDF5 and v5 readers. Motion energy is loaded from the adjacent `motionEnergy_<subject>_<date>.mat` file.

ii.
```python
for manifest in sorted(MANIFEST_ROOT.glob("load*_ALMVideo.m")):
    lines = [line for line in manifest.read_text().splitlines()
             if not line.lstrip().startswith("%")]
...
def load_raw_session(path: Path) -> dict:
    if h5py.is_hdf5(path):
        return load_session_v73(path)
    return load_session_v5(path)
```

iii. The AI says manifest-driven selection reproduces the paper’s curated ALM sessions, avoids behavior-only or commented-out recordings, and normalizes heterogeneous MAT layouts before shared processing.

## 1-b. How are the data split into subjects?

i. The subject is parsed from each manifest filename/session specification. Unique subject strings are sorted and each session receives an index into that list.

ii.
```python
subject = manifest.stem.replace("load", "").replace("_ALMVideo", "")
subjects = sorted({sess["subject"] for sess in processed_sessions})
"subject_idx": np.array([subject_to_idx[sess["subject"]]
                         for sess in processed_sessions], dtype=np.int64),
```

iii. The manifests and filenames consistently identify the animal; the AI reports 14 subjects across 44 sessions.

## 1-c. How are the data split into sessions?

i. Each uncommented manifest date/probe entry whose file exists becomes one `SessionSpec` and then one element of each session-level output list. Sessions are searched in the fixed-delay and randomized-delay cohorts.

ii.
```python
for date, probe_text in zip(dates, probes):
    for cohort in ("Ephys_Behavior", "RandomizedDelay_Ephys_Behavior"):
        data_path = DATA_ROOT / cohort / f"data_structure_{subject}_{date}.mat"
        if data_path.exists():
            sessions.append(SessionSpec(...))
```

iii. The AI treats the authors’ loading scripts as the curated session definition and reports 25 fixed-delay plus 19 randomized-delay sessions.

## 1-d. How are the data split into trials?

i. Raw loaders read `bp.Ntrials` and construct trial-indexed behavioral/video structures. Neural spikes retain their 1-based `unit["trial"]` labels, which are converted to zero-based indices; final session lists contain one matrix per retained trial.

ii.
```python
ntrials = int(np.asarray(bp["Ntrials"][()], dtype=float).reshape(-1)[0])
trial_idx = unit["trial"] - 1
for raw_trial_idx in valid_trials.tolist():
    inputs.append(inp)
    outputs.append(out)
```

iii. The AI relies on explicit Bpod trial fields and spike trial labels rather than reconstructing boundaries.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials that are not early-lick trials and have a finite go cue, then removes all-zero neural trials. It does not remove `stim_enable` trials. Sessions with fewer than two retained trials or ten retained units are skipped.

ii.
```python
candidate_trials = np.flatnonzero(
    ~raw["bp"]["early"] & np.isfinite(raw["bp"]["ev"]["goCue"]))
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials])
...
if len(neural_trials) < 2 or len(keep_unit_idx) < 10:
    return None, info
```

iii. Early trials were excluded because the paper omits them. The all-zero tail removal was justified as eliminating behavioral trials after neural recording ended. The notes call this “early_lick_only”; no justification is given for retaining photostimulation trials despite noting that reference trial logic commonly excludes them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from manifest-selected `obj.clu` probe clusters, principally `quality`, `trial`, and `trialtm`, together with `bp.ev.goCue` for alignment and probe location metadata for region labels.

ii.
```python
trial_idx = unit["trial"] - 1
aligned = unit["trialtm"] - align_times[trial_idx]
...
if matlab_quality_ok(unit["quality"]):
    units.append(unit)
```

iii. The AI selected only manifest-designated ALM probes, including concatenating dual-ALM probes, to follow the authors’ curation.

## 2-b. How is the `neural` data processed?

i. Per-unit spikes are aligned, counted into 10 ms bins, divided by 10 ms to form Hz, causally smoothed with a length-15 half-Gaussian and reflected padding, cast to `float32`, and assembled as neuron-by-time trial matrices.

ii.
```python
DT = 1 / 100
np.add.at(counts, (mapped_trials, bin_idx), 1.0)
fr = smooth_causal_reflect((counts / DT).T, SMOOTH_N, BCTYPE).T
keep_mats.append(fr.astype(np.float32))
```

iii. The AI chose 10 ms because it believed most figure/decoder scripts use it, and described the causal reflected Gaussian as reference-style processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes quality labels `garbage`, `gabrga`, `noisy`, and `real?`, then retains units whose mean smoothed rate in retained trials/window exceeds 1 Hz. It includes multiunits and does not exclude `poor`.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
...
if mean_fr > LOW_FR_HZ:
    keep_idx.append(unit_idx)
```

iii. The quality list was taken from `findClusters`; the 1 Hz cutoff and inclusion of multiunits were justified from the methods’ “all units” statement. The full run retained 2,456 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike’s trial-relative time is shifted by that trial’s go-cue time before bin assignment.

ii.
```python
trial_idx = unit["trial"] - 1
aligned = unit["trialtm"] - align_times[trial_idx]
```

iii. The AI identifies this as the same subtraction performed by the reference `alignSpikes.m` logic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 10 ms (500 bins over `[-2.5, 2.5)` seconds). Raw spike events are binned directly at 10 ms; no later rebinning is applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 100
edges = np.arange(TMIN, TMAX + DT, DT)
```

iii. The AI acknowledged a 5-versus-10 ms discrepancy in the reference repository and chose 10 ms because it considered that grid more common in figure/decoder scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common bin-center grid relative to the go cue, not a separately sampled raw variable; `bp.ev.goCue` defines the zero used to align spikes and video.

ii.
```python
def build_time_axis() -> np.ndarray:
    edges = np.arange(TMIN, TMAX + DT, DT)
    return edges[:-1] + DT / 2
```

iii. The AI chose bin centers so the input directly describes time relative to the required alignment event.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It computes centers of uniform 10 ms bins from -2.5 to +2.5 seconds and repeats the same `1 x 500` float32 vector for every trial.

ii.
```python
inp = time_axis[np.newaxis, :].astype(np.float32)
```

iii. The notes say the common 10 ms grid supports uniform tensors across all task variants.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Neural spike counts use edges corresponding to the same grid whose centers form the input, so each input sample describes the matching neural bin.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
bin_idx = np.floor((aligned[mask] - TMIN) / DT).astype(int)
```

iii. The AI independently checked that saved inputs equal the centered grid and that the go cue falls at the expected location.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
right_choice = (bp["R"][trial_idx] and bp["hit"][trial_idx]) or \
               (bp["L"][trial_idx] and bp["miss"][trial_idx])
left_choice = (bp["L"][trial_idx] and bp["hit"][trial_idx]) or \
              (bp["R"][trial_idx] and bp["miss"][trial_idx])
```

iii. The instructed side plus correctness recovers actual lick side; `no` represents no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials map to the instructed side, misses to the opposite side, and ignores to none. Codes are left 0, right 1, none 2 and are repeated across time.

ii.
```python
if bp["no"][trial_idx]: return 2
if left_choice: return 0
if right_choice: return 1
return 2
```

iii. The AI explicitly chose actual behavioral response rather than instructed/rewarded side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived solely from `bp.autowater`.

ii.
```python
def context_value(bp: dict, trial_idx: int) -> int:
    return 1 if bp["autowater"][trial_idx] else 0
```

iii. The AI interprets autowater trials as WC and all others as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It maps DR to 0 and WC to 1, then repeats the per-trial value over all time bins.

ii.
```python
OUTPUT_VALUES[1] = ["DR", "WC"]
out[1, :] = context_value(raw["bp"], raw_trial_idx)
```

iii. The mapping is a direct relabeling; the chosen numeric order follows the AI’s declared `output_values`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.miss` and `bp.hit`, with all remaining cases (including `bp.no`) treated as ignore.

ii.
```python
if bp["miss"][trial_idx]: return 0
if bp["hit"][trial_idx]: return 1
return 2
```

iii. The three outcome flags are mutually exclusive; the AI retains ignores because the requested output includes that class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss maps to incorrect 0, hit to correct 1, and neither to ignore 2; the label is repeated over time.

ii.
```python
OUTPUT_VALUES[2] = ["incorrect", "correct", "ignore"]
out[2, :] = outcome_value(raw["bp"], raw_trial_idx)
```

iii. This directly follows the requested class semantics.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-view `tongue` feature’s x/y coordinates from `obj.traj`, its `frameTimes` and `NdroppedFrames`, plus bitcode clocks and go-cue times. Unlike the reference, it does not use bottom-view `top_tongue`.

ii.
```python
tongue_x, tongue_y, tongue_vis = align_feature(
    raw, time_axis, align_times, 0, "tongue")
tongue_speed = compute_speed(tongue_x, tongue_y, "tongue")
```

iii. The notes planned side-view tongue tracking and preservation of raw missingness to create the explicit not-visible class.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-view x/y are linearly interpolated onto the 10 ms grid separately within finite runs. Visibility is finite interpolated x and y. Velocity is the Euclidean magnitude of `np.gradient` across grid samples; it is not divided by seconds, smoothed, or combined/normalized across cameras.

ii.
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx],
                        ts[:, 0], time_axis)
...
xvel = np.gradient(xx)
yvel = np.gradient(yy)
speed[:, tr_idx] = np.sqrt(xvel ** 2 + yvel ** 2)
```

iii. The AI called this reference-style alignment/velocity processing and prioritized keeping tongue invisibility instead of filling it.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single session median is computed over visible finite speed samples. Values below it are 0, values at or above it are 1, and invisible/nonfinite samples are 2.

ii.
```python
thresh = float(np.nanpercentile(valid_vals, 50))
out[low_mask] = 0
out[high_mask] = 1
```

iii. This implements the requested per-session 50th-percentile threshold and explicit not-visible category.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-to-behavior offset derived from bitcode modes and the trial go cue are subtracted from frame times, then x/y are interpolated to the common neural bin centers.

ii.
```python
vid_file_offset = mode_float(raw["sglx_bitstart"]) / raw["sglx_fs"]
return vid_file_offset - bit_start
...
frame_times - vidshift - align_times[tr_idx]
```

iii. The AI states this follows `findVideoOffset` and puts all streams on the same go-cue-centered grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-view `top_paw` and `bottom_paw` x/y trajectories, frame metadata, video/behavior bitcode clocks, and go cues.

ii.
```python
paw_feats = ["top_paw", "bottom_paw"]
for feat in paw_feats:
    x, y, vis = align_feature(raw, time_axis, align_times, 1, feat)
```

iii. The AI says paws are bottom-view features and chose the maximum visible speed of both markers to represent paw movement.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each marker is interpolated to the grid; missing aligned x/y values are nearest-filled. Gradients are computed, an unusual scalar x-baseline is subtracted from both x and y velocity, speeds are formed, and the maximum of the two paw speeds is used per bin.

ii.
```python
x = fill_nearest_1d(x); y = fill_nearest_1d(y)
baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
xvel = xvel - baseline
yvel = yvel - baseline
...
paw_speed = np.max(np.where(paw_finite, paw_stack, -np.inf), axis=0)
```

iii. The AI considered max-over-two-markers a useful aggregate and nearest filling consistent with reference treatment of non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session median over samples marked visible and finite splits low (0) from high (1); bins where neither raw marker was visible are class 2.

ii.
```python
paw_visible = np.any(np.stack(paw_vis, axis=0), axis=0)
paw_cat, paw_thr = discretize_visible_signal(paw_speed, paw_visible)
```

iii. The median and third visibility class follow the decoder specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are corrected by the session video offset and trial go cue, then both marker coordinates are interpolated onto neural bin centers.

ii.
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx],
                        ts[:, 0], time_axis + ADVANCE_MOVEMENT)
```

iii. The AI used the same clock correction and common grid as for tongue and motion energy.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is loaded from the standalone per-session `motionEnergy_*.mat` file and paired with side-camera frame times, session clock fields, and go cues. Nested `me.data` wrappers are unwrapped.

ii.
```python
me_path = spec.data_path.with_name(
    f"motionEnergy_{spec.subject}_{spec.date}.mat")
while hasattr(payload, "_fieldnames") and "data" in payload._fieldnames:
    payload = payload.data
```

iii. The AI notes several wrapper layouts and chose the standalone files used by the reference loader. Although planning mentioned embedded `obj.me`, the implementation does not fall back to it.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already spatially reduced signal is linearly interpolated onto 10 ms centers and nearest-filled. No smoothing or differentiation is applied.

ii.
```python
sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
aligned[:, tr_idx] = fill_nearest_1d(sig)
```

iii. The AI considered motion energy already processed upstream and therefore only resampled it, matching its reading of `loadMotionEnergy` edge filling.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session median over all finite aligned values splits low (0) and high (1); unavailable/nonfinite samples remain class 2.

ii.
```python
thresh = float(np.nanpercentile(vals, 50))
out[valid & (me_aligned < thresh)] = 0
out[valid & (me_aligned >= thresh)] = 1
```

iii. The manual movement threshold stored in the file is deliberately ignored because the prompt requests the 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Normally it uses side-camera frame times corrected by video offset and go cue, interpolated to the neural grid. If frame times are absent/nonfinite, it fabricates a 400 Hz sequence and applies a hard-coded 0.5 s offset.

ii.
```python
if frame_times.size == 0 or not np.all(np.isfinite(frame_times)):
    frame_times = (np.arange(y.size) + 1) / 400.0
    tt = frame_times - 0.5 - align_times[tr_idx]
else:
    tt = frame_times - vidshift - align_times[tr_idx]
```

iii. The normal path is justified by reference video-offset logic; the fallback is presented as robustness for missing timing data but is not specifically validated in the notes.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The loaders normalize format/schema variations and missing region labels. Missing feature frame times are replaced with a synthetic 400 Hz clock; trials with NaN `NdroppedFrames` are treated as unavailable. Paw coordinates and motion-energy gaps are nearest-filled, tongue gaps are preserved as class 2, missing motion-energy sessions become class 2, and all-zero neural tail trials are dropped.

ii.
```python
if frame_times.size == 0:
    frame_times = (np.arange(ts.shape[0]) + 1) / 400.0
...
if np.isnan(ndropped):
    continue
...
me_cat = np.full((time_axis.size, nraw), 2, dtype=np.int64)
```

iii. The AI aimed to preserve the requested visibility category for tongue/paw while following reference nearest-fill behavior for non-tongue signals, and documented loader fixes for mixed MAT layouts and missing metadata.

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports per-session loading and neural/video processing at roughly 3 seconds per session, estimating about 130 seconds for 44 sessions. HDF5 traversal, per-unit spike binning/smoothing, and per-trial video interpolation are the main substantive work.

ii.
```python
raw = load_raw_session(spec.data_path)
neural_trials, keep_unit_idx = bin_session_neural(...)
inputs, outputs, thresholds = build_session_outputs(...)
```

iii. Its runtime notes emphasize orientation-safe loading and “vectorized within trials” processing; the full conversion was considered acceptably fast.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops remain over manifests/sessions, HDF5 trials and units, units during spike binning, trials during interpolation/speed computation, paw features, and output trials. The per-spike membership and trial-index mapping list comprehensions in `bin_session_neural` are particularly vectorizable; several output stacking loops are also mechanical.

ii.
```python
valid_mask = np.array([t in trial_map for t in trial_idx], dtype=bool)
mapped_trials = np.array([trial_map[t] for t in trial_idx[mask]], dtype=int)
for tr_idx in range(x.shape[1]):
    ...
```

iii. The AI claimed processing was vectorized within trials “wherever possible,” retaining session-level loops because trial/frame structures are ragged. It did not explicitly analyze all remaining vectorization opportunities.

## 11-c. What processing does the code repeat multiple times?

i. `find_video_offset` is recomputed for every call to `align_feature` (tongue and both paws) and again for motion energy. Every feature independently traverses all trials and reloads/interpolates frame times. `parse_manifest_sessions()` is also called twice in `main`, and identical input arrays are allocated per trial.

ii.
```python
vidshift = find_video_offset(raw)  # in align_feature
...
vidshift = find_video_offset(raw)  # in align_motion_energy
...
print(f"Curated sessions available: {len(parse_manifest_sessions())}")
```

iii. The notes discuss shared helpers and speedups but do not acknowledge these repeated computations; they judged the runtime small enough.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Loaders read unused fields such as lick event arrays, reward/sample/delay times, cluster `tm` and `site`, and dropped-frame counts. Motion-energy `moveThresh` is parsed but discarded. Region strings are collected and normalized per unit but final assembly hard-codes every region index to ALM. Optional plotting also computes diagnostics not stored in the dataset.

ii.
```python
"lickL": [h5_read_cell_numeric_1d(...) for i in range(ntrials)],
"tm": tm,
...
me_data, _ = load_motion_energy(spec)
...
"brain_region_idx": [np.zeros(...) for sess in processed_sessions],
```

iii. The AI prioritized a normalized reusable internal representation and sanity-check plotting; it did not explicitly document most of this discarded work.
