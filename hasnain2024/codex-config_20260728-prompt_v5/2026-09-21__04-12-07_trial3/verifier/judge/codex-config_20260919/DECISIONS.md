# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter hard-codes 25 fixed-delay and 19 randomized-delay `SessionSpec` records, including curated probe numbers, and loads each session’s `data_structure_*.mat` plus its separate `motionEnergy_*.mat`. `load_any_mat` supports HDF5 and older MATLAB files and normalization functions reconcile their layouts.

ii.
```python
ALL_SPECS = FIXED_SPECS + RANDOMIZED_SPECS
def load_any_mat(path):
    if h5py.is_hdf5(path): return mat73.loadmat(str(path))
    return convert_mat_struct(loadmat(path, squeeze_me=True, struct_as_record=False))
```

iii. The notes say the list was transcribed from the authors’ animal-specific loaders so excluded/unselected files are not accidentally included, while dual-format loading was required by the supplied files.

## 1-b. How are the data split into subjects?

i. Each `SessionSpec` explicitly contains a subject ID; assembly creates `subjects` in first-session order and maps every session to `subject_idx`.

ii.
```python
SessionSpec("EKH1", "2021-08-07", "fixed", (2,))
subjects = list(dict.fromkeys(result["subject"] for result in session_results))
subject_idx = np.array([subject_to_idx[result["subject"]] for result in session_results])
```

iii. The notes report 14 unique subjects in the selected neural sessions and treat the curated filename/session metadata as authoritative.

## 1-c. How are the data split into sessions?

i. One curated `<subject>_<date>` file is one session and one element of each top-level session list. Fixed and randomized cohorts differ only in their source directory; selected probes are concatenated within the session.

ii.
```python
def session_id(self): return f"{self.subject}_{self.date}"
for spec in specs:
    result, info = process_session(spec, show_processing=show_processing)
```

iii. This mirrors the original loader organization and preserves the paper’s selected probe per recording day.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the original trial rows. Per-trial behavioral/event/video entries use that index; spike `trial` values are converted from one-based IDs. After a Boolean validity mask, each retained trial becomes one neural/input/output element.

ii.
```python
ntrials = int(float(bp["Ntrials"]))
valid_idx = np.flatnonzero(valid_mask)
original_idx = trial_ids.astype(np.int64) - 1
```

iii. The notes state that the Bpod structure and cluster trial IDs already provide trial boundaries, so they need not be reconstructed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are not early-lick or photostimulation trials, have a finite go cue, have one of hit/miss/no set, and do not exceed the last trial represented in selected-probe spikes. Sessions must retain at least two trials.

ii.
```python
return (~early) & (~stim) & outcome & np.isfinite(go_cue)
if last_neural_trial > 0 and last_neural_trial < ntrials:
    valid_mask &= (np.arange(1, ntrials + 1) <= last_neural_trial)
```

iii. Early/stim removal follows the paper; the final recording cutoff was added after validation exposed all-zero neural trials beyond the end of ephys.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each cluster’s `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
trial_ids = as_1d_numeric(probe["trial"][clu_idx])
trial_times = as_1d_numeric(probe["trialtm"][clu_idx])
go_cue = get_bp_array(bp["ev"], "goCue", ntrials)
```

iii. The notes identify these as the fields used by `findClusters`, `alignSpikes`, and `getSeq` in the reference pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned, counted into 5 ms bins, divided by 0.005 to obtain Hz, and smoothed with a Python port of the reference `mySmooth(...,15,'reflect')`: a causalized 15-sample Gaussian with prepended boundary samples. No normalization or baseline subtraction is applied.

ii.
```python
kern = matlab_gausswin(n)
kern[: math.floor(kern.size / 2)] = 0.0
rates = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The agent says causal Gaussian smoothing matches the methods and MATLAB `mySmooth` implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes cluster labels `garbage`, `gabrga`, `noisy`, and `real?`. It then computes four correct-trial condition PSTHs (DR/WC × left/right), averages those, and retains units above 1 Hz. A session must have at least 10 units.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
mean_fr = float(np.mean(np.stack(psth_stack, axis=1)))
if mean_fr > LOW_FR_HZ: kept_rates.append(rates)
```

iii. The labels and 1 Hz criterion were taken from `findClusters` and `removeLowFRClusters`; condition averaging was intended to reproduce reference low-FR filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For every spike, its trial’s go-cue time is subtracted from `trialtm`; aligned spikes in [-2.5, 2.5) seconds are binned.

ii.
```python
aligned = trial_times - go_cue[original_idx]
keep = (aligned >= TMIN) & (aligned < TMAX)
```

iii. This directly ports the reference `alignSpikes` go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All streams use 1/200 s = 5 ms bins, 1,000 bins from -2.5 to +2.5 seconds. Spikes are histogrammed directly at that resolution; video streams are interpolated onto the 5 ms bin centers.

ii.
```python
DT = 1.0 / 200.0
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
```

iii. The notes cite the paper/reference defaults `dt=1/200`, `tmin=-2.5`, and `tmax=2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common time grid defined relative to each trial’s raw `bp.ev.goCue` alignment event.

ii.
```python
session_input = [TIME_CENTERS[None, :].astype(np.float32).copy()
                 for _ in range(valid_idx.size)]
```

iii. The required decoder input is time relative to the requested event, so the shared neural bin centers are used.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Consecutive bin edges from -2.5 to 2.5 seconds are created and adjacent-edge midpoints form the input; it is copied for every trial.

ii.
```python
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
```

iii. Bin centers represent the timestamp of each neural bin without further transformation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses exactly the centers of the edges used to bin go-cue-aligned spikes, so input column k and neural column k refer to the same interval.

ii.
```python
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
session_input = [TIME_CENTERS[None, :].copy() ...]
```

iii. The common grid was explicitly chosen to guarantee alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses the per-trial event lists `bp.ev.lickL` and `bp.ev.lickR`, together with `goCue`.

ii.
```python
lick_l = ensure_event_list(bp["ev"].get("lickL", []), ntrials)
lick_r = ensure_event_list(bp["ev"].get("lickR", []), ntrials)
```

iii. The notes argue that actual first-lick behavior is preferable to inferring a response from instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The first left/right lick from go cue through three seconds afterward is selected. The earlier side is class 0/1; no qualifying lick is class 2; the trial label is repeated over time.

ii.
```python
left = left[(left >= go_cue[trix]) & (left < go_cue[trix] + RESPONSE_WINDOW_S)]
direction[trix] = 0 if left_first <= right_first else 1
```

iii. This was described as the cleanest direct measure and naturally maps ignore/no-response trials to `none`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from `bp.autowater`.

ii.
```python
autowater = get_bp_array(bp, "autowater", ntrials, dtype=bool)[valid_idx]
```

iii. The field directly identifies water-cued trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater true maps to WC (0); false maps to DR (1), repeated over all time bins.

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)
```

iii. This is a direct relabeling into the requested classes.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = get_bp_array(bp, "hit", ntrials, dtype=bool)[valid_idx]
miss = get_bp_array(bp, "miss", ntrials, dtype=bool)[valid_idx]
no = get_bp_array(bp, "no", ntrials, dtype=bool)[valid_idx]
```

iii. These mutually exclusive Bpod flags directly encode the requested outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is incorrect (0), hit correct (1), and no/otherwise ignore (2), repeated through time.

ii.
```python
outcome = np.full(valid_idx.size, 2, dtype=np.int8)
outcome[miss] = 0; outcome[hit] = 1; outcome[no] = 2
```

iii. The mapping follows the requested category order.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-camera (`obj.traj[0]`) DLC `tongue` x/y trajectories and frame times, `bp.ev.goCue`, and the ephys/Bpod bitcode fields used for video clock correction. It does not use the bottom-camera tongue.

ii.
```python
tongue_x, tongue_y, tongue_visible = extract_feature_traces(obj, "tongue", 0, go_cue)
```

iii. The notes say raw DLC missingness is retained as visibility and the reference kinematics functions guided extraction.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Raw positions are linearly interpolated to the 5 ms grid. Gradients are taken per grid sample; NaN tongue gradient values are set to zero, and speed is Euclidean magnitude. There is no positional smoothing and no conversion from samples to seconds.

ii.
```python
interp_xy = interp_feature(aligned_times, feat_xy, taxis)
xv = np.gradient(trial_xy[:, 0]); yv = np.gradient(trial_xy[:, 1])
xv[~np.isfinite(xv)] = 0.0
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The agent says this follows `findPosition`/`findVelocity`, while preserving an independent visibility mask for the decoder’s third class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single session median is computed over finite, visible tongue-speed samples. Visible values below the median are 0 and values at/above it are 1; invisible points remain 2.

ii.
```python
threshold = float(np.nanpercentile(valid_speed, 50.0))
out[visible] = (speed[visible] >= threshold).astype(np.int8)
```

iii. This follows the task’s per-session 50th-percentile requirement and avoids cross-session scale effects.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A per-session video shift is estimated from median bitcode starts, subtracted with trial go cue from frame times, and positions are linearly interpolated at neural `TIME_CENTERS`.

ii.
```python
vidshift = np.nanmedian(bit_file_offset) / fs_val - np.nanmedian(bit_start)
aligned_times = frame_times - vidshift - go_cue[trix]
interp_xy = interp_feature(aligned_times, feat_xy, TIME_CENTERS)
```

iii. The notes identify this as the clock correction used by `findVideoOffset` and validate it through aligned behavior traces.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both `top_paw` and `bottom_paw` DLC x/y trajectories from bottom camera `obj.traj[1]`, plus frame times and video/go-cue clock fields.

ii.
```python
for feat_name in ("top_paw", "bottom_paw"):
    paw_x, paw_y, paw_visible = extract_feature_traces(obj, feat_name, 1, go_cue)
```

iii. The notes planned to derive paw movement from bottom-view tracked paw points and preserve their raw visibility.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each marker is interpolated to 5 ms, position gaps are nearest-filled, sample gradients are baseline-corrected, and x/y magnitude is computed. When available, the speeds of the two different paw markers are averaged.

ii.
```python
xv = xv - basederiv[0]; yv = yv - basederiv[0]
paw_speed = np.divide(paw_weighted.sum(axis=2), paw_counts, where=paw_counts > 0)
```

iii. The agent cites reference nearest-filling for non-tongue features and describes averaging as combining available paw markers.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The median over finite bins where either paw is visible is the per-session split; below is 0, at/above 1, neither visible 2.

ii.
```python
paw_visible = paw_vis_stack.any(axis=2)
paw_cat, paw_thresh = compute_speed_categories(paw_speed, paw_visible)
```

iii. The session median and explicit missing class implement the task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are bitcode-corrected and go-cue-centered, then paw coordinates are interpolated onto `TIME_CENTERS` before differentiation.

ii.
```python
aligned_times = frame_times - vidshift - go_cue[trix]
interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. This shares the neural time grid and clock correction used for the other video streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads each separate `motionEnergy_<subject>_<date>.mat` trace and uses side-camera frame times plus bitcode/go cue for timing.

ii.
```python
me = load_motion_energy(spec)
view0 = obj["traj"][0] if obj["traj"] else {}
```

iii. The notes say the standalone trace is already spatially reduced and is the reference loader’s motion-energy source.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Nested/raw MATLAB layouts are unwrapped, and each trace is linearly interpolated to the common time centers. No smoothing or recomputation from pixels is performed.

ii.
```python
aligned[:, trix] = np.interp(TIME_CENTERS,
    frame_times[valid] - vidshift - go_cue[trix], trace[valid],
    left=np.nan, right=np.nan)
```

iii. The upstream files already contain the paper’s per-frame motion statistic, so only temporal alignment is needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Its per-session median over finite aligned values splits low (0) and high (1); NaN/no-video bins are class 2.

ii.
```python
motion_visible = np.isfinite(motion)
motion_cat, motion_thresh = compute_speed_categories(motion, motion_visible)
```

iii. This implements the task-required median rather than the paper’s manually stored movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video offset, centered on each go cue, and linearly interpolated onto neural bin centers.

ii.
```python
frame_times[valid] - vidshift - go_cue[trix]
```

iii. The notes report direct spot checks of a raw trace against all 1,000 converted categories.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Loaders normalize differing MATLAB layouts and tolerate absent fields. Missing kinematic samples preserve a raw visibility mask; paw values are nearest-filled only for velocity calculation. Missing video/motion becomes class 2. However, missing or scalar-NaN frame times are replaced with an assumed 400 Hz sequence. Trials beyond ephys are removed.

ii.
```python
if frame_times.size == 1 and np.isnan(frame_times[0]):
    frame_times = np.arange(1, feat_xy.shape[0] + 1) / 400.0
out = np.full(speed.shape, 2, dtype=np.int8)
```

iii. The agent aimed to keep missingness explicit for categorical outputs, while fallback timestamps were intended to retain otherwise usable video; validation motivated the ephys cutoff.

## 11-a. What are the most time-consuming steps of the code?

i. Session MAT loading/normalization and repeated trial/feature video interpolation dominate; the full conversion took 214.6 seconds. Neural cluster smoothing also loops over units and convolution columns.

ii.
```python
obj = normalize_obj(load_any_mat(spec.data_path))
for trix in range(ntrials):
    interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. The notes emphasize single-pass extraction and report per-session/full timings, though they do not provide a formal profiler breakdown.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops in feature extraction, motion interpolation, lick selection, velocity, and output assembly, neuron loops in session matrix assembly, and column loops in `my_smooth` could potentially be vectorized or batched. Ragged frame/event arrays make several loops nontrivial. Spike counting is already vectorized with flattened `np.bincount`.

ii.
```python
counts_flat = np.bincount(flat_idx, minlength=TIME_CENTERS.size * n_valid)
for trix in range(ntrials): ...
for neuron_idx, rates in enumerate(kept_rates): ...
```

iii. The agent explicitly calls out vectorized spike binning and treats the ragged video work as once-per-feature/session.

## 11-c. What processing does the code repeat multiple times?

i. `get_vidshift` is recalculated for motion energy and for each of tongue, top-paw, and bottom-paw extraction. The same trajectory/frame arrays are normalized/read and interpolated separately per feature; time grids and per-trial output constants are repeatedly copied.

ii.
```python
vidshift = get_vidshift(obj)  # in align_motion_energy
vidshift = get_vidshift(obj)  # in each extract_feature_traces call
```

iii. The notes claim each motion/kinematic feature is aligned once and reused thereafter, which is true within a feature, but common clock/frame work is still repeated across features.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It normalizes the full MATLAB object, including unused fields/features. It computes/stores continuous positions, component velocities, speeds, thresholds, condition PSTHs, and condition position arrays even though only categorical outputs and retained rates are saved. Plotting-related continuous intermediates are discarded after each session.

ii.
```python
psth_stack.append(rates[:, cond_pos].mean(axis=1))
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
session_info = {"tongue_threshold": tongue_thresh, ...}
```

iii. Some intermediates are necessary to decide filtering/categories and thresholds are retained only as metadata; the agent prioritized traceability and validation rather than minimizing transient work.
