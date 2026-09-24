# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent parses the authors’ fixed- and randomized-delay MATLAB loader scripts to recover the 44 selected session/date/probe specifications, then loads each `data_structure_*.mat` once with `pymatreader.read_mat`; motion energy is loaded separately when present.

ii.
```python
for loader in FIXED_DELAY_LOADERS:
    sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
for loader in RANDOMIZED_DELAY_LOADERS:
    sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
...
obj = read_mat(spec.data_path)["obj"]
```

iii. The notes say loader-defined sessions are preferable to scanning raw files because extra/commented or behavior-only files exist; this reproduces 25 fixed-delay plus 19 randomized-delay sessions and handles both MATLAB formats.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from loader filenames/session specifications, then unique IDs and per-session indices are assembled.

ii.
```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
...
subjects = stable_unique([x[0]["subject"] for x in processed])
subject_idx = np.asarray([subject_to_idx[x[0]["subject"]] for x in processed], dtype=np.int64)
```

iii. The notes justify following explicit loader metadata and report 14 distinct IDs, while noting a paper/code mouse-count discrepancy.

## 1-c. How are the data split into sessions?

i. Every uncommented loader entry that reaches `datapth = fullfile` becomes one `SessionSpec`; each processed specification becomes one session in the target lists, pooling selected probes where requested.

ii.
```python
if "datapth = fullfile" in line and current_date is not None and current_probe is not None:
    sessions.append(SessionSpec(subject=subject, date=current_date,
                                probe=current_probe, folder=folder, task=task))
```

iii. This follows `loadSessionData` and the paper session counts; the notes explicitly resolve raw-folder extras by using the authors’ inclusion lists.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines trial count. Behavioral arrays are indexed by zero-based trial, while spike `clu.trial` values are mapped from one-based raw trial numbers. Kept trial indices select matching neural, input, and output records.

ii.
```python
n_trials = int(round(float(obj["bp"]["Ntrials"])))
keep_trials = np.flatnonzero(trial_mask)
trial_to_keep[keep_trials_0based + 1] = np.arange(keep_trials_0based.size)
```

iii. The agent says raw trial fields and spike trial identifiers already supply boundaries, so reconstruction is unnecessary.

## 1-e. How are trials filtered based on quality controls?

i. It excludes stimulation, early-lick, and all `bp.no` ignore trials. It additionally removes behavior-valid trials after the maximum trial containing a spike in any selected neuron, skips sessions with fewer than two trials, and skips sessions with fewer than ten retained units.

ii.
```python
valid = (~stim_enable) & (~early) & (~no)
...
refined_trial_mask = trial_mask & neural_coverage_mask
...
if len(selected_neurons) < MIN_UNITS_PER_SESSION:
    return None
```

iii. The notes cite paper omission of early and ignore trials, control-only analyses, avoidance of all-zero post-recording trials, and the paper’s ten-unit session criterion. They chose to omit ignores to keep labels “unambiguous,” despite the requested ignore output class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each cluster’s `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment and probe location metadata for region labels.

ii.
```python
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
```

iii. The notes identify the reference path `alignSpikes -> getSeq -> removeLowFRClusters` and selected probes from the authors’ loaders.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned, binned into counts, converted to Hz, and smoothed with a custom one-sided (causal) half-Gaussian window of length 15 using reflected prefix padding. Selected probes are concatenated.

ii.
```python
rates = counts / DT
return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
...
kern = gaussian_window(n)
kern[: n // 2] = 0.0
kern /= kern.sum()
```

iii. The notes claim this reproduces reference Gaussian smoothing and uses a two-pass neuron selection/build for memory efficiency. The causal choice is not separately justified against the human reference’s symmetric 14 ms Gaussian.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled `garbage`, `gabrga`, `noisy`, or `real?` are removed; remaining units must have mean firing rate strictly above 1 Hz over retained trials and the full window. Sessions must retain at least ten units.

ii.
```python
bad = {"garbage", "gabrga", "noisy", "real?"}
...
if mean_fr > LOW_FR:
    selected.append(...)
```

iii. The notes cite `findClusters`, the paper’s >1 Hz criterion, and the session-level ten-unit inclusion rule. Unlike the human solution, `poor` is not rejected.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike’s trial-relative time has that trial’s go-cue time subtracted before bin assignment.

ii.
```python
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The notes state this matches `alignSpikes` with `params.alignEvent = 'goCue'` and needs no neural/video clock correction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The common grid contains 1000 non-overlapping 5 ms bins over `[-2.5, 2.5)` seconds. Raw spikes are binned directly; video streams are interpolated to bin centers.

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
```

iii. The notes prioritize the paper’s explicit 5 ms decoding bins and matching reference parameters over examples using 10 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is constructed from the configured go-cue-aligned window and bin width; raw `bp.ev.goCue` is used to align spikes but the same relative time vector is reused for every trial.

ii.
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
```

iii. The notes describe the common aligned time axis as the required continuous input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It adds half a bin to the left edges, yielding centers from -2.4975 to 2.4975 seconds, and reshapes to `(1, 1000)` as float32.

ii.
```python
return edges[:-1] + DT / 2.0
...
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The notes report and validate the resulting range and state that centers are the shared decoder time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is made from the centers of the exact edges used to bin aligned spikes, so corresponding columns represent the same intervals.

ii.
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
rates = binned_neuron_trials(..., time_edges)
```

iii. The notes list exact equality of input and converted time axes as a sanity check.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses only `obj.bp.R` after removing ignore trials; it does not combine instructed side with hit/miss to infer the actual lick on incorrect trials.

ii.
```python
R = to_vector(bp["R"], float).astype(int)
lick_direction = R[keep_trials]
```

iii. The mapping plan calls `R/L` a left/right label and says excluding no-response trials makes it defined. It overlooks that `R/L` is instructed side and must be inverted on misses.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No behavioral inference is performed: raw `R` is copied as binary left/right and repeated over all time bins; there is no `none` class because all ignores were filtered.

ii.
```python
np.full(time_centers.size, lick_direction[tr], dtype=np.int64)
```

iii. The agent justified dropping no-response trials for unambiguous binary labels. This conflicts with the task’s explicit left/right/none specification and mislabels incorrect licks.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It derives context from `obj.bp.autowater`.

ii.
```python
autowater = to_vector(bp["autowater"], float).astype(int)
```

iii. The notes identify autowater as the reference proxy for WC versus DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It reverses raw polarity so WC/autowater is 0 and DR is 1, then repeats the per-trial value across time.

ii.
```python
context = 1 - autowater[keep_trials]
```

iii. The notes explicitly choose WC=0 and DR=1 to match requested output ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It reads `bp.hit` and `bp.miss`, but after filtering `bp.no`; the stored outcome itself is just `hit`.

ii.
```python
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)
outcome = hit[keep_trials]
```

iii. The notes planned a miss-versus-hit output and deliberately excluded ignores, citing paper behavioral analyses rather than the requested three categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Kept misses become 0 and hits 1; mutual exclusivity is checked. No class 2 is produced, despite metadata advertising ignore.

ii.
```python
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(...)
```

iii. The agent says ignore trials were removed to avoid ambiguity. This contradicts the decoder output specification and the human solution, both of which retain ignore as category 2.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses x/y DLC tracks from side-camera `tongue`, `left_tongue`, `right_tongue` and bottom-camera `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`, plus frame times, go cues, and clock bitcodes.

ii.
```python
TONGUE_FEATURES = {1: ["tongue", "left_tongue", "right_tongue"],
                   2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]}
```

iii. The notes say the multi-feature reference representation had to be collapsed to one scalar tongue-speed output and chose aggregation across relevant points.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Each coordinate is linearly interpolated to 5 ms centers, gradients are taken per axis, tongue NaNs are converted to zero, Euclidean speeds are averaged across all available tongue features, then remaining NaNs are also converted to zero.

ii.
```python
xy_aligned = interp(taxis)
...
xv = np.nan_to_num(xv, nan=0.0)
yv = np.nan_to_num(yv, nan=0.0)
speed = np.sqrt(np.square(xvel) + np.square(yvel))
agg = np.nan_to_num(agg, nan=0.0, ...)
```

iii. The notes claim this follows reference interpolation/velocity processing and selects only decoder-needed feature groups. It does not preserve visibility or normalize camera scales as the human solution does.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Although a numerical 50th percentile is computed for metadata/plots, categorization rank-sorts every sample and assigns exactly the upper half to 1 and lower half to 0. It never emits class 2 (`not visible`).

ii.
```python
order = np.argsort(flat, kind="mergesort")
out = np.zeros(flat.size, dtype=np.int64)
out[order[flat.size // 2 :]] = 1
```

iii. The notes emphasize exact 50/50 movement bins “by construction.” This departs from literal `< median`/`>= median` behavior under ties and from the required missing-visibility category.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset is estimated from bitcode starts; frame times are transformed by `frameTimes - vidshift - goCue[trial]` and linearly interpolated to neural bin centers.

ii.
```python
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, ..., fill_value=np.nan)
xy_aligned = interp(taxis)
```

iii. The notes cite `findVideoOffset` and reference kinematic interpolation. A 0.5 s fallback is used when offset computation fails.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera DLC tracks for both `top_paw` and `bottom_paw`, with frame times, go cues, and video offset.

ii.
```python
PAW_FEATURES = {2: ["top_paw", "bottom_paw"]}
```

iii. The notes justify aggregating paw feature channels into the single required paw-speed trace. This differs from the human choice of only reliably tracked `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Coordinates are interpolated to the common grid, nearest-filled, differentiated, baseline-corrected by median coordinate differences, converted to Euclidean speed, and averaged across the two paws; missing aggregate values become zero.

ii.
```python
xpos[:, trial] = fill_nearest_1d(xpos[:, trial])
...
xv = xv - base[0]
yv = yv - base[1]
...
agg = np.nan_to_num(agg, nan=0.0, ...)
```

iii. The notes characterize this as reproducing reference per-feature velocity before scalar aggregation.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. All session samples are stable-rank-sorted into exact lower and upper halves (0/1). Class 2 is never produced.

ii.
```python
paw_thr = percentile_threshold(paw_speed, 50.0)
paw_bin = discretize_trace(paw_speed, paw_thr)
```

iii. The notes cite the mandated per-session median and exact class balance, but missing visibility is erased through filling/zero conversion.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times receive the same bitcode offset and per-trial go-cue subtraction, then positions are interpolated at 5 ms neural bin centers.

ii.
```python
old_time = frame_times - vidshift - float(align_times[trial])
xy_aligned = interp(taxis)
```

iii. The notes state that all video streams use the reference clock correction and common time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads per-frame traces from `motionEnergy_<subject>_<date>.mat`, falling back to embedded `obj.me`, and uses side-camera frame times plus go cue/bitcode timing.

ii.
```python
if spec.motion_energy_path.exists():
    raw_me = read_mat(spec.motion_energy_path).get("me")
...
if "me" in obj:
```

iii. The notes identify external motion energy as the ephys reference source and support several nested layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw scalar frame values are linearly interpolated onto bin centers, nearest-filled at missing edges, converted to zero for remaining invalid values, and then discretized.

ii.
```python
aligned[:, trial] = interp(taxis).astype(np.float32)
aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
return np.nan_to_num(aligned, nan=0.0, ...)
```

iii. The notes say the upstream trace is already spatially reduced, so only reference alignment and task-mandated median binning are needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Samples are rank-sorted session-wide into exact binary halves. It never emits class 2 (`no video`), and the computed numerical median is not actually consulted by `discretize_trace`.

ii.
```python
me_thr = percentile_threshold(motion_energy, 50.0)
me_bin = discretize_trace(motion_energy, me_thr)
```

iii. The notes explicitly prefer the task’s median over the paper’s manual movement threshold and celebrate exact 50/50 balance, but do not preserve no-video state.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by video offset and go cue and interpolated to common centers. If frame/value counts mismatch, synthetic 400 Hz times and a fixed 0.5 s offset are used.

ii.
```python
if frame_times.size != me_trial.size:
    frame_times = (np.arange(me_trial.size) + 1.0) / 400.0
    old_time = frame_times - 0.5 - float(align_times[trial])
else:
    old_time = frame_times - vidshift - float(align_times[trial])
```

iii. The notes cite `loadMotionEnergy` reference interpolation and its edge filling.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/invalid frame times are replaced by synthetic 400 Hz times; failed video-offset estimation falls back to 0.5 s; non-tongue coordinates and motion energy are nearest-filled; tongue and remaining missing aggregates become zero. Absent feature groups or motion energy become all-zero traces. Late trials without neural coverage are removed.

ii.
```python
return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
...
return 0.5
...
return np.zeros((time_centers.size, n_trials), dtype=np.float32), used_features
```

iii. The notes frame these as reference-compatible interpolation/filling and document the neural-coverage fix. However, zero filling prevents requested `not visible`/`no video` labels.

## 11-a. What are the most time-consuming steps of the code?

i. The agent identifies large MATLAB session loading, kinematic interpolation, and per-neuron spike binning as dominant, estimating roughly 10–12 seconds per session.

ii.
```python
obj = read_mat(spec.data_path)["obj"]
...
for neuron_index in np.flatnonzero(quality_mask):
```

iii. The notes report sample timing and say these costs dominate the roughly eight-minute projected full conversion.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops in position interpolation and velocity, neuron loops in selection/binning, and a nested neuron-by-trial copy loop could potentially be vectorized or reorganized. Ragged raw video makes complete trial vectorization difficult.

ii.
```python
for trial in range(n_trials):
    ...
for out_idx, selected in enumerate(selected_neurons):
    ...
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]
```

iii. The notes acknowledge per-neuron binning and kinematic interpolation as major costs and describe a two-pass approach as a memory/compute compromise.

## 11-c. What processing does the code repeat multiple times?

i. Neural spikes are traversed once to estimate mean rates and again to build retained matrices. Video trial structures and frame times are revisited independently for every feature. Per-trial neural matrices are allocated and then filled in a nested loop.

ii.
```python
mean_fr = neuron_mean_fr(...)
...
rates = binned_neuron_trials(...)
...
xpos, ypos = aligned_position(obj, view_index, feat_name, ...)
```

iii. The notes explicitly justify two-pass neural work to avoid constructing rejected neurons and say processing only required feature groups limits repetition.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads the full MATLAB object, computes numerical percentile thresholds that are not used by the rank-based discretizer, gathers `moveThresh` but never uses it, and computes/retains plotting summaries and feature provenance only as metadata. Full-data plots are optional.

ii.
```python
move_thresh = raw_me["moveThresh"]
...
tongue_thr = percentile_threshold(tongue_speed, 50.0)
tongue_bin = discretize_trace(tongue_speed, tongue_thr)  # threshold argument unused
```

iii. The notes say manual motion thresholds are intentionally superseded by task medians and diagnostic products support validation; they do not note that `discretize_trace` ignores its threshold argument.
