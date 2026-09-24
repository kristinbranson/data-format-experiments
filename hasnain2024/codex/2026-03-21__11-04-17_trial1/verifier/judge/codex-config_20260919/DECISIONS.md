# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers raw `data_structure_*.mat` files, parses the authors' `load*_ALMVideo.m` scripts to select the referenced subject/date/probe combinations, and intersects those combinations with available files. It loads old MATLAB files with SciPy and v7.3 files with targeted HDF5 readers; motion energy is loaded separately. The full run processes 44 selected sessions one at a time.

ii.
```python
for path in sorted(data_dir.glob("*/*.mat")):
    if path.name.startswith("data_structure_"):
        out[(parts[2], parts[3])] = path
...
for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
    ...
    specs.append(SessionSpec(... probes=current["probes"], session_path=path))
...
return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. The notes say loader scripts define the analyzed session/probe set, the 44-session intersection matches the paper's 25 fixed-delay plus 19 randomized-delay sessions, and dual readers are required because the files mix MATLAB formats. Targeted HDF5 reads were chosen for speed and memory.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from filenames/loader scripts and stored on each `SessionSpec`. During assembly, subjects are added in first-session order and each session receives the corresponding integer index. The full output contains 14 subjects.

ii.
```python
SessionSpec(subject=current["subject"], date=current["date"], ...)
...
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
```

iii. The agent states that filenames and loader scripts reliably identify animals and reports 14 unique subjects in the selected data.

## 1-c. How are the data split into sessions?

i. Each selected subject/date `data_structure` file is one session and becomes one element of `neural`, `input`, and `output`. Sessions with fewer than two retained trials or fewer than ten retained units would be skipped.

ii.
```python
for spec in session_specs:
    session = convert_one_session(spec, ...)
    if session is None:
        continue
    neural.append(session["neural"])
    inputs.append(session["input"])
    outputs.append(session["output"])
```

iii. The notes justify the selected 44 sessions as the intersection of author loader lists and available data and cite the paper's 44 analyzed-session total.

## 1-d. How are the data split into trials?

i. Behavioral arrays are treated as one entry per trial. Original zero-based indices selected by a Boolean validity mask link behavior, event times, video, motion energy, and spike trial numbers. Each retained trial is emitted as one matrix in its session.

ii.
```python
valid = session_valid_trial_mask(raw)
selected_trials = np.flatnonzero(valid)
trial_to_pos[selected_trials] = np.arange(selected_trials.size)
...
for local_idx, trial_idx in enumerate(selected_trials):
    final_neural.append(np.stack(neural_trials[local_idx], axis=0))
```

iii. The agent relied on aligned per-trial raw fields rather than reconstructing boundaries and validated trial labels against raw files.

## 1-e. How are trials filtered based on quality controls?

i. It retains only non-stimulation, non-early trials that are hits or misses and have a left/right instruction. Thus ignore/no-response trials are removed. It also removes trailing behavioral trials beyond the last trial containing spikes, and skips sessions with fewer than two trials.

ii.
```python
return ((raw["stim_enable"] == 0) & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1)))
...
selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. The notes say paper analyses omit ignore trials and therefore chose `~stim & ~early & (hit | miss)`. Neural-coverage filtering was added after validation exposed all-zero late trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu` on the loader-selected probe(s): each unit's `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
{"quality": ..., "trialtm": ..., "trial": ...}
...
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. The agent says the loader scripts identify ALM probes and raw spike times plus trial IDs and go cues are sufficient to reconstruct aligned rates.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned and accumulated as 200 Hz contributions into 5 ms bins. Each unit-by-trial rate matrix is then passed through the agent's reimplementation of `mySmooth`: a length-15 Gaussian-window kernel with its first half zeroed and reflected-prefix padding. Rates are stored as `float32`; there is no z-scoring or baseline subtraction.

ii.
```python
np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
...
kern = matlab_gausswin(n)
kern[: n // 2] = 0.0
kern /= kern.sum()
```

iii. The notes characterize this as matching the causal Gaussian-like kernel and reflected padding in the reference repository and report an independent raw-spike reconstruction matching the converted result.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units on selected probes are rejected for quality labels `garbage`, `gabrga`, `noisy`, or `real?`. Remaining units must have mean processed rate strictly above 1 Hz across retained trials/window. Sessions must retain at least ten units.

ii.
```python
return label not in {"garbage", "gabrga", "noisy", "real?"}
...
if float(unit_mat.mean()) <= LOW_FR_HZ:
    continue
...
if kept_units < 10:
    return None
```

iii. The exclusions are attributed to `findClusters`, while the 1 Hz threshold and ten-unit session criterion are attributed to the paper. The notes acknowledge unit totals differ slightly from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative timestamp is shifted by that trial's `bp.ev.goCue`; shifted times are placed on the common -2.5 to +2.5 s grid.

ii.
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. The agent says this matches `alignSpikes` and the explicitly requested go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 1000 non-overlapping 5 ms bins centered from -2.4975 to +2.4975 s. Raw spikes are binned directly at that resolution; video streams are interpolated onto these bin centers.

ii.
```python
DT = 1.0 / 200.0
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The agent chose the default reference pipeline's `dt=1/200` and `tmin/tmax`, rather than a 10 ms tutorial example.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common bin-center vector defined relative to the alignment event `bp.ev.goCue`, not a separately sampled raw variable.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes say this is the reference trial time base relative to the align event and is exactly the decoder input requested.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The agent builds equally spaced 5 ms bin edges over [-2.5, 2.5] s and uses their centers, repeating the same `(1, 1000)` vector for each trial.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. It was selected to match the reference parameters and common decoder time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact edges used to bin go-cue-shifted spikes, so each input sample corresponds to the same neural bin.

ii.
```python
bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. An independent check in the notes found exact equality with the expected aligned time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived only from the instructed-side flags `bp.R` and `bp.L` after restricting trials to hits/misses; no actual lick-side field or outcome-dependent inversion is used.

ii.
```python
& ((raw["R"] == 1) | (raw["L"] == 1))
...
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. The mapping plan says retained non-early hit/miss trials make direction well defined and maps `R/L` directly to right/left.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Left is encoded 0 and right 1, then repeated across all 1000 time bins. Ignore/no-lick trials and the required `none` category are absent.

ii.
```python
np.full(time_vec.size, lick_direction, dtype=np.int64)
...
"output_values": [["left", "right"], ...]
```

iii. The agent chose constant time series for trial-level labels to give all outputs a shared shape; it justified removing ignores based on paper analyses.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It comes from `bp.autowater`.

ii.
```python
"autowater": np.asarray(bp.autowater, dtype=np.float64).reshape(-1)
```

iii. The notes identify autowater as the raw context flag used by the reference code.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater==1` is WC (0), otherwise DR (1), repeated across time.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. The agent states this directly follows the reference context conditions and requested class order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit` after filtering to hit or miss trials; `bp.miss` participates in the filter and `bp.no` is loaded but excluded.

ii.
```python
& ((raw["hit"] == 1) | (raw["miss"] == 1))
...
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. The mapping plan says reference analyses omit `no` trials, so it retained only hit/miss trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is encoded incorrect (0), hit correct (1), and the value is repeated over time. The requested ignore (2) category is not represented.

ii.
```python
np.full(time_vec.size, outcome, dtype=np.int64)
...
["incorrect", "correct"]
```

iii. The agent justified binary output by excluding ignore trials to match its interpretation of paper curation.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses the side-camera `tongue` feature's x/y coordinates and frame times from `obj.traj`, along with bitcode/video clock fields and `bp.ev.goCue`. It does not use the bottom-camera tongue view or tracking likelihood explicitly.

ii.
```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```

iii. The notes say the side-view canonical marker avoids mixing camera coordinate systems.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y coordinates are linearly interpolated directly onto the 5 ms neural grid. `np.gradient` is applied per grid sample (without division by seconds), NaN derivatives are replaced by zero internally, and speed is Euclidean magnitude. Bins whose interpolated x/y were missing are restored to NaN. There is no coordinate smoothing or likelihood cutoff.

ii.
```python
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
...
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
...
return np.sqrt(xvel**2 + yvel**2)
```

iii. The agent described this as following reference interpolation and velocity logic, though its implementation choices are more specific than that justification.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session median is computed over finite samples in retained trials. Finite samples below/above are 0/1. Missing/not-visible samples are also assigned 0, and only two class names are declared rather than the requested third `not visible` class.

ii.
```python
tongue_thr = summarize_threshold(tongue_sel)
np.where(np.isfinite(tongue_sel[:, local_idx]),
         tongue_sel[:, local_idx] >= tongue_thr, 0).astype(np.int64)
```

iii. The notes report an initial degenerate result and say it was fixed by computing the percentile on visible frames and assigning non-visible frames to bin 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A per-session video offset is estimated from ephys and behavioral bit-start modes. Frame times are shifted by that offset and the trial go cue, then linearly interpolated at neural bin centers. Missing frame times fall back to a 400 Hz synthetic clock.

ii.
```python
return robust_mode(bitstart) / fs - robust_mode(raw["events"]["bitStart"])
...
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. The agent cites the reference video-offset logic and shared time base; fallback clocks were added for malformed/missing frame timing.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw` and `bottom_paw` x/y trajectories and frame times, plus clock-alignment variables.

ii.
```python
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, ...)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
```

iii. The notes say averaging both bottom-view paw markers captures overall paw movement while remaining close to tracked features in the paper.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's x/y is interpolated onto the neural grid and nearest-filled. Per-sample coordinate gradients have a median derivative subtracted, are nearest-filled, and are converted to speed magnitude. The two paw speeds are averaged wherever available; remaining NaNs become zero.

ii.
```python
if "tongue" not in feature_name:
    xpos[:, trix] = nearest_fill_1d(xpos[:, trix])
...
xv = xv - basederiv[0]
...
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), ...)
```

iii. This was presented as a reference-style interpolation/fill and velocity computation, with both paws used as an overall paw measure.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A per-session median across all retained samples divides values into 0 below and 1 at/above. There is no effective `not visible` class because missing values are filled or changed to zero, and only two class labels are declared.

ii.
```python
paw_thr = summarize_threshold(paw_sel)
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64)
```

iii. The agent followed the requested 50th percentile but treated missing paw samples through filling instead of a third category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are video-offset corrected and go-cue shifted, then x/y positions are interpolated onto the same neural bin centers; missing timing uses a synthetic 400 Hz clock.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. The agent says this matches reference video-offset alignment and gives all streams a common grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's separate `motionEnergy_<subject>_<date>.mat` trace, with side-camera frame times, video bitcode offset, and trial go cue for timing.

ii.
```python
me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
motion_energy, motion_thresh = load_motion_energy(me_path)
```

iii. The notes document multiple nested MATLAB layouts and say the loader was generalized to unwrap all forms.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already reduced per-frame trace is linearly interpolated to neural bin centers and nearest-filled; missing residual values become zero. No new image-level energy is calculated and the file's manual `moveThresh` is ignored.

ii.
```python
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
out[:, trix] = nearest_fill_1d(out[:, trix])
...
motion = np.nan_to_num(motion, nan=0.0)
```

iii. The agent says the raw trace is already spatially reduced and the prompt requires a median rather than the paper's manual movement threshold.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A per-session median across retained samples yields binary 0/1 values. There is no third `no video` class; missing data becomes zero before thresholding.

ii.
```python
motion_thr = summarize_threshold(motion_sel)
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64)
```

iii. The agent followed the requested 50th percentile but did not preserve missingness as a category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video offset and trial go cue, and motion values are interpolated onto neural bin centers. With missing frame times it assumes 400 Hz, a 0.5 s offset, and the go cue.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
...
old_t = (np.arange(me.size) + 1.0) / 400.0 - 0.5 - align_times[trix]
```

iii. The notes cite reference clock correction and report a spot check against direct raw interpolation.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The loader supports heterogeneous MATLAB/container layouts and empty probe slots. Missing frame times are replaced by a synthetic 400 Hz clock; non-tongue trajectories and motion energy are nearest/interpolation filled; all-missing features can become zeros; missing tongue samples ultimately become class 0. Trailing trials beyond neural coverage are dropped. Thus missing output information is generally imputed or collapsed into the low class rather than represented by the specified missing-data classes.

ii.
```python
if frame_times is None or ...:
    frame_times = (np.arange(ts.shape[0]) + 1.0) / 400.0
...
motion = np.nan_to_num(motion, nan=0.0)
...
selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. The notes describe these as fixes for nested structs, direct motion arrays, empty probe slots, absent frame times, and behavior extending beyond neural recording. Validation drove the late-trial exclusion.

## 11-a. What are the most time-consuming steps of the code?

i. File loading, neural unit processing, video interpolation/velocity construction, and building the large in-memory output dominate. The full 44-session conversion took about 135 seconds; the notes specifically identify recursive HDF5 loading and naive spike loops as potential bottlenecks.

ii.
```python
raw = load_session(spec)
for unit in raw["units"]:
    unit_mat = compute_unit_trial_matrix(...)
...
tongue_pos = feature_xy(...)
```

iii. The agent says targeted HDF5 reads and vectorized spike accumulation kept runtime well below the optimization threshold.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining loops include per-unit smoothing, per-trial feature interpolation/velocity, per-paw processing, and per-trial assembly (including appending every unit trace). Some could be batched for rectangular data, although ragged raw video and unit spike arrays make complete vectorization awkward.

ii.
```python
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(...)
...
for trix, trial in enumerate(trials):
...
for trix in range(selected_trials.size):
    neural_trials[trix].append(unit_mat[trix])
```

iii. The notes emphasize that spike accumulation was vectorized with `np.add.at` to avoid nested spike/trial histograms and that loading, rather than these loops, dominated runtime.

## 11-c. What processing does the code repeat multiple times?

i. `find_video_offset` is recomputed separately for each feature call and again for motion energy. Feature interpolation and gradient logic is repeated separately for tongue and each paw. The same time vector and constant trial labels are copied for every trial, and per-unit traces are repeatedly appended into trial lists.

ii.
```python
vidshift = find_video_offset(raw)  # in feature_xy
...
vidshift = find_video_offset(raw)  # in aligned_motion_energy
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The agent's notes claim session-by-session processing and shared logic keep work bounded, but do not call out the repeated offset computations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads several unused fields (`L`, `no`, sample/delay/reward events, manual motion threshold, dropped-frame metadata beyond gating), constructs/stores continuous movement arrays in the transient `session_out`, and may create diagnostic plots; these continuous arrays are not placed in the final dataset. It also loads all old-format trajectory features even though only three names are used.

ii.
```python
"no": np.asarray(bp.no, ...),
"events": {"sample": ..., "delay": ..., "reward": ...},
...
"continuous": {"tongue": tongue_sel, "paw": paw_sel, "motion": motion_sel, ...}
```

iii. Continuous values are retained temporarily for threshold diagnostics and optional plots. The notes say targeted HDF5 loading avoids much otherwise-unused data, though old-format loading remains broad.
