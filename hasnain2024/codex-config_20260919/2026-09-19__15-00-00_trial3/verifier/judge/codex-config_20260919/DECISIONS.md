# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 fixed-delay, two-context sessions and their selected ALM probes, then directly reads required fields from each v7.3 HDF5 `data_structure` file with `h5py` and the accompanying v5 motion-energy file with SciPy. It does not load the randomized-delay cohort.

ii. `DATA_DIR = APP / "data" / "Ephys_Behavior"`; `SESSIONS = [("JEB6", "2021-04-18", (2,)), ...]`; `with h5py.File(data_path, "r") as f:`; `scipy_io.loadmat(path, simplify_cells=True)["me"]`.

iii. The notes say these are the 12 sessions explicitly selected by the paper's two-context analysis, and exclude randomized-delay sessions as a different variant without the required alternating WC/DR context. Direct field access was chosen to avoid materializing unused multi-GB fields.

## 1-b. How are the data split into subjects?

i. The animal token in each hard-coded `(animal, date, probes)` tuple defines the subject. Unique IDs are accumulated in first-session order, and each session indexes that list.

ii. `if animal not in subjects: subjects.append(animal)` and `subject_idx = np.asarray([subjects.index(animal) for animal, _, _ in selected], dtype=np.int64)`.

iii. The notes preserve the seven distinct source IDs rather than relabeling them to force the paper's reported six mice.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date tuple and corresponding `data_structure_<animal>_<date>.mat` file becomes one session element in all target lists.

ii. `for index, (animal, date, probes) in enumerate(selected): ... process_session(animal, date, probes, show)` and `session_id = f"{animal}_{date}"`.

iii. The agent says the 12 explicit context-script sessions are the applicable cohort and treats each source file as a session.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the number of trials. All Bpod arrays, go cues, spike trial IDs, DLC entries, and motion-energy entries are indexed by that zero-based Python trial index; retained trials become individual list elements.

ii. `n_trials = int(np.asarray(f["obj/bp/Ntrials"])[0, 0])`; `for pos, source_trial in enumerate(retained): ... neural_trials.append(neural_trial)`.

iii. The notes describe native objects as holding one Bpod record, trajectory entry, and motion-energy vector per trial and record source indices for auditing.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are not early-lick or stimulation trials, have a finite go cue, exactly one hit/miss/no flag, and exactly one R/L flag. Missing-video trials remain. No end-of-ephys cutoff is applied.

ii. `retained = np.flatnonzero(~bp["early"] & ~bp["stim"] & np.isfinite(go) & (outcome_sum == 1) & (side_sum == 1))`.

iii. Early and stimulation exclusion follows the paper; ignore trials are deliberately retained because the target asks for an ignore class. Consistency checks on outcome and side flags were added defensively.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each cluster's `trial`, `trialtm`, and `quality`, together with `bp.ev.goCue`.

ii. `trials = h5_vector(f, group["trial"][cluster_index, 0], ...)`; `trial_times = h5_vector(... "trialtm" ...)`; `quality = h5_string(...)`; `aligned = trial_times[valid_trial] - go[tr0]`.

iii. The agent identifies this as the reference `alignSpikes` mapping and uses only loader-designated ALM probes.

## 2-b. How is the `neural` data processed?

i. Per-unit spikes are go-aligned, counted in 10 ms bins over `[-2.5, 2.5)`, divided by 0.01 to obtain Hz, and smoothed with a 15-sample causal Gaussian kernel plus a reflected leading block. Rates are stored as float32.

ii. `counts = bin_one_unit(...)`; `rates = smooth_reference(counts / DT).astype(np.float32)`; in `matlab_gausswin_causal`, `kernel[: n // 2] = 0`.

iii. The agent says this reproduces the paper tutorial/main choice and kinematics pipeline (`getSeq`/`mySmooth`) rather than the 5 ms default/single-trial settings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled garbage, gabrga, noisy, or real? are rejected case-insensitively. All other labels, including `poor`, pass quality filtering, then units with processed mean rate at or below 1 Hz are removed.

ii. `BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}`; `if quality.lower() in BAD_QUALITIES: continue`; `if mean_rate > 1.0: all_rates.append(rates)`.

iii. The notes attribute the bad-label list to `findClusters(... quality={'all'})` and the strict >1 Hz threshold to the paper; they do not tune it to match the reported unit count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each within-trial spike time has that trial's go-cue time subtracted before binning.

ii. `aligned = trial_times[valid_trial] - go[tr0]`.

iii. The agent states this directly matches the reference `alignSpikes` operation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 10 ms (500 bins from -2.5 to +2.5 s). Native spikes are histogrammed into those bins; video streams are interpolated to their centers.

ii. `DT = 0.010`; `TIME = np.arange(TMIN, TMAX, DT) + DT / 2`; `"time_bin_size": 10.0`.

iii. The agent chose 10 ms because it found that resolution in the main choice/context and kinematics pipeline, acknowledging that other reference settings use 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the configured alignment window and 10 ms bin edges; `bp.ev.goCue` is used to align spikes and behavior but the stored time vector is the common relative grid.

ii. `TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2`.

iii. The notes say bin centers match the reference `obj.time` and the decoder requests only relative time.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The midpoint of every 10 ms bin in `[-2.5, 2.5)` is computed, cast to float32, reshaped to `(1, 500)`, and copied for every trial.

ii. `time_input = TIME.astype(np.float32)[None, :]`; `input_trials.append(time_input.copy())`.

iii. A common grid guarantees identical dimensions and records time rather than a binary event signal because this input is elapsed time.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact edges used to bin go-cue-relative spikes.

ii. `EDGES = TMIN + np.arange(N_TIME + 1) * DT`; `bins = np.searchsorted(EDGES, aligned, side="right") - 1`.

iii. Independent checks reported that all converted time vectors matched reconstructed bin centers.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no`.

ii. `right, left = bool(bp["R"][trial]), bool(bp["L"][trial])`; `hit, miss, no = ...`.

iii. The notes explain that instructed side plus outcome determines the actual response side, while `no` means no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No-response is class 2; a hit uses the instructed side; a miss uses its opposite. The scalar is broadcast across time.

ii. `if no: lick = 2; elif hit: lick = 1 if right else 0; else: lick = 0 if right else 1`; `output[0] = lick`.

iii. This was chosen to represent actual response direction and satisfy the requested left/right/none classes.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It comes from the per-trial `bp.autowater` flag.

ii. `context = 0 if bool(bp["autowater"][trial]) else 1`.

iii. The paper's condition expressions use autowater to identify WC trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater true maps to WC=0 and false to DR=1; the value is broadcast across all bins.

ii. `output[1] = context` with `output_values` equal to `["WC", "DR"]`.

iii. This is a direct relabeling matching the requested ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit`, `bp.miss`, and `bp.no`.

ii. `hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])`.

iii. The three mutually exclusive flags directly encode the requested outcomes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit maps to correct=1, miss to incorrect=0, and otherwise/no-response to ignore=2; it is broadcast across time.

ii. `outcome = 1 if hit else (0 if miss else 2)`; `output[2] = outcome`.

iii. Ignore is retained as a target-required exception to behavioral analyses that omit it.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-camera `obj.traj` feature `tongue`, specifically x/y coordinates and frame times. Go cues and SpikeGLX/Bpod bitcode starts provide alignment. Tracking likelihood is not explicitly read.

ii. `tongue_ix = side_names.index("tongue")`; `x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], TIME)`; similarly for y.

iii. The notes choose the side-view tongue trace and preserve raw coordinate missingness as visibility.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-view x/y are linearly interpolated to 10 ms centers; coordinate gradients are taken per sample and combined with `hypot`. Nonfinite gradients are set to zero, then samples whose interpolated x/y were missing are reset to NaN. No time division or coordinate smoothing is applied.

ii. `xv = np.gradient(xf)`; `yv = np.gradient(yf)`; `speed = np.hypot(xv, yv)`; `speed[~visible] = np.nan`.

iii. The agent calls this reference-style velocity processing and preserves tongue missingness for the requested class 2.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session median is computed over all finite samples from retained trials. Below median is 0, at/above median is 1, and NaN is 2.

ii. `threshold = float(np.nanpercentile(subset, 50))`; `labels[finite & (subset < threshold)] = 0`; `labels[finite & (subset >= threshold)] = 1`.

iii. This exactly implements the task's inequality and finite-only session threshold; ties intentionally enter class 1.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-clock shift is subtracted from side-camera frame times, then the trial go cue is subtracted, and coordinates are interpolated to neural bin centers.

ii. `vidshift = matlab_mode(sglx_start) / fs - matlab_mode(bit_start)`; `aligned_ft = side_ft - vidshift - go[trial]`; `interp_matlab(..., TIME)`.

iii. The notes identify this as `findVideoOffset` plus the reference interpolation and report independent alignment checks.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera DLC x/y and frame times for both `top_paw` and `bottom_paw`, plus synchronization fields and go cues.

ii. `paw_indices = [bottom_names.index(x) for x in ("top_paw", "bottom_paw") if x in bottom_names]`.

iii. The agent averages both available paw landmarks to avoid arbitrarily selecting one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's interpolated x/y missing values are nearest-filled, gradients are computed, a baseline derived from the median x difference is subtracted from both x and y gradients, and speed magnitude is formed. Speeds from the two paws are averaged where available; original visibility restores NaNs.

ii. `xf, yf = fill_nearest(x), fill_nearest(y)`; `baseline_x = np.nanmedian(np.diff(np.column_stack((xf, yf)), axis=0), axis=0)[0]`; `paw[trial] = ... summed / count`.

iii. The agent says this matches `findVelocity` and uses either landmark when the other is unavailable.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same retained-trial, finite-sample per-session median rule is used: 0 below, 1 at/above, 2 missing.

ii. `paw_labels, paw_threshold = discretize_session(paw, retained)`.

iii. This implements the explicit task threshold and retains a not-visible class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are corrected by the same session clock shift and trial go cue, then x/y are interpolated to `TIME`.

ii. `aligned_ft = bottom_ft - vidshift - go[trial]`; `x = interp_matlab(aligned_ft, ..., target_absolute)`.

iii. A shared corrected time base was chosen for all neural and behavioral streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses per-trial `me.data` from the separate motion-energy MATLAB file and side-camera frame times, plus bitcode synchronization and go cues.

ii. `loaded = scipy_io.loadmat(path, simplify_cells=True)["me"]`; `motion_trials = load_motion_file(...)`.

iii. The agent notes the motion trace is already spatially reduced upstream and follows the side-camera clock.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is linearly interpolated to 10 ms centers. If side frame times are unavailable, a synthetic 400 Hz time vector with a 0.5 s offset is used. No smoothing or differentiation is applied.

ii. `motion[trial] = interp_matlab(aligned_ft, me, TIME)`; fallback: `np.arange(1, me.size + 1) / 400.0`.

iii. The notes say the upstream value is already a per-frame scalar, so only alignment/interpolation and target-required discretization are needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Its finite retained-trial samples are split at the per-session median: 0 below, 1 at/above, 2 unavailable.

ii. `motion_labels, motion_threshold = discretize_session(motion, retained)`.

iii. The task-mandated median intentionally replaces the source file's manual `moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted to the behavior clock, made relative to each trial's go cue, then interpolated to the neural time centers.

ii. `aligned_ft = side_ft - vidshift - go[trial]`; `motion[trial] = interp_matlab(aligned_ft, me, target_absolute)`.

iii. This follows the reference motion-energy alignment using the side camera.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid trial flag combinations and nonfinite go cues are excluded. Missing/empty trajectories yield class 2; absent frame times are synthesized at 400 Hz except wholly nonfinite times, which make the trajectory unavailable. Paw coordinates are nearest-filled only for derivative computation, but original visibility controls class 2. Motion traces may be missing and receive class 2. Length mismatches are truncated during interpolation; neural mismatches raise errors.

ii. `if frame_times.size == 0 ... frame_times = np.arange(...) / 400.0`; `if ... all(~np.isfinite(frame_times)): return None, None`; `n = min(x.size, y.size)`; `raise ValueError("Spike trial/time length mismatch...")`.

iii. The agent aims to preserve usable neural/trial data while marking unavailable video honestly, and adds strict validations for structural inconsistencies.

## 11-a. What are the most time-consuming steps of the code?

i. File I/O and per-session neural/video processing are the main costs; the agent specifically identifies loading full MATLAB objects and per-trial smoothing as potential bottlenecks and avoids them. The final run reportedly took about 28–41 seconds.

ii. Direct reads such as `with h5py.File(data_path, "r")` and the trial loop in `load_behavior_streams` dominate remaining work.

iii. The notes report direct field access, vectorized spike work, and sequential session release as speedups over loading whole 100–300 MB objects.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Python loops remain over sessions, clusters, trials, and paw features. Trial video arrays are ragged, limiting straightforward vectorization; spike binning and smoothing are vectorized across each unit's trials.

ii. `for trial in range(n_trials):`; `for cluster_index in range(group["quality"].shape[0]):`; `np.add.at(counts, (tr0[valid], bins[valid]), 1)`.

iii. The notes emphasize eliminating one-spike/one-trial smoothing overhead and using a 2-D convolution, while accepting ragged video loops.

## 11-c. What processing does the code repeat multiple times?

i. The common interpolation path is repeated separately for tongue x/y, each paw x/y, and motion energy. Per-trial output assembly also repeatedly copies the identical time input. Feature lookup and motion loading occur once per session, and clock shift is computed once.

ii. Repeated calls include `interp_matlab(aligned_ft, ..., target_absolute)` and `input_trials.append(time_input.copy())`.

iii. The agent justifies per-stream interpolation because each feature/camera may have different availability, while caching session-wide data and constants.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes unit metadata and behavioral feature lists used only as metadata, continuous behavior arrays later reduced to categorical outputs, and optional diagnostic plots/histograms. `velocity_from_xy` returns a visibility array that callers discard. The `left` and `no` locals in `trial_labels` are read but not needed after trial validation.

ii. `speed, _ = velocity_from_xy(...)`; `return neural, all_meta, stats`; optional `processing_plot(...)`.

iii. The notes regard continuous traces as necessary to calculate session medians and diagnostics, and limit plots to two sessions; metadata supports auditability.
