# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed `data_structure_*.mat` only in `data/Ephys_Behavior`, retained animals in a seven-animal probe map, required exactly 12 files, loaded session structures directly with `h5py`, and loaded companion motion-energy files with `scipy.io.loadmat`. It did not load the randomized-delay folder.

ii. `DATA_DIR = ... / "data" / "Ephys_Behavior"`; `for path in sorted(data_dir.glob("data_structure_*.mat")):`; `if len(sessions) != 12: raise RuntimeError(...)`; `with h5py.File(path, "r") as h5:`

iii. In the trajectory the agent reasoned that the requested WC/DR output identified the fixed-delay electrophysiology collection and described this as the Figure 8 two-context cohort. It later reported 12 sessions, 3,116 trials, and 520 units.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from filenames, deduplicated and sorted; each session receives an integer lookup index. This yields seven subjects in the restricted cohort.

ii. `match = pattern.match(path.name)`; `subjects = sorted({animal for _, animal, _ in files})`; `subject_idx.append(subject_lookup[animal])`

iii. The trajectory inspected filenames/session manifests and treated the filename animal token as the subject identity; no further explicit justification was recorded.

## 1-c. How are the data split into sessions?

i. Every matched MATLAB file is one session and one entry in each top-level session list. Only the 12 files found in `Ephys_Behavior` for mapped animals are included.

ii. `sessions.append((path, match.group(1), match.group(2)))`; `for number, (path, animal, date) in enumerate(files, start=1):`; `neural.append(...)`

iii. The agent justified the cohort as the sessions loaded by the supplied Figure 8 scripts, but did not account for the reference's randomized-delay sessions.

## 1-d. How are the data split into trials?

i. A trial index is the zero-based position in the go-cue vector. Behavioral arrays, spike `trial` identifiers, trajectory cells, and motion-energy entries are indexed with that common original trial index; retained trials become individual matrices.

ii. `go = np.asarray(bp["ev/goCue"]).ravel()`; `n_trials = go.size`; `kept_trials = np.flatnonzero(trial_mask)`; `for local_trial, original_trial in enumerate(kept_trials):`

iii. The agent inspected the MATLAB structures and found trial-indexed Bpod, cluster, trajectory, and motion-energy fields. It did not explicitly discuss why go-cue length, rather than `Ntrials`, is authoritative.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept when they are not early-lick or photostimulation trials, have a finite go cue, and are marked hit, miss, or ignore. The agent retains ignore trials. It does not remove behavioral trials occurring after electrophysiology ended.

ii. `trial_mask = ~early & ~stim & np.isfinite(go) & (hit | miss | ignore)`

iii. Comments and trajectory say paper analyses use control trials and omit early licks, while ignore must remain because it is a required decoder class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the animal-selected `obj/clu` probe: per-unit `trial`, `trialtm`, and `quality`, together with `bp.ev.goCue`. The animal-level `ALM_PROBE` map chooses one probe.

ii. `cluster_group = h5[h5["obj/clu"][probe_number - 1, 0]]`; `spike_trial = referenced_array(...["trial"]...) - 1`; `trial_time = referenced_array(...["trialtm"]...)`; `aligned_time = trial_time - go[spike_trial]`

iii. The trajectory says probe selection was copied from the authors' recording/video scripts and that spike processing followed their alignment pipeline.

## 2-b. How is the `neural` data processed?

i. Each selected unit's aligned spikes are histogrammed into trial-by-time counts, then all counts are converted to Hz and passed through a 15-bin, one-sided causalized Gaussian filter with a custom prefix boundary treatment. No normalization or baseline subtraction is applied.

ii. `DT = 0.01`; `np.add.at(result, (spike_trial[valid], bins[valid]), 1.0)`; `kernel[: n // 2] = 0`; `filtered = lfilter(CAUSAL_KERNEL, [1.0], padded, axis=-1)`; `return ... / DT`

iii. The module docstring and comments claim this reproduces the repository's “15-bin causal Gaussian kernel” and exact `mySmooth(..., 'reflect')` behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units with exact, case-sensitive qualities `garbage`, `gabrga`, `noisy`, or `real?` are rejected. Remaining units are retained when a mean of several condition-PSTH means exceeds 1 Hz; sessions with fewer than ten units raise an error. `poor` is not rejected.

ii. `REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}`; `if quality in REJECTED_QUALITIES: continue`; `mean_rate = unit_mean_rate(...)`; `if mean_rate > 1.0:`

iii. The agent explicitly chose case-sensitive matching to mirror MATLAB `isMember`, and described the condition masks as the Figure 8/source low-firing-rate filter and the threshold as the paper's >1 Hz rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The go cue for each spike's trial is subtracted from its within-trial spike time before binning.

ii. `aligned_time = trial_time - go[spike_trial]`

iii. The trajectory inspected `alignSpikes.m`; the code describes `obj.bp.ev.goCue` as the Bpod alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It uses 500 nonoverlapping 10 ms bins from -2.5 to +2.5 seconds, temporally rebins spikes into those bins, and interpolates video-derived signals onto their centers.

ii. `TMIN = -2.5`; `TMAX = 2.5`; `DT = 0.01`; `TIME = np.arange(TMIN, TMAX, DT) + DT / 2`

iii. The agent stated that the bins and source smoothing match the repository, without explaining the deviation from the reference's 5 ms `params.dt`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a synthetic fixed vector of bin centers defined around `bp.ev.goCue`, rather than values read independently from a raw field.

ii. `TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2`; `time_input = TIME.astype(np.float32)[None, :]`

iii. The agent treated the requested go-cue alignment and common neural time axis as defining this input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are generated at 10 ms spacing over the five-second window and copied for every trial.

ii. `input_trials.append(time_input.copy())`

iii. No separate justification was recorded beyond using the neural time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. `TIME` contains the centers of exactly the bins used to histogram aligned spikes, so input column k corresponds to neural column k.

ii. `bins = np.floor((aligned_time - TMIN) / DT)`; `TIME = np.arange(TMIN, TMAX, DT) + DT / 2`

iii. The agent repeatedly described all streams as interpolated/binned onto the neural time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.hit`, and `bp.no`/ignore; the remaining non-hit, non-ignore retained trials are treated as incorrect responses.

ii. `right_target = np.asarray(bp["R"])...`; `if ignore[original_trial]: ... elif hit[original_trial]: ... else:`

iii. The agent comments that R/L is the instructed side, so an incorrect response must be on the opposite side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Ignore maps to none (2), hit maps to the instructed side, and miss maps to its opposite; the scalar class is repeated through time.

ii. `lick_direction = 1 if right_target[...] else 0`; `lick_direction = 0 if right_target[...] else 1`; `output[0] = lick_direction`

iii. The trajectory did not add reasoning beyond the code comment and required three decoder categories.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived directly from `bp.autowater`.

ii. `autowater = np.asarray(bp["autowater"]).ravel().astype(bool)`

iii. The agent identified autowater as the marker for WC versus DR while inspecting the task code.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is WC (0); otherwise context is DR (1). The per-trial class is repeated at every time point.

ii. `context = 0 if autowater[original_trial] else 1`; `output[1] = context`

iii. This implements the requested WC/DR labeling; no additional justification was recorded.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.no` (ignore) and `bp.hit`; any other retained valid-outcome trial is considered incorrect (the code also loaded `miss`).

ii. `hit = ...`; `miss = ...`; `ignore = np.asarray(bp["no"])...`; `if ignore... elif hit... else...`

iii. The agent explicitly retained ignore because the prompt requires it, unlike some paper analyses.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Incorrect, correct, and ignore are encoded as 0, 1, and 2 respectively and repeated through time.

ii. `outcome = 2`; `outcome = 1`; `outcome = 0`; `output[2] = outcome`

iii. The category order follows the requested output values.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` DLC x/y positions and frame times from `obj.traj`, plus go cues and the session bitcode offset. It does not use the bottom-camera `top_tongue` or DLC likelihood values.

ii. `side = h5[trajectory_refs[0]]`; `tongue_index = find_feature_indices(h5, side, "tongue")`; `tongue = trajectory_signal(h5, side, ...)`

iii. Metadata describes side-camera tongue velocity. The trajectory contains no justification for excluding the second tongue view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X and y positions are linearly interpolated onto the 10 ms target grid; speed is `hypot(gradient(x), gradient(y))` in pixels per target bin. There is no likelihood filtering, position smoothing, division by elapsed seconds, two-camera normalization, or combination.

ii. `x = interp_preserving_missing(..., TIME)`; `y = interp_preserving_missing(..., TIME)`; `speed = np.hypot(np.gradient(x), np.gradient(y))`

iii. The agent called this Euclidean frame-to-frame velocity and claimed it followed the source video offset, but did not justify the omitted reference processing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The median of all finite interpolated tongue-speed samples in a session is the threshold: below is 0, at/above is 1, and NaN is 2 (not visible).

ii. `threshold = float(np.percentile(np.concatenate(finite_parts), 50))`; `out[visible] = (signal[visible] >= threshold).astype(np.int64)`

iii. The agent followed the prompt's per-session 50th-percentile requirement.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session offset is computed as median ephys bitcode time minus median behavioral bit-start time. Side-camera frame times are corrected by that offset and the trial go cue, then x/y are interpolated to the neural bin centers.

ii. `return float(np.nanmedian(video_bit_start / fs) - np.nanmedian(bit_start))`; `aligned_frame_time = frame_time[:n] - offset - go_cue`; `np.interp(target_time, ...)`

iii. The agent said this uses the repository's per-session video offset before interpolation. It did not explain using medians rather than the reference's modes.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera DLC `top_paw` x/y positions and frame times, plus go cues and bitcode timing.

ii. `bottom = h5[trajectory_refs[1]]`; `paw_index = find_feature_indices(h5, bottom, "top_paw")`; `paw = trajectory_signal(h5, bottom, ...)`

iii. Metadata identifies the bottom-camera top paw; no further justification was recorded.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw positions are interpolated to 10 ms bin centers and differentiated by sample index; Euclidean gradient magnitude is used. Likelihood filtering and reference position smoothing are absent.

ii. `x = interp_preserving_missing(..., TIME)`; `speed = np.hypot(np.gradient(x), np.gradient(y))`

iii. The agent described the result as Euclidean frame-to-frame velocity but did not justify differences from the supplied kinematic functions.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. All finite paw-speed samples in a session define a median; below/at-or-above/NaN map to 0/1/2.

ii. `paw_cat, paw_threshold = categorize_session_signal(paw_velocity)`

iii. This directly follows the per-session threshold specified by the prompt.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera times are offset-corrected, made relative to that trial's go cue, and positions are interpolated onto `TIME` before velocity is computed.

ii. `aligned_frame_time = frame_time[:n] - offset - go_cue`; `x = interp_preserving_missing(aligned_frame_time, ..., TIME)`

iii. The agent intended all camera signals to share the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads the standalone companion `motionEnergy_<animal>_<date>.mat` trace for each trial and pairs it with side-camera frame times obtained through the tongue trajectory.

ii. `motion_path = path.with_name(path.name.replace("data_structure_", "motionEnergy_"))`; `motion_trials = load_motion_energy(motion_path)`

iii. The agent inspected the motion-energy layouts and source loader and used the standalone file as the processed per-frame signal.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw per-frame motion energy is truncated to the available side-camera time count, linearly interpolated to 10 ms centers, then every remaining NaN is filled with its nearest finite neighbor before categorization.

ii. `me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)`; `motion_energy.append(nearest_fill(me))`

iii. The agent noted that motion energy has one value per side-camera frame. It did not justify nearest-filling missing observations.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The session-wide median of finite, interpolated/filled motion energy is used; values below and at/above are 0 and 1, while sessions/trials with no usable signal remain 2.

ii. `motion_cat, motion_threshold = categorize_session_signal(motion_energy)`

iii. The per-session 50th percentile follows the prompt.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It uses aligned side-camera times returned by the tongue extraction, then interpolates motion energy onto neural `TIME`. If tongue extraction is rejected, motion energy is also marked absent.

ii. `if tongue is None ... motion_energy.append(np.full(..., np.nan))`; `interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)`

iii. The agent reasoned motion energy and side camera have one value per frame and intended to apply the same video offset and go-cue alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid spike-trial indices are dropped; trials require finite go cues; trajectories with unusable dropped-frame metadata, too few times, or malformed arrays become all-NaN and then category 2. Interpolation does not extrapolate, except motion-energy NaNs are nearest-filled. Array length mismatches are silently truncated to the minimum. Hard minimums on sessions/trials/units raise errors.

ii. `valid = (spike_trial >= 0) & ...`; `return np.full(target_time.shape, np.nan, ...)`; `n = min(frame_time.size, x_raw.size, y_raw.size)`; `motion_energy.append(nearest_fill(me))`

iii. The trajectory shows extensive structure inspection and validation, but no explicit unified missing-data rationale. The code docstrings say interpolation preserves missing intervals, although `np.interp` across internal NaNs and subsequent nearest filling complicate that claim.

## 11-a. What are the most time-consuming steps of the code?

i. The agent did not profile individual stages. Based on the implementation, repeated per-unit condition PSTHs and full trial-by-unit smoothing, plus per-trial HDF5 dereferencing/interpolation, are the main computation; file reads are also substantial. The complete 12-session conversion was run and reported as successful.

ii. `for unit in range(...): ... unit_mean_rate(...)`; `rates = smooth_rates(counts)`; `for trial in kept_trials: ... trajectory_signal(...)`

iii. The trajectory reports completion but does not identify or benchmark bottlenecks.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Unit selection/counting and video extraction loop over units/trials because their raw referenced arrays are ragged. Output assembly and some per-signal categorization loops could readily be vectorized; spike counting already uses `np.add.at` within each unit.

ii. `for unit in range(...)`; `for unit_index, (...) in enumerate(selected_units):`; `for trial in kept_trials:`; `for local_trial, original_trial in enumerate(kept_trials):`

iii. The agent did not explicitly discuss vectorization; its implementation favors direct handling of MATLAB cell references and ragged arrays.

## 11-c. What processing does the code repeat multiple times?

i. Every retained unit is smoothed once during condition-based low-rate screening and again after trial histograms are assembled. `trajectory_signal` repeats interpolation logic for tongue and paw on every trial, and the common time input is copied for every trial.

ii. `psth = smooth_rates(hist[None, :])[0]`; later `rates = smooth_rates(counts)`; `tongue = trajectory_signal(...)`; `paw = trajectory_signal(...)`; `input_trials.append(time_input.copy())`

iii. The first smoothing is used only for selection and the second for output, but the trajectory does not identify this repeated work.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It stores every selected unit's quality and computed mean rate in the session result, then immediately removes both before saving metadata. It also computes continuous movement arrays only to discretize them, which is necessary for thresholds but they are then discarded. `miss` is loaded but final outcome branching can infer it after the trial mask.

ii. `"qualities": [item[2] ...]`; `"mean_rates_hz": [...]`; `result.pop("qualities")`; `result.pop("mean_rates_hz")`

iii. The trajectory does not discuss discarded processing; it focuses on validation and final dataset size/performance.
