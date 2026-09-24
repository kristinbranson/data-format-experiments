# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans the two electrophysiology folders for `data_structure_*.mat`, retains only the 44 names in a hard-coded probe map, associates each with its motion-energy file, and uses separate wrappers for MATLAB v7.3/HDF5 and v5 files.

ii. `files = sorted(glob.glob(os.path.join(dpath, 'data_structure_*.mat')))`; `if key not in PROBE_MAP: continue`; `return H5Session(filepath) if is_h5_format(filepath) else V5Session(filepath)`.

iii. It says the author loading scripts define the included sessions/probes, both ephys task folders belong, and several MATLAB layouts must be supported.

## 1-b. How are the data split into subjects?

i. The mouse ID is parsed from the session key before the underscore. Subjects are accumulated in first-encounter order and every retained session gets an index into that list.

ii. `anm = session_key.split('_')[0]`; `if anm not in all_subjects: all_subjects.append(anm)`; `subject_idx.append(all_subjects.index(anm))`.

iii. The notes treat the filename/session key as the reliable animal identifier and report 44 sessions across the ephys cohorts.

## 1-c. How are the data split into sessions?

i. Each retained `data_structure_<animal>_<date>.mat` is one session and one element of the top-level neural/input/output lists; fixed- and randomized-delay folders are processed uniformly.

ii. `for sess_info in sessions: result = process_session(...)`; `all_neural.append(result['neural'])`.

iii. The 44 author-listed recording sessions are the intended analysis set; unlisted and commented-out files are excluded.

## 1-d. How are the data split into trials?

i. Trials are the zero-based rows of the Bpod arrays (`Ntrials`) and spike membership is obtained from each unit's one-based `trial` field. Each retained trial becomes one matrix in each session list.

ii. `for j in range(n_trials): trial_num = j + 1; spk_mask = trial == trial_num`; `for vi in range(n_valid): neural_trials.append(...)`.

iii. The notes identify Bpod fields as per-trial data and preserve their indexing across behavior, spikes, trajectories, and motion energy.

## 1-e. How are trials filtered based on quality controls?

i. Photostimulation and early-lick trials are removed. After neural processing, any trial whose complete retained, smoothed population is zero is also removed. A session is skipped if fewer than two valid trials remain; sessions with fewer than ten units are skipped.

ii. `valid_mask = (stim_enable == 0) & (early == 0)`; `trial_has_spikes = np.any(trialdat > 0, axis=(0, 1))`; `valid_trials = np.array([t for t in valid_trials if trial_has_spikes[t]])`.

iii. The paper omits early licks and analyzes photoinactivation separately. The added zero-neural filter was introduced after validation exposed behavioral trials continuing beyond ephys recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from selected probes in `obj.clu`: each unit's `trialtm`, `trial`, and `quality`, plus Bpod `ev.goCue` for alignment.

ii. `trialtm = unit['trialtm']; trial = unit['trial']`; `spk_times = trialtm[spk_mask] - goCue[j]`.

iii. This follows `alignSpikes.m` and the probe assignments in the authors' session-loading scripts.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed in 10 ms bins from -2.5 to 2.5 s, divided by 0.01 to obtain Hz, and convolved with a normalized causal half-Gaussian of 15 bins after prefix padding. Selected probes are concatenated; no z-scoring or baseline correction is applied.

ii. `DT = 1.0/100`; `counts, _ = np.histogram(spk_times, bins=EDGES)`; `rate = counts.astype(np.float32) / DT`; `trialdat[:, i, j] = smooth_signal(rate)`.

iii. The agent believed `getSeq.m` and `mySmooth.m` specified 10 ms bins, `gausswin(15)`, causal truncation, and reflect-style boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units with lower-cased quality in `{garbage, gabrga, noisy, real?}` are excluded. Units must have mean smoothed rate over retained trials/time strictly above 1 Hz, and sessions must retain at least ten units.

ii. `if quality in EXCLUDE_QUALITIES: continue`; `mean_fr = np.mean(np.mean(trialdat[:, :, valid_trials], axis=2), axis=0)`; `keep_units = mean_fr > LOW_FR`.

iii. The exclusion set and 1 Hz cutoff are attributed to `findClusters.m` and the paper. The notes acknowledge that trial-weighted FR and lower-casing differ slightly from the original condition-weighted/case-sensitive implementation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time before histogramming.

ii. `spk_times = trialtm[spk_mask] - goCue[j]`.

iii. The agent cites the reference formula `trialtm_aligned = trialtm - goCue(trial)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 non-overlapping 10 ms bins over [-2.5, 2.5) s. Spikes are directly binned to that grid; video and motion energy are interpolated to its centers.

ii. `DT = 1.0/100`; `EDGES = np.arange(TMIN, TMAX + DT, DT)`; `TIME = EDGES[:-1] + DT / 2`.

iii. The notes state this was chosen to match the agent's reading of `params.dt`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a generated common time axis defined by the go-cue-aligned window and bin width, rather than a copied raw variable; raw `goCue` is used to align spikes and video to it.

ii. `TIME = EDGES[:-1] + DT / 2`; `input_trials.append(TIME.reshape(1, -1).astype(np.float32))`.

iii. The decoder explicitly requests time from the alignment event, and the agent uses the bin centers.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Centers are computed by adding half a 10 ms bin to all left bin edges, yielding -2.495 through 2.495 s, then copied into every trial.

ii. `TIME = EDGES[:-1] + DT / 2`.

iii. The notes say this mirrors the reference time-axis construction.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the center coordinate of the exact edges used for neural spike histograms, so input column k describes neural column k.

ii. `counts, _ = np.histogram(spk_times, bins=EDGES)` and `TIME = EDGES[:-1] + DT / 2`.

iii. A single module-level grid is shared across sessions and streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses Bpod `R`, `L`, `hit`, `miss`, and `no` for each retained trial.

ii. `hit = sess.get_bp_field('hit')`; `miss = ...`; `no = ...`; `R = ...`; `L = ...`.

iii. R/L encode the instructed side, while hit/miss determine whether the actual lick matched or opposed it; no means no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials take the instructed side, misses take the opposite side, and no-response/unmatched trials get class 2. Codes are left=0, right=1, none=2 and are broadcast over time.

ii. `elif R[ti] == 1 and hit[ti] == 1: lick_dir[vi] = 1`; `elif R[ti] == 1 and miss[ti] == 1: lick_dir[vi] = 0`; `out[0, :] = lick_dir[vi]`.

iii. The initial attempted loop was superseded by a corrected loop after the agent recognized that R/L are trial type, not observed lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived solely from `obj.bp.autowater`.

ii. `autowater = sess.get_bp_field('autowater')`.

iii. The notes identify autowater trials as water-cued and all others as delayed response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater==1` maps to WC=0; otherwise DR=1, then the scalar is broadcast across the trial.

ii. `context = np.array([1 if autowater[ti] == 0 else 0 for ti in valid_trials])`; `out[1, :] = context[vi]`.

iii. This is a direct relabeling specified in the mapping plan.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses the per-trial Bpod flags `hit`, `miss`, and `no`.

ii. `hit = sess.get_bp_field('hit'); miss = ...; no = ...`.

iii. These flags directly encode correct, incorrect, and ignored trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Defaults to ignore=2, then hit maps to correct=1 and miss to incorrect=0; values are broadcast over time.

ii. `outcome = np.full(len(valid_trials), 2)`; `if hit[ti] == 1: outcome[vi] = 1`; `elif miss[ti] == 1: outcome[vi] = 0`.

iii. The class order follows the decoder specification while retaining ignore trials for the other outputs.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera DLC feature named `tongue`: x/y positions and frame times from `obj.traj`, plus Bpod go cues and SpikeGLX/Bpod bit-start fields for clock correction.

ii. `if fn == 'tongue': tongue_idx_side = fi`; `tx = ts[:, 0, tongue_idx_side]`; `aligned_times = frame_times - vidshift - goCue[ti]`.

iii. The agent chose Euclidean DLC tongue speed and noted that invisible tongue frames need a separate class.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. NaN coordinates define invisibility. Invisible positions are filled with the mean visible x/y position, positions are linearly interpolated to neural-bin centers, x/y gradients are combined by Euclidean norm, and speed is forced to zero where interpolated visibility is false.

ii. `baseline_x = np.nanmean(tx[vis_raw])`; `tx_i = np.interp(taxis, aligned_times, tx_filled)`; `speed = np.sqrt(np.gradient(tx_i)**2 + np.gradient(ty_i)**2)`; `speed[~vis_interp] = 0`.

iii. The agent intended to approximate the reference tongue-baseline fill, interpolation, gradient velocity, and zero velocity when invisible; it acknowledges using all visible positions rather than bout starts.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over visible, finite speed values. Visible values below it are 0 and values at/above it are 1; invisible bins are 2.

ii. `thresh_tongue = np.percentile(visible_tongue_vals, 50)`; `tongue_vel_disc[valid, vi] = (tongue_vel[valid, vi] >= thresh_tongue).astype(int)`.

iii. This implements the requested per-session 50th-percentile split and reserves class 2 for not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session clock offset is estimated as mode(neural bit starts)/sampling rate minus mode(behavior bit starts). The offset and trial go cue are subtracted from frame times, then position and visibility are interpolated to `TIME`.

ii. `vidshift = scipy_mode(bitstart_sglx).mode / fs - scipy_mode(bitstart_ev).mode`; `aligned_times = frame_times - vidshift - goCue[ti]`; `tx_i = np.interp(taxis, aligned_times, tx_filled)`.

iii. The clock formula and interpolation are attributed to `findVideoOffset.m` and the reference kinematics pipeline.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses every bottom-camera DLC feature whose name contains `paw`, including both top and bottom paws, with x/y positions, frame times, and the same clock fields.

ii. `paw_indices_bottom = [fi for fi, fn in enumerate(bottom_feats) if 'paw' in fn]`.

iii. The mapping plan says paw velocity comes from bottom-camera paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw, NaN positions are nearest/interior-linearly filled, positions are interpolated onto the neural grid, gradients have median displacement subtracted, and Euclidean speeds are averaged across available paws. Visibility is true if either paw is visible.

ii. `px_filled = np.interp(np.arange(len(px)), valid_idx, px[valid_idx])`; `vx = np.gradient(px_i) - np.median(np.diff(px_i))`; `paw_vel[:, vi] = np.mean(paw_speeds, axis=0)`.

iii. The agent aimed to match nearest filling and drift-subtracted `findVelocity.m`; it calls the remaining baseline difference negligible.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide median over visible finite paw speeds defines below=0 and at/above=1; bins where neither paw is visible are 2.

ii. `thresh_paw = np.percentile(visible_paw_vals, 50)`; `paw_vel_disc[valid, vi] = (paw_vel[valid, vi] >= thresh_paw).astype(int)`.

iii. This directly follows the requested per-session median and not-visible category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are corrected by the session video offset and trial go cue; filled positions and visibility are interpolated to neural time centers.

ii. `aligned_times = frame_times - vidshift - goCue[ti]`; `px_i = np.interp(taxis, aligned_times, px_filled)`.

iii. The same reference video-clock correction is reused for all camera outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from the session's standalone `motionEnergy_<animal>_<date>.mat` trace, and uses side-camera frame times plus clock/go-cue fields for timing.

ii. `me_data, me_thresh = load_motion_energy(me_filepath)`; `_, _, frame_times, is_valid = sess.get_traj_data(0, ti)`.

iii. The separate files exist beside all ephys sessions and occur in several wrapper layouts, which the loader explicitly handles.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The precomputed per-frame scalar is truncated with frame times to the shorter length, linearly interpolated to neural time centers, and any remaining internal NaNs are nearest/interior-linearly filled. The file's `moveThresh` is read but discarded.

ii. `n_me = min(len(me_trial), len(aligned_times))`; `me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])`; `me_data_aligned[:, vi] = me_interp`.

iii. The agent states the spatial motion-energy calculation was already performed and that reference `loadMotionEnergy.m` interpolates it to neural time.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median of all finite aligned values in a session defines below=0 and at/above=1; missing/no-video bins default to 2.

ii. `thresh_me = np.percentile(me_vals, 50)`; `me_disc[valid_t, vi] = (me_data_aligned[valid_t, vi] >= thresh_me).astype(int)`.

iii. This follows the decoder's per-session 50th-percentile instruction rather than the raw file's movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted by the session video offset and trial go cue, paired 1:1 with motion-energy samples (up to the shorter length), and interpolated to the neural centers.

ii. `aligned_times = frame_times - vidshift - goCue[ti]`; `me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])`.

iii. Motion energy is assumed to share the side-camera frame clock, as in the reference loader.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many malformed/missing video operations are caught and silently left as NaN/default class 2. Missing motion-energy files or load failures also yield class 2. Position gaps are filled for velocity computation, mismatched motion/frame lengths are truncated, failed video-offset calculation falls back to 0.5 s, and all-zero neural trials are removed.

ii. `except Exception: return 0.5`; `except Exception: pass`; `n_me = min(...)`; arrays are initialized with `np.nan` or categorical `2`.

iii. The agent documents fixes for alternate MATLAB layouts and ended-recording trials, and uses explicit missing/not-visible classes to keep otherwise usable trials.

## 11-a. What are the most time-consuming steps of the code?

i. The agent does not profile individual stages, but logs each session's elapsed time and total conversion time. Structurally, nested unit-by-trial spike binning, repeated trajectory reads/interpolation, optional plotting, and pickle writing are the dominant candidates.

ii. `for i, unit in enumerate(all_units): for j in range(n_trials): ... np.histogram(...)`; `for vi, ti in enumerate(valid_trials): sess.get_traj_data(...)`.

iii. The notes report a completed full conversion but provide no explicit timing breakdown or optimization justification.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike counting loops over every unit and every trial, categorical outputs loop over trials, discretization loops over trials, and camera processing repeatedly loops over trials/features. Several could be replaced by vectorized trial masks/histograms or array comparisons, though ragged camera arrays still require some iteration.

ii. `for i, unit in enumerate(all_units): for j in range(n_trials):`; `for vi, ti in enumerate(valid_trials):`; `for vi in range(len(valid_trials)):`.

iii. The agent gives no explicit vectorization analysis; it prioritized fidelity and format validation.

## 11-c. What processing does the code repeat multiple times?

i. `get_traj_data` re-reads/dereferences feature names and trial structures for tongue, paw, and motion energy; behavior label construction makes multiple passes; lick direction contains a discarded first pass followed by the real pass; each trial receives a fresh copy of the identical TIME input.

ii. `side_feats, _, _, _ = sess.get_traj_data(0, 0)` and later `sess.get_traj_data(0, ti)`; two separate `lick_dir = ...; for ...` blocks; `input_trials.append(TIME.reshape(1, -1).astype(np.float32))`.

iii. No justification is supplied for the dead lick pass or repeated dereferencing; repeated calls mainly arise from the wrapper API and separate feature streams.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The first lick-direction loop is overwritten completely; `me_thresh` and some validity/metadata values are read but not used; full raw neural arrays include invalid trials until later subsetting; optional plotting computes figures not used by decoding; per-trial labels are redundantly broadcast across 500 bins.

ii. `pass  # Will redo below` followed by `lick_dir = np.full(...)`; `me_data, me_thresh = ...` (threshold unused); `trialdat = np.zeros(..., n_trials)` before `trialdat_valid = ...`.

iii. The first pass reflects an in-code correction. Broadcasting is accepted by the target time-varying output format, while plotting is explicitly optional diagnostic work.
