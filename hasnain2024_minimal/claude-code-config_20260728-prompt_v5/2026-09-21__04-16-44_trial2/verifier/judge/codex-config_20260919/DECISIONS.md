# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 44 author-selected sessions from the fixed-delay and randomized-delay folders, including the selected probe(s), and loads each `data_structure_<animal>_<date>.mat`. It tries an HDF5/v7.3 reader first and falls back to `scipy.io.loadmat`; motion energy is loaded separately.

ii. `SESSION_META = [('EKH1', '2021-08-07', [2], EPHYS_DIR), ...]` and `try: sess = load_session_h5(fpath, probe_list)\nexcept Exception: sess = load_session_scipy(fpath, probe_list)`.

iii. The trajectory says the agent transcribed sessions/probes from the authors' loading scripts, deliberately excluded behavior-only or author-excluded dates, and chose to include both experiment folders, yielding 44 sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is the `anm` field in each `SESSION_META` tuple. Subjects are accumulated in first-appearance order and each retained session receives the corresponding index.

ii. `if anm not in all_subjects: all_subjects.append(anm)` and `subject_idx_list.append(all_subjects.index(anm))`.

iii. The trajectory identifies animal IDs from loading-script/session names and reports the combined dataset as 14 mice.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date entry and its MATLAB file is one session and one outer-list element in `neural`, `input`, and `output`.

ii. `for anm, date, probes, data_dir in SESSION_META:` followed by `all_sessions.append(result)`.

iii. The agent treated the authors' session lists as authoritative and kept fixed- and randomized-delay recordings as separate sessions in one dataset.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the trial count. Per-trial behavioral arrays, camera entries, and spike `trial` labels are indexed with zero-based Python indices (spike labels are compared to `ti + 1`). Each retained trial becomes one matrix/list entry.

ii. `for ti in range(ntrials):` and `spk_mask = spike_trials_raw == (ti + 1)`; later, `for ti in valid_trials: neural_trials.append(...)`.

iii. The trajectory inspected both MATLAB layouts and repeatedly repaired unwrapping so all 44 files' trial structures could be read.

## 1-e. How are trials filtered based on quality controls?

i. Trials with `stim.enable` or `early` set are removed, and sessions with fewer than two remaining trials are skipped. The agent does not remove trials beyond the last electrophysiology recording, despite noticing zero-neural late trials during verification.

ii. `trial_mask = ~sess['stim_enable'] & ~sess['early']`; `valid_trials = np.where(trial_mask)[0]`; `if len(valid_trials) < 2: return None`.

iii. The trajectory cites the paper for excluding stimulation and early-lick trials. After validation warned about late all-zero neural trials, it called them likely post-recording trials but did not add a cutoff.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probes' `obj.clu` cluster fields `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment.

ii. `spike_trials = ...['trial']`, `spike_trialtm = ...['trialtm']`, and `aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]`.

iii. The trajectory explicitly chose the probes from the authors' scripts and concatenated probes for two-probe sessions.

## 2-b. How is the `neural` data processed?

i. Per unit and trial, aligned spikes are histogrammed into 10 ms bins, divided by 0.01 s to obtain Hz, and smoothed with a custom 15-bin causal half-Gaussian using reflected prefix padding. Probe populations are concatenated; no normalization or baseline subtraction is applied.

ii. `counts, _ = np.histogram(aligned_times, bins=edges)`; `fr = counts / DT`; `trialdat[:, ui, ti] = smooth_data(fr, SMOOTH_WINDOW, BC_TYPE)`.

iii. The agent believed a tutorial's `dt=1/100` and causal `mySmooth` were the relevant reference, acknowledging another reference script used 5 ms but choosing 10 ms because it seemed more accessible.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes quality labels `garbage`, `gabrga`, `noisy`, and `real?` case-insensitively, then keeps units with mean firing rate strictly above 1 Hz over valid trials/window. Sessions would be dropped below 10 units. It does not exclude `poor` units.

ii. `EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}`; `good_units = mean_frs > LOW_FR_THRESH`; `if good_units.sum() < MIN_UNITS: return None`.

iii. The trajectory says these labels and the 1 Hz cutoff came from reference code/methods and added the 10-unit rule from the methods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before binning into the common −2.5 to +2.5 s window.

ii. `aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]`.

iii. The trajectory consistently identifies `goCue` as the required/reference alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 non-overlapping 10 ms bins over five seconds. Raw spikes are histogrammed directly into these bins; camera streams are interpolated to their centers.

ii. `DT = 1.0 / 100`; `edges = np.arange(TMIN, TMAX + DT, DT)`.

iii. The agent explicitly resolved a perceived 5 ms versus 10 ms discrepancy in favor of 10 ms based on a tutorial example.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the chosen −2.5 to +2.5 s bin grid, conceptually relative to `bp.ev.goCue`, rather than read as a raw trace.

ii. `time_centers = edges[:-1] + DT / 2` and `input_trials.append(taxis.reshape(1, -1).astype(np.float32))`.

iii. The trajectory chose the paper's go-cue window and its own 10 ms grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It computes bin centers from the edges, producing −2.495, −2.485, …, 2.495 s, then duplicates this row for every trial.

ii. `edges = np.arange(TMIN, TMAX + DT, DT)`; `time_centers = edges[:-1] + DT / 2`.

iii. No separate trajectory justification was given beyond matching the selected temporal window/binning.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact histogram bins used for go-cue-aligned neural activity, so corresponding columns refer to the same intervals.

ii. `taxis = time_centers` is used both for stored input and as the camera target grid; neural uses the corresponding `edges`.

iii. The trajectory intended every stream to share the go-cue grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `bp.hit`, `bp.miss`, and the instructed side `bp.R`/`bp.L`.

ii. `if sess['hit'][ti]: ... elif sess['miss'][ti]: ... else: lick_dir = 2`.

iii. After validation showed no `none` class, the agent recognized R/L as instructed rather than actual direction and derived response direction jointly with outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hits take the instructed side, misses take the opposite side, and ignore/no-response trials become `none`; codes are left 0, right 1, none 2 and are repeated across time.

ii. `lick_dir = 1 if sess['R'][ti] else 0`; for miss, `lick_dir = 0 if sess['R'][ti] else 1`; otherwise `lick_dir = 2`.

iii. The trajectory explains this correction as necessary to represent the animal's actual lick rather than the cue.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from the per-trial `bp.autowater` flag.

ii. `autowater = ...astype(bool)` and `context = 0 if sess['autowater'][ti] else 1`.

iii. The agent used autowater to distinguish water-cued from delayed-response trials and retained DR-only sessions as all-DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It directly maps autowater true to WC (0), false to DR (1), repeated at every time bin.

ii. `out[1, :] = context`.

iii. The trajectory describes this as a direct condition split.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` and `bp.miss`; `bp.no` is loaded but not needed in the final mapping.

ii. `outcome = 1 if sess['hit'][ti] else (0 if sess['miss'][ti] else 2)`.

iii. The trajectory identifies hit, miss, and no-response as correct, incorrect, and ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss maps to incorrect 0, hit to correct 1, and neither to ignore 2, repeated over time.

ii. `out[2, :] = outcome`.

iii. The agent followed the decoder's requested categorical ordering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only bottom-camera `obj.traj` feature `top_tongue`: x/y coordinates, likelihood, and `frameTimes`. It also uses go cue and the session video/behavior clock offset. It does not use side-camera `tongue`.

ii. `tongue_idx = bottom_feat_names.index('top_tongue')`; `trial_tongue_xy.append(...)`; `trial_tongue_conf.append(...)`.

iii. The trajectory chose bottom-camera tongue/paw tracking; after inspecting likelihoods it used confidence to mark the frequently retracted tongue invisible.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. At 400 Hz it takes first differences of raw x/y, their Euclidean magnitude times 400, copies the first velocity sample, and makes likelihood below 0.9 NaN. It does not fill tongue coordinates, smooth positions, split contiguous valid runs, normalize camera scales, or combine views. Linear interpolation maps velocity to 10 ms centers.

ii. `vel = np.sqrt(dx**2 + dy**2) * VIDEO_FPS`; `vel = np.concatenate([[vel[0]], vel])`; `vel[~visible] = np.nan`; then `interp1d(..., kind='linear')`.

iii. The agent cited the methods' exception to nearest-filling for tongue and introduced the 0.9 DLC visibility threshold after diagnosing an all-above-median result.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The per-session median is computed over non-NaN, DLC-visible samples from retained trials. Visible values below it are 0, values at/above it are 1, and other bins are 2 (`not_visible`).

ii. `tongue_thresh = np.percentile(t_vals, 50)` and assignments to `tongue_disc` initialized with 2.

iii. The trajectory follows the prompt's 50th-percentile split and deliberately preserves invisibility as class 2.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame time is transformed as `frameTimes − vidshift − goCue[trial]`; velocity is linearly interpolated and visibility nearest-neighbor interpolated to neural bin centers.

ii. `ft_aligned = ft - vidshift - goCue[ti]`; `f_interp(taxis)` and `f_vis(taxis)`.

iii. The trajectory checked the bitcode-derived shift (about 0.49 s) and intended the shared go-cue grid to align modalities.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw` x/y coordinates, likelihood, and frame times, plus go cue and the video offset.

ii. `paw_idx = bottom_feat_names.index('top_paw')` and `trial_paw_xy.append(...)`.

iii. The trajectory chose the bottom view for both requested kinematic features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Missing x/y samples are nearest-filled, then unsmoothed frame-to-frame Euclidean displacement is multiplied by a fixed 400 Hz. Low-confidence velocities become NaN; velocity is linearly interpolated to 10 ms centers. The resulting aligned array is then nearest-filled along time.

ii. `compute_velocity(..., fill_missing=True)`; `fill_nearest(paw_vel_aligned)`.

iii. The agent interpreted the methods as nearest-filling all non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session median over retained, treated-as-visible non-NaN values separates 0 (below) and 1 (at/above); remaining unavailable bins are initialized as 2.

ii. `paw_thresh = np.percentile(p_vals, 50)` and `paw_disc = np.full(n_timebins, 2, ...)`.

iii. The 50th-percentile rule comes directly from the decoder task.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are clock-corrected and go-cue-centered, then paw velocity is linearly interpolated onto neural bin centers (visibility by nearest interpolation).

ii. `ft_aligned = ft - vidshift - goCue[ti]`; `interp1d(ft_aligned, vel, kind='linear', ...)`.

iii. The trajectory intended this offset/interpolation to put video on the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads the standalone `motionEnergy_<animal>_<date>.mat` file's `me.data` (and reads but does not use `moveThresh`), and pairs each trace with the bottom-camera frame times loaded for tracking.

ii. `me_data_raw = me['data'].item()`; `sess['me_data'] = [me_data_raw[i] ...]`.

iii. The trajectory planned to use the separate motion-energy files alongside DLC data.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. If trace and frame counts match, the precomputed scalar trace is linearly interpolated to 10 ms centers. NaNs are nearest-filled. No spatial recomputation or smoothing is performed.

ii. `interp1d(ft_aligned, me_trial, kind='linear', ...)`; `fill_nearest(me_aligned)`.

iii. The agent treated motion energy as already computed and needing only alignment/interpolation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median of all non-NaN aligned values in retained trials is the session threshold: below is 0, at/above is 1, and unavailable/no-file bins remain 2.

ii. `me_thresh = np.percentile(m_vals, 50)` and `me_disc = np.full(n_timebins, 2, ...)`.

iii. This follows the prompt's per-session 50th-percentile requirement.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It uses the same bottom-camera `frameTimes − vidshift − goCue` time and linearly interpolates to neural bin centers. The code assumes motion energy follows that camera's frame count.

ii. `if len(me_trial) == len(ft): ... me_aligned[:, ti] = f_interp(taxis)`.

iii. The trajectory intended motion energy to be aligned with the same corrected camera clock.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The two MATLAB layouts get custom readers. Empty camera trials are skipped; broad `try/except` blocks silently leave unavailable kinematics as NaN/class 2. Paw and motion-energy NaNs are nearest-filled, tongue gaps remain unavailable. Missing motion-energy loading silently yields no-video. Sessions that fail loading are printed and skipped. Late trials with no ephys are retained as zero neural matrices.

ii. `except Exception: pass`; `fill_nearest(paw_vel_aligned)`; `fill_nearest(me_aligned)`; and `except Exception as e: ... continue`.

iii. The trajectory devoted substantial debugging to nested MATLAB structures, intentionally avoided tongue filling, but acknowledged and left unresolved the late zero-neural trials.

## 11-a. What are the most time-consuming steps of the code?

i. The likely dominant compute work is the nested unit-by-trial spike histogram/smoothing loop, followed by per-trial camera interpolation and repeated MATLAB loading. The agent did not benchmark individual stages.

ii. `for ui in range(n_units):` nested with `for ti in range(ntrials):`, calling `np.histogram` and `smooth_data` each time.

iii. The trajectory shows repeated full conversion/verification runs but gives no profiling-based justification.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike counting loops over every unit and every trial even though each unit's trial/time labels can be binned across trials together. `smooth_data` also loops over columns, and `fill_nearest` loops over each missing sample with a nearest-distance search. Camera trials remain ragged and are less straightforward to vectorize.

ii. `for ui in range(n_units): for ti in range(ntrials): ...`; `for j in range(x_padded.shape[1]):`; `for idx in np.where(mask)[0]:`.

iii. The trajectory did not discuss vectorization; it prioritized correctness across heterogeneous MATLAB encodings.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly lower-cases the exclusion set for every cluster, computes/smooths even excluded trials before discarding them, constructs interpolators separately for velocity and visibility on every feature/trial, and scans NaNs repeatedly during filling and thresholding.

ii. `if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}` and the all-trials loop preceding `for ti in valid_trials`.

iii. No explicit justification was given; the trajectory reflects iterative additions rather than performance refactoring.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused fields (`L`, `no`, quality strings, `moveThresh`), calculates neural and video arrays for trials later excluded by QC, retains a visibility return that is partly overridden for paw, and stores metadata parameters not consumed by decoding. Neural data for excluded trials is built and then discarded.

ii. `L = ...`, `no = ...`, `sess['me_thresh'] = ...`; processing uses `for ti in range(ntrials)` but output uses `for ti in valid_trials`.

iii. The trajectory does not justify these costs; several fields were useful during development/diagnosis but not in the final downstream representation.
