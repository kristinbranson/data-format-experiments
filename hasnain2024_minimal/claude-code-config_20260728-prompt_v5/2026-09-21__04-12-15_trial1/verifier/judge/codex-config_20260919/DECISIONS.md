# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 44 author-selected sessions (25 `Ephys_Behavior`, 19 `RandomizedDelay_Ephys_Behavior`) and their probes, then loads each `data_structure_<animal>_<date>.mat`. `SessionData` branches between MATLAB v7.3/HDF5 and v5/scipy layouts. Motion energy is loaded separately.

ii. `SESSION_META = [...]`; `sess = SessionData(data_path)`; `me_data = load_motion_energy(me_path)`

iii. The trajectory says it used the authors' loading scripts, excluded inhibition/behavior-only data and commented-out sessions, and selected the 44 sessions (14 mice) whose probe assignments were supported by those scripts.

## 1-b. How are the data split into subjects?

i. Subject identity is the hard-coded `animal` string. Subjects are accumulated in first-encounter order, and each retained session gets its index.

ii. `if animal not in subjects_set: subjects_set.append(animal)` and `all_subject_idx.append(subjects_set.index(animal))`

iii. The trajectory identifies animal IDs from filenames/loading scripts and reports 14 mice across the two paradigms.

## 1-c. How are the data split into sessions?

i. Every `SESSION_META` tuple is one session and becomes one element of the session-level `neural`, `input`, and `output` lists unless a file, trials, or units fail inclusion checks.

ii. `for animal, date, probes, data_dir in SESSION_META:` followed by `all_neural.append(session_neural)`

iii. The agent treated the author loading-script session/date entries as the session definition and combined both task folders into one dataset.

## 1-d. How are the data split into trials?

i. Trials are indexed from `bp.Ntrials`; behavioral arrays and `goCue` are truncated to that count. Spikes use their 1-based `clu.trial` labels, while video and motion-energy entries use zero-based Python trial indices.

ii. `n_trials_total = sess.get_n_trials()`; `aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]`; `for i, ti in enumerate(valid_idx):`

iii. The trajectory recognized that Bpod fields, cluster trial labels, trajectories, and motion-energy cells already encode trials, so boundaries need not be inferred.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials with no photostimulation, no early lick, and a hit/miss/no flag. After neural processing it removes any selected trial whose entire retained neural window is zero, and skips sessions with fewer than two remaining trials.

ii. `valid_mask = (stim_enable == 0) & (early == 0) & ((hit == 1) | (miss == 1) | (no == 1))`; `has_spikes = np.array([np.any(trialdat[:, :, ti] != 0) for ti in valid_idx])`

iii. The trajectory cites the paper for removing early-lick and photostimulation trials. It added the all-zero check after validation exposed late behavioral trials recorded after ephys ended.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probes' cluster `trial`, `trialtm`, and `quality` fields, plus per-trial `bp.ev.goCue`. The loader also reads `tm`, though it is unused.

ii. `units.append({'tm': tm, 'trial': trial, 'trialtm': trialtm, 'quality': quality})`

iii. The agent identified `trialtm` as spike time on the behavioral trial clock and the go cue as the requested alignment event.

## 2-b. How is the `neural` data processed?

i. For each unit and trial, aligned spikes are histogrammed into 5-ms bins, converted to Hz, and passed through the custom `causal_gaussian_smooth` with a 15-sample Gaussian window. Probes are concatenated; no normalization or baseline subtraction is applied.

ii. `counts = np.histogram(aligned[mask], bins=edges)[0]`; `fr = counts.astype(float) / DT`; `trialdat[:, i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, 'reflect')`

iii. The trajectory interpreted the reference as requiring “causal Gaussian smoothing with a window size of 15” and chose reflected boundaries.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It drops clusters with labels `garbage`, `gabrga`, `noisy`, or `real?`, and also drops blank labels. It then retains units with mean processed firing rate above 1 Hz, computed over every trial/window. Sessions must retain at least 10 units.

ii. `u['quality'].lower() not in {q.lower() for q in EXCLUDED_QUALITIES} and u['quality'] != ''`; `fr_mask = mean_frs > LOW_FR`

iii. The trajectory cites manual cluster quality, the paper's “exceeding 1 Hz” rule, and a paper-derived minimum of 10 units per session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time before binning on the common −2.5-to-2.5-s grid.

ii. `aligned = unit['trialtm'] - go_cue_times[unit['trial'] - 1]`

iii. The trajectory explicitly selected `goCue`, matching the prompt and the paper's alignment code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 5 ms (200 Hz), with 1,000 bins from −2.5 to +2.5 s. Spikes are rebinned from event times; video streams are interpolated to bin centers.

ii. `DT = 1 / 200`; `edges = np.linspace(TMIN, TMAX, n_bins + 1)`; `time_axis = edges[:-1] + DT / 2`

iii. The agent chose the paper/default parameter `dt=1/200` rather than a tutorial's 10-ms example.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common time grid defined relative to the raw `bp.ev.goCue` alignment event, not a separately measured raw variable.

ii. `go_cue = sess.get_event_field('goCue')`; `time_axis = edges[:-1] + DT / 2`

iii. The trajectory describes time from go cue as the sole continuous decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It computes the centers of 1,000 equal 5-ms bins and copies the same `(1, 1000)` row into every trial.

ii. `session_input.append(time_axis.reshape(1, -1).copy())`

iii. The chosen window and resolution were taken from the paper's default parameters.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the center coordinate of the exact edges used to bin go-cue-shifted spikes.

ii. `trialdat = bin_and_smooth_spikes(..., edges, time_axis)` and `session_input.append(time_axis.reshape(1, -1).copy())`

iii. The agent intentionally used one common grid for decoder input, neural data, and resampled behavior.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It derives direction from per-trial `bp.hit`, `bp.miss`, `bp.R`, and `bp.L`.

ii. `hit = sess.get_bp_field('hit')`; `miss = ...`; `R = ...`; `L = ...`

iii. The trajectory reasoned that R/L encode the instructed side: a hit uses that side, a miss uses the opposite, and an ignore has no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It assigns left=0, right=1, none=2; hits use the instructed side and misses reverse it. The scalar trial label is broadcast across all time bins.

ii. `elif miss[ti] == 1 and R[ti] == 1: lick_dir[i] = 0`; `out[0, :] = lick_dir[i]`

iii. The agent chose broadcasting to place constant and time-varying outputs in one uniform `(6, n_time)` matrix.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from the per-trial `bp.autowater` flag.

ii. `autowater = sess.get_bp_field('autowater')[:n_trials_total]`

iii. The agent interpreted autowater trials as water-cued and the remainder as delayed-response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is mapped to WC=0 and all other trials to DR=1, then broadcast in time.

ii. `context = np.where(autowater[valid_idx] == 1, 0, 1)`; `out[1, :] = context[i]`

iii. This mapping follows the task semantics and requested class order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` and `bp.miss`; `bp.no` participates in trial validity but is not needed after defaulting outcome to ignore.

ii. `outcome = np.full(n_valid, 2, dtype=np.int64)`; `if hit[ti] == 1 ... elif miss[ti] == 1 ...`

iii. The trajectory treats the three Bpod outcome flags as mutually exclusive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It maps miss to incorrect=0, hit to correct=1, and everything else to ignore=2, then broadcasts the label.

ii. `outcome[i] = 1` / `outcome[i] = 0`; `out[2, :] = outcome[i]`

iii. The coding follows the prompt's output-value order.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses bottom-camera (`traj` camera index 1) DLC `ts` x/y/confidence for hard-coded feature 0 (`top_tongue`), its `frameTimes`, the session video offset from `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.bitStart`, and each trial's go cue.

ii. `ts, ft = sess.get_traj_trial(1, trix)`; `compute_velocity_from_dlc(ts, aligned_ft, 0, time_axis)`

iii. The trajectory chose bottom-camera tongue tracking as a representative view and used bitcode-derived clock correction.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. It takes frame-to-frame Euclidean displacement, divides by a fixed 1/400 s, requires confidence ≥0.6 at both adjacent frames, and linearly interpolates velocity and visibility onto neural bin centers. Missing/interpolation-edge velocity becomes zero but is marked invisible.

ii. `vel = np.sqrt(dx**2 + dy**2) / dt_video`; `vis = (conf[:-1] >= CONF_THRESH) & (conf[1:] >= CONF_THRESH)`; `vel_interp = np.interp(...)`

iii. The agent describes velocity as the derivative magnitude and selected 0.6 as a “typical DLC p_cutoff”; it did not find a paper-specific threshold.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over visible interpolated values. Visible values below it are 0, values at/above it are 1, and invisible bins are 2.

ii. `tongue_disc = discretize_per_session(tv_valid, tvis_valid)`; `thresh = np.percentile(all_valid, 50)`

iii. This directly implements the requested per-session 50th-percentile split and not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by the session video-clock offset and trial go cue, then interpolated at the common 5-ms bin centers.

ii. `aligned_ft = ft - vidshift - go_cue[trix]`; `np.interp(time_axis, ft_vel, vel, ...)`

iii. The trajectory verified that the computed offset is approximately 0.49 s and preferred it to a rough 0.5-s constant.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera DLC features 4 (`top_paw`) and 5 (`bottom_paw`), their x/y/confidence and frame times, plus the same offset and go cue.

ii. `compute_velocity_from_dlc(ts, aligned_ft, 4, time_axis)` and `compute_velocity_from_dlc(ts, aligned_ft, 5, time_axis)`

iii. The trajectory says paws are bottom-view features and decided to use visibility confidence despite discussion that non-tongue gaps may be filled in the reference pipeline.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw is processed like tongue. Where both are visible, their maximum speed is used; where one is visible, that paw is used; neither visible gives zero/invisible.

ii. `combined_vel = np.where(pvis_top & pvis_bot, np.maximum(pv_top, pv_bot), np.where(pvis_top, pv_top, np.where(pvis_bot, pv_bot, 0.0)))`

iii. The agent chose a combined two-paw signal to retain a paw measurement when either tracked paw was visible.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the median of all visible combined-paw values in that session: below=0, at/above=1, invisible=2.

ii. `paw_disc = discretize_per_session(pv_valid, pvis_valid)`

iii. This follows the requested session-specific 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. It uses bottom-camera frame times corrected by video offset and go cue, then linear interpolation at neural bin centers.

ii. `aligned_ft = ft - vidshift - go_cue[trix]`; `np.interp(time_axis, ft_vel, vel, ...)`

iii. The agent used a common time grid to keep output columns synchronized with neural columns.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads per-trial values from the separate `motionEnergy_<animal>_<date>.mat` file and uses side-camera frame times, video offset, and go cue for timing.

ii. `me_data = load_motion_energy(me_path)`; `_, ft = sess.get_traj_trial(0, trix)`

iii. The trajectory identified the standalone files as the complete source and side-camera timing as matching the authors' motion-energy loader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. It unwraps several MATLAB layouts, truncates energy and timestamps to their common length, linearly interpolates to bin centers, and fills internal and edge NaNs by nearest/linear index interpolation.

ii. `me_aligned[:, trix] = np.interp(time_axis, aligned_ft[:min_len], me_trial[:min_len], left=np.nan, right=np.nan)`; `np.interp(np.arange(n_time), np.where(valid)[0], col[valid])`

iii. The trajectory believed nearest filling matched `loadMotionEnergy.m`; it noted this eliminates most/no `no_video` labels.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median over all non-NaN aligned values in the session defines below=0 and at/above=1; remaining NaNs are no-video=2.

ii. `me_disc = discretize_per_session(me_valid, ~np.isnan(me_valid))`

iii. This implements the prompt's session-specific median and no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are offset-corrected and go-cue-centered, then motion energy is interpolated onto neural bin centers.

ii. `aligned_ft = ft - vidshift - go_cue[trix]`; `np.interp(time_axis, aligned_ft[:min_len], me_trial[:min_len], ...)`

iii. The agent used the side camera because the reference motion-energy code associates the trace with that view.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many accessors catch errors and return missing values; failed video-offset calculation falls back to 0.5 s. Missing trajectories yield all invisible, missing motion energy remains class 2, unequal energy/time lengths are truncated, partial motion-energy NaNs are filled, absent data files/sessions are skipped, and late all-zero-neural trials are dropped.

ii. `except Exception: return 0.5`; `if ts is None ... return np.zeros(n_time), np.zeros(n_time, dtype=bool)`; `min_len = min(len(aligned_ft), len(me_trial))`

iii. The trajectory emphasizes robustness to mixed MATLAB formats and added the zero-neural trial filter after verification. It accepted nearest-filled motion energy and silent exception fallbacks as practical recovery.

## 11-a. What are the most time-consuming steps of the code?

i. The conversion repeatedly loops over units/trials for spike histograms and smoothing and over trials for video extraction/interpolation; file loading is also substantial. The later decoder training was far more expensive but is outside conversion.

ii. `for i, unit in enumerate(units):` / `for j in range(n_trials):`; `for trix in range(min(n_trials_total, n_traj_trials)):`

iii. The trajectory explicitly planned efficiency for 44 sessions but gives no conversion profiling; it reports decoder training consuming many CPU cores and minutes.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested unit-by-trial spike loop could be replaced by a 2-D histogram over trial and time. Per-trial lick/outcome loops can be boolean array assignments. Some video loops are inherently ragged, though interpolation remains per trial.

ii. `for i, unit in enumerate(units): for j in range(n_trials):`; `for i, ti in enumerate(valid_idx):`

iii. The trajectory contemplated efficiency but implemented straightforward loops and did not justify why the rectangular spike operation remained nested.

## 11-c. What processing does the code repeat multiple times?

i. It reconstructs the lower-cased exclusion set for every probe, performs a separate spike mask/histogram/smoothing for every unit-trial pair, calls trajectory loading repeatedly for the same trials/cameras, and computes velocity interpolation separately for tongue and both paws.

ii. `{q.lower() for q in EXCLUDED_QUALITIES}` inside the probe loop; `mask = unit['trial'] == (j + 1)`; repeated `compute_velocity_from_dlc(...)`

iii. The trajectory does not discuss these repetitions; its priority was compatibility with heterogeneous files.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads each unit's absolute `tm` but never uses it, reads `bp.no` and `L` where outcome/complement logic could avoid them, bins/smooths all trials before selecting valid trials, and computes video/motion arrays for all trials before retaining `valid_idx`.

ii. `'tm': tm`; `trialdat = bin_and_smooth_spikes(..., n_trials_total, ...)`; `tongue_vel = np.zeros((n_time, n_trials_total))`

iii. The trajectory does not justify these discarded computations; they arise from processing complete sessions before final filtering.
