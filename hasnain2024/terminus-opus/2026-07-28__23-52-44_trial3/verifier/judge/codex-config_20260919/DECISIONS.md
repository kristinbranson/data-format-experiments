# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 44 author-selected sessions, their folder and one probe, then loads each `data_structure_*.mat` and companion motion-energy file. It supports MATLAB v7.3 with HDF5 and v5 with SciPy.

ii. `for sess_meta in sessions_to_process: result = process_session(sess_meta, PARAMS, ...)` and `return load_session_h5(filepath, probe_num)` with fallback `return load_session_v5(filepath, probe_num)`.

iii. The notes say the inventory and probe choices came from the authors' loading scripts and that both MATLAB formats and three motion-energy layouts occur.

## 1-b. How are the data split into subjects?

i. Subject IDs are the `anm` values in `SESSION_META`. Subjects are accumulated in first-session order and each retained session receives an index.

ii. `subj = result['subject']; if subj not in all_subjects: all_subjects.append(subj); subject_idx.append(all_subjects.index(subj))`

iii. The notes report 14 unique animals across the 25 fixed-delay and 19 randomized-delay sessions.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date file is one session and one outer-list element; missing or under-populated sessions can be skipped.

ii. `session_id = f"{anm}_{date}"` and `all_neural.append(result['neural'])`.

iii. The agent followed the loading scripts, excluding randomized-delay files commented out there, and validated 44 resulting sessions.

## 1-d. How are the data split into trials?

i. Behavioral array indices define trials; spike `trial` numbers are converted from 1-based values. Retained indices select neural, input, and output trials.

ii. `for trial_idx in range(ntrials): trial_num = trial_idx + 1` and `for i, trial_idx in enumerate(valid_trial_indices):`.

iii. The notes map trialwise Bpod fields directly and treat the spike trial labels as authoritative.

## 1-e. How are trials filtered based on quality controls?

i. The agent removes early, no-response/ignore, and stimulation trials, requires hit or miss, and later removes trials whose complete retained-neuron matrix is zero.

ii. `valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']` followed by `& (session_data['hit'] | session_data['miss'])`; later `if np.all(ni == 0): continue`.

iii. The notes explicitly say “Exclude early, no-response, stim trials; keep hit+miss”; the all-zero rule was added for behavioral trials extending beyond ephys recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probe's cluster `trialtm`, `trial`, and `quality`, plus trial `goCue` for alignment.

ii. `aligned_times = trialtm[spike_mask] - goCue[trial_idx]`.

iii. The notes identify `clu.trialtm - goCue` and the author-selected probes as the source.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed in 5-ms bins, divided by 0.005 to Hz, and smoothed separately per neuron and trial with the agent's custom one-sided (“causal”) 15-sample kernel.

ii. `counts, _ = np.histogram(aligned_times, bins=edges); fr = counts.astype(np.float32) / dt; fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')`.

iii. The agent says this matches `getSeq.m`/`mySmooth.m`; it chose the code's apparent causal Gaussian interpretation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes quality labels `garbage`, `gabrga`, `noisy`, and `real?`, retains empty labels, removes units with mean window rate at or below 0.5 Hz, and requires at least ten units per session.

ii. `excluded_qualities = {'garbage', 'gabrga', 'noisy', 'real?'}` and `keep_mask = mean_fr > params['low_fr']` where `low_fr` is `0.5`.

iii. The notes acknowledge the paper's 1-Hz rule but choose 0.5 Hz from `getDefaultParams.m`; empty labels are retained to match `findClusters.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before binning.

ii. `aligned_times = trialtm[spike_mask] - goCue[trial_idx]`.

iii. The notes state this matches `alignSpikes.m` and the requested go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The common grid contains 1000 5-ms bins from -2.5 to +2.5 s. Spikes are binned; video variables are linearly interpolated to bin centers rather than averaged within bins.

ii. `edges = np.arange(tmin, tmax + dt, dt); time_axis = edges[:-1] + dt / 2`, with `dt = 1.0 / 200.0`.

iii. The agent cites the paper code's `dt`, `tmin`, and `tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the configured go-cue-aligned window, not copied from a raw field; raw `goCue` establishes the alignment convention.

ii. `time_axis = edges[:-1] + dt / 2`.

iii. The notes define it as the common time axis around the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are calculated and reshaped to `(1, 1000)`, then copied for every trial.

ii. `time_input = time_axis.reshape(1, -1).astype(np.float32)`.

iii. The notes describe a continuous, time-varying decoder input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses centers of the exact edges used to bin go-cue-relative spikes.

ii. `counts, _ = np.histogram(aligned_times, bins=edges)` and `time_axis = edges[:-1] + dt / 2`.

iii. The shared grid was intentionally used for all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is taken directly from Bpod `R` on the retained hit/miss trials; `L` is loaded but not used.

ii. `lick_direction = session_data['R'][valid_trial_indices].astype(int)`.

iii. The mapping plan says `bp.R`, with L=0 and R=1.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Boolean `R` is cast to integer and broadcast across time. Ignore trials are absent, so no “none” category is produced.

ii. `output[0, :] = lick_direction[i]` and `['left', 'right']`.

iii. The agent justified the binary mapping only after deciding to discard no-response trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context derives from `bp.autowater`.

ii. `context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)`.

iii. The notes map autowater trials to WC and all others to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater != 0` becomes WC=0 and zero becomes DR=1; the value is broadcast through the trial.

ii. `output[1, :] = context[i]` and `['WC', 'DR']`.

iii. This direct relabeling follows the requested category ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived solely from `bp.hit` after filtering to hit-or-miss trials.

ii. `outcome = session_data['hit'][valid_trial_indices].astype(int)`.

iii. The notes map hit to correct and the remaining retained miss trials to incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit is cast to correct=1, miss implicitly becomes incorrect=0, and the value is broadcast. Ignore is never represented.

ii. `output[2, :] = outcome[i]` and `['incorrect', 'correct']`.

iii. This follows from the agent's explicit choice to exclude no-response trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side camera's DLC feature named `tongue`: x/y coordinates, frame times, and dropped-frame metadata; it does not read likelihood or combine the bottom-camera tongue view.

ii. `cam0 = session_data['traj'][0]` and `if fn.lower() == 'tongue': tongue_idx = fi`.

iii. The notes describe DLC tongue velocity and baseline filling, but do not justify omitting the second view or likelihood.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y are linearly interpolated to 5-ms centers, all missing positions are filled with session-wide means, gradients are computed per sample (without division by seconds), velocities at originally missing positions are set to zero, and speed is Euclidean magnitude.

ii. `all_x_filled[np.isnan(all_x_filled)] = mean_x`; `xvel = np.gradient(all_x_filled[:, trial_idx])`; `speed = np.sqrt(xvel**2 + yvel**2)`.

iii. The agent believed baseline filling followed `setTongueBaselinePosition`; trajectory reasoning notes recognize that invisibility dominates and intentionally treat it as zero motion.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A median is computed over all non-NaN retained-trial values. Normally values `>=` median are high; if median equals the minimum, strict `>` is used. NaNs become low=0. There is no not-visible=2 category.

ii. `threshold = np.percentile(valid_vals, 50)`; conditional `(vel_data > threshold)` versus `(vel_data >= threshold)`; `discretized[np.isnan(vel_data)] = 0`.

iii. The strict-inequality exception was added because tongue median was zero and `>= 0` made every bin high; the agent favored useful class balance despite the literal specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-clock shift and trial go cue are subtracted from side-camera frame times, then positions are interpolated to neural bin centers.

ii. `aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]` and `fx(time_axis)`.

iii. The agent says the shift follows `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses the second camera and prefers the `top_paw` DLC x/y feature, falling back to any paw feature; frame times and dropped-frame metadata are used, but likelihood is not.

ii. `cam1 = session_data['traj'][1]` and `if 'top_paw' in fn.lower(): paw_idx = fi`.

iii. The notes choose `top_paw` from the top camera and mention baseline-derivative subtraction.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated to 5-ms centers, missing values are nearest/interior-filled, gradients per sample are taken, median first differences are subtracted, remaining gaps are filled, and x/y derivatives are combined.

ii. `x_interp = fx(time_axis)`; `arr[nans] = np.interp(...)`; `xvel = np.gradient(x_interp)`; `xvel -= basederiv_x`.

iii. The agent says this follows the reference's non-tongue baseline derivative treatment, though the code comment admits smoothing is skipped.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the same retained-trial session median and minimum/strict-`>` exception as tongue; missing values become low rather than not-visible.

ii. `paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)`.

iii. The notes target a roughly balanced median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Second-camera frame times are shifted to the behavior clock, made relative to go cue, and x/y positions are linearly interpolated at neural bin centers.

ii. `aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]`.

iii. The shared video-offset method is claimed to match the authors' code.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from each session's separate `motionEnergy_*.mat` trace, with side-camera frame times supplying timestamps.

ii. `me = load_motion_energy(me_path)` and `ft = cam0['trials'][trial_idx]['frameTimes']`.

iii. The agent found and handled direct, standard, and nested file layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-frame trace is linearly interpolated to neural bin centers. If lengths mismatch, a fabricated 400-Hz time axis and offset formula is attempted. Remaining edge NaNs are nearest-filled.

ii. `f_interp = interp1d(aligned_ft, trial_me, kind='linear', ...)`; fallback `me_times = np.arange(len(trial_me)) / 400.0`; then `col[nans] = np.interp(...)`.

iii. The notes describe ME as “interpolated to neural time”; alternate layouts and initial loading failures motivated defensive fallbacks.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It uses the computed retained-trial session median—not the file's `moveThresh`—with the same strict-`>` minimum exception. Missing values become low, and there is no no-video=2 category.

ii. `me_disc, me_thresh = discretize_velocity(valid_me)`.

iii. The agent followed the prompt's per-session 50th percentile, aiming for an approximately 50/50 output.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video shift and go cue, then the trace is interpolated to neural centers.

ii. `aligned_ft = ft - vidshift - goCue[trial_idx]`.

iii. The offset is said to reproduce `findVideoOffset.m`; the notes use the same alignment for all video measures.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Broad fallbacks are used: loader exceptions switch formats; absent ME returns `None`; offset failure becomes zero; dropped/missing video often remains NaN then is filled or classified low; all-zero neural trials are removed. Many processing exceptions are silently ignored.

ii. `except: vidshift = 0.0`, `except: pass`, `discretized[np.isnan(vel_data)] = 0`, and `if np.all(ni == 0): continue`.

iii. The agent framed these as robustness to heterogeneous files and considered zero-neural trials to occur after recording ended.

## 11-a. What are the most time-consuming steps of the code?

i. Per notes, file loading and per-session processing dominate; the full conversion took about 247 seconds. Nested spike binning and repeated video interpolation are prominent computational work.

ii. `for neuron_idx, clu in enumerate(valid_clusters):` nested with `for trial_idx in range(ntrials):`.

iii. The notes estimate roughly 4.4 minutes and report load times for each session.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike counting loops over every neuron and every trial, although a 2-D histogram can bin a neuron's spikes across all trials. Trialwise interpolation, gradients, NaN filling, output assembly, and all-zero checks also loop, with ragged video making some loops reasonable.

ii. `for neuron_idx, clu ...: for trial_idx in range(ntrials): ... np.histogram(...)`.

iii. The agent did not document vectorization opportunities; it focused on correctness and acceptable runtime.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly constructs/interpolates coordinates per feature and trial, repeatedly fills NaNs, copies the identical time input for every trial, and loops once to build then again to filter trial outputs.

ii. `fx = interp1d(...)`, `fy = interp1d(...)` occurs in tongue and paw paths; `time_axis.reshape(1, -1).astype(np.float32)` is inside the trial loop.

iii. No explicit justification is given beyond separate, readable pipelines for each modality.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads lick times, `L`, sample/delay events, likelihood-bearing trajectory channels, dropped-frame counts, and motion-energy `moveThresh` that do not reach the output. It initially creates an incorrect trial-length `brain_region_idx` and immediately replaces it. Optional plotting computes additional aggregates only for diagnostics.

ii. `data['lickL'] = []`, `data['lickR'] = []`; initial `'brain_region_idx': [np.zeros(len(sess_neural), ...)]`, followed by `data['brain_region_idx'] = []`.

iii. Extra fields support exploration/robust parsing; plots were used as sanity checks. The redundant brain-region construction is acknowledged by the inline “Fix” comment.
