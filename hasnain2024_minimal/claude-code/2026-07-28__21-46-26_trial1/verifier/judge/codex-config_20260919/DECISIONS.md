# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent searches both ephys directories for every `data_structure_*.mat`, pairs each with a same-named motion-energy file when present, and loads MATLAB v7.3 with `h5py` or older MATLAB files with `scipy.io`. It therefore discovers files rather than using the authors' curated session/probe list.

ii. `data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))` and `session = load_session_data(session_file)`.

iii. The trajectory says both fixed/randomized-delay datasets were ultimately included to maximize useful data; format fallbacks and per-session exception handling were added after load failures. It acknowledged 45 processed sessions versus the paper/reference counts but accepted the discrepancy.

## 1-b. How are the data split into subjects?

i. Subject is parsed as the filename prefix before the first underscore; unique names are sorted and each session receives an integer index.

ii. `animal = parts[0]`; `unique_subjects = sorted(set(all_animals))`; `subject_idx = np.array([unique_subjects.index(a) for a in all_animals])`.

iii. The trajectory treated filenames as the stable source of animal identity while noting a 14-subject result.

## 1-c. How are the data split into sessions?

i. Each discovered `data_structure_<animal>_<date>.mat` is one session. Sessions that error, lack clusters, have fewer than two retained trials, or have no retained units are skipped.

ii. `for idx, sf in enumerate(session_files): result = process_session(...)`; `all_neural.append(result['neural'])`.

iii. The agent chose file-level sessions and robust skipping so malformed/behavior-only files would not abort conversion.

## 1-d. How are the data split into trials?

i. Behavioral arrays define zero-based trial indices. Spike `trial` values are matched to `trial_idx + 1`; video and motion-energy arrays are indexed by the same zero-based retained trial indices.

ii. `valid_trials = np.where(trial_mask)[0]` and `spike_mask = spike_trials == trial_num`.

iii. The trajectory relied on the recorded Bpod trial assignments and corresponding per-trial camera entries rather than reconstructing boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Only non-stimulation, non-early, responding (`hit` or `miss`) trials are kept. Ignore trials are removed. There is no explicit end-of-ephys truncation.

ii. `trial_mask = ((session['stim_enable'] == 0) & (session['early'] == 0) & ((session['hit'] == 1) | (session['miss'] == 1)))`.

iii. The trajectory explicitly decided to exclude early/stim/ignore trials, although it briefly considered keeping all non-early/non-stim trials; the final rationale was matching the analysis conditions while retaining both correct and incorrect responses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each cluster's `trialtm` spike times and `trial` assignments, cluster `quality`, behavioral `goCue`, and probe location metadata.

ii. `spike_times = unit['trialtm']`; `spike_trials = unit['trial']`; `aligned_times = spike_times[spike_mask] - go_time`.

iii. The agent identified `trialtm` as within-trial spike time and subtracted the trial's go cue, following the MATLAB alignment concept.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed per unit and trial into 5-ms bins, divided by 0.005 to Hz, then convolved with a custom one-sided, nominal 15-sample Gaussian. Probes passing the ALM-name test are concatenated.

ii. `counts, _ = np.histogram(aligned_times, bins=edges)`; `fr = counts / dt`; `fr_smooth = causal_gaussian_smooth(fr, smooth_samples)`.

iii. The agent said this matched `mySmooth.m` by zeroing half a Gaussian kernel, though its description and implementation of causality/kernel width were assumptions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Labels `garbage`, `gabrga`, `noisy`, and `real?` are excluded; empty labels are retained. Only probes whose location contains `ALM` are processed. Units with mean processed firing rate above 0.5 Hz are retained.

ii. `excluded = {'garbage', 'gabrga', 'noisy', 'real?'}`; `if 'ALM' not in loc_upper: continue`; `fr_mask = mean_frs > LOW_FR` with `LOW_FR = 0.5`.

iii. The trajectory noticed the paper's 1-Hz criterion but deliberately retained 0.5 Hz “for consistency,” later attributing count discrepancies to this choice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each spike, the go-cue time of its assigned trial is subtracted; aligned spikes are then binned from −2.5 to +2.5 s.

ii. `aligned_times = spike_times[spike_mask] - go_time`.

iii. The agent stated this follows the paper/MATLAB go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 5 ms (200 Hz), with 1000 bins over [−2.5, 2.5). Spikes are rebinned from event times; video streams are linearly interpolated to bin centers rather than averaged in bins.

ii. `DT = 1 / 200`; `EDGES = np.arange(TMIN, TMAX + DT / 2, DT)`; `TIME_AXIS = EDGES[:-1] + DT / 2`.

iii. The agent selected the paper's `dt`, `tmin`, and `tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a synthetic common time grid defined relative to each trial's raw `goCue`, not another measured variable.

ii. `TIME_AXIS = EDGES[:-1] + DT / 2`.

iii. The grid was chosen to represent the requested decoder input on the reference analysis window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers from −2.4975 to +2.4975 seconds are computed once, cast to `float32`, reshaped to `(1, 1000)`, and copied for every trial.

ii. `input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)`.

iii. No additional transformation was justified beyond matching the neural grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the same edges used to histogram go-cue-relative spikes, so columns correspond exactly.

ii. `counts, _ = np.histogram(aligned_times, bins=edges)` and `TIME_AXIS = EDGES[:-1] + DT / 2`.

iii. The trajectory intended all modalities to share the 1000-bin go-cue grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is taken directly from `bp.R` for retained hit/miss trials; `L` is loaded but unused.

ii. `lick_direction = session['R'][valid_trials].astype(np.int32)`.

iii. Because ignore trials were dropped, the agent treated instructed/right side as lick direction and assumed errors did not require flipping it.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `R=1` becomes right and `R=0` becomes left, then the scalar is tiled across all time bins. No “none” class is represented.

ii. `output_trial[0, :] = lick_direction[ti]`; `['left', 'right']`.

iii. The agent described this as direct R/L coding on responding trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It derives solely from `bp.autowater`.

ii. `context = (1 - session['autowater'][valid_trials]).astype(np.int32)`.

iii. The trajectory identified autowater trials as WC and other trials as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is inverted so WC=0 and DR=1, then tiled across time.

ii. `output_trial[1, :] = context[ti]`; `['WC', 'DR']`.

iii. This encoding follows the requested category order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses only `bp.hit` after filtering to hit/miss trials; `miss` is therefore the complement among retained trials.

ii. `outcome = session['hit'][valid_trials].astype(np.int32)`.

iii. The agent intended to retain correct and incorrect trials but deliberately excluded ignores.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit maps to correct=1 and miss maps to incorrect=0, tiled over time. No ignore category exists.

ii. `output_trial[2, :] = outcome[ti]`; `['incorrect', 'correct']`.

iii. The agent considered this binary coding adequate after its responding-trial filter.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-camera (`cam 0`) DeepLabCut `tongue` x/y positions, trajectory frame times, `NdroppedFrames`, video/behavior bit-start offset, and go-cue times. Likelihood is loaded in `ts` but ignored.

ii. `tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)`.

iii. The agent selected the side-camera tongue feature and described video as 400 Hz tracking aligned through the computed shift.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Raw x/y gradients are computed assuming 1/400 s, combined by Euclidean norm, values at NaN positions are forced to zero, and the series is linearly interpolated to neural times. Remaining NaNs are interpolated or an all-missing trial is set to zero. No likelihood filter, position smoothing, second camera, or cross-view normalization is used.

ii. `dx = np.gradient(x_pos, dt_vid)`; `vel = np.sqrt(dx**2 + dy**2)`; `vel[nan_mask] = 0.0`; `interp1d(...)`.

iii. The trajectory reasoned that zero represented an invisible tongue and accepted this despite recognizing that it collapsed the median split.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median over all finite interpolated values is used; values below it are 0 and values at/above it are 1. Category 2 is never produced.

ii. `threshold = np.percentile(valid, percentile)`; `discretized = (values >= threshold).astype(np.int32)`.

iii. The agent repeatedly noted a zero median made virtually everything class 1, but retained the literal `>= 50th percentile` rule rather than preserving not-visible values.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by the session video offset and trial go cue, then linearly interpolated at neural bin centers.

ii. `aligned_frame_times = frame_times[:n_frames] - vidshift - go_time`; `interp_fn(TIME_AXIS)`.

iii. The agent intended clock correction plus go-cue subtraction to reproduce video/neural alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera (`cam 1`) `top_paw` and `bottom_paw` x/y positions plus frame times, dropped-frame flag, video offset, and go cue.

ii. `paw_top = extract_kinematic_feature(..., 1, 'top_paw', ...)`; `paw_bot = extract_kinematic_feature(..., 1, 'bottom_paw', ...)`.

iii. The agent chose the bottom camera and attempted to use both tracked paw landmarks for robustness.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Missing x/y positions are linearly filled, gradients assume 400 Hz, speed magnitude is computed, each landmark is interpolated to neural time, and the two velocities are averaged when both exist. Missing output is replaced by zeros.

ii. `pos[mask] = np.interp(...)`; `paw_vel = (paw_top + paw_bot) / 2.0`.

iii. The agent described this as averaging available top/bottom paw estimates; it did not justify departure from the reference's single top-paw feature.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide finite-value median produces classes 0/1 via `<` versus `>=`; category 2 is never produced, and absent paw data is all class 0.

ii. `paw_vel_disc = discretize_per_session(paw_vel)`; fallback `np.zeros(...)`.

iii. The agent followed its generic median splitter and treated missing streams as zeros.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. It uses the same video-offset/go-cue correction and linear interpolation to `TIME_AXIS` as tongue velocity.

ii. `aligned_frame_times = frame_times[:n_frames] - vidshift - go_time`; `velocities[:, ti] = interp_fn(TIME_AXIS)`.

iii. The stated goal was exact placement on the common neural time grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses each motion-energy file's per-trial `me.data`, camera-0 frame times when available, video offset, and trial go cue. `moveThresh` is loaded but unused.

ii. `me_interp = interpolate_motion_energy(me['data'], session, valid_trials, go_cue_times)`.

iii. The agent recognized multiple MAT layouts and added fallback loaders; it chose the task-required per-session median rather than the file's supplied movement threshold.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is linearly interpolated to neural bin centers. Frame-time length is trimmed/extended; absent frame times use a synthetic 400-Hz axis and hard-coded 0.5-s fallback shift. Remaining gaps are interpolated; all-missing trials/streams become zero.

ii. `interp_fn = interp1d(aligned_frame_times, me_trial, ...)`; `me_interp[:, ti] = interp_fn(TIME_AXIS)`.

iii. The trajectory prioritized handling heterogeneous files and preventing missing motion energy from failing a session.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median of all finite session values defines classes 0/1; no-video category 2 is not emitted, and a missing stream becomes class 0.

ii. `me_disc = discretize_per_session(me_interp)`; fallback `np.zeros(...)`.

iii. The agent used the same literal 50th-percentile splitter for all continuous outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera frame times are corrected by `vidshift` and go cue and interpolated to neural centers; fallback timing uses `frame/400 - 0.5 - goCue`.

ii. `aligned_frame_times = frame_times - vidshift - go_time`; fallback `frame_times - 0.5 - go_time`.

iii. This was intended to match the motion-energy/video clock to behavior and neural alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Broad exceptions skip unreadable probes or entire sessions. Behavior-only/no-cluster sessions are skipped. Missing metadata defaults to `unknown`/`ALM`, video shift to zero, missing camera/energy data to zeros, and partial temporal gaps are interpolated. A NaN dropped-frame marker skips that trial's feature.

ii. `except Exception as e: ... continue`; `session['vidshift'] = 0.0`; `velocities[:, ti] = 0.0`.

iii. These fallbacks were added reactively after format/load errors so conversion could complete, with the trajectory explicitly accepting two skipped behavior-only sessions and several all-zero modalities.

## 11-a. What are the most time-consuming steps of the code?

i. The likely dominant work is nested unit×trial spike masking/histogramming and per-feature per-trial interpolation; MATLAB/HDF5 loading and the large pickle write are also substantial.

ii. `for ui, ci in enumerate(cluster_indices): ... for ti, trial_idx in enumerate(valid_trials):`; analogous trial loops occur in kinematics and motion energy.

iii. The trajectory did not profile runtime; its attention focused on full conversion and decoder training, so this is inferred from the implementation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike processing can histogram all trial assignments and aligned times for each unit in one 2-D operation rather than repeatedly scanning every spike for every trial. Per-trial output construction and NaN filling can also be array-oriented; repeated `list.index` mappings can use dictionaries.

ii. The principal avoidable loop is `for ti, trial_idx in enumerate(valid_trials): spike_mask = spike_trials == trial_num`.

iii. No vectorization rationale appears in the trajectory; correctness and format compatibility were prioritized.

## 11-c. What processing does the code repeat multiple times?

i. Every unit rescans spike-trial arrays for every retained trial and rebuilds/smooths a histogram. The same missing-value filling, frame alignment, and interpolation logic runs independently for tongue, top paw, bottom paw, and motion energy. The identical input time array and scalar outputs are copied for every trial.

ii. `extract_kinematic_feature(...)` is called three times; `TIME_AXIS.reshape(...).astype(...)` occurs inside the trial loop.

iii. The trajectory does not discuss eliminating repetition; the shared generic extractor was its main reuse mechanism.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused `L`, `no`, `sample`, `delay`, likelihood channels, `moveThresh`, and some metadata. It computes/loads both paw landmarks though the reference uses top paw, computes `all_cluster_indices` but never uses it, and loads excluded probes before rejecting them by location.

ii. `session['L'] = ...`; `threshold = float(thresh_field.flatten()[0])`; `all_cluster_indices = []`.

iii. The trajectory explored these fields and formats broadly; it did not identify the discarded work as an optimization target.
