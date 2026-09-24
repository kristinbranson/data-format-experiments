# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes session, probe, and folder metadata in `SESSION_META`, checks that each data file exists, and loads each session through `SessionData`, which supports HDF5/v7.3 and MATLAB v5 files. Motion energy is loaded separately.

ii. `for (anm, date), (probes, dtype) in sorted(SESSION_META.items()): ... sessions.append((anm, date, probes, dtype))` and `self.f = h5py.File(self.data_file, 'r') ... data = scipy.io.loadmat(self.data_file, squeeze_me=False)`.

iii. The notes say the mapping was taken from the authors' animal-loading functions, and both MATLAB formats had to be supported. They report 25 fixed-delay files and selected randomized-delay files.

## 1-b. How are the data split into subjects?

i. The animal component of each `(animal, date)` session key defines the subject. Subjects are accumulated in first-session order, and every retained session stores its index into that list.

ii. `if anm not in all_subjects: all_subjects.append(anm)` and `subject_idx_list.append(all_subjects.index(anm))`.

iii. The notes treat the animal identifiers in filenames/loading functions as subject IDs and report 14 subjects in the final data.

## 1-c. How are the data split into sessions?

i. Each hard-coded `(animal, date)` entry/file is initially one session. A session is later omitted if it fails bilateral DR-hit, trial-count, neuron-count, or data-availability checks.

ii. `if n_r_hit_dr <= 40 or n_l_hit_dr <= 40: ... return None` and `if n_neurons < 10: ... return None`.

iii. The AI cites the paper's session inclusion requirements (>40 correct DR trials per direction and at least 10 units), ultimately retaining 43 sessions.

## 1-d. How are the data split into trials?

i. Trials are direct indices from `bp.Ntrials`; behavior arrays, spike `trial` labels, camera trajectory entries, and motion-energy entries are indexed by that trial number.

ii. `n_trials_total = sd.get_n_trials()` and `for t_idx in valid_trial_indices: neural_trials.append(trialdat[:, :, t_idx].T...)`.

iii. The AI understood the raw structures as one behavioral row, camera entry, and motion-energy trace per trial, with spike records carrying 1-based trial labels.

## 1-e. How are trials filtered based on quality controls?

i. It removes early-lick, no-response/ignore, and photostimulation trials. It also excludes whole sessions without >40 left-hit and >40 right-hit DR trials, fewer than two retained trials, fewer than ten neurons, or any usable neurons. It does not remove trials after electrophysiology recording ends.

ii. `valid_trials = (early == 0) & (no_resp == 0) & (stim_enable == 0)`; `if n_r_hit_dr <= 40 or n_l_hit_dr <= 40: return None`.

iii. The notes say these filters match the paper and inclusion code. Later notes acknowledge all-zero neural trials after recording ended but call them valid behavioral trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each cluster's `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment.

ii. `trial, trialtm = sd.get_spike_data(probe_idx, clu_idx)` and `trialtm_aligned = trialtm - go_cue[trial_int - 1]`.

iii. The notes identify `alignSpikes`, `findClusters`, and `getSeq` as the reference operations being reproduced.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed in 5-ms bins, divided by 0.005 to produce Hz, and smoothed with a 15-sample one-sided Gaussian whose first half is zeroed, using reflected prefix padding. Probe populations are concatenated.

ii. `N_counts, _ = np.histogram(spk_times, bins=EDGES)`; `fr = N_counts / dt`; `fr_smooth = causal_gaussian_smooth(...)`; `kern[:N//2] = 0`.

iii. The AI believed `mySmooth.m` used this causal 15-sample Gaussian and documented it as matching the analysis pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters with blank quality or labels `garbage`, `gabrga`, `noisy`, or `real?` are removed. Remaining clusters must have mean windowed firing rate strictly above 1 Hz. Sessions also need at least ten remaining neurons.

ii. `if q_clean == '' or q_clean in exclude_lower: continue`; `fr_mask = mean_frs > PARAMS['lowFR']`; `if n_neurons < 10: return None`.

iii. The notes cite `findClusters.m`, the paper's >1-Hz statement, and the paper's ten-unit session minimum.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before binning.

ii. `trialtm_aligned = trialtm - go_cue[trial_int - 1]`.

iii. The AI explicitly identifies this as the operation in `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 1000 non-overlapping 5-ms bins from -2.5 to +2.5 seconds. Raw spikes are binned onto this grid; video streams are interpolated to its bin centers.

ii. `EDGES = np.arange(-2.5, 2.5 + 1/200, 1/200)` and `TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2`.

iii. The AI chose 5 ms from the analysis scripts despite finding a 10-ms example elsewhere.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common time grid defined relative to `bp.ev.goCue`, rather than a copied raw signal.

ii. `TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2`.

iii. The notes state that all trials are aligned to go-cue onset and use the paper's analysis window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are computed from the fixed -2.5-to-2.5-second edges and copied to every trial as a `(1, 1000)` float32 array.

ii. `input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))`.

iii. The AI validated the range `[-2.4975, 2.4975]` and 1000 points.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It consists of the centers of the exact edges used to histogram go-cue-aligned spikes.

ii. `np.histogram(spk_times, bins=EDGES)` and `TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2`.

iii. The shared grid was a stated sanity check.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.L`, `bp.R`, `bp.hit`, `bp.miss`, and `bp.no`.

ii. `lick_dir[(hit == 1) & (L == 1)] = 0`; `lick_dir[(miss == 1) & (R == 1)] = 0`; `lick_dir[no_resp == 1] = 2`.

iii. The notes reason that hits imply the instructed side, misses the opposite side, and ignores no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Left is coded 0, right 1, and no lick 2, then the per-trial code is repeated across time. In practice no-llick trials are filtered out first.

ii. `output_trial[0, :] = lick_dir[t_idx]`.

iii. The AI notes that the `none` class is absent because it intentionally excluded ignore trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived directly from `bp.autowater`.

ii. `autowater = sd.get_bp_field('autowater')`.

iii. The AI identifies autowater as the WC indicator.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The raw flag is cast directly to integer, producing DR=0 and WC=1, and repeated over time.

ii. `context = autowater.astype(np.int64)  # 0=DR, 1=WC`.

iii. The notes explicitly planned and documented DR=0, WC=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit`, `bp.miss`, and `bp.no`.

ii. `outcome[hit == 1] = 1`; `outcome[miss == 1] = 0`; `outcome[no_resp == 1] = 2`.

iii. The AI describes these as correct, incorrect, and ignore flags.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Incorrect is 0, correct 1, and ignore 2, repeated across all time bins. Ignore trials are removed by the trial filter, so class 2 never reaches the output.

ii. `output_trial[2, :] = outcome[t_idx]`.

iii. The AI considered the missing ignore class acceptable because it followed its interpretation of the paper's behavioral filtering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only camera 0's `tongue` DLC x/y coordinates and frame times, plus video-clock fields and go-cue times. It does not use the second camera's tongue feature.

ii. `extract_feature_velocity(sd, 0, 'tongue', go_cue, n_trials_total, is_tongue=True)`.

iii. The notes chose “camera 0 'tongue' feature” after inspecting both camera feature lists.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Raw x/y are linearly interpolated to the 5-ms grid; visibility is inferred from finite interpolated coordinates; gradients are taken per sample (without division by seconds), invisible derivatives are zeroed, and Euclidean magnitude is computed. No coordinate smoothing or two-camera normalization/combination is done.

ii. `x_interp = np.interp(taxis, aligned_ft, x)`; `xvel = np.gradient(x_interp)`; `spd = np.sqrt(xvel**2 + yvel**2)`.

iii. The AI believed this matched the reference `findPosition`/`findVelocity` pipeline and regarded NaNs as DLC invisibility markers.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One session-wide median is computed over visible finite samples: below it is 0, at/above it is 1, and other bins are 2.

ii. `threshold = np.percentile(speed[vis_mask], percentile_thresh)` and `categories = np.full(..., 2)`.

iii. This directly follows the requested per-session 50th-percentile split.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset from bitcode modes and the trial go cue are subtracted from camera frame times, after which positions are interpolated at neural bin centers.

ii. `aligned_ft = ft - vidshift - go_cue[t]`; `np.interp(taxis, aligned_ft, x)`.

iii. The AI cites `findVideoOffset` and reference interpolation onto the neural time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses camera 1's `bottom_paw` x/y tracking, its frame times, the video offset fields, and go-cue times.

ii. `extract_feature_velocity(sd, 1, 'bottom_paw', go_cue, n_trials_total, is_tongue=False)`.

iii. The notes selected this feature as the paw output without documenting a reliability comparison to `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Position is interpolated to neural times, gradients are computed per sample, median coordinate drift is subtracted, missing derivatives are nearest-filled, and x/y derivatives are combined by Euclidean magnitude.

ii. `basederiv_x = np.nanmedian(np.diff(x_interp))`; `arr[mask_nan] = np.interp(...)`; `spd = np.sqrt(xvel**2 + yvel**2)`.

iii. The AI states this reproduces the reference behavior for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide median over bins originally marked visible separates low (0) from high (1); missing bins are 2.

ii. `paw_cat = discretize_velocity(paw_speed, paw_visible, 50)`.

iii. This follows the prompt's requested per-session percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Camera times are offset-corrected, made relative to each trial's go cue, and interpolated onto the common neural grid.

ii. `aligned_ft = ft - vidshift - go_cue[t]`.

iii. The AI cites the same reference video alignment used for tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads per-trial traces and `moveThresh` from separate `motionEnergy_<animal>_<date>.mat` files and uses camera 0 frame times, video-clock fields, and go-cue times. `moveThresh` is loaded but unused.

ii. `me_data, me_thresh = sd.get_motion_energy_data(me_file)` and `ts, ft, ndf = sd.get_traj_trial(0, t)`.

iii. The AI chose the standalone motion-energy files and notes that missing files become the no-video class.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Traces are length-matched to camera times and linearly interpolated to neural bin centers. NaNs are nearest-filled. When frame times are unavailable, synthetic 400-Hz times and an ad hoc 0.5-second shift are used.

ii. `ft = np.arange(len(me_trial)) / 400.0`; `aligned_ft = ft - 0.5 - go_cue[t]`; `me_aligned[:, t] = np.interp(...)`.

iii. The AI believed motion energy only required loading and temporal alignment, and added fallbacks to keep data usable.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A median over all finite session samples yields low (0) and high (1); if the whole file cannot be loaded, every bin is no-video (2).

ii. `threshold = np.percentile(me_data[valid_mask], percentile_thresh)` and `if not has_video: return np.full(..., 2)`.

iii. This is the requested per-session 50th-percentile threshold and no-video category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera 0 frame times are corrected by video offset and trial go cue, then the values are interpolated onto neural centers; missing frame times invoke a synthetic fallback.

ii. `aligned_ft = ft - vidshift - go_cue[t]` and `np.interp(taxis, aligned_ft, me_trial)`.

iii. The notes cite the shared video offset and neural time grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many readers broadly catch exceptions and return empty/missing values. Missing trajectories become category 2; paw derivative gaps and motion-energy NaNs are nearest-filled; missing motion-energy frame times get fabricated 400-Hz times and a 0.5-s shift; absent motion-energy files become all 2. Trials after ephys ends are retained as all-zero neural trials.

ii. `except: return None, None`; `arr[mask_nan] = np.interp(...)`; `aligned_ft = ft - 0.5 - go_cue[t]`.

iii. The notes emphasize robust handling of both file formats and missing video, while explicitly accepting end-of-recording zero-neural trials.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's timing output separately measures neural firing-rate construction and video extraction; nested spike histogram/smoothing loops are a major compute step, while file loading also contributes.

ii. `for neuron_idx, clu_idx ... for t in range(n_trials): ... causal_gaussian_smooth(...)`.

iii. Notes estimate roughly 3.7 seconds per session and 167 seconds overall but do not provide a formal profile.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike processing loops over neurons, trials, and convolution columns; video and motion processing loop over trials; output formatting also loops over trials. The per-neuron trial loop could be replaced by joint trial/time histogramming, and convolution could operate on arrays at once.

ii. `for neuron_idx ...`; `for t in range(n_trials):`; `for j in range(x_filt.shape[1]):`.

iii. The AI did not document vectorization opportunities; it prioritized a direct translation of reference stages.

## 11-c. What processing does the code repeat multiple times?

i. `get_video_offset()` is recomputed for tongue, paw, and motion energy. Camera trajectories are re-read for each output, including camera 0 again for motion-energy timing. The same time input is separately allocated for every trial.

ii. Each extractor contains `vidshift = sd.get_video_offset()`, and `input_trials.append(TIME_AXIS.reshape(1, -1).astype(np.float32))` is inside the trial loop.

iii. The AI's notes do not recognize these repetitions.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused `L` for some derived logic redundancy, `moveThresh`, dropped-frame counts, and `me_thresh`; computes outputs for every raw trial before retaining only `valid_trial_indices`; and creates optional diagnostic plots not used by conversion. Ignore labels are computed even though ignore trials are discarded.

ii. `me_data, me_thresh = ...`; all video arrays use `n_trials_total`; then `for t_idx in valid_trial_indices:` selects output.

iii. The AI did not document these as waste, focusing instead on correctness checks and decoder performance.
