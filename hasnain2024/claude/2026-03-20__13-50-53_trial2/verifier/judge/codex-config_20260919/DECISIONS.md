# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 44 author-selected sessions, their ALM probes, and one of two data folders. It detects MATLAB v7.3 versus v5 and uses dedicated HDF5/scipy loaders; motion energy is loaded separately.

ii. `SESSION_DEFS = [...]`; `fmt = _detect_file_format(data_fn)`; `raw = _load_raw_data_hdf5(...)` / `_load_raw_data_v5(...)`.

iii. The notes say author loading scripts define included sessions/probes; three unlisted files and unavailable JEB4/JEB5 sessions were excluded. Both MATLAB layouts required separate readers.

## 1-b. How are the data split into subjects?

i. Subject is the hard-coded `anm` value. Unique animals are sorted, and each retained session receives an index.

ii. `all_animals = sorted(set(s['animal'] for s in all_sessions))`; `subject_idx.append(animal_to_idx[sess['animal']])`.

iii. The notes describe 14 available animals and acknowledge this is one more DR animal than the paper reports.

## 1-c. How are the data split into sessions?

i. Each `(animal, date)` entry and corresponding `data_structure_<animal>_<date>.mat` is one session; failed or under-10-unit sessions are skipped.

ii. `for i, (anm, date, probes, ddir) in enumerate(session_defs): result = load_session(...)`; `if n_neurons < params['min_units']: return None`.

iii. This follows the author loading scripts and the paper's minimum of ten units per analyzed session.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines trial count. Spike `trial` identifiers and per-trial behavior/video cells associate observations with trials; retained trials become individual matrices in the output.

ii. `valid_trials = np.where(trial_mask)[0] + 1`; `for t in range(n_trials): sess_neural.append(trialdat[:, :, t])`.

iii. The notes treat Bpod trial fields as the authoritative trial partition.

## 1-e. How are trials filtered based on quality controls?

i. Early-lick and stimulation trials are removed. Sessions with fewer than two such trials are skipped. Unlike the reference, late trials after ephys ends are retained, yielding 61 documented all-zero neural trials.

ii. `trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)`; `if len(valid_trials) < 2: return None`.

iii. The first two exclusions are justified from the paper. The notes call the zero-neural late trials “acceptable” because they are only 0.44% of trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data uses selected probes' cluster `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment.

ii. `all_spike_times.append(trialtm)`; `all_spike_trials.append(trial)`; `align_times = ev[params['align_event']]`.

iii. The notes map this to the reference `findClusters`, `alignSpikes`, and `getSeq` pipeline.

## 2-b. How is the `neural` data processed?

i. Per neuron and trial, aligned spikes are histogrammed, divided by 0.01 s to Hz, and causally smoothed with a 15-sample half-Gaussian. The reference instead uses 5 ms bins and a non-causal Gaussian.

ii. `counts, _ = np.histogram(spk_times, bins=edges)`; `fr = causal_gaussian_smooth(counts / params['dt'], params['smooth_window'], ...)`.

iii. The agent believed `WorkingWithDataObjs.m` prescribed 10 ms and causal `mySmooth`, despite noting a conflicting 5 ms default.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Empty labels and `garbage`, `noisy`, `gabrga`, and `real?` are excluded; units must have mean window firing rate strictly above 1 Hz; sessions need at least ten retained units. `poor` is retained.

ii. `keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan' ...])`; `fr_mask = mean_fr > params['low_fr']`.

iii. The notes say this matches quality=`all`, the paper's 1 Hz cutoff, and its ten-unit session minimum.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before binning from −2.5 to +2.5 s.

ii. `spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]`.

iii. The agent identifies go cue as both the requested and reference alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent creates 500 non-overlapping 10 ms bins and interpolates video streams onto their centers. This differs from the human reference's 1000 5 ms bins.

ii. `'dt': 1.0 / 100`; `edges = np.arange(tmin, tmax + dt, dt)`.

iii. The notes explicitly chose the 10 ms tutorial setting over the repository default of 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the configured −2.5-to-2.5 s bin grid, conceptually relative to `bp.ev.goCue`, rather than read as a raw variable.

ii. `time_axis = edges[:-1] + params['dt'] / 2`.

iii. The notes call it a linear ramp defined by the requested go-cue alignment.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are calculated and the same `(1, 500)` float32 array is placed in every trial.

ii. `time_input = time_axis.astype(np.float32).reshape(1, -1)`.

iii. This was intended as the required continuous, time-varying decoder input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact histogram edges used for neural spikes.

ii. `edges = ...`; `time_axis = edges[:-1] + params['dt'] / 2`.

iii. The shared grid was chosen to guarantee matching time dimensions.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived only from `bp.R` on retained trials, not from hit/miss outcome. Thus it represents instructed side, not actual lick direction on errors, and has no no-lick class.

ii. `lick_direction = bp['R'][valid_trial_indices].copy()`.

iii. The mapping plan says `obj.bp.R/L`, with `L=0, R=1`, “from trial info.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The binary `R` flag is copied and broadcast through all time bins; no correction for miss trials or class 2 for ignore trials is performed.

ii. `lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)`.

iii. The notes justify a simple left/right mapping but overlook that actual direction reverses on misses and is absent on ignores.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp.autowater`.

ii. `behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]`.

iii. The notes identify autowater trials as WC and other trials as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is inverted so WC=0 and DR=1, then broadcast across time.

ii. `context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)`.

iii. This is explicitly documented as matching the prompt's value order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived only from `bp.hit`, although `miss` and `no` are loaded.

ii. `outcome = bp['hit'][valid_trial_indices].copy()`.

iii. The mapping plan states hit=1/correct and miss=0/incorrect, but does not account for ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The hit flag is copied and broadcast, collapsing both incorrect and ignore trials to class 0 instead of producing incorrect=0, correct=1, ignore=2.

ii. `outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)`.

iii. The notes say all no-response trials are retained, but fail to implement their required third outcome class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only bottom-camera `top_tongue` x/y positions and frame times from `obj.traj`, plus video/behavior bitcode timing and go cue. It ignores the side-camera tongue and likelihood channel.

ii. `_compute_feature_velocity_generic(... view=2, feat_name='top_tongue', ..., is_tongue=True)`.

iii. The notes intentionally chose the bottom camera, whereas the human solution combines both views to improve visibility.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Valid x/y samples are linearly interpolated directly to neural bin centers, gradients are taken per bin (not per second), NaN gradients are set to zero, and Euclidean magnitude is computed. There is no likelihood filtering, position smoothing, two-view normalization, or frame-bin averaging.

ii. `xpos = interp1d(...)(time_axis)`; `xvel = np.gradient(xpos)`; `xvel[np.isnan(xvel)] = 0`; `speed = np.sqrt(xvel**2 + yvel**2)`.

iii. The notes describe DLC position-to-Euclidean-velocity and interpolation, but overstate fidelity to the authors' processing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median over non-NaN samples creates 0 below and 1 at/above. Missing data is incorrectly class 0; class 2 is never generated.

ii. `threshold = np.percentile(valid_vals, 50)`; `disc = (data >= threshold).astype(np.int64)`; `disc[np.isnan(data)] = 0`.

iii. The median follows the prompt, but the notes treat zero/retracted values as real and do not honor “not visible.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame time is corrected by a session video offset, shifted by trial go cue, and interpolated onto neural bin centers. Offset uses medians and silently defaults to 0.5 s on failure.

ii. `ft_aligned = ft - vidshift - align_times[trix]`; `interp1d(...)(time_axis)`.

iii. The notes cite `findVideoOffset` and the 0.5 s pad as reference synchronization.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw` x/y positions and frame times, timing metadata, and go cue; likelihood is ignored.

ii. `_compute_feature_velocity_generic(... view=2, feat_name='top_paw', ..., is_tongue=False)`.

iii. The notes selected the bottom `top_paw`, consistent with the reliable tracked paw.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated to bin centers and nearest-filled, gradients are computed per bin, median displacement is subtracted from each velocity component, and magnitude is taken. This differs from reference likelihood filtering, smoothing, and differentiation using real time.

ii. `xpos = _fill_nearest(xpos)`; `xvel = np.gradient(xpos)`; `xvel -= np.nanmedian(np.diff(xpos))`.

iii. The agent intended to follow `findVelocity/getKinematicsFromVideo`, with nearest filling for gaps.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A non-NaN session median splits classes 0/1; missing values are assigned 0 and class 2 is absent.

ii. `disc = (data >= threshold).astype(np.int64)`; `disc[np.isnan(data)] = 0`.

iii. The median is justified by the task, but missing-as-low contradicts its “not visible” category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera times have video offset and trial go cue subtracted, then x/y are interpolated to neural centers.

ii. `ft_aligned = ft - vidshift - align_times[trix]`; `interp1d(...)(time_axis)`.

iii. The shared corrected clock/grid is documented as the alignment method.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses per-trial traces in separate `motionEnergy_<animal>_<date>.mat` files and side-camera frame times, plus video timing and go cue.

ii. `me_fn = ... f'motionEnergy_{anm}_{date}.mat'`; `ft = side_cam['trials'][trix]['frameTimes']`.

iii. The notes identify the standalone files as the reference source and report multiple storage layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is linearly interpolated to neural bin centers and internal gaps are nearest-filled; failures leave the whole stream unavailable.

ii. `me_aligned[:, trix] = interp1d(...)(time_axis)`; `me_aligned[:, trix] = _fill_nearest(col)`.

iii. The notes say reference motion energy is loaded and interpolated, without re-deriving the upstream pixel statistic.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Its session-wide non-NaN median produces 0/1. Missing/no-video values and complete load failures become 0 rather than required class 2.

ii. `me_disc = _discretize_velocity(me_data, ...)`; `if data is None: return np.zeros(...)`.

iii. The agent documents four failed files and seven degenerate sessions but accepts all-zero “low” motion energy.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted by video offset and trial go cue, then linearly interpolated at neural time centers; synthetic 400 Hz times are used when frame times are absent.

ii. `ft = np.arange(1, me_trial.size + 1) / 400.0`; `ft_aligned = ft - vidshift - align_times[trix]`.

iii. The notes justify the 400 Hz fallback from the acquisition frame rate.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Broad exceptions often skip sessions/trials/features. Missing frame times get synthetic 400 Hz times; paw/motion gaps are nearest-filled; tongue gaps become zero; missing or all-NaN streams discretize to class 0. Corrupt motion files therefore masquerade as low motion, and late no-ephys trials are kept as zero neural arrays.

ii. `except Exception: continue`; `ft = np.arange(...) / 400.0`; `if data is None: return np.zeros(...)`.

iii. Notes acknowledge corrupt ME files, all-zero neural trials, and degenerate classes, but regard them as acceptable rather than representing unavailable data with class 2 or filtering invalid trials.

## 11-a. What are the most time-consuming steps of the code?

i. File loading and nested neuron-by-trial spike processing dominate; the notes report about 5 s/session and 163 s for conversion.

ii. `for neuron_idx, clu_idx in enumerate(good_indices): ... for t_idx, trial_num in enumerate(valid_trials):`.

iii. Runtime estimates in the notes attribute “full loading” roughly 5 seconds per session.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested neuron/trial spike mask-histogram-smoothing loops are the clearest vectorization target. Video and motion loops are ragged but parts could be batched after interpolation.

ii. `for neuron_idx ...: for t_idx, trial_num ...: spk_mask = spike_trial == trial_num`.

iii. The agent did not discuss this explicitly; its reference mapping prioritized fidelity over optimization.

## 11-c. What processing does the code repeat multiple times?

i. It recreates the identical time input for every trial, repeatedly masks each neuron's full spike-trial vector for every trial, smooths separately per neuron/trial, and calls similar interpolation/fill logic independently for tongue, paw, and motion energy.

ii. `time_input = time_axis.astype(np.float32).reshape(1, -1)` inside the trial loop; `spk_mask = spike_trial == trial_num` inside nested loops.

iii. No specific justification is documented beyond straightforward construction of per-trial output objects.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full conversion retains raw tongue, paw, and motion arrays in each temporary session solely for optional plots, then excludes them from the pickle. It also loads unused behavior fields/events, produces repeated input copies, and optionally creates extensive diagnostic plots.

ii. `'tongue_vel_raw': tongue_vel, 'paw_vel_raw': paw_vel, 'me_raw': me_data`; only discretized arrays are later assembled.

iii. Raw arrays support sanity plots and review, but are not downstream decoder inputs/outputs; notes emphasize these diagnostics.
