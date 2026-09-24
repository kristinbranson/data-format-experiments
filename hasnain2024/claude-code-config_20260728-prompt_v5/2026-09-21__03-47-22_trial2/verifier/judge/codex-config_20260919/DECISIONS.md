# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every `data_structure_*.mat` in the fixed- and randomized-delay directories, skips one named duplicate, and tries an HDF5 loader before a legacy SciPy loader. Separate motion-energy files are located by matching animal and date.

ii. `for fn in sorted(os.listdir(data_dir)):` / `if not fn.startswith('data_structure_'): continue` / `if fn in SKIP_FILES: continue`; and `try: session = load_session_h5(fpath) ... except: ... load_session_scipy(fpath)`.

iii. The notes justify including both ephys directories, excluding behavior-only data, supporting mixed MATLAB formats, and removing `JEB23_2023-10-20` after identifying it as a duplicate.

## 1-b. How are the data split into subjects?

i. The subject is parsed from the filename, unique subject strings are sorted, and each retained session receives the corresponding integer index.

ii. `animal = parts[2]`; `subjects_set = sorted(set(s['animal'] for s in sessions_info))`; `subject_idx_list.append(subject_to_idx[sess_info['animal']])`.

iii. The notes report 14 animals with neural data and treat the filename animal ID as the stable subject identifier.

## 1-c. How are the data split into sessions?

i. Each discovered data-structure file is treated as a session. Sessions without neural data, with fewer than two valid trials, or with fewer than 10 retained units are skipped; one duplicate filename is explicitly skipped.

ii. `sessions.append({... 'session_id': f'{animal}_{date}'})`; `if n_units < MIN_UNITS: ... return None`.

iii. The AI aimed to reproduce 25 fixed-delay plus 19 randomized-delay sessions and used deduplication and minimum-unit filtering to reach 44.

## 1-d. How are the data split into trials?

i. Behavioral arrays are read to `Ntrials`; zero-based trial indices define output trials, while spike records use MATLAB's one-based trial numbers. Each retained index becomes one neural/input/output item.

ii. `n_trials = session['Ntrials']`; `trial_num = t + 1`; `for i in range(n_valid): result['neural'].append(trialdat[:, :, i]...)`.

iii. The notes identify `obj.bp` as the trial table and preserve its indexing across behavior, spikes, tracking, and motion energy.

## 1-e. How are trials filtered based on quality controls?

i. Early-lick, photostimulation, and NaN-go-cue trials are removed. Sessions with fewer than two remaining trials are removed. Trials after ephys recording termination are not removed.

ii. `valid_trials[session['early']] = False`; `valid_trials[session['stim_enable']] = False`; `valid_trials[np.isnan(goCue)] = False`.

iii. Early-lick and stimulation removal are attributed to the paper/reference pipeline; NaN go cues are excluded because alignment is impossible. The notes acknowledge all-zero late trials as an artifact but explicitly leave them in.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each cluster's `obj.clu.trialtm` and `obj.clu.trial`, its `quality`, and per-trial `obj.bp.ev.goCue`.

ii. `trialtm = neuron['trialtm']`; `trial = neuron['trial']`; `aligned = trialtm[spike_mask] - gc`.

iii. The AI states that cluster spike times/trial assignments are aligned to the behavioral go cue and quality-curated before use.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed per neuron and trial into 10 ms bins, divided by 0.01 s to obtain Hz, and passed through a custom 15-sample one-sided Gaussian convolution. No baseline subtraction or normalization is applied.

ii. `counts, _ = np.histogram(aligned, bins=edges)`; `fr = counts.astype(np.float32) / dt`; `fr = smooth_causal(fr, KERNEL, 'reflect')`.

iii. The notes chose the MATLAB code default of 10 ms and interpreted `mySmooth.m` as a causal 15-sample Gaussian, despite recording the paper's 5 ms bins and 35 ms half-width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Empty, garbage, noisy, `gabrga`, and `real?` labels are rejected while poor units are retained. Units with mean post-processing rate below 1 Hz are removed, and sessions below 10 units are discarded.

ii. `EXCLUDE_QUALITY = {'garbage', 'noisy', 'gabrga', 'real?', ''}`; `keep_neurons = mean_fr >= LOW_FR_THRESH`; `if n_units < MIN_UNITS: return None`.

iii. The AI says this matches `findClusters('all')`, the paper's 1 Hz threshold, and a paper-level minimum of 10 units per session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before histogramming.

ii. `aligned = trialtm[spike_mask] - gc`.

iii. The notes identify go cue as the standard reference alignment event and `alignSpikes.m` as the model.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 500 nonoverlapping 10 ms bins from -2.5 to +2.5 seconds. Raw spikes are binned directly; video streams are interpolated to the 10 ms bin centers.

ii. `DT = 0.01`; `EDGES = np.arange(TMIN, TMAX + DT, DT)`; `TIME_CENTERS = EDGES[:-1] + DT / 2`.

iii. The AI preferred the code default `params.dt = 1/100` over the paper's 5 ms statement.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common time grid relative to `obj.bp.ev.goCue`, rather than a separate measured raw variable.

ii. `TIME_CENTERS = EDGES[:-1] + DT / 2`; `time_input = TIME_CENTERS.astype(np.float32)`.

iii. The AI chose the common -2.5 to +2.5 s alignment window from the reference pipeline.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Ten-millisecond edge centers are calculated once and the same 1-by-500 vector is reused for every trial.

ii. `TIME_CENTERS = EDGES[:-1] + DT / 2`; `result['input'].append(time_input[np.newaxis, :])`.

iii. The notes describe this as the continuous time-from-go-cue decoder input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Its values are the centers of the exact edges used to histogram aligned spikes, so columns correspond directly.

ii. `counts, _ = np.histogram(aligned, bins=edges)` and `TIME_CENTERS = EDGES[:-1] + DT / 2`.

iii. A single module-level grid is intentionally shared by neural, input, and resampled behavioral data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `bp.hit`, `bp.miss`, and the instructed-side flags `bp.L`/`bp.R`; non-hit/non-miss trials are no-lick.

ii. `if session['hit'][t]: ... session['L'][t] ... elif session['miss'][t]: ...`.

iii. The AI correctly notes that L/R encode instruction, so misses must be assigned the opposite lick side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hits inherit the instructed side, misses take the opposite side, and all other trials remain class 2; values are broadcast across time.

ii. `lick_dir = np.full(n_valid, 2, dtype=np.int32)`; `out[0, :] = lick_dir[i]`.

iii. This implements left=0, right=1, none=2 as requested.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived solely from `obj.bp.autowater`.

ii. `session['autowater'] = ...`; `if session['autowater'][t] == 1: context[out_idx] = 0`.

iii. The notes identify autowater trials as WC and all others as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is relabeled WC=0; the default is DR=1, then the per-trial value is broadcast across bins.

ii. `context = np.full(n_valid, 1, dtype=np.int32)`; `out[1, :] = context[i]`.

iii. This directly implements the specified two categories.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` and `obj.bp.miss`; `bp.no` is loaded but not required.

ii. `if session['hit'][t]: outcome[out_idx] = 1; elif session['miss'][t]: outcome[out_idx] = 0`.

iii. The AI treats neither-hit-nor-miss as ignore, consistent with mutually exclusive trial outcomes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss becomes incorrect=0, hit becomes correct=1, and the default becomes ignore=2; values are broadcast over time.

ii. `outcome = np.full(n_valid, 2, dtype=np.int32)`; `out[2, :] = outcome[i]`.

iii. This is a direct relabeling into the requested classes.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only camera index 0's DLC feature named `tongue`: x, y, confidence, and frame times, plus go cue and the bitcode-derived video shift. It does not use the second view's `top_tongue`.

ii. `compute_velocity_timeseries(session, cam_idx=0, feat_name='tongue', ...)`.

iii. The planning notes call this the bottom-camera tongue feature, although the reference identifies `tongue` as the side view; no justification is given for discarding the second view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Framewise Euclidean position differences are multiplied by a fixed 400 fps, padded with zero, masked below confidence 0.5, and linearly interpolated to neural bin centers. Positions are not smoothed, gaps can be interpolated across, and the two camera views are not normalized/combined.

ii. `v = np.sqrt(dx**2 + dy**2) * VIDEO_FPS`; `v[low_conf] = np.nan`; `v_interp = np.interp(taxis, aligned_ft[valid], v[valid], ...)`.

iii. The notes justify Euclidean x/y speed and a per-session split, but do not justify these departures from the paper/reference kinematic processing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The median of all session-wide values marked visible is used: below=0, at/above=1, and not visible=2.

ii. `thresh = np.nanpercentile(vis_vals, 50)`; assignments use `< thresh` and `>= thresh`.

iii. This directly follows the prompt's per-session 50th-percentile categories.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video shift is computed from bitcode modes, then frame times subtract the shift and trial go cue and are interpolated at neural bin centers. On offset failure 0.5 s is assumed.

ii. `aligned_ft = ft - vidshift - gc`; `np.interp(taxis, aligned_ft[valid], v[valid], ...)`.

iii. The bitcode correction follows `findVideoOffset.m`; interpolation was chosen to put camera values on the neural grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses camera index 1's `top_paw` DLC x/y/confidence/frame-time data, plus go cue and video shift.

ii. `compute_velocity_timeseries(session, cam_idx=1, feat_name='top_paw', ...)`.

iii. The notes select `top_paw` as the paw signal, describing it as the side-camera feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw processing is identical to the AI's single-view tongue processing: fixed-rate first differences, confidence-0.5 masking, then linear interpolation to bin centers, without position smoothing.

ii. `dx = np.diff(x)`; `v = np.sqrt(dx**2 + dy**2) * VIDEO_FPS`; `v_interp = np.interp(...)`.

iii. The AI intended Euclidean velocity from DLC tracking but did not document why it omitted the reference's smoothing and real-time gradients.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Visible values pooled across the session are median-split into 0/1, while invisible bins are 2.

ii. `paw_disc = discretize_velocity(paw_vel, paw_vis)`.

iii. This follows the requested per-session percentile threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Its own camera frame times are corrected by video shift and go cue, then linearly interpolated to the common bin centers.

ii. `aligned_ft = ft - vidshift - gc`; `v_interp = np.interp(taxis, aligned_ft[valid], v[valid], ...)`.

iii. The AI uses the same clock correction and grid for all camera-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It prefers the separate `motionEnergy_<animal>_<date>.mat`, falls back to embedded `obj.me`, and uses camera-0 frame times, go cue, and video shift for timing.

ii. `if has_me_file: ... load_motion_energy_file(me_path) ... elif me_trials is not None: ...`.

iii. The notes recognize both external and embedded layouts and prefer the external matched file when present.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Existing per-frame motion energy is trimmed to frame-time length and linearly interpolated at 10 ms centers; no smoothing or spatial recomputation is done.

ii. `me_interp = np.interp(taxis, aligned_ft[valid], me_data[valid], left=np.nan, right=np.nan)`.

iii. The AI treats motion energy as already reduced to one value per video frame.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. All finite values in a session are median-split: below=0, at/above=1, NaN/no-video=2.

ii. `thresh = np.nanpercentile(valid_vals, 50)`; `result[valid & (me_aligned >= thresh)] = 1`.

iii. This directly follows the requested session-level 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera-0 frame times subtract the session video shift and trial go cue, then values are interpolated to neural centers. If frame times are absent, synthetic 400 Hz times beginning at 0.5 s are invented.

ii. `aligned_ft = ft - vidshift - gc`; fallback `ft = np.arange(n_frames) / VIDEO_FPS + 0.5`.

iii. The normal path follows the shared video-clock correction; the fallback is intended to retain otherwise missing video.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many loader failures become empty arrays/`None`; missing kinematics become class 2. Broad exceptions switch loaders or skip fields. Missing video shift defaults to 0.5 s and missing motion-energy frame times are synthesized. NaN go-cue trials are dropped. Known late all-zero ephys trials are retained.

ii. `except: session['vidshift'] = 0.5`; `result = np.full(..., 2, ...)`; `valid_trials[np.isnan(goCue)] = False`.

iii. The notes emphasize producing NaN-free decoder data and acknowledge the late-recording artifact, but classify it as a data issue rather than correcting it.

## 11-a. What are the most time-consuming steps of the code?

i. The AI times file loading and neural binning per session; its code structure also makes the nested neuron-by-trial spike loop and the very large pickle materialization costly.

ii. `t_load = time.time() - t0`; `t_bin = time.time() - t1`; nested `for ni ... for t in range(n_trials)`.

iii. The notes report a roughly 2.1 GB result but do not provide an explicit aggregate runtime breakdown.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning loops over every neuron and every trial even though trial and aligned-time dimensions can be histogrammed together. The three per-trial label loops could be vectorized, and output assembly could be array-based; ragged video trials reasonably remain looped.

ii. `for ni, neuron in enumerate(neurons): for t in range(n_trials):`; three instances of `for out_idx, t in enumerate(trial_indices)`.

iii. No specific efficiency justification is documented; the implementation favors straightforward trialwise logic.

## 11-c. What processing does the code repeat multiple times?

i. The same per-session trial traversal is repeated for spike binning, lick direction, context, outcome, tongue, paw, motion energy, and final list assembly. Velocity interpolation is run independently for tongue and paw.

ii. `compute_velocity_timeseries(...)` is called twice; behavioral labels each have a separate `for out_idx, t ...` loop.

iii. The notes describe separate extraction stages for clarity but do not discuss repeated work.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads lick-time arrays, sample/delay times, cluster absolute `tm`, `bp.no`, and motion-energy `moveThresh` without using them in conversion. It bins all trials before discarding invalid trials and loads/aligns video values before retaining only thresholded categories.

ii. `session['lickL'] = []`; `tm = np.array(f[tm_ref]).flatten()`; `me_trials_loaded, me_thresh = ...`; `trialdat = align_and_bin_spikes(... n_trials ...)` then `trialdat = trialdat[:, :, trial_indices]`.

iii. These fields were explored or loaded for completeness; the notes do not claim they affect the final decoder data.
