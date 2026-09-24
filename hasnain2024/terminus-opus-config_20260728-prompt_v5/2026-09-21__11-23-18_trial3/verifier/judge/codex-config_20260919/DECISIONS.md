# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 25 fixed-delay/two-context and 19 randomized-delay sessions, including the selected probe(s), then loads each `data_structure_<animal>_<date>.mat` from its task folder. It supports MATLAB v7.3 with `h5py` and older MATLAB files with `scipy.io.loadmat`; motion energy is loaded separately.

ii. `all_sessions = [(a, d, p, EPHYS_DATA_DIR) for a, d, p in EPHYS_SESSIONS] + [(a, d, p, RD_DATA_DIR) for a, d, p in RD_SESSIONS]`; `load_mat_file` tries `h5py.File(fpath, 'r')` and falls back to `sio.loadmat(fpath, squeeze_me=False)`.

iii. The notes say the lists were taken from the authors' loading scripts, excluding three unused files, and that both MATLAB formats and both datasets must be handled.

## 1-b. How are the data split into subjects?

i. The animal identifier supplied in each hard-coded session tuple defines the subject. Subjects are accumulated in first-appearance order and every retained session stores its index.

ii. `if anm not in subjects: subjects.append(anm)` followed by `subject_idx.append(subjects.index(anm))`.

iii. The notes report 14 unique identifiers and explain the discrepancy from the paper's aggregate mouse count as a difference in paper accounting.

## 1-c. How are the data split into sessions?

i. Every hard-coded animal/date tuple and corresponding MATLAB file is one session and one outer-list element in `neural`, `input`, and `output`.

ii. `for si, (anm, date, probes, ddir) in enumerate(all_sessions): result = process_session(...)`; successful results are appended once to each outer list.

iii. The AI chose the 44 sessions present in the authors' scripts: 25 Ephys and 19 randomized-delay sessions.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` supplies the number of trials. Trial-indexed behavior arrays and trajectory entries are indexed from 0, while each cluster's 1-based `trial` labels select its spikes. One matrix is emitted for every raw trial.

ii. `for trial_num in range(1, n_trials + 1): spk_mask = clu['trial'] == trial_num`; later, `for t in range(n_trials): neural_trials.append(trialdat[:, :, t].T...)`.

iii. The notes state that all trials were included and their conditions represented as labels.

## 1-e. How are trials filtered based on quality controls?

i. They are not filtered. Although `early` and `stim_enable` are read, stimulation trials remain, and early trials are relabeled as no-lick/ignore. Trials after electrophysiology ends also remain as all-zero neural trials.

ii. No trial mask is applied. Instead: `lick_direction[early] = 2` and `outcome[early] = 2`; `stim_enable` is never used.

iii. The notes explicitly decide “No trials excluded,” acknowledge 30 all-zero neural trials, and call those recording gaps expected.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each cluster's `quality`, per-spike `trial`, and `trialtm`, plus `bp.ev.goCue` for alignment.

ii. Cluster extraction stores `{'quality': q_str, 'trialtm': trialtm, 'trial': trial, 'probe': probe_num}`; alignment uses `clu['trialtm'][spk_mask] - goCue[trial_num - 1]`.

iii. The notes identify `findClusters`, `alignSpikes`, and `getSeq` as the reference stages being reproduced.

## 2-b. How is the `neural` data processed?

i. Per cluster and trial, aligned spikes are histogrammed into 10 ms bins, divided by 0.01 s to obtain Hz, and smoothed with a custom one-sided (“causal”) 15-sample Gaussian-like kernel with reflected prefix padding. Selected probes are concatenated; no normalization or baseline subtraction is applied.

ii. `N, _ = np.histogram(aligned_times, bins=edges)`; `fr = N.astype(np.float64) / DT`; `causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)`.

iii. The AI says this matches `getSeq.m` and `mySmooth.m`, interpreting the reference as causal Gaussian smoothing with a 15 ms window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It removes qualities `garbage`, `noisy`, `gabrga`, and `real?`, also removes blank/NUL labels, then keeps neurons with mean binned/smoothed rate strictly above 1 Hz. It skips a session if fewer than 10 remain.

ii. `EXCLUDED_QUALITIES = {'garbage', 'noisy', 'gabrga', 'real?'}`; `keep_mask = trialdat.mean(axis=(0, 2)) > LOW_FR_THRESHOLD`; `if n_kept < 10: return None`.

iii. The notes cite `findClusters.m`, the paper's >1 Hz rule, and the paper's at-least-10-units criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time is shifted by that trial's go-cue time before histogramming into the common window.

ii. `aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]`.

iii. The AI says this directly follows `alignSpikes.m` with `alignEvent='goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 bins of 10 ms from -2.5 to +2.5 s. Raw spikes are rebinned by histogram; video signals are linearly interpolated at the 10 ms bin centers.

ii. `DT = 1.0 / 100`; `edges = np.arange(TMIN, TMAX + DT, DT)`; `time_axis = edges[:-1] + DT / 2`.

iii. The notes acknowledge a 5 ms/10 ms conflict, select 10 ms from `WorkingWithDataObjs.m`, and call it more standard for decoding.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the configured -2.5 to +2.5 s go-cue-relative bin grid, rather than read as a raw variable.

ii. `time_axis = edges[:-1] + DT / 2` and `time_input = time_axis.astype(np.float32).reshape(1, -1)`.

iii. The notes map `time_axis` directly to the sole decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The centers of consecutive 10 ms histogram bins are computed once, converted to `float32`, reshaped to `(1, 500)`, and copied for every trial.

ii. `edges[:-1] + DT / 2`; `input_trials.append(time_input.copy())`.

iii. The chosen grid is justified by the AI's 10 ms bin decision.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of exactly the edges used to bin go-cue-shifted spikes, so corresponding columns represent the same relative intervals.

ii. Spikes use `bins=edges`, while the input uses `edges[:-1] + DT / 2`.

iii. Alignment is inherent in the common grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses per-trial `bp.hit`, `bp.miss`, `bp.R`, `bp.L`, and additionally `bp.early`.

ii. `hit = trial_info['hit']`, `miss = ...`, `R = ...`, `L = ...`, `early = ...`.

iii. The notes describe deriving categorical direction from hit/miss and instructed side, with an early-trial correction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hits are labeled with the instructed side; misses with the opposite side; other trials are none. The AI then overrides every early trial to none. The scalar label is repeated across all time bins.

ii. `lick_direction[hit & R] = 1`; `lick_direction[miss & R] = 0`; `lick_direction[early] = 2`; `output[0, :] = lick_direction[t]`.

iii. The critical review says early trials can also be hit/miss, so they were deliberately changed to none.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived solely from `bp.autowater`.

ii. `autowater = trial_info['autowater']`.

iii. The notes map this raw flag to context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The Boolean flag is cast to integer: 0 is DR and 1 is WC, consistent with the AI's `output_values` ordering. The label is repeated across time.

ii. `context = autowater.astype(int)` and `output_values` contains `['DR', 'WC']`.

iii. The mapping table states context is binary and reports 0=DR, 1=WC.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit`, `bp.miss`, and `bp.early`.

ii. `hit = trial_info['hit']`; `miss = trial_info['miss']`; `early = trial_info['early']`.

iii. The notes explicitly list these three fields.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Default is ignore (2), hits become correct (1), misses incorrect (0), and early trials are overridden back to ignore. The trial label is repeated across time.

ii. `outcome = np.full(n_trials, 2)`; `outcome[hit] = 1`; `outcome[miss] = 0`; `outcome[early] = 2`.

iii. The early override was added during critical review because early trials may carry hit/miss flags.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` feature's DLC x, y, and confidence from `obj.traj`, its `frameTimes`, video/behavior bit-code timing for `vidshift`, and `goCue`. Bottom-camera tongue tracking is loaded but unused.

ii. `ft_side, ts_side, feat_side = ...`; `tx, ty, tc = ts_side[ti, 0, :], ...`; `ft_side_aligned = ft_side - vidshift - goCue[t]`.

iii. The notes explicitly choose “Tongue velocity from side cam” and a 0.5 confidence cutoff.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Consecutive x/y differences are divided by frame-time differences and combined as Euclidean speed. Confidence below 0.5 becomes NaN. Speed and a numeric visibility mask are independently linearly interpolated to neural-bin centers; interpolated bins failing the mask are NaN. There is no coordinate smoothing and no two-camera combination.

ii. `speed = np.sqrt(dx**2 + dy**2) / dt`; `tspeed[~tongue_visible] = np.nan`; `tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)`.

iii. The AI says it follows reference video alignment and intentionally uses the side tongue feature with confidence 0.5.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single median over all non-NaN bins and trials in the session is used: below median=0, at/above=1, missing=2.

ii. `tongue_thresh = np.nanpercentile(tongue_vel_all[tongue_valid], 50)` followed by assignments to `tongue_disc` initialized to 2.

iii. This implements the requested per-session 50th percentile threshold and not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video shift from bit-code modes and the trial go cue are subtracted from side-camera frame times, then speed is interpolated at the common 10 ms centers.

ii. `vidshift = mode(bitstart_sglx)/fs - mode(bitStart_bp)`; `ft_side_aligned = ft_side - vidshift - goCue[t]`; `np.interp(time_axis, ft_side_aligned, tspeed)`.

iii. The notes identify `findVideoOffset.m` as the reference and describe the approximately 0.5 s clock offset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw` DLC x, y, confidence, frame times, the session video shift, and go cue.

ii. `ft_bot, ts_bot, feat_bot = ...`; `px, py, pc = ts_bot[pi, 0, :], ...`.

iii. The notes deliberately select the bottom-camera `top_paw` feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing mirrors tongue velocity: finite differences form speed, confidence below 0.5 is missing, and both speed and visibility are linearly interpolated to bin centers, without smoothing.

ii. `pspeed = compute_velocity(px, py, ft_bot)`; `pspeed[~paw_visible] = np.nan`; `pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)`.

iii. The notes say trajectory format differences are handled and velocities are discretized per session.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session-wide median of all valid paw bins defines low (0) versus high (1); unavailable bins are not visible (2).

ii. `paw_thresh = np.nanpercentile(paw_vel_all[paw_valid], 50)` and `paw_disc = np.full(..., 2)`.

iii. This follows the requested per-session percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera times are shifted to behavior time, made relative to the go cue, and linearly interpolated at the neural time centers.

ii. `ft_bot_aligned = ft_bot - vidshift - goCue[t]`; `np.interp(time_axis, ft_bot_aligned, pspeed)`.

iii. The AI cites the common video-offset procedure.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's standalone `motionEnergy_<animal>_<date>.mat`, supporting direct cell, struct, and nested-struct layouts, plus side-camera frame times and alignment variables.

ii. `me_data = sio.loadmat(me_path, squeeze_me=False)`; wrapper logic extracts `me_s['data']`; processing uses `ft_side_aligned`.

iii. The notes say motion energy was loaded for all sessions and nested formats were handled.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Values are linearly interpolated to time centers. If trace and side-camera lengths differ, synthetic evenly spaced timestamps spanning the side-camera endpoints are created. No smoothing or spatial processing is applied.

ii. `me_interp = np.interp(time_axis, ft_side_aligned, me_trial)`; mismatch fallback: `me_ft = np.linspace(ft_side[0], ft_side[-1], len(me_trial))`.

iii. The AI considers the saved trace already processed and says it properly loaded motion energy for all sessions.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median of every valid motion-energy bin in a session defines low (0) and high (1); missing/no-video bins are 2. The file's `moveThresh` is loaded but not used.

ii. `me_thresh_disc = np.nanpercentile(me_all[me_valid], 50)`; `me_disc` is initialized to 2 and then assigned 0/1.

iii. This follows the prompt's per-session 50th-percentile requirement.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It assumes motion energy follows side-camera frames, subtracts session `vidshift` and trial go cue, and interpolates to 10 ms neural-bin centers.

ii. `ft_side_aligned = ft_side - vidshift - goCue[t]`; `np.interp(time_axis, ft_side_aligned, me_trial)`.

iii. The notes report use of reference video alignment and motion energy interpolation.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Video arrays start as NaN and categorical outputs as class 2. A broad per-trial `try/except` silently converts any tracking or motion-energy error into missing outputs. Missing motion-energy files return `None`; zero frame-time differences are changed to `1e-6`; unequal motion-energy lengths receive synthetic timestamps. Neural trials after recording end are retained as zeros.

ii. `except Exception as e: pass  # Leave as NaN`; `dt[dt == 0] = 1e-6`; `me_ft = np.linspace(...)`.

iii. The notes regard all-zero neural trials as expected gaps and emphasize handling mixed formats and motion-energy wrappers.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's logs imply session file loading plus nested per-neuron/per-trial spike binning and per-trial video interpolation dominate; full conversion took about 148 seconds. The code does not profile individual stages.

ii. The largest loop nest is `for ci, clu in enumerate(clusters): for trial_num in range(1, n_trials + 1): ... np.histogram(...) ... causal_gaussian_smooth(...)`.

iii. The notes extrapolated runtime from a two-session sample and recorded the final 148 s runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron-by-trial spike loop could histogram trial and time jointly; smoothing could operate over a rectangular matrix. Per-trial output assembly and parts of video interpolation could also be batched, though ragged camera arrays make the latter harder.

ii. Relevant loops are the nested loop at lines 383–391, the trial video loop at 440–493, and output-building loop at 523–533.

iii. The AI gave no explicit vectorization analysis; it prioritized correctness and observed acceptable runtime.

## 11-c. What processing does the code repeat multiple times?

i. For every trial it retrieves both camera views, searches feature-name lists, computes/interpolates speed and visibility separately, and creates a fresh copy of the identical time input. Spike masks rescan each cluster's full spike-trial vector once per trial.

ii. `spk_mask = clu['trial'] == trial_num`; `feat_side.index('tongue')`; `feat_bot.index('top_paw')`; `input_trials.append(time_input.copy())`.

iii. The AI did not discuss repeated processing; its notes focus on format handling and overall runtime.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `sample`, `delay`, `stim_enable`, and motion-energy `moveThresh` but never uses them. It loads bottom-camera data even though tongue uses only side view (bottom remains necessary for paw), computes confidence interpolation separately from speed, and stores fields such as cluster quality/probe that are discarded after extraction.

ii. `info['sample'] = ...`, `info['delay'] = ...`, `info['stim_enable'] = ...`; `me_trials, me_thresh = ...` where `me_thresh` is unused.

iii. The notes do not identify these as waste; `stim_enable` was explored as a curation field but the final decision retained all trials.
