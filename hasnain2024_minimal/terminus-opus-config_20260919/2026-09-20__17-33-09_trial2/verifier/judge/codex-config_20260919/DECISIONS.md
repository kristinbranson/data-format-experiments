# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent parses every uncommented animal/date/probe entry in the authors' `load<ANM>_ALMVideo.m` scripts, retains entries with a matching file in either ephys directory, and loads MATLAB v7 files with SciPy and v7.3 files with h5py. Motion-energy files are loaded separately. This produced 44 sessions.

ii. `sessions = parse_sessions()`; `sessions = [s for s in sessions if find_files(s['anm'], s['date'])[0] is not None]`; `if is_v73(fn): return _load_v73(fn, probes, traj_feats)`; `return _load_v7(fn, probes, traj_feats)`

iii. The trajectory says the loading scripts are the paper's curated session/probe list, commented entries should be excluded, and the available files mix MATLAB formats. It explicitly chose all 25 fixed-delay/two-context and 19 randomized-delay sessions.

## 1-b. How are the data split into subjects?

i. The parsed `anm` field identifies the mouse. Unique animal IDs are sorted, and every session receives its index in that list.

ii. `subjects = sorted({r['anm'] for r in res})`; `'subject_idx': np.array([subjects.index(r['anm']) for r in res])`

iii. The agent relied on the animal IDs in the authors' loading scripts and reported 14 mice.

## 1-c. How are the data split into sessions?

i. Each parsed animal/date entry and corresponding `data_structure_<animal>_<date>.mat` file is one session and one outer-list element. The loading script also selects that session's ALM probe(s).

ii. `for d in DATA_DIRS: f = os.path.join(d, 'data_structure_%s_%s.mat' % (anm, date))`; `res = p.map(process_session, sessions, chunksize=1)`

iii. The trajectory states that the 25 plus 19 sessions match the paper's counts and that per-script probe selection is needed.

## 1-d. How are the data split into trials?

i. Trials use zero-based indices into arrays of length `bp.Ntrials`; spikes retain raw 1-based trial labels and are selected with `trials + 1`. Video arrays are indexed by the same zero-based trial index.

ii. `trials = np.where(keep)[0]`; `starts = np.searchsorted(tt, trials + 1, 'left')`; `tr = view['trials'][t]`

iii. The agent checked that `bp.fidx` was absent and video trial counts matched `Ntrials`, concluding no trial offset was needed.

## 1-e. How are trials filtered based on quality controls?

i. It removes early-lick, photostimulation, non-finite-go-cue, and unavailable-video-tail trials. After neural processing it also removes trials whose retained smoothed population is entirely zero. Ignore trials remain. Sessions need at least two retained trials.

ii. `keep = (d['early'] == 0) & (d['stim'] == 0) & np.isfinite(gocue)`; `if ntraj < N: keep[ntraj:] = False`; `nonempty = np.any(rates != 0, axis=(0, 1))`

iii. Early/stim removal was attributed to the paper; ignore trials were kept because requested. After verifier warnings, the agent found 61 trailing all-zero trials in two sessions and treated them as behavior recorded after ephys ended.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected `obj.clu` probes: each cluster's `trial`, `trialtm`, and `quality`, plus `obj.bp.ev.goCue` for alignment.

ii. `trial = ... g['trial'] ...`; `trialtm = ... g['trialtm'] ...`; `q = _h5str(... g['quality'] ...)`; `gocue = d[ALIGN_EVENT]`

iii. The trajectory identified these fields from `alignSpikes.m`, `findClusters.m`, and the data objects.

## 2-b. How is the `neural` data processed?

i. For each unit and trial, aligned spikes are histogrammed, divided by 0.01 s to Hz, and smoothed in time using the agent's port of a causal half-Gaussian `mySmooth` kernel of width 15 bins with reflected prefix padding.

ii. `cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]`; `rates[:, iu, :] = my_smooth(cnt / DT)`; `kern[:N // 2] = 0`

iii. The agent believed the paper's `mySmooth.m` required causal half-Gaussian smoothing with `smooth=15` and reflect boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Exact, case-sensitive labels `garbage`, `gabrga`, `noisy`, and `real?` are excluded at load time. Units with mean processed rate at or below 1 Hz are removed, and sessions with fewer than 10 remaining units are discarded.

ii. `QUAL_EXCLUDE = ('garbage', 'gabrga', 'noisy', 'real?')`; `use = mean_fr > LOW_FR`; `if use.sum() < MIN_UNITS: return None`

iii. The trajectory cites `params.quality='all'`, the paper's 1 Hz cutoff, and a paper-level minimum of ten units. It did not discuss case normalization or the reference's additional `poor` label.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time is shifted by that trial's go-cue time before binning.

ii. `tm = u['trialtm'] - gocue[np.clip(tt - 1, 0, N - 1)]`

iii. The agent states this follows `alignSpikes.m`; it also verified that WC trials store water presentation in `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent creates 500 non-overlapping 10 ms bins over -2.5 to +2.5 s. Raw spikes are rebinned into that grid; video streams are interpolated to its bin centers.

ii. `DT = 0.01`; `EDGES = np.arange(TMIN, TMAX + DT / 2, DT)`; `TAXIS = (EDGES + DT / 2)[:-1]`

iii. The trajectory selected `params.dt=1/100`, noting that dt varied among paper scripts, and treated 10 ms as the reference pipeline setting.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from constants `TMIN`, `TMAX`, and `DT`; raw `goCue` is used to align the streams but not to populate the repeated values directly.

ii. `TAXIS = (EDGES + DT / 2)[:-1]`; `time_in = TAXIS.astype(np.float32)[None, :]`

iii. The agent chose a single continuous, time-varying input as explicitly requested.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers from -2.495 through 2.495 seconds are cast to float32 and copied for every trial.

ii. `time_in = TAXIS.astype(np.float32)[None, :]`; `inputs.append(time_in.copy())`

iii. The trajectory says this fulfills “Time from go cue onset in seconds” and uses the common analysis time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses exactly the same 500 bin centers as the aligned spike histograms.

ii. `EDGES = ...`; `TAXIS = ...`; `np.histogram(..., bins=EDGES)`

iii. The agent designed one shared axis for all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no`.

ii. `hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0`; `R, L = d['R'] > 0, d['L'] > 0`

iii. The agent compared behavioral definitions with the authors' `getPrevChoice.m` and chose outcome/instruction-derived choice rather than reconstructing lick events.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct right trials and incorrect left trials become right (1); ignore/no trials become none (2); all other retained trials become left (0). The value is repeated across time.

ii. `right_lick = (R & hit) | (L & miss)`; `lick_dir = np.where(no, 2, np.where(right_lick, 1, 0))`; `out[0, :] = lick_dir[t]`

iii. The trajectory says this is identical to `getPrevChoice.m` and that ignore trials have no lick.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `bp.autowater`.

ii. `aw = d['autowater'] > 0`; `context = np.where(aw, 0, 1)`

iii. The agent established that autowater marks WC and its absence marks DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater trials are encoded WC=0 and all others DR=1, repeated across every bin.

ii. `context = np.where(aw, 0, 1)`; `out[1, :] = context[t]`

iii. This follows the requested two categorical contexts.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit` and `bp.no` (with all remaining trials treated as incorrect; `miss` is loaded but not directly used here).

ii. `hit, miss, no = ...`; `outcome = np.where(no, 2, np.where(hit, 1, 0))`

iii. The agent interpreted hit as correct, no as ignore, and the remainder as incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It encodes incorrect=0, correct=1, ignore=2 and repeats the trial label across time.

ii. `outcome = np.where(no, 2, np.where(hit, 1, 0))`; `out[2, :] = outcome[t]`

iii. Ignore trials are intentionally retained because ignore is a requested output category.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses x/y coordinates for the `tongue` feature from side-camera `obj.traj`, its `frameTimes` and `NdroppedFrames`, plus ephys/behavior bit-start timing, sampling rate, and go cue.

ii. `TONGUE_VIEW, TONGUE_FEATS = 0, ['tongue']`; `tr['xy'][:, 0, fi]`; `tr['frameTimes']`; `d['bitstart_sglx']`; `d['bitStart']`

iii. The agent chose the side-camera tongue marker and described it as the requested scalar speed.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X and y are linearly interpolated to 10 ms bin centers, differentiated, and combined as Euclidean speed. No likelihood threshold or position smoothing is applied; invalid/dropped-video trials remain NaN.

ii. `x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])`; `vx = np.gradient(x) / DT`; `sp.append(np.sqrt(vx ** 2 + vy ** 2))`

iii. The trajectory says interpolation and gradients follow `findVelocity.m`, and NaNs should represent non-visible tongue samples.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The median of every finite tongue-speed sample in the session is used: finite values below it are 0, at/above it are 1, and NaNs are 2.

ii. `thr = np.percentile(x[vis], 50)`; `out[vis] = (x[vis] >= thr).astype(np.int64)`; `out = np.full(x.shape, 2, dtype=np.int64)`

iii. The agent explicitly reasoned that the percentile should be over visible values only and that category 2 handles invisibility.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by a session clock offset and the trial go cue, then positions are interpolated at neural bin centers. The offset uses medians, not the reference's modes.

ii. `vidshift = (np.median(d['bitstart_sglx']) / d['fs']) - np.median(d['bitStart'])`; `tt = ft - vidshift - gocue[t]`; `interp_nan(TAXIS, tt, ...)`

iii. The agent intended to reproduce `findVideoOffset.m` and later sanity-checked that movement increased near time zero.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera x/y coordinates for both `top_paw` and `bottom_paw`, with the same frame/timing variables as tongue.

ii. `PAW_VIEW, PAW_FEATS = 1, ['top_paw', 'bottom_paw']`

iii. The agent observed that paw features exist in the bottom view and chose to average both markers.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each marker's x/y is interpolated, differentiated, converted to speed, then the two speeds are NaN-averaged. It does not apply DLC likelihood filtering or coordinate smoothing.

ii. `sp.append(np.sqrt(vx ** 2 + vy ** 2))`; `spd = np.where(np.all(np.isnan(sp), axis=0), np.nan, np.nanmean(sp, axis=0))`

iii. The trajectory describes paw velocity as the mean speed of top and bottom paw markers.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session-wide finite-sample median yields below=0 and at/above=1; NaN yields not-visible=2.

ii. `paw_cat, paw_thr = discretise(paw_speed)`

iii. The same visible-only median rationale was used for all movement streams.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times receive the median-derived clock correction and go-cue subtraction; coordinates are interpolated to the neural bin centers.

ii. `tt = ft - vidshift - gocue[t]`; `x = interp_nan(TAXIS, tt, ...)`

iii. The agent intended all video outputs to share the neural axis and checked qualitative go-cue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's `motionEnergy_*.mat` `me.data`, side-camera frame times, ephys and behavioral bit starts, sampling frequency, and go cue.

ii. `me_raw = load_motion_energy(me_fn) if me_fn else None`; `tr = tv['trials'][t]`; `tt = tr['frameTimes'] - vidshift - gocue[t]`

iii. The agent identified the separate files and that motion energy is sampled at video-frame resolution.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is linearly interpolated to bin centers. If its length mismatches frame times, synthetic 400 Hz timestamps and a fixed 0.5 s subtraction are used; missing data stays NaN.

ii. `me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])`; `ftimes = (np.arange(len(me_raw[t])) + 1) / 400.0`; `interp_nan(TAXIS, ftimes - 0.5 - gocue[t], me_raw[t])`

iii. The agent aimed to interpolate motion energy similarly to DLC streams; the trajectory does not specifically justify the 400 Hz/0.5 s fallback.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median over all finite session samples yields below=0 and at/above=1; NaN yields no-video=2.

ii. `me_cat, me_thr = discretise(me_t)`

iii. This implements the requested per-session 50th-percentile split and distinct no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Normally, side-camera frame times are corrected using median bit-start differences, shifted by go cue, and interpolated to neural bin centers. The mismatched-length fallback uses synthetic timing instead.

ii. `tt = tr['frameTimes'] - vidshift - gocue[t]`; `me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])`

iii. The agent cites `findVideoOffset.m` and a shared time axis, though its statistic is median rather than mode.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The loader supports two MATLAB formats, missing region metadata defaults to ALM, missing motion energy produces class 2, invalid/missing video remains NaN/class 2, empty frame times get synthetic 400 Hz times, trajectory-short tails and all-zero neural trials are dropped, and interpolation stays NaN outside coverage or across NaN endpoints.

ii. `except Exception: locs = ['ALM'] * len(probes)`; `me_raw = ... if me_fn else None`; `if ft.size == 0 ...: ft = (np.arange(...) + 1) / 400.0`; `nonempty = np.any(rates != 0, axis=(0, 1))`

iii. The agent investigated mixed formats, missing visibility/video, and late behavior after ephys. It added the zero-neural-trial filter after validation warnings.

## 11-a. What are the most time-consuming steps of the code?

i. MATLAB/HDF5 loading, per-unit/per-trial spike histograms and convolution, per-trial video interpolation/gradients, and writing the roughly 1.9 GB pickle dominate. Sessions are parallelized across 12 processes.

ii. `with Pool(args.nproc, maxtasksperchild=1) as p: res = p.map(process_session, sessions, chunksize=1)`; nested unit/trial loops call `np.histogram` and `interp_nan`.

iii. The trajectory anticipated scale, implemented multiprocessing, and reported a 16-second 44-session conversion; it did not provide a formal profile.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-by-trial histogram loop, trial-by-view-by-feature video loop, output assembly loop, and h5 trial-reading loops could be reduced or batched. In particular, a 2-D trial/time histogram can replace repeated per-trial histograms.

ii. `for iu, u in enumerate(units): ... for j in range(len(trials)): ... np.histogram(...)`; `for j, t in enumerate(trials): for view, feats_idx, dest in ...`

iii. The trajectory focused on multiprocessing rather than documenting vectorization opportunities.

## 11-c. What processing does the code repeat multiple times?

i. It interpolates x and y separately for every marker and trial, recalculates the session median twice in `discretise`, repeatedly searches subject indices, and computes/uses similar camera timing for tongue, paw, and motion energy.

ii. `thr = np.percentile(x[vis], 50)` followed by `float(np.percentile(x[vis], 50))`; `subjects.index(r['anm'])`; repeated `interp_nan(TAXIS, tt, ...)`

iii. No explicit justification was given; these are implementation redundancies visible in the code.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/stores cluster quality and probe fields after initial filtering, loads `sample`, `delay`, `has_fidx`, and dropped-frame arrays mostly for checks, computes detailed session metadata and thresholds unused by decoder tensors, and processes rates before discarding low-rate units and zero trials.

ii. `for k in ['goCue', 'sample', 'delay', 'bitStart']`; `out['has_fidx'] = 'fidx' in bp`; `units.append({... 'probe': p})`; `info = {...}`

iii. Some fields supported investigation and provenance, but the trajectory did not identify them as downstream necessities.
