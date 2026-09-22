# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 25 fixed-delay and 19 randomized-delay sessions and their selected probes, opens each `data_structure_<animal>_<date>.mat` with a custom MATLAB v7/v7.3 `Session` reader, and separately loads the corresponding motion-energy file. Sessions are processed in an eight-process pool.

ii. `SESSIONS = FIXED + RANDOM`; `s = Session(data_path(entry))`; `me_trials, _ = load_motion_energy(me_path(entry))`; `with Pool(nproc) as pool: results = pool.map(_worker, ...)`.

iii. The notes say the list was transcribed from the authors' `load<ANM>_ALMVideo.m` scripts so commented-out, dummy, and non-ephys sessions are excluded. A dual reader is justified by mixed MAT formats, and parallelism by per-trial HDF5 I/O cost.

## 1-b. How are the data split into subjects?

i. The subject is the `anm` field in each hard-coded session tuple. Subjects are added in first-session order, and each session receives the corresponding integer index.

ii. `if r['animal'] not in subjects: subjects.append(r['animal'])`; `data['subject_idx'].append(subjects.index(r['animal']))`.

iii. The notes identify 14 animals and treat the animal token embedded in every session filename as authoritative.

## 1-c. How are the data split into sessions?

i. Each listed `(folder, animal, date, probes)` tuple and its `data_structure` file is one session and one outer-list element. Sessions with fewer than 10 curated units or two usable trials would be skipped; all 44 passed.

ii. `for r in results: ... data['neural'].append(r['neural'])`; `if r['nunits'] < MIN_UNITS: ... continue`.

iii. The agent cites the paper's at-least-10-unit criterion and the target format's two-trial minimum.

## 1-d. How are the data split into trials?

i. Trials are the first `bp.Ntrials` rows. Per-trial behavior arrays, cluster `trial` indices, trajectory cells, and motion-energy cells are indexed by the same zero-based trial index after converting the spike trial numbers from one-based form.

ii. `ntrials = s.ntrials`; `trials = np.where(keep)[0]`; `aligned = tt - gocue[(tr - 1).astype(int)]`; `for i in trials:`.

iii. The notes describe Bpod fields and video/motion-energy arrays as explicitly trial-indexed, so trial boundaries need not be inferred.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials with no photostimulation, no early lick, a finite go cue, and at least one of hit/miss/no. It subsequently drops every kept trial having zero spikes from all retained units anywhere in the five-second window.

ii. `keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])`; `keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0`; `has_spikes = rates[trials].sum(axis=(1, 2)) > 0`.

iii. Early/stim exclusions follow the authors' condition strings. The zero-spike rule was added after verification found trailing trials recorded after ephys had stopped; the notes say ignore trials are deliberately retained because they are requested output classes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected probes in `obj.clu`: each cluster's `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment and probe location for region labels.

ii. `for c in s.clusters(p - 1):`; `tr = np.asarray(c['trial']...)`; `tt = np.asarray(c['trialtm']...)`; `gocue = s.ev(ALIGN_EVENT)`.

iii. The agent maps these to the authors' `findClusters`, `alignSpikes`, and `getSeq` routines.

## 2-b. How is the `neural` data processed?

i. Per unit, aligned spikes are histogrammed into 30 ms bins, divided by 0.03 to obtain spikes/s, and smoothed with a 15-bin causal half-Gaussian using reflected prefix padding. Selected probes are concatenated; no z-scoring is applied.

ii. `cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])`; `rates[:, :, i] = mysmooth(cnt.T / DT).T`; `k[:N // 2] = 0.0`.

iii. The agent chose a Figure 3 setting (`dt=(1/100)*3`) and says causal smoothing avoids future leakage. It claimed this ports `getSeq.m`/`mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters whose trimmed lower-case label is in `{garbage, gabrga, noisy, real?, ''}` are removed; remaining units must have raw mean firing rate strictly above 1 Hz over all trials and the five-second window. Sessions must retain at least 10 units.

ii. `BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}`; `keep_units = meanfr > LOW_FR`; `if r['nunits'] < MIN_UNITS: ... continue`.

iii. The agent cites `findClusters.m`, `removeLowFRClusters.m`, and the paper's >1 Hz and >=10-unit criteria, adding case/whitespace normalization for messy labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time is shifted by that trial's go-cue time before histogramming into the common -2.5 to +2.5 s grid.

ii. `aligned = tt - gocue[(tr - 1).astype(int)]`.

iii. This is identified as a direct port of `alignSpikes.m` with `alignEvent='goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 30 ms bins, yielding 166 bins from nominally -2.5 to +2.5 s. Raw spikes are directly binned at 30 ms; video signals are aggregated/interpolated to that axis. No later rebinning is performed.

ii. `DT = 0.03`; `edges, taxis = time_axis()`; metadata stores `time_bin_size=DT * 1000.0`.

iii. The agent acknowledges reference scripts use 5, 10, and 30 ms, choosing 30 ms for tractable output size and citing Figure 3 plus adequate resolution for licking.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is constructed from the configured `TMIN`, `TMAX`, and `DT`, conceptually relative to each trial's `bp.ev.goCue`; it does not read a separate raw trace.

ii. `edges = TMIN + DT * np.arange(nedges)`; `t = edges + DT / 2.0`; `tin = taxis.astype(np.float32).reshape(1, -1)`.

iii. The agent says the axis mirrors `getSeq.m` and the paper's go-cue window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are computed from the uniform edges and copied unchanged into every trial.

ii. `return edges, t[:-1]`; `inp.append(tin.copy())`.

iii. The notes treat this as the common analysis time axis; no further transformation is justified or applied.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input centers and neural histogram edges are returned by the same `time_axis()` call, while spikes are shifted by go cue before binning.

ii. `edges, taxis = time_axis()` and `cnt, _, _ = np.histogram2d(..., bins=[trial_edges, edges])`.

iii. Sharing one grid is presented as guaranteeing binwise alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.ev.lickL`, `bp.ev.lickR`, and `bp.ev.goCue`.

ii. `lickL, lickR = s.ev_cell('lickL'), s.ev_cell('lickR')`; `tl = np.concatenate([lickL[i] - gocue[i], lickR[i] - gocue[i]])`.

iii. The agent cites the authors' `firstLickTime.m` and reports agreement with outcome/instructed-side logic in its checks.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It finds the earliest strictly post-go-cue left or right lick; no such lick maps to class 2. The per-trial class is broadcast over all bins.

ii. `m = tl > 0`; `lick_dir[i] = int(side[m][np.argmin(tl[m])])`; `o[0] = lick_dir[i]`.

iii. The notes say actual first-lick side is the most direct behavioral measure and was 100% consistent on tested sessions.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from `obj.bp.autowater`.

ii. `autowater = s.bpvec('autowater')`.

iii. The agent identifies autowater trials as water-cued (WC) and all others as delayed-response (DR).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Positive autowater maps to WC=0 and otherwise DR=1; the value is broadcast in time.

ii. `context = np.where(autowater[:ntrials] > 0, 0, 1)`; `o[1] = context[i]`.

iii. This is described as a direct categorical relabeling.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `obj.bp.hit` and `obj.bp.miss`, with the residual class corresponding to `bp.no`/ignore.

ii. `hit = s.bpvec('hit')`; `miss = s.bpvec('miss')`; `no = s.bpvec('no')`.

iii. The notes map this to `getOutcome.m`, retaining ignores because the prompt explicitly requests them.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Default ignore=2 is overwritten by miss=0 and hit=1, then broadcast over time.

ii. `outcome = np.full(ntrials, 2)`; `outcome[miss[:ntrials] > 0] = 0`; `outcome[hit[:ntrials] > 0] = 1`.

iii. The class ordering follows the requested output values.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` feature's x/y coordinates from `obj.traj`, its frame times/drop status, go cues, and the session video-clock shift.

ii. `i_tongue = fn0.index('tongue')`; `xy = ts0[:, :2, i_tongue]`; `vt0 = ft0 - vidshift - gocue[i]`.

iii. The agent says the side view and `findPosition`/`findVelocity` supply the relevant signal; it preserves DLC NaNs to implement the requested not-visible class.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Visibility is finite x and y. Within each contiguous visible run, x/y finite differences are combined as Euclidean speed, divided by median frame interval, and visible frame speeds are averaged in each 30 ms bin. No positional Gaussian smoothing is applied.

ii. `spd = speed_from_xy(xy) / dtf`; `vx = np.gradient(xy[seg, 0])`; `_accumulate(idx, inb, spd, vis, ...)`.

iii. The agent says runwise gradients prevent NaNs leaking into adjacent visible frames and that leaving gaps undefined is required for the explicit visibility category.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session median over finite, visible bins from retained trials defines class 0 below and class 1 at/above; other bins are class 2.

ii. `thr_t = np.nanpercentile(tongue_spd[m_t], 50)`; `tongue_cls = discretize(tongue_spd, tongue_vis, thr_t)`.

iii. This directly follows the prompt's per-session 50th-percentile specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Side-camera frame times are corrected by a session bitcode-derived video shift, shifted by each trial's go cue, digitized into the neural edges, and averaged per bin.

ii. `vidshift = s.vidshift()`; `vt0 = ft0 - vidshift - gocue[i]`; `idx = np.digitize(vt0, bin_edges) - 1`.

iii. The agent cites `findVideoOffset.m` and reports post-go-cue visibility/movement sanity checks.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both `top_paw` and `bottom_paw` x/y tracks from the bottom camera, plus bottom-camera frame times/drop status, go cues, and video shift.

ii. `i_paws = [fn1.index(f) for f in ('top_paw', 'bottom_paw') if f in fn1]`; `pxy = ts1[:, :2, j]`.

iii. The notes explicitly choose the mean over both visible paw markers as a decoder-oriented summary.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's runwise finite-difference speed is computed, the two speeds are NaN-meaned framewise, visibility is true if either is visible, and frame speeds are averaged into 30 ms bins.

ii. `spds.append(speed_from_xy(pxy) / dtf1)`; `spd_p = np.nanmean(np.vstack(spds), axis=0)`; `vis_p = np.any(np.vstack(viss), axis=0)`.

iii. The agent says combining markers retains a paw signal when either marker is visible and preserves missingness only when neither is tracked.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A per-session median over finite visible retained-trial bins gives below=0 and at/above=1; nonvisible bins are 2.

ii. `thr_p = np.nanpercentile(paw_spd[m_p], 50)`; `paw_cls = discretize(paw_spd, paw_vis, thr_p)`.

iii. The median and third visibility class follow the prompt.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera times are independently corrected by video shift and go cue, then digitized into the same edges as spikes.

ii. `vt1 = ft1 - vidshift - gocue[i]`; `idx1 = np.digitize(vt1, bin_edges) - 1`.

iii. The notes justify using the actual bottom-camera frame times rather than assuming cameras have identical frame arrays.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's standalone `motionEnergy_*.mat` `me.data`, with side-camera frame times, video shift, and go cues providing its timestamps.

ii. `me_trials, _ = load_motion_energy(me_path(entry))`; `mev = me_trials[i]`; `vt0 = ft0 - vidshift - gocue[i]`.

iii. The custom loader handles bare, wrapped, and doubly wrapped file layouts, following `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-frame trace is linearly interpolated to 30 ms bin centers, then every interpolation/extrapolation gap is nearest-filled. No smoothing or differentiation is added.

ii. `y = interp_to_axis(vt0[:n], mev[:n], taxis)`; `y = _fill_nearest(y)`.

iii. The agent says this ports MATLAB `interp1` plus `fillmissing(...,'nearest')` because motion energy is already spatially reduced per frame.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The session median over all finite retained-trial bins gives classes 0/1. Class 2 is reserved for a trial without usable video.

ii. `thr_m = np.nanpercentile(me_binned[m_m], 50)`; `me_cls = discretize(me_binned, has_video[:, None] & np.isfinite(me_binned), thr_m)`.

iii. The agent follows the requested per-session median and no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy samples inherit side-camera timestamps, corrected by session video shift and trial go cue, and are interpolated onto the neural bin centers.

ii. `vt0 = ft0 - vidshift - gocue[i]`; `interp_to_axis(vt0[:n], mev[:n], taxis)`.

iii. The agent maps this to `loadMotionEnergy.m` and uses the common analysis axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The loader handles MAT-version and motion-energy-layout variants, trims mismatched spike arrays, filters nonfinite spike times/go cues, normalizes quality text, skips unusable video trials, leaves kinematic gaps as class 2, and nearest-fills motion-energy gaps. Sessions with errors are skipped. Trials with no population spikes are removed.

ii. `n = min(tr.size, tt.size)`; `if ... np.all(~np.isfinite(ft0)) ...: continue`; `y = _fill_nearest(y)`; `except Exception ... return dict(..., error=str(exc))`.

iii. The notes describe these as robust handling of observed quirks. Kinematic gaps are retained because the requested category represents missing visibility; nearest fill for ME is attributed to reference MATLAB behavior.

## 11-a. What are the most time-consuming steps of the code?

i. Per-trial DLC reads from HDF5 dominate; video/DLC plus motion energy takes roughly 1.4–4.5 seconds per session, while neural binning is much cheaper. Parallel processing reduces the full run to about 16 seconds.

ii. `for i in trials: ts0, ft0, nd0 = s.traj_trial(0, i)`; timing records `timing['video']` and `timing['neural_bin']`.

iii. The notes explicitly identify each trajectory trial being a separately referenced HDF5 dataset as the bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-session, per-trial video loop, per-feature/paw loop, per-visible-run gradient loop, and per-unit spike loop could potentially be further batched, though ragged HDF5 datasets constrain vectorization. The agent already vectorizes across trials within each unit via `histogram2d`, across bins via `bincount`, and across sessions via multiprocessing.

ii. `for i, c in enumerate(clusters):`; `for i in trials:`; `for j in i_paws:`; `for a, b in zip(starts, stops):`.

iii. The notes claim roughly 10× improvement from histogramming, FFT convolution, bincount, and eight-way multiprocessing, while identifying per-trial HDF5 access as the remaining inefficiency.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly reads two trajectory datasets per trial, recomputes feature speeds separately, copies the identical time input for every trial, and broadcasts per-trial labels into every time bin. Feature-name and video-shift lookup are sensibly done once per session.

ii. `for i in trials: ... s.traj_trial(0, i); s.traj_trial(1, i)`; `inp.append(tin.copy())`; `o[0] = lick_dir[i]`.

iii. The agent emphasizes reusable session constants and parallel processing, but does not explicitly discuss the storage duplication from copied time axes and broadcast labels.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads both cameras and computes continuous kinematic/ME arrays that are discarded after categorization; it also collects `qualities`, timing/debug metadata, and motion-energy `moveThresh` that do not enter the final decoder arrays. Optional plotting stacks and summarizes outputs solely for diagnostics.

ii. `me_trials, _ = load_motion_energy(...)`; `qualities.append(q)`; `res = dict(... qualities=qualities, timing=timing, ...)`; continuous `tongue_spd`, `paw_spd`, and `me_binned` are not placed in the final pickle.

iii. The notes frame most of this as necessary intermediate work or validation. `moveThresh` is deliberately ignored because the prompt requires a median threshold rather than the source movement threshold.
