# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 44 author-selected session/probe combinations (25 fixed-delay and 19 randomized-delay), opens each `data_structure_<animal>_<date>.mat`, and separately loads its motion-energy file. Custom readers support MATLAB v7.3/HDF5 and v7/MAT5.

ii. `SESSIONS = [...]`; `obj = open_session(fn)`; `me_cells = load_motion_energy(mefn)`

iii. It says the author load scripts are the definitive inclusion/probe list; this avoids three extra files that were excluded or unusable and handles both MATLAB formats found in the release.

## 1-b. How are the data split into subjects?

i. The animal string in each `SESSIONS` tuple is retained as `subject`; final assembly creates the unique subject list and session-level indices.

ii. `res = dict(..., subject=anm)`

iii. The notes identify 14 animals and use the filename/load-list animal identifier because it is consistently available.

## 1-c. How are the data split into sessions?

i. Every hard-coded animal/date tuple and corresponding MATLAB file is one session and one outer-list element.

ii. `def process_session(anm, date, probes, datadir, task, ...)`

iii. This follows the authors' per-session files and loading scripts, yielding 44 sessions.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines trial count; trial-indexed behavior/video fields and each spike's 1-based `clu.trial` index map observations to those trials. Retained trial indices are looped into individual matrices.

ii. `ntrials = obj.Ntrials`; `idx = (tr[m] - 1) * NT + b[m]`; `for j, t in enumerate(kt):`

iii. The notes report that spike indices always lie in `1..Ntrials` and behavioral outcome/side fields are complete.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only when not early-lick, not photostimulation, go cue finite, and the complete smoothed population array has nonzero total activity.

ii. `keep = (early < 0.5) & (stim < 0.5) & np.isfinite(gocue)`; `no_ephys = trialdat.sum(axis=(0, 1)) <= 0`; `keep &= ~no_ephys`

iii. Early/stim removal follows author conditions. The agent adds the zero-spike rule to remove 64 trailing trials after recordings ended, avoiding all-zero decoder inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data use selected probes' cluster `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment and probe metadata for regions.

ii. `tt, tr = obj.spikes(prb, i)`; `al = tt - gocue[tr - 1]`

iii. This is presented as a port of `findClusters`, `alignSpikes`, and `getSeq`.

## 2-b. How is the `neural` data processed?

i. Aligned spikes are histogrammed into 10 ms bins, divided by 0.01 to obtain Hz, and smoothed with a causal, truncated 15-bin Gaussian FIR with reflected prefix padding. No normalization or baseline subtraction is applied.

ii. `out /= DT`; `sm = my_smooth(out.reshape(NT, -1)).astype(np.float32)`

iii. The agent believed this exactly ported the authors' `getSeq.m`/`mySmooth.m` parameters and prevented backward information leakage.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cluster labels are compared case-insensitively against `garbage`, `gabrga`, `noisy`, and `real?`; remaining units must have mean aligned-window firing rate strictly above 1 Hz across all raw trials.

ii. `BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}`; `use = fr > LOW_FR`

iii. It follows `findClusters.m` and the paper's >1 Hz rule, deliberately fixing case sensitivity. Unlike the human solution, it retains `poor` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before binning.

ii. `al = tt - gocue[tr - 1]`

iii. The agent cites `alignSpikes.m` and `params.alignEvent='goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses 500 nonoverlapping 10 ms bins from -2.5 to +2.5 seconds; video streams are interpolated to their centers. No later rebinning is done.

ii. `DT = 0.01`; `TMIN, TMAX = -2.5, 2.5`; `TAXIS = time_axis()[1]`

iii. It selected `params.dt=1/100` from its reading of `WorkingWithDataObjs.m`, despite the human reference using 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the configured analysis window and bin width, not read as a raw trial variable; `bp.ev.goCue` defines zero through the alignment.

ii. `edges = np.arange(TMIN, TMAX + DT / 2, DT)`; `t = edges + DT / 2`

iii. The agent treats bin centers as the required continuous decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin edges are shifted by half a bin and the final extra center is dropped, producing -2.495 through 2.495 seconds as float32.

ii. `return edges, t[:-1]`; `inp = TAXIS.astype(np.float32)[None, :]`

iii. It intended to reproduce the reference time-axis convention.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The identical `TAXIS` associated with neural histogram bins is copied into every trial.

ii. `inputs.append(inp.copy())`

iii. Shared construction guarantees matching timepoints, though the chosen resolution differs from the human solution.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no`.

ii. `hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')`; `R, L = obj.bp('R'), obj.bp('L')`

iii. The agent notes that instructed side plus correctness identifies actual choice and validated it against lick contacts.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials take the instructed side, misses take its opposite, and ignores are class 2; codes are left 0, right 1, none 2 and are broadcast over time.

ii. `lickdir[((R > .5) & (hit > .5)) | ((L > .5) & (miss > .5))] = 1`; `lickdir[no > .5] = 2`

iii. This follows `getPrevChoice.m`, adding the required no-lick class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from per-trial `bp.autowater`.

ii. `autowater = obj.bp('autowater')`

iii. The paper/code describe autowater as the proxy for water-cued blocks.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is mapped to WC=0 and all other trials to DR=1, then broadcast over time.

ii. `context = np.where(autowater > 0.5, 0, 1).astype(np.int8)`

iii. This is a direct categorical relabeling required by the prompt.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit`, `bp.miss`, and `bp.no`.

ii. `hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')`

iii. The notes verify the flags are mutually exhaustive in every session.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss maps to incorrect=0, hit to correct=1, and no-response to ignore=2; values are broadcast over time.

ii. `outcome[miss > 0.5] = 0`; `outcome[hit > 0.5] = 1`; `outcome[no > 0.5] = 2`

iii. This directly matches the requested classes while retaining ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-camera `obj.traj` feature `tongue`: its x/y DLC coordinates, frame times, and dropped-frame marker, plus bitcode metadata and go cues for clock alignment.

ii. `TONGUE_VIEW, TONGUE_FEAT = 0, 'tongue'`; `xy, ft = obj.traj_xy(view, t, featix)`

iii. The agent chose the side tongue feature as the author's named tongue kinematic; it does not combine the bottom-camera tongue used by the human.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Raw x/y are linearly interpolated onto neural bin centers; NaN-aware finite differences are taken per axis and combined by Euclidean magnitude. Missing tracking remains NaN.

ii. `x = interp_matlab(taxis, tv, xy[0])`; `speed[t] = np.sqrt(nan_gradient(x) ** 2 + nan_gradient(y) ** 2)`

iii. It aimed to port `findPosition`/`findVelocity`, while avoiding artificial zero velocity at visibility-bout edges.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The median of all valid tongue-speed samples in retained trials is the per-session threshold: below=0, greater/equal=1, invalid=2.

ii. `thresh = np.nanpercentile(values[valid], 50)`; `v[valid] = (values[valid] >= thresh).astype(np.int8)`

iii. This implements the explicitly requested per-session 50th percentile and visibility class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-clock offset is estimated from SpikeGLX/Bpod bit starts; frame times subtract that offset and the trial go cue before interpolation to `TAXIS`.

ii. `vidshift = mode(bitstart)/fs - mode(bp_bitstart)`; `tv = ft - vidshift - gocue[t]`

iii. The formula follows `findVideoOffset.m`; the shared target grid supplies alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw` x/y coordinates and frame metadata, plus video bitcode offset and go cue.

ii. `PAW_VIEW, PAW_FEAT = 1, 'top_paw'`

iii. The author figure code uses this feature and it is the reliably tracked paw.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It uses the same interpolation, NaN-aware x/y gradient, and magnitude calculation as tongue velocity.

ii. `paw_speed, paw_vis = kinematic_speed(obj, PAW_VIEW, PAW_FEAT, ...)`

iii. The agent intended a direct port of position/velocity routines, omitting the reference per-axis tiny baseline correction.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Valid samples from retained trials are split at their session median; missing samples are class 2.

ii. `paw_d, paw_thr = discretize(paw_speed[keep], paw_vis[keep])`

iii. This follows the requested per-session percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are offset-corrected, made relative to each go cue, and interpolated to the neural bin-center grid.

ii. `tv = ft - vidshift - gocue[t]`; `interp_matlab(taxis, tv, xy[0])`

iii. It applies the common reference clock-correction formula.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads per-trial `me.data` from each standalone `motionEnergy_*.mat`, along with side-camera frame times, bitcode offset, and go cue.

ii. `me_cells = load_motion_energy(mefn)`; `ft = obj.frame_times(0, t)`

iii. Standalone files cover all sessions and their multiple wrapper layouts are handled explicitly.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already spatially reduced per-frame trace is linearly interpolated to the neural time grid; no additional smoothing is performed.

ii. `out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)`

iii. The agent says the upstream motion-energy computation is already complete and ports `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Valid retained-trial samples are split at the session median; bins without video are class 2.

ii. `me_d, me_thr = discretize(me[keep], me_vis[keep])`

iii. This replaces the authors' manual movement threshold because the task explicitly asks for the 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video offset and trial go cue, then interpolated at neural bin centers.

ii. `interp_matlab(taxis, ft - vidshift - gocue[t], m)`

iii. Motion energy has one sample per side-camera frame, making those timestamps the chosen clock.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Both MATLAB layouts and motion-energy wrappers are supported. Missing/mismatched/nonfinite frame times fall back to a nominal 400 Hz clock; unusable video/DLC remains class 2. Missing `stim.enable` becomes zeros, missing region becomes ALM, and zero-neural trials are removed.

ii. `ft = np.arange(1, xy.shape[1] + 1) / 400.0 - 0.5 + vidshift`; `stim = ... if obj.has_bp(...) else np.zeros(ntrials)`

iii. The agent justifies fallbacks from author code and uses explicit invalid classes to avoid NaNs, while preserving trials with only video missing.

## 11-a. What are the most time-consuming steps of the code?

i. The notes identify video interpolation as dominant (about 2.3 seconds/session), with total conversion measured at 107 seconds for 44 sessions; loading/curation varies by file.

ii. `timing['video'] = time.time() - t1`; `timing['neural'] = time.time() - t1`

iii. Timings were measured and recorded per session to guide optimization.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops remain for variable-length video and output-list assembly, and unit/probe loops remain for differently sized spike arrays. Spike counting is vectorized across trials per unit, and smoothing across every unit/trial trace.

ii. `for t in range(ntrials):`; `cnt = np.bincount(idx, minlength=ntrials * NT)`; `my_smooth(out.reshape(NT, -1))`

iii. The agent says it removed per-spike, per-trial histogram, and per-trace smoothing loops; ragged frame/spike sources limit further simple vectorization.

## 11-c. What processing does the code repeat multiple times?

i. `kinematic_speed` repeats the same per-trial interpolation/gradient pipeline separately for tongue and paw; session behavior fields and cluster spikes are also read in separate curation/binning passes.

ii. `tongue_speed, tongue_vis = kinematic_speed(...)`; `paw_speed, paw_vis = kinematic_speed(...)`; `mean_firing_rates(...)`; `bin_spikes(...)`

iii. Separate features require the same transform, while the firing-rate prepass is used to decide which units merit allocation/binning.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads sample/delay times, lick events (when plotting), quality/probe metadata, computes numerous diagnostics, and optionally plots them although most are not decoder arrays. It computes video/neural data for all trials before discarding failed-QC trials.

ii. `sample = obj.bp('ev.sample')`; `delay = obj.bp('ev.delay')`; `info = dict(...)`; `trialdat = bin_spikes(..., ntrials)`

iii. These support provenance, validation, plotting, and thresholds; processing all trials also preserves the reference's all-trial unit-rate criterion, but some work is subsequently discarded.
