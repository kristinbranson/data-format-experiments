# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 12 fixed-delay, two-context ALM sessions and their probe numbers, then opens each `data_structure_*.mat` with `h5py` and its separate motion-energy file with SciPy. It deliberately excludes fixed-delay sessions without both contexts and all randomized-delay sessions.

ii. `SESSIONS = [('JEB6', '2021-04-18', 2), ...]`; `f = h5py.File(data_path, 'r')`; `m = sio.loadmat(me_path, ...)`; `for k, (anm, date, probe) in enumerate(sessions): ...`

iii. The notes say these are exactly the 12 sessions used by Figure 8/ED Figure 2a-left and the only sessions containing both WC and DR, which the AI considered necessary for the context decoder.

## 1-b. How are the data split into subjects?

i. The animal string in each hard-coded `(anm, date, probe)` tuple defines the subject. Unique animals are appended in first-seen order and each retained session receives the corresponding index.

ii. `if anm not in data['subjects']: data['subjects'].append(anm)` and `data['subject_idx'].append(data['subjects'].index(anm))`.

iii. The notes identify seven distinct animal IDs among the selected sessions and explain that filenames/loaders provide the animal identity.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date pair and its `data_structure_<animal>_<date>.mat` file is one session and one element of each top-level session list. Sessions with fewer than two usable trials or ten retained units are skipped.

ii. `session_paths(anm, date)` constructs one file pair; `data['neural'].append(neural)`; `if len(neural) < 2: ... continue`; `if info['n_units'] < 10: ... continue`.

iii. The AI followed the curated MATLAB loader list and the paper's stated minimum of ten units per included session.

## 1-d. How are the data split into trials?

i. `obj.bp.Ntrials` defines the trial count. Bpod vectors are truncated to that count, spike records use their 1-based `clu.trial` field, and video/motion-energy cell entries are indexed by the same zero-based Python trial index. Each kept trial is emitted separately.

ii. `n = int(np.array(bp['Ntrials'])[0, 0])`; `return np.array(bp[name]).flatten()[:n]`; `tr = ...astype(int)`; `for j, tr in enumerate(keep_idx): neural_trials.append(...)`.

iii. The notes describe Bpod fields, spikes, trajectory cells, and motion energy as already trial-indexed, so no boundary reconstruction is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps valid hit/miss/no trials with a finite go cue, excludes early-lick and photostimulation trials, and then excludes trials lacking usable side-camera frame times. It does not implement the reference's end-of-ephys-recording cutoff.

ii. `keep = (bp['hit'] | bp['miss'] | bp['no']) & ~bp['early'] & ~bp['stim'] & ~np.isnan(bp['goCue'])`; later, `keep = keep & kin['has_video']`.

iii. Early/stim removal is justified from reference conditions. Ignore trials are retained because required output classes include ignore/none. Missing-video trials are dropped because three outputs were deemed undefined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected ALM probe in `obj.clu`: cluster `quality`, per-spike `trialtm`, and per-spike `trial`; `obj.bp.ev.goCue` supplies alignment times.

ii. `clu = f[o['clu'][probe - 1, 0]]`; `tm = ...clu['trialtm']`; `tr = ...clu['trial']`; `aligned = tm - gocue[tr - 1]`.

iii. Probe numbers were taken from the authors' animal loaders, and `trialtm - goCue` mirrors `alignSpikes.m`.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins over [-2.5, 2.5) s, divided by 0.01 to obtain Hz, then smoothed with a causal 15-sample Gaussian whose first seven coefficients are zero, using the AI's port of the reference reflect boundary behavior. No normalization or z-scoring is applied.

ii. `h, _, _ = np.histogram2d(...); rate[:, k, :] = h / DT`; `rate_s = my_smooth(rate.reshape(t, -1)).reshape(rate.shape)`; `k[:n // 2] = 0.0`.

iii. The AI chose `params.dt=1/100` and causal `mySmooth` because it found those values in publication figure scripts, despite the default parameters using 5 ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the listed ALM probe is used. Labels exactly matching `garbage`, `gabrga`, `noisy`, or `real?` are removed. Remaining units are retained when their mean firing rate, computed from condition-averaged smoothed PSTHs, exceeds 1 Hz. `poor` is retained.

ii. `BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')`; `qual_keep = [i for i, q in enumerate(quality) if q not in BAD_QUALITY]`; `return mean_fr > LOW_FR`.

iii. The label list and condition-averaged cutoff are justified as ports of `findClusters.m` and `removeLowFRClusters.m`; the 1 Hz threshold comes from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time is shifted by that trial's go-cue time before binning.

ii. `aligned = tm - gocue[tr - 1]`.

iii. The AI explicitly cites the reference `alignSpikes.m` go-cue operation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 bins of 10 ms each over five seconds. Raw spikes are rebinned into that grid; 400 Hz video streams are interpolated onto its bin centers.

ii. `DT = 0.01`; `edges = np.arange(TMIN, TMAX + DT / 2, DT)`; `taxis = (edges[:-1] + edges[1:]) / 2.0`.

iii. The AI judged 10 ms to be the value used by figure scripts, while noting the 5 ms default.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the configured go-cue-aligned window and bin edges, with `obj.bp.ev.goCue` defining zero for the underlying alignment.

ii. `taxis = (edges[:-1] + edges[1:]) / 2.0`; `input_trial = taxis.astype(np.float32)[None, :]`.

iii. The AI chose bin centers to match the reference `getSeq.m` time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Adjacent 10 ms edge pairs are averaged to create centers from -2.495 through 2.495 s; the same row is copied for every trial.

ii. `taxis = (edges[:-1] + edges[1:]) / 2.0`; `input_trials.append(input_trial.copy())`.

iii. This is documented as the `getSeq.m` convention.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It contains the centers of the exact edges used to histogram go-cue-shifted spikes, so its column indices coincide with neural columns.

ii. `aligned = tm - gocue[tr - 1]`; `np.histogram2d(... bins=[edges, trial_bins])`; `input_trial = taxis...[None, :]`.

iii. The common grid was chosen explicitly to align all modalities.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses Bpod `L`, `R`, `hit`, `miss`, and `no` trial flags.

ii. `lick_dir[(bp['L'] & bp['hit']) | (bp['R'] & bp['miss'])] = 0`; analogous expressions set right and no response.

iii. The notes cite the reference choice logic: correct trials use the instructed side, errors the opposite, and ignores have no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Left is encoded 0, right 1, and no response 2; the scalar trial label is repeated across all time bins.

ii. `lick_dir[bp['no']] = 2`; `out[0, :] = lick_dir[tr]`.

iii. The third class is retained to meet the decoder specification.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived solely from `obj.bp.autowater`.

ii. `autowater=g('autowater').astype(bool)`.

iii. The AI identified autowater as the direct WC-versus-DR marker.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater trials become WC=0 and all others DR=1; the value is repeated across time.

ii. `context = np.where(bp['autowater'], 0, 1)`; `out[1, :] = context[tr]`.

iii. The mapping follows the requested output order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses Bpod `miss`, `hit`, and `no` flags.

ii. `outcome[bp['miss']] = 0`; `outcome[bp['hit']] = 1`; `outcome[bp['no']] = 2`.

iii. These mutually exclusive task flags directly represent the three requested outcomes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Incorrect, correct, and ignore are encoded 0, 1, and 2 and repeated over time.

ii. `out[2, :] = outcome[tr]`.

iii. The encoding follows the prompt, with ignores deliberately retained despite their exclusion from paper analyses.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses only the side-camera `tongue` x/y tracks from `obj.traj`, plus side-camera `frameTimes`, video-clock synchronization fields, and go cues. It does not use the bottom-camera tongue track.

ii. `i_tongue = n1.index('tongue')`; `x = resample(ts1[i_tongue, 0])`; `y = resample(ts1[i_tongue, 1])`.

iii. It says this follows `findPosition`/`findVelocity` and describes tongue as a side-camera feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y are linearly interpolated onto the 10 ms grid. Visibility is defined by non-NaN interpolated x. Coordinates are nearest-filled solely for differentiation, velocity is the magnitude of `np.gradient` in samples (not seconds), and invisible bins are later class 2. There is no coordinate smoothing or two-view normalization/combination.

ii. `vis = ~np.isnan(x)`; `vx = np.gradient(nearest_fill(x))`; `tongue_speed[:, tr] = np.hypot(vx, vy)`.

iii. The AI interpreted the MATLAB pipeline as interpolation followed by gradient, with tongue NaNs not treated as observed values.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single median is computed over visible, non-NaN bins of retained trials in each session. Values below it are 0, values at/above it are 1, and invisible bins are 2.

ii. `thresh = np.percentile(vals[vis], 50)`; `out[vis & (vals < thresh)] = 0`; `out[vis & (vals >= thresh)] = 1`.

iii. This directly implements the requested per-session 50th-percentile split and preserves the not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset is estimated from synchronized bitcodes; it and the trial go cue are subtracted from frame times, then positions are interpolated onto neural bin centers.

ii. `vidshift = mode_value(bitstart) / fs - mode_value(bp['bitStart'])`; `return ft - vidshift - gocue_t`; `np.interp(taxis, times, y, ...)`.

iii. The AI cites `findVideoOffset.m` and uses one shared time grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both `top_paw` and `bottom_paw` x/y tracks from the bottom camera, its frame times, synchronization fields, and go cues.

ii. `i_paws = [n2.index(p) for p in ('top_paw', 'bottom_paw') if p in n2]`.

iii. The notes say the Methods use paws from the bottom view; the AI averages available tracked paws.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw is interpolated to 10 ms, nearest-filled, differentiated, corrected by subtracting median positional drift separately in x/y, and converted to speed. Available paw speeds are averaged per bin.

ii. `vx = np.gradient(pxf) - np.median(np.diff(pxf))`; `s = np.hypot(vx, vy)`; `paw_speed[:, tr] = ...np.nansum(sm...)/cnt`.

iii. The AI describes this as a port of `findPosition.m` and `findVelocity.m`, including non-tongue baseline-drift removal.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session median over visible retained-trial bins separates class 0 from class 1; missing/untracked bins are class 2.

ii. `paw_cat, paw_thresh = discretize(kin['paw_speed'], kin['paw_vis'], keep_idx)`.

iii. This implements the prompt's per-session percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are corrected by the bitcode-derived session offset and trial go cue, then paw coordinates are interpolated onto the neural bin centers.

ii. `tt2 = trial_frame_times(... bp['goCue'][tr])`; `px = resample(ts2[i, 0], tt2)`.

iii. The shared aligned grid was intended to reproduce the reference video alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses per-trial traces in separate `motionEnergy_<animal>_<date>.mat` files, along with side-camera frame times, synchronization fields, and go cues.

ii. `me_cells, move_thresh = load_motion_energy(me_path)`; `mv = np.asarray(me_cells[tr], float).flatten()`.

iii. The standalone file is used because that is the active reference loading path.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already spatially reduced trace is linearly interpolated to 10 ms bin centers, and edge NaNs are nearest-filled. The supplied `moveThresh` is recorded but not used for categories.

ii. `me[:, tr] = nearest_fill(np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))`.

iii. The AI says this reproduces `loadMotionEnergy.m`; no new pixel-level energy is computed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median over non-NaN retained-trial session values is used: below=0, at/above=1, missing/no-video=2.

ii. `me_cat, me_thresh = discretize(kin['me'], me_vis, keep_idx)`.

iii. The requested 50th percentile supersedes the paper's stored manual movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times receive the session offset and trial go-cue correction; values are interpolated to neural bin centers.

ii. `tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])`; `np.interp(taxis, tt[:k], mv[:k], ...)`.

iii. Motion energy has one value per side-camera frame, so the same video clock alignment is used.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing stim fields default to all false. Entirely absent/NaN frame-time trials are marked as lacking video and ultimately dropped. Partial coordinate gaps are nearest-filled for velocity calculation but remain invisible in categorical outputs. Length mismatches are truncated to the shorter stream. Assertions catch undefined trial labels.

ii. `except ...: out['stim'] = np.zeros(n, bool)`; `if ft.size == 0 or np.all(np.isnan(ft)): return None`; `k = min(len(times), len(y))`; `assert (outcome[keep_idx] >= 0).all()`.

iii. The AI viewed missing-video trials as unusable for three required outputs and avoided extrapolation beyond available timestamps, while using nearest fill to match MATLAB processing internally.

## 11-a. What are the most time-consuming steps of the code?

i. The code explicitly times spike binning, smoothing, and kinematics per session and reports them; HDF5 loading and trial-wise video interpolation/kinematics are also substantial. Full conversion output reports total elapsed time.

ii. `timing['bin_spikes'] = ...`; `timing['smooth'] = ...`; `timing['kinematics'] = ...`.

iii. The timing instrumentation and diagnostics were added to identify conversion costs; the notes emphasize validation rather than claiming one universal bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. It loops over units for spike histograms, trials and paw features for kinematics, columns for smoothing, and retained trials for final assembly. Some rectangular spike/smoothing work could be more vectorized, while ragged frame/spike arrays make full vectorization difficult.

ii. `for k, ci in enumerate(keep_idx):`; `for tr in range(n):`; `for i in i_paws`; `for j in range(xf.shape[1]):`.

iii. The implementation favors faithful handling of ragged MATLAB cell data and transparent per-session diagnostics.

## 11-c. What processing does the code repeat multiple times?

i. It interpolates each feature and trial separately; nearest-fills x and y separately; copies the identical time input per trial; and calculates the same per-trial frame-time transformation separately for side and bottom views. Diagnostic mode also reopens raw data and recomputes selected interpolations for plots.

ii. `x = resample(...)`; `y = resample(...)`; `input_trials.append(input_trial.copy())`; `f = h5py.File(extras['raw_path'], 'r')`.

iii. Repetition is mainly a consequence of different feature/view lengths and optional validation plots.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads lick timestamps and several event fields used only by diagnostic plots, computes/reporting single-unit counts and the stored motion threshold without using them in converted outputs, retains large `extras` including unsmoothed rates for optional plots, and computes detailed provenance/timing metadata not consumed by the decoder.

ii. `out['lickL'] = [...]`; `move_thresh=move_thresh`; `extras = dict(... rate=rate, ...)`; `n_single = sum(...)`.

iii. These values support sanity checks, paper-statistic comparisons, provenance, and optional processing figures rather than model training.
