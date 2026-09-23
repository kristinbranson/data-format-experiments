# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively identifies every subject/session NWB file with a sorted glob, opens each file directly with `h5py`, processes sessions in a spawn-based multiprocessing pool, caches each session, and combines all usable sessions into one pickle.

ii. `paths = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))` and `f = h5py.File(path, 'r')`; jobs are run with `ctx.Pool(args.workers, maxtasksperchild=1)` and later loaded from per-session pickle files.

iii. The trajectory says it inventoried all 152 NWB files (11 subjects), chose direct HDF5 access after inspecting the NWB structure, and parallelized the expensive 87-GB conversion. It verified that the result contained 152 sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is read from NWB metadata for each file. A unique subject list is built in session order, and each session receives its index into that list.

ii. `subject = f['general/subject/subject_id'][()].decode()`; later, `if sub not in subjects: subjects.append(sub)` and `data['subject_idx'].append(subjects.index(sub))`.

iii. The trajectory confirmed 11 subject IDs during its metadata scan and used file metadata rather than relying solely on directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; each processed file contributes one element to the session-level `neural`, `input`, and `output` lists.

ii. `res = process_session(path, signal=signal)` and then `data['neural'].append(res['neural'])`, with corresponding input/output appends.

iii. The agent inferred this from the `sub-*/sub-*_ses-*_behavior+ophys.nwb` organization and NWB session metadata.

## 1-d. How are the data split into trials?

i. A trial begins at each positive `trial_start` sample and ends just before the corresponding positive `teleport` sample. The same `[s:e]` slice is used for all streams.

ii. `starts = np.where(get('trial_start') > 0)[0]`, `stops = np.where(get('teleport') > 0)[0]`, followed by `for i, (s, e) in enumerate(zip(starts, stops)):` and `act = activity[:, s:e]`.

iii. The trajectory reports that this matches the repository's lap definition and validated equal counts and plausible trial durations on real sessions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped when the lick sensor is deemed erroneous (>30% of trial frames have lick count >2), when neural activity contains non-finite values, or when position, speed, or lick samples are non-finite. Sessions with fewer than two retained trials are dropped. There is no reference-style minimum-50-frame filter.

ii. `lick_error[i] = (np.sum(seg > LICK_ERR_COUNT) / len(seg)) > LICK_ERR_FRAC`; then `if lick_error[i]: continue`, `if not np.all(np.isfinite(act)): continue`, and analogous behavior checks. Combination uses `if len(res['neural']) < 2: ... continue`.

iii. The agent cites `behavior.correct_lick_sensor_error` and notes that exactly 81 trials were removed, matching the paper's reported lick-error count. Other finite-value checks are defensive format protections.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from suite2p raw fluorescence `Fluorescence`, neuropil fluorescence `Neuropil`, ROI curation `iscell`, and plane membership `planeIdx`; it does not use the stored NWB `Deconvolved` signal.

ii. `F[rois] = ophys['Fluorescence'][plane]['data'][:].T`, `Fneu[rois] = ophys['Neuropil'][plane]['data'][:].T`, and `iscell = seg['iscell'][:, 0] > 0`.

iii. The agent reasoned that the paper/repository recomputes dF/F and optional OASIS events from F and Fneu, whereas the stored deconvolution is suite2p's distinct preprocessing.

## 2-b. How is the `neural` data processed?

i. Curated multi-plane ROIs are pooled. Per trial, the code subtracts `0.7*Fneu`, adds back mean neuropil, estimates a maximin baseline (15-sample Gaussian, up-to-300-sample minimum then maximum filters), computes dF/F, smooths it by two samples, and also computes OASIS events (`tau=0.7`). The saved default signal is smoothed dF/F, not events.

ii. `f_ -= NEU_COEF * fneu_`; `seg = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH])`; `seg = minimum_filter1d(seg, win, axis=-1)`; `dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])`; and `activity = events if signal == 'events' else dff`, with `--signal` defaulting to `dff`.

iii. The trajectory initially planned to use events, but a four-session decoder comparison favored dF/F (position accuracy 0.72 versus 0.51). It therefore chose dF/F because it retained graded amplitude, despite acknowledging that the paper's relevant decoding used deconvolved events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `iscell` ROIs are retained, then putative interneurons whose dF/F has Pearson correlation >0.5 with speed are removed.

ii. `F = F[iscell]`; correlation is computed vectorially and `is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH`; finally `activity = activity[~is_int]`.

iii. The agent follows manual suite2p curation and the paper's speed-correlation interneuron exclusion, reporting 301 excluded cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned at `trial_start`; each neural trial starts at that sample, with no additional shifting or interpolation.

ii. `act = activity[:, s:e]`, where `s` is from `np.where(get('trial_start') > 0)[0]`.

iii. The agent found behavior streams already synchronized to imaging frames, so common slicing aligns every trial to its start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging/behavior frames are retained, approximately 64.5 ms (15.5 Hz); no temporal rebinning or resampling is applied. Metadata records the mean session bin duration.

ii. `frame_rate = 1.0 / np.median(np.diff(tstamps))` and `bin_ms = float(np.mean([1000.0 / s['frame_rate_hz'] for s in session_info]))`.

iii. The agent verified synchronized behavior and imaging sample rates and chose the native frame as the natural bin.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior `position/timestamps` array and the trial-start index.

ii. `tstamps = b['position/timestamps'][:]` and `t_rel = tstamps[s:e] - tstamps[s]`.

iii. The trajectory found behavior timestamps shared across synchronized streams and used position timestamps as the common clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of each trial is subtracted from every timestamp in that trial.

ii. `t_rel = tstamps[s:e] - tstamps[s]`; `inp[0] = t_rel`.

iii. This directly expresses elapsed seconds from the requested alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical `[s:e]` indices as neural data and therefore has exactly the same number/order of frames.

ii. `act = activity[:, s:e]`, `T = e - s`, and `t_rel = tstamps[s:e] - tstamps[s]`.

iii. The agent verified the streams were frame-synchronized; rare one-frame length mismatches are truncated before slicing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` (morph) time series.

ii. `morph = get('environment')` and `env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0 ...])`.

iii. Inspection showed this variable represents ENV1/ENV2 and is constant within a trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Each trial is reduced to binary 1 when its maximum environment value exceeds 0.5, otherwise 0, then broadcast across all trial frames.

ii. `inp[1] = env[i]` after the thresholded per-trial calculation above.

iii. The agent used the reduction because observed environment values were binary and trial-constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is taken from the raw behavioral `trial number` time series at the trial's first frame.

ii. `trialnum = get('trial number')` and `inp[2] = float(trialnum[s])`.

iii. The trajectory inspected the trial-number signal but gives no specific justification for preferring its stored value over the sequential extracted-trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The first stored value is cast to float and broadcast across the trial; no renumbering of retained trials occurs.

ii. `inp[2] = float(trialnum[s])`.

iii. This treats trial number as a per-trial contextual variable.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, behavior timestamps, and the `reward_zone` entry signal. Current outcomes are defined as both a reward event during the trial and any reward-zone entry.

ii. `reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])`; `rewarded[i] = int(bool(got_reward) and bool(in_zone))`; `prev_rewarded = np.concatenate([[1], rewarded[:-1]])`.

iii. The agent follows the repository's `get_trial_types` interpretation (reward delivered in-zone). It sets the first imaged trial's previous outcome to 1 because un-imaged warm-up trials were rewarded about 85% of the time.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The session outcome vector is shifted by one trial and prefixed with 1; that scalar is broadcast across the current trial. Filtering does not recompute adjacency, so “previous” means the preceding recorded trial even if that trial is later dropped.

ii. `prev_rewarded = np.concatenate([[1], rewarded[:-1]]).astype(np.float64)` and `inp[3] = prev_rewarded[i]`.

iii. The warm-up rationale above motivated the nonzero first value; shifting preserves chronological recorded-trial outcomes.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone identity parsed from the NWB scene/identifier. Scene names select A/B/C boundaries; switch scenes change zone after trial 30.

ii. `scene = ident.split('/')[-1]`; `zone_labels = scene_reward_zones(scene, ntrials)`; `zstart, zend = REWARD_ZONES[zone_labels[i]]` with fixed A `(80,130)`, B `(200,250)`, C `(320,370)`.

iii. The trajectory found zone definitions and switch timing in repository code and reported validating scene-derived labels against zone-entry positions with zero mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped below at zero. Signed distance is position minus the near edge before the zone, zero within the zone, and position minus the far edge after it.

ii. `p = np.clip(pos[s:e], 0.0, None)` and `dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))`.

iii. This implements “distance to any location in the reward zone,” with sign indicating before versus after.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is mapped explicitly to seven classes: `<-50`, `[-50,-10)`, `[-10,0)`, exactly 0, `(0,10]`, `(10,50]`, and `>50`.

ii. `discretize_reward_distance` assigns class 3 for `d == 0`, classes 2/1/0 for negative ranges, and 4/5/6 for positive ranges.

iii. The thresholds were chosen to reproduce the task specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the same trial slice and frame indices.

ii. Both `act = activity[:, s:e]` and `p = np.clip(pos[s:e], 0.0, None)` have length `T = e-s`.

iii. No interpolation is needed because NWB behavior is frame-synchronized to imaging.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavioral `position` time series.

ii. `pos = get('position')` and `p = np.clip(pos[s:e], 0.0, None)`.

iii. The agent identified this as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Negative values are clipped to zero, then position is divided into 90-cm intervals using floor; results are clipped to classes 0–4.

ii. `out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)`.

iii. The five equal bins follow directly from the 450-cm track; clipping absorbs slight out-of-track measurement noise.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Categories are `<90`, 90–<180, 180–<270, 270–<360, and ≥360 cm (with all values forced into 0–4).

ii. `np.clip(np.floor(p / 90.0), 0, 4)`.

iii. This implements five equal 90-cm bins spanning the track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same `[s:e]` sample interval is used for position and neural data.

ii. `act = activity[:, s:e]` and `p = np.clip(pos[s:e], 0.0, None)`.

iii. The synchronized NWB streams make index alignment sufficient.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii. `lick = get('lick')` and `lk = lick[s:e]`.

iii. The agent identified this as the frame-level lick measurement.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero are mapped to lick=1 and all others to 0; corrupted lick trials are removed beforehand.

ii. `out[3] = (lk > 0).astype(np.int64)`.

iii. Binarization is required by the requested output definition.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data are sliced with the same trial indices.

ii. `act = activity[:, s:e]` and `lk = lick[s:e]`.

iii. They already share the imaging-frame clock.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene name in the NWB `identifier`, not directly from behavioral `reward_zone`/position samples. The scene indicates fixed A/B/C or a switch between two zones.

ii. `scene = ident.split('/')[-1]` and `zone_labels = scene_reward_zones(scene, ntrials)`.

iii. The agent used repository naming conventions and validated inferred labels empirically against zone-entry locations.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regular expressions parse fixed and switch scene formats. Fixed sessions use one label throughout; switch sessions use the first label for 30 trials and the second thereafter. A/B/C are encoded 0/1/2 and broadcast across frames.

ii. `return [m.group(1)] * change_trial + [m.group(2)] * (ntrials - change_trial)` and `out[4] = ZONE_NAMES.index(zone_labels[i])`.

iii. The switch-after-30 rule and zone boundaries came from `behavior.get_reward_zones` and repository metadata.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses raw `Reward/timestamps`, common behavior timestamps, and the `reward_zone` time series.

ii. `reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])`, `rzone_entry = get('reward_zone')`, and `rewarded[i] = int(bool(got_reward) and bool(in_zone))`.

iii. The agent says this matches the paper repository's trial-type definition: reward must be delivered while the animal entered the active zone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame insertion indices. A trial is class 1 only if any mapped reward lies in `[s,e)` and any reward-zone entry sample is positive; otherwise it is 0. The scalar is broadcast across the trial.

ii. `got_reward = np.any((reward_frames >= s) & (reward_frames < e))`; `in_zone = np.any(rzone_entry[s:e] > 0)`; `out[5] = rewarded[i]`.

iii. This guards against counting rewards outside the operative zone/trial-type definition.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Ten two-plane sessions with one extra neural frame are truncated to the common neural/behavior length, and trial endpoints beyond it are removed. Trials with non-finite neural/behavior samples are skipped. Unknown scene names raise an error; sessions with fewer than two usable trials are skipped.

ii. `nsamp = min(F.shape[1], len(pos))`; arrays are sliced to `:nsamp`; `keep = stops <= nsamp`; finite checks use `np.all(np.isfinite(...))`; scene parsing ends with `raise ValueError(...)`.

iii. The mismatch was discovered during the full run and attributed to the last frame of interleaved two-plane scans. The agent chose common-length truncation to preserve sample alignment.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large F/Fneu arrays, per-trial Gaussian/maximin filtering, OASIS deconvolution, and writing/loading the roughly 9.6-GB pickle dominate. Parallel per-session processing and caching mitigate reruns.

ii. The expensive calls are `load_fluorescence`, the filter calls in `compute_activity`, `dcnv.oasis(...)`, and `pickle.dump(...)`.

iii. The trajectory measured an 87-GB source, benchmarked sessions, used spawn workers, and cached 152 session results. Full decoder validation also took several minutes but is outside conversion proper.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops in `compute_activity`, environment/outcome/lick-QC construction, and final trial packaging could be partly replaced by indexed reductions, though variable-length trials make complete vectorization awkward. Plane loading and file/session loops are naturally separate; cell-speed correlation is already vectorized.

ii. Examples are `for s, e in zip(starts, stops):` in `compute_activity`, the `for i, (s, e) ...` quality loop, and the final `for i, (s, e) ...` construction loop.

iii. The agent explicitly optimized the former per-cell correlation loop into matrix operations and relied on process-level parallelism for sessions; it did not discuss further loop vectorization in the final rationale.

## 13-c. What processing does the code repeat multiple times?

i. Each trial is traversed during copying, baseline estimation, smoothing/deconvolution, quality/outcome computation, and output construction. Sessions are also serialized to cache and immediately deserialized for assembly.

ii. `compute_activity` contains three `for s, e in zip(starts, stops)` passes; `process_session` adds two more trial passes; `worker` dumps each result and `main` reloads it.

iii. Repeated passes preserve readable, bounded-memory processing and support per-session caching, but they add overhead.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. With the default `--signal dff`, OASIS events are nevertheless computed for every cell/trial, allocated, returned, and then discarded. `plane_idx` is retained through processing but only used to determine a neuron count before all neurons are labeled CA1; `kept_trials` stores indices but only its length is used.

ii. `events[:, s:e] = dcnv.oasis(...)` is unconditional, followed by `activity = events if signal == 'events' else dff`; `plane_idx = plane_idx[keep_cells]` is returned but final `brain_region_idx` is just zeros.

iii. The trajectory originally supported both signal choices and benchmarked them, which explains unconditional event generation, but after selecting dF/F this became avoidable computation and memory traffic.
