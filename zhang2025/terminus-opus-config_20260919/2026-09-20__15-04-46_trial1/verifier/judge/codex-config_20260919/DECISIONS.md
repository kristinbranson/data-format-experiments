# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM release freeze CSV, optionally restricts it with `DATALIMIT_SUBSET.csv`, groups insertions by session, and uses ONE `SessionLoader`/`SpikeSortingLoader` objects to load trials, behavior, and every probe. Sessions are processed in a multiprocessing pool.

ii. `bwm = pd.read_csv(FREEZE_FILE, index_col=0)`; `for eid, g in bwm.groupby('eid', sort=False): jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))`; `sl = SessionLoader(one=one, eid=eid)`.

iii. The notes say the freeze contains 699 insertions, 459 sessions, and 139 subjects and is the reference release; ONE resolves the cached ALF objects. Session-level parallelism was chosen for speed and memory isolation.

## 1-b. How are the data split into subjects?

i. Subject IDs come directly from the freeze file. After session filtering, unique subjects are sorted and each retained session receives an index into that list.

ii. `g.subject.iloc[0]`; `subjects = sorted(set(s['subject'] for s in sessions))`; `subject_idx = np.array([subj_lut[s['subject']] for s in sessions], dtype=np.int64)`.

iii. The notes treat the release subject field as the authoritative mouse identifier; no filename parsing is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the freeze is a session. All probe insertions with that `eid` are grouped into one job and later merged into one session population.

ii. `for eid, g in bwm.groupby('eid', sort=False):`; `jobs.append((eid, list(g.pid), list(g.probe_name), g.subject.iloc[0]))`.

iii. The notes state that probes within a session are not independent and should be merged, following the reference code.

## 1-d. How are the data split into trials?

i. Rows of `SessionLoader.trials` define trials. Each retained row supplies a stimulus-aligned interval, and the resulting arrays are appended one trial at a time.

ii. `stim_on = trials[PARAMS['align_time']].to_numpy(dtype=float)`; `begs_all = stim_on + WIN[0]`; `ends_all = stim_on + WIN[1]`; `for k in range(n_trials): neural.append(...)`.

iii. The trials table already has one row per trial; the code only masks and windows those rows.

## 1-e. How are trials filtered based on quality controls?

i. It applies RT 0.08–2 s, response, six nonmissing-field, and maximum trial-length checks; requires a supported prior; requires wheel and motion-energy coverage; additionally requires the interval to fall within the spike-time span and contain at least one population spike. Sessions with fewer than two remaining trials are dropped.

ii. `mask = ~sess_loader.trials.eval(query)`; `valid = mask & wheel_ok & me_ok`; `valid &= (np.isclose(pl, 0.2) | ...)`; `valid &= in_span`; `has_spikes = binned.sum(axis=(1, 2)) > 0`.

iii. The agent says the first mask reproduces `load_trials_and_mask(..., max_trial_len=10.0)`, behavior coverage reproduces reference interpolation checks, and neural-coverage checks prevent all-zero trials caused by recording ends/dropouts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each probe's `spikes['times']` and `spikes['clusters']`; cluster labels and acronyms determine unit retention and brain-region metadata.

ii. `spikes, clusters, channels = ssl.load_spike_sorting()`; `cl = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()`; `st = np.asarray(spikes['times'])`; `sc = np.asarray(spikes['clusters'])`.

iii. The notes identify these as the variables used by the reference spike-loading and binning functions.

## 2-b. How is the `neural` data processed?

i. Probes are merged, retained clusters are reindexed, spikes are sorted by time, and spikes are counted in nonoverlapping 20 ms bins. The saved matrices are float32 spike counts, not rates or smoothed activity.

ii. `flat = spike_clusters[a:b] * NBINS + bi`; `counts = np.bincount(flat, minlength=n_units * NBINS)`; `out[k] = counts.reshape(n_units, NBINS)`; `neural.append(np.ascontiguousarray(binned[k]))`.

iii. The agent claims this mirrors reference `bincount2D`, avoids per-spike Python work, and preserves unsmoothed counts. Its README explicitly labels the result “spike counts per 20 ms bin.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains clusters with `label >= 1`, removes Beryl `root` and `void`, retains only regions having at least five good units in that session, and finally retains only regions occurring in at least two sessions.

ii. `good = cl['label'].to_numpy(dtype=float) >= 1.0`; `keep = good & ~np.isin(beryl, EXCLUDE_REGIONS)`; `counts[r] >= MIN_UNITS_PER_REGION`; `if c >= min_sess`.

iii. The agent attributes all these rules to data-paper curation and validates the good-unit count against the reported 108 units/probe.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial spans stimulus onset minus 0.5 s through plus 1.5 s. Absolute spike times are sliced against those interval bounds and binned relative to the interval start.

ii. `PARAMS = {... 'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}`; `begs_all = stim_on + WIN[0]`; `t = spike_times[a:b] - begs[k]`.

iii. The notes cite the methods code’s alignment parameters and report a stimulus-locked PSTH step as a sanity check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The interval is divided into 100 nonoverlapping 20 ms bins. No smoothing or later temporal rebinning is applied.

ii. `BINSIZE = 0.02`; `NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))`.

iii. The agent copied the 20 ms, 100-bin configuration from the methods reference.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the configured stimulus-aligned window/bin grid and `stimOn_times`; every trial receives the same relative-time vector, running from −0.48 to 1.50 s at bin right edges.

ii. `BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)`; `stim_on = trials[PARAMS['align_time']]`.

iii. The agent says the right-edge convention matches its reading of `get_behavior_per_interval`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A 100-element evenly spaced right-edge grid is constructed once, cast to float32, and reused unchanged for every trial.

ii. `time_row = BIN_TIMES.astype(np.float32)`; `inputs.append(np.stack([time_row, ...]))`.

iii. The notes present this as the explicit temporal coordinate of each neural bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Input element `i` is the right edge of neural spike-count bin `i`; thus it labels each interval by its end rather than its center.

ii. `bi = (t / BINSIZE).astype(np.int64)` and `BIN_TIMES = np.linspace(WIN[0] + BINSIZE, WIN[1], NBINS)`.

iii. The agent’s “alignment proof” says spike bin and behavior sample share the same ending instant and therefore have no offset.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full trials table’s `probabilityLeft` sequence; every change starts a new block.

ii. `p = np.asarray(prob_left, dtype=float)`; `new_block[1:] = p[1:] != p[:-1]`.

iii. Since the table has no explicit block ID, the agent infers blocks from the constant block prior.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code computes a zero-based distance from the most recent block start before applying the trial mask, then broadcasts the retained trial’s value over all 100 bins.

ii. `starts = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))`; `return np.arange(len(p)) - starts`; `np.full(NBINS, tib[k], dtype=np.float32)`.

iii. Computing it before filtering preserves the animal’s true position even if intervening trials are excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials['choice']`.

ii. `choice_raw = trials['choice'].to_numpy()[idx]`.

iii. The notes cite the IBL convention: +1 is left, −1 is right, and zero is no response.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No-response trials are filtered; +1 becomes class 0 (left), −1 becomes class 1 (right), and the class is broadcast over time.

ii. `choice = (choice_raw < 0).astype(np.int64)`; `np.full(NBINS, choice[k], dtype=np.int64)`.

iii. This directly implements the requested binary mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials['probabilityLeft']`.

ii. `pl = trials['probabilityLeft'].to_numpy()`; `pleft = pl[idx]`.

iii. The notes identify this as the task’s block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2, then broadcast over time.

ii. `prior = np.select([np.isclose(pleft, 0.2), ...], [0, 1, 2]).astype(np.int64)`.

iii. This is the exact mapping specified by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses the absolute value of the velocity produced by `SessionLoader` from wheel timestamps and positions.

ii. `sl.load_wheel()`; `wheel_t = sl.wheel['times'].to_numpy()`; `wheel_v = np.abs(sl.wheel['velocity'].to_numpy())`.

iii. The agent says this matches reference `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. SessionLoader’s velocity is made nonnegative, finite samples are sorted, one session-wide linear interpolator evaluates all trial grids, and the resulting values are discretized per session.

ii. `f = interp1d(tt, tv, kind='linear', fill_value='extrapolate', ...)`; `vals = f(grid)`; `wheel_d, wheel_thr = discretize_terciles(wheel_k)`.

iii. One interpolator per session was chosen as a vectorized equivalent of the slower reference per-trial interpolation.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained trial-by-time samples in a session are split at that session’s 1/3 and 2/3 quantiles into low/medium/high. If thresholds tie, a rank-based fallback forces three classes.

ii. `q1, q2 = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])`; `np.digitize(values, [q1, q2])`; fallback: `d = (ranks * 3 // max(len(flat), 1))`.

iii. The agent argues per-session thresholds balance classes and avoid letting uncalibrated, camera/session-dependent scale encode session identity.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at stimulus onset plus each neural bin’s right edge.

ii. `grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]`; `vals = f(grid)`.

iii. The agent says each behavior sample and spike bin end at the same instant.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and timestamps from the left camera when available, otherwise the right camera.

ii. `sl.load_motion_energy(views=[view])`; `me_t = sl.motion_energy[key]['times'].to_numpy()`; `me_v = ...['whiskerMotionEnergy'].to_numpy()`.

iii. The notes say this follows the reference left-camera source while adding a right-camera fallback for coverage.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Finite samples are sorted, linearly interpolated to all trial grids with one session-level interpolator, and discretized per session; there is no additional normalization/filtering.

ii. `me_vals, me_ok = interpolate_behavior(me_t, me_v, begs_all, ends_all)`; `me_d, me_thr = discretize_terciles(me_k)`.

iii. The agent says the released trace is used directly and interpolation reproduces the reference behavior loader more efficiently.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses per-session 1/3 and 2/3 quantiles over all retained trial-time samples, with the same rank fallback for tied thresholds.

ii. `me_d, me_thr = discretize_terciles(me_k)` and `return np.digitize(values, [q1, q2]).astype(np.int64)`.

iii. Per-session terciles were chosen because camera motion-energy units are uncalibrated and camera-dependent, while balanced categorical outputs are required.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is interpolated at stimulus onset plus each neural bin’s right edge.

ii. `grid = np.nan_to_num(begs)[:, None] + (BIN_TIMES - WIN[0])[None, :]`; `vals = f(grid)`.

iii. The agent uses the same right-edge alignment proof as for wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid/NaN trial fields are masked; nonfinite behavior samples are removed; missing motion energy, spikes, eligible regions, or too few trials causes session skipping; incomplete behavior/ephys windows and all-zero neural trials are removed. Per-session processing catches exceptions and records a skip reason.

ii. `finite = np.isfinite(tt) & np.isfinite(tv)`; `if me_t is None: return None, ...`; `except Exception as exc: info['skip'] = 'error: %s' % exc`.

iii. The notes frame dropping unusable observations as safer than fabricating missing measurements, and report skip counts/reasons for auditability.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and merging large spike-sorting objects dominates per-session work; full conversion also spends substantial time binning/storing 11.3 GB and serializing it. Sessions are parallelized across workers.

ii. `spikes, clusters, channels = ssl.load_spike_sorting()`; `with ctx.Pool(processes=args.n_workers) as pool:`; `pickle.dump(data, f, protocol=4)`.

iii. The notes benchmark 4–7 s/session and explain that session-level multiprocessing reduced the full run to about three minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning still loops over trials, as does final construction of per-trial neural/input/output lists. The latter mostly packages required ragged session structures; spike binning could potentially assign trial offsets globally where intervals permit.

ii. `for k in range(n_trials): ... np.bincount(...)`; `for k in range(n_trials): neural.append(...); inputs.append(...); outputs.append(...)`.

iii. The agent emphasizes that it already removed reference per-trial process pools, vectorized interpolation over all trials, and uses `np.bincount` rather than looping over spikes.

## 10-c. What processing does the code repeat multiple times?

i. Region membership is counted and applied in a second pass after all sessions; output/input arrays are stacked again for summaries; with `--show-processing`, wheel and motion energy are reloaded after session processing for plots.

ii. `for s in sessions: for r in set(s['regions'].tolist()):`; `allout = np.concatenate([np.stack(s['output']) ...])`; `if args.show_processing: ... sl.load_wheel()`.

iii. The cross-session region pass is required by the chosen two-session rule; summary and plotting repetition is retained for validation rather than conversion semantics.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It stores raw wheel, motion-energy, and stimulus-onset arrays temporarily only for optional plotting, collects detailed `info` dictionaries/tracebacks, computes large concatenated input/output summaries, and optionally reloads raw behavior to plot it; none enters the final decoder arrays. The temporary `raw` field is removed before pickling.

ii. `session = {..., 'raw': {'wheel_vals': wheel_k, 'me_vals': me_k, 'stim_on': stim_on[idx]}}`; `s.pop('raw', None)`; `allout = np.concatenate(...)`.

iii. These operations support sanity checks, diagnostics, progress reporting, and documentation; the agent accepted their cost because they validate an expensive conversion.
