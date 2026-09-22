# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM release manifest, optionally restricts it with `DATALIMIT_SUBSET.csv`, and processes every unique manifest `eid`. All actual trials, wheel, camera, spikes, clusters, and channels are loaded through ONE/brainbox, with probes specified by the release manifest. Sessions are parallelized.

ii. `bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)`; `eids = list(dict.fromkeys(bwm.eid.tolist()))`; `sl = SessionLoader(one=one, eid=eid)`; `sl.load_trials()`; `ssl.load_spike_sorting()`.

iii. The notes say the manifest is the authoritative 699-insertion/459-session release list and that bare `ONE()` is needed to resolve revisioned cache files through cached Alyx metadata without directly reading the data files.

## 1-b. How are the data split into subjects?

i. Subject names come from `bwm_release.csv`. After session filtering, unique subject strings are sorted and each retained session receives an index into that list.

ii. `subj_of = dict(zip(bwm.eid, bwm.subject))`; `subjects = sorted({subj_of[e] for e in kept})`; `subject_idx = np.array([subject_index[subj_of[e]] for e in kept])`.

iii. The manifest already supplies stable subject identifiers, so no path parsing is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` is one session. Probe rows sharing an `eid` are grouped and merged into that session; results are restored to manifest order after parallel processing.

ii. `by_eid = {e: list(g[['pid', 'probe_name']].itertuples(...)) for e, g in bwm.groupby('eid')}`; `results[r['eid']] = r`; `kept = [e for e in eids if results[e].get('skip') is None]`.

iii. The notes treat the ONE `eid` as the native session identifier and session as the natural parallelization unit.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies one table row per trial. Each retained row becomes one neural, input, and output array, indexed by `kt`.

ii. `kt = np.flatnonzero(keep_trial)`; `for j, k in enumerate(kt): neural.append(binned[j]); inputs.append(inp); outputs.append(out)`.

iii. The agent states that the trial table and stimulus-aligned windows define the split directly.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed for RT outside 0.08–2 s, duration over 10 s, missing required events, no-choice, invalid prior, inadequate wheel/camera coverage, inadequate spike coverage, or if fewer than two usable trials remain. Unbiased trials are retained.

ii. `query += ' | (choice == 0)'`; `keep_trial = trial_mask & ws_ok & wm_ok`; `trial_mask &= spk_ok`; `keep_trial = keep_trial & (prior_cls >= 0)`.

iii. The notes attribute the event/RT/no-choice/duration mask to reference `load_trials_and_mask`, behavior coverage to `align_spike_behavior`, and add spike coverage after investigating all-zero trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each probe's `spikes.times` and `spikes.clusters`; merged cluster/channel data provide quality labels and anatomical acronyms used for curation and region indices.

ii. `spikes, clusters, channels = ssl.load_spike_sorting()`; `cdf = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()`.

iii. The notes identify these as the same spike-sorting sources used by the reference pipeline.

## 2-b. How is the `neural` data processed?

i. Probes are merged by offsetting cluster IDs, spikes are time-sorted, retained units are remapped contiguously, and spikes are counted (not divided by bin width) into 100 20-ms bins per trial. Arrays are float32.

ii. `spk_clusters.append(... + offset)`; `np.add.at(flat[k], ci * NBINS + bi, 1.0)`; `binned = bin_spikes_trials(...)`.

iii. The notes explicitly choose unnormalized spike counts because the methods code caches raw counts and say the implementation is equivalent to `bincount2D`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `label >= 1` and a Beryl acronym other than `root` or `void`. A final session must contain at least five such units.

ii. `keep_unit = (labels >= QC_LABEL) & ~np.isin(beryl, NON_GREY)`; `if results[e]['n_neurons'] < MIN_NEURONS_PER_SESSION: ...`.

iii. The agent resolves disagreement between the methods code (all clusters) and data paper in favor of well-isolated grey-matter neurons, citing exact reproduction of 75,708 good units, meaningful regions, memory, and signal quality. The five-unit rule is adapted from a per-region paper criterion to pooled sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial begins at `stimOn_times - 0.5` and spans through `stimOn_times + 1.5`; absolute spike times are located within those windows and binned relative to the window start.

ii. `align = trials[ALIGN_TIME].to_numpy(...)`; `t_beg = align + TIME_WINDOW[0]`; `bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)`.

iii. The agent says all streams share the synchronized session clock and the chosen window matches the decoder instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, with 100 bins across two seconds. Spikes are newly binned from event times; no smoothing or later rebinning is applied.

ii. `BINSIZE = 0.02`; `NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))`.

iii. The notes choose 20 ms because the released reference code and model description use T=100, despite a paper-text discrepancy for some static targets.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the stimulus alignment field `trials.stimOn_times` and the fixed window/bin grid, not from a separate recorded signal.

ii. `ALIGN_TIME = 'stimOn_times'`; `time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1))`.

iii. The notes say this grid directly represents time relative to the required alignment event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs 100 bin-right-edge values from -0.48 through +1.50 s and copies the same row into every trial.

ii. `inp[0] = time_axis`.

iii. It justifies right edges as matching reference behavioral interpolation via `linspace(beg + binsize, end, n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Input element i labels the right edge of neural spike-count bin i; both use the same stimulus-relative two-second grid.

ii. `inp[0] = time_axis`; neural bin i covers `[t_beg + i*BINSIZE, t_beg + (i+1)*BINSIZE)`.

iii. The notes describe the right edge as the representative time used consistently for behavioral streams and the time input.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive runs of `trials.probabilityLeft`; a change starts a new block.

ii. `tib = trial_number_in_block(p_left).astype(np.float32) / TRIAL_IN_BLOCK_SCALE`.

iii. The agent notes that no explicit block identifier exists, while probabilityLeft is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A vectorized zero-based run-length index is computed on all raw trials before filtering, divided by 100, then broadcast across all 100 time bins.

ii. `block_start = np.maximum.accumulate(np.where(is_new, np.arange(n), 0))`; `return np.arange(n) - block_start`; `inp[1] = tib[k]`.

iii. Computing before filtering preserves the animal's true block position. Division by 100 was chosen to keep the feature O(1) beside neural PCs because the decoder does not standardize inputs.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`.

ii. `choice = trials['choice'].to_numpy(dtype=np.float64)`.

iii. The sign convention was empirically checked against stimulus side and feedback.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice +1 becomes left/class 0, -1 becomes right/class 1, and choice 0 trials are excluded. The scalar class is broadcast through time.

ii. `choice_cls = (choice < 0).astype(np.int64)`; `out[0] = choice_cls[k]`.

iii. This is the task-required left=0/right=1 coding and the documented IBL sign convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii. `p_left = trials['probabilityLeft'].to_numpy(dtype=np.float64)`.

iii. The notes confirm the raw field only takes 0.2, 0.5, and 0.8.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2; invalid values are removed, and the class is broadcast through time.

ii. `prior_cls[np.isclose(p_left, 0.2)] = 0`; similarly for 0.5/0.8; `out[1] = prior_cls[k]`.

iii. This exactly follows the decoder specification, including retaining the initial unbiased block.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is the absolute value of `SessionLoader`'s wheel velocity, which originates from wheel position and timestamps.

ii. `sl.load_wheel()`; `wheel_t = sl.wheel['times']...`; `wheel_v = np.abs(sl.wheel['velocity']...)`.

iii. The notes follow the reference definition of wheel speed as absolute velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` supplies processed velocity; the code takes its magnitude, removes nonfinite samples, linearly interpolates it to each trial's bin right edges, and discretizes it by session tertiles.

ii. `ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)`; `ws_cls, ws_thr = tertile_bins(ws_vals, keep_trial)`.

iii. The interpolation grid matches the methods code; session-relative thresholds address large between-session scale differences and produce balanced categorical outputs.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the 33.33rd and 66.67th percentiles pooled over every time bin of retained trials in that session. Values strictly above each threshold increment the class; degenerate traces use unique-value fallbacks.

ii. `lo, hi = np.percentile(pool, [100.0 / 3.0, 200.0 / 3.0])`; `cls = (values > lo).astype(...) + (values > hi).astype(...)`.

iii. The agent says per-session tertiles satisfy the categorical-output requirement while mirroring reference per-session normalization.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at absolute times `stimOn - 0.5 + 0.02*(1..100)`, corresponding to the right edges of the 100 neural bins.

ii. `xi = t_beg[:, None] + grid[None, :]`; `vals = np.interp(safe, times, values)`.

iii. The notes say shared synchronized clocks plus a common trial grid provide alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It comes from `whiskerMotionEnergy` and `times` in left-camera motion energy, falling back to the right camera.

ii. `sl.load_motion_energy(views=[view])`; `wv = me['whiskerMotionEnergy'].to_numpy(...)`.

iii. The left-then-right fallback is described as matching the reference and retaining sessions with either available side camera.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Nonfinite raw samples are dropped, the released trace is otherwise unfiltered, it is linearly interpolated to bin right edges, and then discretized into session tertiles.

ii. `good = np.isfinite(wt) & np.isfinite(wv)`; `wm_vals, wm_ok = interp_behavior(...)`; `wm_cls, wm_thr = tertile_bins(...)`.

iii. The agent says the released ROI trace needs no extra normalization/filtering and uses the same categorical strategy as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same retained-trial pooled 33.33/66.67 session percentiles and strict-greater-than classification as wheel speed.

ii. `wm_cls, wm_thr = tertile_bins(wm_vals, keep_trial)`.

iii. The rationale is balanced classes within sessions despite camera-dependent absolute scales.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is interpolated on the same stimulus-relative right-edge times used for wheel behavior and corresponding neural bins.

ii. `wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)`.

iii. The camera timestamps and spikes share the synchronized session clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite samples are removed; invalid trials are masked; left camera falls back to right; unavailable probes are skipped; sessions missing trials/spikes/wheel/camera, with fewer than two trials, or fewer than five curated neurons are skipped and reported. Worker exceptions become explicit skip records.

ii. `if whisker_t is None: return {'skip': 'no whisker motion energy'}`; `return {'eid': eid, 'skip': f'EXCEPTION ...'}`.

iii. The notes document debugging read-only masks and all-zero neural trials and favor explicit dropping/reporting over imputation.

## 10-a. What are the most time-consuming steps of the code?

i. Spike-sorting I/O (roughly 2–6 s/session) dominates, followed by behavior loading and spike binning; full conversion is parallelized by session.

ii. `spikes, clusters, channels = ssl.load_spike_sorting()`; `with ProcessPoolExecutor(max_workers=args.n_workers)`.

iii. The timing notes estimate 4.3–7.8 s total per session and identify disk loading as the main cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-trial spike loop could potentially be further vectorized; assembly also loops over retained trials. The agent already vectorized trial-number computation and behavior interpolation, and replaced reference per-trial process pools.

ii. `for k in range(n_trials): ... np.add.at(flat[k], ...)`; `for j, k in enumerate(kt): ...`.

iii. The notes claim the new design reduces Python calls substantially, though `bin_spikes_trials` still performs one `np.add.at` call per trial.

## 10-c. What processing does the code repeat multiple times?

i. ONE and atlas objects are created once per session worker call; every session separately loads trials, probes, wheel, and one or more camera views. Per-trial assembly repeatedly allocates input/output arrays and broadcasts static values.

ii. `one = get_one(); br = BrainRegions()` inside `load_session`; `for j, k in enumerate(kt): inp = np.empty(...); out = np.empty(...)`.

iii. The notes emphasize avoiding repeated spike/behavior loading within a session, but do not identify these remaining repetitions as material bottlenecks.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges metadata for all clusters before filtering, interpolates behavior for all trials before applying the final combined mask, computes class arrays for nonretained trials, and records extensive metadata/timings not used by decoder training. It also loads diagnostic-only data only when requested.

ii. `ws_vals, ws_ok = interp_behavior(..., t_beg)` before `keep_trial`; `ws_cls, ws_thr = tertile_bins(ws_vals, keep_trial)` returns classes for the full array; `session_info.append({...})`.

iii. The agent notes that spikes are binned only for kept trials, but does not claim all preliminary behavior/metadata work is required downstream; diagnostics and provenance were retained for validation.
