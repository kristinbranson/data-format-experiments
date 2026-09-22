# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local experiment-table CSV, intersects it with locally present NWB files, then restricts the data to active-behavior, Familiar sessions. It loads each selected experiment through `VisualBehaviorOphysProjectCache.from_local_cache`; thus it deliberately does not load all VisualBehavior data.

ii. `have = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in glob.glob(..., '*.nwb')]`; `sel = tbl[(tbl.behavior_type == 'active_behavior') & (tbl.experience_level == 'Familiar')]`; `ds = cache.get_behavior_ophys_experiment(int(oeid))`

iii. The trajectory says passive sessions lack choices/rewards/outcomes and the paper restricted neural analyses to familiar stimuli; local-file discovery avoids requesting unavailable data.

## 1-b. How are the data split into subjects?

i. Subjects are unique mouse IDs from retained session results; each session receives the index of its mouse in the sorted subject list.

ii. `subjects = sorted({r['info']['mouse_id'] for r in results})`; `subject_idx = np.array([subjects.index(r['info']['mouse_id']) for r in results], dtype=np.int64)`

iii. The mouse ID is treated as the animal identifier; sorting makes indexing deterministic.

## 1-c. How are the data split into sessions?

i. Experiments are grouped by `ophys_session_id`; all simultaneously recorded planes in that session are merged, and failed sessions are omitted.

ii. `return sel.groupby('ophys_session_id').ophys_experiment_id.apply(list).to_dict()`; `neural_trial = np.vstack(neural_parts)`

iii. The trajectory explicitly states that a converted session corresponds to an ophys session and Mesoscope planes are simultaneous, so they are merged.

## 1-d. How are the data split into trials?

i. SDK trial-table rows marked go or catch are used. Each trial consists of complete 250-ms bins inside its `start_time`–`stop_time` interval, with bin edges anchored to `change_time`.

ii. `tr = tr[(tr.go | tr.catch) & tr.change_time.notna()]`; `edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)`

iii. The AI chose SDK trial definitions and variable-length trial coverage while making all streams commensurate on a change-locked grid.

## 1-e. How are trials filtered based on quality controls?

i. Go/catch selection excludes aborted and auto-rewarded trials. Trials are also dropped for ambiguous outcomes, incomplete coverage, empty bins in any stream, or too few edges; sessions with fewer than two retained trials are dropped.

ii. `if len(oc) != 1: continue`; `if len(edges) < 3 or edges[0] < tmin or edges[-1] > tmax: continue`; `if np.any(nfr < 1): ...`; `if len(trials) < 2: return None`

iii. The trajectory cites the explicit go/catch requirement, complete cross-stream coverage, and the decoder's two-trial minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the SDK `events` table, using the selected `events` or optional `filtered_events` column; default is detected calcium events.

ii. `ev = ds.events`; `act = np.vstack(ev[neural_key].values)`; `ap.add_argument('--neural', default='events', choices=['events', 'filtered_events'])`

iii. The AI says detected events are regressed from dF/F by AllenSDK and were used in the paper.

## 2-b. How is the `neural` data processed?

i. Per-neuron events are cumulatively summed, summed into 250-ms bins for each plane, and planes are vertically stacked. Output is float32.

ii. `'cum': np.concatenate([np.zeros((act.shape[0], 1)), np.cumsum(act, axis=1)], axis=1)`; `vals, nfr = bin_sum(p['cum'], p['ts'], edges)`; `np.vstack(neural_parts).astype(np.float32)`

iii. Fixed stimulus-duration bins standardize single-plane and Mesoscope frame rates and lock activity to the stimulus cycle.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra cell-level filter is applied beyond released SDK ROIs and nonempty events tables. A plane with no events rows is skipped; a trial is rejected if any neural bin lacks a frame.

ii. `if len(ev) == 0: continue`; `if np.any(nfr < 1): ok = False`; all remaining rows are retained.

iii. The metadata/trajectory says released ROIs already passed Allen cell-segmentation and ROI QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to real or sham image-change time by putting a 250-ms bin edge exactly at `change_time`; native ophys timestamps determine membership.

ii. `k0 = int(np.ceil((trial.start_time - trial.change_time) / BIN_SIZE))`; `edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)`

iii. The trajectory emphasizes exact change locking and a common grid for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 250 ms. Native event samples are rebinned by summation.

ii. `BIN_SIZE = 0.25`; `'time_bin_size': BIN_SIZE * 1000.0`; `bin_sum(...)`

iii. One bin equals the 250-ms image flash and reconciles recordings with different ophys rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `stimulus_presentations.image_name`, `start_time`, `omitted`, and the change-detection stimulus block.

ii. `sp = sp[sp.stimulus_block_name == 'change_detection_behavior']`; `shown = sp[(~sp.omitted) & (sp.image_name != 'omitted')]`

iii. The AI used actual presentation records rather than only trial initial/change labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Sorted session image names define integer codes. At each bin center, the most recent non-omitted flash is selected, holding identity through gray periods and omissions.

ii. `image_names = sorted(set(sp.image_name.unique()) - {'omitted'})`; `j = np.searchsorted(pres_start, centers, side='right') - 1`; `image_trial = pres_img[j]`

iii. This follows the paper's 750-ms presentation-interval convention and makes Familiar image-set labels common across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the centers of the same change-locked 250-ms bins used for neural activity.

ii. `centers = 0.5 * (edges[:-1] + edges[1:])`; `j = np.searchsorted(pres_start, centers, side='right') - 1`

iii. A shared grid gives one identity value per neural bin.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from `stimulus_presentations.is_change` and presentation `start_time` within the change-detection block.

ii. `change_start = pres_start[shown.is_change.values]`

iii. The AI uses actual change presentations; catch sham changes consequently remain zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each bin center, the most recent real change is found and marked one if it occurred less than 750 ms earlier.

ii. `jc = np.searchsorted(change_start, centers, side='right') - 1`; `change_trial[good] = ((centers[good] - change_start[jc[good]]) < FLASH_INTERVAL)`

iii. The 750-ms interval represents the changed flash plus following gray period.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly binary: 1 within the 750-ms post-change interval and 0 otherwise; no learned threshold is used.

ii. `change_trial = np.zeros(len(centers), dtype=np.int64)`; comparison with `< FLASH_INTERVAL` produces 0/1.

iii. This implements “right after a change” and leaves catch trials at zero.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change status is computed at the center of every neural bin on the common grid.

ii. `centers = 0.5 * (edges[:-1] + edges[1:])`; the resulting `change_trial` is stacked beside outputs matching `neural_trial` columns.

iii. The shared grid guarantees matching lengths and timestamps.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ds.running_speed.speed` and `timestamps`.

ii. `run = ds.running_speed`; `run_t = run.timestamps.values`; `run_v = interp_nans(run.speed.values)`

iii. This is the SDK's standard wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Nonfinite samples are linearly interpolated, native samples are averaged within each 250-ms bin, then values are quantile-coded within each session.

ii. `run_cum = np.concatenate([[0.0], np.cumsum(run_v)])`; `run_trial = rsum / rn`; `run_code = quantile_bins(np.concatenate([t['run'] for t in trials]))`

iii. The AI argues session-wise bins accommodate strong mouse/session differences while producing balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four within-session quantile cut points create five equal-percentile categories coded 0–4.

ii. `edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])`; `np.searchsorted(edges, values, side='right')`

iii. Five percentile bins are required; session-local scaling was judged meaningful.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Native running samples are averaged between the exact same bin edges used for neural event sums.

ii. `rsum, rn = bin_sum(run_cum, run_t, edges)`

iii. Hardware-synchronized native timestamps and common edges provide alignment without pre-interpolation to ophys frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is derived from `eye_tracking.pupil_area` and eye-camera timestamps, converted to equivalent circular diameter.

ii. `pupil_area = interp_nans(eye.pupil_area.values)`; `pupil_v = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)`

iii. The AI interprets area in pixels geometrically and notes NaNs likely represent blinks/lost tracking.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Nonfinite area values are interpolated, area becomes diameter, native samples are averaged per 250-ms bin, and means are quantile-coded within session.

ii. `pupil_cum = np.concatenate([[0.0], np.cumsum(pupil_v)])`; `pupil_trial = psum / pn`; `pupil_code = quantile_bins(...)`

iii. Session-wise quantiles account for camera-pixel scale and between-session pupil differences.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four within-session quantile thresholds create five categories coded 0–4.

ii. `pupil_code = quantile_bins(np.concatenate([t['pupil'] for t in trials]))`

iii. This fulfills the requested five equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye samples are averaged between the same bin edges as neural data; trials without samples in every pupil bin are rejected.

ii. `psum, pn = bin_sum(pupil_cum, eye_t, edges)`; `if np.any(pn < 1): continue`

iii. Native synchronized timestamps avoid assuming equal camera and ophys rates.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the four boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. `oc = [k for k in OUTCOME_NAMES if bool(trial[k])]`

iii. These are the SDK's canonical, mutually exclusive go/catch outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trials without exactly one outcome are removed; the outcome's fixed list index is repeated across every time bin.

ii. `if len(oc) != 1: continue`; `'outcome': OUTCOME_NAMES.index(oc[0])`; `np.full(T, t['outcome'], dtype=np.int64)`

iii. Repetition makes the static trial label compatible with the output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NaNs in running/pupil are linearly interpolated; all-missing behavior, absent eye tracking, empty event planes, uncovered/empty bins, malformed outcomes, and sessions with under two trials are skipped. Worker exceptions drop the session with a message.

ii. `if bad.all(): return None`; `x[bad] = np.interp(...)`; `if eye is None or len(eye) == 0: return None`; `except Exception as exc: ... return None`

iii. The AI prioritized a complete five-output dataset and preventing one bad session from terminating conversion.

## 9-a. What are the most time-consuming steps of the code?

i. NWB/SDK experiment loading, cumulative processing of full neural matrices, and per-session conversion are the expensive steps; multiprocessing across sessions is used.

ii. `with Pool(args.workers) as pool: results = pool.map(_worker, items)`; each worker calls `get_behavior_ophys_experiment` and `np.cumsum(act, axis=1)`.

iii. The trajectory repeatedly timed full conversions and selected 12 workers, indicating session I/O/conversion dominated.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial `iterrows()` loop and per-plane binning loop could be batched/vectorized; image-name coding also uses repeated list lookup.

ii. `for _, trial in tr.iterrows():`; `for p in planes:`; `pres_img = np.array([image_names.index(n) for n in shown.image_name.values])`

iii. No explicit trajectory justification addresses these loops; cumulative sums already optimize within-bin aggregation.

## 9-c. What processing does the code repeat multiple times?

i. Each worker creates a cache object; shared behavior/stimulus tables are accessed from the first plane after all planes are loaded. Trial-wise `searchsorted`/binning is repeated for every stream and trial, and quantile assembly concatenates trial values after extraction.

ii. `cache = VisualBehaviorOphysProjectCache.from_local_cache(...)` inside `convert_session`; repeated `bin_sum(..., edges)` calls for neural, running, and pupil.

iii. The AI accepts repetition to preserve each stream's native timestamp clock and enable parallel session isolation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes/stores extensive session metadata and cell IDs not consumed by decoder outputs; it also retains each loaded dataset object in `planes` although only metadata and traces are needed after extraction. Cumulative full-session arrays include portions outside retained trials.

ii. `'ds': ds` in every plane; `'cell_ids': [...]`; the large `session_info` dictionary is stored in metadata.

iii. The trajectory used metadata for sanity checks and provenance, but these fields are not needed for decoder training.
