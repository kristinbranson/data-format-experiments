# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, intersects it with NWB files actually present on disk, removes passive experiments, groups experiment IDs into sessions, and loads every plane directly with `BehaviorOphysExperiment.from_nwb_path`. Full runs use a multiprocessing pool.

ii. `et = pd.read_csv(META_DIR + 'ophys_experiment_table.csv')`; `et = et[et.ophys_experiment_id.isin(have)]`; `et = et[~et.passive]`; `BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)`

iii. The notes say direct local NWB loading is the same canonical SDK path without unnecessary S3 access; only 284 release files existed locally, passive sessions have no meaningful behavioral outcomes, and parallel session loading is the main speedup.

## 1-b. How are the data split into subjects?

i. Mouse identity comes from each session's SDK metadata; final subjects are the sorted unique mouse IDs and each session indexes that list.

ii. `mouse=str(md['mouse_id'])`; `subjects = sorted(set(r['mouse'] for r in results))`; `data['subject_idx'].append(subjects.index(r['mouse']))`

iii. The notes treat SDK `mouse_id` as the animal identifier and report 38 mice among locally available ophys files.

## 1-c. How are the data split into sessions?

i. A decoder session is one `ophys_session_id`; all experiments/imaging planes with that ID are grouped and their neurons concatenated.

ii. `sessions = [(int(sid), list(map(int, g.ophys_experiment_id.values))) for sid, g in et.groupby('ophys_session_id')]`; `neural = np.concatenate(neural_planes, axis=0)`

iii. The agent reasoned that an experiment is one plane whereas `ophys_session_id` denotes a continuous simultaneous recording; merging planes reconstructs the neural population.

## 1-d. How are the data split into trials?

i. Trials are SDK trial rows labeled go or catch. Each retained trial is represented by eight flash intervals: three before the real/sham change flash, the change flash, and four after it.

ii. `keep = (trials.go.values.astype(bool) | trials['catch'].values.astype(bool))`; `bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]`

iii. The notes identify `change_time` as defined for go and catch trials and choose a fixed flash-centered window to match the paper's 750 ms image-presentation analysis unit.

## 1-e. How are trials filtered based on quality controls?

i. Selecting go or catch excludes aborted and auto-rewarded rows. Trials are further removed if change time is invalid/not exactly a flash onset, the eight-bin window leaves the task block, or no canonical outcome exists. Sessions with fewer than two remaining trials are dropped; passive and unusable-pupil sessions are also dropped.

ii. `good &= np.isfinite(change_times)`; `good &= np.abs(flash_start[j_clipped] - change_times) < 1e-4`; `good &= (j - N_PRE >= 0) & (j + N_POST < len(flash_start))`; `good &= oc >= 0`

iii. The agent cites the explicit go/catch requirement, exact flash alignment, complete windows, valid outcomes, and decoder minimum-trial rule. Passive sessions lack behavioral outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the SDK `events.events` detected calcium-event magnitude for every valid ROI in every plane.

ii. `ev = np.vstack(ds.events.events.values).astype(np.float64)`

iii. The agent quotes the paper's use of “detected calcium events” and therefore chooses events instead of dF/F or additionally smoothed `filtered_events`.

## 2-b. How is the `neural` data processed?

i. Event magnitudes are summed in each of the eight 750 ms flash windows using cumulative sums, reshaped neuron × trial × bin, concatenated across planes, and cast to float32.

ii. `s, _ = binned_sum(ev, ts, flat_t0, flat_t1)`; `neural_planes.append(s.reshape(ev.shape[0], n_trials, N_BINS))`; `neural = np.concatenate(neural_planes, axis=0).astype(np.float32)`

iii. Summing integrates sparse event magnitude without arbitrary smoothing and cumulative sums avoid repeatedly scanning frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit activity filter. Loading via `BehaviorOphysExperiment.from_nwb_path` uses its default `exclude_invalid_rois=True`, so only SDK-valid ROIs remain.

ii. `BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)`

iii. The notes say the Allen ROI pipeline already removes invalid unions, duplicates, border/dendritic, small, or dim ROIs; returned cells were verified `valid_roi=True`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `trials.change_time`, a real image change for go trials and sham change for catch trials; bin index 3 begins at that flash.

ii. `change_times = trials.change_time.values.astype(float)`; `bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]`

iii. The agent argues that this is the event the mouse reports and verified that every retained change time coincides with a stimulus flash onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 750 ms. Native ophys frames are rebinned by summing events within eight flash intervals.

ii. `BIN_SIZE = 0.75`; `bt1 = bt0 + BIN_SIZE`; `time_bin_size=BIN_SIZE * 1000.0`

iii. The paper defines an image-presentation interval as a 250 ms image plus 500 ms gray period, motivating the 750 ms bins.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It comes from change-detection-block `stimulus_presentations.image_name` and `omitted`.

ii. `flash_image = sp.image_name.values.astype(object)`; `flash_image = np.where(flash_omitted, 'omitted', flash_image)`

iii. The agent uses the presentation table as the authoritative per-flash record and preserves omissions rather than fabricating an image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The eight selected flash names are gathered, a global sorted vocabulary is built with `omitted` last, and strings are mapped to integer categories.

ii. `img = flash_image[bin_flash]`; `images = [v for v in images if v != 'omitted'] + ['omitted']`; `img_idx = np.vectorize(img_map.get)(r['image']).astype(np.int64)`

iii. This retains actual omissions and ensures a consistent categorical mapping across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image labels and neural sums use the same `bin_flash`/750 ms windows, so each label describes its neural bin.

ii. `bt0 = flash_start[bin_flash]`; `img = flash_image[bin_flash]`

iii. Assertions check the change-bin image against `trials.change_image_name` and the preceding image against the initial image.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from `stimulus_presentations.is_change` for the selected task flashes.

ii. `flash_ischange = sp.is_change.values.astype(bool)`; `chg = flash_ischange[bin_flash].astype(np.int64)`

iii. The notes emphasize that `is_change` marks real changes but correctly stays false for catch/sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The boolean flag is indexed at the eight trial flashes and converted to int64; no smoothing is applied.

ii. `chg = flash_ischange[bin_flash].astype(np.int64)`

iii. The agent corrected an early mistaken assertion after confirming catch trials should remain zero throughout.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already boolean, so False/True become categories 0/1 (`no_change`, `change`) with no numeric threshold.

ii. `output_values=[..., ['no_change', 'change'], ...]`

iii. This directly implements the binary output specification.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The flag is taken for the same flash bins used for neural event sums; a real change is one at aligned bin 3, catch trials are all zero.

ii. `assert np.all(chg[is_go, N_PRE] == 1)`; `assert np.all(chg[~is_go, N_PRE] == 0)`

iii. Full-data checks found every real-change flag at bin 3 and no catch flags.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It derives from SDK `running_speed.timestamps` and `running_speed.speed`.

ii. `run_t = run.timestamps.values.astype(float)`; `run_v = run.speed.values.astype(float)`

iii. The SDK stream is the standard wheel-derived speed in cm/s on the common session clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Timestamps are sorted, missing values interpolated, samples averaged within each 750 ms neural bin, then values are quantile-classified within session.

ii. `run_v = interpolate_nans(run_v)`; `run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)`; `run_cls, run_edges = quantile_bin(run_binned.ravel())`

iii. Bin means align behavior with neural bins; session-relative percentiles account for large differences in locomotor propensity and balance decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four within-session 20/40/60/80% quantiles divide speed into five integer classes.

ii. `edges = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])`; `np.searchsorted(edges, x, side='right')`

iii. The agent says equal-percentile bins give comparable relative levels and approximately equal class counts; tied edges may collapse a class.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed samples are averaged over the exact same absolute `[bt0, bt1)` flash windows used to sum neural events.

ii. `binned_mean_1d(run_v, run_t, flat_t0, flat_t1)`

iii. All NWB streams share the session clock; identical window boundaries provide temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It derives from `eye_tracking.pupil_width` and eye timestamps.

ii. `pupil = interpolate_nans(eye.pupil_width.values)`; `eye_t = eye.timestamps.values.astype(float)`

iii. Width is a diameter-like measure, unlike ellipse area; the notes report strong agreement with area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaNs/blinks are linearly interpolated (nearest finite value at edges), timestamps sorted, pupil width averaged in each 750 ms bin, and the binned values quantile-classified within session. Sessions with empty/all-NaN eye data are dropped.

ii. `pupil = interpolate_nans(eye.pupil_width.values)`; `pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)`; `pup_cls, pup_edges = quantile_bin(pup_binned.ravel())`

iii. Interpolation bridges blink gaps while fixed-window averaging aligns pupil with neural bins; session percentiles normalize camera/session scale.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four within-session percentile edges split all binned pupil values into five categories.

ii. `pup_cls, pup_edges = quantile_bin(pup_binned.ravel())`

iii. The stated aim is five balanced relative pupil-level classes despite camera-pixel scale differences.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil samples are averaged over the exact `[bt0, bt1)` flash windows used for neural activity.

ii. `binned_mean_1d(pupil, eye_t, flat_t0, flat_t1)`

iii. The common NWB session clock and shared window boundaries guarantee bin-level alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. `for k, name in enumerate(OUTCOMES): oc[trials[name].values.astype(bool)] = k`

iii. These are the SDK's four mutually exclusive canonical outcomes for go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Flags map to fixed codes 0–3; invalid outcomes are filtered, and the static code is repeated over all eight time bins in final output.

ii. `good &= oc >= 0`; `np.full(N_BINS, r['outcome'][t], dtype=np.int64)`

iii. Repetition satisfies the decoder's common time-varying output matrix format while preserving a static per-trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NaNs in running/pupil are interpolated; empty behavioral bins fall back to center interpolation. Empty/all-NaN eye sessions are dropped. Invalid change times/windows/outcomes are dropped, timestamps are sorted, omitted images are explicit, and consistency assertions reject misalignment.

ii. `if bad.all(): return None`; `out[~ok] = np.interp(centres, timestamps, values)`; `good &= np.isfinite(change_times)`

iii. The notes document three sessions dropped for missing eye tracking and extensive raw-data recomputation/assertions rather than silently inventing labels.

## 9-a. What are the most time-consuming steps of the code?

i. Loading large NWBs is dominant; neural event traversal/binning is the other substantial operation. Independent sessions are processed with up to 24 workers.

ii. `with Pool(min(args.workers, len(jobs))) as pool: results = pool.map(process_session, jobs)`

iii. The notes measured roughly 2.6 seconds per plane and report a 51-second full conversion with multiprocessing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorizes trial/bin lookup, behavior binning, and neural accumulation. Remaining per-plane loading/neural loops and final per-trial packaging are difficult or low-value to vectorize.

ii. `bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]`; `csum = np.concatenate([..., np.cumsum(values, axis=1)], axis=1)`

iii. The notes explicitly reject rescanning every bin and replace it with cumulative sums plus `searchsorted`.

## 9-c. What processing does the code repeat multiple times?

i. Each plane is loaded once. Behavioral streams are read only from the first plane. The same binning helpers are called separately for neural, running, and pupil because they are distinct streams; final output is assembled per trial.

ii. `ref = planes[0][1]`; `for eid, ds in planes:`; `binned_mean_1d(run_v, ...)`; `binned_mean_1d(pupil, ...)`

iii. The agent specifically designed against reloading NWBs or rescanning a stream once per bin.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Continuous binned running/pupil values, percentile edges, timings, and several diagnostic fields are retained only in intermediate session results and discarded from the final pickle; they support quantization, checks, logging, and optional plots. Plotting is disabled unless requested.

ii. `run_cont=run_binned, pupil_cont=pup_binned, run_edges=run_edges, pupil_edges=pup_edges, ... timings=timings`; `if show: ... plot_processing(...)`

iii. The notes treat these as validation/provenance intermediates, not downstream decoder features; normal conversion avoids plot generation.
