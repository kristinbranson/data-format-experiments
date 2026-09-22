# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes one NWB file per session with `pynwb.NWBHDF5IO`. It uses a multiprocessing pool by default and assembles successful session results.

ii. `files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))`; `with NWBHDF5IO(path, 'r', load_namespaces=True) as io: nwb = io.read()`; `pool.imap(_worker, ...)`.

iii. The notes say the release contains 174 files and one file per session; sorting is deterministic, `pynwb` is required, and parallel session processing reduces wall time.

## 1-b. How are the data split into subjects?

i. Each session's subject is `nwb.subject.description`, falling back to `subject_id`; unique values are sorted, and `subject_idx` maps sessions to them.

ii. `subject = nwb.subject.description or nwb.subject.subject_id`; `subjects = sorted({s['subject'] for s in sessions})`; `subject_idx = np.array([subj_idx[s['subject']] for s in sessions], dtype=np.int64)`.

iii. The AI treats the descriptive mouse names (for example SC015) as the meaningful paper-facing identifiers and retains the numeric NWB `subject_id` in session metadata.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session, identified by `nwb.identifier`; output order follows the sorted input paths. Sessions with no units or no classifier-good units are skipped.

ii. `session_id = nwb.identifier`; `return {'session_id': session_id, 'subject': subject, 'neural': neural_trials, ...}`.

iii. The AI states that the NWB/file boundary is already the session boundary and keeps all released sessions with usable neural data (173 of 174).

## 1-d. How are the data split into trials?

i. Rows of `nwb.trials.to_dataframe()` define trials. The code asserts one `go_start_times` timestamp per row and uses filtered row indices consistently for all streams.

ii. `trials = nwb.trials.to_dataframe()`; `go_all = np.asarray(be['go_start_times'].timestamps[:])`; `assert len(go_all) == n_trials_all`; `trial_idx = np.where(keep)[0]`.

iii. The trials table is the authoritative boundary; matching event counts and later raw-data spot checks were used to validate indexing.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained if it matches a good unit's `obs_intervals`, is neither auto-water nor free-water, and has at least one population spike in the requested window. Photostim, early-lick, miss, and ignore trials remain. Sessions with fewer than two retained trials are skipped.

ii. `keep = ephys_covered.copy()`; `keep &= (auto_all == 0) & (free_all == 0)`; `has_spikes = rates.sum(axis=(0, 2)) > 0`; `rates = rates[:, has_spikes, :]`.

iii. The notes say uncovered trials contain no ephys; water trials decouple reward from action; required decoder classes must remain; and two zero-population-spike trials were dropped after verification exposed recording termination.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each classifier-good unit's `units['spike_times']`; `BehavioralEvents/go_start_times` supplies trial alignment.

ii. `st = np.asarray(units['spike_times'][int(i)], dtype=np.float64)`; `rates = bin_spike_rates(spike_lists, go)`.

iii. The notes identify sorted unit spike times as the raw neural representation and the go cue as the instructed alignment event.

## 2-b. How is the `neural` data processed?

i. Spike times are sorted if necessary, counted in non-overlapping bins using `searchsorted` and `diff`, divided by 0.05 s to form Hz, and stored as float32 without smoothing or normalization.

ii. `idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)`; `out[i] = np.diff(idx, axis=1)`; `out /= np.float32(BIN_SIZE)`.

iii. The AI says this is a vectorized equivalent of the reference `sliding_histogram(..., rate=True)` and explicitly chooses firing rates rather than counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `units.classification` equals `good` are used; there is no firing-rate threshold. Sessions with no good units are skipped.

ii. `classification = np.asarray(units['classification'][:])`; `good = np.where(classification == 'good')[0]`; `if len(good) == 0: return ...`.

iii. The notes identify this field as the Chen/Liu classifier verdict. They reject the method paper's 2-Hz cutoff as analysis-specific rather than dataset QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges are formed by adding fixed offsets from -2.5 to +1.5 s to each go-cue timestamp, then spikes are counted between those edges.

ii. `edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()`.

iii. All timestamps share the NWB session clock, so no clock correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins over four seconds. Raw spike events are binned once; no subsequent rebinning is applied.

ii. `BIN_SIZE = 0.05`; `N_BINS = int(round((T_PRE + T_POST) / BIN_SIZE))`; `BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE`.

iii. This directly implements the requested 50-ms resolution and requested window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` and each trial's go-cue timestamp, selecting the last sample onset at or before the go cue.

ii. `sample_t = np.asarray(be['sample_start_times'].timestamps[:])`; `si = np.searchsorted(sample_t, go, side='right') - 1`.

iii. Early licking can replay sample/delay epochs, so the AI regards the final pre-go sample onset as the effective instruction tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The go-minus-tone interval is added to every go-relative bin center. If a valid per-trial onset cannot be found, the session median interval (or 1.85 s) is imputed.

ii. `go_minus_tone = go - tone_onset`; `time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)`; `tone_onset[bad_tone] = go[bad_tone] - fallback`.

iii. The notes say this creates a true elapsed-since-instruction clock; the fallback is defensive and was not needed in the full data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 bin centers used for neural bins, with both grids anchored to the same trial go cue.

ii. `BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0`; `time_from_tone = BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]`.

iii. The AI visually checked that the input crosses zero at the raw tone onset.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses the absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times` event streams, plus go times/bin centers.

ii. `ps_start = np.asarray(be['photostim_start_times'].timestamps[:])`; `ps_stop = np.asarray(be['photostim_stop_times'].timestamps[:])`.

iii. The notes report one 0.5-s event per stimulated trial and agreement with trial photostim metadata.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Starts are sorted with their stops. Each absolute bin center is 1 when it lies in a corresponding half-open stimulation interval and 0 otherwise.

ii. `abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]`; `on = valid & (abs_centers >= ps_start[kk]) & (abs_centers < ps_stop[kk])`; `photostim[on] = 1.0`.

iii. A binary time series is required; bin-center classification reproduces the observed 0.5-s pulses.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals are tested at the same absolute go-anchored bin centers as the neural bins.

ii. `abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]`.

iii. Processing plots and spot checks showed the step function coinciding with raw laser events.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is reconstructed from trials-table `outcome` and `trial_instruction`.

ii. `outcome = outcome_all[trial_idx]`; `instr = instr_all[trial_idx]`; `choice_str = np.where(outcome == 'ignore', 'no lick', np.where(outcome == 'hit', instr, opposite))`.

iii. Hit means instructed side, miss means opposite side, and ignore means no lick; the notes report 99.7% agreement with independent lick-event reconstruction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The strings are encoded left=0, right=1, no lick=2 and repeated over all 80 time bins.

ii. `choice = np.select([choice_str == 'left', choice_str == 'right'], [0, 1], default=2)`; `np.full(N_BINS, choice[t], dtype=np.int64)`.

iii. Choice is trial-level, but repetition allows all outputs to share a time-varying array shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` column.

ii. `outcome_all = np.asarray(trials['outcome'].values, dtype=object).astype(str)`.

iii. The raw values already represent exactly the requested ignore/miss/hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Ignore, miss, and hit are encoded 0, 1, and 2 and repeated across 80 bins.

ii. `outcome_code = np.select([outcome == 'ignore', outcome == 'miss'], [0, 1], default=2)`; `np.full(N_BINS, outcome_code[t], dtype=np.int64)`.

iii. Fixed category codes and temporal repetition satisfy the decoder format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii. `early_all = np.asarray(trials['early_lick'].values, dtype=object).astype(str)`.

iii. The NWB trial table explicitly flags early licking.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` becomes 1 and all other expected values (`no early`) become 0; the value is repeated across bins.

ii. `early_code = (early == 'early').astype(np.int64)`; `np.full(N_BINS, early_code[t], dtype=np.int64)`.

iii. This implements the requested no/yes binary trial label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps, column 1 (y), and column 2 (DeepLabCut likelihood).

ii. `ttimes = np.asarray(tongue.timestamps[:])`; `tdata = np.asarray(tongue.data[:])`; `visible = tdata[:, 2] > TONGUE_LIKELIHOOD_THRESH`.

iii. The AI identified this as the dataset's side-camera tongue trace and uses likelihood to distinguish visible measurements.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Nonmonotonic timestamps are stably sorted. Frames with likelihood >0.9 are selected; their y values are averaged within each retained trial's 50-ms bins. Empty bins become not-visible.

ii. `o = np.argsort(ttimes, kind='stable')`; `mean_y, vis_counts = bin_visible_mean(ttimes, tdata[:, 1], visible, go)`; `out = np.full(mean_y.shape, 3, dtype=np.int64)`.

iii. The likelihood distribution is strongly bimodal, making 0.9 insensitive in practice. Sorting handles two malformed sessions, and no imputation is used because not-visible is an explicit class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are computed over visible, binned y means from retained trial windows. Values below p40 are 0, between thresholds are 1, above p60 are 2, and bins without visible frames are 3.

ii. `p40, p60 = np.percentile(mean_y[vis], [TONGUE_PCT_LOW, TONGUE_PCT_HIGH])`; `cls = np.where(y < p40, 0, np.where(y > p60, 2, 1))`.

iii. The notes interpret “over the session” as all visible 50-ms values among kept trials and choose thresholds on the same binned quantity being categorized.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Visible video frames are assigned to the identical absolute go-centered bin edges used for spikes, and their within-bin means are returned in trial-by-bin order.

ii. `edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()`; `idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)`.

iii. Camera and spike timestamps share the session clock; visual/raw spot checks confirmed alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. No-unit/no-good-unit sessions and sessions with fewer than two usable trials are skipped; missing ephys trials and all-zero-population trials are removed; missing/low-confidence tongue frames become class 3; nonmonotonic spike/video timestamps are sorted; and missing tone onset has a median fallback. Worker exceptions are recorded rather than aborting the run.

ii. `if len(good) == 0: return ...`; `if not np.all(np.diff(ttimes) >= 0): ... np.argsort(...)`; `tone_onset[bad_tone] = go[bad_tone] - fallback`; `except Exception as exc: return {..., 'error': repr(exc)}`.

iii. The notes document each anomaly, including two duplicated/nonmonotonic video streams, partial video sessions, truncated trials, and two empty neural trials; absence is represented explicitly where possible rather than imputed.

## 10-a. What are the most time-consuming steps of the code?

i. NWB/trial-table I/O, reading good-unit spike trains, spike binning, multiprocessing overhead, and writing the roughly 12-GB pickle dominate.

ii. `timing['read_spikes'] = ...`; `timing['bin_spikes'] = ...`; `with ctx.Pool(args.jobs) as pool: ...`; `pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)`.

iii. The notes estimate about 0.25 s/session to open, 0.4 s to read spikes, 0.4 s to bin, and about 15 s to pickle; parallelism reduced the realized conversion to roughly 40 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. A Python loop remains over ragged units to read/bin each spike train and loops/list comprehensions remain when making per-trial arrays. The expensive trial and video-bin loops were already vectorized with flattened edges, `searchsorted`, and cumulative sums.

ii. `for i, st in enumerate(spike_times_list): ...`; `spike_lists = [conceptually built by for i in good]`; `neural_trials = [np.ascontiguousarray(rates[:, t, :]) for t in range(n_trials)]`.

iii. The notes say the unit loop is inherent to ragged trains, while replacing neuron×trial histograms and per-bin video masks yielded about 100× and 50× binning speedups.

## 10-c. What processing does the code repeat multiple times?

i. Each session repeats the same NWB reads, curation, binning, and array assembly; within a session it separately constructs trial lists and repeats scalar trial outputs across 80 bins. Plot mode also revisits already computed arrays. Core derived quantities otherwise are computed once.

ii. `np.full(N_BINS, choice[t], ...)`, `np.full(N_BINS, outcome_code[t], ...)`, and `np.full(N_BINS, early_code[t], ...)` occur for every trial; `for i, res in ...` processes each file identically.

iii. The notes emphasize a single pass per file and reuse of module-level bin grids; repetition is chiefly required output formatting rather than redundant scientific computation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal conversion, little scientific processing is discarded. It computes extensive session diagnostics/timings and region metadata that the decoder may not use; optional plot mode performs additional selection and plotting solely for validation. Temporary full rate tensors, visible-frame counts, raw tracking arrays, and spike lists are discarded after assembly.

ii. `timing = OrderedDict()`; `info = {... 'timing': ..., 'frac_tongue_visible': ...}`; `if show_processing: _plot_processing(...)`; `del spike_lists`.

iii. The AI justifies diagnostics and plots as sanity checks; they do not alter converted signals, and plotting is opt-in.
