# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `sub-*/*.nwb` file, sorts the paths, opens each once with `h5py`, and builds one result per retained file.

ii. `nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))` and `with h5py.File(fpath, 'r') as f:`

iii. The trajectory identifies the NWB files as one file per recording session and chose direct HDF5 access for speed.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from each filename's first underscore-delimited token; subjects are accumulated in first-seen file order and sessions index that list.

ii. `subject_id = basename.split('_')[0]` and `subject_idx = np.array([all_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)`

iii. The trajectory treated the `sub-...` filename prefix as the animal identifier; it did not discuss using the NWB subject field.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions failing quality/performance criteria are omitted.

ii. `for i, fpath in enumerate(nwb_files): result = process_session(fpath)`

iii. The agent inferred the dataset's file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`; the same row indices address go cues and behavioral labels. Trials are subsequently masked.

ii. `trials = f['intervals/trials']`; `n_trials = len(trials['id'][:])`; `go_t = go_times[trial_idx]`

iii. The trajectory observed that go-cue count matched trial count and used that positional correspondence.

## 1-e. How are trials filtered based on quality controls?

i. The code first drops whole sessions below 65% responded-control performance or below 50 control hits per side. Within retained sessions it excludes `auto_water`, `free_water`, and trials outside an approximate min/max good-unit spike-time range; it keeps early-lick, ignore, and photostim trials.

ii. `trial_mask = (auto_water == 0) & (free_water == 0)` and `valid_trial_mask = np.array([(go_times[ti] - TIME_BEFORE >= min_spike_time - 1.0) and (go_times[ti] + TIME_AFTER <= max_spike_time + 1.0) for ti in trial_indices])`

iii. The agent cited paper analysis criteria for session filtering, retained labels required as decoder outputs, and added spike-range filtering after discovering all-zero trials. It used a one-second tolerance rather than `obs_intervals`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times`, `units/spike_times_index`, selected unit indices, and trial go-cue timestamps.

ii. `all_spike_times = units['spike_times'][:]`; `spike_times_index = units['spike_times_index'][:]`

iii. The agent identified sorted spike times as the raw neural representation and go cues as the alignment reference.

## 2-b. How is the `neural` data processed?

i. For each retained unit, `searchsorted` at every absolute bin edge produces spike counts, which are divided by 0.05 s to yield Hz. No smoothing or normalization is applied.

ii. `edge_counts = np.searchsorted(st, all_edges_flat).reshape(n_trials, N_BINS + 1)` and `rates_all[i, :, :] = np.diff(edge_counts, axis=1) / BIN_SIZE`

iii. The trajectory explicitly chose non-overlapping spike counts divided by bin width to match the reference method's firing-rate calculation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `unit_quality` string equals `good` are retained; sessions with none are dropped.

ii. `unit_quality = ... units['unit_quality'][:]`; `good_indices = np.where(unit_quality == 'good')[0]`

iii. The agent believed `unit_quality == 'good'` was the paper's classifier-based QC verdict, despite considering classifier logic in the trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 seconds are added to each trial's absolute go-cue timestamp, and spikes are counted between those edges.

ii. `all_edges = go_subset[:, None] + BIN_EDGES[None, :]`

iii. The agent followed the requested go-cue window and relied on NWB streams sharing an absolute session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins. Raw spike timestamps are histogrammed into these bins; no further rebinning occurs.

ii. `BIN_SIZE = 0.05`; `N_BINS = int(round((TIME_BEFORE + TIME_AFTER) / BIN_SIZE))`

iii. These values directly implement the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times` and each trial's `go_start_times`; the last sample start after the previous go and before the current go is selected.

ii. `mask_s = (sample_starts > lower) & (sample_starts < go_t)`; `tone_onsets[i] = candidates[-1]`

iii. The agent reasoned that early licks replay the sample and the final presentation before go is behaviorally relevant.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone's go-relative time is subtracted from each bin center. If no tone is found, a hard-coded 1.85-s tone-to-go interval is imputed.

ii. `time_from_tone = BIN_CENTERS + 1.85` or `time_from_tone = BIN_CENTERS - (tone_t - go_t)`

iii. The trajectory derived 1.85 s from the 0.65-s sample plus 1.2-s delay; it otherwise considered no processing necessary.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 go-cue-relative bin centers used by neural bin edges.

ii. `BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2`

iii. The shared go cue and grid provide alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses absolute `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents`.

ii. `ps_start_abs = ...['photostim_start_times/timestamps'][:]`; `ps_stop_abs = ...['photostim_stop_times/timestamps'][:]`

iii. The agent preferred event timestamps for finer precision and expected absent event streams to mean no stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each trial starts as zeros. Every global stimulation interval overlapping the trial window marks centers satisfying start-inclusive/end-exclusive as one.

ii. `on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)`; `photostim_vec[on_mask] = 1.0`

iii. The agent chose the required binary, time-varying representation.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation times are converted to go-relative times and compared with neural-bin centers.

ii. `ps_s_rel = ps_s - go_t`; `ps_e_rel = ps_e - go_t`

iii. The shared absolute clock and go cue align the streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial `outcome` and `trial_instruction`.

ii. `outcome = outcomes[trial_idx]`; `instruction = instructions[trial_idx]`

iii. The agent recognized choice was not directly stored: hits follow instruction, misses oppose it, and ignores are no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Codes are left=0, right=1, no lick=2 and are repeated across all 80 bins.

ii. `choice = 0 if instruction == 'left' else 1` for hits; the mapping is reversed for misses; `outputs[0, :] = choice`

iii. Repetition makes the per-trial categorical target compatible with the common time-shaped output array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial table's `outcome` column.

ii. `outcomes = ... trials['outcome'][:]`

iii. The raw categories already correspond to the requested target.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps hit=0, miss=1, ignore=2 (unknowns default to ignore) and repeats the code across bins.

ii. `outcome_val = {'hit': 0, 'miss': 1, 'ignore': 2}.get(outcome, 2)`; `outputs[1, :] = outcome_val`

iii. The trajectory only states that categorical outputs are encoded and repeated; it does not justify this category order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes from the trials table's `early_lick` strings.

ii. `early_licks = ... trials['early_lick'][:]`

iii. The field directly provides the requested label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and every other value maps to 1; the result is repeated across bins.

ii. `early_val = 0 if early_licks[trial_idx] == 'no early' else 1`; `outputs[2, :] = early_val`

iii. The agent used the requested no/yes coding and a time-shaped representation.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses side-camera tongue-tracking timestamps plus data column 1 (y) and column 2 (confidence).

ii. `tdata = tt['data'][:]`; `tongue_y = tdata[:, 1]`; `tongue_conf = tdata[:, 2]`

iii. The trajectory identified the DLC-like y and likelihood channels and noted that low confidence corresponds to a retracted/not-visible tongue.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Positions with confidence over 0.9 define session-wide raw-frame percentiles. For each output bin, the nearest camera frame within 5 ms is selected; if confident it is categorized, otherwise class 3 is retained. Frames are not averaged within 50-ms bins.

ii. `vis = tongue_conf > DLC_CONFIDENCE_THRESHOLD`; `tongue_y_p40 = np.percentile(tongue_y[vis], 40)`; `best_idx = np.where(dist_prev < dist_cur, idx_prev, idx_all)`

iii. The agent chose 0.9 as a standard DLC threshold after inspecting the confidence distribution and described its nearest-frame implementation as vectorized.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of all confidence-over-0.9 raw frame y-values define classes: below 40th=0, 40th to below 60th=1, at/above 60th=2, unavailable/low-confidence=3.

ii. `tongue_y_disc[valid & (y_vals < tongue_y_p40)] = 0`; analogous masks assign 1 and 2.

iii. Per-session 40/60 cutoffs and the fourth visibility class follow the instructions; the agent justified confidence filtering as necessary because the tongue is usually retracted.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. At each go-cue-relative neural-bin center, the nearest absolute camera timestamp is used, provided it lies within 5 ms.

ii. `t_abs_all = go_t + BIN_CENTERS`; `close_enough = best_dist < 0.005`

iii. The agent relied on the shared session clock and selected nearest frames to match bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no selected units or too few valid trials are dropped. Trials outside inferred spike coverage are dropped. Missing tones are imputed at 1.85 s before go. Missing tongue tracking or non-close/low-confidence frames become not-visible. Missing photostim events produce zeros; unknown behavioral strings fall into fallback categories.

ii. `if len(good_indices) == 0: return None`; `if np.isnan(tone_t): time_from_tone = BIN_CENTERS + 1.85`; `tongue_y_disc = np.full(N_BINS, 3, dtype=np.int64)`

iii. Spike-range filtering was added after validation exposed all-zero neural trials. Other fallbacks were pragmatic, though the trajectory did not verify each against file semantics.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large NWB arrays and binning spikes dominate; the first nested unit-by-trial approach was too slow, and the optimized per-unit search across all trial edges reduced a session to roughly 2.5 seconds. Full conversion and large pickle writing remain costly.

ii. `for i, st in enumerate(unit_spikes): edge_counts = np.searchsorted(st, all_edges_flat)...`

iii. The trajectory explicitly diagnosed millions of small per-trial histogram/search calls as the bottleneck and optimized them.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The original per-unit/per-trial neural loop was vectorized over trials, but a per-unit loop remains because spike trains are ragged. Session, region, trial-output, photostim-event, and good-unit-range loops remain; the output-building trial loop and global-event scan could be further vectorized/indexed.

ii. `for i, st in enumerate(unit_spikes):` and `for t_i, trial_idx in enumerate(trial_indices):`

iii. The agent specifically optimized neural trial handling and tongue nearest-frame lookup; it considered the remaining structure acceptable.

## 10-c. What processing does the code repeat multiple times?

i. Every trial scans all photostimulation intervals, allocates output arrays, and repeats three scalar trial labels across 80 bins. Subject lookup later repeatedly uses list search. Spike slices are first collected into `unit_spikes`, then iterated again, and good-unit spikes are scanned separately for recording bounds.

ii. `for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):`; `outputs[0, :] = choice`; `all_subjects.index(s['subject_id'])`

iii. The trajectory focused on eliminating repeated per-trial spike searches but did not discuss these smaller repetitions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It parses electrode hemisphere into `side` but discards it, computes/stores metadata bin centers redundant with other metadata, and performs extensive fine-to-major brain-region mapping solely for indexing. It also computes performance counts for sessions that are then discarded when a criterion fails; this is necessary for filtering but produces no output.

ii. `side, region = get_electrode_target_region(...)`; `_, target_region = elec_region_cache[elec_idx]`; `'bin_centers': BIN_CENTERS.tolist()`

iii. The agent discussed whether hemisphere labels should be retained, then ultimately stripped them; it did not identify these operations as waste.
