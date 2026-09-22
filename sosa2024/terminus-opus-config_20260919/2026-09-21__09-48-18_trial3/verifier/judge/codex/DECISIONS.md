# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively identifies every NWB session with a sorted glob, processes each file (eight spawned workers by default), and reads NWB datasets directly with `h5py`.

ii. `files = sorted(glob.glob('/app/data/sub-*/*.nwb'))` and `with h5py.File(fn, 'r') as f:`

iii. The notes say the dataset has 11 subject directories and 152 NWB files, one per subject/session; sorting gives deterministic order and multiprocessing makes the 87-GB conversion tractable.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each file's `general/subject/subject_id`; the final unique subject list is numerically sorted and each session gets an index into it.

ii. `subject=f['general/subject/subject_id'][()].decode()` and `subject_idx=np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64)`

iii. The notes report that these IDs reproduce the 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. A processed file produces one nested session entry; sessions with fewer than two retained trials are excluded.

ii. `sessions = [r[0] for r in results if r[0] is not None and len(r[0]['neural']) >= 2]`

iii. This follows the filename/data organization and the decoder's minimum-two-trials requirement.

## 1-d. How are the data split into trials?

i. Starts are positive `trial_start` samples and stops are positive `teleport` samples. Emitted trial slices are `[start-1, stop-1)`, one frame before each flag.

ii. `starts=np.where(g('trial_start') > 0)[0]`, `stops=np.where(g('teleport') > 0)[0]`, and `sl = slice(s - 1, e - 1)`

iii. The agent says this mirrors `preprocessing.dff`/`glmUtils.get_timeseries_data` and checked paired, nonoverlapping flags, although the human converter emits `[start, stop)`.

## 1-e. How are trials filtered based on quality controls?

i. The first trial of every session is dropped because previous outcome is treated as undefined; lick-sensor-error trials (>30% of frames with lick count >2) and trials with nonfinite events are dropped. Only complete paired laps are considered and sessions must retain at least two trials.

ii. `if i == 0: continue`, `if lick_error[i]: continue`, and `if not np.all(np.isfinite(ev)): ... continue`

iii. The notes say the 30% rule exactly identifies the paper's 81 bad trials; dropping them makes sense because lick is required. First trials were dropped instead of assigning the reference's value 0.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu), after selecting `iscell` ROIs across all planes; stored `Deconvolved` is deliberately not used.

ii. `Fl.append(f['processing/ophys/Fluorescence/plane%d/data' % p][:, keep].T)` and the analogous `Neuropil` read.

iii. The paper recomputes dF/F and OASIS events from F/Fneu; the stored deconvolution is a different suite2p signal.

## 2-b. How is the `neural` data processed?

i. It subtracts `0.7*Fneu`, adds back per-trial mean neuropil, calculates a per-trial maximin baseline (Gaussian sigma 15, 300-frame minimum then maximum), computes `(F-baseline)/abs(baseline)`, smooths with sigma 2, and applies OASIS (`tau=.7`) at the per-plane frame rate. Unlike the reference, it never enables teleport-spanning baseline windows.

ii. `base = gaussian_filter1d(...)`; `base = minimum_filter1d(base, MAXIMIN_WINDOW, axis=-1)`; `events[:, sl] = dcnv.oasis(..., TAU, fs)`

iii. The notes cite the paper preprocessing parameters, but explicitly planned `keep_teleports=False`; this misses the reference's day-specific `teleport_sessions` behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps manually curated `iscell` ROIs, removes cells with dF/F–speed Pearson r>0.5, and additionally removes cells whose minimum baseline is not above 5% of median raw fluorescence.

ii. `keep_cells = (~is_int) & (~bad_baseline)` with `bad_baseline = ~(min_base > BASELINE_MIN_FRAC * scale)`

iii. `iscell` and interneuron removal match the Methods. The notes label the 5% baseline rule as an extra numerical-stability rule not present in the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural events are sliced with the same `[trial_start-1, teleport-1)` indices as behavior, so their first sample is one frame before the trial-start flag.

ii. `ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]`

iii. The agent considered this the paper-code trial window and records `off_start=-dt`; the human solution aligns emitted arrays at the actual start flag.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame, about 64.48 ms (~15.51 Hz), is retained; no rebinning or resampling is applied.

ii. `dt = float(np.median(np.diff(S['t'])))` and `time_bin_size=float(np.mean([s['dt'] for s in sessions]) * 1000.0)`

iii. Behavior is already aligned to imaging frames and the notes state that preserving frames maintains exact alignment.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses timestamps from the raw `position` behavioral series.

ii. `t=beh['position']['timestamps'][()].astype(np.float64)`

iii. The notes found that behavioral series share imaging-frame timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The actual start-flag timestamp `t[s]` is subtracted from a slice beginning at `s-1`, producing a first value of approximately -0.0645 s.

ii. `tt = S['t'][sl] - S['t'][s]`

iii. This deliberately represents the one-frame pre-start window and is reflected in metadata `off_start=-dt`; the reference begins at zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with exactly the same `sl` and length `T` as neural events. Ten mismatched sessions are truncated to the common neural/behavior length first.

ii. `sl = slice(s - 1, e - 1)`, `T = ...`, and `x[0] = tt`

iii. The notes say both streams begin at time zero and the extra imaging frame is at the end.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw `environment` behavioral time series.

ii. `env=g('environment')`

iii. Inspection showed values 0/1 corresponding to ENV1/ENV2 and constant within trials.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median environment value within the trial is rounded and repeated over all trial frames.

ii. `env_trial[i] = int(np.round(np.median(S['env'][sl])))` and `x[1] = env_trial[i]`

iii. This robustly enforces the observed per-trial constant representation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the zero-based loop/lap index from the paired trial flags, not the stored `trial number` values.

ii. `for i, (s, e) in enumerate(zip(starts, stops)):` and `x[2] = i`

iii. The notes preserve original lap indices after dropped trials so switch trial 30 remains interpretable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond repeating the zero-based index at every time point.

ii. `x[2] = i`

iii. This matches the chosen within-session lap definition.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Sparse `Reward` timestamps are mapped to behavioral frames and combined with whether the raw `reward_zone` flag was active in that trial.

ii. `reward_frames = np.searchsorted(S['t'], S['reward_t'])` and `isreward[i] = int(got_reward and np.any(rzflag))`

iii. The agent cites `behavior.get_trial_types`: reward delivery plus active reward zone defines a rewarded trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For retained trial `i`, the already computed binary outcome of trial `i-1` is repeated over time; trial zero is dropped.

ii. `x[3] = isreward[i - 1]`

iii. Dropping the first lap avoids inventing a previous result, while the human reference assigns it 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and zone bounds inferred from the NWB identifier's scene name, with a switch after lap 30.

ii. `rz_coords, rz_labels = scene_reward_zones(S['scene'], n_trials)` and `d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])`

iii. This mirrors `behavior.get_reward_zones` and was checked against observed reward-zone onset positions; the human reference instead uses a Viterbi assignment from position plus `reward_zone` flags.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is negative before the zone, zero everywhere inside it, and positive after it, measured to the nearest boundary.

ii. `d[before] = pos[before] - rz_start`; `d[after] = pos[after] - rz_end`

iii. The notes interpret “distance to any location in the reward zone” as zero throughout the 50-cm zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks create the seven requested categories; exact +10 and +50 remain in the lower respective bins.

ii. `out[(d > 0.0) & (d <= 10.0)] = 4`; `out[(d > 10.0) & (d <= 50.0)] = 5`

iii. The masks directly implement the task's inclusive wording and isolate exact zero as category 3.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and events use the identical `[s-1:e-1]` slice and resulting length.

ii. `pos = S['pos'][sl]` followed by `y[0] = digitize_rz_distance(d_rz)`

iii. Behavior was already sampled on imaging-frame timestamps.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw behavioral `position` series.

ii. `pos=g('position')` and `pos = S['pos'][sl]`

iii. This is the VR corridor coordinate in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial slice is passed directly to `np.digitize`; no smoothing or resampling occurs.

ii. `y[1] = np.digitize(pos, POSITION_EDGES)`

iii. Direct values preserve imaging-frame alignment.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Edges 90, 180, 270, and 360 cm produce five 90-cm categories; exact edges enter the higher bin.

ii. `POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]`

iii. These are five equal divisions of the 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same trial slice and frame count as neural.

ii. `sl = slice(s - 1, e - 1)` and `pos = S['pos'][sl]`

iii. The shared behavioral/imaging indexing requires no interpolation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It uses the raw behavioral `lick` cumulative count series.

ii. `lick=g('lick')` and `lick = S['lick'][sl]`

iii. The notes describe values as cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Counts at least 1 are converted to 1; lower values become 0. Trials with the identified stuck-sensor pattern are removed.

ii. `y[3] = (lick >= 1).astype(np.int64)`

iii. This supplies the required binary output while excluding invalid lick labels.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The same `[s-1:e-1]` frame slice is used for licks and neural events.

ii. `lick = S['lick'][sl]`

iii. The raw behavioral data are already aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from the scene name stored in the NWB `identifier`, plus the within-session trial index for switch scenes.

ii. `scene=f['identifier'][()].decode().split('/')[-1]` and `scene_reward_zones(S['scene'], n_trials)`

iii. The reference paper code derives zones from `sess.scene`; observed zone-entry position was used as a sanity check.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene text is parsed into A/B/C; switch scenes change after 30 trials. The label maps to 0/1/2 and is repeated across frames.

ii. `labels = [first] * min(CHANGE_TRIAL, n_trials)` and `y[4] = RZ_LABELS.index(rz_labels[i])`

iii. This reproduces known A=[80,130], B=[200,250], C=[320,370] and the task switch schedule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from sparse `Reward` timestamps and the per-frame `reward_zone` flag.

ii. `got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))` and `np.any(rzflag)`

iii. The agent follows the paper's `get_trial_types` definition rather than reward presence alone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped with `searchsorted`; a trial is 1 only if it contains a reward and any active-zone flag, otherwise 0. The result is constant over time.

ii. `isreward[i] = int(got_reward and np.any(rzflag))` and `y[5] = isreward[i]`

iii. This distinguishes rewarded from omission trials according to the reference behavioral function.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. One-frame neural/behavior length mismatches are truncated to the common length; trial arrays are clipped accordingly. Nonfinite-event trials are dropped, numerical baseline failures trigger cell removal, worker exceptions omit failed sessions, and assertions/warnings check boundaries and zone consistency.

ii. `n = min(d['F'].shape[1], n_beh)`; `d['F'] = d['F'][:, :n]`; and `_worker` returns `(None, ... error=True)` on exceptions.

iii. The notes found ten sessions with one extra ophys frame and reasoned it was at the end. The baseline filter is an additional, non-reference safeguard.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large F/Fneu arrays, per-trial maximin filtering and OASIS deconvolution, writing the multi-gigabyte pickle, and optional plotting are the dominant work. Sessions are parallelized.

ii. `compute_dff_and_events(...)`, `dcnv.oasis(...)`, and `with ctx.Pool(...)`

iii. Timings (`t_load`, `t_dff`, `t_total`) are explicitly collected; the notes emphasize the 87-GB source and large converted arrays.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops repeatedly build masks/outcomes and output arrays; baseline/smoothing/OASIS must still respect variable trial boundaries. Scene labels and several per-trial summaries could be vectorized. The code already vectorizes all cell–speed correlations with matrix multiplication.

ii. `for i, (s, e) in enumerate(zip(starts, stops)):` appears in neural processing, task-variable calculation, and output construction.

iii. Variable trial lengths make full-session vectorization awkward, while vectorizing across neurons materially improves speed.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trial boundaries multiple times: copying F/Fneu, baseline estimation, dF/F smoothing/OASIS, task summaries, and final trial construction. Optional plots again reconstruct position, speed, distance, and reward-frame mappings.

ii. Repeated `for start, stop in zip(starts, stops):` loops occur in `compute_dff_and_events`, followed by two loops in `process_session`.

iii. The repeated passes preserve bounded memory and keep boundary-specific operations clear, but some masks/slices could be cached.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `rz_onset_pos`, timing/statistics dictionaries, many sanity-check concatenations, and optional figures are not stored in decoder tensors. It also computes full-session dF/F although only events and the interneuron mask are ultimately retained.

ii. `rz_onset_pos = np.full(...)`, `stats = dict(...)`, and `allout = np.concatenate(...)`

iii. These are validation and provenance checks documented by the agent; they do not contribute directly to decoder inputs/outputs but were used to detect conversion mistakes.
