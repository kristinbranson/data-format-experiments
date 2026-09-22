# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively finds every NWB file under `/app/data`, sorts the paths, and reads each directly as HDF5. It loads synchronized behavior arrays, sparse reward timestamps, ROI metadata, and selected deconvolved ophys arrays.

ii. `files=sorted(DATA_ROOT.rglob('*.nwb'))` and `with h5py.File(path, 'r') as f:`; behavior is read with `beh = {k: np.asarray(b[k]['data']) for k in aligned_names}`.

iii. The notes say the dataset has 152 NWBs in subject directories and that processing one session at a time is faster and more memory-efficient than higher-level loading.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken from parent directory names for the global subject list and from NWB metadata for each session; a lookup maps sessions to subject indices.

ii. `subjects = sorted({p.parent.name.replace('sub-','') for p in files}, ...)`; `subject = scalar_text(f['general/subject/subject_id'])`; `subject_lookup = {s:i for i,s in enumerate(subjects)}`.

iii. The notes report 11 subject directories and verify the resulting session distribution (m11 has 12 sessions, all others 14).

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and becomes one element in each of `neural`, `input`, and `output`.

ii. `for i,p in enumerate(files): ... n,x,y,info=load_session(p); neural.append(n); inputs.append(x); outputs.append(y)`.

iii. The notes identify one `*_behavior+ophys.nwb` file per session and retain all 152 supplied sessions.

## 1-d. How are the data split into trials?

i. Every positive `trial_start` sample is paired with the first later positive `teleport` sample. A trial includes its start and excludes teleport.

ii. `starts = np.flatnonzero(start_signal > 0)`; `teleports = np.flatnonzero(teleport_signal > 0)`; `pairs.append((int(s), int(teleports[k])))`; later, `sl = slice(s, e)`.

iii. The notes say the NWBs have no trials table and define complete traversals by these pulses; excluding teleport avoids intertrial artifacts.

## 1-e. How are trials filtered based on quality controls?

i. No duration or quality threshold is applied. A paired, nonempty start-to-teleport interval is retained; malformed or empty intervals are skipped or fail shape/finite-value checks.

ii. `if e <= s: continue` and `if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0): raise ValueError(...)`.

iii. The notes state that all 12,216 complete pulse-defined trials are retained and all sessions already have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `processing/ophys/Deconvolved/plane*/data`, filtered using `iscell` and mapped across planes with `planeIdx`.

ii. `ds = f['processing/ophys/Deconvolved'][plane_name]['data']`; `local_keep = np.flatnonzero(iscell[global_ids])`; `arr = np.asarray(ds[:, local_keep], dtype=np.float32)`.

iii. The agent reasoned that the NWB is a processed export containing Suite2p/OASIS deconvolved activity, so recomputing from fluorescence and neuropil would duplicate processing.

## 2-b. How is the `neural` data processed?

i. Accepted ROI columns are read per plane, terminal rows are trimmed to behavior length, planes are concatenated, trials are sliced, and time-by-neuron data are transposed to neuron-by-time float32. No new dF/F or deconvolution is performed.

ii. `pieces.append(arr[:n_beh])`; `neural_tn = np.concatenate(pieces, axis=1)`; `neu = np.asarray(neural_tn[sl].T, dtype=np.float32)`.

iii. The notes explicitly say to avoid a “second dF/F/deconvolution pass” because the supplied signal was interpreted as already processed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p `iscell > 0` ROIs are retained. There is no speed-correlation putative-interneuron filter or functional place/reward-cell filter.

ii. `iscell = (iscell0[:, 0] if iscell0.ndim > 1 else iscell0) > 0`; `local_keep = np.flatnonzero(iscell[global_ids])`.

iii. The notes regard `iscell` as the appropriate general quality filter and functional subsets as analysis-specific rather than decoder-wide exclusions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural rows share the behavior row index; slicing begins at the `trial_start` sample, so the first neural column is aligned to trial start (time zero).

ii. `sl = slice(s, e)`; `reltime = (timestamps[sl] - timestamps[s])`; `neu = neural_tn[sl].T`.

iii. The notes report direct raw-versus-converted row checks and zero-valued first time samples.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native synchronized volume samples are preserved at 15.5078125 Hz, or 64.4836 ms per bin. There is no temporal rebinning or interpolation.

ii. `DT = 1.0 / 15.5078125` and `'time_bin_size':DT*1000.0`.

iii. The notes explain that two-plane metadata report a per-plane scanner rate, while the stored rows remain volume-synchronized to behavior at about 15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` time-series timestamps and the paired trial-start index.

ii. `timestamps = np.asarray(b['position']['timestamps'], dtype=np.float64)` and `reltime = timestamps[sl] - timestamps[s]`.

iii. The agent found frame-aligned behavior and neural rows synchronized and used position timestamps as the common clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from every timestamp in the trial and the result is cast to float32.

ii. `reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)`.

iii. This makes each retained trial begin at exactly zero without resampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same slice indexes timestamps and neural rows; equal trial lengths and a zero first time are checked.

ii. `neu = neural_tn[sl].T`; `if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0): raise ...`.

iii. The notes report independent comparisons to raw NWB rows and no alignment errors.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the synchronized behavior `environment` time series.

ii. `env_values = np.asarray(beh['environment'][sl])`.

iii. Exploration found native 0/1 values corresponding to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median trial value is rounded to an integer and repeated over every time point.

ii. `env = int(np.rint(np.median(env_values)))`; `np.full(len(pos), env, np.float32)`.

iii. The variable was observed to be constant within complete trials; collapsing it enforces the per-trial specification.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It comes from the synchronized behavior `trial number` value at the trial-start sample.

ii. `source_trial = float(beh['trial number'][s])`.

iii. The notes describe preserving the valid native 0-based label while excluding the pre-track `-1` sentinel.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The source value is converted to float and repeated across the trial.

ii. `np.full(len(pos), source_trial, np.float32)`.

iii. It is semantically per-trial but stored time-expanded for a uniform input matrix.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward` timestamps, position timestamps, and the previous paired trial interval.

ii. `reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)` and `outcomes = np.asarray([int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e]))) ...])`.

iii. The notes identify `Reward` as sparse rather than frame-aligned and assign events by timestamp.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial outcome is whether any reward timestamp lies in its interval. The following trial receives that binary value; the first trial receives 0. It is repeated over time.

ii. `prev = int(outcomes[j-1]) if j > 0 else 0`; `np.full(len(pos), prev, np.float32)`.

iii. The agent followed the required omitted/rewarded binary coding and treated an unknown pre-session outcome as 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses synchronized `position` plus a reward-zone interval inferred from the NWB session identifier and trial index. The raw `reward_zone` stream is not used because the agent found it constant zero.

ii. `identifier = scalar_text(f['identifier'])`; `labels, zone_coords, zone_codes = scene_zone_sequence(...)`; `signed_dist = distance_to_interval(pos, zstart, zstop)`.

iii. The notes say scene labels encode A/B/C and switches, and use the reference intervals A=[80,130], B=[200,250], C=[320,370], with switches at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position minus zone start is used before the interval, zero inside it, and position minus zone stop after it; the signed distance is then categorized.

ii. `return np.where(pos < start, pos - start, np.where(pos > stop, pos - stop, 0.0))`.

iii. This follows the reward-relative coordinate definition and makes all in-zone locations exactly zero.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement seven bins: `<-50`, `[-50,-10)`, `[-10,0)`, exactly 0, `(0,10]`, `(10,50]`, and `>50`.

ii. For example, `y[(x >= -10) & (x < 0)] = 2`, `y[x == 0] = 3`, and `y[(x > 10) & (x <= 50)] = 5`.

iii. The agent states these are the exact task inequalities and tested all boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the identical start-to-teleport row interval.

ii. `pos = beh['position'][sl]`; `neu = neural_tn[sl].T`; `signed_dist = distance_to_interval(pos, ...)`.

iii. The notes cite exact row-level spot checks and equal-length validation.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from synchronized behavior `position`.

ii. `pos = np.asarray(beh['position'][sl], dtype=np.float32)`.

iii. Position is already expressed in centimeters along the virtual corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No smoothing or resampling is applied; each native sample is categorized by explicit masks.

ii. `bin_position(pos)` is placed in the second output row.

iii. The notes intentionally preserve temporal samples rather than apply the paper’s spatial averaging.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Bins are `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360` cm.

ii. `y[(x >= 270) & (x <= 360)] = 3`; `y[x > 360] = 4`.

iii. The agent interprets the stated 450 cm track as five 90 cm bins and explicitly assigns the 360 cm boundary to category 3.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial slice is used for position and neural rows.

ii. `pos = beh['position'][sl]` and `neu = neural_tn[sl].T`.

iii. Shared row synchronization was validated against raw NWB data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the synchronized behavior `lick` stream.

ii. `lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)`.

iii. The notes identify it as a native frame-level behavior stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every positive raw value becomes 1 and all other values become 0.

ii. `(np.asarray(beh['lick'][sl]) > 0).astype(np.int64)`.

iii. This implements the requested binary no/yes output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural rows use the identical trial slice.

ii. `lick = ... beh['lick'][sl]` and `neu = neural_tn[sl].T`.

iii. The agent relies on and spot-checks the NWB’s synchronized row order.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB session `identifier`, the number/index of paired trials, and fixed A/B/C coordinate definitions; the raw `reward_zone` stream is deliberately ignored.

ii. `scene_from_identifier(identifier)` and `scene_zone_sequence(..., len(pairs))`.

iii. The agent found `reward_zone` constant zero and considered the identifier’s scene sequence authoritative, validating it against reward-event positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex parses a fixed location or before/after transition. Transition sessions switch at trial 30; labels map A/B/C to 0/1/2 and are repeated over each trial.

ii. `labels = [before] * c + [after] * (n_trials - c)` with `c = min(int(change_trial), n_trials)`; `np.full(len(pos), zone_codes[j], np.int64)`.

iii. The notes say this reproduces the reference scene logic and supplies labels on omission trials where reward events cannot identify a zone.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from sparse behavior `Reward` timestamps, position timestamps, and paired trial boundaries.

ii. `reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)` and the `outcomes` comprehension over `pairs`.

iii. The agent correctly recognized that Reward is not a frame-length series.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward event occurs from its start timestamp through its teleport timestamp, otherwise 0; the result is repeated over trial time.

ii. `int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))`; `np.full(len(pos), outcomes[j], np.int64)`.

iii. The notes report 10,342 rewarded and 1,874 omitted trials and exact raw-event checks.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Frame-aligned behavior length disagreement raises an error. Neural data shorter than behavior raises an error; known terminal neural excess is truncated. ROI mapping, finite values, trial shapes, and accepted-cell counts are checked. The constant reward-zone field is bypassed via identifier parsing. There is no imputation.

ii. `if any(len(v) != n_beh ...): raise ValueError`; `pieces.append(arr[:n_beh])`; `if neural_tn.shape[0] < n_beh: raise ValueError`; `np.any(~np.isfinite(neu))` also raises.

iii. The notes document ten files with one extra terminal neural row and justify terminal trimming without shifting or interpolation.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large selected neural matrices and serializing the roughly 9.8 GB pickle dominate; session conversion is otherwise vectorized and the full run took about 68 seconds plus 7 seconds to save.

ii. The relevant operations are `np.asarray(ds[:, local_keep], dtype=np.float32)` and `pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)`.

iii. The notes identify HDF5 I/O, neural size, and pickle output as the main costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial reward-outcome comprehension and per-trial construction loop could be partially vectorized, though variable-length list output still requires trial slicing. File and plane loops are structurally necessary. Binning itself is already vectorized.

ii. `outcomes = np.asarray([int(np.any(...)) for s, e in pairs])` and `for j, (s, e) in enumerate(pairs):`.

iii. The notes emphasize vectorized discretization and one-session-at-a-time processing rather than claiming all loops can be removed.

## 13-c. What processing does the code repeat multiple times?

i. Each trial repeatedly allocates constant time-expanded rows for environment, trial number, previous outcome, zone, and outcome. Reward timestamps are also scanned once per trial. Unlike the human survey/conversion workflow, NWBs are not loaded in a separate full survey pass.

ii. Repeated constructs include `np.full(len(pos), ...)` and `np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e]))`.

iii. The agent intentionally avoided repeated dF/F/OASIS computation and a second whole-dataset read.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `plot_info`, `timestamps`, `pairs`, selected ROI IDs, zone labels, and outcomes are assembled in each session’s temporary `info`; most are removed before metadata storage, and `plot_info` is built even when plots are disabled. Diagnostic plotting is optional and not used by decoder training.

ii. `plot_info.append((pos, speed, lick, signed_dist, out, labels[j]))` and `info_clean={k:v for k,v in info.items() if k not in ('plot_info','timestamps','pairs','selected_global','outcomes','zone_labels')}`.

iii. The notes justify plots and spot-check structures for validation, but they are not part of downstream analysis and could be gated behind `show_processing`.
