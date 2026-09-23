# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `Imaging_Exp_info.npy` as the master index, deduplicates recording IDs, loads the 23 behavior dictionaries in parallel, and loads retinotopy and spike files per recording. Duplicate behavior appearances are used to merge wall-to-stimulus mappings.

ii. `exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()`; `with Pool(min(nproc, len(jobs))) as pool: results = pool.map(_load_beh_file, list(jobs.items()))`; `d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()`

iii. The notes say this covers the paper's 89 unique recordings while avoiding repeated loading of large behavior files and resolving stimulus labels across duplicate experiment-type entries.

## 1-b. How are the data split into subjects?

i. `mname` defines the mouse. Unique sorted mouse names form `subjects`, and each session receives its index.

ii. `subjects = sorted({r['mname'] for r in results})`; `'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64)`

iii. The index explicitly identifies mice; the notes report 19 mice.

## 1-c. How are the data split into sessions?

i. A session is the unique `mname_datexp_blk` recording. Repeated appearances under experiment types/stimulus types are deduplicated.

ii. `key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])`; `rec = recs.setdefault(key, {...})`

iii. The agent verified 142 index appearances correspond to 89 recordings and duplicate frame/trial arrays agree.

## 1-d. How are the data split into trials?

i. Frames satisfying corridor, movement, and nonmissing-trial masks are grouped by `ft_trInd`; each group becomes a variable-length trial. Thus stationary frames inside a traversal are omitted.

ii. `valid = (beh['ft_CorrSpc'][:nfr].astype(bool) & (beh['ft_move'][:nfr] > 0) & ~np.isnan(tr))`; `groups = np.split(np.arange(len(frames)), bounds)`

iii. The agent interpreted the reference's `fr_valid = VRmove & isCorridor` as universal frame curation and sought to match analyses excluding non-running data.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level length/outlier filter is applied. All trial IDs represented after frame masking are retained; the full run reports all 38,110 behavioral trials retained.

ii. `for g, tid in zip(groups, trial_ids): ... kept_trials.append(tid)`

iii. The notes say every trial had at least 11 retained frames and therefore none needed dropping.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane `spks` arrays; `iarea` supplies anatomical labels and selection.

ii. `planes = d['spks']`; `iarea = ret['iarea']`

iii. The files already contain Suite2p deconvolved activity, so the agent did not derive fluorescence from another signal.

## 2-b. How is the `neural` data processed?

i. Per-plane deconvolved traces are sliced to selected neurons and retained frames, concatenated, stored as float32, then split by trial. At most 2,000 in-area neurons are stored per session.

ii. `sel = np.sort(rng.choice(in_area, size=max_neurons, replace=False))`; `out.append(np.asarray(p[np.ix_(rows, frames)], dtype=np.float32))`; `neural_out.append(np.ascontiguousarray(spk[:, g]))`

iii. No dF/F or deconvolution is repeated. The 2,000-neuron cap was justified by the otherwise estimated 152 GB size and the decoder's 100-PC reduction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons assigned to V1, mHV, lHV, or aHV are eligible, after which a fixed-seed random sample of 2,000 is taken when necessary. There is no activity-based quality filter.

ii. `in_area = np.flatnonzero(region_idx >= 0)`; `rng = np.random.default_rng(seed)`; `rng.choice(in_area, size=max_neurons, replace=False)`

iii. The notes state released `spks` were already Suite2p-curated; area selection follows the paper, while subsampling controls storage.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial identity is based on `ft_trInd`, and time is expressed relative to interpolated `StartFr`. Arrays contain corridor frames with `ft_move > 0`, so stationary intervals appear as temporal gaps rather than stored bins.

ii. `t_start = np.interp(beh['StartFr'], fr_axis, t_frames)`; `time_since_start = t_sel - t_start[labels]`; `neural_out.append(np.ascontiguousarray(spk[:, g]))`

iii. Corridor entry was selected as the requested alignment event; variable natural trial lengths were retained.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One retained imaging frame is one bin, about 314.7 ms (3.178 Hz); no aggregation or interpolation of neural values is performed.

ii. `dt = float(np.median([r['dt_s'] for r in results]))`; `'time_bin_size': dt * 1000.0`

iii. The agent regarded the native frame grid as the appropriate resolution. Removed stationary frames mean adjacent stored columns need not be adjacent in time.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and frame timestamps `ft`.

ii. `t_cue = np.interp(beh['SoundFr'], fr_axis, t_frames)`

iii. Frame indices and timestamps are the common time base in the reference code.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. MATLAB-day timestamps are converted to seconds, the possibly fractional cue frame is interpolated, and each retained frame time is subtracted from cue time.

ii. `return (ft - ft[0]) * 86400.0`; `time_to_cue = t_cue[labels] - t_sel`

iii. This produces positive values before the cue and negative values afterward.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Values are calculated for the exact `frames` used to slice neural activity and then indexed by the same trial group `g`.

ii. `spk, _, _ = load_spk_selected(key, sel, frames)`; `inp[0] = time_to_cue[g]`

iii. The notes report independent raw-data spot checks and cue-zero-crossing plots.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each recording's `datexp` and that mouse's earliest recorded date.

ii. `rec['date'] = datetime.date(y, m, d)`; `d0 = min(r['date'] for r in rs)`

iii. The agent treated calendar days since first imaging as a continuous training-day measure.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date difference in calendar days is converted to float and broadcast across every bin of the session's trials.

ii. `r['day_of_training'] = float((r['date'] - d0).days)`; `inp[1] = day`

iii. The notes explicitly prefer elapsed days, producing a 0–92 range, rather than ordinal session number.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It comes from `StartFr` and `ft`.

ii. `t_start = np.interp(beh['StartFr'], fr_axis, t_frames)`

iii. `StartFr` is identified as corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame timestamps are converted to seconds, the fractional start frame is interpolated, and its time is subtracted from each retained frame time.

ii. `time_since_start = t_sel - t_start[labels]`

iii. This preserves elapsed time even across frames removed by the movement mask.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is selected using the same retained frame array and per-trial group as neural data.

ii. `inp[2] = time_since_start[g]`; `neural_out.append(np.ascontiguousarray(spk[:, g]))`

iii. Numeric checks against raw data are reported in the notes.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from merged `WallName`/`UniqWalls`/`stim_id` mappings plus whether the session has any non-NaN `RewardFr`, not directly from `isRew`.

ii. `rec['is_task'] = bool(np.any(~np.isnan(beh['RewardFr'])))`; `reward_available = (stim_ids == 2) & rec['is_task']`

iii. The agent distinguishes being in the rewarded corridor from reward delivery and interprets the decoder wording as the former.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A trial is 1 only when its canonical category is rewarded crop 1 and the recording is a task session; the boolean is broadcast over the trial.

ii. `inp[3] = float(reward_available[tid])`

iii. The notes say `isRew` marks actual delivery and misses some rewarded-corridor trials, motivating this reconstruction.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses trial `WallName` and merged `UniqWalls`–`stim_id` maps from all appearances of a recording.

ii. `for w, s in zip(beh['UniqWalls'], np.atleast_1d(beh['stim_id']).astype(float)): ... w2id[str(w)] = int(s)`; `ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn])`

iii. Merging resolves labels masked in individual duplicate behavior records.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names map to eight task-relative canonical IDs (0–7), with unresolved/third nonrewarded crop assigned 7; the ID is broadcast over trial bins.

ii. `STIM_VALUES = ['nonrew_crop1', ..., 'nonrew_crop3']`; `out[0] = stim_ids[tid]`

iii. The agent preferred task-relative categories because physical texture families differ across mice and retained the otherwise unlabeled third crop as its own class.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`.

ii. `lf = np.round(beh['LickFr']).astype(int)`

iii. The reference uses lick frame indices as the common imaging time base.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are rounded to nearest integer; valid indices set a boolean frame vector to true, so multiple licks in a frame remain one.

ii. `lf = lf[(lf >= 0) & (lf < nfr)]`; `lick_any[lf] = True`

iii. The agent describes this as binary presence of at least one lick in an imaging frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session vector is sliced by the same retained frames and trial group used for neural activity.

ii. `licking = lick_any[frames].astype(np.int64)`; `out[1] = licking[g]`

iii. The notes report raw-data equality checks for sample trials.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-wise `ft_Pos`, in decimeters.

ii. `pos = beh['ft_Pos'][:nfr][frames]`

iii. The methods and behavior dictionary establish the position units.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 dm/m and cast to integer category, then clipped to 0–3.

ii. `pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)`

iii. This implements the requested four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories correspond to [0,1), [1,2), [2,3), and [3,4] m.

ii. `POS_BIN_DM = 10.0`; `'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0]`

iii. The four-meter textured corridor naturally supplies these fixed thresholds.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is first sliced with `frames`, then split with the same `g` indices as neural data.

ii. `out[2] = pos_bin[g]`

iii. Sample checks and processing plots were used to verify boundaries and alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `ft_RunSpeed` on corridor-and-moving frames.

ii. `return beh['ft_RunSpeed'][:nfr][frames]`; `speed = beh['ft_RunSpeed'][:nfr][frames]`

iii. The raw variable is already frame-aligned running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All retained-frame speeds from all sessions are concatenated; global 25th, 50th, and 75th percentiles become thresholds, and `np.digitize` assigns classes.

ii. `speed_thr = np.percentile(speeds, [25, 50, 75])`; `speed_bin = np.digitize(speed, speed_thr).astype(np.int64)`

iii. The agent read “each corresponding to 25% of the data” as global quartiles and verified approximately equal global class frequencies.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global value cutoffs (reported as about 12.42, 25.35, and 40.85 cm/s) define four intervals.

ii. `'speed_bin_edges_cm_per_s': [float(x) for x in speed_thr]`

iii. Global fixed thresholds make category meanings consistent across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is selected by `frames` and then by the same trial group `g`.

ii. `out[3] = speed_bin[g]`

iii. The agent reports raw-data spot checks and visual validation.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior is truncated to neural frame count; NaN trial indices and out-of-range lick frames are discarded; unresolved stimulus walls receive ID 7; finite-value assertions fail conversion on remaining bad values.

ii. `nfr = min(p.shape[1] for p in planes)`; `& ~np.isnan(tr)`; `lf = lf[(lf >= 0) & (lf < nfr)]`; `assert np.all(np.isfinite(a))`

iii. The notes identify behavior streams as 1–2 frames longer than spikes and document the special third-crop label rather than dropping those trials.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and slicing the 405 GB of spike files dominates session conversion; serializing the 6.6 GB pickle is the next visible full-run cost.

ii. `d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()`; `pickle.dump(data, f, protocol=4)`

iii. Per-session load timing is recorded; the full notes report 59.6 s conversion and 10.7 s pickling.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Most frame calculations are vectorized. Remaining loops over planes, sessions, trial groups, and stimulus mappings could partly be consolidated, but trials must ultimately be emitted as separate arrays.

ii. `for p in planes:`; `for g, tid in zip(groups, trial_ids):`; `for w, s in zip(beh['UniqWalls'], ...)`

iii. The notes emphasize multiprocessing, per-plane slicing, and vectorized session-level behavior, so these loops were not considered material bottlenecks.

## 12-c. What processing does the code repeat multiple times?

i. Spike files are opened twice per session: once to discover dimensions and again to load selected values. Trial-frame masks are also computed once for global speed thresholds and again during conversion.

ii. `_, n_neu_spk, nfr = load_spk_selected(key, None, None)`; `spk, _, _ = load_spk_selected(key, sel, frames)`; `session_speed_values(...)` and later `trial_frames(beh, nfr)`

iii. The first spike pass avoids materializing the full matrix, and the preliminary frame pass is needed to establish global speed thresholds before parallel conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and merges behavior fields/maps used mainly for metadata or diagnostics, computes extensive per-session statistics, and optionally constructs plots; these do not affect decoder arrays. It also reads spike metadata in the first of two file loads.

ii. `out.append((reckey, {f: beh[f] for f in BEH_FIELDS if f in beh}))`; `res = {... 'n_licks': int(len(beh['LickFr'])), ...}`; `if show_processing: plot_processing(...)`

iii. These operations support validation, provenance, and visualization; plotting is opt-in and was not part of the full conversion.
