# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master index and every experiment-type behavior file, collapses duplicate views into 89 recording keys, merges their non-NaN `stim_id` assignments, and later loads spikes and retinotopy per recording.

ii. `exp_info = np.load(...'Imaging_Exp_info.npy'...).item()` and `beh_all = np.load(...'Beh_%s.npy' % exp_type...).item()`.

iii. The notes say 142 experiment records are duplicate views of 89 recordings; merging avoids duplicated neural sessions and recovers conflict-free stimulus labels.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique `mname` values and sessions store the corresponding index.

ii. `subjects = sorted({sess_beh[k]['mname'] for k in keys})`; `data['subject_idx'].append(subjects.index(sb['mname']))`.

iii. Mouse identity is explicit in the master index.

## 1-c. How are the data split into sessions?

i. A session is unique `(mname, datexp, blk)`; duplicate experiment views are merged and sorted by mouse/date/block.

ii. `key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])`.

iii. This reproduces the paper's 89 recordings and prevents duplication.

## 1-d. How are the data split into trials?

i. Trial labels come from `ft_trInd`. The agent keeps finite in-range samples inside the textured corridor while `ft_move > 0`, removes unmapped-stimulus and boundary-artifact samples, then groups the remaining contiguous samples by trial.

ii. `keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)`; `uniq, starts = np.unique(trial, return_index=True)`.

iii. It cites the paper's `ft_CorrSpc & (ft_move > 0)` analysis mask and says a StartFr/GrayFr guard fixes 21 wraparound samples.

## 1-e. How are trials filtered based on quality controls?

i. Trials with NaN stimulus roles are removed; trials with fewer than two retained running samples are removed (reportedly none). It does not use a long-trial percentile filter.

ii. `kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]`; `if T < MIN_SAMPLES_PER_TRIAL: continue`.

iii. The notes justify excluding 309 `circle3` trials because reference `stim_id` never labels them and rely on running-only selection to remove long stops.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each spike file's `spks` planes; regions come from retinotopy `iarea`.

ii. `planes = d['spks']`; `iarea = np.asarray(d['iarea'])`.

iii. The data are already Suite2p-deconvolved and `iarea` is aligned with concatenated plane rows.

## 2-b. How is the `neural` data processed?

i. Planes are sliced and concatenated; zero-variance neurons are removed, at most 2,000 region-stratified neurons are sampled, and each is z-scored over retained session samples. Outputs are float32 trial matrices.

ii. `sel = stratified_subsample(region_all, max_neurons, rng)`; `Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / Xs.std(axis=1, keepdims=True)`.

iii. Z-scoring is justified by reference population-readout routines and scale control; the cap is justified by storage/training cost and decoder ablations.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes neurons outside V1/mHV/lHV/aHV, zero-variance neurons, and visual neurons beyond the 2,000-neuron stratified cap.

ii. `region_all[sd_all == 0] = -1`; `sel = stratified_subsample(region_all, max_neurons, rng)`.

iii. Visual-area exclusion follows reference code; dead cells cannot be z-scored, and subsampling allegedly avoids a 151.5-GB dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials begin at corridor entry but contain only running corridor frames; stopped frames disappear. Trials are variable length and unpadded.

ii. `keep = ft_corr & running`; `trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]`.

iii. The agent says temporal alignment does not require equal trial lengths and uses a running clock after deleting stopped samples.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One native imaging frame is one bin, about 314.85 ms; no temporal rebinning or positional interpolation occurs.

ii. `dt = float(np.median(np.diff(ft)) * SEC_PER_DAY)`; `time_bin_size=float(np.mean(dts) * 1000.0)`.

iii. Behavior is already on the neural-frame grid and native frames preserve temporal analysis.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It uses `SoundFr`, timestamps `ft`, and `ft_move > 0`.

ii. `tau_cue = np.interp(np.asarray(beh['SoundFr'], dtype=float)[:ntrials], grid, tau)`.

iii. The running mask supplies the agent's chosen time coordinate.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It constructs cumulative running time using median frame duration, interpolates SoundFr onto it, and subtracts sample time, positive before and negative after cue.

ii. `np.cumsum(running[:-1] * dt, out=tau[1:])`; `time_to_cue=(tau_cue[trial] - tau[kept]).astype(np.float32)`.

iii. The notes argue this eliminates wall-clock outliers caused by stops.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Values use exactly the retained frames and trial slices used for neural columns.

ii. `inp[0] = sb['time_to_cue'][a:b]`.

iii. Shared retained-frame indices guarantee column alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses session `datexp` and mouse `mname`.

ii. `d = date(*map(int, entry['datexp'].split('_')))`.

iii. Explicit `days` is available only for some records, so dates cover all sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It calculates calendar days since the mouse's first imaging session and broadcasts that scalar across the trial.

ii. `(date(*map(int, e['datexp'].split('_'))) - first[e['mname']]).days`; `inp[1] = sb['day']`.

iii. The notes define training age this way and report a 0–92 range.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses `StartFr`, `ft`, and `ft_move`.

ii. `tau_start = np.interp(start_fr_all, grid, tau)`.

iii. StartFr is corridor entry and the running mask defines the time base.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. StartFr is interpolated onto cumulative running time and subtracted from each retained sample.

ii. `time_since_start=(tau[kept] - tau_start[trial]).astype(np.float32)`.

iii. This intentionally omits stopped time.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same retained frame indices and `(a,b)` trial slices.

ii. `inp[2] = sb['time_since_start'][a:b]`.

iii. All streams share the retained-sample axis.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It uses `isRew` and `WallName`: the first rewarded trial identifies the rewarded wall, then all trials with that wall are labeled one.

ii. `rew_wall = str(wall_name[is_rew][0])`; `trial_rew = (wall_name == rew_wall).astype(np.float32)`.

iii. The agent cites reference `get_cat_id`; sessions without any `isRew` get zeros.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Wall equality becomes a binary per-trial value broadcast over time.

ii. `inp[3] = sb['trial_rew'][t]`.

iii. This is meant to encode rewarded-corridor identity rather than reward delivery.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses `WallName`, `UniqWalls`, and merged `stim_id` values.

ii. `wall2id = {w: s for w, s in zip(entry['uniq_walls'], entry['stim_id'])}`.

iii. The agent regards `stim_id` as a role-aligned taxonomy shared across mice.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Non-NaN IDs become seven integer classes and are broadcast across trials; unmapped `circle3` trials are dropped.

ii. `STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']`; `out[0] = int(sb['trial_stim'][t])`.

iii. The agent argues broad categories discard exemplar information and raw names differ across mice.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It comes from `LickFr`.

ii. `lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)`.

iii. LickFr is already in imaging-frame coordinates.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are floored and valid frames marked as a binary event.

ii. `lick_bin = np.zeros(n, dtype=bool)`; `lick_bin[lf] = True`.

iii. Frame k is treated as interval `[k,k+1)`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector uses the same retained frames and trial slices as neural data.

ii. `lick=lick_bin[kept].astype(np.int64)`; `out[1] = sb['lick'][a:b]`.

iii. Both streams share native imaging indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from per-frame `ft_Pos` in decimeters.

ii. `position=ft_pos[kept]`.

iii. The notes identify 0–40 dm as the textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 dm and cast to integer categories.

ii. `pos_bin = np.clip((sb['position'] / 10).astype(np.int64), 0, 3)`.

iii. This implements four equal one-meter bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Edges at 10, 20, and 30 dm yield classes 0–3, clipped at endpoints.

ii. `TEXTURE_LENGTH_DM = 40.0`; `N_POSITION_BINS = 4`.

iii. The task explicitly requests four one-meter bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is selected by `kept` and the same trial slices.

ii. `out[2] = pos_bin[a:b]`.

iii. Position and neural activity share the frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `ft_RunSpeed` at retained frames.

ii. `speed=ft_spd[kept]`.

iii. No other raw magnitude is used.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from all retained samples are concatenated, global 25/50/75 percentile thresholds computed, and `np.digitize` applied.

ii. `speed_edges = np.percentile(all_speed, [25, 50, 75])`; `spd_bin = np.digitize(sb['speed'], speed_edges)`.

iii. Global thresholds make class meaning consistent between sessions and sample/full runs.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global value edges (about 12.40, 25.28, 40.75) create four ordered categories.

ii. `speed_edges = np.percentile(all_speed, [25, 50, 75])`.

iii. Pooled class frequencies are approximately 25% each, though individual sessions are not balanced.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed uses the same retained frames and `(a,b)` trial slices.

ii. `out[3] = spd_bin[a:b]`.

iii. It is frame-synchronous with neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior is truncated to neural length; invalid trial indices and licks are excluded; unmapped stimuli are dropped; assertions catch conflicts/dimension mismatches; a boundary guard removes 21 artifacts.

ii. `n = min(len(ft_full), int(nframes_neural))`; `lf = lf[(lf >= 0) & (lf < n)]`; `assert len(region_all) == n_total`.

iii. The notes document observed edge cases and say truncation matches reference `[:nfr]` behavior.

## 12-a. What are the most time-consuming steps of the code?

i. Spike-plane loading/slicing, concatenation, and z-scoring dominate; sessions are parallelized.

ii. `d = np.load(path, allow_pickle=True).item()`; `X = np.concatenate(chunks, axis=0)`.

iii. The notes call neural I/O expensive and behavior loading cheap.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial array construction and small per-region sampling loops could be reduced, although variable-length lists still require iteration; most sample selection is vectorized.

ii. `for t, a, b in zip(uniq, starts, stops):`; `for r in range(len(BRAIN_REGIONS)):`.

iii. The agent says major earlier inefficiencies were removed and remaining loops are minor beside I/O.

## 12-c. What processing does the code repeat multiple times?

i. `build_session_behavior` runs once before neural loading to establish global speed edges and again per session after the true neural frame count is known.

ii. `sess_beh = OrderedDict((key, build_session_behavior(...)) ...)`; `sb = build_session_behavior(key, entry, day, nframes_neural=nframes)`.

iii. Repetition fixes global bins while ensuring the final pass truncates exactly to imaged frames.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Normal conversion mainly discards temporary full retained-neuron arrays and registry details. `--show-processing` additionally creates raw copies, concatenations, and figures used only for diagnostics.

ii. `raw = Xs.copy() if return_raw else None`; `del res['X'], res['raw']`.

iii. Diagnostics are optional; the agent views z-scoring/subsampling as decoder-oriented processing rather than discarded work.
