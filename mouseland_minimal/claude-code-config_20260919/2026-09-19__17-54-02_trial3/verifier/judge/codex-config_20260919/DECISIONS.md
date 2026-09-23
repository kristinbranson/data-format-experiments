# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy`, then every `Beh_<experiment>.npy`; it merges duplicate recording entries by `mname_datexp_blk`, loads each recording's plane-wise `spks`, and loads its retinotopy `iarea`.

ii. `exp_info = np.load(... 'Imaging_Exp_info.npy', allow_pickle=1).item()`; `Beh = np.load(... 'Beh_%s.npy' % exp_type, allow_pickle=1).item()`; `planes = np.load(... '%s_neural_data.npy' % key, allow_pickle=True).item()['spks']`; `iarea = np.load(fn, allow_pickle=True)['iarea']`

iii. The trajectory says duplicate behavior copies have identical arrays but complementary `stim_id` labels, so it reads behavior once per recording while merging those labels. It reports all 89 recordings from 19 mice.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname`; sessions are assigned an index in the first-seen `subjects` list.

ii. `if rec['mname'] not in subjects: subjects.append(rec['mname']); subject_idx.append(subjects.index(rec['mname']))`

iii. The mouse identifier is explicit in the experiment index; no inferred split is needed.

## 1-c. How are the data split into sessions?

i. A session is the unique tuple mouse, date, block. Repeated appearances across experiment types are merged, not duplicated.

ii. `key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk']); rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})`

iii. The trajectory verified recordings recur in multiple experiment types and chose one physical recording per session.

## 1-d. How are the data split into trials?

i. The agent iterates `ntrials`; frames are assigned using `ft_trInd`, restricted to textured-corridor frames and additionally to `ft_move > 0`.

ii. `keep = (rec['ft_CorrSpc'][:n] & (rec['ft_move'][:n] > 0) & ~np.isnan(trind))`; `f = frames[trials == t]`

iii. It defines a trial as one corridor traversal aligned to entry and says running-only frames match the paper's analysis selection and avoid long idle stretches.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if their stimulus has no merged canonical `stim_id`, or if no frame survives the corridor/running/imaging-length filters. There is no explicit long-trial percentile filter.

ii. `if stim_of_trial[t] < 0: ... continue`; `if len(f) == 0: continue`

iii. It says unlabeled `circle3` trials cannot share its chosen canonical scale (309 dropped), and running filtering makes traversals comparable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from plane-wise `spks`; `iarea` supplies region labels and selection.

ii. `planes = np.load(...).item()['spks']`; `iarea = np.load(fn, allow_pickle=True)['iarea']`

iii. The agent identifies `spks` as already-deconvolved Suite2p calcium activity.

## 2-b. How is the `neural` data processed?

i. Selected rows are copied plane by plane into float32, then trial columns are sliced. No normalization, padding, or rebinning is applied.

ii. `out = np.empty((len(sel), nframes), dtype=np.float32)`; `neural_s.append(spk[:, f])`

iii. It says the supplied signal is already deconvolved and describes it as “not normalised.” Plane-wise selection avoids materializing a full concatenation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only V1/mHV/lHV/aHV neurons are eligible, after which at most 2,000 are uniformly sampled per session with seed 0.

ii. `valid = np.nonzero(area_idx >= 0)[0]`; `valid = np.sort(rng.choice(valid, NEURONS_PER_SESSION, replace=False))`

iii. It argues all neurons would require hundreds of GB and 2,000 equals the decoder's exact-SVD cutoff, while random sampling preserves area proportions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials begin at corridor entry conceptually, but stored columns are the surviving same-trial corridor/running frames; idle frames can be omitted. Length remains variable.

ii. `f = frames[trials == t]`; `neural_s.append(spk[:, f])`; metadata: `'off_start': 0.0, 'off_end': None`

iii. The agent states corridor entry is trial start and retains native frames without padding.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are used, about 3.18 Hz or 315 ms. No rebinning occurs; metadata uses the mean session median interval.

ii. `dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)`; `'time_bin_size': float(np.mean(dts)) * 1000.0`

iii. It says imaging frames are already the common behavior/neural grid and therefore are the natural finest bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It uses per-trial `SoundFr`, selected frame indices, and a session `dt` derived from `ft`.

ii. `time_to_cue = (rec['SoundFr'][t] - f) * dt`

iii. The trajectory describes this as real elapsed seconds to the cue.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame difference is multiplied by median seconds/frame; values are positive before and negative after the cue.

ii. `time_to_cue = (rec['SoundFr'][t] - f) * dt`

iii. It explicitly chose signed “time to” and checked that clipping rare long tails barely affected accuracy, so left them unmodified.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the frame indices `f` used for neural columns.

ii. `input_s.append(np.stack([time_to_cue, ...]))`; `neural_s.append(spk[:, f])`

iii. The shared imaging-frame index provides alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses `datexp` and `mname`.

ii. `date = {k: datetime.date(*map(int, recordings[k]['datexp'].split('_'))) for k in keys}`

iii. It says date is the only training-time information in metadata.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It computes calendar days since that mouse's earliest recording, then broadcasts the value over the trial.

ii. `recordings[k]['day'] = float((date[k] - first[recordings[k]['mname']]).days)`; `day = np.full(len(f), rec['day'])`

iii. It intended day 0 for naive/before-learning sessions and elapsed days thereafter.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses `StartFr`, selected frame indices, and `dt` from `ft`.

ii. `time_since_start = (f - rec['StartFr'][t]) * dt`

iii. The agent identifies corridor entry (`StartFr`) as trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame offset from `StartFr` is multiplied by median seconds/frame.

ii. `time_since_start = (f - rec['StartFr'][t]) * dt`

iii. It aims to represent real elapsed seconds from corridor entry.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed at the identical selected indices `f` as neural data, though omitted idle frames create gaps in elapsed time.

ii. `time_since_start = (f - rec['StartFr'][t]) * dt`; `neural_s.append(spk[:, f])`

iii. Alignment is by shared frame number.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It derives the rewarded wall(s) from `WallName[isRew]`, then labels every trial whose `WallName` matches; sessions with no rewarded wall are all zero.

ii. `rewarded_walls = set(rec['WallName'][rec['isRew']])`; `rew_of_trial = np.isin(rec['WallName'], list(rewarded_walls))`

iii. It argues raw `isRew` indicates actual delivery in active sessions and would leak licking, whereas the requested variable is availability.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A Boolean wall match is converted to 1/0 and broadcast across all selected trial frames.

ii. `rew = np.full(len(f), 1.0 if rew_of_trial[t] else 0.0)`

iii. It verified at most one rewarded texture per session and no rewards for unsupervised/naive cohorts.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses `WallName`, `UniqWalls`, and merged `stim_id` values from duplicate behavior views.

ii. `for wall, cat in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], float)): ...`; `stim_of_trial = np.array([wallmap.get(w, -1) for w in rec['WallName']])`

iii. The agent sought the paper's cross-mouse canonical role labels.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. It produces seven canonical variant/role categories, broadcasts the trial value, and drops unmapped trials.

ii. `STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']`; `out[0] = stim_of_trial[t]`

iii. It claims these labels make different trained texture pairs comparable across mice.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses session `LickFr` and the spike-file frame count.

ii. `lick_fr = np.round(rec['LickFr']).astype(int)`

iii. Licks are directly supplied as imaging-frame positions.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame values are rounded, bounds-checked, and marked in a Boolean vector; multiple licks in a frame remain one.

ii. `lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]; licks[lick_fr] = True`

iii. The agent represents the requested binary per-frame target and safely excludes out-of-range events.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary vector is indexed by the same `f` as neural columns.

ii. `out[1] = licks[f]`; `neural_s.append(spk[:, f])`

iii. Alignment is by shared imaging frame.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It uses per-frame `ft_Pos`.

ii. `pos = rec['ft_Pos']; out[2] = np.digitize(pos[f], POSITION_EDGES)`

iii. The code comments that 40 position units equal the 4 m corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Native position is selected at trial frames and digitized into integer categories; no smoothing is done.

ii. `out[2] = np.digitize(pos[f], POSITION_EDGES)`

iii. This directly implements four requested spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 raw units, yielding nominal 0–1, 1–2, 2–3, and 3–4 m categories.

ii. `POSITION_EDGES = [10.0, 20.0, 30.0]`; `np.digitize(pos[f], POSITION_EDGES)`

iii. Equal 1 m bins follow from the 40-unit/4-m corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same `f` as neural activity.

ii. `out[2] = np.digitize(pos[f], POSITION_EDGES)`; `neural_s.append(spk[:, f])`

iii. Both streams are on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses per-frame `ft_RunSpeed` at frames accepted by `add_frame_selection`.

ii. `speeds.append(rec['ft_RunSpeed'][frames])`; `speed = rec['ft_RunSpeed']`

iii. Running speed is directly present in behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three percentile thresholds are computed globally over selected behavior frames from all sessions, then applied with `np.digitize`.

ii. `speed_edges = np.percentile(speeds, [25, 50, 75])`; `out[3] = np.digitize(speed[f], speed_edges)`

iii. It chose global edges so a category has the same physical meaning across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below/between/above global 25th, 50th, and 75th percentiles become categories 0–3.

ii. `speed_edges = np.percentile(speeds, [25, 50, 75])`; `np.digitize(speed[f], speed_edges)`

iii. The trajectory says these are dataset-wide quartile edges over included timepoints.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed at the identical `f` used for neural columns.

ii. `out[3] = np.digitize(speed[f], speed_edges)`; `neural_s.append(spk[:, f])`

iii. Alignment is by shared frame number.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior is truncated to available imaging frames; NaN trial indices and empty trials are removed; lick indices are bounds-checked; neuron/retinotopy sizes and rewarded-wall uniqueness are asserted. Unmapped stimuli are dropped.

ii. `n = min(nframes, len(rec['ft']))`; `& ~np.isnan(trind)`; `if len(f) == 0: continue`; `assert len(iarea) == nneurons`

iii. The comments cite the paper's same behavior-to-imaging truncation and treat absent canonical labels as unusable for its selected output scale.

## 12-a. What are the most time-consuming steps of the code?

i. Loading hundreds of GB of spike files and selecting/copying each session's neural rows dominate; full decoder validation is also lengthy but outside conversion.

ii. `planes = np.load(...).item()['spks']`; `out[m] = plane[sel[m] - offsets[p]]`

iii. The agent explicitly designed plane-wise loading around very large (20k–90k-neuron) recordings and reported the storage/runtime concern.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly selects `frames[trials == t]`; subject lookup uses repeated linear `list.index`; plane and record loops are largely necessitated by ragged/file-separated data.

ii. `for t in range(rec['ntrials']): f = frames[trials == t]`; `subjects.index(rec['mname'])`

iii. The trajectory gives no explicit vectorization discussion; these are apparent from code, though I/O dominates.

## 12-c. What processing does the code repeat multiple times?

i. Frame selection is run once over behavior-only lengths to calculate speed thresholds and again after spike loading with the true frame count. Trial masks are also recomputed once per trial.

ii. `frames, _ = add_frame_selection(rec, len(rec['ft']))`; later `frames, trials = add_frame_selection(rec, nframes)`

iii. The first pass supports global speed edges; the second enforces actual neural length. No explicit trajectory justification addresses the duplicate scan.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/stores extensive session metadata and experiment-type lists unused by decoder training, calculates `texture_length` and `reward_mode` for metadata, and reads all eligible neural rows before per-trial slicing. The deliberate 2,000-neuron selection limits the latter.

ii. `exp_types=sorted(set(rec['exp_types']))`; `texture_length=float(beh['Texture_Length'])`; `reward_mode=str(beh['Reward_Mode'])`

iii. These values aid provenance and sanity checking, but the trajectory does not claim the decoder consumes them.
