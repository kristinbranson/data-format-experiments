# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The master `Imaging_Exp_info.npy` index is loaded, physical recordings are deduplicated by mouse/date/block, and sorted. Required behavior files are cached by experiment type; spike planes and retinotopy are loaded per session.

ii. `exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()`; `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)`; `beh_cache[et] = load_beh(et)`

iii. The notes say 142 session-experiment entries describe 89 physical recordings, so each recording is included once and the first matching behavior view is used.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mname` values; sessions map into the sorted subject list.

ii. `subjects = sorted(set(s['mname'] for s in sessions))`; `subject_idx_list.append(subject_to_idx[session['mname']])`

iii. The mouse identifier is directly supplied by the experiment index; the reported result is 19 mice.

## 1-c. How are the data split into sessions?

i. A physical session is the tuple `(mname, datexp, blk)`. Duplicate appearances across experiment types are removed, keeping the first.

ii. `key = (db['mname'], db['datexp'], db['blk'])`; `if key in seen: continue`

iii. The notes explain that repeated entries are different analytical views of the same recording, not new recordings.

## 1-d. How are the data split into trials?

i. Each trial is sliced from integer-cast `StartFr[t]` (inclusive) to `EndFr[t]` (exclusive), covering the textured corridor and gray space.

ii. `StartFr = beh['StartFr'].astype(int)`; `EndFr = beh['EndFr'].astype(int)`; `neural = spk[:, start:end].astype(np.float16)`

iii. The notes explicitly choose `StartFr` to `EndFr` to preserve a contiguous, corridor-entry-aligned time series.

## 1-e. How are trials filtered based on quality controls?

i. Trials are skipped if their bounds are invalid/outside neural data, behavior streams end before the start, or fewer than two time points remain. No outlier-duration or movement filter is applied.

ii. `if start < 0 or end > n_total_frames or end <= start: ... continue`; `if n_tp < 2: ... continue`

iii. The notes state that all trials are used and stationary frames are deliberately retained for temporal continuity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` arrays for all imaging planes in each session's neural-data file. `iarea` supplies region labels.

ii. `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)`; `return dtrans['iarea']`

iii. The traces were already Suite2p-deconvolved, so the notes say no dF/F or further deconvolution is required.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, trial columns are sliced without temporal interpolation or normalization, and arrays are cast to float16.

ii. `neural = spk[:, start:end].astype(np.float16)`

iii. Float16 was chosen to reduce the very large output; the notes describe the values as raw deconvolved traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. All `iarea` values are mapped to V1, mHV, lHV, aHV, or an added `other` category.

ii. `BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']`; `idx = np.full(len(iarea), region_to_idx['other'], dtype=np.int64)`

iii. The notes deliberately retain all neurons so the decoder can learn relevance, despite acknowledging that `-1` and `7` are outside the four selected visual areas.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every variable and neural matrix starts at integer `StartFr`, treated as corridor entry, and ends at `EndFr`; trials retain variable lengths.

ii. `start = StartFr[t]`; `end = EndFr[t]`; `neural = spk[:, start:end]`

iii. Metadata records `off_start=0`, `off_end=None`, and corridor entry as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One original imaging frame is one bin; no rebinning is applied. A single median frame period from the first selected session is used globally (about 314.7 ms).

ii. `frame_period = float(np.nanmedian(np.diff(ft) * 24 * 3600))`; `'time_bin_size': frame_period * 1000`

iii. The notes report stable approximately 3.17 Hz acquisition and conclude the single value is adequate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the integer imaging-frame indices in the trial.

ii. `sound_fr = SoundFr[t]`; `frame_indices = np.arange(start, end, dtype=np.float64)`

iii. `SoundFr` is documented as the per-trial cue frame.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue-frame minus current-frame is multiplied by the global frame period, yielding seconds positive before and negative after the cue.

ii. `time_to_sound = (sound_fr - frame_indices) * frame_period`

iii. The notes and README explicitly describe that sign convention.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the identical `start:end` integer frame sequence and therefore has one value per neural column.

ii. `frame_indices = np.arange(start, end, dtype=np.float64)`

iii. Frame indices are the shared alignment grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and chronological `datexp` ordering among the retained sessions.

ii. `mouse_sessions[s['mname']].append((s['datexp'], i))`

iii. The notes call it the ordinal session index within each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are sorted by date per mouse, numbered from zero, and that number is broadcast across every bin of a trial.

ii. `for day_idx, (datexp, global_idx) in enumerate(sess_list): days[global_idx] = float(day_idx)`; `day = np.full(n_tp, training_day, dtype=np.float32)`

iii. This represents progress by recorded training session rather than elapsed calendar days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the trial's integer-cast `StartFr` and each integer frame index.

ii. `start = StartFr[t]`; `frame_indices = np.arange(start, end, dtype=np.float64)`

iii. The notes define trial start as corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Current-frame minus start-frame is multiplied by the global frame period, so the first saved bin is zero.

ii. `time_since_start = (frame_indices - start) * frame_period`

iii. It is a direct frame-to-seconds conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is generated over the exact same `start:end` indices as neural data.

ii. `neural = spk[:, start:end]`; `frame_indices = np.arange(start, end, dtype=np.float64)`

iii. Both arrays share length `end-start`.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from trial-level `isRew`.

ii. `isRew = beh['isRew']`; `rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)`

iii. The notes identify it as 1 for a rewarded corridor and 0 otherwise.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial flag is converted to float and broadcast over all trial bins.

ii. `rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)`

iii. No transformation beyond broadcasting is claimed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses per-trial `WallName`; the global category list is collected from each behavior object's `UniqWalls`.

ii. `for wn in beh['UniqWalls']: all_stim.add(str(wn))`; `stim_name = str(WallName[t])`

iii. The notes chose the wall names directly and report 15 distinct crop/swap labels.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Sorted wall names are mapped to integer indices and the per-trial index is broadcast across time. Crop and swap variants are not collapsed to four base textures.

ii. `stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}`; `stim_out = np.full(n_tp, stim_idx, dtype=np.int64)`

iii. The notes describe the 15 unique wall labels as the stimulus categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session-level fractional `LickFr` values.

ii. `lick_frs = beh['LickFr']`

iii. The notes planned a binary per-frame indicator; `LickTrind` was mentioned in planning but is not used by the code.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are rounded to nearest integers, filtered to `[start,end)`, shifted to trial-relative coordinates, clipped, and marked binary.

ii. `lick_frs_int = np.round(lick_frs).astype(int)`; `lick_vec[trial_lick_frs] = 1`

iii. The stated decision is to round fractional lick frames and create a binary vector.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Rounded absolute lick frames are selected within the same `start:end` interval and shifted by `start`, giving the neural trial's length.

ii. `mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)`; `trial_lick_frs = lick_frs_int[mask] - start_fr`

iii. The common imaging-frame grid supplies alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is taken from frame-wise `ft_Pos` in decimeters.

ii. `pos = ft_Pos[start:end]`

iii. The notes document a 0–60 dm corridor with 40 dm texture and 20 dm gray space.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The sliced raw positions are passed to `np.digitize`; no interpolation is applied.

ii. `pos_bin = digitize_position(pos)`

iii. The goal was four categorical position bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds at 10, 20, and 30 dm form bins 0–1 m, 1–2 m, 2–3 m, and 3 m+, with the last bin also containing all gray space.

ii. `bins = np.digitize(pos, [10, 20, 30])`; `POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']`

iii. The notes explicitly chose to fold 3–6 m into the final category.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sliced with the same absolute `start:end` frame bounds.

ii. `pos = ft_Pos[start:end]`

iii. Position is already sampled per imaging frame.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-wise `ft_RunSpeed` across all cached sessions.

ii. `speeds = beh['ft_RunSpeed']`; `all_speeds = np.concatenate(all_speeds)`

iii. The notes specify global speed quartiles across all frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global percentile values (25th, 50th, 75th) are computed, then each trial's raw speed values are digitized against them.

ii. `quartiles = np.percentile(all_speeds, [25, 50, 75])`; `speed_bin = digitize_speed(speed, speed_quartiles)`

iii. This was intended to create four quartile classes, though ties at zero make their populations unequal.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` assigns 0–3 using the global 25th/50th/75th percentile thresholds; values equal to a threshold go to the higher bin.

ii. `bins = np.digitize(speed, quartiles)`

iii. The notes acknowledge the resulting distribution is 9.2%, 40.9%, 25.0%, 24.9% because many values equal zero.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sliced with the identical `start:end` bounds before categorization.

ii. `speed = ft_RunSpeed[start:end]`

iii. `ft_RunSpeed` is already frame-aligned.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code asserts neuron/retinotopy counts match; invalid neural bounds, empty/inverted behavior-clipped intervals, and sub-two-frame trials are skipped. It suppresses warnings globally and does not explicitly repair NaNs or mismatched stimulus keys.

ii. `assert len(iarea) == n_neurons`; `end = min(end, len(ft_Pos), len(ft_RunSpeed))`; `if n_tp < 2: ... continue`

iii. Full conversion notes report no skipped trials, errors, or verification warnings, so no further handling was considered necessary.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating enormous spike files, copying every trial's neural slice into float16, retaining the full converted dataset in memory, and serializing the roughly 202 GB pickle dominate. The full run took about 33 minutes.

ii. `spk = np.concatenate(...)`; `neural = spk[:, start:end].astype(np.float16)`; `pickle.dump(data, f, protocol=5)`

iii. Notes estimate large per-session data and periodically report accumulated neural memory; float16 was adopted to control size.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial slicing/array construction, per-session collection of speeds/stimuli, per-area boolean mapping, and the loop creating day indices could be partly vectorized or grouped. Trial arrays are ragged, so the main per-trial loop cannot be eliminated cleanly.

ii. `for t in range(ntrials):`; `for area_val, region_name in AREA_MAP.items():`; `for s in sessions: all_speeds.append(...)`

iii. The agent did not document specific vectorization opportunities; its emphasis was memory and I/O.

## 12-c. What processing does the code repeat multiple times?

i. It traverses every behavior stream once to collect stimuli, again to concatenate all speeds, and later again session-by-session to make trials. It also creates `arange`, constant vectors, and scans all lick events anew for every trial.

ii. `get_all_stimuli(sessions, beh_cache)`; `compute_speed_quartiles(sessions, beh_cache)`; `for t in range(ntrials): ... make_lick_vector(beh, start, end)`

iii. No rationale for these repeated passes is documented; they simplify implementation and are small relative to neural data copying.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and bins behavior from gray-space frames, retains neurons labeled `other`, collects metadata such as VR constants, and computes timing/diagnostic summaries that the decoder need not use. Optional plotting computes mean neural traces. More importantly, gray-space samples are later given the same final position class as 3–4 m texture, reducing their distinct downstream meaning.

ii. `BRAIN_REGIONS = [..., 'other']`; `POS_BIN_EDGES = [0, 10, 20, 30, 60.01]`; `ax.plot(time_axis, neural.mean(axis=0), ...)`

iii. These are deliberate choices in the notes: gray space and all neurons were retained for continuity and decoder flexibility; plotting is optional validation.
