# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy`, deduplicates recordings by mouse/date/block, caches the relevant `Beh_<type>.npy` dictionaries, and loads each session's neural and retinotopy files. It makes a full first pass through behavior for speed thresholds, then a second pass to build the dataset.

ii. `exp_info = np.load(os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()`; `all_sessions = sorted(session_map.keys())`; `beh_cache[exp_type] = np.load(beh_path, allow_pickle=True).item()`; `spk = load_spk(...)`; `iarea = load_retino(...)`.

iii. The trajectory says the master experiment index describes the 89 recordings and notes duplicate listings across experiment types. The agent chose the duplicate entry with the most non-NaN stimuli, intending to retain the most complete behavior, and used caching to avoid repeated behavior-file reads.

## 1-b. How are the data split into subjects?

i. A subject is the mouse-name prefix of the session key. Unique names are sorted and each retained session gets the corresponding integer index.

ii. `all_mice = sorted(set(k.split('_')[0] for k in all_sessions))`; `subject_idx_list.append(mouse_to_idx[mname])`.

iii. The trajectory identifies `mname` as the mouse ID and uses it consistently with the paper's 19-mouse description.

## 1-c. How are the data split into sessions?

i. A session is uniquely keyed by `mname_datexp_blk`; duplicate appearances in experiment groups are collapsed, preferring the entry with the greatest stimulus count.

ii. `key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'`; `if new_nstim > old_nstim: session_map[key] = (exp_type, beh_key, ndb)`.

iii. The trajectory concluded that mouse/date/block names the spike recording and that duplicates in the experiment index should not create duplicate sessions.

## 1-d. How are the data split into trials?

i. The agent uses `beh['ntrials']` and spatially interpolates the concatenated running frames into an array of exactly 60 bins per declared trial; trial `t` is slice `interp_spk[:, t, :]`.

ii. `interp_spk = get_interpPos_spk(..., ntrials, n_bins=int(CL), lengths=CL)`; `for trial in range(ntrials): neural_trial = interp_spk[:, trial, :]`.

iii. It reasoned that the reference analysis represents each trial on 60 one-decimeter spatial samples (40 textured plus 20 gray) and that accumulated position separates successive trials.

## 1-e. How are trials filtered based on quality controls?

i. Individual trials are not filtered. A session is skipped only if it declares fewer than two trials, neural/retinotopy loading fails, or fewer than ten area-valid neurons remain.

ii. `if ntrials < 2: ... continue`; the later loop is `for trial in range(ntrials):` with no trial mask.

iii. The trajectory did not identify or justify a trial-duration/outlier filter; it treated all declared trials as usable after spatial interpolation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane `spks` arrays in each neural-data file. `iarea` from the retinotopy file supplies neuron inclusion and region labels; behavior `ft_move` and `ft_PosCum` supply the interpolation coordinate.

ii. `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)`; `VRmove = beh['ft_move'][:nfr] > 0`; `ft_AcumPos = beh['ft_PosCum'][:nfr]`.

iii. The agent identified `spks` as Suite2p deconvolved activity and followed the paper statement that analyses considered running timepoints.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, non-running frames removed, and every neuron's trace is linearly interpolated against cumulative position to 60 spatial bins per trial. Results are stored as float32 and non-finite values are replaced by zero.

ii. `Model_ = interpolate.interp1d(vind, v, fill_value='extrapolate')`; `spk_filtered[:, VRmove]`; `neural_trial = interp_spk[:, trial, :].astype(np.float32)`; `np.nan_to_num(...)`.

iii. The trajectory explicitly chose the paper repository's spatial interpolation helper and running-frame restriction, viewing one-decimeter bins as the analysis representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea` -1 or 7 are removed; retained codes are assigned to V1, mHV, lHV, or aHV. Sessions with fewer than ten retained neurons are skipped.

ii. `valid_neuron_mask = (iarea != -1) & (iarea != 7)`; `if n_valid < 10: ... continue`; `spk_filtered = spk[valid_neuron_mask]`.

iii. The agent says this matches the reference area mapping and excludes neurons outside visual cortex. The ten-neuron session threshold was a defensive decoder choice, not supported by a cited paper criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every fixed-length spatial trial begins at accumulated-position bin 0, interpreted as corridor entry, and extends through 60 dm including gray space.

ii. `linPos = np.arange(0, new_shape[0], 1 / new_shape[1])`; `np.reshape(..., (int(new_shape[0]), int(new_shape[1])))`; metadata says `'temporal_alignment_event': 'Trial start (corridor entry)'`.

iii. The agent equated the start of each interpolated spatial traversal with trial start and relied on constant virtual-corridor speed to interpret spatial bins temporally.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It reports 166.67 ms per bin, inferred from 1 dm at 6 dm/s. The original 3.17-Hz imaging frames are resampled by spatial interpolation to 60 fixed bins.

ii. `time_bin_sec = 1.0 / 6.0`; `'time_bin_size': time_bin_sec * 1000`; `get_interpPos_spk(...)`.

iii. The trajectory justified this from the paper's 60 cm/s constant VR motion while running and its selected one-decimeter spatial grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundPos` and the synthetic positions `0..59`.

ii. `SoundPos = beh['SoundPos']`; `positions = np.arange(n_bins)`; `sound_pos = SoundPos[trial]`.

iii. The agent chose position rather than recorded frame timestamps because all output had been placed on its fixed spatial grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Sound position minus bin position is multiplied by 1/6 second per dm, producing positive values before and negative values after the cue.

ii. `time_to_cue = (sound_pos - positions) * time_bin_sec`.

iii. The trajectory states this sign convention and derives the conversion from constant VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It has the same 60 synthetic spatial bins as interpolated neural activity.

ii. `input_trial = np.stack([time_to_cue.astype(np.float32), ...], axis=0)` alongside `neural_trial = interp_spk[:, trial, :]`.

iii. The agent considered common position-bin indices sufficient alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and the calendar date string `datexp` in every experiment-index entry.

ii. `date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')`; `mouse_dates[ndb['mname']].append(date)`.

iii. The trajectory interpreted “day” as elapsed calendar days since that mouse's earliest indexed recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The earliest calendar date for each mouse is subtracted from the session date; that integer day offset is repeated across all 60 bins.

ii. `session_days[key] = (date - mouse_first_date[mname]).days`; `day_array = np.full(n_bins, day, dtype=np.float32)`.

iii. The agent described this as a relative training-day measure beginning at zero.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from recorded timestamps or `StartFr`; it is synthesized from spatial-bin indices and assumed VR speed.

ii. `positions = np.arange(n_bins)`; `time_since_start = positions * time_bin_sec`.

iii. The agent reasoned that one spatial bin always corresponds to 1/6 second while VR is moving.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each index 0 through 59 is multiplied by 1/6 second, yielding a uniform 0-to-9.83-second ramp.

ii. `time_since_start = positions * time_bin_sec`.

iii. The trajectory treats corridor entry as zero and uses the synthetic fixed-resolution time axis.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The 60-value ramp shares indices with the 60 interpolated neural bins.

ii. `input_trial = np.stack([... time_since_start.astype(np.float32), ...], axis=0)`.

iii. The agent considered the shared spatial grid to be temporal alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` flag.

ii. `isRew = beh['isRew']`; `float(isRew[trial])`.

iii. The trajectory identifies `isRew` as the rewarded-corridor indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is cast to float and broadcast over 60 bins.

ii. `reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)`.

iii. The agent treated it as a per-trial contextual input that must share the time dimension.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from the trial's raw `WallName` string.

ii. `WallName = beh['WallName']`; `stim_idx = stim_to_idx[str(WallName[trial])]`.

iii. The trajectory chose `WallName` because it is populated in swap sessions where another stimulus field may be masked.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct raw wall name across behavior files is sorted and assigned its own category, then broadcast over the trial. Variants are not collapsed to circle/leaf/rock/wood.

ii. `all_stim_names = sorted(all_stim_names_set)`; `stim_category = np.full(n_bins, stim_idx, dtype=np.int64)`.

iii. The trajectory aimed to preserve distinctions such as crops and swaps as visual categories rather than imposing a base-texture map.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from each lick's `LickPos` and `LickTrind`.

ii. `make_lick_spatial_bins(beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL))`.

iii. The agent selected spatial lick fields to match the position-resampled neural grid.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick position is floored and clipped to a bin; that trial/bin is set to one. Multiple licks in a bin remain one.

ii. `bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))`; `lick_arr[tr, bin_idx] = 1`.

iii. The trajectory describes the requested output as binary occupancy per spatial bin.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are assigned to the same nominal 60 position bins used for neural interpolation, but not aligned through common imaging-frame indices.

ii. `lick_trial = lick_spatial[trial, :]`; `neural_trial = interp_spk[:, trial, :]`.

iii. The agent relied on shared spatial coordinates for alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is synthesized from bin indices `0..59`, rather than using per-frame `ft_Pos`.

ii. `positions = np.arange(n_bins)`; `pos_category = np.digitize(positions, position_bins_edges)`.

iii. The agent reasoned that the interpolated grid itself encodes position exactly.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The synthetic position vector is digitized at 10, 20, 30, and 40 dm and copied to every trial.

ii. `position_bins_edges = [10, 20, 30, 40]`; `pos_trial = pos_category.copy()`.

iii. It intended four one-meter texture bins plus a separate gray-space class.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Values 0–9, 10–19, 20–29, and 30–39 map to classes 0–3; 40–59 map to a fifth `gray_space` class.

ii. `position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']`; `np.digitize(positions, [10, 20, 30, 40])`.

iii. The trajectory explicitly decided to retain the paper's 2 m gray interval as an additional class.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Category index `j` describes the nominal position of neural interpolation bin `j`.

ii. Both `pos_trial` and `neural_trial` have length 60 and are stacked/stored for the same trial.

iii. Alignment is by the common synthetic spatial grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from precomputed behavioral `run_pos` arrays for every session/trial/spatial position.

ii. `run_pos = beh['run_pos']`; `run_pos_all.append(beh['run_pos'])`.

iii. The agent selected the already position-binned behavior stream to match its neural representation.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All finite `run_pos` values across all sessions are pooled to obtain three percentile thresholds; each value is then digitized into 0–3.

ii. `edges = np.percentile(all_speeds, [25, 50, 75])`; `result = np.digitize(run_pos, edges)`.

iii. The trajectory interpreted “each corresponding to 25% of the data” as global value-based quartile thresholds.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` applies the global 25th, 50th, and 75th percentile values, with ties assigned together.

ii. `speed_edges = discretize_speed_quartiles(run_pos_all)`; `speed_bins = speed_to_bins(run_pos, speed_edges)`.

iii. The agent intended four globally comparable quartile-labeled categories.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The trial's 60-element `run_pos` row is assumed to use the same spatial grid as the 60 neural bins.

ii. `speed_trial = speed_bins[trial, :].astype(np.int64)`.

iii. The trajectory relied on identical trial and position indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and position-related behavior arrays are truncated to the number of neural frames; neural NaN/Inf values after interpolation become zero. Missing neural or retinotopy files skip a whole session. NaNs are removed only when estimating speed thresholds; other missing behavioral fields are not handled.

ii. `VRmove = beh['ft_move'][:nfr] > 0`; `np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)`; `except Exception ... continue`; `all_speeds = all_speeds[~np.isnan(all_speeds)]`.

iii. The trajectory frames these as defensive safeguards so a malformed session/value does not abort conversion, but gives no data-specific evidence for zero imputation or broad session skipping.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating very large spike files and per-neuron interpolation over all sessions dominate. The full first behavior pass and stimulus scan add overhead.

ii. `np.concatenate([...['spks']], 0)`; `for s in range(raw_spk.shape[0]): spk_resh.append(... interp_value(...))`.

iii. The trajectory anticipated a multi-gigabyte result and used chunking/caching, implicitly recognizing neural I/O and interpolation as the expensive work.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop interpolating one neuron at a time, lick-event loop, trial construction loop, area-assignment loop, and repeated behavior/stimulus scans could potentially be vectorized or grouped.

ii. `for s in range(raw_spk.shape[0]): ...`; `for i in range(len(lick_pos)): ...`; `for trial in range(ntrials): ...`.

iii. The trajectory does not explicitly discuss vectorization; it instead copied the repository's neuron-wise interpolation pattern and used a nominal 10,000-neuron chunk loop.

## 12-c. What processing does the code repeat multiple times?

i. Behavior is traversed once for speed data, again across every behavior file for stimulus names, and again to build sessions. Duplicate experiment-index entries are also repeatedly parsed for mapping and training dates.

ii. `for sess_key in all_sessions_full: ... run_pos_all.append(...)`; `for exp_type_key in exp_info.keys(): ... np.load(beh_path, ...)`; later `for sess_idx, sess_key in enumerate(all_sessions):`.

iii. The agent deliberately used two passes so global speed and label definitions were known before outputs were encoded; it did not justify the separate uncached stimulus-file scan.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `n_position_categories` is computed but unused; `data_root` passed to `get_session_beh_mapping` is unused; interpolation extrapolates gray-space neural values that should not be part of the requested four corridor-position bins; metadata/summary work and optional sample copying do not affect decoder arrays.

ii. `n_position_categories = 5`; `def get_session_beh_mapping(exp_info, data_root='data/beh')`; `fill_value='extrapolate'`; the optional `sample_data = {...}` block.

iii. The trajectory does not identify these as unnecessary; the gray space and extensive metadata were intentional attempts to mirror the repository and document the conversion.
