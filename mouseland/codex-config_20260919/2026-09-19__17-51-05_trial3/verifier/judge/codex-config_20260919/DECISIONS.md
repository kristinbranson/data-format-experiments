# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as the master manifest, deduplicates physical recordings, groups them by experiment group, loads each `Beh_<group>.npy`, and loads retinotopy and `spks` files per session. It asserts 89 sessions.

ii. `exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()`; `behavior_dict = np.load(BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True).item()`; `raw = np.load(spk_path, allow_pickle=True).item()`

iii. The notes justify this as matching the released loaders and avoiding repeated loading of large behavior dictionaries and neural sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is `db["mname"]`; unique names are sorted and each session receives an integer `subject_idx`.

ii. `subjects = sorted({r["db"]["mname"] for r in records})`; `subject_idx.append(subject_lookup[record["db"]["mname"]])`

iii. The agent notes that the experiment index explicitly provides mouse identity and verifies 19 subjects.

## 1-c. How are the data split into sessions?

i. A session ID combines mouse, date, and block. Duplicate index appearances are removed by a `seen` set; behavior keys additionally include `stimtype` when present.

ii. `sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"`; `if sid in seen: continue`

iii. The notes describe the duplicates as analysis aliases of the same physical recording and report 89 unique sessions.

## 1-d. How are the data split into trials?

i. Frames are grouped by integer `ft_trInd`, but only frames satisfying finite trial ID, `ft_CorrSpc`, and `ft_move > 0` are assigned to trials. Every source trial is retained if validation succeeds.

ii. `valid = np.isfinite(trial_id) & np.asarray(beh["ft_CorrSpc"], dtype=bool) & (np.asarray(beh["ft_move"]) > 0)`; `frames = [selected[selected_trial == trial] for trial in range(ntrials)]`

iii. The agent says this matches the movement/corridor mask used in the paper's main neural-response analyses and avoids stationary reward-consumption periods.

## 1-e. How are trials filtered based on quality controls?

i. No trials are intentionally filtered. Instead, conversion aborts if any trial has fewer than two selected moving frames; all 38,110 trials passed.

ii. `if np.any(lengths < 2): ... raise ValueError(...)`

iii. The notes justify retaining every imaging trial because each has valid aligned running/corridor frames and every session has many trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the per-plane arrays in `raw["spks"]`; neuron regions and the keep mask come from retinotopy `iarea`.

ii. `planes = raw["spks"]`; `iarea = np.asarray(ret["iarea"])`

iii. The agent identifies `spks` as released Suite2p non-negative deconvolved fluorescence and states no dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. Kept cells are copied plane-by-plane into a float32 session matrix. Each trial then copies columns at its selected frame indices; no normalization, interpolation, or temporal rebinning is applied.

ii. `activity[kept_offset : kept_offset + nkeep] = plane[plane_keep]`; `ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)`

iii. The notes say raw deconvolved activity is appropriate for the decoder and float32 preserves native values while allowing the session matrix to be released.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells with `iarea` codes mapping to V1, mHV, lHV, or aHV are kept; codes -1 and 7 are excluded. Finite values, matching frame counts, and cell-count agreement are checked. No stimulus-selectivity filter is used.

ii. `keep = region >= 0`; `if not np.all(np.isfinite(activity)): raise ValueError(...)`

iii. The agent argues area filtering matches reference visual-cortex curation, while d-prime selection would leak the visual-category target.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are conceptually aligned to corridor entry, but only moving textured-corridor frames are stored. Arrays begin near trial start, remain variable length, and can contain temporal gaps where stationary frames were removed.

ii. `ntrial = np.array(activity[:, frames], ...)`; metadata: `"temporal_alignment_event": "trial start (entry into the 4-m textured corridor)"` and `"timepoints_may_have_gaps": True`

iii. The agent says exact timestamps preserve elapsed time across omitted pauses, and variable lengths avoid padding or truncation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The nominal native imaging resolution is 3.17 Hz, or 315.457 ms per frame. No resampling/rebinning is done, although removal of stationary frames makes adjacent stored columns sometimes farther apart than one nominal bin.

ii. `TIME_BIN_MS = 1000.0 / IMAGING_RATE_HZ`; `"time_bin_size": float(TIME_BIN_MS)`

iii. The notes say native frames are the appropriate temporal samples and spatial interpolation would violate the temporal decoder format.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and per-frame timestamp `ft`.

ii. `ft = np.asarray(beh["ft"])[frames]`; `time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0`

iii. The notes prefer released exact timestamps so omitted stationary periods do not compress time.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue time minus frame time is converted from MATLAB-day units to seconds, yielding positive values before and negative values after the cue.

ii. `time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0`

iii. The documentation explicitly defines it as signed seconds until cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed at exactly the same selected `frames` used as neural columns.

ii. `ft = np.asarray(beh["ft"])[frames]`; `ntrial = np.array(activity[:, frames], ...)`

iii. Independent raw-file checks reportedly confirmed equality for sampled trials.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses the experiment-index descriptor's `days` field, falling back to `sess#`.

ii. `day_value = float(db["days"])` or `day_value = float(db["sess#"])`

iii. The agent treats these released fields as the native training-day/stage annotation.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The selected scalar is broadcast unchanged over every retained timepoint in the trial.

ii. `np.full(T, record["day_value"])`

iii. The notes report a full-data range of 0–15 and preserve the source field in metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-frame `ft` and per-trial `Trial_start_time`.

ii. `time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0`

iii. The agent identifies corridor entry as trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start time is subtracted from each selected timestamp and converted from days to seconds.

ii. `time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0`

iii. Exact timestamps are retained so elapsed time includes pauses even though pause frames are omitted.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same frame-index array as neural data, so each stored value corresponds to its neural column, but the stored sequence may have gaps.

ii. `ft = np.asarray(beh["ft"])[frames]`; `ntrial = np.array(activity[:, frames], ...)`

iii. The notes state all-session frame-index and sampled numeric checks passed.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew` identifies the unique rewarded `WallName` in a session; availability is then 1 for every trial with that wall, not merely trials with reward delivery.

ii. `reward_walls = np.unique(walls[np.asarray(beh["isRew"], dtype=bool)])`; `reward_available = (walls == rw_wall).astype(np.float32)`

iii. The agent distinguishes availability from delivery and reports 4,446 availability-positive versus 4,336 rewarded trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. It validates that at most one wall is rewarded, compares every trial wall to it, uses all zeroes if none exists, and broadcasts the result across trial time.

ii. `np.full(T, reward_available[trial])`

iii. The notes argue this better represents whether reward is available in the corridor than directly copying delivery flags.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from per-trial `WallName`.

ii. `stim = stimulus_class(str(walls[trial]))`

iii. The notes say wall names encode the native texture family.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. String prefixes `circle`, `leaf`, `rock`, and `wood` map to codes 0–3; the code is broadcast through the trial.

ii. `STIMULUS_PREFIX_TO_CLASS = {"circle": 0, "leaf": 1, "rock": 2, "wood": 3}`; `np.full(T, stim, dtype=np.int16)`

iii. The agent groups crops and swapped variants into the four requested broad categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived jointly from `LickFr` and `LickTrind`.

ii. `lick_fr = np.asarray(beh["LickFr"])`; `lick_tr = np.asarray(beh["LickTrind"])`

iii. The agent uses both fields to associate finite lick-frame events with their trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Nonfinite events are removed; lick frames for the current trial are selected; each retained frame is labeled 1 if present in that event list, otherwise 0.

ii. `trial_lick_frames = lick_fr_int[lick_tr_int == trial]`; `lick = np.isin(frames, trial_lick_frames).astype(np.int16)`

iii. The notes describe this as exact native-frame event binning and report independent equality checks.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary vector is evaluated on the exact selected neural frame indices.

ii. `lick = np.isin(frames, trial_lick_frames)`; `ntrial = np.array(activity[:, frames], ...)`

iii. The agent reports raw alignment spot checks across multiple sessions and trials.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position comes from frame-level `ft_Pos` at selected frames.

ii. `position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)`

iii. The agent notes the raw position scale has 10 units per metre.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Raw position is divided by 10 and floored to metres, then clipped and cast to integer.

ii. `position = np.clip(position, 0, 3).astype(np.int16)`

iii. The notes say this directly implements four equal 1-m bins over the textured 4-m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are [0,10), [10,20), [20,30), and [30,40) raw units, labeled 0–3; clipping protects the bounds.

ii. `np.floor(... / 10.0)` followed by `np.clip(position, 0, 3)`

iii. The processing plots and audit reportedly confirm approximately equal spatial occupancy and positions within [0,40).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same `frames` used for neural columns.

ii. `np.asarray(beh["ft_Pos"])[frames]`; `activity[:, frames]`

iii. The notes report exact all-session trial-length and sampled value checks.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Speed comes from frame-level `ft_RunSpeed` on the selected moving corridor frames.

ii. `speed = np.asarray(beh["ft_RunSpeed"])[frames]`

iii. The notes say it uses the native frame-aligned speed without interpolation.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior-only prepass pools speeds from every retained observation in the conversion and calculates global 25th, 50th, and 75th percentiles; each trial is digitized with those edges.

ii. `thresholds = np.quantile(speeds, [0.25, 0.50, 0.75])`; `speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)`

iii. The agent argues global edges give consistent class meaning across a shared decoder and approximately equal full-dataset class counts.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below the three global edges receive labels 0–3 via `np.digitize`; full-data edges are approximately 12.4224, 25.3526, and 40.8546.

ii. `np.digitize(speed, speed_edges, right=False)`

iii. The notes report each class occupies essentially 25% of selected observations and require strictly increasing thresholds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed and categorized at the same selected frame indices as neural activity.

ii. `speed = np.asarray(beh["ft_RunSpeed"])[frames]`; `activity[:, frames]`

iii. The notes report independent raw-value checks.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script validates shapes, geometry, finite values, trial IDs, region codes, neural/retinotopy counts, and selected indices, generally raising errors instead of imputing. Nonfinite lick events are discarded, and a unique fallback behavior key is accepted. It does not trim behavior to neural length; it rejects selected indices beyond neural data.

ii. `finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)`; `if max_selected >= nframes_neural: raise ValueError(...)`; `matches = [k for k in behavior_dict if k.startswith(record["session_id"])]`

iii. The notes emphasize fail-fast validation and state that all 89 sessions passed.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/filtering multi-GB neural files, copying retained trial matrices, and serializing the 141-GiB pickle dominate. Full in-memory conversion took about 1,394 s and writing about 123 s.

ii. `raw = np.load(spk_path, allow_pickle=True).item()`; `ntrial = np.array(activity[:, frames], ..., copy=True)`; `pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)`

iii. The notes attribute the unexpectedly long full run to cumulative neural reads and retained float32 copies.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial construction, per-trial lick filtering, and `frames = [selected[selected_trial == trial] ...]` could be grouped/vectorized, although variable-length output still requires assembly. Plane and session loops are structurally necessary for memory control.

ii. `for trial, frames in enumerate(frames_by_trial):`; `frames = [selected[selected_trial == trial] for trial in range(ntrials)]`

iii. The notes characterize direct indexing/stacking as already vectorized within trials and prioritize bounded memory over parallel loading.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files and frame-selection/validation are processed twice: once in `speed_quartile_prepass` and again during conversion. Arrays such as `ft`, `ft_Pos`, and `ft_RunSpeed` are repeatedly wrapped with `np.asarray` inside trial loops.

ii. `frames = selected_frames_by_trial(beh, sid)` appears in the prepass and conversion; repeated examples include `np.asarray(beh["ft_Pos"])[frames]`.

iii. The agent accepts the small behavior-only prepass to establish global thresholds before costly neural I/O; it reports the prepass as inexpensive.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The prepass computes audit summaries not required by training; conversion loads/returns full `iarea` for metadata/optional plots; repeated validation and `gc.collect()` add overhead. Optional plots compute several display-only statistics when requested.

ii. `summary = {...}`; `return activity, region_idx, iarea`; `gc.collect()`; `if args.show_processing ... make_processing_plot(...)`

iii. The notes justify these costs as sanity checks, audit metadata, visualization, and memory management; plotting is disabled for the full run.
