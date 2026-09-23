# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All `Beh_*.npy` dictionaries are catalogued, alias keys are collapsed to physical recordings, `Imaging_Exp_info.npy` is loaded separately, and spike/retinotopy inventories must match before each session is loaded.

ii. `for path in sorted((DATA_ROOT / "beh").glob("Beh_*.npy")): loaded = np.load(path, allow_pickle=True).item()`; `spk_bases = {p.name.removesuffix("_neural_data.npy") ...}`

iii. The notes say aliases are alternate analyses, not recordings, and report exact four-way correspondence for 89 sessions.

## 1-b. How are the data split into subjects?

i. The mouse-prefix of each physical session ID defines subject; sorted unique prefixes define `subjects` and sessionwise indices.

ii. `subjects = sorted({base.split("_")[0] for base in bases})`; `subject_idx = np.asarray([subject_lookup[base.split("_")[0]] for base in bases])`

iii. Session IDs natively encode mouse/date/block; 19 mice were found.

## 1-c. How are the data split into sessions?

i. A session is `mouse_date_block`; terminal `_swap1/_swap2` aliases are removed and duplicate behavior content is validated.

ii. `return re.sub(r"_swap[12]$", "", session_key)`; `base = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"`

iii. Swap keys are documented as alternate views of the same physical recording.

## 1-d. How are the data split into trials?

i. For each integer trial, frames must match `ft_trInd`, lie in `ft_CorrSpc`, and have `ft_move > 0`; stationary frames are removed, so trial samples can be non-contiguous.

ii. `valid = np.isfinite(trial_stamp) & corridor & movement`; `frames = np.flatnonzero(valid & (trial_stamp == trial))`

iii. The agent claims this matches the paper's running-timepoint curation and preserves gaps in timestamp inputs.

## 1-e. How are trials filtered based on quality controls?

i. Only trials with no remaining curated frames are excluded; sessions below two trials fail. There is no duration-outlier filter, and all 38,110 full-data trials survived.

ii. `if frames.size == 0: excluded_trials.append(trial); continue`; `if len(neural_trials) < 2: raise ValueError(...)`

iii. The justification is that moving-frame curation addresses stationary periods directly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values use plane-wise `spks`; neuron inclusion/regions use retinotopy `iarea`.

ii. `planes = neural_dict["spks"]`; `iarea = np.asarray(retino["iarea"])`

iii. The notes identify `spks` as supplied Suite2p deconvolved activity.

## 2-b. How is the `neural` data processed?

i. Mapped rows are copied plane-by-plane into float32, then curated trial columns are copied. There is no dF/F, normalization, deconvolution, or resampling.

ii. `selected = np.empty((int(mapped.sum()), nframes), dtype=np.float32)`; `neural = spk[:, frames].copy()`

iii. The agent preserves supplied float32 values because the signal is already deconvolved.

## 2-c. How is the `neural` data filtered based on quality controls?

i. `iarea` 8, 0/1/2/9, 5/6, and 3/4 map to V1/mHV/lHV/aHV; -1/7 are removed. Nonfinite values fail; no activity/selectivity filter is used.

ii. `mapped = np.isin(iarea, [0,1,2,3,4,5,6,8,9])`; `if not np.all(np.isfinite(neural)): raise ValueError(...)`

iii. The notes say cells are already Suite2p-classified and target-selectivity filtering risks leakage.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials begin in the textured corridor (corridor entry), retain native moving frames only, and use no padding/fixed window; gaps may separate adjacent columns.

ii. `frames = np.flatnonzero(valid & (trial_stamp == trial))`; `neural = spk[:, frames].copy()`

iii. The agent says original timestamps preserve elapsed time through omitted stationary periods.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning occurs; retained columns are native 3.17-Hz frames (315.46 ms nominal), though removal creates nonuniform elapsed gaps.

ii. `FS_HZ = 3.17`; `MS_PER_FRAME = 1000.0 / FS_HZ`

iii. Metadata explicitly distinguishes native resolution from curated gaps.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Per-trial `SoundTime` and per-frame `ft`.

ii. `time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS`

iii. Direct timestamps were preferred to fractional `SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame time is subtracted from cue time, converted from days to seconds, and cast into float32 inputs; positive means before cue.

ii. `DAY_TO_SECONDS = 86_400.0`; formula above.

iii. Units and sign are documented in metadata.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `ft[frames]` with exactly the neural indices.

ii. `neural = spk[:, frames].copy()` and `... ft[frames] ...`

iii. Strict validation requires identical time lengths.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Registry `days` is used if present, otherwise `sess#`; the minimum across aliases is chosen.

ii. `training_day[base] = float(min(explicit_days))` / `float(min(session_indices))`

iii. The agent says explicit metadata avoids counting aliases as physical days.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The numeric day/stage is repeated across each retained trial frame.

ii. `np.full(frames.size, day, dtype=np.float64)`

iii. The heterogeneous `days`/`sess#` rule is disclosed in metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Per-frame `ft` and per-trial `Trial_start_time`.

ii. `time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS`

iii. `Trial_start_time` is treated as the direct corridor-entry timestamp.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Start is subtracted from frame time, days are converted to seconds, and values enter the float32 matrix.

ii. The preceding formula implements the processing.

iii. Actual timestamps retain gaps left by frame curation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses exactly the selected neural frame indices.

ii. Both expressions index with `frames`.

iii. Shape and independent raw spot checks reportedly passed.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Per-trial `isRew`.

ii. `np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64)`

iii. Notes clarify this denotes rewarded corridor, not observed delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. It is Boolean-coded 0/1 and broadcast over trial frames.

ii. Same `np.full(...)` snippet.

iii. Broadcasting ensures the uniform `(4,T)` layout.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Per-trial `WallName`.

ii. `category = stimulus_category(str(beh["WallName"][trial]))`

iii. Direct labels are invariant across swap aliases.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The leading alphabetic name is lowercased and mapped from circle/leaf/rock/wood to 0–3, then broadcast.

ii. `match = re.match(r"([A-Za-z]+)", str(name))`; `decoder_output[0] = category`

iii. This pools exemplars and swaps into four physical categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Session lick-frame indices `LickFr`.

ii. `lick_float = np.asarray(beh["LickFr"], dtype=np.float64)`

iii. The notes call integer lick-frame events reference-compatible.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite indices are integer-truncated, bounds-filtered, and set in a binary frame vector.

ii. `lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)`; `lick_global[lick_frames] = 1`

iii. This safely drops malformed/post-imaging events; multiple licks remain binary.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is indexed by the same neural `frames`.

ii. `decoder_output[1] = lick_global[frames]`

iii. Exact alignment was reportedly spot-checked and plotted.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Per-frame `ft_Pos`.

ii. `position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]`

iii. Source units are documented as decimeters.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Divide by 10, floor, clip to 0–3, cast int8.

ii. `position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)`

iii. This creates four requested 1-m categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Boundaries are 10/20/30 dm: 0–1, 1–2, 2–3, 3–4 m.

ii. `np.clip(np.floor(position / 10.0), 0, 3)`

iii. Near-quarter occupancy was used as a sanity check.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position and neural activity share `frames`.

ii. `...ft_Pos... [frames]`; `spk[:, frames]`

iii. Common indexing and shape checks enforce alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed` at retained frames.

ii. `speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]`

iii. Native speed units are retained.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from all retained trials/sessions are concatenated and global 25/50/75% quantiles computed.

ii. `all_speed = np.concatenate([...])`; `thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])`

iii. Global thresholds were chosen for comparable physical class meaning and one-pass neural I/O.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `searchsorted(..., side="right")` maps global thresholds to 0–3; ties are not rank-split.

ii. `labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)`

iii. Reported full thresholds yield approximately equal counts.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speeds first share neural indices; nested strict zips later fill the matching output trial.

ii. `for out_trial, speed_trial in zip(out_session, speed_session, strict=True): out_trial[3] = labels`

iii. Strict pairing and shape validation prevent reordering.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Streams truncate to their common length; invalid trial stamps and lick indices are excluded. Alias/count/inventory mismatches, unknown categories, nonfinite numerical data, or too few trials fail loudly.

ii. `common_nframes = min([neural_nframes] + [len(beh[k]) for k in frame_fields])`; `if not np.all(np.isfinite(decoder_input)): raise ValueError(...)`

iii. The notes favor safe length repair but explicit failure for ambiguity/corruption.

## 12-a. What are the most time-consuming steps of the code?

i. Reading ~404 GiB of spikes, trial copies, and writing a 141-GiB pickle dominate; full conversion took ~743 s.

ii. `neural_dict = np.load(path, allow_pickle=True).item()`; `neural = spk[:, frames].copy()`; `pickle.dump(...)`

iii. The notes identify neural I/O as dominant and enforce a single pass.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial `flatnonzero` rescans all session frames and could be one-pass grouped. Plane and ragged speed loops are less amenable and I/O-dominated.

ii. `for trial in range(ntrials): frames = np.flatnonzero(valid & (trial_stamp == trial))`

iii. The agent discusses memory/I/O optimization but not this vectorization.

## 12-c. What processing does the code repeat multiple times?

i. Each trial re-slices behavior fields and rescans trial stamps; validation revisits every trial. Retaining small speed vectors avoids a repeated full neural pass.

ii. `position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]`; analogous speed slicing occurs per trial.

iii. Notes say this avoids rereading ~404 GiB merely for speed thresholds.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Detailed aliases/provenance, timing stats, raw speed vectors, and optional plot context are not decoder features. Raw speeds are discarded after labeling.

ii. `speed_all.append(speeds)`; `info["registry_aliases"] = aliases[base]`; `del speed_all`

iii. The agent justifies them for global thresholds, auditability, and diagnostics.
