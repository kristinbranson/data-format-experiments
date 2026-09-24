# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every `Beh_*.npy` file, merges records by a normalized session key, then loads that session's spike and retinotopy files. It does not use `Imaging_Exp_info.npy` as the master index.

ii. ```python
beh_files = sorted(fname for fname in os.listdir(BEH_ROOT) if fname.startswith("Beh_") and fname.endswith(".npy"))
beh = np.load(os.path.join(BEH_ROOT, fname), allow_pickle=True).item()
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
```

iii. The trajectory says the behavior, spike, and retinotopy stores were inspected and duplicate behavior entries were found to be alternate annotations of the same recordings, so they were collapsed to 89 recordings.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed as the first underscore-delimited component of each session key. Unique IDs are sorted and each session receives the corresponding index.

ii. ```python
subject = parts[0]
subjects = sorted(per_subject_dates)
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The trajectory treats the normalized recording key as carrying the mouse and date; no separate justification was recorded.

## 1-c. How are the data split into sessions?

i. A normalized behavior key (mouse, date, block) defines a session. A trailing `swap...` component is removed and duplicate entries are merged, yielding 89 sorted unique recordings.

ii. ```python
if parts[-1].startswith("swap"):
    return "_".join(parts[:-1])
if base_key not in merged:
    merged[base_key] = {k: v for k, v in record.items()}
session_keys = sorted(behaviors)
```

iii. The agent explicitly checked duplicates and concluded they represented identical physical recordings with alternate stimulus annotations, so each should appear once.

## 1-d. How are the data split into trials?

i. The code loops over `ntrials` and selects frames whose `ft_trInd` equals the trial, are in `ft_CorrSpc`, and have `ft_move > 0`. Thus it keeps only moving corridor frames and can omit pauses within a trial.

ii. ```python
move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
for tr in range(int(beh["ntrials"])):
    mask = (trial_idx == tr) & move_corr_mask
```

iii. The trajectory says many paper analyses restricted to visible-corridor, moving frames and adopts that "running-only analysis rule."

## 1-e. How are trials filtered based on quality controls?

i. Trials with no moving corridor frames after truncation to imaged frames are dropped. There is no whole-trial duration/outlier filter; stationary frames are removed from every retained trial, and sessions with fewer than two retained trials cause an error.

ii. ```python
if not np.any(mask):
    continue
if len(neural_trials) < 2:
    raise RuntimeError(...)
```

iii. The movement restriction was justified as matching running-only paper analyses. The trajectory later attributed very wide time ranges to long pauses and retained those trials rather than adding an outlier rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each plane's deconvolved `spks`; neuron area labels come from retinotopy `iarea`. Behavior masks are also used for neuron scoring and frame selection.

ii. ```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
```

iii. The agent identified these as the paper's deconvolved traces and retinotopic grouping variables.

## 2-b. How is the `neural` data processed?

i. Selected plane rows are concatenated, sliced to each trial's moving corridor frames, and cast to `float16`; no temporal padding, resampling, or signal transformation is applied.

ii. ```python
selected_spk = np.concatenate(selected_chunks, axis=0)
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The agent chose the existing deconvolved traces and `float16` to keep the very large export tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons must belong to V1/mHV/lHV/aHV, have greater mean activity during moving corridor than moving gray-space frames, and then rank among the 128 highest corridor-variance neurons in their area. This normally forces 512 neurons per session.

ii. ```python
responsive = corr_mean > gray_mean
keep = responsive & region_mask
order = np.lexsort((row_arr, plane_arr, -score_arr))
top = order[:MAX_NEURONS_PER_AREA]
```

iii. The trajectory calls this a deterministic, paper-motivated responsive subset and says the cap was necessary to make the pickle and decoder training tractable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are labeled as aligned to corridor entry, but arrays contain only moving corridor frames. Their first retained frame may be after `StartFr`, and pauses create temporal gaps; arrays remain variable length and unpadded.

ii. ```python
mask = (trial_idx == tr) & move_corr_mask
neural_trials.append(selected_spk[:, mask].astype(np.float16))
"temporal_alignment_event": "corridor entry (trial start)"
```

iii. The agent considered corridor alignment plus the paper's movement restriction appropriate; it accepted long elapsed-time gaps caused by removed pauses.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is performed. The reported bin size is the median frame-timestamp difference across sessions (and a session-specific median is used for timing inputs), approximately one imaging frame or 315 ms.

ii. ```python
dts.append(np.median(np.diff(ft)) * MS_PER_DAY)
time_bin_size = float(np.median(dts))
```

iii. The agent kept the native deconvolved imaging frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, retained imaging-frame indices, and a session median frame duration computed from `ft`.

ii. ```python
selected_frames = frame_numbers[mask]
time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds)
```

iii. After verification exposed bad values from absolute timestamp fields, the agent switched to frame-index timing because it considered that the neural alignment space.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue-frame minus each retained frame is multiplied by the session's median seconds per frame; positive values precede the cue and negative values follow it. No interpolation through the actual `ft` timestamps is used.

ii. ```python
dt_seconds = float(np.median(np.diff(np.asarray(beh["ft"], dtype=np.float64))) * SECONDS_PER_DAY)
time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
```

iii. The agent described this revision as correcting mixed absolute timestamps and imaging-frame alignment.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed for exactly the boolean mask used to select neural columns, so lengths and retained frame identities coincide.

ii. ```python
selected_frames = frame_numbers[mask]
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. Frame-index construction was explicitly chosen to align it with imaging data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from dates parsed from normalized session keys and the earliest recording date for each subject.

ii. ```python
date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
first_dates = {subject: min(dates) for subject, dates in per_subject_dates.items()}
```

iii. The final trajectory says a consistent explicit training-day count was unavailable for every recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days since that mouse's first imaging session are computed and broadcast across every retained bin of a trial.

ii. ```python
day_since_first[key] = float((date - first_dates[subject]).days)
day_arr = np.full(mask.sum(), day_value, dtype=np.float32)
```

iii. The agent selected calendar elapsed days as a fallback for missing explicit training-day metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-trial `StartFr`, retained frame indices, and the session median frame duration from `ft`.

ii. ```python
time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds)
```

iii. The agent chose frame-index timing after the verifier revealed implausible results from absolute timestamp fields.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each retained frame index minus `StartFr` is multiplied by median seconds per frame. Fractional `StartFr` is preserved algebraically but actual timestamp irregularity is ignored.

ii. ```python
dt_seconds = float(np.median(np.diff(np.asarray(beh["ft"], dtype=np.float64))) * SECONDS_PER_DAY)
time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. The trajectory calls frame indices the appropriate reference for the neural data.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Values are calculated for the same moving-corridor mask used for neural columns.

ii. ```python
selected_frames = frame_numbers[mask]
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The agent explicitly revised the timing inputs to make them frame-aligned.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` value.

ii. ```python
reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. No separate justification was recorded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. `isRew` is converted to Boolean/float 0 or 1 and broadcast over all retained frames in the trial.

ii. ```python
np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. This implements the requested per-trial binary decoder input; no further justification was given.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial's `WallName` string; `stim_id` is merged between duplicates but is not used for this output.

ii. ```python
wall_name = str(beh["WallName"][tr])
stim_category = category_to_idx[wall_to_category(wall_name)]
```

iii. The trajectory says duplicate annotation merging preserved the richest mapping and the final summary identifies four wall families.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A leading alphabetic token is lowercased, validated as circle/leaf/rock/wood, mapped through a sorted global category list, and broadcast across the trial.

ii. ```python
match = re.match(r"([A-Za-z]+)", str(name))
category = match.group(1).lower()
visual_category = np.full(mask.sum(), stim_category, dtype=np.int16)
```

iii. The agent intended to reduce wall variants to the requested four stimulus families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session lick-frame indices in `LickFr`.

ii. ```python
lick_fr = np.asarray(beh["LickFr"], dtype=int)
```

iii. No separate justification was recorded.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are integer-cast, out-of-range entries dropped, duplicates collapsed, and corresponding imaging frames marked 1 in an otherwise zero array.

ii. ```python
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
lick_binary[np.unique(lick_fr)] = 1
```

iii. The choice follows the requested binary, time-varying representation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The full-session frame flag is sliced with the exact trial mask used for neural columns.

ii. ```python
lick = lick_binary[mask].astype(np.int16)
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The shared frame mask provides direct imaging-frame alignment; no additional rationale was recorded.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from framewise `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. The agent recognized the corridor as 4 m and the source position scale as compatible with four 1 m bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Retained positions are clipped to `[0, 39.999]`, divided by 10, floored, and converted to integers.

ii. ```python
pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. This was chosen to create the four requested equal 1 m corridor bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 source units (decimeters), yielding categories 0–3 labeled 0–1 m through 3–4 m; outliers are clipped to an endpoint bin.

ii. ```python
np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0)
["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. The thresholds directly implement the task's four equal-length spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is truncated to imaging length and selected with the identical moving-corridor trial mask.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
pos_bin = ... ft_pos[mask] ...
```

iii. Position and neural data therefore share retained imaging-frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`; `ft_CorrSpc` and `ft_move` determine which values contribute to global thresholds and trials.

ii. ```python
mask = np.asarray(beh["ft_CorrSpc"], dtype=bool) & (np.asarray(beh["ft_move"]) > 0)
speeds.append(np.asarray(beh["ft_RunSpeed"])[mask])
```

iii. The agent followed its running-only interpretation of the paper analyses.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All moving-corridor speeds from all behavior records are concatenated once, global 25/50/75% value quantiles are computed, and each retained trial speed is digitized against them.

ii. ```python
all_speeds = np.concatenate(speeds)
return np.quantile(all_speeds, [0.25, 0.5, 0.75]).astype(np.float32)
speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False)
```

iii. The agent interpreted “each corresponding to 25% of the data” as global quartile thresholds.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global numeric quantile edges form four categories via `np.digitize`; ties at an edge go to the higher bin and results are clipped to 0–3.

ii. ```python
speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The thresholds were intended to represent quartiles and are reported in metadata/output labels.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sliced by the same moving-corridor trial mask used for neural columns.

ii. ```python
speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False)
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. Both streams retain the same imaging frames.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Duplicate records are merged only where `stim_id` shapes match, filling missing old IDs from nonmissing new IDs. Behavior is truncated to neural length; invalid lick indices and empty trials are dropped. Neuron-count mismatch, no selected neurons, or fewer than two trials raises an error.

ii. ```python
fill = np.isnan(stim_old) & ~np.isnan(stim_new)
stim_old[fill] = stim_new[fill]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
if not np.any(mask): continue
```

iii. The agent investigated duplicates as alternate annotations and used verification to catch timing problems. Other safeguards are implementation checks rather than separately documented decisions.

## 12-a. What are the most time-consuming steps of the code?

i. Loading very large per-session spike files and computing per-neuron corridor/gray means and corridor variances dominate; full conversion also serializes an approximately 825 MB pickle.

ii. ```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
corr_mean = plane[:, corr_mask].mean(axis=1)
corr_var = plane[:, corr_mask].var(axis=1)
```

iii. The trajectory reports raw recordings large enough to require tractability decisions, an initial process around 10 GB RAM, and identifies streaming all 89 recordings as expensive.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly builds a full-length equality mask; session-key/date loops, per-area candidate loops, and selected-neuron reconstruction could also be consolidated, though I/O and array reductions dominate.

ii. ```python
for tr in range(int(beh["ntrials"])):
    mask = (trial_idx == tr) & move_corr_mask
for region in BRAIN_REGIONS:
    region_mask = np.isin(plane_areas, AREA_CODES[region])
```

iii. The trajectory does not discuss vectorization specifically; it focused on throughput, memory, and progress visibility.

## 12-c. What processing does the code repeat multiple times?

i. Behavior arrays are converted/sliced in several functions; every trial rescans all session frames to create its mask; dates/session keys are parsed repeatedly; moving corridor masks are independently created for speed edges, neuron selection, and trial building.

ii. ```python
mask = np.asarray(beh["ft_CorrSpc"], dtype=bool) & ...
corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & ...
move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & ...
```

iii. No explicit justification was recorded; the agent prioritized a clear streaming implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It merges missing `stim_id` annotations but later derives stimulus solely from `WallName`. It computes/retains full category and session-key metadata, and performs responsiveness/variance ranking whose scores are discarded after selection.

ii. ```python
merged[base_key]["stim_id"] = stim_old
wall_name = str(beh["WallName"][tr])
score_arr = np.concatenate(candidate["score"])
```

iii. The merge was justified as preserving the “richest mapping,” although the final converter never consumes that mapping; selection scores exist only to enforce the tractability cap.
