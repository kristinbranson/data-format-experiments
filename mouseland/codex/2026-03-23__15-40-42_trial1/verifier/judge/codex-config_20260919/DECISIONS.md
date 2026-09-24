# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every `Beh_*.npy`, loads each behavior dictionary, groups keys by the first five underscore-delimited fields (mouse/date/block), and retains one representative behavior record per recording. It separately loads the master experiment map, each selected recording's concatenated `spks`, and its retinotopy file.

ii.
```python
for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
    beh_dict = np.load(beh_path, allow_pickle=True).item()
    for key, record in beh_dict.items():
        base = "_".join(key.split("_")[:5])
```
```python
spk_item = np.load(path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
```

iii. The notes distinguish 142 experiment records, 99 behavior keys, and 89 unique neural recordings. The agent chose 89 unique recording bases to match the paper and avoid duplicated neural data and leakage.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from the first field of the recording base. Sorted unique IDs form `subjects`; every retained session receives the corresponding `subject_idx`.

ii.
```python
subject = parts[0]
subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
subject_idx.append(subject_to_idx[spec.subject])
```

iii. The mouse name is explicit in every recording ID; the notes expected and obtained 19 mice.

## 1-c. How are the data split into sessions?

i. A session is a unique `<mouse>_<year>_<month>_<day>_<block>` base. Duplicate behavior keys are resolved with a score preferring non-swap keys, more finite `stim_id` entries, and more walls.

ii.
```python
base = "_".join(key.split("_")[:5])
if session_key_score(key, record) < session_key_score(current.key, current.record):
    current.key = key
    current.record = record
```

iii. The paper reports 89 recordings, while experiment labels and swap aliases reuse recordings. The agent therefore deduplicated by recording base and selected a canonical behavior record.

## 1-d. How are the data split into trials?

i. Trials are identified by integer `ft_trInd`. Within each trial, only finite frame indices satisfying both `ft_CorrSpc` and `ft_move > 0` are retained, so stopped frames inside a traversal are removed and a trial can be temporally discontinuous.

ii.
```python
trial_ids = ft_tr[valid].astype(int)
keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
trial_frames = valid_idx[keep & (trial_ids == trial)]
```

iii. The notes say the paper's analyses considered only running timepoints in the textured 0–4 m corridor and describe this as the closest match to the published analyses.

## 1-e. How are trials filtered based on quality controls?

i. A trial is removed if it has fewer than five retained running-corridor frames, lasts more than 60 seconds from raw trial start to its last retained frame, or has a retained-frame gap over 10 seconds. Sessions with fewer than two valid trials are removed.

ii.
```python
if frame_idx.size < MIN_TRIAL_TIMEPOINTS: return False, "too_few_timepoints"
if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S: return False, "retained_duration_gt_60s"
if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S: return False, "interframe_gap_gt_10s"
```

iii. The agent called these decoder-specific filters for unusable/pathological stalled trials. The full run removed 2,217 trials and retained 35,893.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the `spks` list in each `<base>_neural_data.npy`, concatenated across planes. `iarea` from the matching retinotopy file supplies region labels used in neuron selection and metadata.

ii.
```python
spk_item = np.load(path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
iarea = np.asarray(ret["iarea"], dtype=float)
```

iii. The paper and source code identify `spks` as already-deconvolved fluorescence, so the agent did not recompute dF/F or deconvolve it.

## 2-b. How is the `neural` data processed?

i. The planes are concatenated as float32, a stimulus-selective subset is chosen, and each trial takes the selected neurons at its retained running frames. No normalization, interpolation, padding, or temporal resampling is applied.

ii.
```python
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The agent argued a compact subset was needed to make full decoding tractable and described the selection as paper-grounded top/bottom 5% stimulus selectivity per visual area.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It computes stimulus-pair d-prime on running corridor frames, requires corridor responsiveness relative to gray space, then normally retains the highest and lowest 5% per V1/mHV/lHV/aHV. It has fallbacks to mapped neurons when comparison data or selected cells are absent.

ii.
```python
hi, lo = np.nanpercentile(dp[candidates], [95, 5])
selected |= candidates & ((dp >= hi) | (dp <= lo))
```
```python
corr_neu = ((np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1)) |
            (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1)))
```

iii. The notes link this rule to coding-direction analyses and size reduction, reducing 4,691,034 raw neurons to 304,548 exported neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are described as aligned to corridor entry/trial start, but arrays contain only frames labeled as moving inside the corridor. Timing inputs are relative to `Trial_start_time`; there is no resampling or explicit insertion of a time-zero bin.

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
neural_trial = spk[:, frame_idx]
```

iii. The agent preferred frame-level `ft_trInd` and raw timestamps to avoid boundary ambiguity and preserve native data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained without rebinning. Metadata uses the median of session median frame intervals, about 314.7 ms.

ii.
```python
dt_ms = np.diff(ft) * MS_PER_DAY
spec.median_frame_dt_ms = float(np.median(dt_ms))
"time_bin_size": float(np.median(frame_dt_medians))
```

iii. The notes state frame spacing is consistent and native timestamps are more faithful than a synthetic resampled clock.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundTime` and frame-level `ft` timestamps.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. The notes identify raw cue timing as the direct, reproducible timing source.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. MATLAB-day timestamps are subtracted and converted to seconds. Values are positive before the cue and negative after it, then cast to float32.

ii.
```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. This directly represents continuous time remaining to the cue as requested.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `ft` at exactly the same `frame_idx` columns used for the neural trial.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
neural_trial = spk[:, frame_idx]
```

iii. The agent notes that all frame-varying streams share the imaging-frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in each recording base and the earliest selected recording date for that subject.

ii.
```python
date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
first_date_by_subject.setdefault(spec.subject, spec.date)
```

iii. The paper does not define this decoder-required variable, so the agent chose dates as an objective source.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It is elapsed calendar days since that mouse's first selected imaging session, represented as float and repeated over every retained frame.

ii.
```python
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. The agent considered elapsed days a reproducible continuous proxy; the resulting range was 0–92.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses frame timestamps `ft` and per-trial `Trial_start_time`.

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. The notes say this is consistent with the raw timestamp-based analyses.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The trial-start timestamp is subtracted from each retained frame timestamp, converted from days to seconds, and cast to float32.

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. This makes corridor entry approximately zero and later bins positive.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated at the same retained `frame_idx` used for neural columns.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
neural_trial = spk[:, frame_idx]
```

iii. The shared native frame grid supplies alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the trial-level `isRew` flag.

ii.
```python
float(bool(spec.record["isRew"][trial_idx]))
```

iii. The notes state this is 1 only for rewarded-corridor task trials and 0 for unrewarded, unsupervised, naive, and control trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is coerced to Boolean/float and broadcast across every retained frame in the trial.

ii.
```python
reward_available = np.full(frame_idx.size,
    float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. Broadcasting gives all input trials the uniform `(4, T)` shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`; `UniqWalls` across selected records is used to construct the global category list.

ii.
```python
categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
visual_to_idx[str(spec.record["WallName"][trial_idx])]
```

iii. The agent treated every raw wall name as a globally consistent stimulus identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Fifteen raw wall names, including crop and swap variants, are sorted and integer-encoded. The trial's code is broadcast over all timepoints; variants are not collapsed to circle/leaf/rock/wood.

ii.
```python
stim_idx = np.full(frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])], dtype=np.int16)
```

iii. The notes explicitly chose 15 global categories found in the imaging data rather than a broad texture-family mapping.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickFr` and `LickTrind`.

ii.
```python
lick_frames = np.asarray(record["LickFr"], dtype=float)
lick_trial = np.asarray(record["LickTrind"], dtype=float)
```

iii. These fields directly identify the imaging frame and trial of each lick.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Nonfinite/out-of-range events are removed, frame numbers are cast to integers, duplicates are collapsed per trial, and retained frames are labeled 1 when present in the lick lookup.

ii.
```python
lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The notes describe projecting lick events to a binary imaging-frame vector.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frame numbers are compared directly with the same retained frame indices used for neural columns.

ii.
```python
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
neural_trial = spk[:, frame_idx]
```

iii. Both are on the native imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos` in decimeters.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. The paper and raw metadata establish a 40-decimeter textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions are clipped to `[0, 39.999]` decimeters, divided by 10 using floor division, and cast to categorical integers 0–3.

ii.
```python
pos = np.clip(pos_decimeters, 0.0, 39.999)
return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. This converts raw decimeters into the requested one-meter bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 decimeters, yielding 0–1, 1–2, 2–3, and 3–4 m.

ii.
```python
POSITION_OUTPUT_VALUES = ["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. These are exactly four equal-length one-meter spatial bins as requested.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same `frame_idx` as neural activity.

ii.
```python
np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx]
```

iii. No interpolation is necessary because behavior is already frame-aligned.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-level `ft_RunSpeed` at retained trial frames.

ii.
```python
run_speed = np.asarray(record["ft_RunSpeed"][:spec.estimated_nfr], dtype=np.float32)
all_speeds.append(run_speed[frame_idx])
```

iii. The raw stream directly supplies speed on the imaging-frame grid.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from all trials passing preliminary filters across all selected sessions are concatenated. Global 25th/50th/75th percentile value edges are computed, then `searchsorted(..., side="right")` assigns bins 0–3.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
bins = np.searchsorted(edges, speed, side="right")
```

iii. The agent interpreted “each corresponding to 25% of the data” as requiring full-dataset global quartiles rather than session-specific bins.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global numeric quartile edges define `q1`–`q4`; ties remain together, so the observed bin proportions need not be exactly 25%.

ii.
```python
RUNNING_SPEED_OUTPUT_VALUES = ["q1", "q2", "q3", "q4"]
return np.clip(bins, 0, 3).astype(np.int16)
```

iii. Global edges make category meanings consistent across sessions, though the notes observed a large tied-speed imbalance in sample data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is selected at exactly the same `frame_idx` as neural columns and then digitized.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The shared frame indices provide direct alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Frame-level behavior arrays are initially limited to their common minimum length and later clipped to the true neural length. Nonfinite trial IDs and lick records are discarded, out-of-range licks are dropped, retinotopy/neuron count mismatches raise an error, unusable trials are filtered, and sessions with fewer than two trials are skipped.

ii.
```python
return min(int(np.asarray(record[key]).shape[0]) for key in frame_keys)
frame_idx = frame_idx[frame_idx < nfr]
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
```

iii. The notes found behavior arrays exceed neural recordings by 1–3 frames and followed the reference convention of truncating behavior to neural length, while adding documented decoder-pathology filters.

## 12-a. What are the most time-consuming steps of the code?

i. Loading each very large neural file, concatenating all planes, computing whole-session selectivity, and repeatedly materializing selected trial matrices dominate runtime and storage.

ii.
```python
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
```

iii. The notes explicitly identify full neural-file I/O as heavy and report about 10.4 seconds per sample session; selective export was introduced largely for tractability.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial frame construction scans the complete valid-frame array once for every trial; lick lookup similarly loops over unique trials, and processing constructs each trial array in Python. Grouping/sorting frame indices once could reduce repeated masks.

ii.
```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]
```

iii. The agent did not explicitly propose this vectorization, focusing instead on I/O and one-session-at-a-time processing.

## 12-c. What processing does the code repeat multiple times?

i. Trial quality checks run during the behavior preparation pass and again during session export. Frame arrays such as `ft_RunSpeed` are repeatedly converted/indexed inside trial loops, and neuron selection repeatedly computes means over stimulus/gray masks.

ii.
```python
trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)
```
```python
keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
```

iii. The notes acknowledge a behavior-derived first pass followed by clipping and checking against exact neural frame counts in a second pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and concatenates every raw neuron before discarding roughly 93.5% through selectivity filtering. It also reserves an `other` brain-region class that the usual selection excludes, gathers experiment aliases chiefly for metadata, and computes optional plotting information unused by decoder training.

ii.
```python
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
spk = spk[selected_neurons]
BRAIN_REGION_NAMES = ["V1", "mHV", "lHV", "aHV", "other"]
```

iii. The notes explicitly say full neural recordings must be loaded before selection and call this the principal remaining inefficiency.
