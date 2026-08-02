# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every `Beh_*.npy` file under `data/beh/`, uses `Imaging_Exp_info.npy` to recover experiment-type metadata, deduplicates behavior records onto one session per recording base, and then loads one matching spike file and one retinotopy file for each selected session during the main processing pass.

ii. ```python
for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
    beh_dict = np.load(beh_path, allow_pickle=True).item()
    for key, record in beh_dict.items():
        base = "_".join(key.split("_")[:5])
        ...

path = root / "data" / "spk" / f"{base}_neural_data.npy"
spk_item = np.load(path, allow_pickle=True).item()
ret_path = root / "data" / "retinotopy" / f"{subject}_{date.strftime('%Y_%m_%d')}_trans.npz"
```

iii. In `CONVERSION_NOTES.md`, the agent says the paper reports 89 recordings in 19 mice, while behavior metadata reuses recordings across experiment labels. That led it to treat unique recording bases as the session unit and pair each with the raw spike and retinotopy files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are parsed from the recording-base prefix before the first underscore, collected into a sorted unique list, and each kept session stores the corresponding subject index.

ii. ```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    subject = parts[0]
    ...

subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
subject_idx.append(subject_to_idx[spec.subject])
```

iii. The notes explicitly state that mouse names are parsed from the recording base and that 19 unique subjects should remain after deduplication.

## 1-c. How are the data split into sessions?

i. A session is one unique recording base `<mouse>_<date>_<blk>`. If multiple behavior keys point at the same base, the agent keeps one canonical key, preferring non-`swap` keys with more finite `stim_id` values and more unique wall names.

ii. ```python
def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)

if session_key_score(key, record) < session_key_score(current.key, current.record):
    current.key = key
    current.record = record
```

iii. The notes say the dataset contains repeated experiment labels and `swap` aliases for the same raw recording, so the session unit should be the unique recording base to avoid duplicate use of the same recording.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from frame-level `ft_trInd`. For each trial id, the agent keeps only finite frame labels that are in the textured corridor and have `ft_move > 0`, then stores the corresponding frame indices for that trial.

ii. ```python
ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
valid_idx = np.flatnonzero(np.isfinite(ft_tr))
trial_ids = ft_tr[np.isfinite(ft_tr)].astype(int)
keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[np.isfinite(ft_tr)]
keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[np.isfinite(ft_tr)] > 0
...
trial_frames = valid_idx[keep & (trial_ids == trial)]
```

iii. The notes say the agent deliberately used `ft_trInd` rather than `StartFr`/`EndFr` because that is how the reference code groups frames into trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent removes trials with fewer than 5 retained timepoints, retained duration longer than 60 s from `Trial_start_time`, or any retained inter-frame gap over 10 s. It also discards sessions with fewer than 2 remaining trials.

ii. ```python
if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
    return False, "too_few_timepoints"
...
if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
    return False, "retained_duration_gt_60s"
...
if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
    return False, "interframe_gap_gt_10s"
...
if len(session_neural) < 2:
    removed_sessions.append((spec.base, len(session_neural), "fewer_than_two_valid_trials"))
```

iii. The notes describe these as decoder-specific filters added to remove pathological timing outliers and to satisfy the decoder requirement that each session have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported neural activity comes from the raw `spks` arrays stored in each `<base>_neural_data.npy` file.

ii. ```python
spk_item = np.load(path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
return spk.astype(np.float32, copy=False)
```

iii. The notes say the reference code treats `spks` as the already deconvolved fluorescence traces and never recomputes dF/F.

## 2-b. How is the `neural` data processed?

i. The agent concatenates imaging planes, casts to `float32`, keeps the raw deconvolved values, maps neurons to retinotopic regions, and then applies a paper-inspired neuron-selection step that keeps the top 5% positive and top 5% negative stimulus-selective neurons per visual area with several fallbacks.

ii. ```python
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
...
dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
...
hi, lo = np.nanpercentile(dp[candidates], [95, 5])
selected |= candidates & ((dp >= hi) | (dp <= lo))
...
spk = spk[selected_neurons]
```

iii. The notes say this was done to keep the export tractable while staying “paper-grounded” by reusing the reference coding-direction neuron-selection logic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are effectively filtered to visual-cortex areas and then to corridor-responsive, stimulus-selective neurons. If that yields too few candidates, the code falls back to all mapped visual-area neurons, and finally to any neuron with a known visual-area mapping.

ii. ```python
candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
...
if candidates.sum() < 20:
    selected[candidates] = True
...
if not np.any(selected):
    selected = (region_idx < 4) & corr_neu & np.isfinite(dp)
if not np.any(selected):
    selected = region_idx < 4
```

iii. The notes explicitly distinguish this from any Suite2p cell-quality filter: the agent says it is using the paper’s functional selection logic rather than an electrophysiology-style QC mask.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural matrices are formed by grouping raw frames by `ft_trInd` and keeping the retained running/corridor frames from each trial. The code does not resample or pad around trial start; instead, it carries the trial-start timing in `time_since_trial_start_s`.

ii. ```python
frame_idx = frame_idx[frame_idx < nfr]
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
...
"temporal_alignment_event": "corridor entry / trial start",
```

iii. The notes say the agent preserved native frame timestamps and aligned streams by trial membership rather than inventing a new time grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the native imaging-frame grid. The metadata stores the median frame interval across sessions as `time_bin_size`, and the script does not perform temporal rebinning or resampling.

ii. ```python
dt_ms = np.diff(ft) * MS_PER_DAY
spec.median_frame_dt_ms = float(np.median(dt_ms))
...
"time_bin_size": float(np.median(frame_dt_medians)),
"frame_bin_source": "native imaging frame timestamps; no temporal resampling",
```

iii. The notes say the reference code mainly works in raw frame time or in position bins, so keeping native frame time was the most faithful choice.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime` at the trial level and `ft` at the retained frame level.

ii. ```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. The notes say cue timing should come from the same raw timestamps used in the reference cue-alignment functions.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the code subtracts the frame timestamp from the trial’s `SoundTime` and converts the day units to seconds.

ii. ```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. The notes describe this as a direct continuous timing variable, with positive values before the cue and negative values after it.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on the exact same retained `frame_idx` values used to slice the neural matrix, so every neural frame gets one time-to-cue value.

ii. ```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
input_trial = np.vstack([time_to_cue, training_day, time_since_start, reward_available])
```

iii. The notes say all decoder inputs should share the native frame grid of the exported neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Training day is derived from the recording date encoded in each session base, compared against the earliest date seen for that subject.

ii. ```python
subject, date, blk = parse_base(base)
...
first_date_by_subject.setdefault(spec.subject, spec.date)
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. The notes say this variable is required by the decoder task rather than the paper, so the agent chose the reproducible proxy “elapsed days since the subject’s first retained imaging session.”

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code computes an integer day difference in calendar days and then repeats that scalar across every retained frame in the trial.

ii. ```python
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
...
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. The notes say the agent wanted a uniform `(4, T)` input matrix, so per-trial variables are repeated over time.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. No environment-type input is created at all. The converted dataset has four inputs only: time to sound cue, training day, time since trial start, and reward availability.

ii. ```python
INPUT_NAMES = [
    "time_to_sound_cue_s",
    "training_day",
    "time_since_trial_start_s",
    "reward_available",
]
```

iii. The notes never map an environment-type input because it was not part of the decoder task specification the agent followed.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None, because the agent never computes or exports an environment-type variable.

ii. ```python
INPUT_NAMES = [
    "time_to_sound_cue_s",
    "training_day",
    "time_since_trial_start_s",
    "reward_available",
]
```

iii. This follows directly from the decision to exclude environment type from the decoder inputs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from frame timestamps `ft` and trial start timestamps `Trial_start_time`.

ii. ```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. The notes say this variable should preserve native frame timing relative to the alignment event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the code subtracts the trial-start timestamp from the frame timestamp and converts the resulting day units to seconds.

ii. ```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. The notes treat this as the explicit timing channel that anchors all retained frames to trial start without resampling.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame indices used for the neural slice, so it is frame-by-frame aligned with `neural`.

ii. ```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
input_trial = np.vstack([time_to_cue, training_day, time_since_start, reward_available])
```

iii. The notes repeatedly emphasize using the native frame grid for every exported stream.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived from the per-trial `isRew` flag.

ii. ```python
reward_available = np.full(
    frame_idx.size,
    float(bool(spec.record["isRew"][trial_idx])),
    dtype=np.float32,
)
```

iii. The notes say the decoder task wanted a trial-level binary “rewarded corridor vs not” variable, and `isRew` is the direct raw source for that.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code converts `isRew[trial_idx]` to a float and repeats it across every retained frame of the trial.

ii. ```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. The notes say all decoder inputs were stored in a uniform `(4, T)` format, so per-trial values were tiled over time.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The output category comes from each trial’s `WallName`, with the global category vocabulary collected from session `UniqWalls` values across all selected sessions.

ii. ```python
def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
...
stim_idx = np.full(frame_idx.size, visual_to_idx[str(spec.record["WallName"][trial_idx])], dtype=np.int16)
```

iii. The notes say the agent wanted a single global categorical mapping that covers all imaging-session stimuli, including rock and wood variants.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent sorts the unique stimulus names globally, maps each trial’s `WallName` to its integer category id, and repeats that id across all retained frames in the trial.

ii. ```python
visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}
stim_idx = np.full(
    frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])],
    dtype=np.int16,
)
```

iii. The notes say the repeated per-frame representation was chosen to keep every output in a uniform `(4, T)` array.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from the raw lick frame indices `LickFr` and lick trial indices `LickTrind`.

ii. ```python
lick_frames = np.asarray(record["LickFr"], dtype=float)
lick_trial = np.asarray(record["LickTrind"], dtype=float)
```

iii. The notes say the agent wanted a time-varying lick raster aligned to imaging frames, and `LickFr`/`LickTrind` provide that directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code removes non-finite and out-of-range lick events, groups remaining lick frames by trial, and then marks each retained frame as 1 if it appears in that trial’s lick-frame list.

ii. ```python
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
...
lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
...
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The notes describe this as a binary per-frame lick output rather than the trial-level lick summaries used in some figure analyses.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick raster is evaluated on the same retained `frame_idx` array used to slice the neural matrix.

ii. ```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The notes say the lick output should live on the same native imaging frames as the exported neural activity.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos` on the retained frame indices.

ii. ```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. The notes say the raw behavior files store corridor position in decimeters, which matches the paper’s 4 m texture corridor after unit conversion.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent takes the raw decimeter positions on retained frames and then discretizes them; it does not interpolate or smooth them first.

ii. ```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes say the export should preserve raw frame timing and use the paper’s corridor geometry rather than introduce a new interpolation step.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The code clips position to `[0, 39.999]` decimeters and maps it into bins `0-9.999`, `10-19.999`, `20-29.999`, and `30-39.999`, which correspond to four 1 m bins.

ii. ```python
pos = np.clip(pos_decimeters, 0.0, 39.999)
return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes explicitly interpret the raw units as decimeters and the texture corridor as 40 decimeters long.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Each position-bin value is read from `ft_Pos[frame_idx]`, so it is aligned one-to-one with the neural frames in the trial.

ii. ```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. The notes say all time-varying outputs should be evaluated on retained imaging frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the frame-level `ft_RunSpeed` values on retained frames.

ii. ```python
run_speed = np.asarray(record["ft_RunSpeed"][:spec.estimated_nfr], dtype=np.float32)
...
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The notes say the paper’s speed analyses use frame-aligned speed measurements, so `ft_RunSpeed` is the correct raw source.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first gathers all retained running-speed values across sessions and valid trials, computes global quartile edges, and then digitizes each trial’s retained `ft_RunSpeed` values using those common edges.

ii. ```python
for trial_idx, (keep_trial, frame_idx) in enumerate(zip(spec.kept_trial_mask, spec.trial_frame_indices)):
    if keep_trial and trial_passes_quality_filters(...)[0]:
        all_speeds.append(run_speed[frame_idx])
...
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
...
speed_bin = digitize_speed(..., speed_edges)
```

iii. The notes say global quartiles were chosen because the decoder task asked for bins covering 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The code uses the 25th, 50th, and 75th percentiles of all retained speed values and then bins each speed with `np.searchsorted(..., side="right")`, clipped to category ids 0-3.

ii. ```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
...
bins = np.searchsorted(edges, speed, side="right")
return np.clip(bins, 0, 3).astype(np.int16)
```

iii. The notes explicitly say the quartiles were global rather than per-session because the task specified 25% bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sampled at the same retained frame indices as the neural data and then discretized, so it is frame-aligned with `neural`.

ii. ```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The notes say all time-varying outputs were intentionally built on the native imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or inconsistent values by ignoring non-finite `ft_trInd` and lick entries, clipping all trial frame indices to the available neural frame count, choosing a canonical behavior key when multiple labels share a recording, erroring on retinotopy/neuron count mismatches, and using fallbacks when the preferred stimulus pair or selectivity pool is unavailable.

ii. ```python
valid = np.isfinite(ft_tr)
frame_idx = frame_idx[frame_idx < nfr]
...
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
...
if iarea.shape[0] != nneurons:
    raise ValueError(...)
...
if stim_pair is None:
    return np.arange(spk.shape[0], dtype=np.int32)
```

iii. The notes say the reference data have slight behavior/neural frame-length mismatches and repeated metadata keys, so the agent adopted the reference convention of truncating behavior arrays to neural frame count and added a few defensive fallbacks.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading each full spike matrix from disk, computing the neuron-selection statistics on that full matrix, looping over every trial to build per-trial arrays, and finally writing the 9.7 GiB pickle.

ii. ```python
spk = load_spike_matrix(root, spec.base)
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
...
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...
write_pickle(out_path, data)
```

iii. The notes explicitly call the workflow I/O-heavy because each full neural recording has to be loaded before the compact subset can be chosen.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-building loop in `build_trial_frame_indices`, the per-trial speed-collection loop in `prepare_session_specs`, the per-area loop in `select_decoder_neurons`, the per-trial lick-lookup construction, and parts of the per-trial export loop all could be vectorized or cached more aggressively.

ii. ```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]
...
for trial_idx, (keep_trial, frame_idx) in enumerate(...):
    ...
for area in range(4):
    ...
for trial in np.unique(lick_trial):
    ...
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...
```

iii. The notes mention that several preparation loops still iterate trial-by-trial and that repeated array conversions remain in the session loop.

## 12-c. What processing does the code repeat multiple times?

i. It repeats trial-quality filtering in both the preparation pass and the main export pass, repeatedly clips frame indices to `< nfr`, and repeatedly converts the same record arrays with `np.asarray(...)` inside trial loops.

ii. ```python
if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
    all_speeds.append(run_speed[frame_idx])
...
keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
...
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The notes call out that quality filtering is computed once to get speed quartiles and then again during export, and that several behavior arrays are reconverted inside the hot loop.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes diagnostic information that is not used by downstream decoder training, including full `frame_dt_medians`, `kept_trial_mask`, the full `removed_trials` list before truncating it for metadata, and optional processing plots. These support validation and notes, not the decoder itself.

ii. ```python
spec.kept_trial_mask = np.array([...], dtype=bool)
frame_dt_medians.append(spec.median_frame_dt_ms)
...
removed_trials.append((spec.base, int(trial_idx), int(frame_idx.size), remove_reason))
...
"removed_trials": removed_trials[:1000],
"removed_trials_truncated": len(removed_trials) > 1000,
...
if processing_plots_remaining > 0:
    make_processing_plot(...)
```

iii. The notes frame these as sanity-check and reporting aids rather than core decoder inputs or outputs.
