# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans every top-level `Beh_*.npy` file under `/app/data/beh`, creates one `SessionSpec` per recording base (mouse + date + block), uses `Imaging_Exp_info.npy` only to attach experiment-type labels, and later loads spikes and retinotopy per kept session. Trial-level arrays are taken from the chosen behavior record inside each `SessionSpec`.

ii. 
```python
for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
    exp_type = beh_path.stem.replace("Beh_", "")
    beh_dict = np.load(beh_path, allow_pickle=True).item()
    for key, record in beh_dict.items():
        base = "_".join(key.split("_")[:5])
```
```python
spk = load_spike_matrix(root, spec.base)
region_idx = load_region_index(root, spec.base, nneurons)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this as using the paper’s 89 unique recording bases while avoiding leakage from duplicated experiment labels and `swap1`/`swap2` aliases.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are parsed from the first token of the recording base and then deduplicated/sorted at assembly time. Each kept session gets a `subject_idx` into that sorted subject list.

ii.
```python
subject, date, blk = parse_base(base)
```
```python
subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
subject_idx.append(subject_to_idx[spec.subject])
```

iii. The notes say the mouse id is already encoded in each recording base, so no further inference was needed.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique recording base made from the first five underscore-separated fields of a behavior key. If multiple behavior keys share a base, the AI keeps one “representative” key using a heuristic score that prefers non-swap keys and keys with more usable stimulus annotations.

ii.
```python
base = "_".join(key.split("_")[:5])
if base not in selected:
    selected[base] = SessionSpec(...)
else:
    current = selected[base]
    if session_key_score(key, record) < session_key_score(current.key, current.record):
        current.key = key
        current.record = record
```
```python
def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    stim_id = np.asarray(record.get("stim_id", []), dtype=float)
    non_nan_stim = int(np.isfinite(stim_id).sum()) if stim_id.size else 0
    unique_walls = int(len(np.unique(record.get("WallName", []))))
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)
```

iii. The trajectory says this was done to avoid leakage when multiple behavior keys point at the same underlying recording, especially `swap1`/`swap2` variants.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from frame indices, not from `StartFr`/`EndFr`. For each trial id `0..ntrials-1`, the AI keeps only frames where `ft_trInd` matches the trial, `ft_CorrSpc` is true, and `ft_move > 0`.

ii.
```python
ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
valid = np.isfinite(ft_tr)
trial_ids = ft_tr[valid].astype(int)
keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
```
```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]
    frame_indices.append(trial_frames.astype(np.int32, copy=False))
```

iii. The notes and trajectory justify this as a “paper-style running corridor mask,” i.e. keeping only running timepoints inside the 0-4 m textured corridor.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have fewer than 5 retained timepoints, if the retained portion lasts more than 60 s after `Trial_start_time`, or if any retained inter-frame gap exceeds 10 s. Sessions with fewer than 2 surviving trials are also dropped.

ii.
```python
if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
    return False, "too_few_timepoints"
...
if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
    return False, "retained_duration_gt_60s"
...
if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
    return False, "interframe_gap_gt_10s"
```
```python
if len(session_neural) < 2:
    removed_sessions.append((spec.base, len(session_neural), "fewer_than_two_valid_trials"))
```

iii. In Step 10 of `CONVERSION_NOTES.md`, the AI says these filters were introduced after finding stalled/outlier trials whose retained running frames occurred long after nominal trial start and created extreme timing values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from concatenating the `spks` arrays in each session’s spike file. The retinotopy file’s `iarea` is also loaded because the AI uses it for region labels and neuron filtering.

ii.
```python
spk_item = np.load(path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
```
```python
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=float)
```

iii. The notes explicitly say the stored `spks` arrays are already deconvolved calcium traces and should be used directly.

## 2-b. How is the `neural` data processed?

i. After loading, the AI casts spikes to `float32`, selects only a subset of neurons with `select_decoder_neurons`, and then slices those neurons by each trial’s retained frame indices. Trial arrays are stored without padding.

ii.
```python
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
return spk.astype(np.float32, copy=False)
```
```python
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
...
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The notes say the all-neuron export was too large, so the AI switched to a “paper-grounded” selective-neuron subset to make full decoding tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first maps all neurons into `V1`, `mHV`, `lHV`, `aHV`, or `other` via `iarea`. It then keeps a selective subset: corridor-responsive neurons in each of the four named visual areas, taking top and bottom 5% by `d'` within each area, with fallbacks to all mapped neurons or even all neurons if the stimulus-pair logic fails.

ii.
```python
region_idx = np.full(nneurons, 4, dtype=np.int16)
region_idx[iarea == 8] = 0
region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1
region_idx[np.isin(iarea, [5, 6])] = 2
region_idx[np.isin(iarea, [3, 4])] = 3
```
```python
for area in range(4):
    candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
    ...
    hi, lo = np.nanpercentile(dp[candidates], [95, 5])
    selected |= candidates & ((dp >= hi) | (dp <= lo))
```

iii. `CONVERSION_NOTES.md` Step 5 says this mirrors the paper’s coding-direction analyses and reduces the exported dataset size enough for decoder training.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata labels the alignment event as “corridor entry / trial start,” but the actual per-trial neural matrix contains only the retained running corridor frames belonging to that trial. So the trial is anchored to trial start conceptually, but non-running and non-corridor frames are removed.

ii.
```python
"temporal_alignment_event": "corridor entry / trial start",
```
```python
trial_frames = valid_idx[keep & (trial_ids == trial)]
...
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The notes repeatedly justify this as preserving the paper’s running-only corridor analysis rather than exporting every frame from entry to exit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on native imaging-frame timestamps. The script does not resample or rebin in time; it stores one column per retained imaging frame and reports the median inter-frame interval in milliseconds as `metadata["time_bin_size"]`.

ii.
```python
dt_ms = np.diff(ft) * MS_PER_DAY
spec.median_frame_dt_ms = float(np.median(dt_ms))
```
```python
"time_bin_size": float(np.median(frame_dt_medians)),
"frame_bin_source": "native imaging frame timestamps; no temporal resampling",
```

iii. The notes say native frame bins were kept because they are already the paper’s time base and because no synthetic clock was needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from session frame timestamps `ft` and the per-trial cue timestamp `SoundTime`.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. The notes say the AI preferred raw timestamps over resampling to another grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every retained frame of a trial, the script subtracts the frame timestamp from that trial’s `SoundTime`, converts days to seconds, and stores the result as `float32`.

ii.
```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. The justification given in the notes is that native timestamps preserve the paper’s timing while satisfying the decoder’s need for a time-varying input.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same `frame_idx` used to slice the neural matrix for that trial.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
```

iii. The notes and raw sanity checks say all exported streams were reconstructed from the same retained-frame mask.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date encoded in each session base id, relative to the first date seen for that mouse.

ii.
```python
subject = parts[0]
date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
```
```python
first_date_by_subject.setdefault(spec.subject, spec.date)
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. In Step 5, the notes justify this as a reproducible decoder-required proxy using raw session dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script sorts sessions by subject/date/block, finds each subject’s first recording date, computes elapsed calendar days since that date, and repeats that scalar across every retained frame of every trial in the session.

ii.
```python
specs = sorted(selected.values(), key=lambda s: (s.subject, s.date, int(s.blk)))
...
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```
```python
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. The notes say “elapsed days since the mouse’s first retained imaging session” was chosen because the paper does not define a training-day variable but the decoder task requires one.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `ft` frame timestamps and the per-trial absolute timestamp `Trial_start_time`.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. The notes say raw timestamps were used directly instead of reconstructing time from frame numbers.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the script subtracts the trial’s `Trial_start_time` from the frame timestamp and converts from MATLAB-day units to seconds.

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. The notes frame this as a direct timestamp difference on the same frame grid as the neural data.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame indices as the neural array for that trial.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
```

iii. The notes and trajectory describe the export as reconstructing all streams on a shared retained-frame mask.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` field.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. The notes say this matches rewarded vs unrewarded corridor identity, with unsupervised/naive sessions expected to be all zeros.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is converted to a boolean/float and broadcast across all retained frames of the trial.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
input_trial = np.vstack(
    [time_to_cue, training_day, time_since_start, reward_available]
).astype(np.float32, copy=False)
```

iii. The notes justify the broadcast as a decoder-format requirement so every input trial has a uniform `(4, T)` shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trial-level `WallName` labels, while the set of allowed output values is collected from `UniqWalls` across sessions.

ii.
```python
categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
```
```python
stim_idx = np.full(
    frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])],
    dtype=np.int16,
)
```

iii. The notes say the AI intentionally kept the global set of distinct wall names present in the imaging data.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script does not collapse textures into four broad families. Instead, it builds a global vocabulary of raw wall names (15 categories in the notes) and repeats the selected category index across all retained frames of the trial.

ii.
```python
def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories
```
```python
output_trial = np.vstack([stim_idx, licking, position_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. Step 5 of the notes explicitly says the output uses a “global categorical mapping over all unique stimulus names across retained sessions.”

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived mainly from `LickFr`, with `LickTrind` used to organize lick frames by trial before alignment.

ii.
```python
lick_frames = np.asarray(record["LickFr"], dtype=float)
lick_trial = np.asarray(record["LickTrind"], dtype=float)
```
```python
lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
```

iii. The notes justify this as building a binary imaging-frame lick vector aligned to the exported trial frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script filters out non-finite and out-of-range lick frames, integer-casts the remaining lick frame numbers, uniquifies them per trial, and marks a retained frame as 1 when its index appears in that trial’s lick set.

ii.
```python
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
lick_frames = lick_frames[valid].astype(int)
...
lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
```
```python
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The notes and checks say the output is a binary time-varying lick trace on the same imaging-frame grid as the rest of the export.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick output is aligned by testing the same `frame_idx` used for the neural slice against the trial’s lick frames.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The notes describe the lick vector as projected directly onto the retained imaging frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level `ft_Pos` signal.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. The notes say raw positions are in decimeters and should be converted to 1 m corridor bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position values are clipped to `[0, 39.999]` decimeters so they stay within the 0-4 m textured corridor, then floor-divided by 10 decimeters to produce four 1 m bins.

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes explicitly map raw decimeter positions to 1 m bins covering the textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories correspond to thresholds at 0, 10, 20, 30, and 40 decimeters, yielding bins `0-1m`, `1-2m`, `2-3m`, and `3-4m`.

ii.
```python
POSITION_OUTPUT_VALUES = ["0-1m", "1-2m", "2-3m", "3-4m"]
```
```python
return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes justify this as matching the decoder task’s required four equal 1 m position categories.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read on the same retained frame indices used for the neural trial.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. The notes state that all exported streams share the same retained-frame alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-level `ft_RunSpeed` signal.

ii.
```python
run_speed = np.asarray(record["ft_RunSpeed"][:spec.estimated_nfr], dtype=np.float32)
...
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The notes say running speed was discretized only because the decoder output must be categorical.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first collects `ft_RunSpeed` values from all retained frames of all valid trials across the selected sessions, computes global 25th/50th/75th percentile edges, and then bins each trial’s retained frame speeds against those edges.

ii.
```python
for trial_idx, (keep_trial, frame_idx) in enumerate(zip(spec.kept_trial_mask, spec.trial_frame_indices)):
    if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
        all_speeds.append(run_speed[frame_idx])
...
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
bins = np.searchsorted(edges, speed, side="right")
return np.clip(bins, 0, 3).astype(np.int16)
```

iii. Step 5 of the notes says the AI intentionally used global quartiles so each speed bin would contain 25% of the retained dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The category thresholds are the global quantile edges returned by `np.quantile(..., [0.25, 0.5, 0.75])`. `np.searchsorted(..., side="right")` assigns a frame to bin 0, 1, 2, or 3.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
def digitize_speed(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. The notes justify this as the decoder task’s “25% bins” requirement.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained frame indices used for the neural matrix.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The notes describe the outputs as time-varying labels defined on the same exported frame grid as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles non-finite `ft_trInd`, `LickFr`, and `LickTrind` by masking them out; clips trial frame indices and licks to the available neural frame count; raises on retinotopy/neuron-count mismatch; and removes trials that fail the minimum-length, duration, or inter-frame-gap filters.

ii.
```python
valid = np.isfinite(ft_tr)
...
frame_idx = frame_idx[frame_idx < nfr]
```
```python
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
...
valid = (lick_frames >= 0) & (lick_frames < nfr)
```
```python
if iarea.shape[0] != nneurons:
    raise ValueError(...)
```

iii. The notes say behavior arrays can be slightly longer than neural recordings, and the stalled-trial filters were added after critical review of extreme timing artifacts.

## 12-a. What are the most time-consuming steps of the code?

i. The script’s main expensive step is loading each full spike recording before neuron selection. Per-trial export is secondary, but the notes emphasize full-session spike I/O as the dominant cost.

ii.
```python
spk = load_spike_matrix(root, spec.base)
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
```

iii. `CONVERSION_NOTES.md` Step 6 says the implementation “still has to load each neural recording file in full before selecting the compact neuron subset, so full conversion will still be I/O-heavy.”

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the repeated per-trial scans in `build_trial_frame_indices`, the per-trial speed accumulation in `prepare_session_specs`, the per-trial lick lookup construction, and the per-trial assembly loop in `process_sessions`.

ii.
```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]
```
```python
for trial in np.unique(lick_trial):
    lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
```
```python
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...
    session_neural.append(neural_trial)
```

iii. The notes say the AI already accepted some extra looping in exchange for a behavior-only preparation pass and per-session streaming of neural data.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats trial filtering and frame-index handling: `trial_passes_quality_filters` is called once during speed-edge preparation and again during final session export, and the same retained-frame indices are re-clipped and used in multiple passes.

ii.
```python
if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
    all_speeds.append(run_speed[frame_idx])
```
```python
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    frame_idx = frame_idx[frame_idx < nfr]
    keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
```

iii. Step 6 of the notes explicitly says the script performs a behavior-only preparation pass before the main conversion pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script keeps extra bookkeeping that the decoder does not use, including `source_files`/`all_keys` tracking in `SessionSpec`, per-session timing/removed-trial metadata, optional diagnostic plotting infrastructure, and median-frame-interval bookkeeping that is only summarized into metadata.

ii.
```python
source_files: set[str] = field(default_factory=set)
all_keys: set[str] = field(default_factory=set)
experiment_types: set[str] = field(default_factory=set)
median_frame_dt_ms: float = np.nan
```
```python
if processing_plots_remaining > 0:
    make_processing_plot(...)
```
```python
"removed_trials": removed_trials[:1000],
"removed_trials_truncated": len(removed_trials) > 1000,
```

iii. The notes frame these as sanity-check and documentation aids rather than parts of the final decoder representation.
