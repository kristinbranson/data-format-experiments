# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans every `Beh_*.npy` behavior file under `data/beh`, groups keys by the first five underscore-separated fields of the behavior key (`<mouse>_<YYYY>_<MM>_<DD>_<blk>`), and chooses one representative behavior record per recording base with a heuristic `session_key_score`. It reads `Imaging_Exp_info.npy` only to attach experiment-type labels, not to drive the main session list. During processing it then loads one spike file from `data/spk` and one retinotopy file from `data/retinotopy` per selected session.

ii. 
```python
def load_experiment_type_map(root: Path) -> dict[str, set[str]]:
    exp_info = np.load(root / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
```

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

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies using “89 unique recording bases as sessions” to match the paper’s “89 recordings in 19 mice” and avoid duplicate experiment labels or `swap1`/`swap2` aliases.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the mouse name parsed from the recording base. The script stores all unique subject names in sorted order and builds `subject_idx` from that sorted list.

ii. 
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    subject = parts[0]
```

```python
subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
subject_idx.append(subject_to_idx[spec.subject])
```

iii. The notes say the session unit is the unique recording base and the mouse name from that base is the subject identity.

## 1-c. How are the data split into sessions?

i. A session is one unique recording base `<mouse>_<date>_<blk>`. If multiple behavior keys map to the same base, the AI keeps only one representative key and treats that as the session’s behavior record.

ii. 
```python
base = "_".join(key.split("_")[:5])
if base not in selected:
    selected[base] = SessionSpec(
        base=base,
        key=key,
        record=record,
        subject=subject,
        date=date,
        blk=blk,
    )
```

```python
if session_key_score(key, record) < session_key_score(current.key, current.record):
    current.key = key
    current.record = record
```

iii. Step 5 says this matches the paper’s 89 recordings and avoids duplicated metadata entries that share the same raw neural recording.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from frame-level behavior arrays. For each trial index from `0` to `ntrials - 1`, the AI collects frame indices where `ft_trInd` equals that trial, `ft_CorrSpc` is true, and `ft_move > 0`. It keeps variable-length frame lists rather than enforcing a fixed trial length.

ii. 
```python
def build_trial_frame_indices(record: dict, nfr: int) -> list[np.ndarray]:
    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    valid_idx = np.flatnonzero(valid)
    trial_ids = ft_tr[valid].astype(int)
    keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
    keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
```

```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]
    frame_indices.append(trial_frames.astype(np.int32, copy=False))
```

iii. Step 5 explicitly says the AI chose “paper-style running corridor frames (`ft_CorrSpc` and `ft_move > 0`)” because the published analyses consistently use running timepoints in the textured corridor.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have fewer than 5 retained timepoints, if the retained segment extends more than 60 s after `Trial_start_time`, or if any retained inter-frame gap exceeds 10 s. Sessions with fewer than 2 surviving trials are removed entirely.

ii. 
```python
if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
    return False, "too_few_timepoints"
```

```python
retained_duration_s = float((frame_times[-1] - float(record["Trial_start_time"][trial_idx])) * SEC_PER_DAY)
if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
    return False, "retained_duration_gt_60s"
```

```python
if frame_idx.size > 1:
    max_gap_s = float(np.max(np.diff(frame_times)) * SEC_PER_DAY)
    if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
        return False, "interframe_gap_gt_10s"
```

iii. Step 10 says these filters were added after finding sparse stalled trials with extreme timing outliers that distorted trial-start alignment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the stored deconvolved traces `spks` in `data/spk/<base>_neural_data.npy`, concatenated across planes. Per-neuron region labels come from `iarea` in `data/retinotopy/<mouse>_<date>_trans.npz`.

ii. 
```python
spk_item = np.load(path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
```

```python
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=float)
```

iii. The notes state that the paper and reference code treat the stored `spks` arrays as the deconvolved fluorescence signal directly, without any dF/F recomputation.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, selects a subset of neurons with a d-prime-based filter, and for each kept trial extracts only the retained running-corridor frames. Trials are stored at variable length and cast to `float32`; there is no fixed 32-frame crop or padding.

ii. 
```python
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
region_idx = region_idx[selected_neurons]
```

```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
session_neural.append(neural_trial)
```

iii. In Step 5 and Step 6, the AI justifies this as a “compact, paper-grounded subset” that uses the paper’s stimulus-selective logic to keep full-dataset decoding tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first maps neurons into `V1`, `mHV`, `lHV`, `aHV`, or `other` from `iarea`. It then keeps a corridor-responsive subset of neurons, selecting the top 5% and bottom 5% of d-prime values within each of the first four visual-area groups; if there are too few candidates, it falls back to all neurons in those four areas.

ii. 
```python
region_idx = np.full(nneurons, 4, dtype=np.int16)
region_idx[iarea == 8] = 0
region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1
region_idx[np.isin(iarea, [5, 6])] = 2
region_idx[np.isin(iarea, [3, 4])] = 3
```

```python
dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
...
hi, lo = np.nanpercentile(dp[candidates], [95, 5])
selected |= candidates & ((dp >= hi) | (dp <= lo))
```

iii. Step 5 says the AI chose this because the paper’s coding-direction analyses use top 5% positive and negative selective neurons per area, and because the raw dataset is too large for practical full decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats corridor entry / trial start as the alignment event, but it keeps only the running textured-corridor frames from each trial after that event. Alignment is implicit through the trial-specific `frame_idx` lists and the paired time inputs derived from `Trial_start_time`.

ii. 
```python
keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
```

```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

```python
"temporal_alignment_event": "corridor entry / trial start",
"off_start": 0.0,
"off_end": None,
```

iii. The notes argue that keeping native running-corridor frames is the closest match to the paper’s analyses while still satisfying the decoder format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the native imaging-frame grid. The time bin size is recorded as the median frame interval across sessions in milliseconds, and no temporal resampling is applied.

ii. 
```python
dt_ms = np.diff(ft) * MS_PER_DAY
spec.median_frame_dt_ms = float(np.median(dt_ms))
```

```python
"time_bin_size": float(np.median(frame_dt_medians)),
"frame_bin_source": "native imaging frame timestamps; no temporal resampling",
```

iii. Step 5 explicitly says the AI chose raw frame timestamps because the median spacing is already very consistent across sessions.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the absolute cue timestamp `SoundTime` and the frame timestamps `ft` for the retained frames of each trial.

ii. 
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. The notes say cue timing should come directly from raw timestamps and stay on the native frame grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the AI computes `SoundTime - ft` in seconds, producing a continuous time-varying input that is positive before the cue and negative after it.

ii. 
```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. Step 10 describes this as direct construction from the raw behavior timestamps `SoundTime` and `ft`, without resampling to a synthetic clock.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same retained `frame_idx` used to slice the per-trial neural matrix, so it is frame-by-frame aligned to the neural data.

ii. 
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
input_trial = np.vstack(
    [time_to_cue, training_day, time_since_start, reward_available]
).astype(np.float32, copy=False)
```

iii. The notes repeatedly state that native frame timestamps are preserved so behavioral inputs stay on the same frame grid as neural activity.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the subject id and session date parsed from the recording base string.

ii. 
```python
subject, date, blk = parse_base(base)
```

```python
first_date_by_subject.setdefault(spec.subject, spec.date)
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. Step 5 says the paper does not define this decoder variable, so the AI derives it from raw session dates as a reproducible proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are sorted by subject and date, then `training_day` is set to the elapsed calendar days since that subject’s first retained imaging session. That scalar is repeated across all timepoints of each trial.

ii. 
```python
specs = sorted(selected.values(), key=lambda s: (s.subject, s.date, int(s.blk)))
...
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

```python
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. Step 5 explicitly justifies this as “elapsed days since the subject’s first retained imaging session.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the absolute trial-start timestamp `Trial_start_time` and the frame timestamps `ft`.

ii. 
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. The notes say raw timestamps should be used directly wherever possible instead of interpolating to a new time grid.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the AI computes `ft - Trial_start_time` in seconds, producing a continuous time-varying signal beginning at or after trial start.

ii. 
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. Step 10 cites `Trial_start_time` and `ft` as the direct raw sources for this decoder input.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is built on the same retained `frame_idx` used for the per-trial neural matrix, so it is sample-aligned to the neural frames.

ii. 
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
```

iii. The notes describe the whole export as preserving native frame-time alignment between neural and behavioral streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` flag in the behavior record.

ii. 
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. Step 5 maps `beh['isRew']` directly to `reward_available`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the per-trial `isRew` value to a boolean/float and repeats it across every retained frame of that trial.

ii. 
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. The notes describe this as a decoder-format broadcast of a trial-level variable rather than a paper-specific transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The per-trial output is taken from `WallName`. The global label vocabulary is collected from `UniqWalls` across the selected sessions.

ii. 
```python
def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories
```

```python
stim_idx = np.full(
    frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])],
    dtype=np.int16,
)
```

iii. Step 5 says the AI wanted a global categorical mapping over all unique stimulus names seen in the imaging data.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps the fine-grained stimulus names rather than collapsing them to broader texture families. It builds a sorted global vocabulary of unique wall names, maps each trial’s `WallName` to that integer index, and repeats the category across the retained frames of the trial.

ii. 
```python
visual_categories = collect_visual_categories(specs)
visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}
```

```python
stim_idx = np.full(
    frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])],
    dtype=np.int16,
)
```

iii. In Step 5, the notes justify this by listing 15 global categories in the imaging data and tying the choice to the decoder task’s requirement for categorical stimulus labels.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`. `LickFr` provides lick frame numbers and `LickTrind` assigns those licks to trials.

ii. 
```python
lick_frames = np.asarray(record["LickFr"], dtype=float)
lick_trial = np.asarray(record["LickTrind"], dtype=float)
```

```python
lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
```

iii. Step 10 cites `LickFr` as the primary source and notes that the converted licking output was checked directly against the raw files.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI filters invalid lick entries, truncates lick frame numbers to integers, clips them to the imaged range, groups them by trial, and marks each retained frame as `1` if it appears in that trial’s lick-frame set and `0` otherwise.

ii. 
```python
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
lick_frames = lick_frames[valid].astype(int)
lick_trial = lick_trial[valid].astype(int)
valid = (lick_frames >= 0) & (lick_frames < nfr)
```

```python
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The notes describe this as a direct binary per-frame reconstruction of licking aligned to imaging frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking vector is constructed only on the retained `frame_idx` for that trial, so it is aligned frame-by-frame with the sliced neural data.

ii. 
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Step 10 reports raw sanity checks confirming exact agreement between converted licking traces and raw data on sampled trials.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level `ft_Pos`, which is measured in decimeters.

ii. 
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes say the raw corridor is represented in decimeters and the decoder output should use 1 m bins across the 0-4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips positions to the textured corridor range, floors by 10 decimeters, and stores the result as one of four categorical bin ids for each retained frame.

ii. 
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Step 5 explicitly maps raw `ft_Pos` to four 1 m bins using edges `[0, 10, 20, 30, 40]`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1 m bins are used: `0-1m`, `1-2m`, `2-3m`, and `3-4m`, corresponding to decimeter ranges `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40)`.

ii. 
```python
POSITION_OUTPUT_VALUES = ["0-1m", "1-2m", "2-3m", "3-4m"]
```

```python
return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes say this thresholding is driven by the decoder task specification and the paper’s 4 m textured-corridor geometry.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read at the same retained frame indices as the neural data and therefore stays frame-aligned to each per-trial neural slice.

ii. 
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. The notes emphasize that all time-varying variables are exported on the same retained native frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed`.

ii. 
```python
run_speed = np.asarray(record["ft_RunSpeed"][:spec.estimated_nfr], dtype=np.float32)
...
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Step 5 maps `beh['ft_RunSpeed']` directly to the running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI pools all retained running-corridor frame speeds across the selected sessions, computes global quartile thresholds with `np.quantile`, and then bins each retained frame’s `ft_RunSpeed` using those thresholds.

ii. 
```python
speed_values = np.concatenate(all_speeds).astype(np.float32, copy=False)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```

```python
def digitize_speed(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. Step 5 explicitly justifies “global quartiles” because the decoder task asks for 25% bins across the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four categories `q1` through `q4` are defined by three global speed thresholds at the 25th, 50th, and 75th percentiles of all retained running-corridor frames.

ii. 
```python
RUNNING_SPEED_OUTPUT_VALUES = ["q1", "q2", "q3", "q4"]
```

```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
bins = np.searchsorted(edges, speed, side="right")
```

iii. The notes say these edges are saved in metadata and intended to make the full dataset split into 25% speed bins overall.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained `frame_idx` as the neural data and converted to bins frame-by-frame on that shared grid.

ii. 
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The notes describe this as keeping all decoder streams on the retained native imaging frames.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI adds several defensive checks. It ignores non-finite `ft_trInd`, `LickFr`, and `LickTrind` values, clips all trial and lick frame indices to the neural frame count, raises an error if retinotopy neuron counts do not match spikes, and removes trials with too few retained points or pathological timing gaps/durations.

ii. 
```python
valid = np.isfinite(ft_tr)
valid_idx = np.flatnonzero(valid)
```

```python
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
...
valid = (lick_frames >= 0) & (lick_frames < nfr)
```

```python
if iarea.shape[0] != nneurons:
    raise ValueError(
        f"Retinotopy neuron count mismatch for {base}: iarea={iarea.shape[0]}, spikes={nneurons}"
    )
```

iii. Step 10 says the additional trial filters were introduced after inspecting severe timing outliers in sparse retained trial segments.

## 12-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify full spike-file loading as the dominant cost, because each session’s entire neural recording must be loaded before neuron selection can discard most neurons. Trial-by-trial session processing is secondary.

ii. 
```python
spk = load_spike_matrix(root, spec.base)
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
```

iii. Step 6 says the implementation is still I/O-heavy because each neural recording is loaded in full before the compact subset is chosen.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code loops over every trial at least twice: once to build per-trial frame indices and again to build exported trial arrays. It also loops over areas inside neuron selection. The notes mention this kind of repeated per-trial work as an efficiency limitation, although I/O remains the dominant cost.

ii. 
```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]
    frame_indices.append(trial_frames.astype(np.int32, copy=False))
```

```python
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...
    session_neural.append(neural_trial)
```

iii. Step 6 notes that behavior-derived trial frame indices are prepared in one pass, but the export still iterates trial-by-trial during the main conversion.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats some work across passes. It first computes `trial_frame_indices` and trial-level filtering inputs in `prepare_session_specs`, then checks trial quality again in `process_sessions`. It also repeatedly re-reads arrays like `ft_Pos` and `ft_RunSpeed` inside the per-trial loop.

ii. 
```python
spec.trial_frame_indices = build_trial_frame_indices(record, spec.estimated_nfr)
...
if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
    all_speeds.append(run_speed[frame_idx])
```

```python
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...
    keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
```

iii. Step 6 explicitly describes a behavior-only preparation pass followed by a second per-session conversion pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does extra bookkeeping and optional diagnostics that are not used by the decoder itself, including collecting `source_files`/`all_keys`, computing and saving diagnostic plots with `--show-processing`, and keeping large removal logs in metadata. It also computes several session-level preparation fields only to support conversion logistics rather than downstream modeling.

ii. 
```python
selected[base].source_files.add(beh_path.name)
selected[base].all_keys.add(key)
selected[base].experiment_types.add(exp_type)
```

```python
if processing_plots_remaining > 0:
    ...
    make_processing_plot(...)
```

```python
"removed_trials": removed_trials[:1000],
"removed_trials_truncated": len(removed_trials) > 1000,
```

iii. The notes present these as sanity-check and documentation machinery rather than part of the decoder-facing dataset construction.
