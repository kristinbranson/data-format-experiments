# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy`, scans the imaging `Beh_*.npy` files (excluding three behavior-only pretraining files), groups duplicate behavior views by raw mouse/date/block key, and loads each raw session's spike planes and retinotopy file. Full mode processes all 89 unique raw recordings, with up to four threads.

ii.
```python
exp_info = load_exp_info()
for beh_path in iter_behavior_files():
    beh_all = np.load(beh_path, allow_pickle=True).item()
planes = load_spike_planes(spec.raw_key)
iarea = load_retinotopy(spec.raw_key)
```

iii. The notes say raw imaging recordings, rather than repeated figure-specific behavior views, are the proper session unit because the paper reports 89 recordings and duplicate views reuse the same neural data.

## 1-b. How are the data split into subjects?

i. Subject identity is the mouse name from the experiment index/session key. A sorted unique subject list is built and each session receives its index.

ii.
```python
raw_to_subject[raw_key] = rec["mname"]
subjects = sorted({spec.subject for spec in selected_specs})
subject_idx = np.array([subject_to_idx[spec.subject] for spec in selected_specs], dtype=np.int8)
```

iii. The agent states that the index directly supplies one mouse per raw recording; full conversion yields 19 mice.

## 1-c. How are the data split into sessions?

i. A session is a unique `<mouse>_<year>_<month>_<day>_<block>` raw recording. Duplicate behavior-file views are validated and merged into one `SessionSpec`, giving 89 sessions.

ii.
```python
raw_key = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
for raw_key in sorted(raw_to_views):
    views = raw_to_views[raw_key]
    validate_duplicate_behavior_views(raw_key, views)
```

iii. The notes justify de-duplication because 142 experiment entries and 99 behavior keys refer to only 89 neural recordings.

## 1-d. How are the data split into trials?

i. The agent iterates `ntrials` and selects frames simultaneously labeled with that trial, inside the textured corridor, and moving. Trials therefore omit stationary frames and may be temporally discontinuous.

ii.
```python
for trial in range(ntrials):
    idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
```

iii. The agent interpreted the paper's “only running timepoints” analysis mask as applicable to the decoder and said it removes reward-stop periods while matching published imaging analyses.

## 1-e. How are trials filtered based on quality controls?

i. A trial is discarded only when it has no frame satisfying the trial/corridor/movement mask. The session errors if fewer than two trials remain. No long-trial percentile filter is applied.

ii.
```python
if frame_idx.size == 0:
    continue
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. The notes treat movement filtering as the means of removing long stops. They report 38,110 included trials and do not identify or remove whole extreme-duration trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes directly from the `spks` list in each session's neural file, with one neuron-by-frame array per imaging plane. `iarea` from retinotopy supplies region metadata.

ii.
```python
obj = np.load(path, allow_pickle=True).item()
return obj["spks"]
iarea = load_retinotopy(spec.raw_key)
```

iii. The agent notes that these are already deconvolved Suite2p traces, so no dF/F computation or further deconvolution is needed.

## 2-b. How is the `neural` data processed?

i. Each plane is sliced at the selected running-corridor frames, plane slices are concatenated along the neuron axis, and the result is stored as `float16`. There is no padding or interpolation.

ii.
```python
neural_trial = np.concatenate(
    [plane[:, frame_idx] for plane in planes], axis=0
).astype(np.float16, copy=False)
```

iii. The notes justify plane-wise slicing as a memory optimization and `float16` as reducing an otherwise oversized pickle without affecting the decoder's later float32 buffers.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. Retinotopy codes are mapped to V1, mHV, lHV, aHV, or an added `other` class; all 4,691,034 neurons are retained.

ii.
```python
idx = np.full(iarea.shape, 4, dtype=np.int8)
idx[iarea == 8] = 0
idx[np.isin(iarea, [0, 1, 2, 9])] = 1
return idx
```

iii. The agent says the public code has no additional global neuronal QC and regards region labels as metadata rather than a filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Arrays begin at the first moving textured-corridor frame assigned to a trial, described in metadata as corridor entry. Nonmoving frames occurring later are also removed; trials remain variable length.

ii.
```python
frame_idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
"temporal_alignment_event":
    "first imaging frame assigned to the current corridor trial (corridor entry)"
```

iii. The agent found `StartFr` can precede the first usable corridor/moving frame and therefore chose the combined frame mask instead of slicing from `StartFr`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No explicit temporal rebinning is applied. Stored columns are selected original imaging frames, and metadata reports the nominal 3.17 Hz interval, 315.46 ms. Because nonmoving frames are deleted, adjacent stored columns need not actually be adjacent in time.

ii.
```python
TIME_BIN_MS = 1000.0 / NOMINAL_FRAME_RATE_HZ
"time_bin_size": TIME_BIN_MS
```

iii. The agent says original framewise traces are more suitable than the paper's 60-bin spatial interpolation, while retaining the reference running mask.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundPos` and frame-level `ft_Pos`, not from `SoundFr` and frame timestamps.

ii.
```python
sound_pos = np.asarray(beh["SoundPos"][:int(beh["ntrials"])])
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
```

iii. After raw timestamps produced long pause-inflated values, the agent switched to position and the nominal virtual-corridor speed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue-position minus current-position distance (decimeters) is divided by 6 dm/s. It is positive before and negative after the cue and stored as `float16`.

ii.
```python
cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. The stated rationale is to represent movement time and prevent excluded pauses from inflating the time variable.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. `ft_Pos` is indexed by exactly the same `frame_idx` used to slice neural columns, producing one cue-time value per neural column.

ii.
```python
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
```

iii. The notes' spot checks reconstructed neural and input arrays from raw data and reported equality.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the calendar date and block encoded in each raw session key, relative to that mouse's earliest imaging date.

ii.
```python
date, block = parse_date_and_block(raw_key)
subject_first_date = {subject: min(date for date, _ in date_blocks) ...}
```

iii. The agent says a mouse-relative continuous session day preserves within-mouse chronology.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days since first imaging are computed, with `0.01 * (block - 1)` added to distinguish same-day blocks, then broadcast across every retained trial frame.

ii.
```python
training_day = float((date - subject_first_date[subject]).days) + 0.01 * (block - 1)
day_trial = np.full(frame_idx.size, spec.training_day, dtype=np.float16)
```

iii. The agent preferred elapsed calendar time over an ordinal recording-session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived solely from nonnegative `ft_Pos` at selected frames and a fixed virtual speed, rather than from `StartFr` and `ft` timestamps.

ii.
```python
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. The agent changed from wall-clock time after finding that pause gaps produced values up to roughly 1,765 seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Corridor position in decimeters is divided by 6 dm/s, yielding an idealized movement-time axis of approximately 0–6.7 seconds, stored as `float16`.

ii.
```python
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. The notes argue this matches the paper's fixed 60 cm/s virtual speed and excludes pause duration.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Position is indexed at the same selected frame indices as the neural traces, so each stored neural column has one derived movement-time value.

ii.
```python
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
```

iii. The agent reports raw-to-converted spot checks for input alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the per-trial binary `isRew` field.

ii.
```python
is_rew = np.asarray(beh["isRew"][:int(beh["ntrials"])], dtype=np.int8)
```

iii. The notes define this as rewarded-corridor identity, not actual reward delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial value is cast to a floating binary value and repeated across all retained frames.

ii.
```python
reward_trial = np.full(frame_idx.size, float(is_rew[trial]), dtype=np.float16)
```

iii. No additional processing was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The trial's `WallName` is mapped using `UniqWalls` and `stim_id` pooled across duplicate behavior views, with literal-name fallback for already canonical labels.

ii.
```python
for wall, stim_id in zip(uniq_walls, stim_ids):
    label_map[wall] = CANONICAL_STIM_BY_ID[int(stim_id)]
literal_wall = str(wall_name[trial])
canonical_wall = spec.wall_to_stimulus[literal_wall]
```

iii. The agent says merging views recovers complete labels while avoiding duplicated neural recordings.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Labels are encoded into eight fine-grained categories (`circle1/2/3`, `leaf1/2/3`, and two leaf swaps) and broadcast across all retained frames in the trial.

ii.
```python
OUTPUT_STIMULI = ["circle1", "circle2", "circle3", "leaf1", "leaf2",
                  "leaf3", "leaf1_swap1", "leaf1_swap2"]
np.full(frame_idx.size, stimulus_idx, dtype=np.uint8)
```

iii. The agent chose to preserve distinct presented stimuli and cross-mouse canonical mappings rather than collapse them into broad texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking comes from `LickFr` and is additionally restricted to licks whose `LickTrind` equals the current trial.

ii.
```python
lick_frames = np.asarray(beh["LickFr"], dtype=int)
lick_trials = np.asarray(beh["LickTrind"], dtype=int)
lick_trial_mask = lick_trials == trial
```

iii. The notes describe these as direct event-stream fields and use trial identity as a consistency restriction.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are converted to integers, intersected with retained frame indices, and represented as a binary vector with one if at least one lick lands on the frame.

ii.
```python
frames = np.asarray(lick_frames[trial_lick_mask], dtype=np.int64)
kept = np.intersect1d(frames, frame_idx, assume_unique=False)
out[offsets[valid]] = 1
```

iii. The agent wanted a sparse frame-local binary decoder output and reports checking it against raw lick events.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Only lick frames present in the exact neural `frame_idx` are retained, and their positions are found within that sorted index.

ii.
```python
kept = np.intersect1d(frames, frame_idx, assume_unique=False)
offsets = np.searchsorted(frame_idx, kept)
```

iii. The notes state that this yields one lick label per selected neural frame with no temporal smearing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level `ft_Pos`, expressed in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
```

iii. The agent relies on the raw framewise corridor position and the known 0–4 m textured region.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Selected positions are converted to float, divided by 10 decimeters per meter, floored, and cast to compact integer output.

ii.
```python
bins = np.floor(pos_dm / 10.0).astype(np.int16)
pos_bins = position_to_bins(ft_pos[frame_idx]).astype(np.uint8, copy=False)
```

iii. The notes say this directly implements four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 dm, corresponding to `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` m; values outside are clipped to categories 0 or 3.

ii.
```python
bins = np.floor(pos_dm / 10.0).astype(np.int16)
return np.clip(bins, 0, 3)
```

iii. This follows the requested four equal-length spatial categories.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sliced with the identical `frame_idx` used for neural columns.

ii.
```python
pos_bins = position_to_bins(ft_pos[frame_idx])
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
```

iii. The agent's sanity checks found position progresses from category 0 to 3 on sample trials.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed` at the retained running-corridor frames.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float32)
speed_trial = ft_speed[frame_idx]
```

iii. The notes identify this as the direct per-frame speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from every retained frame in every processed session are concatenated, global 0/25/50/75/100% quantiles are computed, and each trial is digitized using the three interior thresholds.

ii.
```python
all_speeds = np.concatenate([speed for session in processed_sessions
                             for speed in session["speed_trials"]])
quantiles = np.quantile(all_speeds, [0.0, 0.25, 0.5, 0.75, 1.0])
bins = np.digitize(speed_trial, quantiles[1:-1], right=False)
```

iii. The agent says global thresholds provide consistent class meanings across sessions for a cross-session decoder.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories are defined by the three global empirical quartile values. Ties at a boundary all go into the higher bin because `right=False`, so equal class counts are not guaranteed in general.

ii.
```python
np.digitize(speed_trial, quantiles[1:-1], right=False)
```

iii. The agent reports an exactly 25% distribution per category in the final data and records the numeric edges in metadata.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Raw speed is first sliced with each trial's neural frame indices; digitization later preserves the shape and ordering.

ii.
```python
speed_trial = ft_speed[frame_idx]
session["output"][trial_idx][3] = bins
```

iii. The agent treats all frame-level streams as aligned by the common selected frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior arrays are truncated to the neural frame count; empty trials are skipped. Plane frame counts, neuron/retinotopy counts, duplicate behavior arrays, and stimulus mappings are validated, generally by raising errors. Licks outside selected frames disappear through intersection.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nframes])
if any(int(arr.shape[1]) != nframes for arr in planes): raise ValueError(...)
if frame_idx.size == 0: continue
if missing: raise ValueError(...)
```

iii. The agent describes the dataset as internally consistent and favors explicit validation of duplicate views and dimensions rather than silent repair.

## 12-a. What are the most time-consuming steps of the code?

i. Loading hundreds of gigabytes of spike planes, slicing/concatenating them into per-trial matrices, holding and serializing an 81 GiB pickle, and decoder training are the dominant costs. Full conversion is parallelized over four sessions at a time.

ii.
```python
planes = load_spike_planes(spec.raw_key)
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
with ThreadPoolExecutor(max_workers=max_workers) as pool:
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes report an initial artifact over 100 GB, an estimated 18-minute serial run, and a final 5.16-minute parallel conversion after masking and dtype changes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. `get_trial_frame_indices` rescans all session frames once per trial. Lick conversion also loops over trials and performs intersections/searches, and speed assignment makes another nested session/trial pass. Grouping frame indices once by trial and vectorized event lookup could reduce overhead.

ii.
```python
for trial in range(ntrials):
    idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
for session in processed_sessions:
    for trial_idx, speed_trial in enumerate(session["speed_trials"]):
```

iii. The notes focus more on I/O and memory than these loops; they deliberately defer speed binning until all included speeds are available.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly scans frame masks per trial, slices every spike plane per trial, and revisits all trials to install speed bins. Sample-selection diagnostics separately recompute trial frame sets and load retinotopy neuron counts before conversion.

ii.
```python
frame_sets = get_trial_frame_indices(beh, len(beh["ft"]))
nneurons = get_session_neuron_count(spec.raw_key)
for trial, frame_idx in enumerate(trial_frame_sets):
```

iii. The agent accepted these repetitions to keep memory bounded, select representative sample sessions, and postpone globally defined speed thresholds.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `speed_trials` duplicates retained continuous speeds only to compute global bins and is not saved in the final dictionary. Duplicate-view validation and merged label construction inspect arrays not otherwise used by the decoder. Optional plotting computes diagnostics that are also discarded.

ii.
```python
speed_trials.append(speed_trial)
speed_edges = apply_speed_bins(processed_sessions)
"output": [session["output"] for session in processed_sessions]
```

iii. The agent considers the temporary speeds necessary for global quartiles and the other work necessary validation; it explicitly avoided a full-session spike concatenation that would create a much larger discarded copy.
