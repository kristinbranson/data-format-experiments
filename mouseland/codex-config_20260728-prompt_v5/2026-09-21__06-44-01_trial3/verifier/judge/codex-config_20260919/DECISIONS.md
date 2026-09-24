# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy`, loads each referenced behavior file to enumerate every `(mouse, date, block)` view, chooses one canonical behavior view per unique recording, and then loads that session's behavior, all spike planes, and retinotopy assignments. Full mode processes all 89 unique recordings; sample mode selects two scored sessions.

ii.
```python
exp_info = load_exp_info()
session_views = collect_session_views(exp_info)
canonical_sessions = choose_canonical_view(session_views)
beh = load_behavior(view)
planes = load_spike_planes(triplet)
iarea = load_iarea(triplet)
```

iii. The notes justify deduplicating 142 experiment entries/99 behavior keys to the 89 underlying recordings reported in the paper. Canonical views are ranked for finite `stim_id` coverage and wall coverage to avoid analysis-specific duplicate views.

## 1-b. How are the data split into subjects?

i. Subjects are mouse IDs (`mname`) from each session triplet. The sorted unique IDs form `subjects`, and each session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({triplet[0] for triplet, _view, _exp_types in sessions})
subject_lookup = {name: idx for idx, name in enumerate(subjects)}
data["subject_idx"].append(subject_lookup[mouse])
```

iii. The notes state that mouse identity is directly supplied by `Imaging_Exp_info.npy`; the full output contains 19 mice.

## 1-c. How are the data split into sessions?

i. A session is a unique `(mname, datexp, blk)` recording. Duplicate experiment-specific views are grouped by that triplet and one view is selected using `behavior_view_priority` rather than treating duplicates as separate sessions.

ii.
```python
triplet = (rec["mname"], rec["datexp"], rec["blk"])
views[triplet].append(SessionView(...))
candidates.append((behavior_view_priority(beh, view.has_stimtype), view))
out.append((triplet, candidates[0][1], sorted(set(exp_types))))
```

iii. The agent argues that duplicate views are reinterpretations of the same neural recording and that 89 unique triplets match the paper's 89 recordings.

## 1-d. How are the data split into trials?

i. Within each session the agent loops over `range(ntrials)` and selects native imaging frames whose rounded `ft_trInd` equals the trial and whose `ft_CorrSpc` flag is true. Trials therefore span the variable-length 4 m texture corridor and exclude gray space.

ii.
```python
frame_trial[valid_frame_trial] = np.rint(frame_trial_raw[valid_frame_trial]).astype(int)
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
```

iii. The notes say corridor-only trials align at corridor entry, make the requested four 1 m position bins well defined, and retain native time needed for timing inputs.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two corridor frames, an unknown stimulus family, nonfinite cue/start times, or any nonfinite neural/input/output/speed value are skipped. The agent does not remove excessively long stalled trials; in the full run no trial was skipped.

ii.
```python
if mask.sum() < 2:
    skipped_trials += 1
    continue
if family not in family_lookup:
    ...
if not np.isfinite(cue_time) or not np.isfinite(start_time):
    ...
if not (np.all(np.isfinite(neural_trial)) and ...):
    skipped_trials += 1
    continue
```

iii. The agent treated a 5,607-frame, 29-minute stalled trial as genuine behavior rather than an indexing error and retained all 38,110 raw trials. It did not adopt the reference's 99th-percentile traversal-length exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each plane in `['spks']` of `<mouse>_<date>_<block>_neural_data.npy`. Retinotopy `iarea`, behavior `ft_WallID`, `ft_move`, `ft_CorrSpc`, `ft_GraySpc`, `stim_id`, and `UniqWalls` are additionally used to select neurons.

ii.
```python
obj = np.load(path, allow_pickle=True).item()
return [np.asarray(x) for x in obj["spks"]]
selected_blocks, region_idx, region_counts, selection_meta = select_neurons(planes, beh, iarea)
```

iii. The notes recognize `spks` as already-deconvolved fluorescence and use behavior/retinotopy for a paper-inspired selective-neuron subset.

## 2-b. How is the `neural` data processed?

i. The agent computes stimulus d′ and corridor responsiveness, takes up to the 64 strongest eligible neurons in each of four regions, extracts native corridor frames per trial, concatenates selected plane blocks, and casts to `float16`. It performs no dF/F computation, temporal resampling, normalization, or padding.

ii.
```python
dp = compute_dprime(plane[:, stim_a_fr], plane[:, stim_b_fr])
candidates.sort(key=lambda x: x[0], reverse=True)
chosen = candidates[:N_PER_REGION]
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays], axis=0
).astype(np.float16, copy=False)
```

iii. The stated reason is to keep the dense export tractable while approximating the paper's selective-neuron logic and preserving real timing for the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons must lie in V1, mHV, lHV, or aHV, be corridor responsive, have finite d′, and preferably satisfy `abs(d′) >= 0.3`; if a region has no strong neurons, the threshold is relaxed to any responsive neuron. At most 64 highest-absolute-d′ neurons per region are retained.

ii.
```python
region_pool = region_mask & corr_neu & np.isfinite(dp)
strong_pool = region_pool & (np.abs(dp) >= 0.3)
chosen_pool = strong_pool if strong_pool.any() else region_pool
chosen = candidates[:N_PER_REGION]
```

iii. The agent calls this a balanced paper-style selective subset needed to bound memory/output size. The full output retains 22,163 of 4,691,034 raw neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural columns use the same corridor-frame mask defined by trial identity and `ft_CorrSpc`; the first retained frame is corridor entry/trial start. Trials remain variable length and are neither padded nor cut to a fixed duration.

ii.
```python
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
neural_trial = np.concatenate([selected_plane[:, mask] ...], axis=0)
```

iii. The agent says native corridor frames preserve timing and that variable-length trials are supported; metadata records corridor entry with `off_start = 0.0` and `off_end = None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One native imaging frame is one bin. No rebinning is applied. The metadata bin size is the median, across selected sessions, of each session's median `diff(ft)`, converted from days to milliseconds (about 315 ms).

ii.
```python
dts.append(float(np.nanmedian(np.diff(ft)) * SECONDS_PER_DAY * 1000.0))
return float(np.median(dts)), dts
```

iii. The notes favor native frame timing because all framewise streams share this grid and timing inputs would be degraded by position interpolation.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundTime` and frame-level `ft`.

ii.
```python
trial_times = ft[mask]
cue_time = float(beh["SoundTime"][trial_idx])
```

iii. The notes identify these as the stored raw timing variables used by the paper for cue alignment.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue time minus each frame time is multiplied by 86,400 to convert MATLAB-day units to seconds; it is positive before and negative after the cue.

ii.
```python
(cue_time - trial_times) * SECONDS_PER_DAY
```

iii. The agent chose a continuous time-varying value matching the requested variable and documented negative values after cue onset.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. `trial_times` is indexed by exactly the same trial corridor mask used to select neural columns, yielding one timing value per neural bin.

ii.
```python
trial_times = ft[mask]
neural_trial = np.concatenate([selected_plane[:, mask] ...], axis=0)
```

iii. The justification is exact shared framewise alignment; a raw-data sanity check reportedly matched the reconstruction exactly.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp`, block number, and the earliest recording date for each mouse in the selected session list.

ii.
```python
first_date = entries[0][0]
out[triplet] = float((dt - first_date).days + 0.01 * (blk - 1))
```

iii. The agent states that the data lack an explicit training-day field, so elapsed calendar days from the first recording are its preferred continuous proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are chronologically sorted per mouse. Calendar-day difference from the first session is computed, with `0.01*(block-1)` added to distinguish same-day blocks, then broadcast over all frames of every trial.

ii.
```python
per_subject[mouse].append((parse_date(datexp), int(blk), triplet))
np.full(mask.sum(), day_value, dtype=np.float32)
```

iii. The notes call calendar elapsed time the most defensible continuous training-day proxy and report a 0–92 day range.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from trial-level `Trial_start_time` and frame-level `ft`.

ii.
```python
start_time = float(beh["Trial_start_time"][trial_idx])
trial_times = ft[mask]
```

iii. These are identified as direct raw timing fields, with corridor entry treated as trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The trial start timestamp is subtracted from every retained frame timestamp and multiplied by 86,400 to obtain seconds.

ii.
```python
(trial_times - start_time) * SECONDS_PER_DAY
```

iii. The agent chose the direct continuous time difference required by the decoder task.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses `ft[mask]` with the identical corridor/trial mask used for neural columns.

ii.
```python
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
trial_times = ft[mask]
```

iii. The agent reports exact raw-versus-converted sanity-check agreement.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from per-trial `isRew`.

ii.
```python
float(bool(beh["isRew"][trial_idx]))
```

iii. The notes describe `isRew` as the raw binary reward-availability field.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is cast through `bool` to 0/1 float and repeated across every frame of the trial.

ii.
```python
np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)
```

iii. This preserves unrewarded sessions as zero and produces the requested per-trial binary input in the required time-shaped tensor.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`.

ii.
```python
wall_name = str(beh["WallName"][trial_idx])
family = family_from_wall_name(wall_name)
```

iii. The notes prefer `WallName` because it records the presented texture consistently, whereas `stim_id`/`TrialStim` can be remapped across duplicate analysis views.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The lowercase leading family is mapped to circle, leaf, rock, or wood, converted to an integer index, and repeated over the trial. Unknown families are skipped.

ii.
```python
for prefix in ("circle", "leaf", "rock", "wood", "brick"):
    if lowered.startswith(prefix): return prefix
stim_out = np.full(mask.sum(), family_lookup[family], dtype=np.int16)
```

iii. The agent says coarse families best match the requested category (e.g. circle, leaf) and avoid treating crops/swaps as distinct categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session lick-event frame numbers in `LickFr`.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
```

iii. The notes identify `LickFr` as the frame-indexed lick source; `LickTrind` was considered but is not used in the implementation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are floored to integers, invalid/out-of-range events are dropped, and valid frames are marked 1 in an otherwise-zero binary vector. Multiple licks in one frame remain 1.

ii.
```python
lick_idx = np.asarray(np.floor(lick_fr), dtype=int)
valid = (lick_idx >= 0) & (lick_idx < nfr)
lick_vec[lick_idx[valid]] = 1
```

iii. This follows the reference convention of integer frame indexing and yields the required binary time-varying output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session-wide frame vector is sliced with the same trial corridor mask used for neural data.

ii.
```python
lick_out = lick_vec[mask].astype(np.int16)
```

iii. The shared imaging-frame grid is the stated alignment mechanism.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level `ft_Pos` on retained corridor frames.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)[:nfr]
pos_out = position_to_bin(ft_pos[mask])
```

iii. The notes explain that raw position is in decimeter units and the texture corridor spans 0–40 (4 m).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10, floored, cast to integer, and clipped to category indices 0–3.

ii.
```python
bins = np.floor(np.asarray(pos, dtype=float) / 10.0).astype(np.int16)
return np.clip(bins, 0, 3)
```

iii. This directly implements four equal 1 m bins across the requested 4 m texture corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 raw decimeter units, corresponding to `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` m; clipping sends any edge/out-of-range value to the nearest end bin.

ii.
```python
np.floor(np.asarray(pos, dtype=float) / 10.0)
np.clip(bins, 0, 3)
```

iii. The paper/data unit reconciliation in the notes supports these fixed physical thresholds.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed with exactly the same frame mask as neural activity.

ii.
```python
pos_out = position_to_bin(ft_pos[mask])
```

iii. Both streams already live on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed` for every retained corridor frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)[:nfr]
speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
```

iii. The notes identify this as the direct raw speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All speeds from all retained trials and sessions are concatenated, stable-sorted globally, divided by rank into four nearly equal chunks, and then scattered back into their original trial arrays. Quantile edges/ranges are metadata only.

ii.
```python
all_speed_values = np.concatenate([...])
order = np.argsort(all_speed, kind="mergesort")
for bin_idx, idx_chunk in enumerate(np.array_split(order, 4)):
    speed_bins[idx_chunk] = bin_idx
```

iii. The agent wanted exact 25% occupancy despite a large tie at zero and chose global bins to avoid session-specific category definitions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is not thresholded by unique speed values. Categories are global rank quartiles: the lowest quarter of frame ranks is 0 through the highest quarter as 3; stable ordering arbitrarily separates tied values across bins.

ii.
```python
for bin_idx, idx_chunk in enumerate(np.array_split(order, 4)):
    speed_bins[idx_chunk] = bin_idx
edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
```

iii. Threshold digitization initially produced imbalanced occupancy because many speeds equal zero, so the agent changed to rank assignment and obtained exact quarter counts.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Raw speed is first selected with the neural frame mask. After global rank assignment, bins are returned sequentially using each saved trial's frame count, preserving original order.

ii.
```python
speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
trial_speed_bins = all_speed_bins[cursor : cursor + n_time]
data["output"][session_idx][trial_idx] = np.vstack([...])
```

iii. The cursor is checked against the global vector length, and the notes report exact raw reconstruction checks.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural frame count is authoritative: recognized framewise behavior arrays are trimmed to `nfr`; out-of-range lick events are discarded; nonfinite trial identifiers are invalid; trials with insufficient frames, unknown families, nonfinite times, or nonfinite converted values are skipped. Count mismatches and sessions with fewer than two retained trials raise errors.

ii.
```python
trimmed[key] = value[:nfr]
valid = (lick_idx >= 0) & (lick_idx < nfr)
if total_neurons_raw != iarea.shape[0]: raise ValueError(...)
if len(neural_trials) < 2: raise ValueError(...)
```

iii. The notes say behavior commonly extends 1–3 frames beyond imaging and that trimming to neural length matches the reference. Full conversion skipped no trials.

## 12-a. What are the most time-consuming steps of the code?

i. Loading very large dense spike files and per-session neuron selection/d′ computation dominate. The notes estimate about 15.4 seconds per session and 22.8 minutes for 89 sessions.

ii.
```python
planes = load_spike_planes(triplet)
selected_blocks, region_idx, region_counts, selection_meta = select_neurons(planes, beh, iarea)
```

iii. The agent notes 4.69 million raw neurons and large dense files; it processes one session at a time and caps neurons to control runtime and memory.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loop over every trial repeatedly builds masks over all session frames; candidate tuple construction and per-region/per-plane grouping also use Python loops. Trial segmentation could be grouped in one pass, though variable-length outputs still require per-trial assembly.

ii.
```python
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
for score, idx in zip(scores.tolist(), local_idx.tolist()):
    ...
```

iii. The agent did not explicitly document these as remaining vectorization targets. It did document reverting a pre-concatenation optimization after it slowed sample conversion.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files/dictionaries are repeatedly loaded during view collection, canonical-view scoring, frame-timing computation, sample scoring, and final conversion. Each trial also rescans the full session trial-index vector to construct its mask.

ii.
```python
beh_all = np.load(beh_path, allow_pickle=True).item()
beh = np.load(view.beh_path, allow_pickle=True).item()[view.key]
beh = load_behavior(view)
```

iii. The notes mention a lightweight timing pre-pass but do not acknowledge all repeated behavior loads; canonical deduplication is presented as preventing more expensive repeated spike work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes canonical-view priorities from behavior dictionaries, per-session d′/responsiveness statistics, speed quantile edges/value ranges, and detailed metadata; these are not used by decoder training. In non-plot runs it still stores `session_plot_info`, and `selected_blocks` carries a region ID that is not used when extracting arrays.

ii.
```python
selection_meta = {"stimulus_pair_for_selection": ..., "corridor_responsive_fraction": ...}
edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
session_plot_info.append({...})
for plane_idx, local_idx, _region_id in selected_blocks:
```

iii. Most of this is justified as provenance, sanity checking, or optional plotting. The agent does not identify it as discarded downstream processing.
