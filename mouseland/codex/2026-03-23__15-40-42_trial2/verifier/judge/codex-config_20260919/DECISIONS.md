# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Loads the master `Imaging_Exp_info.npy` index, all canonical behavior dictionaries, per-session deconvolved spike files through `utils.load_spk`, and matching retinotopy files. It first collects the global stimulus vocabulary and timing metadata, then processes all 89 canonical sessions.

ii.
```python
exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. The notes say this covers all matched imaging recordings and deduplicates 142 analysis entries to the paper's 89 recordings.

## 1-b. How are the data split into subjects?

i. Uses `mname` as subject, sorts unique names, and creates one `subject_idx` per processed session.

ii.
```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
"subject_idx": np.asarray([subject_to_idx[s.subject] for s in processed_sessions])
```

iii. The notes report 19 unique mice and say `mname` is the explicit subject identifier.

## 1-c. How are the data split into sessions?

i. Defines a session by mouse, date, and block; groups duplicate index entries by this ID and chooses one canonical entry, preferring entries without `stimtype`.

ii.
```python
session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
grouped[session_id].append(SessionCandidate(...))
return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]
```

iii. Duplicate analysis labels refer to the same neural recording; notes state either swap copy has duplicated framewise data and `WallName` preserves identity.

## 1-d. How are the data split into trials?

i. For each raw trial, constructs the half-open frame interval `StartFr:GrayFr`, then removes every frame where `ft_move <= 0`; retained trials therefore contain only moving texture-corridor frames and can have temporal gaps.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
```

iii. The agent interpreted the paper's running-only analysis rule as requiring dropped stationary frames and used `GrayFr` to end at the 4 m texture boundary.

## 1-e. How are trials filtered based on quality controls?

i. Drops only trials having no moving frames after the mask. It retains all other trials and does not apply a trial-duration outlier cutoff.

ii.
```python
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. The notes say removing stopped frames eliminates extremely long pauses while retaining all 38,110 trials in this dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Derives neural activity from the per-plane deconvolved `spks` loaded by `utils.load_spk`; retinotopy `iarea` supplies region labels used during selection.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
_, region_idx_all = area_labels_from_iarea(ret["iarea"])
```

iii. The notes identify these as already deconvolved Suite2p traces, so no dF/F or deconvolution is recomputed.

## 2-b. How is the `neural` data processed?

i. Selects a neuron subset, slices each trial to moving texture frames, and stores float32 framewise traces without rebinning, normalization, or padding.

ii.
```python
spk_sel = spk[selected_mask]
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. It sought paper-style task-relevant neurons and native-frame data suitable for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Keeps visual-area neurons with absolute stimulus d-prime at least 0.3, plus qualifying aHV reward-prediction neurons; if fewer than 64 survive, keeps top-|d-prime| visual neurons. `other` neurons are excluded by the selection masks.

ii.
```python
stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)
reward_pred = ahv_mask & ... & (dp_sound > DP_THRESHOLD) & (reward_dp >= reward_dp_thr)
selected = stim_selective | reward_pred
```

iii. The notes link these thresholds to two paper analyses and say the subset makes conversion tractable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligns each trial at corridor entry (`StartFr`) but retains only moving frames until `GrayFr`; arrays start at the first retained frame and are not padded.

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
neural = spk_sel[:, frames]
```

iii. The agent says corridor entry is the requested event and running-only frames follow the paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Reports the median native imaging interval across sessions (about 315 ms) as bin size and performs no aggregation/resampling, but deletes non-moving frames.

ii.
```python
time_bin_ms = compute_time_bin_ms(full_catalog)
"time_bin_size": time_bin_ms
```

iii. The imaging frame is treated as the native bin; the notes claim no temporal interpolation is needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Uses trial `SoundFr`, the retained frame indices, and a median frame interval derived from `ft`.

ii.
```python
sound_fr = np.asarray(beh["SoundFr"], dtype=int)
dft = np.diff(ft); frame_dt = float(np.median(dft[dft > 0]) * SECONDS_PER_DAY)
```

iii. The notes originally planned actual cue/frame timestamps, but the trajectory records a later switch to a retained-running-frame axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Finds where `SoundFr` would be inserted among retained frames and multiplies the difference in retained-sample indices by the median native frame duration; values are positive before and negative after cue.

ii.
```python
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The trajectory says absolute elapsed time looked too large after stationary frames were dropped, so the agent intentionally switched to compressed running-frame time.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Produces one cue-time value for every retained neural column using the same `frames` ordering.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_to_cue = (cue_idx - retained_idx) * frame_dt
```

iii. The agent emphasizes uniform time-varying arrays aligned to neural samples.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derives training day from the calendar date embedded in each catalog entry, grouped by `mname`.

ii.
```python
date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
```

iii. A complete explicit training-day label was unavailable, so calendar date was chosen as a proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Computes elapsed calendar days from each mouse's first recording and broadcasts that value over every retained trial frame.

ii.
```python
float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
day = np.full(frames.shape, session_day, dtype=np.float32)
```

iii. The notes explicitly call this a training-day proxy and report a 0–92 range.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Uses `StartFr`, retained frame order, and the median interval calculated from `ft`.

ii.
```python
start_fr = np.asarray(beh["StartFr"], dtype=int)
retained_idx = np.arange(frames.size, dtype=np.float32)
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
```

iii. The intended source was corridor-entry and frame timing; the later implementation uses compressed moving-frame time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Sets the first retained moving sample to zero and increments by one median frame interval per retained sample, ignoring elapsed stationary intervals and any offset between `StartFr` and the first retained sample.

ii.
```python
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The switch was justified as making time consistent with running-only analysis rather than wall-clock pauses.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Creates one time value in order for each retained neural frame.

ii.
```python
frames = frames[ft_move[frames] > 0]
t_since = np.arange(frames.size, dtype=np.float32) * frame_dt
neural = spk_sel[:, frames]
```

iii. The notes require all time-varying streams to share the neural time dimension.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Uses the raw per-trial `isRew` flag.

ii.
```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
```

iii. The notes say it represents rewarded-corridor identity, including sessions with no water delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Broadcasts the trial's 0/1 reward value over all retained timepoints without other transformation.

ii.
```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Uniform time-varying input matrices were chosen for decoder convenience.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Uses each trial's exact raw `WallName`; it also scans all canonical sessions to build the vocabulary.

ii.
```python
names.update(map(str, np.asarray(beh["WallName"]).tolist()))
wall_name = np.asarray(beh["WallName"]).astype(str)
```

iii. Exact names avoid masked `stim_id` values and preserve swap and exemplar identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Maps all 15 exact wall strings to sorted integer categories and broadcasts the category over the trial.

ii.
```python
stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. The agent intentionally preserved subtypes because it believed pooling should be downstream.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Uses `LickFr` and `LickTrind` to select lick frames belonging to each trial.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
```

iii. The notes describe binary projection of raw lick events onto imaging frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. After removing nonfinite events and casting lick frames to integers, marks a retained sample 1 when its imaging-frame index matches any lick; otherwise 0.

ii.
```python
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)].astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. This implements the requested per-frame binary licking output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Evaluates licks on exactly the same retained `frames` indices used for neural columns.

ii.
```python
neural = spk_sel[:, frames]
licking = np.isin(frames, lick_frames_trial)
```

iii. The notes report direct raw-data spot checks of this alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Uses framewise raw `ft_Pos` at retained neural frames.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
pos = ft_pos[frames]
```

iii. Raw positions are in decimeters over a 4 m texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Divides decimeter position by 10, floors it, casts to integer, and clips to labels 0–3.

ii.
```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. This directly implements four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Uses thresholds at 10, 20, and 30 dm: [0,10), [10,20), [20,30), and [30,40] dm, with clipping outside the range.

ii.
```python
np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The 0–40 dm texture length makes these four equal spatial categories.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Indexes position with the identical retained-frame vector used for neural data.

ii.
```python
neural = spk_sel[:, frames]
pos = ft_pos[frames]
```

iii. Framewise behavior and neural data share imaging-frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Uses framewise `ft_RunSpeed` at retained neural frames.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. This is the direct raw speed stream specified in the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Concatenates speeds from every retained frame across all sessions, computes global numerical quantiles, then applies their three interior thresholds to every trial.

ii.
```python
all_speed_concat = np.concatenate(all_speed_values)
speed_edges = np.quantile(all_speed_concat, [0, .25, .5, .75, 1])
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right")
```

iii. The agent interpreted 'each corresponding to 25% of the data' globally and restricted the distribution to included running texture frames.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds at the global 25th, 50th, and 75th percentiles using right-sided insertion; if quantile edges tie, nudges later edges upward with `nextafter`.

ii.
```python
for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```

iii. This was intended to provide four numeric speed ranges and avoid non-increasing labels.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Reads raw speeds at the same retained frames as neural data, then category assignment preserves their order and length.

ii.
```python
raw_speed = ft_speed[frames]
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right")
```

iii. The notes report direct spot checks against raw speed at representative session/trial pairs.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Filters nonfinite lick events and cue positions, skips trials with no retained moving frame, has fallback neuron selection, and validates stream lengths. It does not explicitly clip `StartFr`/`GrayFr` to imaged frame count or truncate all behavior streams to neural length before constructing trial ranges.

ii.
```python
lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
if frames.size == 0: continue
if input_arr.shape[1] != T or output_arr.shape[1] != T: raise ValueError(...)
```

iii. The agent characterized the dataset as clean and relied on successful full conversion plus raw spot checks.

## 12-a. What are the most time-consuming steps of the code?

i. Identifies loading the roughly 404 GB of spike files and neuron-selection calculations over full matrices as dominant; full conversion was estimated around 14.5 minutes.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
```

iii. The notes' size-weighted timing says I/O dominates and each spike file is loaded once during session processing.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in reward-neuron selection, trial construction, sample scoring, and final stacking could partly be grouped/vectorized, although variable-length arrays limit full vectorization.

ii.
```python
for trial_idx in np.flatnonzero(valid_reward_trials): ...
for trial_idx in range(int(beh["ntrials"])): ...
for neural_trial, input_raw, output_raw in zip(...): ...
```

iii. The notes focus more on I/O than loop optimization and do not claim these loops dominate.

## 12-c. What processing does the code repeat multiple times?

i. Behavior dictionaries are repeatedly loaded during stimulus collection, time-bin calculation, sample ranking, and session processing; raw running speeds are also stored per trial, re-collected, concatenated, and revisited for binning.

ii.
```python
stimulus_values = collect_stimulus_values(full_catalog)
time_bin_ms = compute_time_bin_ms(full_catalog)
beh = load_behavior(candidate.exp_type, candidate.beh_key)
for trial in session.output_raw: all_speed_values.append(trial["running_speed_raw"])
```

iii. The notes explicitly flag repeated behavior reloads for vocabulary and frame-interval metadata.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Computes and retains extensive `processing_info` (d-primes, cue-position summaries, trial-length summaries, optional examples), raw speed intermediates, and sample-selection diagnostics that are not part of the saved decoder dataset. It also loads all behaviors to collect metadata before loading them again.

ii.
```python
proc_info = {**selection_info, **trial_summary, "cue_positions_dm": cue_positions, "example_trial": example_trial}
output_trials.append({..., "running_speed_raw": raw_speed})
```

iii. These support plots, logging, sanity checks, and global speed bins; after finalization much of the diagnostic material is discarded.
