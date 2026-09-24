# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `data/beh/Imaging_Exp_info.npy`, orders experiment types, loads each `Beh_<exp_type>.npy`, deduplicates mouse/date/block recordings, and stores behavior objects in a session list. During session processing it loads and concatenates all planes from the matching spike file and loads the matching retinotopy file.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
beh_data = np.load(beh_path, allow_pickle=True).item()
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
```

iii. The trajectory says the index is the authoritative inventory, behavior files contain multiple keyed sessions, spike planes must be concatenated, and retinotopy supplies area labels. It also deliberately preferred test/supervised/after-learning annotations for duplicates.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mname` values. Each retained session gets the index of its mouse in that list.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions))
all_subject_idx.append(all_subjects.index(result['mname']))
```

iii. The agent reasoned that `mname` directly identifies the mouse and reported 19 unique mice.

## 1-c. How are the data split into sessions?

i. A session is the unique `(mname, datexp, blk)` tuple. Duplicate appearances across experiment types are removed, with a custom ordering deciding which behavior annotation wins.

ii.
```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
```

iii. The trajectory recognized repeated recordings across experiment types and chose one representation per physical recording; it preferred test and supervised entries as allegedly more complete.

## 1-d. How are the data split into trials?

i. The agent uses `beh['ntrials']` and reshapes a continuous, position-interpolated neural array into `(neurons, n_trials, 60)`. It then emits every trial with the first 40 spatial bins; it does not use `ft_trInd` or native frame boundaries.

ii.
```python
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving], corridor_len,
                                      n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The agent justified this as matching the paper's position-interpolation pipeline and using the 4 m texture section.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. Every declared trial is retained unless an entire session has fewer than two trials; session exceptions are caught and skipped.

ii.
```python
for tr in range(n_trials):
    ...
if result['n_trials'] < 2:
    continue
```

iii. The trajectory focused on visual-neuron and moving-frame filtering and did not identify or justify the reference's empty/extreme-duration trial exclusions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from concatenated `spk_data['spks']`; interpolation coordinates come from `ft_PosCum`, `ft_move`, `Corridor_Length`, and `ntrials`; neuron selection/region labels come from retinotopy `iarea`.

ii.
```python
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
ft_move = beh['ft_move'][:n_frames]
ft_pos_cum = beh['ft_PosCum'][:n_frames]
iarea = ret_data['iarea']
```

iii. The agent cited the paper code's `spk_pos_interp`/`get_interpPos_spk` processing and visual-area mapping.

## 2-b. How is the `neural` data processed?

i. Visual-cortex spike traces are restricted to `ft_move > 0`, linearly interpolated against normalized cumulative position to 60 bins/trial, truncated to bins 0–39, and stored as float32.

ii.
```python
valid_spk = spk[valid_neurons][:, vr_moving]
interp_spk = position_interpolate_spk(valid_spk, ft_pos_cum[vr_moving],
                                      corridor_len, n_trials, n_bins=60)
interp_spk = interp_spk[:, :, :40]
```

iii. The agent believed reproducing the paper's spatial analysis was required by “SAME processing,” and replaced SciPy interpolation with `np.interp` for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` or `iarea == 7` are removed; the remaining codes are mapped to V1, mHV, lHV, or aHV. Neural frames with `ft_move <= 0` are excluded before interpolation.

ii.
```python
valid_mask = (iarea != -1) & (iarea != 7)
vr_moving = ft_move > 0
valid_spk = spk[valid_neurons][:, vr_moving]
```

iii. The agent interpreted the paper as analyzing visual-cortex cells and running frames only.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is not temporally aligned to corridor-entry frames. Each trial is represented by fixed spatial positions 0–39, implicitly beginning at corridor position zero.

ii.
```python
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
norm_pos = accum_pos / corridor_len
return interp_spk.reshape(n_neurons, n_trials, n_bins)
```

iii. The agent treated spatial bin zero as trial start and described it in metadata as corridor entry, but did not use `StartFr`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent declares 166.67 ms from an assumed 6 dm/s VR speed. The data are actually spatially resampled into 1 dm bins, not temporally rebinned at a constant elapsed-time resolution.

ii.
```python
VR_SPEED = 6.0
BIN_SIZE_SEC = 1.0 / VR_SPEED
'time_bin_size': BIN_SIZE_MS,
```

iii. It reasoned that one 1 dm spatial bin at 6 dm/s equals 1/6 s, despite also recording the native imaging rate as 3.17 Hz.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived only from per-trial `SoundPos` and the generated spatial-bin index.

ii.
```python
sound_pos = beh['SoundPos']
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)])
```

iii. The agent chose sound position because all converted streams were placed on a position grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It subtracts sound position from bin index and multiplies by 1/6 s. This produces negative values before and positive values after the cue—the opposite sign of “time to cue.”

ii.
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(40)], dtype=np.float32)
```

iii. The code comment explicitly states “negative before cue, positive after,” so the sign was intentional, though inconsistent with the requested/reference meaning.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It has the same 40 spatial bins as the interpolated neural array; alignment is by assumed position, not native time/frame.

ii.
```python
input_trial = np.stack([time_to_cue, day_of_training,
                        time_since_start, reward_avail], axis=0)
```

iii. The agent considered shared position-bin indices sufficient alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It comes from `datexp` for each session and the earliest recording date for that mouse.

ii.
```python
d = compute_day_of_training(s['datexp'])
days = (d - mouse_first_date[s['mname']]).days
```

iii. The agent interpreted day of training as elapsed calendar days from a mouse's first included recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Dates are parsed, calendar-day differences are computed per mouse, and the result is broadcast across all 40 bins of every trial in the session.

ii.
```python
trial_input[1, :] = day_val
```

iii. It chose a per-mouse relative scale so mice with different calendar dates are comparable.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from generated spatial-bin indices and the constant `BIN_SIZE_SEC`, not raw timestamps or `StartFr`.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)])
```

iii. The agent assumed fixed traversal speed converts spatial distance to elapsed time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Bin numbers 0–39 are multiplied by 1/6 s, yielding a fixed ramp for every trial.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(40)], dtype=np.float32)
```

iii. This followed the agent's fixed-speed spatial representation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is stacked on the same 40 position-bin columns, but does not reflect the actual frame times of the neural observations.

ii.
```python
input_trial = np.stack([... time_since_start ...], axis=0)
```

iii. The agent equated matching array columns with temporal alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is taken from per-trial `beh['isRew']`.

ii.
```python
is_rew = beh['isRew'].astype(float)
```

iii. The agent treated the supplied reward flag as authoritative.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is broadcast unchanged across all 40 bins of its trial.

ii.
```python
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. Reward availability is per-trial, so the agent made it constant over the trial.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`; the global label vocabulary is collected from each session's `UniqWalls`.

ii.
```python
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. The agent noted that wall names directly encode the shown stimulus.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct wall variant is sorted and assigned its own index, then broadcast over 40 bins. No mapping collapses variants into the four base categories.

ii.
```python
all_stim_names = sorted(all_stim_set)
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. The agent favored preserving all observed stimulus names and reported 15 categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses lick positions and trial indices, `LickPos` and `LickTrind`, rather than lick frame numbers.

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
```

iii. This was selected to match the spatially interpolated trial representation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick's position is truncated to an integer bin; that trial/bin becomes 1. Multiple licks collapse to a single binary value and out-of-range events are ignored.

ii.
```python
bin_idx = int(pos)
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

iii. The agent wanted a binary lick indicator per spatial bin.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Both are indexed by trial and nominal 1 dm position bins, but neural values are interpolated and licks are integer-binned, so alignment is spatial rather than temporal.

ii.
```python
lick_tr = lick_binary[tr, :].astype(np.int64)
output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)
```

iii. The agent regarded the common 40-bin position grid as aligned.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is generated from the known indices of the 40 spatial bins; it does not read `ft_Pos`.

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. Because neural data had already been position-interpolated, the agent treated position as deterministic from column index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A fixed four-class vector is created once and copied into every trial.

ii.
```python
pos_tr = pos_bins.astype(np.int64)
```

iii. The fixed vector represents the four 1 m segments of the retained 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Integer position bins 0–9, 10–19, 20–29, and 30–39 map to classes 0–3 using floor division by 10.

ii.
```python
pos_bins[i] = min(i // 10, 3)
```

iii. Ten 1 dm bins equal each requested 1 m category.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position classes are assigned directly by each neural column's spatial interpolation index.

ii.
```python
neural_trial = interp_spk[:, tr, :]
pos_tr = pos_bins.astype(np.int64)
```

iii. The agent viewed this as exact spatial alignment, although it is not native frame/time alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `beh['run_pos']`, the already position-interpolated trial-by-position speed array.

ii.
```python
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]
```

iii. The agent selected the paper-provided position-grid speed to match its neural columns.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global percentile thresholds are computed from all sessions' first 40 position bins; `np.digitize` assigns classes 0–3.

ii.
```python
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
speed_bins = np.clip(np.digitize(run_speed, speed_quantile_edges), 0, 3)
```

iii. The trajectory says the agent changed to dataset-wide quartile thresholds to satisfy “each corresponding to 25% of the data.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below Q25, between Q25/Q50, Q50/Q75, and at/above Q75 receive 0, 1, 2, and 3. Ties are not rank-split, so class counts need not be equal.

ii.
```python
speed_bins = np.digitize(run_speed, speed_quantile_edges)
```

iii. The agent considered percentile edges the natural definition of quartile bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos[:, :40]` and interpolated neural data share trial and spatial-bin indices, but are not aligned on imaging frames/time.

ii.
```python
speed_tr = speed_bins[tr, :].astype(np.int64)
```

iii. The agent relied on both streams' position-interpolated representation.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior files/keys are warned about and skipped; duplicate sessions are ignored; session-processing exceptions are caught and skipped; sessions with fewer than two trials are skipped. Arrays tied to frames are sliced to `n_frames`, but inconsistent lengths, NaNs, empty trials, and anomalous durations are not explicitly repaired or filtered.

ii.
```python
if not os.path.exists(beh_path):
    continue
if beh_key not in beh_data:
    continue
except Exception as e:
    ...
    continue
```

iii. The trajectory emphasized completing conversion despite isolated bad sessions, but did not document the reference's targeted handling of behavior extending beyond imaging.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating hundreds of gigabytes of spike planes and interpolating each neuron separately are the dominant operations; full conversion was repeatedly run and monitored for a long duration.

ii.
```python
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(...)
```

iii. The trajectory explicitly optimized interpolation and used garbage collection because full conversion was expensive.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-neuron interpolation, per-lick assignment, position-bin creation, per-trial construction, date scans, and repeated subject lookup are Python loops. Position bins and time ramps are trivially vectorizable; lick assignment can use indexed writes; subject lookup can use a dictionary. Per-neuron interpolation is harder because `np.interp` is one-dimensional.

ii.
```python
for s in range(n_neurons): ...
for i in range(len(lick_pos)): ...
for i in range(N_TEXTURE_BINS): ...
for tr in range(n_trials): ...
```

iii. The agent only called out optimizing interpolation; it did not provide a systematic efficiency justification for the remaining loops.

## 12-c. What processing does the code repeat multiple times?

i. It parses session dates in multiple passes, scans sessions to build stimuli/subjects/speeds/first dates/days, recreates identical time and position vectors for every trial, casts the fixed position vector each trial, and loads behavior files once per experiment type rather than grouping all work around a file cache.

ii.
```python
for s in sessions: d = compute_day_of_training(s['datexp'])
for tr in range(n_trials):
    time_since_start = np.array([i * BIN_SIZE_SEC for i in range(40)])
    pos_tr = pos_bins.astype(np.int64)
```

iii. The trajectory prioritized clarity and memory release, not eliminating these smaller repeated computations.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It interpolates all 60 corridor bins and then discards the last 20, computes/stores `session_date` only for the returned intermediate, creates day-of-training zeros that are overwritten later, and retains metadata fields not required by the decoder. It also calculates `n_frames` then does not use the variable after slicing.

ii.
```python
interp_spk = position_interpolate_spk(..., n_bins=60)
interp_spk = interp_spk[:, :, :40]
day_of_training = np.full(N_TEXTURE_BINS, 0.0)
trial_input[1, :] = day_val
```

iii. The first choice was meant to mirror the paper's 60-bin interpolation before selecting the texture region; placeholders simplified later session-level filling.
