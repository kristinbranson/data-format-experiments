# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads experiment metadata from `data/beh/Imaging_Exp_info.npy`, then walks the experiment-type dictionaries, loads the matching `Beh_<exp_type>.npy` behavior file, deduplicates recordings by `(mname, datexp, blk)`, and later loads one neural `.npy` file and one retinotopy `.npz` file per retained recording.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
sessions = collect_all_sessions(exp_info, ROOT)
```

```python
for exp_type in exp_order:
    db_list = exp_info[exp_type]
    beh_path = os.path.join(root, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_path, allow_pickle=True).item()
```

```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
```

iii. The justification in `CONVERSION_NOTES.md` is that there are 23 experiment types but only 89 unique recordings, so recordings should be collected once; the trajectory adds that duplicated entries largely share the same behavioral streams and differ mainly in analysis annotations such as `stim_id`.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The script builds a sorted unique subject list and records one `subject_idx` per session by indexing that list.

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions))
...
all_subject_idx.append(all_subjects.index(result['mname']))
...
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=int),
```

iii. The agent’s notes treat the paper’s “89 recordings in 19 mice” as the subject/session structure to preserve, so `mname` becomes the subject identifier.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `(mname, datexp, blk)` recordings. If a recording appears in multiple experiment-type tables, only the first one encountered in the agent’s preferred sort order is kept.

ii.
```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
...
sessions.append({
    'mname': ndb['mname'],
    'datexp': ndb['datexp'],
    'blk': ndb['blk'],
    'exp_type': exp_type,
    'beh_key': beh_key,
    'beh': beh_data[beh_key],
    'ndb': ndb,
})
```

```python
exp_order = sorted(exp_info.keys(), key=lambda x: (
    0 if 'test' in x else 1,
    0 if 'sup_' in x else 1,
    0 if 'after' in x else 1,
    x
))
```

iii. The justification is explicit in both the code comments and notes: use each recording once, and prefer “more complete” behavior annotations by ordering test before train, supervised before unsupervised, and after-learning before before-learning.

## 1-d. How are the data split into trials?

i. The number of trials comes from `beh['ntrials']`. Neural data are reshaped into `(neurons, n_trials, 60)` by position interpolation and then accessed one trial at a time in a `for tr in range(n_trials)` loop; behavioral variables already exist as per-trial arrays and are indexed by the same `tr`.

ii.
```python
n_trials = beh['ntrials']
...
interp_spk = position_interpolate_spk(
    valid_spk,
    ft_pos_cum[vr_moving],
    corridor_len,
    n_trials,
    n_bins=N_POS_BINS
)
...
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The agent’s justification is that this matches the paper code’s `get_interpPos_spk(...)`, which turns framewise running data into trial-by-position tensors using `ntrials`.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. All trials are kept unless the whole session fails processing or has fewer than two trials. Trialwise effects come indirectly from using only running frames for neural interpolation and from ignoring lick events outside valid trial/bin ranges.

ii.
```python
vr_moving = ft_move > 0
...
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

```python
try:
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
except Exception as e:
    ...
    continue
...
if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
```

iii. The notes frame the main curation as frame-level running-only selection and neuron-level retinotopy filtering; they do not claim any extra trial rejection rule from the reference pipeline.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived primarily from `spk_data['spks']`, with `beh['ft_move']`, `beh['ft_PosCum']`, `beh['ntrials']`, and `beh['Corridor_Length']` used to select running frames and interpolate by trial/position. `ret_data['iarea']` is used for neuron filtering and region labels.

ii.
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
iarea = ret_data['iarea']
...
n_trials = beh['ntrials']
corridor_len = beh['Corridor_Length']
ft_move = beh['ft_move'][:n_frames]
ft_pos_cum = beh['ft_PosCum'][:n_frames]
```

iii. The agent cites the paper’s use of Suite2p deconvolved calcium traces and the notebook/utils preprocessing that interpolates those traces with `ft_move` and `ft_PosCum`.

## 2-b. How is the `neural` data processed?

i. The agent concatenates the per-plane `spks`, keeps only valid neurons, restricts to running frames, linearly interpolates onto 60 evenly spaced position bins per trial, then truncates to the first 40 texture bins and stores each trial as `float32`.

ii.
```python
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
valid_spk = spk[valid_neurons][:, vr_moving]
interp_spk = position_interpolate_spk(
    valid_spk,
    ft_pos_cum[vr_moving],
    corridor_len,
    n_trials,
    n_bins=N_POS_BINS
)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

iii. The notes justify this as matching `utils.spk_pos_interp()` / `utils.get_interpPos_spk()`, with `numpy.interp` substituted for `scipy.interpolate.interp1d` purely for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered to neurons in visual cortex (`iarea != -1` and `iarea != 7`) and to timepoints where the VR is moving (`ft_move > 0`). No additional neuron reliability, d-prime, or baseline-normalization filter is applied.

ii.
```python
valid_mask = (iarea != -1) & (iarea != 7)
...
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
...
valid_spk = spk[valid_neurons][:, vr_moving]
```

iii. The notes explicitly say “Only visual cortex neurons are included” and “Only running frames are used,” pointing to `load_retino()` / `neu_area_ID()` and the reference notebook’s interpolation loop.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data to trial start by treating the first interpolated corridor bin as corridor entry. Each trial is represented by position bins 0–39, so alignment is implicit in the per-trial bin index after interpolation.

ii.
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': N_TEXTURE_BINS * BIN_SIZE_SEC,
```

```python
interp_spk = position_interpolate_spk(..., n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The notes justify this by combining the paper’s corridor-entry trial structure with the reference code’s trial-by-position interpolation and the fixed VR speed during running.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent treats each 1 dm position bin as `1/6 s`, so the reported bin size is about `166.67 ms`. No framewise temporal resampling is done directly; instead, the raw traces are rebinned indirectly by interpolation into 60 position bins and then cropped to 40.

ii.
```python
VR_SPEED = 6.0
BIN_SIZE_SEC = 1.0 / VR_SPEED
BIN_SIZE_MS = BIN_SIZE_SEC * 1000
N_POS_BINS = 60
N_TEXTURE_BINS = 40
```

```python
'time_bin_size': BIN_SIZE_MS,
...
interp_spk = position_interpolate_spk(..., n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

iii. The trajectory shows the agent explicitly reasoning from the methods text: 60 cm/s fixed VR speed and 1 dm position bins imply 6 bins/s, or about 167 ms per bin.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `beh['SoundPos']` plus the implicit per-bin corridor position index and the fixed-speed conversion constant `BIN_SIZE_SEC`.

ii.
```python
sound_pos = beh['SoundPos']
...
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The notes justify this by stating that the cue was positioned randomly inside the corridor and that, under fixed VR speed, position offsets can be converted to time offsets.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the script subtracts the cue position from each texture-bin index and multiplies by `1/6 s`, producing negative values before the cue and positive values after it.

ii.
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` states this formula directly and explains the sign convention as “negative before cue, positive after.”

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built with exactly one value per neural position bin, over the same 40 per-trial bins used for `neural`, and then stacked into the per-trial `(4, 40)` input array.

ii.
```python
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail],
                        axis=0)  # (4, 40)
...
neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The notes say the cue-time input is aligned to the same trial-start/position-bin representation as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date string `datexp` and the subject identity `mname`, which together define each mouse’s earliest recorded session and each session’s offset from that date.

ii.
```python
session_date = compute_day_of_training(datexp)
...
mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d
```

iii. The justification in the notes is that “day of training” should be continuous and constant within trial; the trajectory shows the agent choosing recording-date offsets because no richer per-day training log is loaded in the conversion script.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script parses `YYYY_MM_DD` strings to `datetime`, computes `days = current_date - earliest_date_for_mouse`, converts that to `float`, and broadcasts the resulting scalar across all 40 bins in every trial of the session.

ii.
```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')
```

```python
days = (d - mouse_first_date[s['mname']]).days
session_days[key] = float(days)
...
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. The notes call this “Days since first recording for each mouse.” That is the agent’s explicit proxy for training day.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. No environment-type input is created. The script defines only four inputs: time to sound cue, day of training, time since trial start, and reward availability.

ii.
```python
'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start',
                'reward_availability'],
```

iii. There is no separate justification in the notes beyond following the decoder-task input list from the instructions, which also contains only those four variables.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The agent does not compute, encode, or store an environment-type variable.

ii.
```python
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail],
                        axis=0)  # (4, 40)
```

iii. The omission is consistent with the notes and README, which document only the four requested decoder inputs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from an explicit raw timestamp array. Instead, it is derived from the per-bin index `i` and the fixed-speed constant `BIN_SIZE_SEC`, after the neural data have already been converted to trial-by-position bins.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The notes justify this by treating each 1 dm bin as a constant-duration step because the corridor moves at 60 cm/s whenever running timepoints are included.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script generates a length-40 ramp from `0` to `39 * BIN_SIZE_SEC`, the same for every trial in every session.

ii.
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` summarizes this as “Linear from 0 to ~6.5 seconds across 40 bins.”

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is created on the same 40-bin trial grid as the neural data and stacked into the same per-trial input tensor.

ii.
```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
...
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail],
                        axis=0)
```

iii. The notes say all inputs are aligned to “Trial start (corridor entry)” with 40 timepoints per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `beh['isRew']`.

ii.
```python
is_rew = beh['isRew'].astype(float)
...
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The notes state this directly: reward availability is “From `beh['isRew']`.”

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script converts `isRew` to float and broadcasts the per-trial value across all 40 bins of that trial.

ii.
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The notes justify this as a per-trial binary context variable: 1 in rewarded corridors and 0 otherwise.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh['WallName']`, with the global label vocabulary assembled from all sessions’ `beh['UniqWalls']`.

ii.
```python
all_stim_set = set()
for s in sessions:
    for wn in s['beh']['UniqWalls']:
        all_stim_set.add(str(wn))
all_stim_names = sorted(all_stim_set)
```

```python
wall_names = beh['WallName']
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. The notes justify this as a 15-category trial label covering all stimulus names seen across the full dataset.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial’s string `WallName` is converted to a global integer category index and then broadcast across all 40 bins of the trial.

ii.
```python
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
...
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. The notes document the integer mapping and explain that this is a per-trial categorical output expanded across time.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the event-position array `beh['LickPos']` and the event-to-trial index array `beh['LickTrind']`.

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
```

iii. The notes explicitly cite `beh['LickPos']` and `beh['LickTrind']` as the source variables.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script initializes a zero `(n_trials, 40)` array, floors each lick position to an integer dm bin with `int(pos)`, and writes a `1` at `(trial, bin)` if the event falls inside the 40 texture bins.

ii.
```python
lick_binary = np.zeros((n_trials, n_bins), dtype=int)
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(pos)
    if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
        lick_binary[tr, bin_idx] = 1
```

iii. The notes justify the binary encoding as “1 if any lick event occurred in that position bin, 0 otherwise.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is converted directly onto the same 40 per-trial position bins as the neural data, then copied trial by trial into the aligned output tensor.

ii.
```python
lick_tr = lick_binary[tr, :].astype(np.int64)
...
output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)
```

iii. The notes say the output is binary and time-varying on the same per-bin grid as the interpolated neural activity.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is effectively derived from the standardized 40-bin corridor representation itself rather than from a raw behavior field at conversion time. The script uses the known 40 texture-bin geometry after interpolation.

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The notes justify this as “4 bins (0-3), time-varying,” each covering 1 m of the 4 m texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A fixed 40-element vector is created once per session and reused for every trial.

ii.
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
...
pos_tr = pos_bins.astype(np.int64)
```

iii. The notes explain that the position output is deterministic from the corridor layout after interpolation.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The 40 bins are grouped into four equal 10-bin chunks, corresponding to 0–1 m, 1–2 m, 2–3 m, and 3–4 m.

ii.
```python
pos_bins[i] = min(i // 10, 3)  # 0-9 -> 0, 10-19 -> 1, 20-29 -> 2, 30-39 -> 3
```

iii. The notes justify this with the decoder specification: 4 equal-length, 1 m spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Because both neural activity and position share the same 40-bin trial grid, alignment is direct and one-to-one.

ii.
```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
pos_tr = pos_bins.astype(np.int64)
output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)
```

iii. The notes describe both as living on the same position-binned trial representation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['run_pos']`, the behavior matrix that already stores running speed interpolated into `(trial, position)` bins.

ii.
```python
run_pos = beh['run_pos']  # (n_trials, 60) - already position-interpolated
run_speed = run_pos[:, :N_TEXTURE_BINS]  # (n_trials, 40)
```

iii. The notes explicitly say the quartiles are computed from `beh['run_pos']`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script takes the first 40 texture bins from `run_pos`, computes global quartile edges across all sessions, and digitizes each speed value into a bin index 0–3.

ii.
```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
```

```python
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. The notes justify this as satisfying the decoder requirement that running-speed categories correspond to 25% quantiles of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global 25th, 50th, and 75th percentiles of all texture-bin running-speed values across the retained sessions.

ii.
```python
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. `CONVERSION_NOTES.md` records the actual edges and explains that `np.digitize` produces the four bins directly.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Alignment is direct because `beh['run_pos']` is already trial-by-position and the script uses the same first 40 bins as the interpolated neural tensor.

ii.
```python
run_speed = run_pos[:, :N_TEXTURE_BINS]
...
speed_tr = speed_bins[tr, :].astype(np.int64)
...
output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)
```

iii. The notes state that running speed is based on the same 40-bin texture-area representation as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is minimal. Missing behavior files/keys produce warnings and the session is skipped; exceptions during session processing also skip the session; sessions with fewer than two trials are skipped; invalid lick events are ignored; there is no explicit imputation or NaN repair for the used fields.

ii.
```python
if not os.path.exists(beh_path):
    print(f"  Warning: behavior file not found for {exp_type}")
    continue
...
if beh_key not in beh_data:
    print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
    continue
```

```python
try:
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
except Exception as e:
    ...
    continue
```

```python
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

iii. The notes do not claim any sophisticated missing-data policy; the overall stance is that the necessary fields are dense enough for the conversion and that obvious failures should be skipped rather than repaired.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large `spks` arrays and interpolating every neuron’s activity trace across all retained running frames into position bins. A secondary cost is scanning all sessions to compute global running-speed quartiles.

ii.
```python
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

```python
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
```

iii. The trajectory shows the agent explicitly optimizing interpolation by replacing `scipy.interpolate.interp1d` with `numpy.interp` for speed.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron interpolation loop, the per-lick event binning loop, the per-trial assembly loop, and the fixed `pos_bins` construction loop are all vectorizable.

ii.
```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

```python
for i in range(len(lick_pos)):
    ...
```

```python
for tr in range(n_trials):
    ...
```

```python
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The agent’s own notes already acknowledge speed sensitivity around interpolation; the remaining Python loops were left in simple scalar form.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly parses dates, repeatedly fills the day-of-training row after first creating it as zeros, repeatedly performs `all_stim_names.index(wn)` lookups for every trial, and repeatedly broadcasts constant per-trial values inside the trial loop.

ii.
```python
session_date = compute_day_of_training(datexp)
...
d = compute_day_of_training(s['datexp'])
...
d = compute_day_of_training(s['datexp'])
```

```python
day_val = 0.0
day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)
...
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

```python
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. There is no explicit justification for these repetitions; they are implementation conveniences rather than documented requirements.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It interpolates all 60 corridor bins and then discards the last 20 gray-space bins; it computes and stores a placeholder zero day-of-training row before overwriting it; it computes `session_date` and returns it from `process_session` even though the saved dataset uses the later `day_val`; and it builds some metadata/summary values that the decoder itself does not consume.

ii.
```python
interp_spk = position_interpolate_spk(..., n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

```python
day_val = 0.0
day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)
...
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

```python
session_date = compute_day_of_training(datexp)
...
return {
    ...
    'session_date': session_date,
}
```

iii. The notes justify the 60-to-40 crop as matching the paper’s position interpolation while keeping only the texture corridor for decoding; the other discarded work is not justified separately.
