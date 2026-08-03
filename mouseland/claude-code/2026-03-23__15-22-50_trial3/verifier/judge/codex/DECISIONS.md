# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `data/beh/Imaging_Exp_info.npy` as the master index, deduplicates physical sessions by `(mname, datexp, blk)`, loads each needed `Beh_<exp_type>.npy` once into a cache, and then loads spike and retinotopy files per session during session processing.

ii. 
```python
exp_info = np.load(
    os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
).item()
```
```python
for et in exp_types_needed:
    beh_cache[et] = load_beh(et)
```
```python
spk = load_spk(session['db_entry'])
iarea = load_retino(session['db_entry'])
```

iii. The notes say the same physical session can appear in multiple experiment types, so the converter should keep each physical session once and then reuse the relevant behavior file from the first experiment-type entry encountered.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The final `subjects` list is the sorted set of unique mouse names, and each retained session gets a `subject_idx` from that mapping.

ii. 
```python
subjects = sorted(set(s['mname'] for s in sessions))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```
```python
subject_idx_list.append(subject_to_idx[session['mname']])
```

iii. The notes explicitly describe mouse name as the subject identifier and report 19 unique mice.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` triple. Duplicate appearances of the same physical recording in multiple experiment types are dropped, and the remaining sessions are sorted by mouse name and date.

ii. 
```python
key = (db['mname'], db['datexp'], db['blk'])
if key in seen:
    continue
seen.add(key)
```
```python
sessions.sort(key=lambda s: (s['mname'], s['datexp']))
```

iii. The notes justify this as “use each physical session once” because the same recording may appear under several behavior/analysis groupings.

## 1-d. How are the data split into trials?

i. Trials are split using `StartFr` and `EndFr`, with each trial taken as the full frame interval `[StartFr[t], EndFr[t])`. This includes the corridor and grey-space portion of the trial, not just texture-corridor frames.

ii. 
```python
StartFr = beh['StartFr'].astype(int)
EndFr = beh['EndFr'].astype(int)
```
```python
for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
```
```python
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes say the chosen “trial window” is `StartFr` to `EndFr` and describe that as corridor plus gray space.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a paper-matching quality-control filter to trials. It only skips trials that are out of bounds, collapse to empty after clipping, or have fewer than 2 timepoints.

ii. 
```python
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1
    continue
```
```python
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue
```
```python
if n_tp < 2:
    skipped += 1
    continue
```

iii. The notes frame these as sanity/safety checks for malformed or truncated trials rather than scientific curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural activity itself is derived from the concatenated `spks` arrays in each session’s neural data file. The per-neuron brain-region metadata is derived from retinotopy `iarea`.

ii. 
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The notes say the data are already Suite2p deconvolved traces, so no dF/F or extra deconvolution is needed.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, slices trial windows directly from `StartFr` to `EndFr`, and stores each trial as `float16`. It keeps variable trial lengths and does not pad, truncate to 32 frames, or otherwise reformat trials to a common length.

ii. 
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
neural = spk[:, start:end].astype(np.float16)
```

iii. The code comment says `float16` is used to reduce memory. The notes say the neural signal should remain raw frame-level deconvolved spikes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps all neurons. It maps `iarea` into `V1`, `mHV`, `lHV`, `aHV`, or `other`, but it does not drop `other`, does not restrict to the four named visual regions, and does not apply d-prime or activity filtering.

ii. 
```python
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
    -1: 'other', 7: 'other',
}
```
```python
idx = np.full(len(iarea), region_to_idx['other'], dtype=np.int64)
for area_val, region_name in AREA_MAP.items():
    mask = iarea == area_val
    idx[mask] = region_to_idx[region_name]
```

iii. The notes explicitly justify this as “include ALL neurons (no filtering by area), as the decoder should learn to use relevant neurons.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to corridor entry by using `StartFr` as the first frame of each trial. The extracted window then runs through `EndFr`, so the alignment event is trial start, but the retained duration is variable.

ii. 
```python
for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
```
```python
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes describe the temporal alignment event as “Trial start (corridor entry)” and treat `StartFr` as the anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging-frame resolution. The AI estimates one global `frame_period` from the first sampled session’s `ft` timestamps and uses that as the time-bin size for all sessions. No rebinning is applied.

ii. 
```python
sample_beh = beh_cache[sessions[0]['exp_type']][sessions[0]['beh_key']]
ft = sample_beh['ft']
dt = np.diff(ft) * 24 * 3600
frame_period = float(np.nanmedian(dt))
```
```python
'time_bin_size': frame_period * 1000,
```

iii. The notes justify this as using the raw frame-level data at about 3.17 Hz with no additional temporal binning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` together with the per-frame indices inside the selected trial window, scaled by the global frame period estimated from `ft`.

ii. 
```python
SoundFr = beh['SoundFr']
```
```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The notes describe this variable as the signed time difference between the current frame and the sound-cue frame.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the AI computes `time_to_sound = (SoundFr[t] - frame_index) * frame_period`, producing a continuous per-frame signal that is positive before the cue and negative after it.

ii. 
```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The code comment states the sign convention directly. The notes also describe the variable as “time until cue” / “time since cue,” although the notes contain a sign-description inconsistency in one table.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same frame indices used to slice the neural trial, so it is frame-aligned with the neural array for that trial.

ii. 
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes repeatedly say that all streams are aligned by imaging-frame number.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session’s mouse name and date (`mname`, `datexp`) after sessions are grouped by mouse and sorted chronologically.

ii. 
```python
for i, s in enumerate(sessions):
    mouse_sessions[s['mname']].append((s['datexp'], i))
```
```python
sess_list.sort(key=lambda x: x[0])
for day_idx, (datexp, global_idx) in enumerate(sess_list):
    days[global_idx] = float(day_idx)
```

iii. The notes justify this as using ordinal session order within each mouse as a training-day proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Within each mouse, sessions are sorted by `datexp` and assigned day indices `0, 1, 2, ...`. The chosen scalar day is then broadcast across all timepoints of each trial.

ii. 
```python
days = np.zeros(len(sessions), dtype=np.float32)
for mname, sess_list in mouse_sessions.items():
    sess_list.sort(key=lambda x: x[0])
    for day_idx, (datexp, global_idx) in enumerate(sess_list):
        days[global_idx] = float(day_idx)
```
```python
day = np.full(n_tp, training_day, dtype=np.float32)
```

iii. The notes call this the “ordinal session index (by date) within each mouse.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and the frame indices within the current trial window, scaled by the global frame period.

ii. 
```python
StartFr = beh['StartFr'].astype(int)
```
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. The notes describe this as elapsed time from corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the AI subtracts the trial’s `StartFr` from each frame index and multiplies by `frame_period`, giving a continuous per-frame elapsed-time signal starting at zero.

ii. 
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. The notes justify it as a straightforward time-from-alignment variable.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is built from the same `[start, end)` frame indices as the neural slice for each trial, so it is directly aligned to the neural data.

ii. 
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes treat frame-number alignment as the common alignment rule for all variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial behavior variable `isRew`.

ii. 
```python
isRew = beh['isRew']
```
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The notes describe this as a direct rewarded-vs-unrewarded corridor indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. There is no substantive transformation. The trial’s `isRew` value is cast to float and broadcast across all timepoints of the trial.

ii. 
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The notes say unsupervised and naive sessions simply have no available reward, so this can remain a per-trial constant.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives visual-stimulus labels from `WallName` during per-trial processing and from `UniqWalls` when it builds the global category list. It keeps the original named wall variants rather than collapsing them to four base textures.

ii. 
```python
for wn in beh['UniqWalls']:
    all_stim.add(str(wn))
```
```python
stim_name = str(WallName[t])
stim_idx = stim_to_idx[stim_name]
```

iii. The notes explicitly say the converter should keep 15 unique stimuli.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects all unique wall names across sessions, sorts them, maps each trial’s `WallName` to that 15-class index, and broadcasts the resulting class label across the trial’s timepoints.

ii. 
```python
all_stimuli = get_all_stimuli(sessions, beh_cache)
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
```
```python
stim_name = str(WallName[t])
stim_idx = stim_to_idx[stim_name]
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. The notes justify this as preserving all observed wall variants instead of grouping them into coarse texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`. The AI does not use `LickTrind`; it uses the global lick frame indices directly.

ii. 
```python
lick_frs = beh['LickFr']
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. The notes state that licking is represented as a binary per-frame variable built from lick-frame events.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds each fractional lick frame to the nearest integer frame, selects licks that fall inside the trial’s `[start_fr, end_fr)` interval, shifts them into trial-relative indices, clips to valid bounds, and writes `1` at those bins in a binary vector.

ii. 
```python
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
if mask.any():
    trial_lick_frs = lick_frs_int[mask] - start_fr
    trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
    lick_vec[trial_lick_frs] = 1
```

iii. The notes justify this as “binary per frame, 1 if lick in frame.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick vector is defined on the same frame interval used for the neural slice, using trial-relative frame indices after subtracting `start_fr`.

ii. 
```python
trial_lick_frs = lick_frs_int[mask] - start_fr
```
```python
lick = make_lick_vector(beh, start, end)
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes say all streams are aligned on the frame grid, and the code follows that rule for licking.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, the per-frame corridor position.

ii. 
```python
ft_Pos = beh['ft_Pos']
```
```python
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. The notes describe `ft_Pos` as position in decimeters along the 6 m corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI takes `ft_Pos[start:end]` and bins it with edges at 10, 20, and 30 dm, producing four bins over the full trial window. Because the trial window includes grey space, the last bin absorbs both the last meter of texture and the grey-space segment.

ii. 
```python
def digitize_position(pos):
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)
```
```python
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. The notes explicitly justify “position bin 3 includes gray space.”

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are `[0,10)`, `[10,20)`, `[20,30)`, and `[30,60+]` dm, labeled `0-1m`, `1-2m`, `2-3m`, and `3m+`.

ii. 
```python
POS_BIN_EDGES = [0, 10, 20, 30, 60.01]
POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']
```
```python
bins = np.digitize(pos, [10, 20, 30])
```

iii. The notes say this choice intentionally folds grey-space frames into the last position category.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by taking `ft_Pos[start:end]` on the same trial frame interval as the neural slice.

ii. 
```python
pos = ft_Pos[start:end]
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes treat all time-varying variables as frame-synchronous with the neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed']
```
```python
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. The notes map `ft_RunSpeed` directly to the running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first pools `ft_RunSpeed` across all sessions, computes global 25th/50th/75th percentile thresholds, and then digitizes each trial’s speed trace against those thresholds.

ii. 
```python
all_speeds = []
for s in sessions:
    beh = beh_cache[s['exp_type']][s['beh_key']]
    speeds = beh['ft_RunSpeed']
    all_speeds.append(speeds)

all_speeds = np.concatenate(all_speeds)
quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. The notes justify this as “quartile bins across all data.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded with `np.percentile(..., [25, 50, 75])` on the pooled speed values, and `np.digitize` converts each frame into one of four global quartile bins.

ii. 
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
def digitize_speed(speed, quartiles):
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)
```

iii. The notes say the intent was to make 25%-of-data bins, though the notes later acknowledge the zero-speed pileup makes the realized proportions uneven.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by taking `ft_RunSpeed[start:end]` on the same frame interval used for neural data.

ii. 
```python
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes use the same frame-alignment rationale here as for licking and position.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles obvious inconsistencies defensively: it asserts retinotopy/neuron-count agreement, skips trials with invalid or too-short frame windows, clips `end` to the available `ft_Pos`/`ft_RunSpeed` lengths, and clips lick indices to valid trial-relative bounds.

ii. 
```python
assert len(iarea) == n_neurons, \
    f"Neuron count mismatch: spk={n_neurons}, retinotopy={len(iarea)}"
```
```python
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue
```
```python
trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
```

iii. The notes describe these as sanity checks and missing-data safeguards rather than dataset-specific corrections.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is reading and concatenating the very large spike files for each session, followed by the per-trial session-processing loop and the global pass used to compute speed quartiles.

ii. 
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
for t in range(ntrials):
    ...
```
```python
all_speeds = []
for s in sessions:
    ...
```

iii. The notes estimate about 15 seconds per session and repeatedly discuss memory and full-dataset runtime as the main bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loop in `process_session`, the per-trial lick reconstruction that rescans all `LickFr` values, the global speed-collection loop, and the loop that maps `iarea` values into region ids.

ii. 
```python
for t in range(ntrials):
    ...
```
```python
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
```
```python
for s in sessions:
    beh = beh_cache[s['exp_type']][s['beh_key']]
    speeds = beh['ft_RunSpeed']
    all_speeds.append(speeds)
```
```python
for area_val, region_name in AREA_MAP.items():
    mask = iarea == area_val
    idx[mask] = region_to_idx[region_name]
```

iii. The notes say the script should be efficient, but they do not provide a separate optimization justification for these loops.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds frame-index arrays and constant broadcasts for every trial, and `make_lick_vector` rescans the full `LickFr` array separately for each trial instead of precomputing a session-level lick raster once.

ii. 
```python
frame_indices = np.arange(start, end, dtype=np.float64)
```
```python
day = np.full(n_tp, training_day, dtype=np.float32)
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```
```python
lick = make_lick_vector(beh, start, end)
```

iii. There is no explicit justification in the notes beyond keeping the implementation simple and decoder-compatible.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the default conversion path there is not much truly discarded processing. The optional `--show-processing` plots are diagnostic only, and the script also computes summary/metadata fields that are not used by the decoder itself. Brain-region bookkeeping is extra relative to decoder training but is required by the target file format.

ii. 
```python
if args.show_processing and i < 2:
    plot_processing(...)
```
```python
'speed_quartiles': speed_quartiles.tolist(),
'position_bin_edges_dm': POS_BIN_EDGES,
'total_trials': sum(len(s) for s in neural_all),
```

iii. The notes justify the plotting and summaries as validation and documentation steps rather than core decoder inputs.
